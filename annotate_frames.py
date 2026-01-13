import sys
import json
import shutil
import math
from pathlib import Path
import cv2
import numpy as np

def update_intrinsic_matrix(K_array, crop_param, resize_param, original_h, original_w):
    """
    カメラ行列(K)をCrop/Resizeに合わせて更新する関数
    K_array: np.array (3x3)
    """
    K = K_array.copy()

    # 1. Cropによる補正
    current_h, current_w = original_h, original_w
    if crop_param is not None:
        (x1, y1), (x2, y2) = crop_param
        K[0, 2] -= x1
        K[1, 2] -= y1
        current_w = x2 - x1
        current_h = y2 - y1

    # 2. Resizeによる補正
    if resize_param is not None:
        target_h, target_w = resize_param
        scale_x = target_w / current_w
        scale_y = target_h / current_h

        K[0, 0] *= scale_x
        K[1, 1] *= scale_y
        K[0, 2] *= scale_x
        K[1, 2] *= scale_y

    return K.tolist()


def process_and_save_data(src_path, dst_path, crop_rect, resize_target, interpolation, undistort_maps=None):
    """
    画像(.png等)またはNumpy配列(.npy)を読み込み、歪み補正 -> Crop -> Resizeを行って保存する関数
    
    Args:
        undistort_maps: (map1, map2) のタプル。指定された場合、cv2.remapで歪み補正を行う
    """
    if not src_path.exists():
        return False

    is_npy = src_path.suffix == '.npy'

    # 読み込み
    if is_npy:
        img = np.load(str(src_path))
    else:
        img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False

    h_org, w_org = img.shape[:2]

    # 0. 歪み補正 (Undistort)
    # Cropの前に実施する必要があります（座標がずれるため）
    if undistort_maps is not None:
        map1, map2 = undistort_maps
        # セグメンテーション(inst-seg)やDepthの場合はNearest、RGBはLinearなどが望ましいが
        # ここでは呼び出し元の interpolation 引数に従う形にするか、あるいはremap用に分離するか検討が必要。
        # 今回は process_and_save_data の interpolation 引数を remap にも適用します。
        img = cv2.remap(img, map1, map2, interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT)
        
        # 補正によって画像サイズが変わる可能性がある場合、h_org, w_orgを更新したいところですが
        # initUndistortRectifyMapで元サイズと同じに指定していればそのままでOK。

    # 1. Crop処理
    if crop_rect:
        (x1, y1), (x2, y2) = crop_rect
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_org, x2), min(h_org, y2)
        img = img[y1:y2, x1:x2]

    # 2. Resize処理
    if resize_target:
        tgt_h, tgt_w = resize_target
        if img.shape[:2] != (tgt_h, tgt_w):
            img = cv2.resize(img, (tgt_w, tgt_h), interpolation=interpolation)

    # 保存
    if is_npy:
        np.save(str(dst_path), img)
    else:
        cv2.imwrite(str(dst_path), img)
    
    return True


def get_undistort_maps_and_newK(K_list, D_list, img_w, img_h):
    """
    歪み補正用のマップと、補正後の新しいカメラ行列を計算する
    """
    K = np.array(K_list, dtype=np.float64)
    D = np.array(D_list, dtype=np.float64)

    # 最適な新しいカメラ行列を計算 (alpha=0: 黒い部分を削除, alpha=1: 全画素保持)
    # ここではデータセット作成用として、重要な領域を残すため alpha=1 とするか、
    # あるいは 0 にして有効領域のみにするかはお好みですが、今回は 0 (有効領域のみ) とします。
    # 必要に応じて alpha=1 に変更してください。
    alpha = 0
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (img_w, img_h), alpha, (img_w, img_h))
    
    # マップ生成
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (img_w, img_h), cv2.CV_16SC2)
    
    return map1, map2, new_K


def main():
    if len(sys.argv) != 2:
        print("Usage: python annotate_frames.py <datacfg.json>")
        sys.exit(1)

    # 設定読み込み
    cfg_path = Path(sys.argv[1])
    with open(cfg_path, "r") as f:
        config = json.load(f)

    # --- パス設定 ---
    src_root_dir = Path(config["target_mov"])
    dst_root_dir = Path(config["out_dir"])

    def get_root_path(key_name, folder_name):
        if key_name in config and config[key_name]:
            return Path(config[key_name])
        candidate = src_root_dir / folder_name
        if candidate.exists():
            return candidate
        return None

    # 画像サイズ取得のために一旦ダミーでLeftの最初の画像を読み込みたいが、
    # 確実なのは crop/resize 設定の前の元解像度を知ること。
    # ここでは config の "crop" の情報などから推測せず、最初の画像を実際に読んでサイズ取得等を後半で行う。

    # データタイプ定義
    data_types = {
        "inst_seg": {
            "root": get_root_path("inst_seg_root", "inst-seg"),
            "sub_dir": "inst-seg",
            "ext": ".png",
            "interp": cv2.INTER_NEAREST,
            "use_undistort": "left" # Leftカメラ基準
        },
        "inst_seg_overlay": {
            "root": get_root_path("inst_seg_overlay_root", "inst-seg_overlay"),
            "sub_dir": "inst-seg_overlay",
            "ext": ".png",
            "interp": cv2.INTER_LINEAR,
            "use_undistort": "left" # RGB可視化なのでLeft基準かつLinear
        },
        "depth": {
            "root": get_root_path("depth_root", "depth"),
            "sub_dir": "depth",
            "ext": ".npy", 
            "interp": cv2.INTER_NEAREST,
            "use_undistort": "left" # Depthも通常Left基準であればLeftで補正
        },
        "depth_view": {
            "root": get_root_path("depth_view_root", "depth_view"),
            "sub_dir": "depth_view",
            "ext": ".png",
            "interp": cv2.INTER_LINEAR,
            "use_undistort": "left"
        },
        "occ": {
            "root": get_root_path("occ_root", "occ"),
            "sub_dir": "occ",
            "ext": ".png",
            "interp": cv2.INTER_NEAREST,
            "use_undistort": "left"
        }
    }

    # 画像処理パラメータ
    crop_rect = config.get("crop")
    resize_target = config.get("resize")

    if not dst_root_dir.exists():
        dst_root_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(cfg_path, dst_root_dir / "datacfg.json")

    source_fps = config.get("source_fps", 30)
    target_fps = config.get("target_fps", 30)
    frame_step = source_fps / target_fps
    
    dataset_meta = []
    
    # ---------------------------------------------------------
    # キャリブレーション準備 (ループの外で計算)
    # ---------------------------------------------------------
    calib_cfg = config["calibration"]
    undistort_maps = {} # 'left': (m1, m2), 'right': (m1, m2)
    new_Ks = {}         # 'left': new_K_array, 'right': new_K_array
    
    # 初期化フラグ（画像サイズ取得用）
    maps_initialized = False

    for i, seg_info in enumerate(config["segments"]):
        src_start, src_end = seg_info["range"]
        split_name = seg_info["split"]
        seq_name = f"seq{i+1}"
        seq_dir = dst_root_dir / seq_name

        sub_dirs_to_create = ["left", "right", "depth", "depth_view", "occ", "inst-seg", "inst-seg_overlay"]
        for d in sub_dirs_to_create:
            (seq_dir / d).mkdir(parents=True, exist_ok=True)

        dst_start_idx = math.ceil(src_start / frame_step)
        dst_end_idx = math.floor(src_end / frame_step)

        print(f"Generating {seq_name} ({split_name}): {dst_start_idx}-{dst_end_idx}")

        for dst_frame_idx in range(dst_start_idx, dst_end_idx + 1):
            src_frame_idx = int(dst_frame_idx * frame_step)
            
            src_fname_png = f"{src_frame_idx:05d}.png"
            src_fname_npy = f"{src_frame_idx:05d}.npy"
            dst_fname_png = f"{dst_frame_idx:05d}.png"
            dst_fname_npy = f"{dst_frame_idx:05d}.npy"

            files_processed = False

            # --- 1. Left / Right 画像処理 ---
            for side in ["left", "right"]:
                src_path = src_root_dir / side / src_fname_png
                dst_path = seq_dir / side / dst_fname_png

                if src_path.exists():
                    # --- 初回のみ：画像サイズ取得してマップ生成 ---
                    if not maps_initialized:
                        img_tmp = cv2.imread(str(src_path))
                        if img_tmp is not None:
                            h_org, w_org = img_tmp.shape[:2]
                            
                            # Left Map & New K
                            m1_l, m2_l, nk_l = get_undistort_maps_and_newK(
                                calib_cfg["K1"], calib_cfg["dist1"], w_org, h_org
                            )
                            undistort_maps["left"] = (m1_l, m2_l)
                            new_Ks["left"] = nk_l

                            # Right Map & New K
                            m1_r, m2_r, nk_r = get_undistort_maps_and_newK(
                                calib_cfg["K2"], calib_cfg["dist2"], w_org, h_org
                            )
                            undistort_maps["right"] = (m1_r, m2_r)
                            new_Ks["right"] = nk_r

                            maps_initialized = True
                    # ----------------------------------------------

                    # マップがあれば渡す
                    current_maps = undistort_maps.get(side)
                    if process_and_save_data(src_path, dst_path, crop_rect, resize_target, cv2.INTER_LINEAR, current_maps):
                        files_processed = True

            # --- 2. その他 (Inst-Seg, Depth等) ---
            for key, info in data_types.items():
                if info["root"] is None:
                    continue

                input_fname = src_fname_npy if info["ext"] == ".npy" else src_fname_png
                output_fname = dst_fname_npy if info["ext"] == ".npy" else dst_fname_png
                src_p = info["root"] / input_fname
                dst_p = seq_dir / info["sub_dir"] / output_fname

                # このデータタイプがどのカメラ(left/right)に基づいているか
                side_ref = info.get("use_undistort")
                current_maps = undistort_maps.get(side_ref) if side_ref else None

                process_and_save_data(src_p, dst_p, crop_rect, resize_target, info["interp"], current_maps)


            # --- メタデータ保存 ---
            if files_processed and maps_initialized:
                # dataset.json に書き込む Calibration 情報を作成
                # ここでは new_K をさらに crop/resize したものを計算する
                
                final_calib = calib_cfg.copy()
                
                # 歪み係数は補正済みなので 0 にする
                final_calib["dist1"] = [0.0] * 5
                final_calib["dist2"] = [0.0] * 5
                
                # K1 (Left) の更新
                final_calib["K1"] = update_intrinsic_matrix(
                    new_Ks["left"], crop_rect, resize_target, h_org, w_org
                )
                
                # K2 (Right) の更新
                final_calib["K2"] = update_intrinsic_matrix(
                    new_Ks["right"], crop_rect, resize_target, h_org, w_org
                )

                frame_data = {
                    "frame_index": dst_frame_idx,
                    "original_frame_index": src_frame_idx,
                    "source_video": str(src_root_dir),
                    "split": split_name,
                    "paths": {
                        "left": f"{seq_name}/left/{dst_fname_png}",
                        "right": f"{seq_name}/right/{dst_fname_png}",
                        "depth": f"{seq_name}/depth/{dst_fname_npy}",
                        "depth_view": f"{seq_name}/depth_view/{dst_fname_png}",
                        "occ": f"{seq_name}/occ/{dst_fname_png}",
                        "inst-seg": f"{seq_name}/inst-seg/{dst_fname_png}",
                        "inst-seg_overlay": f"{seq_name}/inst-seg_overlay/{dst_fname_png}",
                    },
                    "calibration": final_calib,
                }
                dataset_meta.append(frame_data)

    with open(dst_root_dir / "dataset.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)
    print(f"Saved dataset metadata to {dst_root_dir / 'dataset.json'}")

if __name__ == "__main__":
    main()