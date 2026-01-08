import sys
import json
import shutil
from pathlib import Path
import cv2
import numpy as np


def update_intrinsic_matrix(K_list, crop_param, resize_param, original_h, original_w):
    """
    K_list: リスト形式の3x3行列
    crop_param: [[x1, y1], [x2, y2]] or None
    resize_param: [h, w] or None
    original_h, original_w: 元画像のサイズ
    """
    K = np.array(K_list, dtype=np.float64)

    # 1. Cropによる補正 (主点の移動)
    current_h, current_w = original_h, original_w
    if crop_param is not None:
        (x1, y1), (x2, y2) = crop_param
        # cx, cy をシフト
        K[0, 2] -= x1
        K[1, 2] -= y1

        # サイズ更新
        current_w = x2 - x1
        current_h = y2 - y1

    # 2. Resizeによる補正 (焦点距離と主点のスケール)
    if resize_param is not None:
        target_h, target_w = resize_param

        scale_x = target_w / current_w
        scale_y = target_h / current_h

        # fx, fy, cx, cy にスケールを適用
        K[0, 0] *= scale_x  # fx
        K[1, 1] *= scale_y  # fy
        K[0, 2] *= scale_x  # cx
        K[1, 2] *= scale_y  # cy

    return K.tolist()


def main():
    if len(sys.argv) != 2:
        print("Usage: python annotate_frames.py <datacfg.json>")
        sys.exit(1)

    # 設定読み込み
    cfg_path = Path(sys.argv[1])
    with open(cfg_path, "r") as f:
        config = json.load(f)

    # --- 入力元と出力先の設定 ---
    # target_mov: 画像読み込み元のルート (例: /work/.../src_dir)
    src_root_dir = Path(config["target_mov"])

    # out_dir: 書き出し先のルート (例: /work/.../dst_dir)
    dst_root_dir = Path(config["out_dir"])
    # ---------------------------

    # 画像処理パラメータの取得
    crop_rect = config.get("crop")  # 例: [[100, 100], [500, 400]]
    resize_target = config.get("resize")  # 例: [256, 512] (H, W)

    # 出力先ディレクトリの作成
    if not dst_root_dir.exists():
        dst_root_dir.mkdir(parents=True, exist_ok=True)

    # datacfg.jsonを出力先にコピー
    shutil.copy(cfg_path, dst_root_dir / "datacfg.json")

    # FPS設定
    source_fps = config.get("source_fps", 30)
    target_fps = config.get("target_fps", 30)

    if target_fps > source_fps:
        print(
            f"Error: target_fps ({target_fps}) cannot be higher than source_fps ({source_fps})."
        )
        sys.exit(1)

    frame_step = int(source_fps / target_fps)
    print(f"FPS Conversion: {source_fps} -> {target_fps} (Step: {frame_step})")

    dataset_meta = []

    # 補正後のキャリブレーションデータを保持する変数
    modified_calibration = None

    # セグメントごとに処理
    for i, seg_info in enumerate(config["segments"]):
        start, end = seg_info["range"]
        split_name = seg_info["split"]

        seq_name = f"seq{i+1}"

        # 出力先のシーケンスディレクトリ (dst_root_dir 直下に作成)
        seq_dir = dst_root_dir / seq_name

        # ディレクトリ作成 (left_pngは削除)
        # depth, occ, inst-seg は保存先として確保
        for sub in ["left", "right", "depth", "occ", "inst-seg"]:
            (seq_dir / sub).mkdir(parents=True, exist_ok=True)

        print(
            f"Generating {seq_name} ({split_name}): Target Frames {start}-{end} (Source step: {frame_step})..."
        )

        for dst_frame_idx in range(start, end + 1):
            src_frame_idx = dst_frame_idx * frame_step
            
            # 入力ファイル名
            src_fname = f"{src_frame_idx:05d}.png"
            
            # 出力ファイル名 (.png と .npy)
            dst_fname_png = f"{dst_frame_idx:05d}.png"
            dst_fname_npy = f"{dst_frame_idx:05d}.npy"

            files_processed = False

            # 左右画像の処理
            for side in ["left", "right"]:
                src_path = src_root_dir / side / src_fname
                
                # PNGとして保存
                dst_path_png = seq_dir / side / dst_fname_png

                if src_path.exists():
                    # 画像読み込み
                    img = cv2.imread(str(src_path))
                    if img is None:
                        continue

                    h_org, w_org = img.shape[:2]

                    # 最初の画像のタイミングでキャリブレーション補正値を計算・固定
                    if modified_calibration is None:
                        modified_calibration = config["calibration"].copy()
                        if "K1" in modified_calibration:
                            modified_calibration["K1"] = update_intrinsic_matrix(
                                modified_calibration["K1"],
                                crop_rect,
                                resize_target,
                                h_org,
                                w_org,
                            )
                        if "K2" in modified_calibration:
                            modified_calibration["K2"] = update_intrinsic_matrix(
                                modified_calibration["K2"],
                                crop_rect,
                                resize_target,
                                h_org,
                                w_org,
                            )

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
                            img = cv2.resize(
                                img, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR
                            )

                    # --- 保存処理 (全てPNG) ---
                    cv2.imwrite(str(dst_path_png), img)

                    files_processed = True
                else:
                    pass

            if files_processed:
                # メタデータ作成
                final_calib = (
                    modified_calibration
                    if modified_calibration
                    else config["calibration"]
                )

                # パスを out_dir からの相対パスとして生成
                frame_data = {
                    "frame_index": dst_frame_idx,
                    "original_frame_index": src_frame_idx,
                    "source_video": str(
                        src_root_dir
                    ),  # 元動画のパスは参考用に絶対パスで記録
                    "split": split_name,
                    "paths": {
                        "left": f"{seq_name}/left/{dst_fname_png}",         # .png
                        "right": f"{seq_name}/right/{dst_fname_png}",       # .png
                        "depth": f"{seq_name}/depth/{dst_fname_npy}",       # .npy
                        "depth_view": f"{seq_name}/depth/{dst_fname_png}",  # .png (view)
                        "occ": f"{seq_name}/occ/{dst_fname_png}",           # .png
                        "inst-seg": f"{seq_name}/inst-seg/{dst_fname_png}", # .png
                    },
                    "calibration": final_calib,
                }
                dataset_meta.append(frame_data)
            else:
                print(
                    f"Warning: Source frame {src_fname} not found for target frame {dst_frame_idx}"
                )

    # dataset.json の書き出し
    with open(dst_root_dir / "dataset.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)
    print(f"Saved dataset metadata to {dst_root_dir / 'dataset.json'}")


if __name__ == "__main__":
    main()