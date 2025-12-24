import sys
import json
import shutil
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: python annotate_frames.py <datacfg.json>")
        sys.exit(1)

    # 設定読み込み
    cfg_path = Path(sys.argv[1])
    with open(cfg_path, "r") as f:
        config = json.load(f)

    target_mov = config["target_mov"]
    root_dir = Path(config["out_dir"]) / target_mov
    
    # datacfg.jsonを出力先にコピー
    if not root_dir.exists():
        print(f"Error: Directory {root_dir} does not exist. Run extract_stereo.py first.")
        sys.exit(1)
    shutil.copy(cfg_path, root_dir / "datacfg.json")

    dataset_meta = [] # dataset.json用のリスト

    # セグメントごとに処理
    for i, seg_info in enumerate(config["segments"]):
        start, end = seg_info["range"]
        split_name = seg_info["split"]
        
        seq_name = f"seq{i+1}"
        seq_dir = root_dir / seq_name

        # ディレクトリ作成
        for sub in ["left", "right", "depth", "occ", "inst-seg"]:
            (seq_dir / sub).mkdir(parents=True, exist_ok=True)

        print(f"Generating {seq_name} ({split_name}): Frame {start}-{end}...")

        # フレーム処理
        for frame_idx in range(start, end + 1):
            fname = f"{frame_idx:05d}.png"
            
            # 画像コピー
            for side in ["left", "right"]:
                src = root_dir / side / fname
                dst = seq_dir / side / fname
                if src.exists():
                    shutil.copy(src, dst)

            # メタデータ作成
            frame_data = {
                "frame_index": frame_idx,
                "source_video": target_mov,
                "split": split_name,
                "paths": {
                    "left": f"{target_mov}/{seq_name}/left/{fname}",
                    "right": f"{target_mov}/{seq_name}/right/{fname}",
                    "depth": f"{target_mov}/{seq_name}/depth/{fname}",
                    "occ": f"{target_mov}/{seq_name}/occ/{fname}",
                    "inst-seg": f"{target_mov}/{seq_name}/inst-seg/{fname}"
                },
                "calibration": config["calibration"]
            }
            dataset_meta.append(frame_data)

    # dataset.json の書き出し
    with open(root_dir / "dataset.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)
    print(f"Saved dataset metadata to {root_dir / 'dataset.json'}")

if __name__ == "__main__":
    main()