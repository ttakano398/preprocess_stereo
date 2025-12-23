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

    root_dir = Path(config["out_dir"]) / config["target_mov"]
    segments = config["segments"]

    # datacfg.jsonを出力先にコピー
    if not root_dir.exists():
        print(
            f"Error: Directory {root_dir} does not exist. Run extract_stereo.py first."
        )
        sys.exit(1)
    shutil.copy(cfg_path, root_dir / "datacfg.json")

    # セグメントごとに処理
    for i, (start, end) in enumerate(segments):
        seq_dir = root_dir / f"seq{i+1}"

        # ディレクトリ作成 (left, right, depth, occ)
        for sub in ["left", "right", "depth", "occ"]:
            (seq_dir / sub).mkdir(parents=True, exist_ok=True)

        print(f"Generating seq{i+1} (Frame: {start}-{end})...")

        # フレームのコピー
        for frame_idx in range(start, end + 1):
            fname = f"{frame_idx:05d}.png"  # 5桁埋め

            for side in ["left", "right"]:
                src = root_dir / side / fname
                dst = seq_dir / side / fname

                if src.exists():
                    shutil.copy(src, dst)


if __name__ == "__main__":
    main()
