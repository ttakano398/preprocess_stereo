import cv2
import sys
from pathlib import Path


def extract_frames(video_path, out_dir):
    path = Path(video_path)
    out_root = Path(out_dir) / path.stem

    # ディレクトリ作成 (parents=Trueで親も作成、exist_okで上書き許容)
    (out_root / "left").mkdir(parents=True, exist_ok=True)
    (out_root / "right").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"Error: Cannot open {video_path}")

    # 自動認識した情報の表示
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Start: {width}x{height} @ {fps:.2f}fps ({total_frames} frames)")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Line-by-line分割
        # 奇数行(1,3,5...行目) -> index 0,2,4... -> Right
        # 偶数行(2,4,6...行目) -> index 1,3,5... -> Left
        left_img = frame[1::2, :]
        right_img = frame[0::2, :]

        # 保存
        cv2.imwrite(str(out_root / "left" / f"{idx:05d}.png"), left_img)
        cv2.imwrite(str(out_root / "right" / f"{idx:05d}.png"), right_img)

        print(f"\rProgress: {idx+1}/{total_frames}", end="")
        idx += 1

    cap.release()
    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_stereo.py <video_path> <out_dir>")
    else:
        extract_frames(sys.argv[1], sys.argv[2])
