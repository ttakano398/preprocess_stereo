import cv2
import sys
import argparse
from pathlib import Path


def add_frame_number(video_path, out_dir, debug=False):
    path = Path(video_path)
    # debugモード時はファイル名に _debug を付与
    suffix = "_debug" if debug else ""
    out_path = Path(out_dir) / f"{path.stem}_frame{suffix}.mp4"

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit("Error opening video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # デバッグ時の上限フレーム数計算 (10秒)
    max_frames = int(fps * 10) if debug else float("inf")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    print(f"Processing: {path.name} (Debug: {debug})")

    n = 0
    while True:
        if debug and n >= max_frames:
            print("\nDebug limit (10s) reached.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(
            frame,
            f"{n}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        print(f"\rFrame: {n}", end="")
        n += 1

    cap.release()
    writer.release()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add frame numbers to video.")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("out_dir", help="Path to output directory")
    parser.add_argument(
        "--debug", action="store_true", help="Process only the first 10 seconds"
    )

    args = parser.parse_args()
    add_frame_number(args.video_path, args.out_dir, args.debug)
