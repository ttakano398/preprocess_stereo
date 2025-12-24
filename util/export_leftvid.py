import argparse
import cv2
import sys
import ast

def main():
    parser = argparse.ArgumentParser(description="Extract left view from line-by-line video.")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("out_path", type=str, help="Path to output video")
    parser.add_argument("range_str", type=str, help="Frame range, e.g. '[0, 100]'")
    parser.add_argument("--width", type=int, default=None, help="Output video width")
    parser.add_argument("--height", type=int, default=None, help="Output video height")
    
    args = parser.parse_args()

    # 範囲の解析
    try:
        start_frame, end_frame = ast.literal_eval(args.range_str)
    except (ValueError, SyntaxError):
        print("Error: Range format must be [start, end]")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {args.video_path}")
        sys.exit(1)

    # 元動画の情報
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(end_frame, total_frames - 1)

    # Line-by-Line抽出直後のサイズ (高さが半分)
    extracted_w = orig_w
    extracted_h = orig_h // 2

    # 最終的な出力サイズを決定 (指定がなければ抽出サイズそのまま)
    target_w = args.width if args.width is not None else extracted_w
    target_h = args.height if args.height is not None else extracted_h
    
    # リサイズが必要かどうかのフラグ
    do_resize = (target_w != extracted_w) or (target_h != extracted_h)

    # VideoWriterの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.out_path, fourcc, fps, (target_w, target_h))

    # シーク
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"Processing: {args.out_path}")
    print(f"Range: {start_frame} -> {end_frame}")
    print(f"Native Extracted Size: {extracted_w}x{extracted_h}")
    print(f"Output Size: {target_w}x{target_h} (Resize: {'Yes' if do_resize else 'No'})")

    current_idx = start_frame
    while current_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Line-by-Line処理: 偶数行のみ取得
        left_img = frame[0::2, :]

        # 必要であればリサイズ
        if do_resize:
            left_img = cv2.resize(left_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        writer.write(left_img)
        
        current_idx += 1
        if (current_idx - start_frame) % 100 == 0:
            print(f"Progress: {current_idx}/{end_frame}", end='\r')

    cap.release()
    writer.release()
    print(f"\nDone.")

if __name__ == "__main__":
    main()