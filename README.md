ステレオ画像に対して前処理を行うコード

## 機能
- `extract_stereo.py`: **Line-by-Line形式の左右フレーム分割**
  - `Usage: python extract_stereo.py <video_path> <out_dir>`
  - 動画ファイル名に基づいたディレクトリを作成し、その中に `left` / `right` フォルダを生成
  - フレームレートを自動認識
- `visualize_frames.py`: **アノテーション用動画の生成**
  - `Usage: python visualize_frames.py <video_path> <out_dir>`
  - フレーム番号を画面上に可視化した動画を出力
  - 動画が長すぎる and 細かいアノテーションが不要な場合は動画秒数からframe numを逆算した方が効率的

### `extract_stereo.py`: フレーム抽出
- **入力**: 
  - ステレオ動画ファイル (`{mov}.mp4`) ※Line-by-Line形式
  - 出力先親ディレクトリ (`outdir`)
- **出力**:
  ```text
  outdir/
  └── {mov}/          # 動画ファイル名と同名のディレクトリ
      ├── left/
      │   ├── 00000.png
      │   └── ...
      └── right/
          ├── 00000.png
          └── ...
  ```

### `visualize_frames.py`: フレーム可視化
- **入力**: `{mov}.mp4`, 出力先親ディレクトリ (`outdir`)
- **出力**: `{outdir}/{mov}_frame.mp4`
  - 左上にフレーム番号が明記された動画ファイル

### `annotate_frames.py`: フレームの選別 & データセットの整形
- **入力**: データ要件を明記した`datacfg.json`
- **出力**:
  ```text
  outdir/
  └── {mov}/          # 動画ファイル名と同名のディレクトリ
      ├── datacfg.json    # 入力設定のコピー
      ├── dataset.json    # 全フレームのメタデータ（パス, Split, Param）
      ├── seq1/
      │   ├── left/
      │   ├── right/
      │   ├── depth/
      │   ├── occ/
      │   └── inst-seg/
      └── seq2/
          ├── left/
          ├── right/
          └── .../
  ```
- `datacfg.json`の仕様
  - segments: 抽出フレームとsplitを指定する辞書のリスト
  - calibration: カメラパラメータ
  - 全フレーム情報を格納した以下の形式の`dataset.json`が生成される
```
{
  "out_dir": "./dataset_root",
  "target_mov": "surgery_video_01",
  "segments": [
    { "range": [10, 150], "split": "train" },
    { "range": [300, 450], "split": "val" },
    { "range": [600, 800], "split": "test" }
  ],
  "calibration": {
    "K1": [[1408.24, 0.0, 988.07], ...],
    "dist1": [[-0.15, ...]],
    "K2": [[1416.04, ...], ...],
    "dist2": [[-0.13, ...]],
    "R": [[0.99, ...], ...],
    "T": [[5.22], ...],
    "baseline": 5.258
  }
}
```

## インストール
```
python -m venv .venv
source .venv/bin/activate
pip install opencv-python
```