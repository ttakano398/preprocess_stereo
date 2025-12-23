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
      ├── datacfg.json
      ├── seq1/
      │   ├── left/
      │   ├── right/
      │   ├── depth/
      │   └── occ/
      └── seq2/
          ├── left/
          └── right/
  ```
- `datacfg.json`の例
  - 出力ディレクトリ: {outdir}
  - 選別する対象のディレクトリ: {mov}
  - アノテーションのフレーム数のリスト: 例: [[0, 100], [200, 409], ...]
    - このリストに含まれるフレームを利用
    - 連続するフレームは同じseqとして利用
    - このリストのlenを参照し、seq1, seq2, ...とseqを作成される
```
{
  "out_dir": "./dataset_root",
  "target_mov": "surgery_video_01",
  "segments": [
    [10, 150],
    [300, 450],
    [600, 800]
  ]
}
```

## インストール
```
python -m venv .venv
source .venv/bin/activate
pip install opencv-python
```