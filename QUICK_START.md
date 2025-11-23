# クイックスタートガイド

このガイドでは、ゲーム入力解析アプリの基本的な使い方を説明します。

## 必要な環境

- Rust 1.70以上
- GStreamer 1.20以上（動画のデコードに使用）
- Windows環境（このガイドではWindows向けに説明）

## インストール

### 1. GStreamerのインストール

1. [GStreamer公式サイト](https://gstreamer.freedesktop.org/download/)からMSVC 64-bit版をダウンロード
2. 開発版とランタイム版の両方をインストール
3. 環境変数を設定：
   - `GSTREAMER_1_0_ROOT_MSVC_X86_64`: インストールディレクトリ（例：`C:\gstreamer\1.0\msvc_x86_64\`）
   - `PATH`に`%GSTREAMER_1_0_ROOT_MSVC_X86_64%\bin`を追加

### 2. プロジェクトのビルド

```bash
# プロジェクトディレクトリに移動
cd input_analyzer

# ビルド
cargo build --release
```

## 基本的な使い方

### ステップ1: 動画からフレームを抽出

```bash
# サンプル動画からフレームを抽出
cargo run --release

# または特定の動画を指定
cargo run --release -- path/to/your/video.mp4
```

**出力先**: `output/frames/` ディレクトリに30フレームごとにPNG画像が保存されます。

**例**: `frame_000630.png`, `frame_000660.png`, ...

### ステップ2: 入力インジケータを解析

```bash
# デフォルト設定で解析
cargo run --release --bin analyze_input

# または特定のフレームを指定
cargo run --release --bin analyze_input_custom -- output/frames/frame_000630.png
```

**出力先**: `output/analysis/` ディレクトリ

**生成されるファイル**:
- `*_debug_grid.png` - グリッド線が描かれたデバッグ画像
- `*_indicator_region.png` - 入力インジケータ領域全体
- `*_input_rows/` - 各セルの画像（16行×7列=112個）

## 出力ファイルの見方

### デバッグ画像（debug_grid.png）

元の画像に以下が描画されています：
- **赤線**: 入力インジケータ領域の外枠
- **緑線**: セルの境界線（16行×7列）

このファイルで領域設定が正しいか確認できます。

### セル画像（input_rows/）

各セルが個別の画像として保存されます：

```
row00_col00_frame_count.png  # 行0, 列0（フレームカウント）
row00_col01_input.png        # 行0, 列1（入力アイコン）
row00_col02_input.png        # 行0, 列2（入力アイコン）
...
row15_col00_frame_count.png  # 行15, 列0（フレームカウント）
row15_col06_input.png        # 行15, 列6（入力アイコン）
```

**ファイル名の意味**:
- `rowXX`: 行番号（00～15）
  - 00 = 最も古い入力
  - 15 = 最新の入力
- `colXX`: 列番号（00～06）
  - 00 = フレームカウント
  - 01～06 = 入力アイコン
- サイズ: すべて48×48ピクセル

## 入力インジケータの構造

### レイアウト

```
┌─────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ COUNT   │ ICON 1 │ ICON 2 │ ICON 3 │ ICON 4 │ ICON 5 │ ICON 6 │ ← 行0（古）
├─────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│ COUNT   │ ICON 1 │ ICON 2 │ ICON 3 │ ICON 4 │ ICON 5 │ ICON 6 │ ← 行1
├─────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│   ...   │  ...   │  ...   │  ...   │  ...   │  ...   │  ...   │
├─────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│ COUNT   │ ICON 1 │ ICON 2 │ ICON 3 │ ICON 4 │ ICON 5 │ ICON 6 │ ← 行15（新）
└─────────┴────────┴────────┴────────┴────────┴────────┴────────┘
   列0       列1      列2      列3      列4      列5      列6
```

### 各列の意味

- **列0（フレームカウント）**: その入力状態が続いているフレーム数（0-99）
- **列1-6（入力アイコン）**: 押されているボタンや方向（最大6個まで）

### 入力の優先度（左から右へ）

1. 方向入力（↑↗→↘↓↙←↖、ニュートラルは非表示）
2. A_1 ボタン
3. A_2 ボタン
4. B ボタン
5. W ボタン
6. Start ボタン

## 領域設定のカスタマイズ

### デフォルト設定

- **位置**: (x, y) = (204, 182)
- **サイズ**: (width, height) = (336, 768)
- **グリッド**: 16行 × 7列
- **セルサイズ**: 48×48ピクセル

### カスタム設定で解析

異なる解像度や位置の場合：

```bash
cargo run --release --bin analyze_input_custom -- <画像パス> <x> <y> <width> <height> <rows> <cols>
```

例：
```bash
cargo run --release --bin analyze_input_custom -- output/frames/frame_000630.png 204 182 336 768 16 7
```

### 設定の調整方法

1. **デバッグ画像を確認**
   ```bash
   cargo run --release --bin analyze_input_custom -- output/frames/frame_000630.png
   ```

2. **`output/analysis/frame_000630_debug_grid.png`を開く**

3. **グリッド線と実際のセルの位置を比較**
   - 左にずれている → X座標を減らす
   - 右にずれている → X座標を増やす
   - 上にずれている → Y座標を減らす
   - 下にずれている → Y座標を増やす

4. **調整して再実行**

## よくある質問

### Q: フレーム抽出の間隔を変更できますか？

A: はい、`src/main.rs`の`frame_interval`を変更してください。

```rust
let config = FrameExtractorConfig {
    frame_interval: 60,  // 60フレームごと（約1秒ごと）
    // ...
};
```

### Q: すべてのフレームを抽出できますか？

A: はい、`frame_interval: 1`に設定してください。

### Q: JPEG形式で保存できますか？

A: はい、設定を変更してください。

```rust
let config = FrameExtractorConfig {
    image_format: "jpg".to_string(),
    jpeg_quality: 85,  // 品質 0-100
    // ...
};
```

### Q: グリッドがずれています

A: `analyze_input_custom`を使って座標を調整してください。

```bash
# 座標を微調整（例：X+5、Y-3）
cargo run --release --bin analyze_input_custom -- output/frames/frame_000630.png 209 179 336 768 16 7
```

### Q: 複数のフレームを一括で解析できますか？

A: 現在は1フレームずつですが、シェルスクリプトで自動化できます。

```bash
# Windows PowerShell
Get-ChildItem output\frames\*.png | ForEach-Object {
    cargo run --release --bin analyze_input_custom -- $_.FullName
}

# Linux/macOS
for file in output/frames/*.png; do
    cargo run --release --bin analyze_input_custom -- "$file"
done
```

## トラブルシューティング

### GStreamerが見つからない

```
error: failed to run custom build command for `gstreamer`
```

**解決方法**:
1. GStreamerが正しくインストールされているか確認
2. 環境変数`GSTREAMER_1_0_ROOT_MSVC_X86_64`が設定されているか確認
3. システムを再起動

### 画像ファイルが開けない

```
Error: 画像ファイルを開けませんでした
```

**解決方法**:
1. ファイルパスが正しいか確認
2. ファイルが存在するか確認
3. ファイルが破損していないか確認

### 入力インジケータ領域が画像範囲外

```
Error: 入力インジケータ領域が画像範囲外です
```

**解決方法**:
1. 画像の解像度を確認（1920x1080を想定）
2. `analyze_input_custom`で座標を調整
3. デバッグ画像で位置を確認

## 次のステップ

1. **入力アイコンの認識** - OCRや画像マッチングで入力内容を識別
2. **フレームカウントの読み取り** - 数字認識で持続時間を取得
3. **統計分析** - APM（Actions Per Minute）の計算
4. **可視化** - グラフやヒートマップの生成

詳細は以下のドキュメントを参照してください：
- `README.md` - プロジェクト全体の概要
- `USAGE.md` - 詳細な使用方法
- `INPUT_ANALYSIS.md` - 入力解析の詳細ガイド

## サンプルワークフロー

```bash
# 1. 動画からフレームを抽出
cargo run --release

# 2. 特定のフレームを解析
cargo run --release --bin analyze_input_custom -- output/frames/frame_000630.png

# 3. デバッグ画像で確認
explorer output\analysis\frame_000630_debug_grid.png

# 4. 抽出されたセル画像を確認
explorer output\analysis\frame_000630_input_rows

# 5. 必要に応じて座標を調整して再実行
cargo run --release --bin analyze_input_custom -- output/frames/frame_000630.png 204 182 336 768 16 7
```

## サポート

問題が発生した場合は、以下を確認してください：
1. デバッグ画像でグリッド位置を確認
2. ビルドログでエラーメッセージを確認
3. ドキュメントのトラブルシューティングセクションを参照

それでも解決しない場合は、Issuesで報告してください。