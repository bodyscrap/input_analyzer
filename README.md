# Input Analyzer - ゲーム入力解析アプリ

ゲームの入力を解析するためのアプリケーションです。GStreamerで動画をデコードし、フレームを抽出してゲームの状態を分析します。

## 機能

### フレーム抽出
- GStreamerを使用した高速な動画デコード
- 動画ファイルからフレームを抽出
- フレーム抽出間隔の設定（全フレーム、またはN フレームごと）
- PNG/JPEG形式での出力
- 特定のフレーム番号や時間での抽出

### 入力セル一括抽出
- 複数の動画から全フレームの入力セルを自動抽出
- フレーム間隔の指定（全フレーム or N フレームごと）
- 並列処理による高速な一括処理
- 動画名/フレーム名/行列番号で階層的に整理された出力

### 入力インジケータ解析
- フレームから入力インジケータ領域を自動抽出
- グリッド状の入力行を個別のセル画像に分割
- フレームカウントと入力アイコンの分離
- デバッグ用グリッド表示
- カスタマイズ可能な領域設定

### 入力アイコン自動認識
- テンプレートマッチングによる入力アイコン認識
- 14種類の入力タイプに対応（方向8種類、ボタン5種類、空）
- インジケータ映り込み対応（0-2行目）
- 信頼度スコア付き認識結果
- テンプレート準備支援ツール付き

## 必要なもの

### システム要件
- Rust 1.70以上
- GStreamer 1.20以上
- OpenCV 4.x (C:\opencvにインストール済みを想定)

### GStreamerのインストール

#### Windows
1. GStreamer公式サイトからインストーラーをダウンロード: https://gstreamer.freedesktop.org/download/
2. **MSVC 64-bit**版をインストール（開発版とランタイム版の両方）
3. 環境変数に追加:
   - `GSTREAMER_1_0_ROOT_MSVC_X86_64`: GStreamerのインストールディレクトリ（例: `C:\gstreamer\1.0\msvc_x86_64\`）
   - `PATH`に`%GSTREAMER_1_0_ROOT_MSVC_X86_64%\bin`を追加

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav
```

#### macOS
```bash
brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly
```

### OpenCVの設定

このプロジェクトでは、OpenCVが`C:\opencv`にインストールされていることを前提としています。
異なる場所にインストールされている場合は、`.cargo/config.toml`を編集してパスを変更してください。

## ビルド方法

```bash
# 依存関係を含めてビルド
cargo build --release

# 開発ビルド
cargo build
```

## 使い方

### 1. フレーム抽出

```bash
# サンプル動画からフレームを抽出
cargo run
```

デフォルトでは、`sample_data/input_sample.mp4` から30フレームごとにフレームを抽出し、`output/frames/` ディレクトリに保存します。

### 2. 入力セルの一括抽出

すべての動画から入力セルを一括で抽出します：

```bash
# 全フレームを抽出（デフォルト）
cargo run --bin extract_all_cells -- sample_data

# 30フレームごとに抽出
cargo run --bin extract_all_cells -- sample_data input_cells 30

# 出力先を指定
cargo run --bin extract_all_cells -- sample_data my_cells 60
```

出力ディレクトリ構造：
```
input_cells/
├── video_name_01/
│   ├── frame_000000/
│   │   ├── row00_col00_frame_count.png
│   │   ├── row00_col01_input.png
│   │   └── ... (112個のセル画像)
│   ├── frame_000030/
│   └── ...
├── video_name_02/
└── ...
```

### 3. テンプレート準備

入力認識の前に、テンプレート画像を準備します：

```bash
# テンプレート候補を自動抽出
cargo run --release --bin prepare_templates -- input_cells templates 50

# 分類支援HTMLを開く
start templates\classify.html

# 画像を手動で分類（詳細はTEMPLATE_PREPARATION.mdを参照）
```

テンプレートディレクトリ構造：
```
templates/
├── dir_up/              # 通常（3-15行目用）
├── dir_up_indicator/    # インジケータ映り込み（0-2行目用）
├── dir_right/
├── dir_right_indicator/
├── button_a1/
├── button_a1_indicator/
├── empty/
└── ...
```

**重要**: 1-2行目にはインジケータ（メーター）が映り込むため、通常版とインジケータ版の両方のテンプレートが必要です。

### 4. 入力インジケータ解析

```bash
# デフォルト設定で解析
cargo run --bin analyze_input

# 特定のフレームを解析（カスタム設定可能）
cargo run --bin analyze_input_custom -- output/frames/frame_000630.png

# カスタム領域で解析
cargo run --bin analyze_input_custom -- output/frames/frame_000630.png 216 189 326 759 16 7
```

解析結果は以下のファイルとして保存されます：
- `output/analysis/*_debug_grid.png` - グリッド線が描画されたデバッグ画像
- `output/analysis/*_indicator_region.png` - 抽出された入力インジケータ領域
- `output/analysis/*_input_rows/` - 各行・各列のセル画像

### 5. 分類精度テスト（機械学習モデル使用）

学習済みモデルで動画から入力アイコンを分類し、クラスごとに振り分けます。

```bash
# デフォルト設定で実行（sample_data配下の動画を3フレームごとに処理）
cargo run --release --features ml --bin test_classification

# カスタム設定で実行
cargo run --release --features ml --bin test_classification -- <動画ディレクトリ> <フレーム間隔> <モデルパス> <出力ディレクトリ>

# 例：60フレームごとに処理
cargo run --release --features ml --bin test_classification -- sample_data 60 models/model.mpk test_results
```

出力構造：
```
test_results/
├── input_sample_01/
│   ├── btn_a1/
│   │   ├── frame_000000_0_conf0.952.png
│   │   └── ...
│   ├── btn_a2/
│   ├── dir_6/
│   ├── empty/
│   └── ...
├── input_sample_02/
│   └── ...
└── ...
```

各ファイル名には信頼度スコアが含まれます（例：`conf0.952` = 95.2%）

### プログラムのカスタマイズ

`src/main.rs` のコードを編集することで、以下の設定を変更できます：

```rust
let config = FrameExtractorConfig {
    frame_interval: 30,  // 抽出間隔（フレーム数）
    output_dir: PathBuf::from("output/frames"),  // 出力先
    image_format: "png".to_string(),  // 画像形式（png/jpg/jpeg）
    jpeg_quality: 95,  // JPEG品質（0-100）
};
```

### API使用例

#### すべてのフレームを抽出
```rust
use input_analyzer::frame_extractor::{FrameExtractor, FrameExtractorConfig};
use std::path::PathBuf;

let config = FrameExtractorConfig {
    frame_interval: 1,  // すべてのフレーム
    output_dir: PathBuf::from("output/all_frames"),
    image_format: "png".to_string(),
    jpeg_quality: 95,
};

let extractor = FrameExtractor::new(config);
extractor.extract_frames("path/to/video.mp4")?;
```

#### 特定のフレームを抽出
```rust
// フレーム番号で指定
extractor.extract_frame_at("path/to/video.mp4", 100)?;

// 時間で指定（秒）
extractor.extract_frame_at_time("path/to/video.mp4", 3.5)?;
```

#### 動画情報を取得
```rust
let info = FrameExtractor::get_video_info("path/to/video.mp4")?;
println!("解像度: {}x{}", info.width, info.height);
println!("FPS: {}", info.fps);
println!("再生時間: {:.2}秒", info.duration_sec);
```

## プロジェクト構造

```
input_analyzer/
├── src/
│   ├── lib.rs                  # ライブラリエントリポイント
│   ├── main.rs                 # メインプログラム（フレーム抽出）
│   ├── frame_extractor.rs      # フレーム抽出モジュール
│   ├── input_analyzer.rs       # 入力インジケータ解析モジュール
│   ├── input_recognizer.rs     # 入力認識モジュール
│   └── bin/
│       ├── analyze_input.rs           # 入力解析バイナリ
│       ├── analyze_input_custom.rs    # カスタム設定版
│       ├── extract_all_cells.rs       # 一括セル抽出
│       └── prepare_templates.rs       # テンプレート準備ツール
├── sample_data/
│   ├── input_sample.mp4        # サンプル動画
│   └── frame_sample.png        # サンプルフレーム
├── output/                     # 出力ディレクトリ（自動生成）
│   ├── frames/                 # 抽出されたフレーム
│   └── analysis/               # 入力解析結果
│       ├── *_debug_grid.png
│       ├── *_indicator_region.png
│       └── *_input_rows/
├── input_cells/                # 一括抽出された入力セル（自動生成）
│   ├── video_name_01/
│   │   └── frame_NNNNNN/
│   │       └── rowXX_colYY_*.png
│   └── ...
├── templates/                  # 入力認識用テンプレート画像（要作成）
│   ├── dir_up/                # 方向：上（通常）
│   ├── dir_up_indicator/      # 方向：上（インジケータ映り込み）
│   ├── button_a1/             # A_1ボタン（通常）
│   ├── button_a1_indicator/   # A_1ボタン（インジケータ映り込み）
│   ├── empty/                 # 空（入力なし）
│   └── ...
├── Cargo.toml                  # プロジェクト設定
├── README.md                   # このファイル
├── USAGE.md                    # 詳細な使用例
├── TEMPLATE_PREPARATION.md     # テンプレート準備ガイド
└── BATCH_EXTRACTION.md         # 一括抽出ガイド
```

## トラブルシューティング

### GStreamerが見つからない
```
error: failed to run custom build command for `gstreamer`
```
→ GStreamerが正しくインストールされているか確認してください。環境変数`GSTREAMER_1_0_ROOT_MSVC_X86_64`が正しく設定されているか確認してください。

### OpenCVが見つからない
```
error: failed to run custom build command for `opencv`
```
→ OpenCVが`C:\opencv`にインストールされているか確認してください。異なる場所にインストールされている場合は、`.cargo/config.toml`のパスを編集してください。

### 動画ファイルが開けない
```
Error: 動画ファイルを開けませんでした
```
→ 動画ファイルのパスが正しいか確認してください。GStreamerのプラグインが正しくインストールされているか確認してください（特にgst-plugins-ugly、gst-libavなど）。

### テンプレートが見つからない
```
警告: ↑のテンプレートが見つかりません
```
→ テンプレート画像を準備してください。`prepare_templates`を使用してテンプレート候補を抽出し、手動で分類してください。詳細は`TEMPLATE_PREPARATION.md`を参照。

### 入力インジケータ領域が画像範囲外
```
Error: 入力インジケータ領域が画像範囲外です
```
→ `analyze_input_custom` を使用して領域設定を調整してください。デバッグ画像を確認して適切な座標を決定してください。

### デコーダーが見つからない
```
Error: decodebinの作成に失敗しました
```
→ GStreamerのプラグインパッケージ（base、good、bad、ugly）がすべてインストールされているか確認してください。

## 技術スタック

- **言語**: Rust
- **動画デコード**: GStreamer
- **画像処理**: image クレート
- **エラーハンドリング**: anyhow

## 入力インジケータの仕様

入力インジケータは以下の構造を持ちます：
- **位置**: (x, y) = (204, 182)
- **サイズ**: (width, height) = (336, 768)
- **グリッド**: 16行 × 7列
- **セルサイズ**: 48×48ピクセル

### 列の意味
- **1列目**: フレームカウント（0-255、初期値1、255を超えると0に戻る）
- **2～7列目**: 入力アイコン（最大6個）

### 入力の優先度
1. 方向入力（8種類、ニュートラルは非表示）
2. A_1
3. A_2
4. B
5. W
6. Start

### 行の意味
- **最下行（15行目）**: 最新の入力
- **最上行（0行目）**: 最も古い入力
- 入力状態が変わるたびに行が更新される

### インジケータの映り込み（重要）
- **0-2行目**: 画面上部のインジケータ（メーター）が固定位置に映り込む
- インジケータには複数のパターンが存在
- 認識精度を上げるため、インジケータ映り込み用のテンプレートが必要

## 入力認識の仕様

### 認識可能な入力タイプ

| カテゴリ | 種類 | 説明 |
|---------|-----|------|
| 方向入力 | 8種類 | ↑↗→↘↓↙←↖ |
| ボタン入力 | 5種類 | A_1, A_2, B, W, Start |
| その他 | 1種類 | 空（入力なし） |

### テンプレート構造

各入力タイプに対して2種類のテンプレートを用意：
- **通常テンプレート**: 3-15行目用（インジケータなし）
- **インジケータ映り込み用**: 0-2行目用（インジケータあり）

### 認識アルゴリズム

1. 行番号を確認（0-2行目 or 3-15行目）
2. 適切なテンプレートセットを選択
3. 正規化相互相関によるテンプレートマッチング
4. 最も類似度の高い入力タイプを判定
5. 信頼度スコア（0.0-1.0）を算出

## パフォーマンス

### 一括セル抽出の処理時間（参考値）

| 動画 | フレーム数 | 抽出間隔 | 抽出セル数 | 処理時間 |
|------|-----------|---------|-----------|---------|
| 39秒 (60fps) | 2,355 | 30フレーム | 8,848 | 約30秒 |
| 31秒 (60fps) | 1,862 | 30フレーム | 7,000 | 約25秒 |
| 14秒 (60fps) | 835 | 30フレーム | 3,136 | 約12秒 |

※ 全フレーム抽出の場合は処理時間が大幅に増加します（30倍程度）

## 今後の予定

- [x] フレーム抽出機能
- [x] 入力インジケータ領域の抽出
- [x] 入力セルの一括抽出
- [x] テンプレート準備支援ツール
- [x] 入力アイコンの認識（テンプレートマッチング）
- [x] インジケータ映り込み対応
- [ ] 入力認識の実行ツール（次のステップ）
- [ ] フレームカウントの数字認識（OCR）
- [ ] 入力履歴の時系列データ化
- [ ] 統計情報の出力（APM、入力頻度など）
- [ ] 機械学習による認識精度向上
- [ ] GUI の実装
- [ ] リアルタイムフレーム解析
- [ ] マルチスレッド処理による高速化

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します！バグ報告や機能要望は Issues でお願いします。