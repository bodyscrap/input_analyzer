# Sample Data Directory

このディレクトリには、入力解析用のサンプル動画ファイルを配置します。

## 動画ファイルの配置

以下の形式の動画ファイルをこのディレクトリに配置してください：

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.webm`
- `.flv`

## 動画の要件

- **解像度**: 1920x1080 (Full HD)
- **フレームレート**: 60 FPS 推奨
- **コーデック**: H.264 推奨
- **入力インジケータ領域**: 画面上の (204, 182) に 336x768 ピクセルの領域に表示されていること

## 使用方法

### セル画像の一括抽出

```bash
# 30フレームごとに抽出（推奨）
cargo run --bin extract_all_cells --release -- sample_data input_cells 30

# 全フレームを抽出
cargo run --bin extract_all_cells --release -- sample_data input_cells 1
```

### 個別の動画を解析

```bash
cargo run --bin analyze_input --release -- sample_data/your_video.mp4
```

## .gitignore の設定

動画ファイルは `.gitignore` で除外されているため、Git リポジトリには含まれません。
各自で動画ファイルを用意してこのディレクトリに配置してください。

## 注意事項

- 動画ファイルは大容量になるため、Git で管理しないようにしてください
- バックアップは別途行うことを推奨します
- プライバシーに配慮し、個人情報が含まれる動画は使用しないでください