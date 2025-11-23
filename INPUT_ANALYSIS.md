# 入力インジケータ解析ガイド

## 概要

このドキュメントでは、ゲーム動画から入力インジケータを解析し、プレイヤーの入力履歴を抽出する方法について説明します。

## 入力インジケータの構造

### 基本情報

入力インジケータは、ゲーム画面内に表示される入力履歴を視覚的に示す領域です。

- **デフォルト位置**: (x, y) = (204, 182)
- **デフォルトサイズ**: (width, height) = (336, 768)
- **グリッド構造**: 16行 × 7列
- **セルサイズ**: 48×48ピクセル

### グリッドレイアウト

```
┌────────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ COUNT  │ IN 1 │ IN 2 │ IN 3 │ IN 4 │ IN 5 │ IN 6 │  ← 行0（最古）
├────────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ COUNT  │ IN 1 │ IN 2 │ IN 3 │ IN 4 │ IN 5 │ IN 6 │  ← 行1
├────────┼──────┼──────┼──────┼──────┼──────┼──────┤
│   ...  │  ... │  ... │  ... │  ... │  ... │  ... │
├────────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ COUNT  │ IN 1 │ IN 2 │ IN 3 │ IN 4 │ IN 5 │ IN 6 │  ← 行15（最新）
└────────┴──────┴──────┴──────┴──────┴──────┴──────┘
  列0      列1    列2    列3    列4    列5    列6
```

### 列の役割

#### 1列目: フレームカウント
- **表示内容**: 2桁の数字（00～99）
- **意味**: その入力状態が維持されているフレーム数
- **特徴**: 99を超えると0に戻る（ループ）
- **画像サイズ**: 48×48ピクセル

#### 2～7列目: 入力アイコン
- **表示内容**: 入力の種類を示すアイコン
- **最大表示数**: 6個（同時押しの組み合わせ）
- **画像サイズ**: 各48×48ピクセル
- **特徴**: 優先度順に左から右へ詰めて表示

### 入力の優先度

入力アイコンは以下の優先度で表示されます（高い順）：

1. **方向入力**（8種類）
   - ↗（右上）、→（右）、↘（右下）
   - ↓（下）、↙（左下）、←（左）
   - ↖（左上）、↑（上）
   - ニュートラル（中立）の場合は非表示

2. **A_1ボタン**
3. **A_2ボタン**
4. **Bボタン**
5. **Wボタン**
6. **Startボタン**

### 行の意味

- **最下行（行15）**: 最新の入力状態
- **最上行（行0）**: 最も古い入力状態
- **更新タイミング**: 入力状態が1つでも変化した時に新しい行が追加される
- **スクロール**: 新しい行が追加されると、古い行は上にスクロールする

## 使用方法

### 基本的な解析

デフォルト設定で入力インジケータを解析します：

```bash
cargo run --bin analyze_input
```

これにより以下が生成されます：
- `output/analysis/debug_grid.png` - グリッド線が描画されたデバッグ画像
- `output/analysis/indicator_region.png` - 抽出された入力インジケータ領域
- `output/analysis/input_rows/` - 各セルの画像ファイル

### カスタム設定での解析

特定のフレーム画像を解析する場合：

```bash
cargo run --bin analyze_input_custom -- output/frames/frame_000630.png
```

領域設定をカスタマイズする場合：

```bash
cargo run --bin analyze_input_custom -- <画像パス> <x> <y> <width> <height> <rows> <cols>
```

例：
```bash
cargo run --bin analyze_input_custom -- output/frames/frame_000630.png 204 182 336 768 16 7
```

### 出力ファイル

#### デバッグ画像
ファイル名: `*_debug_grid.png`

元の画像にグリッド線を重ねて表示します：
- 赤線: 入力インジケータ領域の外枠
- 緑線: セルの境界線

この画像を確認することで、領域設定が正しいかを視覚的に確認できます。

#### インジケータ領域画像
ファイル名: `*_indicator_region.png`

入力インジケータ領域全体を切り出した画像です。
グリッド構造全体を確認する際に便利です。

#### セル画像
ディレクトリ: `*_input_rows/`

各セルを個別の画像ファイルとして保存します：

```
row00_col00_frame_count.png  # 行0のフレームカウント
row00_col01_input.png        # 行0の入力アイコン1
row00_col02_input.png        # 行0の入力アイコン2
...
row15_col00_frame_count.png  # 行15のフレームカウント
row15_col01_input.png        # 行15の入力アイコン1
...
```

ファイル命名規則：
- `rowXX`: 行番号（00～15）
- `colXX`: 列番号（00～06）
- `frame_count`: フレームカウント画像（列0）
- `input`: 入力アイコン画像（列1～6）

## 領域設定の調整

### 手順

1. **デバッグ画像を確認**
   ```bash
   cargo run --bin analyze_input_custom -- output/frames/frame_000630.png
   ```
   
2. **`output/analysis/frame_000630_debug_grid.png`を開く**
   - グリッド線が入力インジケータに正しく重なっているか確認
   - ずれている場合は座標を調整

3. **座標を微調整**
   ```bash
   # X座標を+5、Y座標を-3調整する例
   cargo run --bin analyze_input_custom -- output/frames/frame_000630.png 221 186 326 759 16 7
   ```

4. **再度確認**
   - 満足できる結果が得られるまで繰り返す

### 調整のヒント

- **X座標/Y座標**: 入力インジケータの左上角の位置
- **Width/Height**: 入力インジケータ全体のサイズ
- **Rows/Cols**: グリッドの行数と列数（通常は変更不要）

一般的な調整：
- グリッドが右にずれている → X座標を減らす
- グリッドが下にずれている → Y座標を減らす
- セルが小さすぎる → Width/Heightを増やす
- セルが大きすぎる → Width/Heightを減らす

## プログラムからの使用

### Rustコードでの使用例

```rust
use input_analyzer::input_analyzer::{InputAnalyzer, InputIndicatorRegion};
use image;

fn main() -> anyhow::Result<()> {
    // 入力インジケータ領域を定義
    let region = InputIndicatorRegion::new(
        204,  // x
        182,  // y
        336,  // width
        768,  // height
        16,   // rows
        7     // cols
    );
    
    // 解析器を作成
    let analyzer = InputAnalyzer::new(region);
    
    // 画像を読み込み
    let image = image::open("output/frames/frame_000630.png")?;
    
    // すべての入力行を抽出
    let rows = analyzer.extract_all_rows(&image)?;
    
    // 各行を処理
    for (i, row) in rows.iter().enumerate() {
        println!("行{}: {}個の入力アイコン", i, row.input_icons.len());
        
        // フレームカウント画像を保存
        row.save_frame_count(format!("frame_count_{}.png", i))?;
        
        // 各入力アイコンを保存
        for (j, _icon) in row.input_icons.iter().enumerate() {
            row.save_input_icon(j, format!("input_{}_{}.png", i, j))?;
        }
    }
    
    Ok(())
}
```

### デフォルト設定を使用

```rust
use input_analyzer::input_analyzer::InputAnalyzer;

fn main() -> anyhow::Result<()> {
    // デフォルト設定で解析器を作成
    let analyzer = InputAnalyzer::default();
    
    // すべての入力行を抽出して保存
    analyzer.extract_and_save_all_rows(
        "output/frames/frame_000630.png",
        "output/my_analysis"
    )?;
    
    Ok(())
}
```

### デバッグ画像の作成

```rust
use input_analyzer::input_analyzer::InputAnalyzer;

fn main() -> anyhow::Result<()> {
    let analyzer = InputAnalyzer::default();
    
    // デバッグ画像を作成
    analyzer.save_debug_image(
        "output/frames/frame_000630.png",
        "output/debug.png"
    )?;
    
    Ok(())
}
```

## 次のステップ

入力インジケータからセル画像を抽出した後、以下の処理を行うことができます：

### 1. フレームカウントの認識
- OCR（光学文字認識）を使用して数字を読み取る
- テンプレートマッチングで0～99の画像パターンを識別
- → 各入力状態の持続時間を定量化

### 2. 入力アイコンの認識
- 各アイコンの画像パターンを学習
- テンプレートマッチングまたは機械学習で分類
- → 具体的な入力内容（方向、ボタン）を識別

### 3. 入力履歴の構築
- 認識結果を時系列データに変換
- JSON、CSV、SQLiteなどに保存
- → プレイヤーの入力パターンを分析可能に

### 4. 統計分析
- APM（Actions Per Minute）の計算
- ボタンの使用頻度
- コンボの検出
- 入力タイミングの精度分析

## トラブルシューティング

### 問題: グリッドがずれている

**原因**: 入力インジケータの位置や解像度が想定と異なる

**解決方法**:
1. デバッグ画像を確認
2. `analyze_input_custom`で座標を微調整
3. 異なる解像度の動画の場合は、スケール比率を計算して適用

### 問題: セル画像が正しく切り出されない

**原因**: Width/Heightの設定が不適切

**解決方法**:
1. 元画像で入力インジケータ全体のピクセル数を測定
2. Width/Heightを調整
3. デバッグ画像で確認

### 問題: 抽出されたセル画像が空白または意図しない内容

**原因**: 
- 入力インジケータが画面外にある
- 領域設定が大きくずれている
- 画像の解像度が異なる

**解決方法**:
1. 元の画像ファイルを開いて入力インジケータの位置を目視確認
2. 画像編集ソフトで正確な座標を測定
3. 測定した座標で再度実行

### 問題: 一部のフレームでしか解析できない

**原因**: 
- ゲーム中にカメラが動く
- 入力インジケータの表示/非表示が切り替わる

**解決方法**:
- 入力インジケータが常に表示されているフレームのみを選択
- 複数の領域設定を用意し、動的に切り替える処理を実装

## パフォーマンス最適化

### 大量のフレームを処理する場合

```rust
use rayon::prelude::*;
use input_analyzer::input_analyzer::InputAnalyzer;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let analyzer = InputAnalyzer::default();
    
    // フレームファイルのリストを取得
    let frame_files: Vec<PathBuf> = std::fs::read_dir("output/frames")?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|s| s == "png").unwrap_or(false))
        .collect();
    
    // 並列処理で解析（rayon クレートを使用）
    frame_files.par_iter().for_each(|frame_path| {
        let output_dir = format!("output/analysis/{}", 
            frame_path.file_stem().unwrap().to_str().unwrap());
        
        if let Err(e) = analyzer.extract_and_save_all_rows(frame_path, &output_dir) {
            eprintln!("エラー {}: {}", frame_path.display(), e);
        }
    });
    
    Ok(())
}
```

## まとめ

このガイドでは、入力インジケータの構造、解析方法、カスタマイズ方法について説明しました。

基本的な流れ：
1. 動画からフレームを抽出
2. 入力インジケータの位置を特定
3. グリッド構造に基づいてセル画像を抽出
4. 各セル画像を解析（OCR、パターン認識）
5. 入力履歴データを構築
6. 統計分析・可視化

この基盤を使って、ゲームプレイの詳細な分析が可能になります。