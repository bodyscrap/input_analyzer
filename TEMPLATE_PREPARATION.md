# テンプレート準備ガイド

## 概要

入力アイコンの自動認識には、各アイコンの代表的な画像（テンプレート）が必要です。このガイドでは、既に抽出した入力セル画像からテンプレートを準備する方法を説明します。

## 前提条件

`extract_all_cells`を使用して、動画から入力セル画像を抽出済みであること。

```bash
# まだ実行していない場合
cargo run --release --bin extract_all_cells -- sample_data input_cells 30
```

## ステップ1: テンプレート候補の自動抽出

`prepare_templates`ツールを使用して、テンプレート候補を自動抽出します。

```bash
cargo run --release --bin prepare_templates -- input_cells templates 50
```

### 引数

- `input_cells`: 入力セル画像が格納されているディレクトリ
- `templates`: テンプレート画像を保存するディレクトリ
- `50`: 各カテゴリから収集するサンプル数

### 出力

```
templates/
├── empty/              # 空（入力なし）のセル候補
├── unclassified/       # 未分類のセル（手動分類が必要）
└── classify.html       # 分類支援HTML
```

## ステップ2: 手動分類

### 2-1. 分類支援HTMLを開く

`templates/classify.html`をブラウザで開きます。

```bash
# Windows
start templates\classify.html

# Linux/macOS
open templates/classify.html
```

### 2-2. 画像を確認

HTMLには以下のカテゴリの画像が表示されます：

- **empty**: 空（入力なし）のセル候補
- **unclassified**: 未分類のセル（アイコンがあるセル）

各画像は96x96ピクセルに拡大表示されます（元は48x48ピクセル）。

### 2-3. 重要：インジケータの映り込みについて

**注意**: 1行目（row00）と2行目（row01）のセルには、画面上部のインジケータ（メーター）が固定位置に映り込みます。このインジケータは複数のパターンが存在します。

そのため、以下の2種類のテンプレートを準備する必要があります：

1. **通常テンプレート**（3-15行目用）: インジケータが映り込まないセル
2. **インジケータ映り込み用テンプレート**（0-2行目用）: インジケータが映り込んだセル

`prepare_templates`ツールは自動的にこれらを分類します：
- `unclassified/` - 通常のセル（3-15行目から抽出）
- `unclassified_indicator/` - インジケータ映り込みセル（0-2行目から抽出）

### 2-4. アイコンの種類を特定

`unclassified`および`unclassified_indicator`フォルダ内の画像を確認し、以下のアイコンを特定します：

#### 方向入力（8種類）

| アイコン | フォルダ名 | 説明 |
|---------|-----------|------|
| ↑ | `dir_up` | 上 |
| ↗ | `dir_up_right` | 右上 |
| → | `dir_right` | 右 |
| ↘ | `dir_down_right` | 右下 |
| ↓ | `dir_down` | 下 |
| ↙ | `dir_down_left` | 左下 |
| ← | `dir_left` | 左 |
| ↖ | `dir_up_left` | 左上 |

#### ボタン入力（5種類）

| アイコン | フォルダ名 | 説明 |
|---------|-----------|------|
| A_1 | `button_a1` | A_1ボタン |
| A_2 | `button_a2` | A_2ボタン |
| B | `button_b` | Bボタン |
| W | `button_w` | Wボタン |
| Start | `button_start` | Startボタン |

#### その他

| アイコン | フォルダ名 | 説明 |
|---------|-----------|------|
| (なし) | `empty` | 空（入力なし） |

### 2-5. フォルダ構造を作成

`templates`ディレクトリ内に、各アイコン用のフォルダを作成します。

**重要**: 通常用とインジケータ映り込み用の両方のフォルダを作成してください。

```bash
cd templates

# 【通常テンプレート用】方向入力用フォルダ
mkdir dir_up dir_up_right dir_right dir_down_right
mkdir dir_down dir_down_left dir_left dir_up_left

# 【通常テンプレート用】ボタン入力用フォルダ
mkdir button_a1 button_a2 button_b button_w button_start

# 【インジケータ映り込み用】方向入力用フォルダ
mkdir dir_up_indicator dir_up_right_indicator dir_right_indicator dir_down_right_indicator
mkdir dir_down_indicator dir_down_left_indicator dir_left_indicator dir_up_left_indicator

# 【インジケータ映り込み用】ボタン入力用フォルダ
mkdir button_a1_indicator button_a2_indicator button_b_indicator button_w_indicator button_start_indicator

# 空フォルダ（既に存在する場合はスキップ）
# mkdir empty
```

### 2-6. 画像を分類

`unclassified`および`unclassified_indicator`フォルダ内の画像を、適切なカテゴリフォルダに移動します。

**重要な分類ルール**:
- `unclassified/`（3-15行目）の画像 → 通常フォルダ（例：`dir_up/`, `button_a1/`）
- `unclassified_indicator/`（0-2行目）の画像 → インジケータ用フォルダ（例：`dir_up_indicator/`, `button_a1_indicator/`）

#### Windowsの場合

1. エクスプローラーで`templates/unclassified`を開き、画像を確認しながら該当する通常フォルダにドラッグ＆ドロップします。
2. エクスプローラーで`templates/unclassified_indicator`を開き、画像を確認しながら該当するインジケータ用フォルダにドラッグ＆ドロップします。

#### Linux/macOSの場合

```bash
cd templates/unclassified

# 例：sample_0001.pngが右方向の場合
mv sample_0001.png ../dir_right/

# 例：sample_0002.pngがA_1ボタンの場合
mv sample_0002.png ../button_a1/

# インジケータ映り込み版の分類
cd ../unclassified_indicator

# 例：sample_0010.pngが右方向（インジケータ映り込み）の場合
mv sample_0010.png ../dir_right_indicator/

# 例：sample_0011.pngがA_1ボタン（インジケータ映り込み）の場合
mv sample_0011.png ../button_a1_indicator/
```

#### 分類のコツ

1. **最初に明確なアイコンを分類**
   - はっきりと見えるアイコンから始める
   - 通常版とインジケータ版の両方で、各カテゴリに3-5個のサンプルがあれば十分

2. **不明瞭な画像はスキップ**
   - ぼやけている画像
   - 複数のアイコンが重なっている画像
   - 判別が困難な画像

3. **各カテゴリに最低3個のサンプルを用意**
   - 認識精度を上げるため、複数のバリエーションが重要
   - 理想的には5-10個のサンプル
   - **特に重要**: インジケータ映り込み版も必ず用意する（0-2行目の認識に必須）

4. **インジケータのパターン**
   - インジケータには複数のパターンが存在します
   - 異なるパターンのサンプルを含めると認識精度が向上します

## ステップ3: テンプレートの検証

分類が完了したら、各フォルダにサンプルが配置されているか確認します。

```bash
# Windows PowerShell
Get-ChildItem templates -Directory | ForEach-Object {
    $count = (Get-ChildItem $_.FullName -Filter *.png).Count
    Write-Host "$($_.Name): $count samples"
}

# Linux/macOS
for dir in templates/*/; do
    count=$(ls -1 "$dir"*.png 2>/dev/null | wc -l)
    echo "$(basename "$dir"): $count samples"
done
```

### 理想的な分類結果

```
empty: 50 samples
dir_up: 5 samples
dir_up_indicator: 5 samples           # インジケータ映り込み版
dir_up_right: 5 samples
dir_up_right_indicator: 4 samples
dir_right: 8 samples
dir_right_indicator: 7 samples
dir_down_right: 4 samples
dir_down_right_indicator: 3 samples
dir_down: 6 samples
dir_down_indicator: 5 samples
dir_down_left: 3 samples
dir_down_left_indicator: 3 samples
dir_left: 7 samples
dir_left_indicator: 6 samples
dir_up_left: 4 samples
dir_up_left_indicator: 4 samples
button_a1: 10 samples
button_a1_indicator: 8 samples
button_a2: 8 samples
button_a2_indicator: 7 samples
button_b: 9 samples
button_b_indicator: 8 samples
button_w: 6 samples
button_w_indicator: 5 samples
button_start: 5 samples
button_start_indicator: 4 samples
unclassified: 0 samples               # すべて分類済み
unclassified_indicator: 0 samples     # すべて分類済み
```

**注**: インジケータ映り込み版のサンプルが少ない場合でも、通常版のテンプレートで代用できる場合があります。

## ステップ4: テンプレートディレクトリの最終確認

最終的なディレクトリ構造：

```
templates/
├── dir_up/                  # 通常（3-15行目用）
│   ├── sample_0001.png
│   ├── sample_0002.png
│   └── ...
├── dir_up_indicator/        # インジケータ映り込み（0-2行目用）
│   ├── sample_0001.png
│   └── ...
├── dir_right/
│   └── ...
├── dir_right_indicator/
│   └── ...
├── button_a1/
│   └── ...
├── button_a1_indicator/
│   └── ...
├── button_a2/
│   └── ...
├── empty/
│   └── ...
└── classify.html  # 不要になったら削除可能
```

## トラブルシューティング

### 問題: 画像が小さくて見えにくい

**解決方法**: HTMLの画像サイズを大きく調整

`classify.html`を編集：

```css
.image-container img {
    width: 192px;  /* 96px から 192px に変更 */
    height: 192px;
}
```

### 問題: アイコンの種類が判別できない

**解決方法**: 元のフレーム画像を確認

```bash
# セル画像のパスから元のフレームを特定
# 例: input_cells/input_sample_01/frame_000630/row00_col01_input.png

# 元のフレーム画像を確認
start output/frames/frame_000630.png
```

### 問題: 空セルの判定が間違っている

**解決方法**: 

1. `empty`フォルダ内の画像を確認
2. アイコンが含まれている場合は適切なカテゴリに移動
3. `unclassified`または`unclassified_indicator`フォルダから空セルを`empty`に移動

### 問題: インジケータ映り込み版のサンプルが不足

**解決方法**:

1. より多くの0-2行目のセルを収集
   ```bash
   # サンプル数を増やして再実行
   cargo run --release --bin prepare_templates -- input_cells templates_more 100
   ```

2. `unclassified_indicator`フォルダを確認
3. 最低でも各カテゴリに3個のインジケータ版サンプルを用意

4. インジケータ版がない場合でも通常版で代用可能（精度は低下）

### 問題: サンプル数が少なすぎる

**解決方法**: より多くのサンプルを収集

```bash
# サンプル数を増やして再実行
cargo run --release --bin prepare_templates -- input_cells templates_more 200
```

## 次のステップ

テンプレートの準備が完了したら、入力認識を実行できます。

```bash
# 入力認識の実行（次のステップで実装）
cargo run --release --bin recognize_inputs -- input_cells/input_sample_01/frame_000630 templates
```

## ベストプラクティス

### 1. 多様なサンプルを収集

- 異なるフレームからサンプルを選ぶ
- 明るさが異なるサンプルを含める
- 同じアイコンでも微妙に違うものを選ぶ
- **インジケータのパターン**: 複数のインジケータパターンを含める（0-2行目）

### 2. 品質を重視

- ぼやけた画像は避ける
- 明確に判別できる画像のみを使用
- 少数の高品質なサンプルの方が、多数の低品質なサンプルより良い

### 3. バランスを考慮

- すべてのカテゴリに最低3-5個のサンプルを用意
- 特定のカテゴリに偏らないようにする
- **通常版とインジケータ版の両方**をバランスよく用意

### 4. 定期的な見直し

- 認識精度が低い場合は、テンプレートを追加・更新
- 誤認識されやすいアイコンは、より多くのサンプルを用意

## アノテーションの自動化（将来の拡張）

現在は手動分類ですが、将来的には以下の方法で自動化できます：

1. **クラスタリング**: 類似した画像を自動的にグループ化
2. **半教師あり学習**: 少数のラベル付きデータから学習
3. **アクティブラーニング**: 不確実な画像のみ人間が確認

## まとめ

テンプレート準備の流れ：

1. ✅ `extract_all_cells`で入力セルを抽出
2. ✅ `prepare_templates`でテンプレート候補を自動抽出（通常版＋インジケータ版）
3. ✅ `classify.html`で画像を確認
4. ✅ 画像を手動でカテゴリ別に分類
   - 通常版：`dir_up/`, `button_a1/` など
   - インジケータ版：`dir_up_indicator/`, `button_a1_indicator/` など
5. ✅ 各カテゴリに最低3-5個のサンプルを配置（通常版＋インジケータ版の両方）
6. ⏭️ 入力認識を実行（次のステップ）

これで、入力アイコン認識の準備が整いました！