# train_model - 入力解析モデル学習ツール

学習データディレクトリから入力種別ごとに分類する機械学習モデルを作成し、tar.gz形式で保存します。

## 機能

- 学習データディレクトリ内のサブディレクトリ名を自動的にクラスラベルとして検出
- オプションでボタンラベルのリストを指定可能
- 学習済みモデル、入力動画解像度情報、解析対象範囲情報、ボタンラベルリストをtar.gzにパッケージング
- コマンドラ インン引数で柔軟に設定変更可能

## 使用方法

### 基本的な使い方

```bash
cargo run --bin train_model --features ml --release -- --data-dir training_data
```

### オプション一覧

```
Options:
  -d, --data-dir <DATA_DIR>
          学習データディレクトリ（各サブディレクトリがクラスラベル）
          [default: training_data]

  -o, --output <OUTPUT>
          出力モデルのパス（.tar.gz拡張子は自動追加）
          [default: models/icon_classifier]

  -b, --buttons <BUTTONS>
          ボタンラベルのカンマ区切りリスト（方向入力とothersを除く）
          例: "A1,A2,B,W,Start"

  -e, --epochs <EPOCHS>
          エポック数
          [default: 50]

      --batch-size <BATCH_SIZE>
          バッチサイズ
          [default: 8]

      --learning-rate <LEARNING_RATE>
          学習率
          [default: 0.001]

      --val-ratio <VAL_RATIO>
          検証データの割合（0.0-1.0）
          [default: 0.2]

  -h, --help
          Print help
```

### 使用例

#### 1. デフォルト設定で学習

```bash
cargo run --bin train_model --features ml --release
```

- 学習データ: `training_data/`
- 出力: `models/icon_classifier.tar.gz`
- エポック: 50
- バッチサイズ: 8
- 学習率: 0.001
- 検証データ割合: 20%

#### 2. カスタム設定で学習

```bash
cargo run --bin train_model --features ml --release -- \
  --data-dir my_training_data \
  --output models/my_model \
  --buttons "A1,A2,B,W,Start" \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 0.0005
```

#### 3. 高速テスト学習（少ないエポック）

```bash
cargo run --bin train_model --features ml --release -- \
  --epochs 10 \
  --batch-size 4
```

## 学習データディレクトリの構造

```
training_data/
├── btn_a1/          # A1ボタンの画像
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── btn_a2/          # A2ボタンの画像
│   ├── image_001.png
│   └── ...
├── btn_b/           # Bボタンの画像
│   └── ...
├── btn_w/           # Wボタンの画像
│   └── ...
├── btn_start/       # Startボタンの画像
│   └── ...
├── dir_1/           # 方向キー（左下）の画像
│   └── ...
├── dir_2/           # 方向キー（下）の画像
│   └── ...
├── dir_3/           # 方向キー（右下）の画像
│   └── ...
├── dir_4/           # 方向キー（左）の画像
│   └── ...
├── dir_6/           # 方向キー（右）の画像
│   └── ...
├── dir_7/           # 方向キー（左上）の画像
│   └── ...
├── dir_8/           # 方向キー（上）の画像
│   └── ...
├── dir_9/           # 方向キー（右上）の画像
│   └── ...
└── empty/           # 入力なし（ニュートラル）の画像
    └── ...
```

**重要:** サブディレクトリ名がクラスラベルとして自動検出されます。

## 出力ファイル（tar.gz形式）

学習後、指定した出力パスに `.tar.gz` ファイルが生成されます。

### ファイル内容

```
icon_classifier.tar.gz
├── metadata.json    # メタデータ
└── model.bin        # 学習済みモデルの重み
```

### metadata.json の内容

```json
{
  "button_labels": ["A1", "A2", "B", "W", "Start"],
  "image_width": 48,
  "image_height": 48,
  "tile_x": 252,
  "tile_y": 902,
  "tile_width": 48,
  "tile_height": 48,
  "columns_per_row": 6,
  "model_input_size": 48,
  "num_epochs": 50,
  "trained_at": "2025-12-09T12:34:56+09:00"
}
```

- **button_labels**: ボタンラベルのリスト（方向入力とothersを除く）
- **image_width, image_height**: 学習データの画像サイズ
- **tile_x, tile_y**: 解析対象タイルの開始座標
- **tile_width, tile_height**: タイルのサイズ
- **columns_per_row**: 解析対象の列数
- **model_input_size**: モデルへの入力画像サイズ
- **num_epochs**: 学習エポック数
- **trained_at**: 学習完了時刻

## GUIアプリでの使用

生成された `.tar.gz` ファイルは、GUIアプリケーション（`input_editor_gui`）で直接読み込むことができます：

1. GUIアプリを起動
2. 「設定」→「モデルを選択」
3. 生成された `.tar.gz` ファイルを選択
4. 動画から入力履歴を自動抽出

## 学習の流れ

1. **データ収集**: `input_editor_gui` の「学習データ生成」機能でタイル画像を収集
2. **手動分類**: タイル画像を各クラスフォルダに手動で振り分け
3. **学習**: `train_model` でモデルを学習
4. **評価**: GUIアプリで動画から入力を抽出して精度を確認
5. **反復**: 不正確な部分のデータを追加して再学習

## 注意事項

- 画像ファイルは PNG 形式を推奨
- 各クラスに十分な数の画像（少なくとも50枚以上）を用意することを推奨
- GPU (WGPU) バックエンドを使用（CUDA/OpenCL不要）
- メモリ効率のため、画像は処理時にオンデマンドで読み込まれます

## トラブルシューティング

### エラー: "データディレクトリが見つかりません"

`--data-dir` オプションで正しいパスを指定してください。

### エラー: "トレーニングディレクトリにサブディレクトリが見つかりませんでした"

学習データディレクトリ内にクラスフォルダ（サブディレクトリ）が存在することを確認してください。

### 学習が遅い

- `--batch-size` を増やす（メモリに余裕がある場合）
- `--epochs` を減らす（テスト用）
- GPUが正しく認識されているか確認

### メモリ不足

- `--batch-size` を減らす（4 または 2）
- 学習データの枚数を減らす
