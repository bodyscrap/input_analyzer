# 設定ファイル (config.json)

このドキュメントでは、アプリケーションの設定ファイル `config.json` について説明します。

## 概要

`config.json` は、アプリケーションの動作設定を保存するJSONファイルです。
このファイルが存在する場合、アプリケーションは起動時に設定を読み込み、前回使用した設定を自動的に適用します。

## 設定ファイルの場所

設定ファイルは、プロジェクトのルートディレクトリに `config.json` という名前で保存されます。

```
input_analyzer/
├── config.json          <- ここに配置
├── Cargo.toml
├── src/
└── ...
```

## デフォルト設定ファイルの生成

設定ファイルが存在しない場合、アプリケーションは自動的にデフォルト設定を使用します。
明示的にデフォルト設定ファイルを生成したい場合は、以下のコマンドを実行してください：

```bash
cargo run --bin create_default_config
```

## 設定ファイルの構造

### 完全な例

```json
{
  "device_type": "Wgpu",
  "model": {
    "model_path": "models/icon_classifier.bin",
    "num_classes": 14,
    "dropout": 0.5
  },
  "training": {
    "num_epochs": 50,
    "batch_size": 8,
    "num_workers": 1,
    "learning_rate": 0.001,
    "seed": 42,
    "train_ratio": 0.8
  },
  "last_video_path": "sample_data/input_sample.mp4",
  "last_output_dir": "output"
}
```

### 設定項目の説明

#### `device_type` (文字列)

計算デバイスの種類を指定します。

- **`"Wgpu"`**: GPU (WGPU) バックエンドを使用（推奨・高速）
- **`"Cpu"`**: CPU (NdArray) バックエンドを使用（互換性重視）

**デフォルト値**: `"Wgpu"`

**例**:
```json
"device_type": "Wgpu"
```

#### `model` (オブジェクト)

機械学習モデルに関する設定です。

##### `model.model_path` (文字列)

使用するモデルファイルのパス。

**デフォルト値**: `"models/icon_classifier.bin"`

**例**:
```json
"model_path": "models/icon_classifier.bin"
```

##### `model.num_classes` (整数)

分類クラス数。入力アイコンの種類数に対応します。

**デフォルト値**: `14`

**例**:
```json
"num_classes": 14
```

##### `model.dropout` (小数)

ドロップアウト率（0.0〜1.0）。モデルの過学習を防ぎます。

**デフォルト値**: `0.5`

**例**:
```json
"dropout": 0.5
```

#### `training` (オブジェクト)

モデルの学習に関する設定です。

##### `training.num_epochs` (整数)

学習エポック数。学習の反復回数を指定します。

**デフォルト値**: `50`

**推奨範囲**: 10〜200

**例**:
```json
"num_epochs": 50
```

##### `training.batch_size` (整数)

バッチサイズ。一度に処理するサンプル数です。

**デフォルト値**: `8`

**推奨範囲**: 4〜32（GPUメモリに依存）

**例**:
```json
"batch_size": 8
```

##### `training.num_workers` (整数)

データローダーのワーカー数。

**デフォルト値**: `1`

**例**:
```json
"num_workers": 1
```

##### `training.learning_rate` (小数)

学習率。モデルの重み更新の大きさを制御します。

**デフォルト値**: `0.001` (1e-3)

**推奨範囲**: 0.0001〜0.01

**例**:
```json
"learning_rate": 0.001
```

##### `training.seed` (整数)

ランダムシード。再現性のために固定します。

**デフォルト値**: `42`

**例**:
```json
"seed": 42
```

##### `training.train_ratio` (小数)

トレーニングデータの割合（0.0〜1.0）。残りは検証データになります。

**デフォルト値**: `0.8` (80%学習、20%検証)

**例**:
```json
"train_ratio": 0.8
```

#### `last_video_path` (文字列、オプション)

最後に使用したビデオファイルのパス。
アプリケーションが自動的に記録します。

**デフォルト値**: `null`

**例**:
```json
"last_video_path": "sample_data/input_sample.mp4"
```

#### `last_output_dir` (文字列、オプション)

最後に使用した出力ディレクトリ。
アプリケーションが自動的に記録します。

**デフォルト値**: `null`

**例**:
```json
"last_output_dir": "output"
```

## 設定の動作

### 読み込み優先順位

1. **コマンドライン引数**: コマンドラインで明示的に指定された設定
2. **設定ファイル**: `config.json` に保存された設定
3. **デフォルト値**: プログラムに組み込まれたデフォルト値

### 自動保存

以下の操作を行うと、設定ファイルが自動的に更新されます：

- **モデル学習 (`train_model`)**: 学習設定とモデルパスを保存
- **入力履歴抽出 (`extract_input_history`)**: ビデオパス、出力ディレクトリ、モデルパスを保存
- **GUI編集 (`input_editor_gui`)**: デバイスタイプ、ファイルパスを保存

## 使用例

### GPU/CPUの切り替え

設定ファイルの `device_type` を変更することで、計算デバイスを切り替えられます。

**GPUを使用する場合**:
```json
{
  "device_type": "Wgpu"
}
```

**CPUを使用する場合**:
```json
{
  "device_type": "Cpu"
}
```

### カスタムモデルパスの設定

独自のモデルファイルを使用する場合：

```json
{
  "model": {
    "model_path": "models/my_custom_model.mpk"
  }
}
```

### 学習パラメータの調整

より高精度なモデルを学習したい場合：

```json
{
  "training": {
    "num_epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.0005
  }
}
```

## コマンドラインでの設定上書き

設定ファイルの値は、コマンドライン引数で上書きできます。

### モデル学習

```bash
# エポック数とバッチサイズを指定
cargo run --release --features ml --bin train_model -- training_data 100 16

# デバイスタイプも指定
cargo run --release --features ml --bin train_model -- training_data 100 16 gpu
```

### 入力履歴抽出

```bash
# モデルパスとデバイスを指定
cargo run --release --features ml --bin extract_input_history -- video.mp4 output.csv models/custom.mpk gpu
```

## トラブルシューティング

### 設定ファイルが読み込まれない

- ファイル名が `config.json` であることを確認してください
- ファイルの配置場所がプロジェクトルートであることを確認してください
- JSON形式が正しいか確認してください（オンラインのJSONバリデーターを使用）

### GPU設定でエラーが発生する

WGPUバックエンドでエラーが発生する場合は、一時的にCPUに切り替えてください：

```json
{
  "device_type": "Cpu"
}
```

### 設定をリセットしたい

`config.json` ファイルを削除すると、次回起動時にデフォルト設定が使用されます。

```bash
# Windowsの場合
del config.json

# または、デフォルト設定ファイルを再生成
cargo run --bin create_default_config
```

## 設定のバックアップ

重要な設定は、別の場所にコピーして保管することをお勧めします：

```bash
# バックアップを作成
copy config.json config.backup.json

# バックアップから復元
copy config.backup.json config.json
```

## まとめ

- 設定ファイルは `config.json` として保存される
- 前回使用した設定が自動的に適用される
- デバイスタイプ、モデル設定、学習パラメータをカスタマイズ可能
- コマンドライン引数で設定を上書きできる
- 設定は使用後に自動的に保存される

設定ファイルを活用することで、毎回同じオプションを指定する手間が省け、効率的にアプリケーションを使用できます。