# .gitignore 設定ガイド

このドキュメントでは、プロジェクトの `.gitignore` ファイルの設定内容について説明します。

## 概要

このプロジェクトでは、大容量のデータファイル（画像、動画、モデルのweightなど）をGit管理から除外しています。これにより、リポジトリのサイズを小さく保ち、クローン時間を短縮します。

## 除外されるファイル・ディレクトリ

### 🔴 大容量データ（必須除外）

以下のディレクトリは数GB～数十GBになる可能性があるため、必ず除外されます：

| ディレクトリ | サイズ目安 | 内容 |
|------------|-----------|------|
| `input_cells_all/` | **4.1GB** | 全動画から抽出されたセル画像（数十万枚） |
| `training_data/` | 12MB | 学習用データセット（数千枚） |
| `models/` | 29MB | 学習済みモデルのweight（.mpkファイルなど） |
| `templates/` | 数MB | テンプレート画像 |
| `output/` | 可変 | 解析結果の出力 |

### 📹 動画ファイル

すべての動画形式が除外されます：

```
*.mp4, *.avi, *.mov, *.mkv, *.webm, *.flv, *.m4v, *.3gp, *.wmv, *.mpg, *.mpeg
```

動画ファイルは通常数十MB～数GBになるため、Gitには含めません。

### 🖼️ 画像ファイル

一般的な画像形式が除外されます：

```
*.png, *.jpg, *.jpeg, *.bmp, *.gif, *.tiff, *.webp, *.ico, *.svg
```

**例外：** 以下の画像は管理対象として残されます：
- `input_icon_samples/` - サンプルアイコン（小さいサイズ）
- ドキュメント内の画像 (`docs/**/*.png`, `.github/**/*.png`)

### 🤖 モデルファイル

機械学習モデルのweightファイルが除外されます：

```
*.mpk       # Burn形式
*.pt, *.pth # PyTorch形式
*.ckpt      # チェックポイント
*.h5        # Keras/TensorFlow形式
*.onnx      # ONNX形式
*.bin       # バイナリweight
```

## 管理されるファイル

以下のファイルはGit管理対象として残されます：

### ✅ ソースコード

- `src/**/*.rs` - Rustソースコード
- `Cargo.toml` - パッケージ設定
- `.github/` - GitHub Actions設定

### ✅ ドキュメント

- `README.md` - プロジェクト説明
- `*.md` - すべてのMarkdownドキュメント
- `input_icon_samples/` - サンプル画像（参考用）

### ✅ 設定ファイル

- `.gitignore` - Git除外設定
- `.github/copilot-instructions.md` - 開発ガイドライン

## ディレクトリサイズの確認

現在のディレクトリサイズを確認するには：

```bash
du -sh models/ training_data/ input_cells_all/
```

出力例：
```
29M     models/
12M     training_data/
4.1G    input_cells_all/
```

## データの再生成方法

除外されたデータは、以下のコマンドで再生成できます：

### 1. セル画像の抽出

```bash
# すべての動画からセル画像を抽出
cargo run --release --bin extract_all_cells -- sample_data input_cells_all 1
```

### 2. トレーニングデータの収集

```bash
# input_icon_samplesからトレーニングデータを作成
cargo run --release --bin collect_training_data
```

### 3. モデルの学習

```bash
# モデルを学習（50エポック、バッチサイズ16）
cargo run --release --bin train_model --features ml -- training_data 50 16
```

### 4. トレーニングデータの更新（オプション）

```bash
# 学習済みモデルを使って高品質なデータを自動収集
cargo run --release --bin collect_with_model --features ml -- --target 120 --confidence 0.98
```

## 注意事項

### ⚠️ 大容量ファイルのコミット防止

以下のサイズのファイルは絶対にコミットしないでください：

- **100MB以上** - GitHubの制限によりpushできません
- **10MB以上** - リポジトリが肥大化します
- **1MB以上** - 画像や動画の可能性があります

### 💡 Git LFS について

非常に大きなファイルを管理する必要がある場合は、Git LFSの使用を検討してください：

```bash
# Git LFSのインストール（必要に応じて）
git lfs install

# 特定のファイルタイプをLFSで管理
git lfs track "*.mp4"
git lfs track "*.mpk"
```

ただし、このプロジェクトでは基本的にデータの再生成が容易なため、Git LFSは不要です。

## カスタマイズ

### CSV出力を除外したい場合

`.gitignore` の以下の行のコメントを解除してください：

```gitignore
# CSV出力（必要に応じてコメントアウト解除）
# *.csv
```

↓

```gitignore
# CSV出力
*.csv
```

### 特定のディレクトリを管理対象にしたい場合

除外されているディレクトリ内の特定のファイルを管理対象にするには：

```gitignore
# 基本的にtraining_dataを除外
/training_data/

# ただしREADMEは管理対象
!/training_data/README.md
```

## トラブルシューティング

### 既に追跡されているファイルを除外したい

既にGitで追跡されているファイルを除外するには：

```bash
# キャッシュから削除（ファイル自体は残る）
git rm --cached -r models/
git rm --cached -r training_data/

# コミット
git commit -m "大容量データをgit管理から除外"
```

### .gitignoreが機能しない場合

キャッシュをクリアしてください：

```bash
git rm -r --cached .
git add .
git commit -m ".gitignoreを適用"
```

## まとめ

- ✅ 大容量データ（4.1GB以上）は除外済み
- ✅ モデルweightは除外済み
- ✅ 動画・画像ファイルは除外済み
- ✅ ソースコードとドキュメントは管理対象
- ✅ データの再生成は簡単なコマンドで可能

これにより、リポジトリは軽量に保たれ、クローンやpull操作が高速になります。