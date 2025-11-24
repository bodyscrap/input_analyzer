# サイバーボッツ入力履歴エディタ - リリースノート

## バイナリ配布について

### ダウンロード

GitHubのReleasesページから最新版をダウンロードできます。

### 配布ファイル

- **サイバーボッツ入力履歴エディタ-vX.X.X.zip**
  - `input_editor_gui.exe` - メインアプリケーション
  - `INSTALL_GUIDE.md` - インストール手順
  - `USER_MANUAL.md` - 操作マニュアル
  - `README.md` - プロジェクト概要

### 機械学習モデル（別配布）

動画からの入力抽出機能を使用するには、別途機械学習モデルが必要です。

- **model.mpk** - 入力アイコン分類モデル
- リポジトリのReleasesページまたは別途指定された場所からダウンロード
- アプリケーション起動後、設定メニューからモデルファイルを選択

### モデルの配布方法

モデルファイルは以下のいずれかの方法で配布できます:

#### 方法1: GitHub Releases（推奨）

リポジトリ管理者がモデルファイルをReleasesページにアップロード

#### 方法2: 外部ストレージ

モデルファイルサイズが大きい場合は、Google Drive、OneDrive等にアップロード

## 開発者向け情報

### ビルド方法

```powershell
# GUIアプリケーションのビルド
cargo build --release --features gui,ml --bin input_editor_gui

# 出力先
target\release\input_editor_gui.exe
```

### リリース手順

1. バージョンタグを作成
   ```powershell
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. GitHub Actionsが自動的にビルド・リリース

3. モデルファイルを手動でReleasesページにアップロード

### 必要な依存関係

- Rust 1.70以降
- GStreamer 1.24.5以降（MSVC 64-bit版）
- Windows SDK

## ライセンス

このプロジェクトのライセンスについては、LICENSEファイルを参照してください。
