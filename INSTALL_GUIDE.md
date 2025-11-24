# サイバーボッツ入力履歴エディタ - インストールガイド

## 必要な環境

- Windows 11 (Windows 10でも動作する可能性があります)
- GStreamer 1.24.5以降

## インストール手順

### 1. GStreamerのインストール

動画ファイルの処理にGStreamerが必要です。

#### 方法1: Chocolateyを使用（推奨）

PowerShellを管理者権限で開き、以下のコマンドを実行:

```powershell
choco install gstreamer gstreamer-devel --version=1.24.5 -y
```

#### 方法2: 手動インストール

1. [GStreamer公式サイト](https://gstreamer.freedesktop.org/download/)からインストーラーをダウンロード
2. **MSVC 64-bit版**をダウンロード（MinGW版ではありません）
   - `gstreamer-1.0-msvc-x86_64-1.24.5.msi` (Runtime)
   - `gstreamer-1.0-devel-msvc-x86_64-1.24.5.msi` (Development)
3. 両方のインストーラーを実行
4. インストール時に「Complete」インストールを選択
5. デフォルトのインストール先: `C:\gstreamer\1.0\msvc_x86_64\`

#### インストール確認

コマンドプロンプトまたはPowerShellで以下を実行:

```powershell
gst-launch-1.0 --version
```

バージョン情報が表示されれば成功です。

### 2. アプリケーションのセットアップ

1. ダウンロードしたZIPファイルを任意のフォルダに解凍
2. 解凍したフォルダに以下のファイルがあることを確認:
   - `input_editor_gui.exe` - メインアプリケーション
   - `INSTALL_GUIDE.md` - このファイル
   - `USER_MANUAL.md` - 操作マニュアル
   - `README.md` - プロジェクト概要

### 3. 機械学習モデルのダウンロード

1. 別途配布される`model.mpk`ファイルをダウンロード
2. 任意の場所に保存（例: `C:\Users\<ユーザー名>\Documents\CyberbotsInputEditor\model.mpk`）

### 4. 初回起動と設定

1. `input_editor_gui.exe`をダブルクリックして起動
2. メニューバーの「設定」をクリック
3. 「モデルファイル」の「選択...」ボタンをクリック
4. ダウンロードした`model.mpk`ファイルを選択
5. 「現在: <ファイルパス>」と表示されれば設定完了

## トラブルシューティング

### アプリケーションが起動しない

- **GStreamerがインストールされているか確認**
  ```powershell
  gst-launch-1.0 --version
  ```
- **環境変数の確認**
  - `Path`に`C:\gstreamer\1.0\msvc_x86_64\bin`が含まれているか確認
  - 含まれていない場合は手動で追加してPCを再起動

### 動画ファイルを開くとエラーになる

- **モデルファイルが選択されているか確認**
  - 設定メニューでモデルファイルのパスを確認
  - 「未選択（動画抽出不可）」と表示されている場合は、モデルファイルを選択

- **対応している動画形式か確認**
  - 対応形式: MP4, AVI, MOV, MKV
  - コーデック: H.264推奨

### 動画が長すぎるエラー

- デフォルトでは2分(120秒)までの動画に対応
- 設定メニューの「動画長さ上限 (秒)」で上限を変更可能（最大600秒=10分）

## アンインストール

1. アプリケーションフォルダを削除
2. 必要に応じてGStreamerをアンインストール
   - コントロールパネル → プログラムと機能 → GStreamer を選択して削除

## サポート

問題が解決しない場合は、以下の情報を添えてIssueを作成してください:

- Windowsのバージョン
- GStreamerのバージョン
- エラーメッセージの全文
- 実行した操作の手順
