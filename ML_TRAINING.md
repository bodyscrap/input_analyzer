# 機械学習による入力アイコン認識

このドキュメントでは、Burnフレームワークを使用した機械学習ベースの入力アイコン認識システムの実装について説明します。

## 概要

現在、テンプレートマッチングベースの認識を使用していますが、機械学習（CNN）を使用することでより高精度な認識が可能になります。

## トレーニングデータの収集

### ステップ1: データ収集

`collect_training_data` ツールを使用して、各カテゴリのサンプル画像を自動収集します。

```bash
# 各カテゴリから100枚ずつ収集
cargo run --bin collect_training_data --release -- \
    input_icon_samples \
    input_cells \
    training_data \
    100 \
    0.5
```

**引数:**
- `input_icon_samples`: テンプレート画像ディレクトリ
- `input_cells`: 抽出済みセル画像ディレクトリ
- `training_data`: 出力先ディレクトリ
- `100`: 各カテゴリのサンプル数
- `0.5`: 空白判定の閾値

### ステップ2: データセット構造

収集されたデータは以下の構造で保存されます：

```
training_data/
├── dir_1/          # 左下入力
│   ├── sample_0000_0.850.png
│   ├── sample_0001_0.845.png
│   └── ...
├── dir_2/          # 下入力
├── dir_3/          # 右下入力
├── dir_4/          # 左入力
├── dir_6/          # 右入力
├── dir_7/          # 左上入力
├── dir_8/          # 上入力
├── dir_9/          # 右上入力
├── btn_a1/         # A1ボタン
├── btn_a2/         # A2ボタン
├── btn_b/          # Bボタン
├── btn_w/          # Wボタン
├── btn_start/      # Startボタン
├── empty/          # 空白（入力なし）
└── labels.txt      # ラベル定義
```

ファイル名の形式: `sample_{番号}_{類似度スコア}.png`

### ステップ3: データの確認

```bash
# 各カテゴリのサンプル数を確認
for dir in training_data/*/; do
    count=$(ls "$dir"*.png 2>/dev/null | wc -l)
    echo "$(basename $dir): ${count}枚"
done
```

## Burnフレームワークについて

### 現状の課題

Burn 0.15以降でビルドエラーが発生するため、現在は以下の代替アプローチを推奨します：

### 代替アプローチ1: PyTorchでの学習 + ONNX変換

最も実用的なアプローチです。

#### 1. PyTorchで学習スクリプトを作成

```python
# train_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class IconDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # ラベルマッピングを読み込み
        with open(os.path.join(root_dir, 'labels.txt'), 'r') as f:
            self.label_map = {line.split(':')[1].strip(): int(line.split(':')[0]) 
                            for line in f.readlines()}
        
        # サンプルを収集
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path):
                continue
            
            label = self.label_map.get(category)
            if label is None:
                continue
            
            for img_name in os.listdir(category_path):
                if img_name.endswith('.png'):
                    self.samples.append(os.path.join(category_path, img_name))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class IconCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(IconCNN, self).__init__()
        
        # 48x48 input
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # 48 -> 24 -> 12 -> 6
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 48 -> 24
        x = self.pool(self.relu(self.conv2(x)))  # 24 -> 12
        x = self.pool(self.relu(self.conv3(x)))  # 12 -> 6
        
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def train(data_dir, epochs=50, batch_size=32, lr=0.001):
    # データ変換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # データセット
    dataset = IconDataset(data_dir, transform=transform)
    
    # トレーニング/検証分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # モデル
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IconCNN(num_classes=14).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"デバイス: {device}")
    print(f"トレーニングサンプル: {train_size}")
    print(f"検証サンプル: {val_size}")
    print()
    
    # トレーニングループ
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # 検証
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')
        
        # ベストモデルを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'icon_classifier_best.pth')
            print(f'  ✓ ベストモデル保存 (Val Acc: {val_acc:.2f}%)')
        print()
    
    print(f'トレーニング完了！最高検証精度: {best_val_acc:.2f}%')
    
    # ONNXエクスポート
    model.load_state_dict(torch.load('icon_classifier_best.pth'))
    model.eval()
    
    dummy_input = torch.randn(1, 3, 48, 48).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        'icon_classifier.onnx',
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print('ONNXモデルをエクスポートしました: icon_classifier.onnx')

if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'training_data'
    train(data_dir)
```

#### 2. 学習の実行

```bash
# PyTorchと依存関係をインストール
pip install torch torchvision pillow

# 学習を実行
python train_pytorch.py training_data
```

#### 3. Rustで推論

ONNXモデルをRustで使用するには、`tract`クレートを使用します：

```toml
# Cargo.toml に追加
tract-onnx = "0.21"
```

```rust
// 推論コード例
use tract_onnx::prelude::*;

fn load_model(path: &str) -> TractResult<SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>> {
    let model = tract_onnx::onnx()
        .model_for_path(path)?
        .into_optimized()?
        .into_runnable()?;
    Ok(model)
}

fn predict(model: &SimplePlan<...>, image: &[f32]) -> TractResult<usize> {
    let input = tract_ndarray::Array4::from_shape_vec((1, 3, 48, 48), image.to_vec())?;
    let result = model.run(tvec!(input.into()))?;
    let output = result[0].to_array_view::<f32>()?;
    let predicted_class = output.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    Ok(predicted_class)
}
```

### 代替アプローチ2: Burnの安定版を待つ

Burn 0.14系でビルドが通る場合：

```bash
# ml機能を有効にしてビルド
cargo build --features ml --release
```

### 代替アプローチ3: 他のRust MLフレームワーク

- **tract**: ONNX推論に特化、最も安定
- **candle**: HuggingFaceのRust ML、PyTorchライク
- **tch-rs**: PyTorchのRustバインディング

## モデルアーキテクチャ

### 推奨CNN構造

```
Input (3x48x48)
    ↓
Conv2D(32, 3x3) + ReLU + MaxPool(2x2)  → 32x24x24
    ↓
Conv2D(64, 3x3) + ReLU + MaxPool(2x2)  → 64x12x12
    ↓
Conv2D(128, 3x3) + ReLU + MaxPool(2x2) → 128x6x6
    ↓
Flatten → 4608
    ↓
Dense(256) + ReLU + Dropout(0.5)
    ↓
Dense(14) → Output (14クラス)
```

### パラメータ数

- Conv層: 約100K
- FC層: 約1.2M
- 合計: 約1.3M パラメータ

## データオーグメンテーション

精度向上のため、以下の変換を適用：

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomRotation(5),           # ±5度回転
    transforms.ColorJitter(                 # 色調整
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.RandomAffine(                # アフィン変換
        degrees=0,
        translate=(0.05, 0.05)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

## 学習のヒント

### ハイパーパラメータ

```python
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
WEIGHT_DECAY = 1e-4
```

### 学習曲線の確認

TensorBoardを使用：

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/icon_classifier')

# トレーニングループ内
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/train', train_acc, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)
```

```bash
tensorboard --logdir=runs
```

### 過学習の防止

1. **Dropout**: 0.5を使用
2. **Early Stopping**: 検証精度が5エポック改善しなければ停止
3. **Weight Decay**: L2正則化（1e-4）
4. **Data Augmentation**: 上記の変換を使用

## デプロイ

### 1. ONNXモデルの配置

```
models/
└── icon_classifier.onnx
```

### 2. Rustでの推論実装

```rust
// src/ml_recognizer.rs
use tract_onnx::prelude::*;
use image::DynamicImage;

pub struct MLRecognizer {
    model: SimplePlan<...>,
}

impl MLRecognizer {
    pub fn new(model_path: &str) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }
    
    pub fn predict(&self, image: &DynamicImage) -> Result<usize> {
        // 前処理
        let rgb = image.to_rgb8();
        let mut input = vec![0.0f32; 3 * 48 * 48];
        
        for (i, pixel) in rgb.pixels().enumerate() {
            input[i] = (pixel[0] as f32 / 255.0 - 0.5) / 0.5;
            input[i + 48*48] = (pixel[1] as f32 / 255.0 - 0.5) / 0.5;
            input[i + 48*48*2] = (pixel[2] as f32 / 255.0 - 0.5) / 0.5;
        }
        
        // 推論
        let result = self.model.run(tvec!(input.into()))?;
        
        // 後処理
        let output = result[0].to_array_view::<f32>()?;
        let predicted = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        
        Ok(predicted)
    }
}
```

## 期待される精度

- **テンプレートマッチング**: 85-90%
- **CNN（軽量）**: 95-98%
- **CNN（最適化）**: 98-99%

## トラブルシューティング

### Q: データが不足している

A: 動画を追加で処理するか、データオーグメンテーションを強化してください。

### Q: 特定のカテゴリの精度が低い

A: そのカテゴリのサンプル数を増やすか、類似カテゴリとの混同行列を確認してください。

### Q: 過学習している

A: Dropoutを増やす、Weight Decayを増やす、Early Stoppingを使用してください。

### Q: GPUが使えない

A: PyTorchでCUDA版をインストールするか、CPUで学習してください（遅くなります）。

```bash
# CUDA版PyTorchのインストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 参考リンク

- [Burn Framework](https://github.com/tracel-ai/burn)
- [Burn Models](https://github.com/tracel-ai/models)
- [Tract ONNX](https://github.com/sonos/tract)
- [PyTorch公式チュートリアル](https://pytorch.org/tutorials/)

## まとめ

現時点では、**PyTorch + ONNX + tract** のアプローチが最も実用的です：

1. `collect_training_data`でデータ収集 ✓
2. PyTorchでモデル訓練
3. ONNXでエクスポート
4. tractでRust推論

Burnフレームワークが安定したら、純粋なRust実装に移行できます。