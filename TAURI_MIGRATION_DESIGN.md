# Tauri移植設計書 - Input Analyzer

## 1. 概要

### 1.1 移植の目的
- クロスプラットフォーム対応（Windows, macOS, Linux）
- モダンなUIフレームワーク（React/Vue/Svelte）の活用
- パフォーマンスの最適化
- 配布の簡素化（単一実行ファイル）

### 1.2 技術スタック

#### フロントエンド
- **フレームワーク**: React 18 + TypeScript
- **UIライブラリ**: Material-UI (MUI) v5
- **状態管理**: Zustand
- **ビルドツール**: Vite

#### バックエンド（Rust）
- **フレームワーク**: Tauri 2.x
- **機械学習**: Burn 0.19.1 + WGPU
- **動画処理**: GStreamer 1.20+
- **並行処理**: Tokio + async/await

## 2. アーキテクチャ設計

### 2.1 全体構成

```
┌─────────────────────────────────────────────────────┐
│                  Tauri Window                        │
│  ┌───────────────────────────────────────────────┐  │
│  │         React Frontend (TypeScript)           │  │
│  │  ┌─────────────┐  ┌──────────────────────┐  │  │
│  │  │ UI Components│  │  State Management    │  │  │
│  │  │  (MUI)      │  │    (Zustand)         │  │  │
│  │  └──────┬──────┘  └──────────┬───────────┘  │  │
│  │         │                     │               │  │
│  │         └─────────┬───────────┘               │  │
│  │                   │ IPC (Invoke)              │  │
│  └───────────────────┼───────────────────────────┘  │
│                      │                               │
│  ┌───────────────────▼───────────────────────────┐  │
│  │         Rust Backend (Tauri Core)             │  │
│  │  ┌─────────────────────────────────────────┐ │  │
│  │  │  Command Handlers (async)               │ │  │
│  │  ├─────────────────────────────────────────┤ │  │
│  │  │  ┌─────────────┐  ┌──────────────────┐ │ │  │
│  │  │  │ ML Engine   │  │ Video Processor  │ │ │  │
│  │  │  │ (Burn+WGPU) │  │  (GStreamer)     │ │ │  │
│  │  │  └─────────────┘  └──────────────────┘ │ │  │
│  │  │  ┌─────────────┐  ┌──────────────────┐ │ │  │
│  │  │  │ Model Mgmt  │  │  Config Mgmt     │ │ │  │
│  │  │  └─────────────┘  └──────────────────┘ │ │  │
│  │  └─────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 2.2 モジュール構成

#### フロントエンド（src-ui/）

```
src-ui/
├── src/
│   ├── components/
│   │   ├── ModelManager.tsx        # モデル読み込み・管理UI
│   │   ├── VideoAnalyzer.tsx       # 動画解析UI
│   │   ├── TrainingDataExtractor.tsx  # 学習データ抽出UI
│   │   ├── CsvEditor.tsx           # CSV編集UI
│   │   ├── ProgressMonitor.tsx     # 進捗表示
│   │   └── SettingsPanel.tsx       # 設定パネル
│   ├── stores/
│   │   ├── appStore.ts             # アプリ全体の状態
│   │   ├── modelStore.ts           # モデル関連状態
│   │   └── analysisStore.ts        # 解析関連状態
│   ├── services/
│   │   └── tauri.ts                # Tauri API ラッパー
│   ├── types/
│   │   └── index.ts                # 型定義
│   └── App.tsx                     # メインアプリ
├── package.json
└── tsconfig.json
```

#### バックエンド（src-tauri/）

```
src-tauri/
├── src/
│   ├── main.rs                     # エントリポイント
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── model.rs                # モデル関連コマンド
│   │   ├── video.rs                # 動画処理コマンド
│   │   ├── training.rs             # 学習関連コマンド
│   │   └── config.rs               # 設定関連コマンド
│   ├── ml/
│   │   ├── mod.rs
│   │   ├── model.rs                # MLモデル定義
│   │   ├── inference.rs            # 推論エンジン
│   │   ├── training.rs             # 学習エンジン
│   │   └── batch_processor.rs      # バッチ処理最適化
│   ├── video/
│   │   ├── mod.rs
│   │   ├── extractor.rs            # フレーム抽出
│   │   └── stream_processor.rs     # ストリーム処理
│   ├── utils/
│   │   ├── mod.rs
│   │   ├── config.rs               # 設定管理
│   │   └── metadata.rs             # メタデータ管理
│   └── state.rs                    # アプリケーション状態
├── Cargo.toml
└── tauri.conf.json
```

## 3. データフロー設計

### 3.1 動画からの入力履歴抽出フロー

```
[UI: 動画選択]
     ↓
[Tauri Command: start_video_analysis]
     ↓
[Video Extractor: フレームストリーム開始]
     ↓ (コールバック: 1フレームずつ)
[Frame] → [Tile Extraction] → [Inference] → [State Update]
     ↓                            ↓              ↓
  [削除]                    [GPU/CPU]      [CSV Record]
     ↓
[Event: progress_update] → [UI: プログレスバー更新]
     ↓
[Event: analysis_complete] → [UI: 結果表示]
```

### 3.2 モデル学習フロー

```
[UI: 学習データパス選択]
     ↓
[Tauri Command: start_training]
     ↓
[Training Engine: データセット読み込み]
     ↓
[Train/Valid分割 (80/20)]
     ↓
[エポックループ]
  ├→ [Batch処理] → [GPU Forward] → [Loss計算] → [Backward]
  ├→ [Event: epoch_progress] → [UI: 進捗更新]
  ├→ [Validation] → [Accuracy計算]
  └→ [Event: validation_result] → [UI: 精度表示]
     ↓
[モデル保存（tar.gz + metadata）]
     ↓
[Event: training_complete] → [UI: 完了通知]
```

## 4. 順次推論の実装設計

### 4.1 現状の問題点

現在のegui実装では：
- フレーム抽出が完了してから推論を開始
- メモリ上に全フレームのPathBufを保持
- 大容量動画では数GB単位のメモリ消費

### 4.2 改善アーキテクチャ（Tauri版）

#### 4.2.1 ストリーム処理パイプライン

```rust
// src-tauri/src/video/stream_processor.rs

use tokio::sync::mpsc;
use futures::stream::StreamExt;

pub struct StreamProcessor {
    extractor: FrameExtractor,
    inference_engine: InferenceEngine,
}

impl StreamProcessor {
    pub async fn process_video_stream(
        &self,
        video_path: PathBuf,
        progress_tx: mpsc::Sender<ProgressEvent>,
    ) -> Result<Vec<InputRecord>> {
        let (frame_tx, mut frame_rx) = mpsc::channel(1); // バッファサイズ=1
        
        // フレーム抽出タスク（別スレッド）
        let extractor_handle = tokio::task::spawn_blocking({
            let extractor = self.extractor.clone();
            let video_path = video_path.clone();
            move || {
                extractor.extract_frames_streaming(
                    video_path,
                    |frame_path| {
                        // 各フレーム抽出後、即座にチャネル送信
                        frame_tx.blocking_send(frame_path).ok();
                    }
                )
            }
        });
        
        let mut records = Vec::new();
        let mut current_state = None;
        let mut duration = 0u32;
        let mut frame_count = 0;
        
        // 推論タスク（メインスレッド、GPUアクセス）
        while let Some(frame_path) = frame_rx.recv().await {
            frame_count += 1;
            
            // 1. タイル切り出し
            let tiles = extract_tiles_from_frame(&frame_path).await?;
            
            // 2. GPU推論（バッチ処理）
            let state = self.inference_engine.infer_batch(&tiles).await?;
            
            // 3. 状態記録
            if let Some(prev) = current_state {
                if state == prev {
                    duration += 1;
                } else {
                    records.push(state_to_record(&prev, duration));
                    current_state = Some(state);
                    duration = 1;
                }
            } else {
                current_state = Some(state);
                duration = 1;
            }
            
            // 4. フレーム削除（メモリ解放）
            tokio::fs::remove_file(frame_path).await.ok();
            
            // 5. 進捗通知
            progress_tx.send(ProgressEvent::FrameProcessed { 
                count: frame_count 
            }).await.ok();
        }
        
        extractor_handle.await??;
        
        // 最後の状態を記録
        if let Some(state) = current_state {
            records.push(state_to_record(&state, duration));
        }
        
        Ok(records)
    }
}
```

#### 4.2.2 メモリ管理の最適化

```rust
// src-tauri/src/ml/batch_processor.rs

pub struct BatchProcessor {
    model: IconClassifier,
    device: WgpuDevice,
    batch_size: usize,
    // GPUバッファを事前確保（再利用）
    input_buffer: Tensor<B, 4>,
}

impl BatchProcessor {
    pub fn new(model: IconClassifier, device: WgpuDevice) -> Self {
        let batch_size = 6; // 最大6タイル/フレーム
        
        // [batch_size, 3, 48, 48] のバッファを事前確保
        let input_buffer = Tensor::zeros([batch_size, 3, 48, 48], &device);
        
        Self {
            model,
            device,
            batch_size,
            input_buffer,
        }
    }
    
    pub async fn infer_tiles(&mut self, tiles: &[RgbImage]) -> Result<Vec<Classification>> {
        let num_tiles = tiles.len();
        assert!(num_tiles <= self.batch_size);
        
        // 画像データをバッファにコピー（GPUメモリ再利用）
        for (i, tile) in tiles.iter().enumerate() {
            let tensor = image_to_tensor(tile, &self.device);
            self.input_buffer = self.input_buffer.slice_assign(
                [i..i+1, 0..3, 0..48, 0..48],
                tensor
            );
        }
        
        // 使用する部分だけスライス
        let batch = self.input_buffer.clone().slice([0..num_tiles]);
        
        // フォワードパス（GPU）
        let output = self.model.forward(batch);
        
        // 結果を取得（GPU→CPU転送は最小限）
        let predictions = output.argmax(1).into_data().await.value;
        let confidences = output.max_dim(1).into_data().await.value;
        
        Ok((0..num_tiles)
            .map(|i| Classification {
                class_id: predictions[i] as usize,
                confidence: confidences[i],
            })
            .collect())
    }
}
```

### 4.3 メモリ使用量の比較

| 実装方式 | 動画1分間 | 動画10分間 | 備考 |
|---------|----------|-----------|------|
| **現状（egui）** | 約500MB | 約5GB | 全フレームのPathBuf保持 |
| **Tauri改善版** | 約200MB | 約200MB | ストリーム処理、即削除 |

### 4.4 スループット向上

```rust
// パイプライン並列化（オプション）

pub struct PipelinedProcessor {
    // 抽出 → 推論の2段パイプライン
    extraction_queue: mpsc::Receiver<PathBuf>,
    inference_queue: mpsc::Sender<InferenceTask>,
}

// 抽出と推論を並行実行
// - スレッド1: フレームN+1を抽出
// - スレッド2: フレームNを推論
// → レイテンシ半減
```

## 5. 学習時のバッチ処理とメモリ管理

### 5.1 データローダーの実装

```rust
// src-tauri/src/ml/training.rs

use burn::data::{dataloader::DataLoaderBuilder, dataset::Dataset};
use std::sync::Arc;

pub struct ImageDataset {
    samples: Vec<(PathBuf, usize)>, // (画像パス, クラスID)
    cache_size: usize,
    // LRUキャッシュで頻繁にアクセスされる画像をメモリ保持
    cache: Arc<Mutex<lru::LruCache<PathBuf, Tensor<B, 3>>>>,
}

impl ImageDataset {
    pub fn new(data_dir: PathBuf, cache_size: usize) -> Result<Self> {
        let mut samples = Vec::new();
        
        // 各クラスディレクトリをスキャン
        for (class_id, class_name) in CLASSES.iter().enumerate() {
            let class_dir = data_dir.join(class_name);
            for entry in std::fs::read_dir(class_dir)? {
                let path = entry?.path();
                if path.extension().map_or(false, |e| e == "png") {
                    samples.push((path, class_id));
                }
            }
        }
        
        // シャッフル
        samples.shuffle(&mut rand::thread_rng());
        
        Ok(Self {
            samples,
            cache_size,
            cache: Arc::new(Mutex::new(lru::LruCache::new(cache_size))),
        })
    }
    
    fn load_image(&self, path: &Path) -> Result<Tensor<B, 3>> {
        // キャッシュチェック
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(tensor) = cache.get(path) {
                return Ok(tensor.clone());
            }
        }
        
        // ディスクから読み込み
        let img = image::open(path)?.to_rgb8();
        let tensor = image_to_tensor(&img, &self.device);
        
        // キャッシュに追加
        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(path.to_path_buf(), tensor.clone());
        }
        
        Ok(tensor)
    }
}

impl Dataset for ImageDataset {
    type Item = (Tensor<B, 3>, usize);
    
    fn get(&self, index: usize) -> Option<Self::Item> {
        let (path, class_id) = &self.samples[index];
        let tensor = self.load_image(path).ok()?;
        Some((tensor, *class_id))
    }
    
    fn len(&self) -> usize {
        self.samples.len()
    }
}
```

### 5.2 学習ループの実装

```rust
pub async fn train_model(
    config: TrainingConfig,
    progress_tx: mpsc::Sender<TrainingEvent>,
) -> Result<TrainedModel> {
    let device = WgpuDevice::default();
    
    // データセット準備
    let dataset = ImageDataset::new(&config.data_dir, 1000)?; // 1000画像キャッシュ
    let (train_set, valid_set) = dataset.split(0.8);
    
    // データローダー
    let train_loader = DataLoaderBuilder::new()
        .batch_size(config.batch_size)
        .shuffle(true)
        .num_workers(4) // 並列読み込み
        .build(train_set);
    
    let valid_loader = DataLoaderBuilder::new()
        .batch_size(config.batch_size)
        .build(valid_set);
    
    // モデル初期化
    let mut model = IconClassifier::new(&device);
    let mut optimizer = AdamConfig::new().init();
    
    for epoch in 0..config.num_epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        
        // トレーニング
        for (batch_idx, (images, labels)) in train_loader.iter().enumerate() {
            // GPU転送
            let images = images.to_device(&device);
            let labels = labels.to_device(&device);
            
            // フォワード
            let output = model.forward(images);
            let loss = CrossEntropyLoss::new().forward(output.clone(), labels.clone());
            
            // バックワード
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(config.learning_rate, model, grads);
            
            total_loss += loss.into_scalar();
            batch_count += 1;
            
            // 進捗通知（10バッチごと）
            if batch_idx % 10 == 0 {
                progress_tx.send(TrainingEvent::BatchComplete {
                    epoch,
                    batch: batch_idx,
                    loss: total_loss / batch_count as f32,
                }).await.ok();
            }
        }
        
        // バリデーション
        let valid_acc = evaluate(&model, &valid_loader, &device).await?;
        
        progress_tx.send(TrainingEvent::EpochComplete {
            epoch,
            train_loss: total_loss / batch_count as f32,
            valid_accuracy: valid_acc,
        }).await.ok();
        
        // 早期終了判定
        if valid_acc > 0.99 {
            break;
        }
    }
    
    Ok(TrainedModel { model, device })
}
```

### 5.3 メモリ使用量の目安

```
学習時のメモリ内訳（バッチサイズ32）:

GPU VRAM:
- モデルパラメータ: 約50MB
- オプティマイザ状態: 約50MB
- バッチデータ (32×3×48×48): 約1.1MB
- 中間アクティベーション: 約10MB
- 勾配テンソル: 約50MB
→ 合計: 約160MB

RAM:
- データローダーキャッシュ (1000画像): 約200MB
- 並列ワーカーバッファ: 約50MB
- その他: 約50MB
→ 合計: 約300MB
```

## 6. Tauri Command設計

### 6.1 モデル管理コマンド

```rust
// src-tauri/src/commands/model.rs

#[tauri::command]
pub async fn load_model(
    path: String,
    state: State<'_, AppState>,
) -> Result<ModelInfo, String> {
    let model = ModelStorage::load_targz(&path)
        .map_err(|e| e.to_string())?;
    
    state.set_model(model.clone()).await;
    
    Ok(ModelInfo {
        num_classes: model.metadata.num_classes,
        button_labels: model.metadata.button_labels,
        image_width: model.metadata.image_width,
        image_height: model.metadata.image_height,
        trained_at: model.metadata.trained_at,
    })
}

#[tauri::command]
pub async fn start_training(
    data_dir: String,
    config: TrainingConfig,
    window: Window,
) -> Result<(), String> {
    let (tx, mut rx) = mpsc::channel(100);
    
    // バックグラウンドで学習実行
    tokio::spawn(async move {
        let result = train_model(config, tx).await;
        // 完了イベント送信
        window.emit("training_complete", result).ok();
    });
    
    // 進捗イベント転送
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            window.emit("training_progress", event).ok();
        }
    });
    
    Ok(())
}
```

### 6.2 動画処理コマンド

```rust
// src-tauri/src/commands/video.rs

#[tauri::command]
pub async fn analyze_video(
    video_path: String,
    window: Window,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let model = state.get_model().await
        .ok_or("モデルが読み込まれていません")?;
    
    let (tx, mut rx) = mpsc::channel(100);
    
    let processor = StreamProcessor::new(model);
    
    // バックグラウンドで解析実行
    tokio::spawn(async move {
        let result = processor.process_video_stream(
            PathBuf::from(video_path),
            tx,
        ).await;
        
        window.emit("analysis_complete", result).ok();
    });
    
    // 進捗イベント転送
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            window.emit("analysis_progress", event).ok();
        }
    });
    
    Ok(())
}

#[tauri::command]
pub async fn cancel_analysis(
    state: State<'_, AppState>,
) -> Result<(), String> {
    state.cancel_current_task().await;
    Ok(())
}
```

## 7. フロントエンド実装例

### 7.1 動画解析コンポーネント

```typescript
// src-ui/src/components/VideoAnalyzer.tsx

import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';
import { useState, useEffect } from 'react';

export function VideoAnalyzer() {
  const [progress, setProgress] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    // 進捗イベントリスナー
    const unlisten = listen('analysis_progress', (event) => {
      const { count, total } = event.payload;
      setProgress(count);
      setTotalFrames(total);
    });

    return () => {
      unlisten.then(f => f());
    };
  }, []);

  const handleAnalyze = async (videoPath: string) => {
    setIsAnalyzing(true);
    
    try {
      await invoke('analyze_video', { videoPath });
    } catch (error) {
      console.error('解析エラー:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleCancel = async () => {
    await invoke('cancel_analysis');
    setIsAnalyzing(false);
  };

  return (
    <Box>
      <LinearProgress 
        variant="determinate" 
        value={(progress / totalFrames) * 100} 
      />
      <Typography>
        {progress} / {totalFrames} フレーム処理完了
      </Typography>
      <Button onClick={handleCancel} disabled={!isAnalyzing}>
        キャンセル
      </Button>
    </Box>
  );
}
```

## 7. モデル評価・改善サイクル設計

### 7.1 概要

学習したモデルの精度を反復的に向上させるためのワークフローを実装します。

```
学習 → 自動分類 → 確認・修正 → 統合 → 再学習 → ...
```

### 7.2 タイル自動分類コマンド

```rust
// src-tauri/src/commands/classification.rs

#[tauri::command]
pub async fn classify_video_tiles(
    model_path: String,
    video_path: String,
    output_dir: String,
    app_handle: tauri::AppHandle,
) -> Result<ClassificationResult, String> {
    let model = ModelStorage::load_targz(&model_path)
        .map_err(|e| e.to_string())?;
    
    let inference_engine = InferenceEngine::new(model)?;
    let extractor = FrameExtractor::new()?;
    
    // 出力ディレクトリ構造を作成
    create_class_directories(&output_dir, &inference_engine.class_labels)?;
    
    let (frame_tx, frame_rx) = tokio::sync::mpsc::channel(32);
    let cancel_flag = Arc::new(AtomicBool::new(false));
    
    // フレーム抽出スレッド
    let extractor_handle = tokio::task::spawn_blocking({
        let video_path = video_path.clone();
        move || {
            extractor.extract_frames_with_callback(
                &video_path,
                |frame_path| {
                    frame_tx.blocking_send(frame_path).ok();
                    Ok(())
                }
            )
        }
    });
    
    let mut classified_counts = HashMap::new();
    let mut total_tiles = 0;
    
    // 分類タスク
    while let Some(frame_path) = frame_rx.recv().await {
        // タイル切り出し
        let tiles = extract_tiles_from_frame(&frame_path).await?;
        
        for (tile_idx, tile_image) in tiles.iter().enumerate() {
            // 推論
            let class_id = inference_engine.infer_single(tile_image).await?;
            let class_name = &inference_engine.class_labels[class_id];
            
            // 分類先ディレクトリに保存
            let filename = format!(
                "{}_{}_tile{}.png",
                frame_path.file_stem().unwrap().to_str().unwrap(),
                frame_number,
                tile_idx + 1
            );
            let output_path = PathBuf::from(&output_dir)
                .join(class_name)
                .join(filename);
            
            tile_image.save(&output_path)
                .map_err(|e| e.to_string())?;
            
            *classified_counts.entry(class_name.clone()).or_insert(0) += 1;
            total_tiles += 1;
        }
        
        // 進捗通知
        app_handle.emit_all("classification_progress", 
            json!({ "total": total_tiles })
        ).ok();
        
        // フレーム削除
        tokio::fs::remove_file(frame_path).await.ok();
        
        if cancel_flag.load(Ordering::Relaxed) {
            break;
        }
    }
    
    extractor_handle.await.map_err(|e| e.to_string())??;
    
    Ok(ClassificationResult {
        total_tiles,
        class_counts: classified_counts,
        output_dir,
    })
}

fn create_class_directories(
    base_dir: &str,
    class_labels: &[String],
) -> Result<(), String> {
    for label in class_labels {
        let dir = PathBuf::from(base_dir).join(label);
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("ディレクトリ作成エラー: {}", e))?;
    }
    Ok(())
}
```

### 7.3 分類結果確認UI（React）

```typescript
// src-ui/src/components/ClassificationReviewer.tsx

interface ClassificationReviewerProps {
  outputDir: string;
  classLabels: string[];
}

export const ClassificationReviewer: React.FC<ClassificationReviewerProps> = ({
  outputDir,
  classLabels,
}) => {
  const [selectedClass, setSelectedClass] = useState<string>(classLabels[0]);
  const [images, setImages] = useState<string[]>([]);
  const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set());
  
  useEffect(() => {
    loadClassImages(selectedClass);
  }, [selectedClass]);
  
  const loadClassImages = async (className: string) => {
    const files = await invoke<string[]>('list_directory_files', {
      path: `${outputDir}/${className}`,
    });
    setImages(files);
  };
  
  const moveImagesToClass = async (targetClass: string) => {
    for (const imagePath of selectedImages) {
      await invoke('move_file', {
        from: imagePath,
        to: `${outputDir}/${targetClass}/${path.basename(imagePath)}`,
      });
    }
    
    // リロード
    setSelectedImages(new Set());
    loadClassImages(selectedClass);
  };
  
  return (
    <Box>
      <Typography variant="h6">分類結果の確認と修正</Typography>
      
      {/* クラス選択タブ */}
      <Tabs value={selectedClass} onChange={(e, v) => setSelectedClass(v)}>
        {classLabels.map(label => (
          <Tab key={label} label={label} value={label} />
        ))}
      </Tabs>
      
      {/* 画像グリッド */}
      <Grid container spacing={2}>
        {images.map(imagePath => (
          <Grid item xs={2} key={imagePath}>
            <Card
              onClick={() => {
                const newSet = new Set(selectedImages);
                if (newSet.has(imagePath)) {
                  newSet.delete(imagePath);
                } else {
                  newSet.add(imagePath);
                }
                setSelectedImages(newSet);
              }}
              sx={{
                border: selectedImages.has(imagePath) 
                  ? '2px solid blue' 
                  : 'none',
              }}
            >
              <CardMedia
                component="img"
                image={`asset://localhost/${imagePath}`}
                alt="tile"
              />
            </Card>
          </Grid>
        ))}
      </Grid>
      
      {/* 移動先選択 */}
      {selectedImages.size > 0 && (
        <Box mt={2}>
          <Typography>
            {selectedImages.size}枚選択中 - 移動先を選択:
          </Typography>
          <Box>
            {classLabels.map(label => (
              <Button
                key={label}
                onClick={() => moveImagesToClass(label)}
                disabled={label === selectedClass}
              >
                {label}へ移動
              </Button>
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
};
```

### 7.4 学習データ統合コマンド

```rust
// src-tauri/src/commands/data_integration.rs

#[tauri::command]
pub async fn integrate_classified_data(
    source_dir: String,
    training_data_dir: String,
) -> Result<IntegrationResult, String> {
    let mut integrated_counts = HashMap::new();
    
    // 各クラスディレクトリを走査
    for entry in std::fs::read_dir(&source_dir)
        .map_err(|e| e.to_string())? 
    {
        let entry = entry.map_err(|e| e.to_string())?;
        let class_name = entry.file_name().to_string_lossy().to_string();
        let source_class_dir = entry.path();
        let target_class_dir = PathBuf::from(&training_data_dir).join(&class_name);
        
        std::fs::create_dir_all(&target_class_dir)
            .map_err(|e| e.to_string())?;
        
        let mut count = 0;
        for img_entry in std::fs::read_dir(&source_class_dir)
            .map_err(|e| e.to_string())? 
        {
            let img_path = img_entry.map_err(|e| e.to_string())?.path();
            if img_path.extension().map_or(false, |e| e == "png") {
                let target_path = target_class_dir.join(img_path.file_name().unwrap());
                std::fs::copy(&img_path, &target_path)
                    .map_err(|e| e.to_string())?;
                count += 1;
            }
        }
        
        integrated_counts.insert(class_name, count);
    }
    
    Ok(IntegrationResult {
        integrated_counts,
        training_data_dir,
    })
}
```

### 7.5 評価・改善サイクルのワークフロー

1. **初回学習**: 手動で分類した少量のデータでモデル学習
2. **自動分類**: 検証用動画でタイルを自動分類
3. **確認UI**: 分類結果をグリッド表示、誤分類を視覚的に確認
4. **手動修正**: 誤っているタイルを正しいクラスに移動
5. **データ統合**: 修正したデータを学習データセットに統合
6. **再学習**: より大きく正確なデータセットで再学習
7. **精度確認**: 2に戻り、満足いく精度まで繰り返す

### 7.6 メモリ最適化

分類時のメモリ使用を最小化：
- フレームを即座に削除
- タイルをバッチ化せず1枚ずつ推論
- 画像保存後は即座にメモリ解放
- 大量のタイル画像でもRAM使用量を抑制

## 8. ビルド・配布

### 8.1 ビルド設定

```toml
# src-tauri/tauri.conf.json

{
  "build": {
    "beforeBuildCommand": "npm run build",
    "beforeDevCommand": "npm run dev",
    "devPath": "http://localhost:5173",
    "distDir": "../dist"
  },
  "bundle": {
    "identifier": "com.example.input-analyzer",
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/icon.icns",
      "icons/icon.ico"
    ],
    "resources": [
      "models/*",
      "config.json"
    ],
    "externalBin": []
  }
}
```

### 8.2 配布形式

- **Windows**: NSIS installer (.exe)
- **macOS**: DMG (.dmg)
- **Linux**: AppImage, deb, rpm

## 9. 移行計画

### フェーズ1: 基盤構築（2週間）
- Tauriプロジェクト初期化
- React UIスケルトン作成
- 基本的なIPC通信実装

### フェーズ2: コア機能移植（3週間）
- モデル読み込み機能
- ストリーム処理パイプライン実装
- 順次推論の実装

### フェーズ3: 学習機能（2週間）
- データローダー実装
- 学習ループ実装
- 進捗表示

### フェーズ4: モデル評価・改善機能（1.5週間）
- タイル自動分類機能実装
- 分類結果の確認UI
- 誤分類修正ワークフロー
- 学習データ統合機能

### フェーズ5: UI/UX改善（1週間）
- マテリアルデザイン適用
- エラーハンドリング強化
- 設定画面

### フェーズ6: テスト・最適化（1週間）
- パフォーマンス測定
- メモリプロファイリング
- バグ修正

合計: 約10.5週間

## 10. まとめ

Tauri移植により以下の改善が期待できます：

- **メモリ効率**: 既存のストリーミング処理を維持し、約200MB（動画の長さに関わらず）
- **クロスプラットフォーム**: Windows/macOS/Linux対応
- **モダンUI**: React + MUIによる洗練されたUI
- **配布簡素化**: 単一実行ファイル
- **パフォーマンス**: 既存のストリーム処理を活用した高速化
- **モデル形式**: tar.gz形式に統一（メタデータ付き）

既存の順次推論とメモリ管理の実装を活用し、大容量動画でも安定した処理を実現します。
