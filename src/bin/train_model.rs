//! Burn機械学習フレームワークを使用した入力アイコン分類モデルのトレーニング
//!
//! 48x48のゲーム入力アイコン画像を14クラスに分類するCNNモデルを学習します。
//! Burn 0.19.1 + AutodiffBackend (WGPU) を使用します。

#[cfg(feature = "ml")]
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    module::Module,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::{backend::AutodiffBackend, Int, Tensor},
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, LearningStrategy, TrainOutput, TrainStep, ValidStep,
    },
};

#[cfg(feature = "ml")]
use input_analyzer::ml_model::{IconClassifier, ModelConfig, CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES, load_and_normalize_image};

#[cfg(feature = "ml")]
use rand::seq::SliceRandom;
#[cfg(feature = "ml")]
use std::path::{Path, PathBuf};

// WGPUバックエンド（GPU使用）
#[cfg(feature = "ml")]
type MyBackend = burn_wgpu::Wgpu;
#[cfg(feature = "ml")]
type MyAutodiffBackend = burn_autodiff::Autodiff<MyBackend>;

// CPUバックエンド（メモリ効率的・安定）- 必要に応じてコメントを切り替え
// #[cfg(feature = "ml")]
// type MyBackend = burn_ndarray::NdArray<f32>;
// #[cfg(feature = "ml")]
// type MyAutodiffBackend = burn_autodiff::Autodiff<MyBackend>;



/// データセットアイテム（画像パスのみ保持）
#[cfg(feature = "ml")]
#[derive(Clone, Debug)]
struct IconItem {
    path: PathBuf,
    label: usize,
}

/// アイコン画像データセット
#[cfg(feature = "ml")]
#[derive(Clone)]
struct IconDataset {
    items: Vec<IconItem>,
}

#[cfg(feature = "ml")]
impl IconDataset {
    /// データセットを読み込み
    fn load(data_dir: &Path) -> anyhow::Result<Self> {
        let mut items = Vec::new();

        println!("=== データセット読み込み中 ===");

        for (class_idx, class_name) in CLASS_NAMES.iter().enumerate() {
            let class_dir = data_dir.join(class_name);
            if !class_dir.exists() {
                println!("警告: {} ディレクトリが存在しません", class_name);
                continue;
            }

            let mut class_items = 0;
            for entry in std::fs::read_dir(&class_dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.extension().and_then(|s| s.to_str()) == Some("png") {
                    items.push(IconItem {
                        path: path.clone(),
                        label: class_idx,
                    });
                    class_items += 1;
                }
            }

            println!("  {:12}: {:4} 枚", class_name, class_items);
        }

        println!("\n総サンプル数: {}", items.len());

        if items.is_empty() {
            anyhow::bail!("データが見つかりませんでした");
        }

        Ok(Self { items })
    }

    /// データセットを分割
    fn split(mut self, train_ratio: f32) -> (Self, Self) {
        let mut rng = rand::thread_rng();
        self.items.shuffle(&mut rng);

        let train_size = (self.items.len() as f32 * train_ratio) as usize;
        let mut train_items = self.items;
        let val_items = train_items.split_off(train_size);

        (
            Self { items: train_items },
            Self { items: val_items },
        )
    }
}

#[cfg(feature = "ml")]
impl Dataset<IconItem> for IconDataset {
    fn get(&self, index: usize) -> Option<IconItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}



/// バッチデータ
#[cfg(feature = "ml")]
#[derive(Clone, Debug)]
struct IconBatch<B: Backend> {
    pub images: Tensor<B, 4>, // [Batch, Channel, Height, Width]
    pub targets: Tensor<B, 1, Int>,
}

/// バッチャー
#[cfg(feature = "ml")]
#[derive(Clone)]
struct IconBatcher<B: Backend> {
    device: B::Device,
}

#[cfg(feature = "ml")]
impl<B: Backend> IconBatcher<B> {
    fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[cfg(feature = "ml")]
impl<B: Backend> burn::data::dataloader::batcher::Batcher<B, IconItem, IconBatch<B>>
    for IconBatcher<B>
{
    fn batch(&self, items: Vec<IconItem>, device: &B::Device) -> IconBatch<B> {
        let batch_size = items.len();

        // バッチ生成時に画像を読み込む（メモリ効率的）
        let mut images_vec = Vec::with_capacity(batch_size * 3 * IMAGE_SIZE * IMAGE_SIZE);
        let mut targets_vec = Vec::with_capacity(batch_size);

        for item in items {
            // 画像をロード
            match load_and_normalize_image(&item.path) {
                Ok(image_data) => {
                    images_vec.extend(image_data);
                    targets_vec.push(item.label as i64);
                }
                Err(e) => {
                    eprintln!("警告: 画像読み込み失敗 {}: {}", item.path.display(), e);
                    // エラーの場合はゼロで埋める（またはスキップ）
                    images_vec.extend(vec![0.0; 3 * IMAGE_SIZE * IMAGE_SIZE]);
                    targets_vec.push(item.label as i64);
                }
            }
        }

        // Tensorを作成してreshape [Batch, Channel, Height, Width]
        let images = Tensor::<B, 1>::from_floats(images_vec.as_slice(), device)
            .reshape([batch_size, 3, IMAGE_SIZE, IMAGE_SIZE]);

        let targets = Tensor::<B, 1, Int>::from_ints(targets_vec.as_slice(), device);

        IconBatch { images, targets }
        // ← ここでimages_vecとtargets_vecは自動的にドロップされる
    }
}

#[cfg(feature = "ml")]
impl<B: AutodiffBackend> TrainStep<IconBatch<B>, ClassificationOutput<B>> for IconClassifier<B> {
    fn step(&self, batch: IconBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

#[cfg(feature = "ml")]
impl<B: Backend> ValidStep<IconBatch<B>, ClassificationOutput<B>> for IconClassifier<B> {
    fn step(&self, batch: IconBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[cfg(feature = "ml")]
#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 50)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 2)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-3)]
    pub learning_rate: f64,
}

#[cfg(feature = "ml")]
fn create_artifact_dir(artifact_dir: &str) {
    // 既存のアーティファクトを削除
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

#[cfg(feature = "ml")]
pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    dataset_train: IconDataset,
    dataset_val: IconDataset,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    // CPUバックエンドの場合はseedメソッドが異なる
    // B::seed(&device, config.seed);

    // ランダムシード設定（クロスプラットフォーム）
    use rand::SeedableRng;
    let _ = rand::rngs::StdRng::seed_from_u64(config.seed);

    let batcher_train = IconBatcher::<B>::new(device.clone());
    let batcher_val = IconBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train);

    let dataloader_val = DataLoaderBuilder::new(batcher_val)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_val);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .learning_strategy(LearningStrategy::SingleDevice(device.clone()))
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_val);

    model_trained
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

#[cfg(feature = "ml")]
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("================================================================================");
        eprintln!("アイコン分類モデル学習 (Burn 0.19.1)");
        eprintln!("================================================================================");
        eprintln!();
        eprintln!("使用方法:");
        eprintln!("  train_model <トレーニングデータディレクトリ> [エポック数] [バッチサイズ]");
        eprintln!();
        eprintln!("引数:");
        eprintln!("  <トレーニングデータディレクトリ> : training_data ディレクトリ (必須)");
        eprintln!("  [エポック数]                    : 学習エポック数 (デフォルト: 50)");
        eprintln!("  [バッチサイズ]                  : バッチサイズ (デフォルト: 32)");
        eprintln!();
        eprintln!("例:");
        eprintln!("  cargo run --bin train_model --features ml --release -- training_data");
        eprintln!("  cargo run --bin train_model --features ml --release -- training_data 30 16");
        eprintln!();
        eprintln!("注意: 現在はCPUバックエンド(NdArray)を使用します");
        eprintln!("      GPUを使用する場合はソースコードでWGPUバックエンドに切り替えてください");
        eprintln!();
        std::process::exit(1);
    }

    let data_dir = PathBuf::from(&args[1]);
    let num_epochs = if args.len() >= 3 {
        args[2].parse().unwrap_or(50)
    } else {
        50
    };
    let batch_size = if args.len() >= 4 {
        args[3].parse().unwrap_or(32)
    } else {
        32
    };

    println!("================================================================================");
    println!("アイコン分類モデル学習 (Burn + WGPU/GPU)");
    println!("================================================================================");

    // デバイス設定（WGPU/GPU）
    let device = burn_wgpu::WgpuDevice::default();
    println!("\n使用デバイス: {:?}", device);

    // CPUの場合はこちらを使用:
    // let device = burn_ndarray::NdArrayDevice::Cpu;
    // println!("\n使用デバイス: CPU (NdArray)");

    // データセット読み込み
    let dataset = IconDataset::load(&data_dir)?;
    let (dataset_train, dataset_val) = dataset.split(0.8);

    println!("\n学習データ: {} 枚", dataset_train.len());
    println!("検証データ: {} 枚", dataset_val.len());

    // モデル構築
    println!("\n=== モデル構築 ===");
    println!("モデル構造:");
    println!("  Conv1: 3 -> 32 (48x48 -> 24x24)");
    println!("  Conv2: 32 -> 64 (24x24 -> 12x12)");
    println!("  Conv3: 64 -> 128 (12x12 -> 6x6)");
    println!("  FC: 128*6*6 -> 256 -> {}", NUM_CLASSES);

    // 学習設定
    let config = TrainingConfig::new(
        ModelConfig::new(NUM_CLASSES),
        AdamConfig::new(),
    )
    .with_num_epochs(num_epochs)
    .with_batch_size(batch_size)
    .with_learning_rate(1.0e-3);

    println!("\n=== 学習設定 ===");
    println!("エポック数: {}", config.num_epochs);
    println!("バッチサイズ: {}", config.batch_size);
    println!("学習率: {}", config.learning_rate);

    // 学習実行
    println!("\n=== 学習開始 ===\n");
    train::<MyAutodiffBackend>("models", config, device, dataset_train, dataset_val);

    println!("\n✓ 学習完了!");
    println!("\nモデル保存先: models/model");
    println!();
    println!("次のステップ:");
    println!("  1. モデルを使用してinput_cells_allから新しいデータを収集");
    println!("  2. training_dataを更新");
    println!("  3. 再度学習（反復的にデータセット品質を向上）");

    Ok(())
}

#[cfg(not(feature = "ml"))]
fn main() {
    eprintln!("エラー: このプログラムはml機能を有効にしてビルドする必要があります。");
    eprintln!();
    eprintln!("ビルドコマンド:");
    eprintln!("  cargo build --bin train_model --features ml --release");
    eprintln!();
    std::process::exit(1);
}
