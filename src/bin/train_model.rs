//! Burn機械学習フレームワークを使用した入力アイコン分類モデルのトレーニング
//!
//! 48x48のゲーム入力アイコン画像を14クラスに分類するCNNモデルを学習します。
//! Burn 0.19.1 + AutodiffBackend (WGPU) を使用します。

#[cfg(feature = "ml")]
use input_analyzer::config::{AppConfig, DeviceType};

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
use input_analyzer::ml_model::{
    load_and_normalize_image, IconClassifier, ModelConfig, BUTTON_LABELS, IMAGE_SIZE,
};

#[cfg(feature = "ml")]
use input_analyzer::model_metadata::ModelMetadata;
#[cfg(feature = "ml")]
use input_analyzer::model_storage;

#[cfg(feature = "ml")]
use rand::seq::SliceRandom;
#[cfg(feature = "ml")]
use std::path::{Path, PathBuf};
#[cfg(feature = "ml")]
use anyhow::Context;

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
    /// トレーニングディレクトリからクラスラベルを自動生成
    fn detect_classes(data_dir: &Path) -> anyhow::Result<Vec<String>> {
        let entries: Vec<_> = std::fs::read_dir(data_dir)?
            .filter_map(Result::ok)
            .filter_map(|e| {
                let path = e.path();
                if path.is_dir() {
                    if let Some(name) = path.file_name() {
                        if let Some(name_str) = name.to_str() {
                            return Some(name_str.to_string());
                        }
                    }
                }
                None
            })
            .collect();
        
        // アルファベット順でソート（一貫性を保つため）
        let mut classes = entries;
        classes.sort();
        
        if classes.is_empty() {
            anyhow::bail!("トレーニングディレクトリにサブディレクトリが見つかりませんでした");
        }
        
        Ok(classes)
    }

    /// データセットを読み込み
    fn load(data_dir: &Path, class_names: &[String]) -> anyhow::Result<Self> {
        let mut items = Vec::new();

        println!("=== データセット読み込み中 ===");
        println!("検出されたクラス: {}", class_names.len());

        for (class_idx, class_name) in class_names.iter().enumerate() {
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

        (Self { items: train_items }, Self { items: val_items })
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
    fn batch(&self, items: Vec<IconItem>, _device: &B::Device) -> IconBatch<B> {
        let batch_size = items.len();

        // 各画像を個別に処理してテンソルを作成
        let mut batch_images = Vec::with_capacity(batch_size);
        let mut targets_vec = Vec::with_capacity(batch_size);

        for item in items {
            // 画像をロードして即座にTensorに変換
            match load_and_normalize_image(&item.path) {
                Ok(image_data) => {
                    // 即座にTensorに変換(CPUメモリからGPUメモリへ)
                    let image_tensor =
                        Tensor::<B, 1>::from_floats(image_data.as_slice(), &self.device)
                            .reshape([1, 3, IMAGE_SIZE, IMAGE_SIZE]);
                    batch_images.push(image_tensor);
                    targets_vec.push(item.label as i64);
                    // image_dataはここでドロップされる
                }
                Err(e) => {
                    eprintln!("警告: 画像読み込み失敗 {}: {}", item.path.display(), e);
                    // エラーの場合はゼロテンソルを作成
                    let zero_tensor =
                        Tensor::<B, 4>::zeros([1, 3, IMAGE_SIZE, IMAGE_SIZE], &self.device);
                    batch_images.push(zero_tensor);
                    targets_vec.push(item.label as i64);
                }
            }
        }

        // バッチテンソルを結合
        let images = Tensor::cat(batch_images, 0);
        let targets = Tensor::<B, 1, Int>::from_ints(targets_vec.as_slice(), &self.device);

        // batch_imagesとtargets_vecは既に消費されているため、dropは不要

        IconBatch { images, targets }
    }
}

#[cfg(feature = "ml")]
impl<B: AutodiffBackend> TrainStep<IconBatch<B>, ClassificationOutput<B>> for IconClassifier<B> {
    fn step(&self, batch: IconBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let images = batch.images;
        let targets = batch.targets;
        let item = self.forward_classification(images, targets);
        let grads = item.loss.backward();
        let output = TrainOutput::new(self, grads, item);
        output
    }
}

#[cfg(feature = "ml")]
impl<B: Backend> ValidStep<IconBatch<B>, ClassificationOutput<B>> for IconClassifier<B> {
    fn step(&self, batch: IconBatch<B>) -> ClassificationOutput<B> {
        let images = batch.images;
        let targets = batch.targets;
        self.forward_classification(images, targets)
    }
}

#[cfg(feature = "ml")]
#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 50)]
    pub num_epochs: usize,
    #[config(default = 8)]
    pub batch_size: usize,
    #[config(default = 1)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

#[cfg(feature = "ml")]
fn create_artifact_dir(artifact_dir: &str) {
    // 既存のアーティファクトを削除
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

#[cfg(feature = "ml")]
fn train<B: AutodiffBackend>(
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

/// 学習データディレクトリから画像サイズを検出
#[cfg(feature = "ml")]
fn detect_image_size_from_dataset(data_dir: &Path) -> anyhow::Result<(u32, u32)> {
    use image::GenericImageView;
    
    // 各クラスディレクトリから最初の画像を探す
    for entry in std::fs::read_dir(data_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_dir() {
            // サブディレクトリ内の最初の画像ファイルを探す
            for file_entry in std::fs::read_dir(&path)? {
                let file_entry = file_entry?;
                let file_path = file_entry.path();
                
                if let Some(ext) = file_path.extension() {
                    let ext_lower = ext.to_string_lossy().to_lowercase();
                    if ext_lower == "png" || ext_lower == "jpg" || ext_lower == "jpeg" {
                        // 画像を読み込んでサイズを取得
                        let img = image::open(&file_path)
                            .with_context(|| format!("画像の読み込みに失敗: {}", file_path.display()))?;
                        let (width, height) = img.dimensions();
                        return Ok((width, height));
                    }
                }
            }
        }
    }
    
    // 画像が見つからない場合はエラー
    anyhow::bail!("学習データディレクトリに画像ファイルが見つかりません: {}", data_dir.display())
}

#[cfg(feature = "ml")]
fn main() -> anyhow::Result<()> {
    // 設定ファイルを読み込み（存在しない場合はデフォルト設定）
    let mut config = AppConfig::load_or_default();

    // 設定情報を表示
    config.display();

    let args: Vec<String> = std::env::args().collect();
    let data_dir = if args.len() >= 2 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("training_data")
    };

    // コマンドライン引数で上書き可能
    let num_epochs = if args.len() >= 3 {
        args[2].parse().unwrap_or(config.training.num_epochs)
    } else {
        config.training.num_epochs
    };

    let batch_size = if args.len() >= 4 {
        args[3].parse().unwrap_or(config.training.batch_size)
    } else {
        config.training.batch_size
    };

    // デバイスタイプをコマンドライン引数で指定可能
    if args.len() >= 5 {
        match args[4].to_lowercase().as_str() {
            "cpu" => config.set_device_type(DeviceType::Cpu),
            "gpu" | "wgpu" => config.set_device_type(DeviceType::Wgpu),
            _ => println!(
                "警告: 不明なデバイスタイプ '{}' - 設定ファイルの値を使用します",
                args[4]
            ),
        }
    }

    println!("=================================================================================");
    println!("アイコン分類モデル学習 (Burn)");
    println!("=================================================================================");
    println!("\nデータディレクトリ: {}", data_dir.display());

    // クラスラベルを自動検出
    let class_names = IconDataset::detect_classes(&data_dir)?;
    let num_classes = class_names.len();
    
    println!("\n検出されたクラス:");
    for (i, class_name) in class_names.iter().enumerate() {
        println!("  {}: {}", i, class_name);
    }

    // デバイス設定（設定ファイルの値を使用）
    let device = match config.device_type {
        DeviceType::Wgpu => {
            let dev = burn_wgpu::WgpuDevice::default();
            println!("使用デバイス: WGPU (GPU) - {:?}", dev);
            dev
        }
        DeviceType::Cpu => {
            println!("警告: CPU (NdArray) モードは現在このバイナリではサポートされていません");
            println!("WGPU (GPU) を使用します");
            let dev = burn_wgpu::WgpuDevice::default();
            println!("使用デバイス: WGPU (GPU) - {:?}", dev);
            dev
        }
    };

    // データセット読み込み
    let dataset = IconDataset::load(&data_dir, &class_names)?;
    let (dataset_train, dataset_val) = dataset.split(config.training.train_ratio);

    println!("\n学習データ: {} 枚", dataset_train.len());
    println!("検証データ: {} 枚", dataset_val.len());

    // モデル構築
    println!("\n=== モデル構築 ===");
    println!("モデル構造:");
    println!("  Conv1: 3 -> 32 (48x48 -> 24x24)");
    println!("  Conv2: 32 -> 64 (24x24 -> 12x12)");
    println!("  Conv3: 64 -> 128 (12x12 -> 6x6)");
    println!("  FC: 128*6*6 -> 256 -> {}", num_classes);

    // 学習設定
    let training_config = TrainingConfig::new(ModelConfig::new(num_classes), AdamConfig::new())
        .with_num_epochs(num_epochs)
        .with_batch_size(batch_size)
        .with_learning_rate(config.training.learning_rate);

    println!("\n=== 学習設定 ===");
    println!("エポック数: {}", training_config.num_epochs);
    println!("バッチサイズ: {}", training_config.batch_size);
    println!("学習率: {}", training_config.learning_rate);

    // 学習実行
    println!("\n=== 学習開始 ===\n");
    train::<MyAutodiffBackend>(
        "models",
        training_config,
        device,
        dataset_train,
        dataset_val,
    );

    println!("\n✓ 学習完了!");

    // モデルとメタデータを保存
    println!("\n=== モデルを保存中 ===");
    
    let model_path = std::path::PathBuf::from("models/icon_classifier");
    
    // モデルバイナリを読み込み
    let model_binary = std::fs::read("models/model.mpk")
        .context("Failed to read compiled model file")?;

    // 学習データから画像サイズを取得
    let (image_width, image_height) = detect_image_size_from_dataset(&data_dir)?;
    println!("検出された学習データ画像サイズ: {}x{}", image_width, image_height);

    // メタデータを作成（設定と検出値を使用）
    let button_labels: Vec<String> = BUTTON_LABELS.iter().map(|s| s.to_string()).collect();
    let metadata = ModelMetadata::new(
        button_labels,
        image_width,
        image_height,
        config.button_tile.x,
        config.button_tile.y,
        config.button_tile.width,
        config.button_tile.height,
        config.button_tile.columns_per_row,
        IMAGE_SIZE as u32,  // model_input_size
        num_epochs as u32,
    );

    // Tar.gz形式で保存
    model_storage::save_model_with_metadata(&model_path, &metadata, &model_binary)
        .context("Failed to save model with metadata")?;

    let tar_gz_path = model_path.with_extension("tar.gz");
    println!("✓ モデルを保存しました: {}", tar_gz_path.display());
    println!("\nTar.gzファイル内容:");
    println!("  metadata.json - メタデータ（ボタン情報、タイル設定など）");
    println!("  model.bin     - モデルの重み");

    // メタデータを表示
    model_storage::print_metadata_info(&metadata);

    // 設定を更新して保存
    config.training.num_epochs = num_epochs;
    config.training.batch_size = batch_size;
    config.set_model_path(tar_gz_path.to_string_lossy().to_string());

    if let Err(e) = config.save_default() {
        eprintln!("警告: 設定ファイルの保存に失敗しました: {}", e);
    }

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
