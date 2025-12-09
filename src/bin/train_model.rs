//! Burn機械学習フレームワークを使用した入力アイコン分類モデルのトレーニング
//!
//! 48x48のゲーム入力アイコン画像を分類するCNNモデルを学習します。
//! Burn 0.19.1 + AutodiffBackend (WGPU) を使用します。
//!
//! ## 使用方法
//! ```bash
//! cargo run --bin train_model --features ml --release -- \
//!   --data-dir training_data \
//!   --output models/my_model \
//!   --buttons "A1,A2,B,W,Start" \
//!   --epochs 50 \
//!   --batch-size 8
//! ```

#[cfg(feature = "ml")]
use clap::Parser;

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
    fn detect_classes(data_dir: &Path, button_labels: &[String]) -> anyhow::Result<Vec<String>> {
        let mut entries: Vec<String> = std::fs::read_dir(data_dir)?
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
        
        // 方向キーを分離
        let mut dir_classes: Vec<String> = entries.iter()
            .filter(|name| name.starts_with("dir_"))
            .cloned()
            .collect();
        dir_classes.sort(); // dir_1, dir_2, ..., dir_9
        
        // ボタンクラス（button_labelsの順序）
        let button_classes: Vec<String> = button_labels.iter()
            .filter(|label| entries.contains(label))
            .cloned()
            .collect();
        
        // 順序: dir_1~9, ボタン順
        let mut classes = dir_classes;
        classes.extend(button_classes);
        
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

        // 全画像データをCPUメモリで一度にまとめてから、GPUへ1回で転送
        let mut all_pixels = Vec::with_capacity(batch_size * 3 * IMAGE_SIZE * IMAGE_SIZE);
        let mut targets_vec = Vec::with_capacity(batch_size);

        for item in items {
            // 画像をロードして正規化（CPUメモリ上）
            match load_and_normalize_image(&item.path) {
                Ok(image_data) => {
                    all_pixels.extend_from_slice(&image_data);
                    targets_vec.push(item.label as i64);
                    // image_dataはここでドロップ（すぐにメモリ解放）
                }
                Err(e) => {
                    eprintln!("警告: 画像読み込み失敗 {}: {}", item.path.display(), e);
                    // エラーの場合はゼロで埋める
                    all_pixels.extend(vec![0.0f32; 3 * IMAGE_SIZE * IMAGE_SIZE]);
                    targets_vec.push(item.label as i64);
                }
            }
        }

        // 1回の転送でバッチ全体をGPUメモリへ
        let images = Tensor::<B, 1>::from_floats(all_pixels.as_slice(), &self.device)
            .reshape([batch_size, 3, IMAGE_SIZE, IMAGE_SIZE]);
        let targets = Tensor::<B, 1, Int>::from_ints(targets_vec.as_slice(), &self.device);

        // CPUメモリを明示的に解放
        drop(all_pixels);
        drop(targets_vec);

        IconBatch { images, targets }
    }
}

#[cfg(feature = "ml")]
impl<B: AutodiffBackend> TrainStep<IconBatch<B>, ClassificationOutput<B>> for IconClassifier<B> {
    fn step(&self, batch: IconBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        let grads = item.loss.backward();
        TrainOutput::new(self, grads, item)
    }
}

#[cfg(feature = "ml")]
impl<B: Backend> ValidStep<IconBatch<B>, ClassificationOutput<B>> for IconClassifier<B> {
    fn step(&self, batch: IconBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[cfg(feature = "ml")]
#[derive(Parser, Debug)]
#[command(name = "train_model")]
#[command(about = "入力アイコン分類モデルのトレーニング", long_about = None)]
struct Args {
    /// 学習データディレクトリ（各サブディレクトリがクラスラベル）
    #[arg(short, long, default_value = "training_data")]
    data_dir: String,

    /// 出力モデルのパス（.tar.gz拡張子は自動追加）
    #[arg(short, long, default_value = "models/icon_classifier")]
    output: String,

    /// ボタンラベルのカンマ区切りリスト（方向入力とothersを除く）
    /// 例: "A1,A2,B,W,Start"
    #[arg(short, long)]
    buttons: Option<String>,

    /// エポック数
    #[arg(short, long, default_value_t = 50)]
    epochs: usize,

    /// バッチサイズ
    #[arg(long, default_value_t = 8)]
    batch_size: usize,

    /// 学習率
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f64,

    /// 検証データの割合（0.0-1.0）
    #[arg(long, default_value_t = 0.2)]
    val_ratio: f32,
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
    #[config(default = 0)]
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

    eprintln!("📊 バッチャーを作成中...");
    let batcher_train = IconBatcher::<B>::new(device.clone());
    let batcher_val = IconBatcher::<B::InnerBackend>::new(device.clone());

    eprintln!("📊 データローダーを作成中...");
    // num_workers=0: 各バッチを学習ループ内でオンデマンド読み込み（メモリ効率的）
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(0)
        .build(dataset_train);

    let dataloader_val = DataLoaderBuilder::new(batcher_val)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(0)
        .build(dataset_val);

    eprintln!("🧠 モデルを初期化中...");
    let model = config.model.init::<B>(&device);
    eprintln!("✓ モデル初期化完了");

    eprintln!("📚 Learnerを構築中...");
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .learning_strategy(LearningStrategy::SingleDevice(device.clone()))
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            model,
            config.optimizer.init(),
            config.learning_rate,
        );

    eprintln!("🚀 学習ループ開始...");
    let model_trained = learner.fit(dataloader_train, dataloader_val);
    eprintln!("✓ 学習ループ完了");

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
    let args = Args::parse();
    let config = AppConfig::load_or_default();

    let data_dir = PathBuf::from(&args.data_dir);

    if !data_dir.exists() {
        return Err(anyhow::anyhow!(
            "データディレクトリが見つかりません: {}",
            data_dir.display()
        ));
    }

    println!("=================================================================================");
    println!("アイコン分類モデル学習 (Burn)");
    println!("=================================================================================");
    println!("\nデータディレクトリ: {}", data_dir.display());
    println!("出力先: {}.tar.gz", args.output);
    println!("エポック数: {}", args.epochs);
    println!("バッチサイズ: {}", args.batch_size);
    println!("学習率: {}", args.learning_rate);
    println!("検証データ割合: {:.1}%", args.val_ratio * 100.0);

    // ボタンラベルを先に読み込む
    let button_labels: Vec<String> = if let Some(buttons_str) = &args.buttons {
        buttons_str.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        // buttons.txtから読み込む
        let buttons_file = data_dir.join("buttons.txt");
        if buttons_file.exists() {
            let content = std::fs::read_to_string(&buttons_file)?;
            content.trim().split(',').map(|s| s.trim().to_string()).collect()
        } else {
            // デフォルト: BUTTONLABELSから方向入力とothersを除いたもの
            BUTTON_LABELS.iter()
                .filter(|&&label| !label.starts_with("dir_") && label != "empty")
                .map(|&s| s.to_string())
                .collect()
        }
    };

    println!("\nボタンラベル（CSV/表示順）:");
    for (i, label) in button_labels.iter().enumerate() {
        println!("  {}: {}", i, label);
    }

    // クラス順序を生成: dir_1~9, button_labelsの順
    let class_names = IconDataset::detect_classes(&data_dir, &button_labels)?;
    let num_classes = class_names.len();
    
    println!("\nモデルのクラス順序:");
    for (i, class_name) in class_names.iter().enumerate() {
        println!("  {}: {}", i, class_name);
    }

    // デバイス設定
    let device = burn_wgpu::WgpuDevice::default();
    println!("\n使用デバイス: WGPU (GPU) - {:?}", device);

    // データセット読み込み
    let dataset = IconDataset::load(&data_dir, &class_names)?;
    let (dataset_train, dataset_val) = dataset.split(1.0 - args.val_ratio);

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
        .with_num_epochs(args.epochs)
        .with_batch_size(args.batch_size)
        .with_learning_rate(args.learning_rate);

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
    
    let model_path = PathBuf::from(&args.output);
    
    // モデルバイナリを読み込み
    let model_binary = std::fs::read("models/model.mpk")
        .context("モデルファイルの読み込みに失敗しました")?;

    // 学習データから画像サイズを取得
    let (image_width, image_height) = detect_image_size_from_dataset(&data_dir)?;
    println!("検出された学習データ画像サイズ: {}x{}", image_width, image_height);
    // メタデータを作成
    let metadata = ModelMetadata::new(
        button_labels,
        image_width,
        image_height,
        config.button_tile.source_video_width,
        config.button_tile.source_video_height,
        config.button_tile.x,
        config.button_tile.y,
        config.button_tile.tile_size,
        config.button_tile.tile_size,
        config.button_tile.columns_per_row,
        IMAGE_SIZE as u32,  // model_input_size
        args.epochs as u32,
    );

    // Tar.gz形式で保存
    model_storage::save_model_with_metadata(&model_path, &metadata, &model_binary)
        .context("モデルとメタデータの保存に失敗しました")?;

    let tar_gz_path = model_path.with_extension("tar.gz");
    println!("\n✓ モデルを保存しました: {}", tar_gz_path.display());
    println!("\nTar.gzファイル内容:");
    println!("  metadata.json - メタデータ（ボタン情報、タイル設定、解像度情報）");
    println!("  model.bin     - 学習済みモデルの重み");

    // メタデータを表示
    model_storage::print_metadata_info(&metadata);

    println!("\n=== 保存されたメタデータの詳細 ===");
    println!("ボタンラベル: {:?}", metadata.button_labels);
    println!("学習データ画像サイズ: {}x{}", metadata.image_width, metadata.image_height);
    println!("解析対象タイル:");
    println!("  位置: ({}, {})", metadata.tile_x, metadata.tile_y);
    println!("  サイズ: {}x{}", metadata.tile_width, metadata.tile_height);
    println!("  列数: {}", metadata.columns_per_row);
    println!("モデル入力サイズ: {}x{}", metadata.model_input_size, metadata.model_input_size);
    println!("学習エポック数: {}", metadata.num_epochs);

    println!("\n次のステップ:");
    println!("  1. tar.gzファイルをGUIアプリで読み込み");
    println!("  2. 動画から入力履歴を自動抽出");
    println!("  3. 必要に応じてデータ収集と再学習");

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
