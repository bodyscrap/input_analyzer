//! 分類精度テストツール
//!
//! 動画から入力アイコンを抽出し、学習済みモデルで分類して結果を保存します。
//! 
//! # 使用方法
//! ```bash
//! cargo run --release --features ml --bin test_classification -- [オプション]
//! ```
//! 
//! # オプション
//! - 第1引数: 動画ディレクトリ（デフォルト: sample_data）
//! - 第2引数: フレーム抽出間隔（デフォルト: 3）
//! - 第3引数: モデルファイルパス（デフォルト: models/model.mpk）
//! - 第4引数: 出力ディレクトリ（デフォルト: test_results）

#[cfg(feature = "ml")]
use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::Tensor,
};

#[cfg(feature = "ml")]
use input_analyzer::frame_extractor::FrameExtractor;
#[cfg(feature = "ml")]
use input_analyzer::input_analyzer::{InputAnalyzer, InputIndicatorRegion};
#[cfg(feature = "ml")]
use input_analyzer::ml_model::{IconClassifier, ModelConfig, CLASS_NAMES, load_and_normalize_image};

#[cfg(feature = "ml")]
use anyhow::{Context, Result};
#[cfg(feature = "ml")]
use std::env;
#[cfg(feature = "ml")]
use std::fs;
#[cfg(feature = "ml")]
use std::path::{Path, PathBuf};

#[cfg(feature = "ml")]
type MyBackend = burn_wgpu::Wgpu;
#[cfg(feature = "ml")]
type MyDevice = burn_wgpu::WgpuDevice;

/// コマンドライン引数
#[cfg(feature = "ml")]
struct Args {
    video_dir: PathBuf,
    frame_interval: u32,
    model_path: PathBuf,
    output_dir: PathBuf,
}

#[cfg(feature = "ml")]
impl Args {
    fn parse() -> Self {
        let args: Vec<String> = env::args().collect();
        
        let video_dir = if args.len() >= 2 {
            PathBuf::from(&args[1])
        } else {
            PathBuf::from("sample_data")
        };
        
        let frame_interval = if args.len() >= 3 {
            args[2].parse().unwrap_or(3)
        } else {
            3
        };
        
        let model_path = if args.len() >= 4 {
            PathBuf::from(&args[3])
        } else {
            PathBuf::from("models/model.mpk")
        };
        
        let output_dir = if args.len() >= 5 {
            PathBuf::from(&args[4])
        } else {
            PathBuf::from("test_results")
        };
        
        Self {
            video_dir,
            frame_interval,
            model_path,
            output_dir,
        }
    }
}

/// モデルをロード
#[cfg(feature = "ml")]
fn load_model(model_path: &Path, device: &MyDevice) -> Result<IconClassifier<MyBackend>> {
    println!("モデルをロード中: {}", model_path.display());
    
    let config = ModelConfig::new(14);
    let model = config.init(device);
    
    let record = CompactRecorder::new()
        .load(model_path.to_path_buf(), device)
        .context("モデルファイルの読み込みに失敗しました")?;
    
    let model = model.load_record(record);
    println!("✓ モデルをロードしました\n");
    
    Ok(model)
}

/// 動画ファイルを収集
#[cfg(feature = "ml")]
fn collect_video_files(video_dir: &Path) -> Result<Vec<PathBuf>> {
    let video_extensions = ["mp4", "avi", "mov", "mkv", "webm", "flv"];
    let mut videos = Vec::new();
    
    for entry in fs::read_dir(video_dir)
        .with_context(|| format!("ディレクトリが開けません: {}", video_dir.display()))? 
    {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if video_extensions.contains(&ext.to_str().unwrap_or("")) {
                    videos.push(path);
                }
            }
        }
    }
    
    videos.sort();
    Ok(videos)
}

/// フレームから最下行の入力アイコンを抽出
#[cfg(feature = "ml")]
fn extract_bottom_row_icons(frame_path: &Path) -> Result<Vec<image::RgbImage>> {
    let img = image::open(frame_path)
        .with_context(|| format!("画像を開けません: {}", frame_path.display()))?;
    
    // 入力インジケータ領域の設定（デフォルト）
    let region = InputIndicatorRegion::default();
    let analyzer = InputAnalyzer::new(region);
    
    // すべての行を抽出
    let rows = analyzer.extract_all_rows(&img)?;
    
    // 最下行（行15）の入力アイコンを取得
    if let Some(last_row) = rows.last() {
        Ok(last_row.input_icons.clone())
    } else {
        anyhow::bail!("入力行が抽出できませんでした");
    }
}

/// 画像を分類
#[cfg(feature = "ml")]
fn classify_image(
    model: &IconClassifier<MyBackend>,
    image_path: &Path,
    device: &MyDevice,
) -> Result<(usize, f32)> {
    // 画像を読み込んで正規化
    let image_data = load_and_normalize_image(image_path)?;
    
    // Tensorに変換 [1, 3, 48, 48]
    let tensor = Tensor::<MyBackend, 1>::from_floats(image_data.as_slice(), device)
        .reshape([1, 3, 48, 48]);
    
    // 予測
    let (predictions, logits) = model.predict(tensor);
    
    // 予測クラスと信頼度を取得
    let pred_class = predictions.into_data().to_vec::<i32>().unwrap()[0] as usize;
    let logits_vec = logits.into_data().to_vec::<f32>().unwrap();
    
    // Softmax計算（簡易版）
    let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits_vec.iter().map(|x| (x - max_logit).exp()).sum();
    let confidence = (logits_vec[pred_class] - max_logit).exp() / exp_sum;
    
    Ok((pred_class, confidence))
}

/// 動画を処理
#[cfg(feature = "ml")]
fn process_video(
    video_path: &Path,
    model: &IconClassifier<MyBackend>,
    device: &MyDevice,
    frame_interval: u32,
    video_output_dir: &Path,
) -> Result<()> {
    let video_name = video_path.file_stem().unwrap().to_str().unwrap();
    println!("\n=== 動画を処理中: {} ===", video_name);
    
    // 一時フレーム出力ディレクトリ
    let temp_frames_dir = PathBuf::from(format!("temp_frames_{}", video_name));
    fs::create_dir_all(&temp_frames_dir)?;
    
    // フレーム抽出
    println!("フレームを抽出中 (間隔: {} フレーム)...", frame_interval);
    let config = input_analyzer::frame_extractor::FrameExtractorConfig {
        frame_interval,
        output_dir: temp_frames_dir.clone(),
        image_format: "png".to_string(),
        jpeg_quality: 95,
    };
    
    let extractor = FrameExtractor::new(config);
    let frame_paths = extractor.extract_frames(video_path)?;
    println!("✓ {} フレームを抽出しました", frame_paths.len());
    
    // 各フレームを処理
    println!("各フレームのアイコンを分類中...");
    let mut total_icons = 0;
    let mut class_counts = std::collections::HashMap::new();
    
    for frame_path in &frame_paths {
        // 最下行のアイコンを抽出
        let icons = match extract_bottom_row_icons(frame_path) {
            Ok(icons) => icons,
            Err(e) => {
                eprintln!("  警告: {} のアイコン抽出に失敗: {}", frame_path.display(), e);
                continue;
            }
        };
        
        let frame_name = frame_path.file_stem().unwrap().to_str().unwrap();
        
        // 各アイコンを分類
        for (icon_idx, icon_img) in icons.iter().enumerate() {
            // 一時ファイルに保存
            let temp_icon_path = temp_frames_dir.join(format!("{}_{}.png", frame_name, icon_idx));
            icon_img.save(&temp_icon_path)?;
            
            // 分類
            let (class_id, confidence) = match classify_image(model, &temp_icon_path, device) {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("  警告: アイコン分類に失敗: {}", e);
                    fs::remove_file(&temp_icon_path)?;
                    continue;
                }
            };
            
            let class_name = CLASS_NAMES[class_id];
            *class_counts.entry(class_name).or_insert(0) += 1;
            
            // クラスごとのディレクトリに保存
            let class_dir = video_output_dir.join(class_name);
            fs::create_dir_all(&class_dir)?;
            
            // ファイル名: videoname_framename_iconidx.png (confidenceは含めない)
            let output_filename = format!("{}_{}_{}.png", video_name, frame_name, icon_idx);
            let output_path = class_dir.join(output_filename);
            
            fs::copy(&temp_icon_path, &output_path)?;
            fs::remove_file(&temp_icon_path)?;
            
            total_icons += 1;
        }
    }
    
    // 一時ディレクトリを削除
    fs::remove_dir_all(&temp_frames_dir)?;
    
    // 結果サマリー
    println!("\n--- 処理結果 ---");
    println!("総アイコン数: {}", total_icons);
    println!("\nクラス別内訳:");
    let mut class_list: Vec<_> = class_counts.iter().collect();
    class_list.sort_by_key(|(name, _)| *name);
    for (class_name, count) in class_list {
        println!("  {}: {} 枚", class_name, count);
    }
    
    Ok(())
}

#[cfg(feature = "ml")]
fn main() -> Result<()> {
    println!("=== 分類精度テストツール ===\n");
    
    // 引数解析
    let args = Args::parse();
    
    println!("設定:");
    println!("  動画ディレクトリ: {}", args.video_dir.display());
    println!("  フレーム間隔: {} フレーム", args.frame_interval);
    println!("  モデルファイル: {}", args.model_path.display());
    println!("  出力ディレクトリ: {}", args.output_dir.display());
    println!();
    
    // 出力ディレクトリを作成
    fs::create_dir_all(&args.output_dir)?;
    
    // デバイス初期化
    let device = MyDevice::default();
    println!("デバイス: {:?}\n", device);
    
    // モデルをロード
    let model = load_model(&args.model_path, &device)?;
    
    // 動画ファイルを収集
    println!("動画ファイルを検索中...");
    let video_files = collect_video_files(&args.video_dir)?;
    
    if video_files.is_empty() {
        anyhow::bail!("動画ファイルが見つかりません: {}", args.video_dir.display());
    }
    
    println!("✓ {} 個の動画ファイルを発見しました\n", video_files.len());
    for (i, video) in video_files.iter().enumerate() {
        println!("  {}. {}", i + 1, video.file_name().unwrap().to_str().unwrap());
    }
    
    // 各動画を処理
    for (i, video_path) in video_files.iter().enumerate() {
        let video_name = video_path.file_stem().unwrap().to_str().unwrap();
        let video_output_dir = args.output_dir.join(video_name);
        fs::create_dir_all(&video_output_dir)?;
        
        println!("\n\n[{}/{}] 処理中...", i + 1, video_files.len());
        
        if let Err(e) = process_video(video_path, &model, &device, args.frame_interval, &video_output_dir) {
            eprintln!("エラー: {} の処理に失敗しました: {}", video_name, e);
            continue;
        }
    }
    
    println!("\n\n=== すべての処理が完了しました ===");
    println!("結果は {} に保存されています", args.output_dir.display());
    
    Ok(())
}

#[cfg(not(feature = "ml"))]
fn main() {
    eprintln!("このツールを使用するには、mlフィーチャーを有効にしてビルドしてください:");
    eprintln!("cargo run --release --features ml --bin test_classification");
    std::process::exit(1);
}
