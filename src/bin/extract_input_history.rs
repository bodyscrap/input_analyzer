//! 動画から入力履歴をCSV形式で抽出するアプリケーション
//!
//! # 機能
//! - 動画の各フレームから入力アイコン領域を抽出
//! - 学習済みモデルで入力を分類
//! - 連続する同じ入力をまとめて持続フレーム数とともにCSV出力
//!
//! # 使用方法
//! ```bash
//! cargo run --release --features ml --bin extract_input_history -- <動画ファイル> [出力CSVパス] [モデルパス]
//! ```
//!
//! # 出力CSV形式
//! ```
//! duration,direction,btn_a1,btn_a2,btn_b,btn_w,btn_start
//! 1,5,0,0,0,0,0
//! 3,6,1,0,0,0,0
//! ```

#[cfg(feature = "ml")]
use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::Tensor,
};

#[cfg(feature = "ml")]
use input_analyzer::frame_extractor::FrameExtractor;
#[cfg(feature = "ml")]
use input_analyzer::ml_model::{IconClassifier, ModelConfig, CLASS_NAMES, load_and_normalize_image};
#[cfg(feature = "ml")]
use input_analyzer::input_history_extractor::{InputState, update_input_state, extract_bottom_row_icons};

#[cfg(feature = "ml")]
use anyhow::{Context, Result};
#[cfg(feature = "ml")]
use std::env;
#[cfg(feature = "ml")]
use std::fs::{self, File};
#[cfg(feature = "ml")]
use std::io::Write;
#[cfg(feature = "ml")]
use std::path::{Path, PathBuf};

#[cfg(feature = "ml")]
type MyBackend = burn_wgpu::Wgpu;
#[cfg(feature = "ml")]
type MyDevice = burn_wgpu::WgpuDevice;

/// 画像を分類
#[cfg(feature = "ml")]
fn classify_image(
    model: &IconClassifier<MyBackend>,
    image_path: &Path,
    device: &MyDevice,
) -> Result<usize> {
    let image_data = load_and_normalize_image(image_path)?;
    let tensor = Tensor::<MyBackend, 1>::from_floats(image_data.as_slice(), device)
        .reshape([1, 3, 48, 48]);
    
    let (predictions, _) = model.predict(tensor);
    let pred_class = predictions.into_data().to_vec::<i32>().unwrap()[0] as usize;
    
    Ok(pred_class)
}

/// フレームから入力状態を抽出
#[cfg(feature = "ml")]
fn extract_input_from_frame(
    frame_path: &Path,
    model: &IconClassifier<MyBackend>,
    device: &MyDevice,
    temp_dir: &Path,
) -> Result<InputState> {
    let mut state = InputState::new();
    
    // 最下行のアイコンを抽出
    let icons = extract_bottom_row_icons(frame_path)?;
    
    // 各アイコンを分類
    for (icon_idx, icon_img) in icons.iter().enumerate() {
        // 一時ファイルに保存
        let temp_icon_path = temp_dir.join(format!("temp_icon_{}.png", icon_idx));
        icon_img.save(&temp_icon_path)?;
        
        // 分類
        let class_id = classify_image(model, &temp_icon_path, device)?;
        let class_name = CLASS_NAMES[class_id];
        
        // 状態を更新
        update_input_state(&mut state, class_name);
        
        // 一時ファイルを削除
        fs::remove_file(&temp_icon_path)?;
    }
    
    Ok(state)
}

/// 動画から入力履歴を抽出
#[cfg(feature = "ml")]
fn extract_input_history(
    video_path: &Path,
    output_csv_path: &Path,
    model: &IconClassifier<MyBackend>,
    device: &MyDevice,
) -> Result<()> {
    println!("=== 入力履歴抽出 ===");
    println!("動画: {}", video_path.display());
    println!("出力CSV: {}", output_csv_path.display());
    
    // 一時ディレクトリ作成
    let temp_dir = PathBuf::from("temp_extract");
    fs::create_dir_all(&temp_dir)?;
    
    // フレーム抽出ディレクトリ
    let temp_frames_dir = PathBuf::from("temp_frames");
    fs::create_dir_all(&temp_frames_dir)?;
    
    // フレーム抽出
    println!("\nフレームを抽出中...");
    let config = input_analyzer::frame_extractor::FrameExtractorConfig {
        frame_interval: 1,
        output_dir: temp_frames_dir.clone(),
        image_format: "png".to_string(),
        jpeg_quality: 95,
    };
    
    let extractor = FrameExtractor::new(config);
    let frame_paths = extractor.extract_frames(video_path)?;
    println!("✓ {} フレームを抽出しました", frame_paths.len());
    
    // CSV出力準備
    let mut csv_file = File::create(output_csv_path)?;
    writeln!(csv_file, "duration,direction,btn_a1,btn_a2,btn_b,btn_w,btn_start")?;
    
    // 入力履歴抽出
    println!("\n入力を解析中...");
    let mut current_state: Option<InputState> = None;
    let mut duration = 0u32;
    let mut total_records = 0usize;
    
    for (frame_idx, frame_path) in frame_paths.iter().enumerate() {
        // 進捗表示
        if frame_idx % 100 == 0 {
            print!("\r処理中: {}/{} フレーム", frame_idx + 1, frame_paths.len());
            std::io::stdout().flush()?;
        }
        
        // 入力状態を抽出
        let state = match extract_input_from_frame(frame_path, model, device, &temp_dir) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("\n警告: フレーム {} の処理に失敗: {}", frame_idx, e);
                continue;
            }
        };
        
        // 状態比較
        if let Some(ref prev_state) = current_state {
            if &state == prev_state {
                // 同じ入力が続いている
                duration += 1;
            } else {
                // 入力が変化した - 前の入力を記録
                writeln!(csv_file, "{}", prev_state.to_csv_line(duration))?;
                total_records += 1;
                
                // 新しい入力を開始
                current_state = Some(state);
                duration = 1;
            }
        } else {
            // 最初のフレーム
            current_state = Some(state);
            duration = 1;
        }
    }
    
    // 最後の入力を記録
    if let Some(ref state) = current_state {
        writeln!(csv_file, "{}", state.to_csv_line(duration))?;
        total_records += 1;
    }
    
    println!("\r✓ 処理完了: {}/{} フレーム", frame_paths.len(), frame_paths.len());
    
    // 一時ディレクトリを削除
    fs::remove_dir_all(&temp_dir)?;
    fs::remove_dir_all(&temp_frames_dir)?;
    
    println!("\n=== 完了 ===");
    println!("出力レコード数: {}", total_records);
    println!("CSV: {}", output_csv_path.display());
    
    Ok(())
}

#[cfg(feature = "ml")]
fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("使用方法: {} <動画ファイル> [出力CSVパス] [モデルパス]", args[0]);
        eprintln!();
        eprintln!("例:");
        eprintln!("  {} video.mp4", args[0]);
        eprintln!("  {} video.mp4 output.csv", args[0]);
        eprintln!("  {} video.mp4 output.csv models/custom_model.mpk", args[0]);
        std::process::exit(1);
    }
    
    let video_path = PathBuf::from(&args[1]);
    let output_csv_path = if args.len() >= 3 {
        PathBuf::from(&args[2])
    } else {
        let video_stem = video_path.file_stem().unwrap().to_str().unwrap();
        PathBuf::from(format!("{}_input_history.csv", video_stem))
    };
    let model_path = if args.len() >= 4 {
        PathBuf::from(&args[3])
    } else {
        PathBuf::from("models/model.mpk")
    };
    
    // 動画ファイルの存在確認
    if !video_path.exists() {
        anyhow::bail!("動画ファイルが見つかりません: {}", video_path.display());
    }
    
    // モデルの存在確認
    if !model_path.exists() {
        anyhow::bail!("モデルファイルが見つかりません: {}", model_path.display());
    }
    
    // デバイス設定
    let device = MyDevice::default();
    
    // モデル読み込み
    println!("モデルを読み込み中: {}", model_path.display());
    let record = CompactRecorder::new()
        .load(model_path.clone(), &device)
        .context("モデルの読み込みに失敗しました")?;
    
    let model = ModelConfig::new(CLASS_NAMES.len())
        .init::<MyBackend>(&device)
        .load_record(record);
    
    println!("✓ モデル読み込み完了");
    
    // 入力履歴抽出
    extract_input_history(&video_path, &output_csv_path, &model, &device)?;
    
    Ok(())
}

#[cfg(not(feature = "ml"))]
fn main() {
    eprintln!("エラー: このプログラムはml機能を有効にしてビルドする必要があります。");
    eprintln!();
    eprintln!("ビルドコマンド:");
    eprintln!("  cargo build --bin extract_input_history --features ml --release");
    eprintln!();
    std::process::exit(1);
}
