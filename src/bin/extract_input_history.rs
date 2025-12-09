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
//! duration,direction,A1,A2,B,W,Start
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
use input_analyzer::config::{AppConfig, DeviceType};
#[cfg(feature = "ml")]
use input_analyzer::frame_extractor::FrameExtractor;
#[cfg(feature = "ml")]
use input_analyzer::input_history_extractor::{
    extract_bottom_row_icons, update_input_state, InputState,
};
#[cfg(feature = "ml")]
use input_analyzer::ml_model::{
    load_and_normalize_image, IconClassifier, ModelConfig,
};
#[cfg(feature = "ml")]
use input_analyzer::model_metadata::ModelMetadata;
#[cfg(feature = "ml")]
use input_analyzer::model_storage;

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
    let tensor =
        Tensor::<MyBackend, 1>::from_floats(image_data.as_slice(), device).reshape([1, 3, 48, 48]);

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
    metadata: &ModelMetadata,
    class_names: &[String],
) -> Result<InputState> {
    let mut state = InputState::new();

    // メタデータから解析領域を取得
    use input_analyzer::input_analyzer::InputIndicatorRegion;
    
    // tile_x, tile_y = 解析対象の左上座標（継続フレーム数列を除く）
    // tile_width/height = 1セルのサイズ（正方形）
    // columns_per_row = 解析対象列数（方向1 + ボタン5 = 6）
    
    let region = InputIndicatorRegion {
        x: metadata.tile_x,
        y: metadata.tile_y,
        width: metadata.tile_width * metadata.columns_per_row,
        height: metadata.tile_height,
        rows: 1, // 最下行のみを抽出するので1行
        cols: metadata.columns_per_row,
    };

    // 最下行のアイコンを抽出
    let icons = extract_bottom_row_icons(frame_path, &region)?;

    // 各列を分類
    // - 1列目（icon_idx=0）: 方向キー、ボタン、その他すべてが入る可能性
    // - 2列目以降: ボタンまたはその他のみ（方向キーは最左列のみに出現）
    for (icon_idx, icon_img) in icons.iter().enumerate() {
        // 一時ファイルに保存
        let temp_icon_path = temp_dir.join(format!("temp_icon_{}.png", icon_idx));
        icon_img.save(&temp_icon_path)?;

        // 分類
        let class_id = classify_image(model, &temp_icon_path, device)?;
        let class_name = if class_id < class_names.len() {
            &class_names[class_id]
        } else {
            "others"
        };

        // 方向キーは最左列（icon_idx=0）のみで有効
        // 2列目以降で方向キーが検出された場合は無視（学習データが正しければ発生しない）
        if icon_idx > 0 && class_name.starts_with("dir_") {
            // 2列目以降で方向キーが検出された場合は警告のみ（ボタンとしては扱わない）
            eprintln!("警告: {}列目で方向キー {} が検出されました（無視）", icon_idx + 1, class_name);
        } else {
            // 状態を更新
            update_input_state(&mut state, class_name);
        }

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
    metadata: &ModelMetadata,
    class_names: &[String],
) -> Result<()> {
    println!("=== 入力履歴抽出 ===");
    println!("動画: {}", video_path.display());
    println!("出力: {}\n", output_csv_path.display());

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
    
    // 動画解像度を検証
    if let Some(first_frame_path) = frame_paths.first() {
        let img = image::open(first_frame_path)
            .context("最初のフレームの読み込みに失敗しました")?;
        let video_width = img.width();
        let video_height = img.height();
        
        if video_width != metadata.video_width || video_height != metadata.video_height {
            return Err(anyhow::anyhow!(
                "動画解像度が学習時と異なります。\n  学習時: {}x{}\n  入力動画: {}x{}\n学習時と同じ解像度の動画を使用してください。",
                metadata.video_width, metadata.video_height,
                video_width, video_height
            ));
        }
        println!("✓ 動画解像度を検証: {}x{}", video_width, video_height);
    }
    println!("✓ {} フレームを抽出しました", frame_paths.len());

    // CSV出力準備（メタデータからボタンラベルを取得）
    let mut csv_file = File::create(output_csv_path)?;
    let button_header = metadata.button_labels.join(",");
    writeln!(
        csv_file,
        "duration,direction,{}",
        button_header
    )?;

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
        let state = match extract_input_from_frame(frame_path, model, device, &temp_dir, metadata, &class_names) {
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
                writeln!(csv_file, "{}", prev_state.to_csv_line(duration, &metadata.button_labels))?;
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
        writeln!(csv_file, "{}", state.to_csv_line(duration, &metadata.button_labels))?;
        total_records += 1;
    }

    println!(
        "\r✓ 処理完了: {}/{} フレーム",
        frame_paths.len(),
        frame_paths.len()
    );

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
    // 設定ファイルを読み込み（存在しない場合はデフォルト設定）
    let mut config = AppConfig::load_or_default();

    println!("=== 入力履歴抽出 ===\n");

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "使用方法: {} <動画ファイル> [出力CSVパス] [モデルパス] [デバイス]",
            args[0]
        );
        eprintln!();
        eprintln!("例:");
        eprintln!("  {} video.mp4", args[0]);
        eprintln!("  {} video.mp4 output.csv", args[0]);
        eprintln!("  {} video.mp4 output.csv models/custom_model.mpk", args[0]);
        eprintln!(
            "  {} video.mp4 output.csv models/custom_model.mpk gpu",
            args[0]
        );
        eprintln!();
        eprintln!("デバイス: gpu (または wgpu) / cpu");
        std::process::exit(1);
    }

    let video_path = PathBuf::from(&args[1]);
    let output_csv_path = if args.len() >= 3 {
        PathBuf::from(&args[2])
    } else {
        let video_stem = video_path.file_stem().unwrap().to_str().unwrap();
        PathBuf::from(format!("{}_input_history.csv", video_stem))
    };
    let model_path_arg = if args.len() >= 4 {
        PathBuf::from(&args[3])
    } else {
        PathBuf::from(&config.model.model_path)
    };
    
    // tar.gz形式のモデルパスに自動変換
    let model_path = if model_path_arg.extension().and_then(|s| s.to_str()) == Some("gz") {
        model_path_arg
    } else {
        // 拡張子がない、または.mpk/.binの場合は.tar.gzを追加
        let mut tar_gz_path = model_path_arg.clone();
        tar_gz_path.set_extension("tar.gz");
        if tar_gz_path.exists() {
            tar_gz_path
        } else {
            model_path_arg // 元のパスを使用
        }
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

    // 動画ファイルの存在確認
    if !video_path.exists() {
        anyhow::bail!("動画ファイルが見つかりません: {}", video_path.display());
    }

    // モデルの存在確認
    if !model_path.exists() {
        anyhow::bail!("モデルファイルが見つかりません: {}", model_path.display());
    }

    // デバイス設定（設定ファイルの値を使用）
    let device = match config.device_type {
        DeviceType::Wgpu => {
            let dev = MyDevice::default();
            println!("使用デバイス: WGPU (GPU) - {:?}", dev);
            dev
        }
        DeviceType::Cpu => {
            println!("警告: CPU (NdArray) モードは現在このバイナリではサポートされていません");
            println!("WGPU (GPU) を使用します");
            let dev = MyDevice::default();
            println!("使用デバイス: WGPU (GPU) - {:?}", dev);
            dev
        }
    };

    // モデル読み込み（tar.gz形式）
    println!("モデルを読み込み中: {}", model_path.display());
    
    let (metadata, model_binary) = model_storage::load_model_with_metadata(&model_path)
        .context("モデルの読み込みに失敗しました")?;
    
    // クラス順序: dir_1~9 (ニュートラルの5を除く), button_labelsの順
    let mut class_names: Vec<String> = vec![
        "dir_1".to_string(), "dir_2".to_string(), "dir_3".to_string(),
        "dir_4".to_string(), "dir_6".to_string(), "dir_7".to_string(),
        "dir_8".to_string(), "dir_9".to_string(),
    ];
    class_names.extend(metadata.button_labels.clone());
    
    let num_classes = class_names.len();
    
    println!("モデル情報:");
    println!("  画像サイズ: {}x{}", metadata.image_width, metadata.image_height);
    println!("  クラス数: {}", num_classes);
    println!("  ボタンラベル: {:?}", metadata.button_labels);
    println!("  タイル領域: ({}, {}) {}x{}", 
        metadata.tile_x, metadata.tile_y, 
        metadata.tile_width, metadata.tile_height);
    
    // 一時ファイルに保存してロード
    let temp_model_file = std::env::temp_dir().join("temp_model.mpk");
    std::fs::write(&temp_model_file, &model_binary)?;
    
    let record = CompactRecorder::new()
        .load(temp_model_file.clone(), &device)
        .context("モデルレコードの読み込みに失敗しました")?;
    
    std::fs::remove_file(&temp_model_file)?;

    let model = ModelConfig::new(num_classes)
        .init::<MyBackend>(&device)
        .load_record(record);

    println!("✓ モデル読み込み完了\n");

    // 入力履歴抽出
    extract_input_history(&video_path, &output_csv_path, &model, &device, &metadata, &class_names)?;

    // 設定を更新して保存
    config.update_last_video_path(&video_path);
    config.update_last_output_dir(output_csv_path.parent().unwrap_or(Path::new(".")));
    config.set_model_path(model_path.to_string_lossy().to_string());

    if let Err(e) = config.save_default() {
        eprintln!("警告: 設定ファイルの保存に失敗しました: {}", e);
    }

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
