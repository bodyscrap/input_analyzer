use anyhow::{Context, Result};
use image::{DynamicImage, ImageEncoder, RgbImage};
use image::codecs::png::{PngEncoder, CompressionType, FilterType};

use input_analyzer::frame_extractor::FrameExtractor;
use input_analyzer::input_analyzer::{InputAnalyzer, InputIndicatorRegion};
use std::env;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

/// PNG画像を非圧縮で保存（DynamicImage用）
fn save_png_uncompressed(img: &DynamicImage, path: &Path) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("ファイルの作成に失敗: {}", path.display()))?;
    let writer = BufWriter::new(file);

    let encoder = PngEncoder::new_with_quality(
        writer,
        CompressionType::Fast,  // 非圧縮（最速）
        FilterType::NoFilter,   // フィルタなし
    );

    let color_type = match img.color() {
        image::ColorType::Rgb8 => image::ExtendedColorType::Rgb8,
        image::ColorType::Rgba8 => image::ExtendedColorType::Rgba8,
        image::ColorType::L8 => image::ExtendedColorType::L8,
        image::ColorType::La8 => image::ExtendedColorType::La8,
        _ => image::ExtendedColorType::Rgba8,
    };

    encoder.write_image(
        img.as_bytes(),
        img.width(),
        img.height(),
        color_type,
    ).context("PNG画像の書き込みに失敗しました")?;

    Ok(())
}

/// PNG画像を非圧縮で保存（RgbImage用）
fn save_png_uncompressed_rgb(img: &RgbImage, path: &Path) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("ファイルの作成に失敗: {}", path.display()))?;
    let writer = BufWriter::new(file);

    let encoder = PngEncoder::new_with_quality(
        writer,
        CompressionType::Fast,  // 非圧縮（最速）
        FilterType::NoFilter,   // フィルタなし
    );

    encoder.write_image(
        img.as_raw(),
        img.width(),
        img.height(),
        image::ExtendedColorType::Rgb8,
    ).context("PNG画像の書き込みに失敗しました")?;

    Ok(())
}

/// コマンドライン引数
struct Args {
    /// 動画ディレクトリ
    video_dir: PathBuf,
    /// 出力ディレクトリ
    output_dir: PathBuf,
    /// フレーム抽出間隔（デフォルト: 1 = 全フレーム）
    frame_interval: u32,
}

impl Args {
    fn parse() -> Result<Self> {
        let args: Vec<String> = env::args().collect();

        if args.len() < 2 {
            print_usage();
            anyhow::bail!("動画ディレクトリが指定されていません");
        }

        let video_dir = PathBuf::from(&args[1]);
        let output_dir = if args.len() >= 3 {
            PathBuf::from(&args[2])
        } else {
            PathBuf::from("input_cells")
        };

        let frame_interval = if args.len() >= 4 {
            args[3]
                .parse::<u32>()
                .context("フレーム間隔の解析に失敗しました")?
        } else {
            1 // デフォルトは全フレーム
        };

        Ok(Self {
            video_dir,
            output_dir,
            frame_interval,
        })
    }
}

fn print_usage() {
    println!("=== 入力セル一括抽出ツール ===");
    println!();
    println!("使用方法:");
    println!("  extract_all_cells <動画ディレクトリ> [出力ディレクトリ] [フレーム間隔]");
    println!();
    println!("引数:");
    println!("  <動画ディレクトリ> : 動画ファイルが格納されているディレクトリ（必須）");
    println!("  [出力ディレクトリ] : 抽出したセルを保存するディレクトリ（デフォルト: input_cells）");
    println!("  [フレーム間隔]     : フレーム抽出間隔（デフォルト: 1 = 全フレーム）");
    println!();
    println!("例:");
    println!("  # 全フレームを抽出（デフォルト）");
    println!("  extract_all_cells sample_data");
    println!();
    println!("  # 30フレームごとに抽出");
    println!("  extract_all_cells sample_data input_cells 30");
    println!();
    println!("  # 出力先を指定して全フレーム抽出");
    println!("  extract_all_cells sample_data my_output 1");
}

/// 動画ファイルのリストを取得
fn get_video_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut videos = Vec::new();

    if !dir.exists() {
        anyhow::bail!("ディレクトリが存在しません: {}", dir.display());
    }

    for entry in fs::read_dir(dir).context("ディレクトリの読み込みに失敗しました")? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_str().unwrap_or("").to_lowercase();
                if matches!(ext_str.as_str(), "mp4" | "avi" | "mov" | "mkv" | "webm" | "flv") {
                    videos.push(path);
                }
            }
        }
    }

    videos.sort();
    Ok(videos)
}

/// 1つの動画を処理
fn process_video(
    video_path: &Path,
    output_base_dir: &Path,
    frame_interval: u32,
) -> Result<()> {
    let video_name = video_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    println!("\n{}", "=".repeat(80));
    println!("動画: {}", video_path.display());
    println!("{}", "=".repeat(80));

    // フレーム抽出用の一時ディレクトリ
    let temp_frames_dir = output_base_dir.join(format!("temp_frames_{}", video_name));
    fs::create_dir_all(&temp_frames_dir)?;

    // フレーム抽出の設定
    let frame_config = input_analyzer::frame_extractor::FrameExtractorConfig {
        frame_interval,
        output_dir: temp_frames_dir.clone(),
        image_format: "png".to_string(),
        jpeg_quality: 95,
    };

    println!("\nステップ1: フレーム抽出");
    println!("  抽出間隔: {}フレームごと", frame_interval);

    let extractor = FrameExtractor::new(frame_config);
    let frame_paths = extractor
        .extract_frames(video_path)
        .context("フレーム抽出に失敗しました")?;

    println!("  抽出フレーム数: {}", frame_paths.len());

    // 入力インジケータ解析器を作成
    let region = InputIndicatorRegion::default();

    println!("\nステップ2: 入力セル抽出");
    println!("  入力インジケータ領域: ({}, {}), サイズ: {}x{}",
             region.x, region.y, region.width, region.height);
    println!("  セルサイズ: {}x{}", region.cell_width(), region.cell_height());

    let analyzer = InputAnalyzer::new(region);

    // 各フレームを処理
    let mut total_cells = 0;
    for (i, frame_path) in frame_paths.iter().enumerate() {
        let frame_name = frame_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("frame");

        // 画像を読み込み
        let image = image::open(frame_path)
            .with_context(|| format!("画像の読み込みに失敗: {}", frame_path.display()))?;

        // 入力行を抽出
        let rows = analyzer
            .extract_all_rows(&image)
            .with_context(|| format!("入力行の抽出に失敗: {}", frame_path.display()))?;

        // 各セルを保存
        let frame_output_dir = output_base_dir
            .join(video_name)
            .join(frame_name);
        fs::create_dir_all(&frame_output_dir)?;

        for row in &rows {
            // フレームカウント画像を保存（非圧縮PNG）
            let frame_count_path = frame_output_dir.join(format!(
                "row{:02}_col00_frame_count.png",
                row.row_index
            ));
            save_png_uncompressed_rgb(&row.frame_count_image, &frame_count_path)?;
            total_cells += 1;

            // 入力アイコンを保存（非圧縮PNG）
            for (col_idx, icon) in row.input_icons.iter().enumerate() {
                let icon_path = frame_output_dir.join(format!(
                    "row{:02}_col{:02}_input.png",
                    row.row_index,
                    col_idx + 1
                ));
                save_png_uncompressed_rgb(icon, &icon_path)?;
                total_cells += 1;
            }
        }

        if (i + 1) % 10 == 0 || i + 1 == frame_paths.len() {
            println!(
                "  進捗: {}/{} フレーム処理完了 ({:.1}%)",
                i + 1,
                frame_paths.len(),
                (i + 1) as f64 / frame_paths.len() as f64 * 100.0
            );
        }
    }

    // 一時ディレクトリを削除
    println!("\nステップ3: 一時ファイルのクリーンアップ");
    fs::remove_dir_all(&temp_frames_dir)
        .context("一時ディレクトリの削除に失敗しました")?;

    println!("\n✓ 動画の処理が完了しました");
    println!("  抽出フレーム数: {}", frame_paths.len());
    println!("  抽出セル数: {}", total_cells);
    println!("  出力先: {}", output_base_dir.join(video_name).display());

    Ok(())
}

fn main() -> Result<()> {
    println!("=== 入力セル一括抽出ツール ===\n");

    // 引数を解析
    let args = Args::parse()?;

    println!("設定:");
    println!("  動画ディレクトリ: {}", args.video_dir.display());
    println!("  出力ディレクトリ: {}", args.output_dir.display());
    println!("  フレーム間隔: {}フレームごと", args.frame_interval);

    // 動画ファイルを取得
    let video_files = get_video_files(&args.video_dir)?;

    if video_files.is_empty() {
        println!("\n警告: 動画ファイルが見つかりませんでした");
        return Ok(());
    }

    println!("\n検出された動画ファイル: {}個", video_files.len());
    for (i, video) in video_files.iter().enumerate() {
        println!("  {}. {}", i + 1, video.display());
    }

    // 出力ディレクトリを作成
    fs::create_dir_all(&args.output_dir)?;

    // 各動画を処理
    let mut success_count = 0;
    let mut error_count = 0;

    for video_path in &video_files {
        match process_video(video_path, &args.output_dir, args.frame_interval) {
            Ok(_) => success_count += 1,
            Err(e) => {
                eprintln!("\n✗ エラー: {}: {}", video_path.display(), e);
                error_count += 1;
            }
        }
    }

    // 最終サマリー
    println!("\n{}", "=".repeat(80));
    println!("=== 処理完了 ===");
    println!("{}", "=".repeat(80));
    println!("総動画数: {}", video_files.len());
    println!("成功: {}", success_count);
    println!("失敗: {}", error_count);
    println!("\n全ての入力セルは以下に保存されました:");
    println!("  {}", args.output_dir.display());

    if error_count > 0 {
        println!("\n警告: {}個の動画でエラーが発生しました", error_count);
    }

    Ok(())
}
