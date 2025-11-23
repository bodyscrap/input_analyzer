use anyhow::{Context, Result};
use image::{DynamicImage, ImageEncoder};
use image::codecs::png::{PngEncoder, CompressionType, FilterType};
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

/// PNG画像を非圧縮で保存
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

/// テンプレート準備ツール
/// 既存のセル画像をカテゴリ別に分類して、テンプレートとして使用できるようにする

fn print_usage() {
    println!("=== テンプレート準備ツール ===");
    println!();
    println!("使用方法:");
    println!("  prepare_templates <入力セルディレクトリ> <テンプレート出力ディレクトリ> [サンプル数]");
    println!();
    println!("引数:");
    println!("  <入力セルディレクトリ>       : extract_all_cellsで抽出したセルが格納されているディレクトリ");
    println!("  <テンプレート出力ディレクトリ> : テンプレート画像を保存するディレクトリ");
    println!("  [サンプル数]                  : 各カテゴリから収集するサンプル数（デフォルト: 150）");
    println!();
    println!("例:");
    println!("  # 各カテゴリから150枚ずつサンプルを収集");
    println!("  prepare_templates input_cells templates 150");
    println!();
    println!("出力:");
    println!("  templates/");
    println!("    ├── dir_up/              # 方向：上（通常）");
    println!("    ├── dir_up_indicator/    # 方向：上（インジケータ映り込み）");
    println!("    ├── dir_right/           # 方向：右（通常）");
    println!("    ├── button_a1/           # A_1ボタン（通常）");
    println!("    ├── empty/               # 空（入力なし）");
    println!("    └── ...");
    println!();
    println!("注意:");
    println!("  - 0-2行目のセルにはインジケータが映り込むため、");
    println!("    別途 *_indicator フォルダにサンプルを収集してください");
}

/// セル画像を収集
fn collect_cell_images<P: AsRef<Path>>(input_dir: P) -> Result<Vec<PathBuf>> {
    let mut cells = Vec::new();

    println!("セル画像を収集中...");

    for entry in walkdir::WalkDir::new(input_dir)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("png") {
            // フレームカウント以外の画像（入力アイコン）のみ収集
            if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
                if file_name.contains("_input.png") {
                    cells.push(path.to_path_buf());
                }
            }
        }
    }

    println!("  収集したセル画像: {}個", cells.len());

    // 行番号別に分類
    let mut cells_by_row: HashMap<u32, Vec<PathBuf>> = HashMap::new();
    for path in &cells {
        if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
            // row00_col01_input.png -> 行番号0を抽出
            if let Some(row_str) = file_name.strip_prefix("row").and_then(|s| s.split('_').next()) {
                if let Ok(row_num) = row_str.parse::<u32>() {
                    cells_by_row.entry(row_num).or_insert_with(Vec::new).push(path.clone());
                }
            }
        }
    }

    println!("  行0-2（インジケータ映り込み）: {}個",
             cells_by_row.get(&0).map(|v| v.len()).unwrap_or(0) +
             cells_by_row.get(&1).map(|v| v.len()).unwrap_or(0) +
             cells_by_row.get(&2).map(|v| v.len()).unwrap_or(0));
    println!("  行3-15（通常）: {}個",
             (3..=15).map(|i| cells_by_row.get(&i).map(|v| v.len()).unwrap_or(0)).sum::<usize>());

    Ok(cells)


}

/// 画像の平均輝度を計算
fn calculate_brightness(image: &image::RgbImage) -> f64 {
    let (width, height) = image.dimensions();
    let mut sum = 0u64;
    let total_pixels = (width * height) as u64;

    for pixel in image.pixels() {
        let brightness = (pixel[0] as u64 + pixel[1] as u64 + pixel[2] as u64) / 3;
        sum += brightness;
    }

    sum as f64 / total_pixels as f64
}

/// 画像のヒストグラムを計算（簡易版）
fn calculate_histogram(image: &image::RgbImage) -> Vec<u32> {
    let mut histogram = vec![0u32; 256];

    for pixel in image.pixels() {
        let gray = ((pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32) / 3) as usize;
        histogram[gray] += 1;
    }

    histogram
}

/// 画像の分散を計算
fn calculate_variance(image: &image::RgbImage, mean: f64) -> f64 {
    let (width, height) = image.dimensions();
    let mut sum_sq = 0.0;
    let total_pixels = (width * height) as f64;

    for pixel in image.pixels() {
        let brightness = (pixel[0] as f64 + pixel[1] as f64 + pixel[2] as f64) / 3.0;
        let diff = brightness - mean;
        sum_sq += diff * diff;
    }

    sum_sq / total_pixels
}

/// 画像を特徴量で分類
fn classify_images(
    cell_paths: &[PathBuf],
    sample_count: usize,
) -> Result<HashMap<String, Vec<PathBuf>>> {
    println!("\n画像を分析して分類中...");

    let mut empty_cells = Vec::new();
    let mut non_empty_cells = Vec::new();
    let mut indicator_cells = Vec::new(); // 0-2行目のセル

    // 空と非空を分類、さらに行番号で分類
    for path in cell_paths {
        let img = image::open(path)?.to_rgb8();
        let brightness = calculate_brightness(&img);
        let variance = calculate_variance(&img, brightness);

        // 行番号を取得
        let row_num = if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
            if let Some(row_str) = file_name.strip_prefix("row").and_then(|s| s.split('_').next()) {
                row_str.parse::<u32>().ok()
            } else {
                None
            }
        } else {
            None
        };

        // 輝度と分散で空のセルを判定（閾値は調整が必要）
        if brightness < 30.0 && variance < 100.0 {
            empty_cells.push(path.clone());
        } else if let Some(row) = row_num {
            if row <= 2 {
                // 0-2行目はインジケータ映り込み
                indicator_cells.push(path.clone());
            } else {
                // 3-15行目は通常
                non_empty_cells.push(path.clone());
            }
        } else {
            non_empty_cells.push(path.clone());
        }
    }

    println!("  空セル候補: {}個", empty_cells.len());
    println!("  非空セル候補（通常）: {}個", non_empty_cells.len());
    println!("  非空セル候補（インジケータ映り込み）: {}個", indicator_cells.len());

    let mut categories: HashMap<String, Vec<PathBuf>> = HashMap::new();

    // 空セルのサンプルを追加（一定間隔でサンプリング）
    let empty_sample_count = sample_count.min(empty_cells.len());
    let empty_step = if empty_cells.len() > empty_sample_count {
        empty_cells.len() / empty_sample_count
    } else {
        1
    };
    categories.insert(
        "empty".to_string(),
        empty_cells.into_iter().step_by(empty_step).take(empty_sample_count).collect(),
    );

    // 通常の非空セルを未分類として保存（一定間隔でサンプリング）
    let non_empty_sample_count = sample_count.min(non_empty_cells.len());
    let non_empty_step = if non_empty_cells.len() > non_empty_sample_count {
        non_empty_cells.len() / non_empty_sample_count
    } else {
        1
    };
    categories.insert(
        "unclassified".to_string(),
        non_empty_cells
            .into_iter()
            .step_by(non_empty_step)
            .take(non_empty_sample_count)
            .collect(),
    );

    // インジケータ映り込みセルを別カテゴリとして保存（一定間隔でサンプリング）
    let indicator_sample_count = sample_count.min(indicator_cells.len());
    if indicator_sample_count > 0 {
        let indicator_step = if indicator_cells.len() > indicator_sample_count {
            indicator_cells.len() / indicator_sample_count
        } else {
            1
        };
        categories.insert(
            "unclassified_indicator".to_string(),
            indicator_cells
                .into_iter()
                .step_by(indicator_step)
                .take(indicator_sample_count)
                .collect(),
        );
    }

    Ok(categories)
}

/// カテゴリ別にテンプレート画像を保存
fn save_templates<P: AsRef<Path>>(
    categories: HashMap<String, Vec<PathBuf>>,
    output_dir: P,
) -> Result<()> {
    let output_dir = output_dir.as_ref();
    println!("\nテンプレート画像を保存中...");

    for (category, paths) in categories {
        let category_dir = output_dir.join(&category);
        fs::create_dir_all(&category_dir)?;

        println!("  {}: {}個のサンプル", category, paths.len());

        for (i, path) in paths.iter().enumerate() {
            // 画像を読み込んでコピー（HTMLから参照できるように）
            let img = image::open(path)?;
            let output_path = category_dir.join(format!("sample_{:04}.png", i));
            save_png_uncompressed(&img, &output_path)?;
        }
    }

    Ok(())
}

/// 手動分類用のサマリーHTMLを生成
fn generate_classification_html<P: AsRef<Path>>(
    categories: &HashMap<String, Vec<PathBuf>>,
    output_dir: P,
) -> Result<()> {
    let output_dir = output_dir.as_ref();
    let html_path = output_dir.join("classify.html");

    println!("\n分類支援HTMLを生成中: {}", html_path.display());

    let mut html = String::from(
        r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>入力アイコン分類</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .category { margin-bottom: 40px; border: 2px solid #333; padding: 10px; }
        .category h2 { margin-top: 0; }
        .images { display: flex; flex-wrap: wrap; gap: 5px; }
        .image-container { text-align: center; }
        .image-container img {
            border: 1px solid #999;
            image-rendering: pixelated;
            width: 96px;
            height: 96px;
        }
        .image-container .filename { font-size: 10px; color: #666; }
        .instructions {
            background: #ffffcc;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid #cccc00;
        }
    </style>
</head>
<body>
    <h1>入力アイコン分類支援ツール</h1>
    <div class="instructions">
        <h3>使い方:</h3>
        <ol>
            <li>以下の画像を確認し、同じ種類のアイコンを見つけてください</li>
            <li>画像ファイルを適切なカテゴリフォルダに移動してください</li>
            <li>カテゴリフォルダ名を以下のいずれかに変更してください：
                <ul>
                    <li><strong>通常（3-15行目）:</strong>
                        <ul>
                            <li><code>dir_up</code>, <code>dir_up_right</code>, <code>dir_right</code>, <code>dir_down_right</code></li>
                            <li><code>dir_down</code>, <code>dir_down_left</code>, <code>dir_left</code>, <code>dir_up_left</code></li>
                            <li><code>button_a1</code>, <code>button_a2</code>, <code>button_b</code>, <code>button_w</code>, <code>button_start</code></li>
                        </ul>
                    </li>
                    <li><strong>インジケータ映り込み（0-2行目）:</strong>
                        <ul>
                            <li>上記のフォルダ名に <code>_indicator</code> を付ける</li>
                            <li>例: <code>dir_up_indicator</code>, <code>button_a1_indicator</code></li>
                        </ul>
                    </li>
                    <li><code>empty</code> - 入力なし</li>
                </ul>
            </li>
        </ol>
    </div>
"#,
    );

    for (category, paths) in categories {
        html.push_str(&format!("<div class='category'>\n<h2>{}</h2>\n", category));
        html.push_str("<div class='images'>\n");

        for (i, path) in paths.iter().enumerate() {
            if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
                // HTMLと同じディレクトリにある画像への相対パス
                let rel_path = format!("{}/sample_{:04}.png", category, i);

                html.push_str(&format!(
                    "<div class='image-container'>\
                     <img src='{}' title='{}'>\
                     <div class='filename'>{}</div>\
                     </div>\n",
                    rel_path, file_name, file_name
                ));
            }
        }

        html.push_str("</div>\n</div>\n");
    }

    html.push_str("</body>\n</html>");

    fs::write(html_path, html)?;

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        print_usage();
        anyhow::bail!("引数が不足しています");
    }

    let input_dir = PathBuf::from(&args[1]);
    let output_dir = PathBuf::from(&args[2]);
    let sample_count = if args.len() >= 4 {
        args[3].parse::<usize>().unwrap_or(150)
    } else {
        150
    };

    println!("=== テンプレート準備ツール ===\n");
    println!("設定:");
    println!("  入力ディレクトリ: {}", input_dir.display());
    println!("  出力ディレクトリ: {}", output_dir.display());
    println!("  サンプル数: {}個/カテゴリ", sample_count);

    // セル画像を収集
    let cell_paths = collect_cell_images(&input_dir)?;

    if cell_paths.is_empty() {
        anyhow::bail!("セル画像が見つかりませんでした");
    }

    // 画像を分類
    let categories = classify_images(&cell_paths, sample_count)?;

    // 出力ディレクトリを作成
    fs::create_dir_all(&output_dir)?;

    // テンプレート画像を保存
    save_templates(categories.clone(), &output_dir)?;

    // 分類支援HTMLを生成
    generate_classification_html(&categories, &output_dir)?;

    println!("\n✓ 完了!");
    println!("\n次のステップ:");
    println!("  1. {}を開いてください", output_dir.join("classify.html").display());
    println!("  2. 画像を確認し、同じ種類のアイコンを見つけてください");
    println!("  3. 画像ファイルを適切なカテゴリフォルダに移動してください");
    println!("  4. フォルダ名を以下のいずれかに変更してください:");
    println!("     【通常（3-15行目）】");
    println!("       - 方向: dir_up, dir_right, dir_down, dir_left, など");
    println!("       - ボタン: button_a1, button_a2, button_b, button_w, button_start");
    println!("     【インジケータ映り込み（0-2行目）】");
    println!("       - 上記に _indicator を付ける");
    println!("       - 例: dir_up_indicator, button_a1_indicator");
    println!("     【共通】");
    println!("       - 空: empty");

    Ok(())
}
