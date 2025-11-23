use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// カテゴリ定義
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum IconCategory {
    Dir1,      // 左下
    Dir2,      // 下
    Dir3,      // 右下
    Dir4,      // 左
    Dir6,      // 右
    Dir7,      // 左上
    Dir8,      // 上
    Dir9,      // 右上
    BtnA1,     // A1ボタン
    BtnA2,     // A2ボタン
    BtnB,      // Bボタン
    BtnW,      // Wボタン
    BtnStart,  // Startボタン
    Empty,     // 空白
}

impl IconCategory {
    fn all() -> Vec<Self> {
        vec![
            Self::Dir1,
            Self::Dir2,
            Self::Dir3,
            Self::Dir4,
            Self::Dir6,
            Self::Dir7,
            Self::Dir8,
            Self::Dir9,
            Self::BtnA1,
            Self::BtnA2,
            Self::BtnB,
            Self::BtnW,
            Self::BtnStart,
            Self::Empty,
        ]
    }

    fn template_name(&self) -> &str {
        match self {
            Self::Dir1 => "dir_1.png",
            Self::Dir2 => "dir_2.png",
            Self::Dir3 => "dir_3.png",
            Self::Dir4 => "dir_4.png",
            Self::Dir6 => "dir_6.png",
            Self::Dir7 => "dir_7.png",
            Self::Dir8 => "dir_8.png",
            Self::Dir9 => "dir_9.png",
            Self::BtnA1 => "btn_a1.png",
            Self::BtnA2 => "btn_a2.png",
            Self::BtnB => "btn_b.png",
            Self::BtnW => "btn_w.png",
            Self::BtnStart => "btn_start.png",
            Self::Empty => "", // 空白は特別処理
        }
    }

    fn folder_name(&self) -> &str {
        match self {
            Self::Dir1 => "dir_1",
            Self::Dir2 => "dir_2",
            Self::Dir3 => "dir_3",
            Self::Dir4 => "dir_4",
            Self::Dir6 => "dir_6",
            Self::Dir7 => "dir_7",
            Self::Dir8 => "dir_8",
            Self::Dir9 => "dir_9",
            Self::BtnA1 => "btn_a1",
            Self::BtnA2 => "btn_a2",
            Self::BtnB => "btn_b",
            Self::BtnW => "btn_w",
            Self::BtnStart => "btn_start",
            Self::Empty => "empty",
        }
    }

    fn label(&self) -> usize {
        match self {
            Self::Dir1 => 0,
            Self::Dir2 => 1,
            Self::Dir3 => 2,
            Self::Dir4 => 3,
            Self::Dir6 => 4,
            Self::Dir7 => 5,
            Self::Dir8 => 6,
            Self::Dir9 => 7,
            Self::BtnA1 => 8,
            Self::BtnA2 => 9,
            Self::BtnB => 10,
            Self::BtnW => 11,
            Self::BtnStart => 12,
            Self::Empty => 13,
        }
    }
}

/// テンプレートマッチング用の構造体
struct TemplateSet {
    templates: HashMap<IconCategory, DynamicImage>,
}

impl TemplateSet {
    fn load(template_dir: &Path) -> Result<Self> {
        let mut templates = HashMap::new();

        for category in IconCategory::all() {
            if category == IconCategory::Empty {
                continue; // 空白はテンプレートなし
            }

            let template_path = template_dir.join(category.template_name());
            let img = image::open(&template_path)
                .with_context(|| format!("テンプレート読み込み失敗: {}", template_path.display()))?;
            templates.insert(category, img);
        }

        Ok(Self { templates })
    }

    /// 画像との類似度を計算（正規化相互相関）
    fn match_score(&self, category: &IconCategory, target: &DynamicImage) -> f32 {
        if *category == IconCategory::Empty {
            // 空白の場合は、他のすべてのテンプレートとの類似度が低いかチェック
            let max_score = self
                .templates
                .values()
                .map(|template| Self::calculate_ncc(template, target))
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            // 空白度 = 1.0 - 最大類似度
            return 1.0 - max_score;
        }

        if let Some(template) = self.templates.get(category) {
            Self::calculate_ncc(template, target)
        } else {
            0.0
        }
    }

    /// 正規化相互相関（NCC）を計算
    fn calculate_ncc(template: &DynamicImage, target: &DynamicImage) -> f32 {
        let template_rgb = template.to_rgb8();
        let target_rgb = target.to_rgb8();

        if template_rgb.dimensions() != target_rgb.dimensions() {
            return 0.0;
        }

        let (width, height) = template_rgb.dimensions();
        let mut sum_t = 0.0f64;
        let mut sum_i = 0.0f64;
        let mut sum_ti = 0.0f64;
        let mut sum_tt = 0.0f64;
        let mut sum_ii = 0.0f64;
        let count = (width * height * 3) as f64;

        for y in 0..height {
            for x in 0..width {
                let t_pixel = template_rgb.get_pixel(x, y);
                let i_pixel = target_rgb.get_pixel(x, y);

                for i in 0..3 {
                    let t = t_pixel[i] as f64;
                    let i_val = i_pixel[i] as f64;

                    sum_t += t;
                    sum_i += i_val;
                    sum_ti += t * i_val;
                    sum_tt += t * t;
                    sum_ii += i_val * i_val;
                }
            }
        }

        let mean_t = sum_t / count;
        let mean_i = sum_i / count;

        let numerator = sum_ti - count * mean_t * mean_i;
        let denominator = ((sum_tt - count * mean_t * mean_t)
            * (sum_ii - count * mean_i * mean_i))
            .sqrt();

        if denominator < 1e-10 {
            return 0.0;
        }

        (numerator / denominator).max(0.0).min(1.0) as f32
    }

    /// 最も類似度の高いカテゴリを取得
    fn classify(&self, target: &DynamicImage, empty_threshold: f32) -> (IconCategory, f32) {
        let mut best_category = IconCategory::Empty;
        let mut best_score = 0.0f32;

        for category in IconCategory::all() {
            let score = self.match_score(&category, target);
            if score > best_score {
                best_score = score;
                best_category = category;
            }
        }

        // 空白判定の閾値チェック
        if best_category != IconCategory::Empty && best_score < empty_threshold {
            best_category = IconCategory::Empty;
            best_score = 1.0 - best_score;
        }

        (best_category, best_score)
    }
}

/// セル画像を収集
fn collect_cell_images(input_cells_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut cells = Vec::new();

    println!("セル画像を収集中...");

    for entry in walkdir::WalkDir::new(input_cells_dir)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file()
            && path.extension().and_then(|s| s.to_str()) == Some("png")
            && path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.contains("_input.png"))
                .unwrap_or(false)
        {
            cells.push(path.to_path_buf());
        }
    }

    println!("  収集完了: {}個のセル画像", cells.len());
    Ok(cells)
}

/// トレーニングデータの収集と分類
fn collect_training_data(
    template_dir: &Path,
    input_cells_dir: &Path,
    output_dir: &Path,
    samples_per_category: usize,
    empty_threshold: f32,
) -> Result<()> {
    println!("=== トレーニングデータ収集ツール ===\n");

    // テンプレートを読み込み
    println!("ステップ1: テンプレート読み込み");
    let templates = TemplateSet::load(template_dir)?;
    println!("  読み込み完了: {}個のテンプレート\n", templates.templates.len());

    // セル画像を収集
    println!("ステップ2: セル画像収集");
    let cell_paths = collect_cell_images(input_cells_dir)?;
    println!();

    // 各カテゴリごとに候補を分類
    println!("ステップ3: 画像分類");
    let mut category_candidates: HashMap<IconCategory, Vec<(PathBuf, f32)>> = HashMap::new();
    for category in IconCategory::all() {
        category_candidates.insert(category, Vec::new());
    }

    let total = cell_paths.len();
    for (i, path) in cell_paths.iter().enumerate() {
        if (i + 1) % 1000 == 0 || i + 1 == total {
            print!("\r  進捗: {}/{} ({:.1}%)", i + 1, total, (i + 1) as f64 / total as f64 * 100.0);
        }

        if let Ok(img) = image::open(path) {
            let (category, score) = templates.classify(&img, empty_threshold);
            if let Some(candidates) = category_candidates.get_mut(&category) {
                candidates.push((path.clone(), score));
            }
        }
    }
    println!("\n");

    // 各カテゴリの統計を表示
    println!("ステップ4: 分類結果");
    for category in IconCategory::all() {
        if let Some(candidates) = category_candidates.get(&category) {
            println!(
                "  {:12} : {:5}個の候補",
                category.folder_name(),
                candidates.len()
            );
        }
    }
    println!();

    // ランダムサンプリングして出力
    println!("ステップ5: ランダムサンプリングと保存");
    let mut rng = thread_rng();

    for category in IconCategory::all() {
        let folder_name = category.folder_name();
        let category_dir = output_dir.join(folder_name);
        fs::create_dir_all(&category_dir)?;

        if let Some(candidates) = category_candidates.get_mut(&category) {
            // 類似度でソート（降順）
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // 上位の候補からランダムサンプリング
            let top_candidates = if candidates.len() > samples_per_category * 3 {
                &mut candidates[..samples_per_category * 3]
            } else {
                candidates.as_mut_slice()
            };

            top_candidates.shuffle(&mut rng);

            let sample_count = top_candidates.len().min(samples_per_category);
            let samples = &top_candidates[..sample_count];

            println!(
                "  {:12} : {}個をサンプリング",
                folder_name, sample_count
            );

            for (idx, (path, score)) in samples.iter().enumerate() {
                let img = image::open(path)?;
                let output_path = category_dir.join(format!("sample_{:04}_{:.3}.png", idx, score));
                img.save(&output_path)?;
            }
        }
    }
    println!();

    // labels.txtを生成
    println!("ステップ6: ラベルファイル生成");
    let labels_path = output_dir.join("labels.txt");
    let mut labels = Vec::new();
    for category in IconCategory::all() {
        labels.push(format!(
            "{}: {}",
            category.label(),
            category.folder_name()
        ));
    }
    fs::write(&labels_path, labels.join("\n"))?;
    println!("  保存: {}", labels_path.display());
    println!();

    println!("✓ 完了!");
    println!("  出力先: {}", output_dir.display());

    Ok(())
}

fn print_usage() {
    println!("=== トレーニングデータ収集ツール ===");
    println!();
    println!("使用方法:");
    println!("  collect_training_data <テンプレートディレクトリ> <入力セルディレクトリ> <出力ディレクトリ> [サンプル数] [空白閾値]");
    println!();
    println!("引数:");
    println!("  <テンプレートディレクトリ> : input_icon_samples ディレクトリ");
    println!("  <入力セルディレクトリ>     : input_cells ディレクトリ");
    println!("  <出力ディレクトリ>         : トレーニングデータの出力先");
    println!("  [サンプル数]               : 各カテゴリから収集する数（デフォルト: 100）");
    println!("  [空白閾値]                 : 空白判定の閾値（デフォルト: 0.5）");
    println!();
    println!("例:");
    println!("  # 各カテゴリ100枚ずつ収集");
    println!("  collect_training_data input_icon_samples input_cells training_data 100 0.5");
    println!();
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        print_usage();
        anyhow::bail!("引数が不足しています");
    }

    let template_dir = PathBuf::from(&args[1]);
    let input_cells_dir = PathBuf::from(&args[2]);
    let output_dir = PathBuf::from(&args[3]);

    let samples_per_category = if args.len() >= 5 {
        args[4].parse::<usize>().context("サンプル数の解析に失敗")?
    } else {
        100
    };

    let empty_threshold = if args.len() >= 6 {
        args[5].parse::<f32>().context("閾値の解析に失敗")?
    } else {
        0.5
    };

    if !template_dir.exists() {
        anyhow::bail!(
            "テンプレートディレクトリが存在しません: {}",
            template_dir.display()
        );
    }

    if !input_cells_dir.exists() {
        anyhow::bail!(
            "入力セルディレクトリが存在しません: {}",
            input_cells_dir.display()
        );
    }

    collect_training_data(
        &template_dir,
        &input_cells_dir,
        &output_dir,
        samples_per_category,
        empty_threshold,
    )?;

    Ok(())
}
