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

/// 既存のトレーニングデータをカウント
fn count_existing_samples(training_dir: &Path) -> Result<HashMap<IconCategory, usize>> {
    let mut counts = HashMap::new();

    for category in IconCategory::all() {
        let category_dir = training_dir.join(category.folder_name());
        let count = if category_dir.exists() {
            fs::read_dir(&category_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path().extension().and_then(|s| s.to_str()) == Some("png")
                })
                .count()
        } else {
            0
        };
        counts.insert(category, count);
    }

    Ok(counts)
}

/// セル画像を収集
fn collect_cell_images(input_cells_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut cells = Vec::new();

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

    Ok(cells)
}

/// トレーニングデータの補充
fn supplement_training_data(
    template_dir: &Path,
    input_cells_dir: &Path,
    training_dir: &Path,
    target_samples: usize,
    empty_threshold: f32,
) -> Result<()> {
    println!("=== トレーニングデータ補充ツール ===\n");

    // 既存のサンプル数をカウント
    println!("ステップ1: 既存トレーニングデータの確認");
    let existing_counts = count_existing_samples(training_dir)?;

    println!("\n現在のサンプル数:");
    let mut categories_to_supplement = Vec::new();
    for category in IconCategory::all() {
        let count = existing_counts.get(&category).unwrap_or(&0);
        let status = if *count < target_samples {
            categories_to_supplement.push(category.clone());
            format!("→ {}枚追加が必要", target_samples - count)
        } else if *count > target_samples {
            format!("({}枚超過)", count - target_samples)
        } else {
            "(目標達成)".to_string()
        };
        println!("  {:12} : {:3}枚 {}", category.folder_name(), count, status);
    }
    println!();

    if categories_to_supplement.is_empty() {
        println!("✓ すべてのカテゴリが目標サンプル数に達しています");
        return Ok(());
    }

    println!("補充対象: {}カテゴリ\n", categories_to_supplement.len());

    // テンプレートを読み込み
    println!("ステップ2: テンプレート読み込み");
    let templates = TemplateSet::load(template_dir)?;
    println!("  読み込み完了: {}個のテンプレート\n", templates.templates.len());

    // セル画像を収集
    println!("ステップ3: セル画像収集");
    let cell_paths = collect_cell_images(input_cells_dir)?;
    println!("  収集完了: {}個のセル画像\n", cell_paths.len());

    // 補充が必要なカテゴリのみ分類
    println!("ステップ4: 画像分類（補充対象のみ）");
    let mut category_candidates: HashMap<IconCategory, Vec<(PathBuf, f32)>> = HashMap::new();
    for category in &categories_to_supplement {
        category_candidates.insert(category.clone(), Vec::new());
    }

    let total = cell_paths.len();
    for (i, path) in cell_paths.iter().enumerate() {
        if (i + 1) % 1000 == 0 || i + 1 == total {
            print!("\r  進捗: {}/{} ({:.1}%)", i + 1, total, (i + 1) as f64 / total as f64 * 100.0);
        }

        if let Ok(img) = image::open(path) {
            let (category, score) = templates.classify(&img, empty_threshold);
            if categories_to_supplement.contains(&category) {
                if let Some(candidates) = category_candidates.get_mut(&category) {
                    candidates.push((path.clone(), score));
                }
            }
        }
    }
    println!("\n");

    // 各カテゴリの候補数を表示
    println!("ステップ5: 候補数確認");
    for category in &categories_to_supplement {
        if let Some(candidates) = category_candidates.get(category) {
            println!(
                "  {:12} : {:5}個の候補",
                category.folder_name(),
                candidates.len()
            );
        }
    }
    println!();

    // 補充サンプリングして追加
    println!("ステップ6: 補充サンプリングと保存");
    let mut rng = thread_rng();
    let mut total_added = 0;

    for category in &categories_to_supplement {
        let folder_name = category.folder_name();
        let category_dir = training_dir.join(folder_name);
        fs::create_dir_all(&category_dir)?;

        let existing_count = existing_counts.get(category).unwrap_or(&0);
        let needed = target_samples.saturating_sub(*existing_count);

        if needed == 0 {
            continue;
        }

        if let Some(candidates) = category_candidates.get_mut(category) {
            // 類似度でソート（降順）
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // 上位の候補からランダムサンプリング
            let top_candidates = if candidates.len() > needed * 3 {
                &mut candidates[..needed * 3]
            } else {
                candidates.as_mut_slice()
            };

            top_candidates.shuffle(&mut rng);

            let sample_count = top_candidates.len().min(needed);
            let samples = &top_candidates[..sample_count];

            println!(
                "  {:12} : {}枚追加 (既存{}枚 → 合計{}枚)",
                folder_name,
                sample_count,
                existing_count,
                existing_count + sample_count
            );

            // 既存の最大インデックスを取得
            let existing_files: Vec<_> = fs::read_dir(&category_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path().extension().and_then(|s| s.to_str()) == Some("png")
                })
                .collect();

            let mut max_idx = 0;
            for entry in existing_files {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(idx_str) = name.strip_prefix("sample_").and_then(|s| s.split('_').next()) {
                        if let Ok(idx) = idx_str.parse::<usize>() {
                            max_idx = max_idx.max(idx);
                        }
                    }
                }
            }

            // 新しいサンプルを追加
            for (i, (path, score)) in samples.iter().enumerate() {
                let img = image::open(path)?;
                let new_idx = max_idx + i + 1;
                let output_path = category_dir.join(format!("sample_{:04}_{:.3}.png", new_idx, score));
                img.save(&output_path)?;
            }

            total_added += sample_count;
        }
    }
    println!();

    println!("✓ 完了!");
    println!("  補充カテゴリ数: {}", categories_to_supplement.len());
    println!("  追加サンプル数: {}", total_added);
    println!("  トレーニングデータ: {}", training_dir.display());

    // 最終統計を表示
    println!("\n最終サンプル数:");
    let final_counts = count_existing_samples(training_dir)?;
    for category in IconCategory::all() {
        let count = final_counts.get(&category).unwrap_or(&0);
        println!("  {:12} : {:3}枚", category.folder_name(), count);
    }

    Ok(())
}

fn print_usage() {
    println!("=== トレーニングデータ補充ツール ===");
    println!();
    println!("使用方法:");
    println!("  supplement_training_data <テンプレートディレクトリ> <入力セルディレクトリ> <トレーニングデータディレクトリ> [目標サンプル数] [空白閾値]");
    println!();
    println!("引数:");
    println!("  <テンプレートディレクトリ>       : input_icon_samples ディレクトリ");
    println!("  <入力セルディレクトリ>           : input_cells ディレクトリ");
    println!("  <トレーニングデータディレクトリ> : training_data ディレクトリ");
    println!("  [目標サンプル数]                 : 各カテゴリの目標数（デフォルト: 100）");
    println!("  [空白閾値]                       : 空白判定の閾値（デフォルト: 0.5）");
    println!();
    println!("説明:");
    println!("  既存のtraining_dataを確認し、目標数に達していないカテゴリのみを補充します。");
    println!("  既存のサンプルは保持され、新しいサンプルが追加されます。");
    println!();
    println!("例:");
    println!("  # 各カテゴリを100枚に補充");
    println!("  supplement_training_data input_icon_samples input_cells training_data 100 0.5");
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
    let training_dir = PathBuf::from(&args[3]);

    let target_samples = if args.len() >= 5 {
        args[4].parse::<usize>().context("目標サンプル数の解析に失敗")?
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

    if !training_dir.exists() {
        anyhow::bail!(
            "トレーニングデータディレクトリが存在しません: {}",
            training_dir.display()
        );
    }

    supplement_training_data(
        &template_dir,
        &input_cells_dir,
        &training_dir,
        target_samples,
        empty_threshold,
    )?;

    Ok(())
}
