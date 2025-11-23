//! 学習済みBurnモデルを使用してtraining_dataを更新するツール
//!
//! input_cells_allから高信頼度のサンプルを自動収集し、
//! 既存のtraining_dataを補充・更新します。

#[cfg(feature = "ml")]
use burn::{
    module::Module,
    record::CompactRecorder,
    tensor::Tensor,
};

#[cfg(feature = "ml")]
use input_analyzer::ml_model::{ModelConfig, CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES, load_and_normalize_image};

#[cfg(feature = "ml")]
use rand::seq::SliceRandom;
#[cfg(feature = "ml")]
use std::collections::HashMap;
#[cfg(feature = "ml")]
use std::path::{Path, PathBuf};

// バックエンド設定（train_model.rsと同じ）
#[cfg(feature = "ml")]
type MyBackend = burn_wgpu::Wgpu;



/// input_cellsから画像パスを収集
#[cfg(feature = "ml")]
fn collect_cell_images(input_cells_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut cells = Vec::new();

    println!("\n=== セル画像を収集中: {} ===", input_cells_dir.display());

    for entry in std::fs::read_dir(input_cells_dir)? {
        let entry = entry?;
        let sample_dir = entry.path();

        if !sample_dir.is_dir() {
            continue;
        }

        let sample_name = sample_dir.file_name().unwrap().to_str().unwrap();
        let mut count = 0;

        for cell_entry in walkdir::WalkDir::new(&sample_dir)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = cell_entry.path();
            if path.is_file()
                && path.extension().and_then(|s| s.to_str()) == Some("png")
                && path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.contains("_input.png"))
                    .unwrap_or(false)
            {
                cells.push(path.to_path_buf());
                count += 1;
            }
        }

        println!("  {}: {} 枚", sample_name, count);
    }

    println!("\n総セル画像数: {}", cells.len());
    Ok(cells)
}

/// 既存のtraining_dataをカウント
#[cfg(feature = "ml")]
fn count_existing_samples(training_dir: &Path) -> anyhow::Result<HashMap<String, usize>> {
    let mut counts = HashMap::new();

    for class_name in CLASS_NAMES.iter() {
        let class_dir = training_dir.join(class_name);
        let count = if class_dir.exists() {
            std::fs::read_dir(&class_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("png"))
                .count()
        } else {
            0
        };
        counts.insert(class_name.to_string(), count);
    }

    Ok(counts)
}

/// モデルを使ってデータ収集
#[cfg(feature = "ml")]
fn collect_with_model(
    model_path: &Path,
    input_cells_dir: &Path,
    training_dir: &Path,
    target_per_class: usize,
    confidence_threshold: f32,
    batch_size: usize,
    max_samples: Option<usize>,
) -> anyhow::Result<()> {
    println!("================================================================================");
    println!("モデルベースのトレーニングデータ更新");
    println!("================================================================================");

    // デバイス設定（WGPU/GPU）
    let device = burn_wgpu::WgpuDevice::default();
    println!("\n使用デバイス: {:?}", device);

    // モデルロード
    println!("\n=== モデル読み込み ===");
    let model_config = ModelConfig::new(NUM_CLASSES);
    let model = model_config.init::<MyBackend>(&device);

    let model = model
        .load_file(model_path, &CompactRecorder::new(), &device)
        .map_err(|e| anyhow::anyhow!("モデルの読み込みに失敗: {}", e))?;

    println!("モデル読み込み完了: {}", model_path.display());
    println!("信頼度閾値: {}", confidence_threshold);

    // 既存データ確認
    println!("\n=== 既存トレーニングデータ確認 ===");
    let existing_counts = count_existing_samples(training_dir)?;

    println!("\n現在のサンプル数:");
    for class_name in CLASS_NAMES.iter() {
        let count = existing_counts.get(*class_name).unwrap_or(&0);
        let needed = target_per_class.saturating_sub(*count);
        if needed > 0 {
            println!("  {}: {:3}枚 → {}枚追加が必要", class_name, count, needed);
        } else {
            println!("  {}: {:3}枚 (目標達成)", class_name, count);
        }
    }

    // セル画像収集
    let mut cell_images = collect_cell_images(input_cells_dir)?;

    // サンプリング（指定された場合）
    if let Some(max) = max_samples {
        if cell_images.len() > max {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            cell_images.shuffle(&mut rng);
            cell_images.truncate(max);
            println!("\nランダムサンプリング: {} 枚に制限", max);
        }
    }

    // 分類実行
    println!("\n=== セル画像を分類中 ===");

    let mut candidates: HashMap<String, Vec<(PathBuf, f32)>> = HashMap::new();
    for class_name in CLASS_NAMES.iter() {
        candidates.insert(class_name.to_string(), Vec::new());
    }

    let total_batches = (cell_images.len() + batch_size - 1) / batch_size;
    let mut processed = 0;

    for (batch_idx, chunk) in cell_images.chunks(batch_size).enumerate() {
        // バッチ画像をロード
        let mut batch_data = Vec::with_capacity(chunk.len() * 3 * IMAGE_SIZE * IMAGE_SIZE);
        let mut valid_paths = Vec::new();

        for path in chunk {
            match load_and_normalize_image(path) {
                Ok(image_data) => {
                    batch_data.extend(image_data);
                    valid_paths.push(path.clone());
                }
                Err(_) => {
                    // エラーの場合はスキップ
                    continue;
                }
            }
        }

        if valid_paths.is_empty() {
            continue;
        }

        let actual_batch_size = valid_paths.len();

        // Tensorに変換
        let images = Tensor::<MyBackend, 1>::from_floats(batch_data.as_slice(), &device)
            .reshape([actual_batch_size, 3, IMAGE_SIZE, IMAGE_SIZE]);

        // 推論
        let (predictions, logits) = model.predict(images);

        // Softmax計算して信頼度を取得
        let probs = burn::tensor::activation::softmax(logits, 1);

        // 結果を収集
        for i in 0..actual_batch_size {
            let pred_class_idx = predictions.clone().slice([i..i+1]).into_scalar() as usize;
            let confidence = probs.clone().slice([i..i+1, pred_class_idx..pred_class_idx+1]).into_scalar();

            if confidence >= confidence_threshold {
                let class_name = CLASS_NAMES[pred_class_idx];
                candidates
                    .get_mut(class_name)
                    .unwrap()
                    .push((valid_paths[i].clone(), confidence));
            }
        }

        processed += chunk.len();
        if (batch_idx + 1) % 100 == 0 || batch_idx + 1 == total_batches {
            print!(
                "\r  進捗: {}/{} バッチ ({:.1}%)",
                batch_idx + 1,
                total_batches,
                100.0 * processed as f32 / cell_images.len() as f32
            );
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }
    println!();

    // 統計表示
    println!("\n=== 候補数 ===");
    for class_name in CLASS_NAMES.iter() {
        let count = candidates.get(*class_name).map(|v| v.len()).unwrap_or(0);
        let existing = existing_counts.get(*class_name).unwrap_or(&0);
        let needed = target_per_class.saturating_sub(*existing);
        let status = if count >= needed { "✓" } else { "⚠" };
        println!(
            "  {} {}: {} 候補 (既存: {}, 必要: {})",
            status, class_name, count, existing, needed
        );
    }

    // サンプリングと保存
    println!("\n=== サンプリングと保存 ===");
    let mut total_added = 0;
    let mut rng = rand::thread_rng();

    for class_name in CLASS_NAMES.iter() {
        let class_dir = training_dir.join(class_name);
        std::fs::create_dir_all(&class_dir)?;

        let existing = existing_counts.get(*class_name).unwrap_or(&0);
        let needed = target_per_class.saturating_sub(*existing);

        if needed == 0 {
            println!("  {}: スキップ（既に目標達成）", class_name);
            continue;
        }

        let mut class_candidates = candidates.get_mut(*class_name).unwrap().clone();

        if class_candidates.is_empty() {
            println!("  {}: 候補なし", class_name);
            continue;
        }

        // 信頼度でソート（降順）
        class_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // 上位からランダムサンプリング
        let sample_pool_size = (class_candidates.len()).min(needed * 3);
        let mut sample_pool = class_candidates[..sample_pool_size].to_vec();
        sample_pool.shuffle(&mut rng);

        let actual_count = sample_pool.len().min(needed);
        let samples = &sample_pool[..actual_count];

        // 既存の最大インデックスを取得
        let existing_files: Vec<_> = std::fs::read_dir(&class_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("png"))
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

        // 保存
        for (i, (src_path, conf)) in samples.iter().enumerate() {
            let idx = max_idx + i + 1;
            let dst_name = format!("sample_{:04}_{:.3}.png", idx, conf);
            let dst_path = class_dir.join(dst_name);
            std::fs::copy(src_path, dst_path)?;
        }

        total_added += actual_count;
        let avg_conf = samples.iter().map(|(_, c)| c).sum::<f32>() / samples.len() as f32;

        println!(
            "  {}: {}枚追加 (平均信頼度: {:.3})",
            class_name, actual_count, avg_conf
        );
    }

    // 最終統計
    println!("\n=== 最終統計 ===");
    let final_counts = count_existing_samples(training_dir)?;

    println!("\nクラス別サンプル数:");
    let mut total_samples = 0;
    for class_name in CLASS_NAMES.iter() {
        let count = final_counts.get(*class_name).unwrap_or(&0);
        total_samples += count;
        let status = if *count >= target_per_class { "✓" } else { "⚠" };
        println!("  {} {}: {}枚", status, class_name, count);
    }

    println!("\n総サンプル数: {}", total_samples);
    println!("新規追加: {}", total_added);

    println!("\n✓ 完了!");
    println!("  出力先: {}", training_dir.display());

    Ok(())
}

#[cfg(feature = "ml")]
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 1 || (args.len() >= 2 && args[1] == "--help") {
        eprintln!("================================================================================");
        eprintln!("モデルベースのトレーニングデータ更新");
        eprintln!("================================================================================");
        eprintln!();
        eprintln!("使用方法:");
        eprintln!("  collect_with_model [オプション]");
        eprintln!();
        eprintln!("オプション:");
        eprintln!("  --model <パス>          モデルファイル (デフォルト: models/model)");
        eprintln!("  --input <パス>          入力セルディレクトリ (デフォルト: input_cells_all)");
        eprintln!("  --output <パス>         出力ディレクトリ (デフォルト: training_data)");
        eprintln!("  --target <数>           各クラスの目標サンプル数 (デフォルト: 100)");
        eprintln!("  --confidence <値>       信頼度閾値 (デフォルト: 0.95)");
        eprintln!("  --batch-size <数>       バッチサイズ (デフォルト: 32)");
        eprintln!("  --max-samples <数>      処理する最大セル数 (デフォルト: 全て)");
        eprintln!();
        eprintln!("例:");
        eprintln!("  # デフォルト設定で実行");
        eprintln!("  cargo run --bin collect_with_model --features ml --release");
        eprintln!();
        eprintln!("  # 各クラス150枚、信頼度0.90以上で収集");
        eprintln!("  cargo run --bin collect_with_model --features ml --release -- \\");
        eprintln!("    --target 150 --confidence 0.90");
        eprintln!();
        eprintln!("  # 高速化: 10万枚からサンプリング");
        eprintln!("  cargo run --bin collect_with_model --features ml --release -- \\");
        eprintln!("    --max-samples 100000 --batch-size 128");
        eprintln!();
        return Ok(());
    }

    // 引数パース
    let mut model_path = PathBuf::from("models/model");
    let mut input_cells_dir = PathBuf::from("input_cells_all");
    let mut training_dir = PathBuf::from("training_data");
    let mut target_per_class = 100;
    let mut confidence_threshold = 0.95;
    let mut batch_size = 32;
    let mut max_samples: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                if i < args.len() {
                    model_path = PathBuf::from(&args[i]);
                }
            }
            "--input" => {
                i += 1;
                if i < args.len() {
                    input_cells_dir = PathBuf::from(&args[i]);
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    training_dir = PathBuf::from(&args[i]);
                }
            }
            "--target" => {
                i += 1;
                if i < args.len() {
                    target_per_class = args[i].parse()?;
                }
            }
            "--confidence" => {
                i += 1;
                if i < args.len() {
                    confidence_threshold = args[i].parse()?;
                }
            }
            "--batch-size" => {
                i += 1;
                if i < args.len() {
                    batch_size = args[i].parse()?;
                }
            }
            "--max-samples" => {
                i += 1;
                if i < args.len() {
                    max_samples = Some(args[i].parse()?);
                }
            }
            _ => {}
        }
        i += 1;
    }

    // ファイル存在確認（.mpk拡張子付きでチェック）
    let model_file = if model_path.extension().is_some() {
        model_path.clone()
    } else {
        model_path.with_extension("mpk")
    };

    if !model_file.exists() {
        anyhow::bail!("モデルファイルが存在しません: {} ({}も確認しました)",
                      model_path.display(), model_file.display());
    }

    if !input_cells_dir.exists() {
        anyhow::bail!("入力ディレクトリが存在しません: {}", input_cells_dir.display());
    }

    if !training_dir.exists() {
        anyhow::bail!("トレーニングデータディレクトリが存在しません: {}", training_dir.display());
    }

    // 実行
    collect_with_model(
        &model_path,
        &input_cells_dir,
        &training_dir,
        target_per_class,
        confidence_threshold,
        batch_size,
        max_samples,
    )?;

    Ok(())
}

#[cfg(not(feature = "ml"))]
fn main() {
    eprintln!("エラー: このプログラムはml機能を有効にしてビルドする必要があります。");
    eprintln!();
    eprintln!("ビルドコマンド:");
    eprintln!("  cargo run --bin collect_with_model --features ml --release");
    eprintln!();
    std::process::exit(1);
}
