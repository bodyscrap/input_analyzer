//! test_resultsをtraining_dataにマージするツール
//!
//! 目視確認・修正済みのtest_resultsデータをtraining_dataに追加します。
//!
//! # 使用方法
//! ```bash
//! cargo run --release --bin merge_test_results -- [オプション]
//! ```
//!
//! # オプション
//! - 第1引数: test_resultsディレクトリ（デフォルト: test_results）
//! - 第2引数: training_dataディレクトリ（デフォルト: training_data）
//! - 第3引数: バックアップを作成するか（デフォルト: true）

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const CLASS_NAMES: [&str; 14] = [
    "A1", "A2", "B", "Start", "W", "dir_1", "dir_2", "dir_3", "dir_4",
    "dir_6", "dir_7", "dir_8", "dir_9", "others",
];

/// コマンドライン引数
struct Args {
    test_results_dir: PathBuf,
    training_data_dir: PathBuf,
    create_backup: bool,
}

impl Args {
    fn parse() -> Self {
        let args: Vec<String> = env::args().collect();

        let test_results_dir = if args.len() >= 2 {
            PathBuf::from(&args[1])
        } else {
            PathBuf::from("test_results")
        };

        let training_data_dir = if args.len() >= 3 {
            PathBuf::from(&args[2])
        } else {
            PathBuf::from("training_data")
        };

        let create_backup = if args.len() >= 4 {
            args[3].to_lowercase() != "false" && args[3] != "0"
        } else {
            true
        };

        Self {
            test_results_dir,
            training_data_dir,
            create_backup,
        }
    }
}

/// training_dataのバックアップを作成
fn create_backup(training_data_dir: &Path) -> Result<PathBuf> {
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let backup_dir = PathBuf::from(format!("training_data_backup_{}", timestamp));

    println!("バックアップを作成中: {}", backup_dir.display());

    // ディレクトリ構造をコピー
    copy_dir_all(training_data_dir, &backup_dir)?;

    println!("✓ バックアップ完了: {}\n", backup_dir.display());

    Ok(backup_dir)
}

/// ディレクトリを再帰的にコピー
fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if src_path.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }

    Ok(())
}

/// test_resultsからサンプルを収集
fn collect_test_results(test_results_dir: &Path) -> Result<HashMap<String, Vec<PathBuf>>> {
    let mut class_samples: HashMap<String, Vec<PathBuf>> = HashMap::new();

    println!("=== test_resultsからサンプルを収集中 ===");

    // test_results配下の各動画ディレクトリを走査
    for video_entry in fs::read_dir(test_results_dir)
        .with_context(|| format!("ディレクトリが開けません: {}", test_results_dir.display()))?
    {
        let video_entry = video_entry?;
        let video_dir = video_entry.path();

        if !video_dir.is_dir() {
            continue;
        }

        let video_name = video_dir.file_name().unwrap().to_str().unwrap();
        println!("\n動画: {}", video_name);

        // 各クラスディレクトリを走査
        for class_name in CLASS_NAMES.iter() {
            let class_dir = video_dir.join(class_name);

            if !class_dir.exists() {
                continue;
            }

            let mut count = 0;
            for entry in fs::read_dir(&class_dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("png") {
                    class_samples
                        .entry(class_name.to_string())
                        .or_insert_with(Vec::new)
                        .push(path);
                    count += 1;
                }
            }

            if count > 0 {
                println!("  {:12}: {:4} 枚", class_name, count);
            }
        }
    }

    // 総計を表示
    println!("\n--- 収集サマリー ---");
    let mut total = 0;
    let mut class_list: Vec<_> = class_samples.iter().collect();
    class_list.sort_by_key(|(name, _)| *name);

    for (class_name, samples) in class_list {
        println!("  {:12}: {:4} 枚", class_name, samples.len());
        total += samples.len();
    }
    println!("  総計: {} 枚\n", total);

    Ok(class_samples)
}

/// training_dataの現在のサンプル数をカウント
fn count_training_samples(training_data_dir: &Path) -> Result<HashMap<String, usize>> {
    let mut counts = HashMap::new();

    for class_name in CLASS_NAMES.iter() {
        let class_dir = training_data_dir.join(class_name);
        let count = if class_dir.exists() {
            fs::read_dir(&class_dir)?
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

/// test_resultsのサンプルをtraining_dataにコピー
fn merge_samples(
    class_samples: &HashMap<String, Vec<PathBuf>>,
    training_data_dir: &Path,
) -> Result<HashMap<String, usize>> {
    let mut copied_counts = HashMap::new();

    println!("=== training_dataにマージ中 ===\n");

    for (class_name, samples) in class_samples.iter() {
        let class_dir = training_data_dir.join(class_name);
        fs::create_dir_all(&class_dir)?;

        let mut copied = 0;
        for (idx, src_path) in samples.iter().enumerate() {
            // 新しいファイル名を生成
            let new_filename = format!("test_result_{:04}.png", idx);
            let mut dst_path = class_dir.join(&new_filename);

            // 既に存在する場合は番号を増やす
            let mut counter = 0;
            while dst_path.exists() {
                counter += 1;
                let new_filename = format!("test_result_{:04}_{}.png", idx, counter);
                dst_path = class_dir.join(&new_filename);
            }

            // ファイルをコピー
            fs::copy(src_path, &dst_path)
                .with_context(|| format!("コピー失敗: {} -> {}", src_path.display(), dst_path.display()))?;
            copied += 1;
        }

        copied_counts.insert(class_name.clone(), copied);
        println!("  {:12}: {:4} 枚をコピー", class_name, copied);
    }

    println!("\n✓ マージ完了\n");

    Ok(copied_counts)
}

fn main() -> Result<()> {
    println!("=== test_results → training_data 新規作成ツール ===\n");

    let args = Args::parse();

    println!("設定:");
    println!("  test_results: {}", args.test_results_dir.display());
    println!("  training_data: {}", args.training_data_dir.display());
    println!("  バックアップ作成: {}\n", args.create_backup);

    // test_resultsの存在確認
    if !args.test_results_dir.exists() {
        anyhow::bail!(
            "test_resultsディレクトリが見つかりません: {}",
            args.test_results_dir.display()
        );
    }

    // 既存のtraining_dataをバックアップ
    if args.training_data_dir.exists() {
        if args.create_backup {
            println!("既存のtraining_dataをバックアップ中...");
            create_backup(&args.training_data_dir)?;
        }
        
        // training_dataを削除
        println!("既存のtraining_dataを削除中...");
        fs::remove_dir_all(&args.training_data_dir)?;
        println!("✓ 削除完了\n");
    }

    // 新しいtraining_dataディレクトリを作成
    fs::create_dir_all(&args.training_data_dir)?;
    
    // 各クラスディレクトリを作成
    for class_name in CLASS_NAMES.iter() {
        let class_dir = args.training_data_dir.join(class_name);
        fs::create_dir_all(&class_dir)?;
    }

    // test_resultsからサンプルを収集
    let class_samples = collect_test_results(&args.test_results_dir)?;

    if class_samples.is_empty() {
        println!("追加するサンプルがありません。");
        return Ok(());
    }

    // training_dataにコピー
    let copied_counts = merge_samples(&class_samples, &args.training_data_dir)?;

    // labels.txtを作成
    let labels_path = args.training_data_dir.join("labels.txt");
    let labels_content = CLASS_NAMES.join("\n") + "\n";
    fs::write(&labels_path, labels_content)?;
    println!("✓ labels.txt を作成\n");

    // 最終的なサンプル数を確認
    println!("=== 新規training_data ===");
    let final_counts = count_training_samples(&args.training_data_dir)?;
    let mut final_list: Vec<_> = final_counts.iter().collect();
    final_list.sort_by_key(|(name, _)| *name);
    let final_total: usize = final_counts.values().sum();

    for (class_name, count) in final_list {
        println!("  {:12}: {:4} 枚", class_name, count);
    }
    println!("  総計: {} 枚\n", final_total);

    println!("=== 完了 ===");
    println!("次のステップ:");
    println!("  1. training_dataの内容を確認:");
    println!("     cargo run --release --bin inspect_training_data");
    println!("  2. モデルを再トレーニング:");
    println!("     cargo run --release --features ml --bin train_model");

    Ok(())
}
