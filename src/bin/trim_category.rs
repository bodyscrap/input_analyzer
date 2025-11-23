use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// サンプルファイルとスコアの情報
#[derive(Debug, Clone)]
struct SampleInfo {
    path: PathBuf,
    score: f32,
}

/// ファイル名からスコアを抽出
fn extract_score(filename: &str) -> Option<f32> {
    // ファイル名の形式: sample_XXXX_Y.YYY.png
    let name = filename.strip_suffix(".png")?;
    let parts: Vec<&str> = name.split('_').collect();

    if parts.len() >= 3 {
        // 最後の部分がスコア
        parts.last()?.parse::<f32>().ok()
    } else {
        None
    }
}

/// カテゴリ内のサンプルを収集
fn collect_samples(category_dir: &Path) -> Result<Vec<SampleInfo>> {
    let mut samples = Vec::new();

    for entry in fs::read_dir(category_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("png") {
            if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                if let Some(score) = extract_score(filename) {
                    samples.push(SampleInfo {
                        path: path.clone(),
                        score,
                    });
                }
            }
        }
    }

    Ok(samples)
}

/// サンプルを高スコア順に並べ替えて上位のみ保持
fn trim_category(category_dir: &Path, target_count: usize, dry_run: bool) -> Result<()> {
    let category_name = category_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    println!("=== カテゴリ: {} ===", category_name);

    // サンプルを収集
    println!("サンプルを収集中...");
    let mut samples = collect_samples(category_dir)?;

    if samples.is_empty() {
        println!("  サンプルが見つかりませんでした");
        return Ok(());
    }

    println!("  現在のサンプル数: {}", samples.len());

    if samples.len() <= target_count {
        println!("  目標数({})以下です。削減の必要はありません。", target_count);
        return Ok(());
    }

    // スコアで降順ソート
    samples.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    // 統計情報
    let avg_score_all = samples.iter().map(|s| s.score).sum::<f32>() / samples.len() as f32;
    let avg_score_top = samples[..target_count].iter().map(|s| s.score).sum::<f32>() / target_count as f32;
    let min_score_kept = samples[target_count - 1].score;
    let max_score_removed = samples[target_count].score;

    println!("\n統計情報:");
    println!("  全体の平均スコア: {:.3}", avg_score_all);
    println!("  保持分の平均スコア: {:.3}", avg_score_top);
    println!("  保持する最低スコア: {:.3}", min_score_kept);
    println!("  削除する最高スコア: {:.3}", max_score_removed);
    println!("  削除予定: {}個", samples.len() - target_count);

    if dry_run {
        println!("\n[ドライラン] 実際の削除は行いません");
        println!("\n削除予定のファイル（最初の10個）:");
        for sample in samples.iter().skip(target_count).take(10) {
            println!("  {} (スコア: {:.3})",
                sample.path.file_name().unwrap().to_str().unwrap(),
                sample.score);
        }
        if samples.len() - target_count > 10 {
            println!("  ... 他 {}個", samples.len() - target_count - 10);
        }
    } else {
        // 実際に削除
        println!("\nファイルを削除中...");
        let mut deleted = 0;
        for sample in samples.iter().skip(target_count) {
            if let Err(e) = fs::remove_file(&sample.path) {
                eprintln!("  警告: 削除失敗: {} - {}", sample.path.display(), e);
            } else {
                deleted += 1;
            }
        }
        println!("  削除完了: {}個", deleted);
        println!("\n✓ 完了!");
        println!("  残りのサンプル数: {}", target_count);
    }

    Ok(())
}

fn print_usage() {
    println!("=== カテゴリサンプル削減ツール ===");
    println!();
    println!("使用方法:");
    println!("  trim_category <カテゴリディレクトリ> <目標サンプル数> [--dry-run]");
    println!();
    println!("引数:");
    println!("  <カテゴリディレクトリ> : 削減対象のカテゴリフォルダ（例: training_data/empty）");
    println!("  <目標サンプル数>       : 残すサンプル数");
    println!("  --dry-run              : 実際の削除を行わず、削減予定を表示");
    println!();
    println!("説明:");
    println!("  カテゴリ内のサンプルをスコア（ファイル名から抽出）で並べ替え、");
    println!("  高スコアの上位N個のみを残し、残りを削除します。");
    println!();
    println!("例:");
    println!("  # emptyカテゴリを100個に削減（ドライラン）");
    println!("  trim_category training_data/empty 100 --dry-run");
    println!();
    println!("  # 実際に削減");
    println!("  trim_category training_data/empty 100");
    println!();
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        print_usage();
        anyhow::bail!("引数が不足しています");
    }

    let category_dir = PathBuf::from(&args[1]);
    let target_count = args[2]
        .parse::<usize>()
        .context("目標サンプル数の解析に失敗")?;

    let dry_run = args.len() >= 4 && args[3] == "--dry-run";

    if !category_dir.exists() {
        anyhow::bail!(
            "カテゴリディレクトリが存在しません: {}",
            category_dir.display()
        );
    }

    if !category_dir.is_dir() {
        anyhow::bail!(
            "指定されたパスはディレクトリではありません: {}",
            category_dir.display()
        );
    }

    trim_category(&category_dir, target_count, dry_run)?;

    Ok(())
}
