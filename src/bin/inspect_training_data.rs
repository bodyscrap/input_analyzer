//! トレーニングデータ検証ツール
//!
//! 収集されたトレーニングデータをHTMLビューアーで確認し、
//! 品質チェックと誤分類の検出を支援します。

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// カテゴリ情報
#[derive(Debug, Clone)]
struct CategoryInfo {
    name: String,
    label: usize,
    samples: Vec<SampleInfo>,
}

/// サンプル情報
#[derive(Debug, Clone)]
struct SampleInfo {
    path: PathBuf,
    filename: String,
    score: Option<f32>,
}

impl SampleInfo {
    fn from_path(path: PathBuf) -> Self {
        let filename = path.file_name().unwrap().to_str().unwrap().to_string();

        // ファイル名から類似度スコアを抽出
        // 形式: sample_0000_0.850.png
        let score = filename
            .strip_prefix("sample_")
            .and_then(|s| s.rsplit_once('_'))
            .and_then(|(_, score_ext)| score_ext.strip_suffix(".png"))
            .and_then(|s| s.parse::<f32>().ok());

        Self {
            path,
            filename,
            score,
        }
    }
}

/// トレーニングデータを読み込み
fn load_training_data(data_dir: &Path) -> Result<Vec<CategoryInfo>> {
    println!("トレーニングデータを読み込み中...");

    // ラベルマッピングを読み込み
    let labels_path = data_dir.join("labels.txt");
    let labels_content = fs::read_to_string(&labels_path)
        .context("labels.txtの読み込みに失敗しました")?;

    let mut label_map = HashMap::new();
    for line in labels_content.lines() {
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() == 2 {
            let label_id = parts[0].trim().parse::<usize>()?;
            let category = parts[1].trim().to_string();
            label_map.insert(category.clone(), label_id);
        }
    }

    let mut categories = Vec::new();

    // 各カテゴリのサンプルを収集
    for entry in fs::read_dir(data_dir)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        let category_name = path.file_name().unwrap().to_str().unwrap().to_string();

        if let Some(&label) = label_map.get(&category_name) {
            let mut samples = Vec::new();

            for img_entry in fs::read_dir(&path)? {
                let img_entry = img_entry?;
                let img_path = img_entry.path();

                if img_path.extension().and_then(|s| s.to_str()) == Some("png") {
                    samples.push(SampleInfo::from_path(img_path));
                }
            }

            // スコアでソート（降順）
            samples.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let sample_count = samples.len();
            categories.push(CategoryInfo {
                name: category_name.clone(),
                label,
                samples,
            });

            println!("  {}: {}個のサンプル", category_name, sample_count);
        }
    }

    // ラベル順にソート
    categories.sort_by_key(|c| c.label);

    Ok(categories)
}

/// 統計情報を計算
fn calculate_statistics(categories: &[CategoryInfo]) {
    println!("\n=== 統計情報 ===");

    let total_samples: usize = categories.iter().map(|c| c.samples.len()).sum();
    println!("総サンプル数: {}", total_samples);
    println!("カテゴリ数: {}", categories.len());

    println!("\nカテゴリ別サンプル数:");
    for cat in categories {
        let avg_score = if cat.samples.is_empty() {
            0.0
        } else {
            cat.samples
                .iter()
                .filter_map(|s| s.score)
                .sum::<f32>()
                / cat.samples.len() as f32
        };

        let min_score = cat.samples
            .iter()
            .filter_map(|s| s.score)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let max_score = cat.samples
            .iter()
            .filter_map(|s| s.score)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        println!(
            "  {:12} : {:3}枚 (平均スコア: {:.3}, 範囲: {:.3}-{:.3})",
            cat.name, cat.samples.len(), avg_score, min_score, max_score
        );
    }

    // 潜在的な問題を検出
    println!("\n=== 品質チェック ===");

    let mut warnings = Vec::new();

    for cat in categories {
        // サンプル数が少ない
        if cat.samples.len() < 50 {
            warnings.push(format!(
                "⚠️  {}: サンプル数が少ない ({}枚)",
                cat.name, cat.samples.len()
            ));
        }

        // 平均スコアが低い
        let avg_score = if !cat.samples.is_empty() {
            cat.samples
                .iter()
                .filter_map(|s| s.score)
                .sum::<f32>()
                / cat.samples.len() as f32
        } else {
            0.0
        };

        if avg_score < 0.7 && !cat.name.starts_with("empty") {
            warnings.push(format!(
                "⚠️  {}: 平均スコアが低い ({:.3}) - 誤分類の可能性",
                cat.name, avg_score
            ));
        }

        // 低スコアサンプルが多い
        let low_score_count = cat.samples
            .iter()
            .filter(|s| s.score.unwrap_or(1.0) < 0.6)
            .count();

        if low_score_count > 10 {
            warnings.push(format!(
                "⚠️  {}: スコア0.6未満のサンプルが{}枚 - 確認推奨",
                cat.name, low_score_count
            ));
        }
    }

    if warnings.is_empty() {
        println!("✓ 大きな問題は検出されませんでした");
    } else {
        for warning in warnings {
            println!("{}", warning);
        }
    }
}

/// HTMLビューアーを生成
fn generate_html_viewer(categories: &[CategoryInfo], output_path: &Path) -> Result<()> {
    println!("\nHTMLビューアーを生成中...");

    let mut html = String::from(
        r#"<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>トレーニングデータ検証</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            background-color: #e8f5e9;
            padding: 10px;
            border-radius: 5px;
            cursor: pointer;
            user-select: none;
        }
        h2:hover {
            background-color: #c8e6c9;
        }
        .category {
            margin-bottom: 30px;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stats {
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
            font-size: 14px;
        }
        .samples {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .sample {
            border: 2px solid #ddd;
            padding: 10px;
            text-align: center;
            background-color: #fafafa;
            border-radius: 5px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .sample:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            border-color: #4CAF50;
        }
        .sample img {
            width: 96px;
            height: 96px;
            image-rendering: pixelated;
            border: 1px solid #ccc;
            background-color: white;
        }
        .sample-info {
            margin-top: 8px;
            font-size: 12px;
            color: #666;
        }
        .score {
            font-weight: bold;
            padding: 3px 8px;
            border-radius: 3px;
            display: inline-block;
            margin-top: 5px;
        }
        .score-high { background-color: #4CAF50; color: white; }
        .score-medium { background-color: #FFC107; color: black; }
        .score-low { background-color: #F44336; color: white; }
        .summary {
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .summary h3 {
            margin-top: 0;
            color: #1976D2;
        }
        .warning {
            background-color: #fff3cd;
            border-left: 4px solid #FFC107;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        .collapsible-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }
        .collapsible-content.active {
            max-height: 10000px;
        }
        .toggle-all {
            margin-bottom: 20px;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .toggle-all:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <h1>🔍 トレーニングデータ検証ビューア</h1>

    <div class="summary">
        <h3>📊 サマリー</h3>
        <p><strong>総サンプル数:</strong> "#
    );

    let total_samples: usize = categories.iter().map(|c| c.samples.len()).sum();
    html.push_str(&format!("{}</p>\n", total_samples));
    html.push_str(&format!("<p><strong>カテゴリ数:</strong> {}</p>\n", categories.len()));

    html.push_str(r#"
        <h4>クレンジングが必要なケース:</h4>
        <ul>
            <li>❌ 明らかに間違ったカテゴリに分類されている</li>
            <li>⚠️ スコアが低い（0.6未満）画像は誤分類の可能性</li>
            <li>🔍 emptyカテゴリに実際にはアイコンが写っている</li>
            <li>🔍 アイコンカテゴリに空白画像が混入している</li>
            <li>⚡ インジケータの映り込みが激しい画像</li>
        </ul>
    </div>

    <button class="toggle-all" onclick="toggleAll()">全て展開/折りたたみ</button>
"#);

    // 各カテゴリ
    for cat in categories {
        let avg_score = if cat.samples.is_empty() {
            0.0
        } else {
            cat.samples
                .iter()
                .filter_map(|s| s.score)
                .sum::<f32>()
                / cat.samples.len() as f32
        };

        html.push_str(&format!(
            r#"
    <div class="category">
        <h2 onclick="toggleCategory(this)">
            📁 {} (ラベル: {}) - {}枚
        </h2>
        <div class="collapsible-content">
            <div class="stats">
                <strong>平均スコア:</strong> {:.3} |
                <strong>サンプル数:</strong> {}
            </div>
            <div class="samples">
"#,
            cat.name, cat.label, cat.samples.len(), avg_score, cat.samples.len()
        ));

        // サンプル画像（最大50枚表示）
        for sample in cat.samples.iter().take(50) {
            let score = sample.score.unwrap_or(0.0);
            let score_class = if score >= 0.8 {
                "score-high"
            } else if score >= 0.6 {
                "score-medium"
            } else {
                "score-low"
            };

            let rel_path = sample.path.strip_prefix(output_path.parent().unwrap()).unwrap();
            let rel_path_str = rel_path.to_str().unwrap().replace('\\', "/");

            html.push_str(&format!(
                r#"
                <div class="sample">
                    <img src="{}" alt="{}">
                    <div class="sample-info">
                        <div>{}</div>
                        <div class="score {}">スコア: {:.3}</div>
                    </div>
                </div>
"#,
                rel_path_str,
                sample.filename,
                sample.filename,
                score_class,
                score
            ));
        }

        html.push_str(
            r#"
            </div>
        </div>
    </div>
"#,
        );
    }

    html.push_str(
        r#"
    <script>
        function toggleCategory(element) {
            const content = element.nextElementSibling;
            content.classList.toggle('active');
        }

        function toggleAll() {
            const contents = document.querySelectorAll('.collapsible-content');
            const anyActive = Array.from(contents).some(c => c.classList.contains('active'));

            contents.forEach(content => {
                if (anyActive) {
                    content.classList.remove('active');
                } else {
                    content.classList.add('active');
                }
            });
        }

        // 最初のカテゴリを展開
        document.querySelector('.collapsible-content').classList.add('active');
    </script>
</body>
</html>
"#,
    );

    fs::write(output_path, html)?;
    println!("✓ HTMLビューアーを生成しました: {}", output_path.display());

    Ok(())
}

fn print_usage() {
    println!("=== トレーニングデータ検証ツール ===");
    println!();
    println!("使用方法:");
    println!("  inspect_training_data <トレーニングデータディレクトリ> [出力HTMLパス]");
    println!();
    println!("引数:");
    println!("  <トレーニングデータディレクトリ> : training_data ディレクトリ");
    println!("  [出力HTMLパス]                    : HTMLビューアーの出力先（デフォルト: training_data_inspect.html）");
    println!();
    println!("例:");
    println!("  inspect_training_data training_data");
    println!("  inspect_training_data training_data my_review.html");
    println!();
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        anyhow::bail!("引数が不足しています");
    }

    let data_dir = PathBuf::from(&args[1]);
    let output_html = if args.len() >= 3 {
        PathBuf::from(&args[2])
    } else {
        PathBuf::from("training_data_inspect.html")
    };

    if !data_dir.exists() {
        anyhow::bail!("ディレクトリが存在しません: {}", data_dir.display());
    }

    println!("=== トレーニングデータ検証ツール ===\n");

    // データ読み込み
    let categories = load_training_data(&data_dir)?;

    // 統計情報表示
    calculate_statistics(&categories);

    // HTMLビューアー生成
    generate_html_viewer(&categories, &output_html)?;

    println!("\n=== 次のステップ ===");
    println!("1. ブラウザでHTMLを開く:");
    println!("   {}", output_html.display());
    println!();
    println!("2. 各カテゴリの画像を確認:");
    println!("   - 明らかな誤分類があれば手動で移動");
    println!("   - スコアが低い（赤色）画像は特に注意");
    println!("   - emptyカテゴリに実際にアイコンがないか確認");
    println!();
    println!("3. クレンジング方法:");
    println!("   - 誤分類画像を正しいカテゴリフォルダに移動");
    println!("   - 明らかに品質が低い画像を削除");
    println!("   - 必要に応じてデータ再収集");
    println!();
    println!("推奨:");
    println!("  少量のノイズ（5-10%）は機械学習で許容されます。");
    println!("  まずは現状のデータで学習し、精度が低ければクレンジングを検討してください。");

    Ok(())
}
