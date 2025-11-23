use input_analyzer::input_analyzer::{InputAnalyzer, InputIndicatorRegion};
use std::env;

fn print_usage() {
    println!("使用方法:");
    println!("  analyze_input_custom <画像パス> [x] [y] [width] [height] [rows] [cols]");
    println!();
    println!("引数:");
    println!("  <画像パス>  : 解析する画像ファイルのパス（必須）");
    println!("  [x]         : 入力インジケータ領域のX座標（デフォルト: 216）");
    println!("  [y]         : 入力インジケータ領域のY座標（デフォルト: 189）");
    println!("  [width]     : 入力インジケータ領域の幅（デフォルト: 326）");
    println!("  [height]    : 入力インジケータ領域の高さ（デフォルト: 759）");
    println!("  [rows]      : 行数（デフォルト: 16）");
    println!("  [cols]      : 列数（デフォルト: 7）");
    println!();
    println!("例:");
    println!("  # デフォルト設定で解析");
    println!("  analyze_input_custom output/frames/frame_000630.png");
    println!();
    println!("  # カスタム領域で解析");
    println!("  analyze_input_custom output/frames/frame_000630.png 220 190 320 750 16 7");
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("エラー: 画像パスが指定されていません\n");
        print_usage();
        std::process::exit(1);
    }

    let image_path = &args[1];

    // 領域設定を解析
    let region = if args.len() >= 8 {
        let x = args[2].parse::<u32>()
            .map_err(|_| anyhow::anyhow!("X座標の解析に失敗しました: {}", args[2]))?;
        let y = args[3].parse::<u32>()
            .map_err(|_| anyhow::anyhow!("Y座標の解析に失敗しました: {}", args[3]))?;
        let width = args[4].parse::<u32>()
            .map_err(|_| anyhow::anyhow!("幅の解析に失敗しました: {}", args[4]))?;
        let height = args[5].parse::<u32>()
            .map_err(|_| anyhow::anyhow!("高さの解析に失敗しました: {}", args[5]))?;
        let rows = args[6].parse::<u32>()
            .map_err(|_| anyhow::anyhow!("行数の解析に失敗しました: {}", args[6]))?;
        let cols = args[7].parse::<u32>()
            .map_err(|_| anyhow::anyhow!("列数の解析に失敗しました: {}", args[7]))?;

        println!("カスタム領域設定を使用します");
        InputIndicatorRegion::new(x, y, width, height, rows, cols)
    } else {
        println!("デフォルト領域設定を使用します");
        InputIndicatorRegion::default()
    };

    println!("\n=== ゲーム入力解析 - 入力インジケータ抽出 ===\n");
    println!("画像ファイル: {}", image_path);
    println!("\n入力インジケータ領域設定:");
    println!("  位置: ({}, {})", region.x, region.y);
    println!("  サイズ: {}x{}", region.width, region.height);
    println!("  グリッド: {}行 x {}列", region.rows, region.cols);
    println!("  セルサイズ: {}x{}\n", region.cell_width(), region.cell_height());

    // 入力解析器を作成
    let analyzer = InputAnalyzer::new(region);

    // 出力ディレクトリ名を画像ファイル名から生成
    let output_base = format!("output/analysis/{}",
        std::path::Path::new(image_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown"));

    // デバッグ画像を作成（グリッド線付き）
    println!("デバッグ画像を作成中...");
    let debug_path = format!("{}_debug_grid.png", output_base);
    analyzer.save_debug_image(image_path, &debug_path)?;

    // 入力インジケータ領域全体を抽出
    println!("入力インジケータ領域を抽出中...");
    let indicator_path = format!("{}_indicator_region.png", output_base);
    analyzer.extract_and_save_indicator(image_path, &indicator_path)?;

    // すべての入力行を抽出して保存
    println!("\nすべての入力行を抽出中...");
    let rows_dir = format!("{}_input_rows", output_base);
    let rows = analyzer.extract_and_save_all_rows(image_path, &rows_dir)?;

    // 結果のサマリーを表示
    println!("\n--- 抽出結果サマリー ---");
    println!("総行数: {}", rows.len());
    println!("\n各行の詳細:");
    for (i, row) in rows.iter().enumerate() {
        let frame_count_size = (
            row.frame_count_image.width(),
            row.frame_count_image.height()
        );
        println!("  行{:2}: フレームカウント画像 ({}x{}) + {}個の入力アイコン",
                 i, frame_count_size.0, frame_count_size.1, row.input_icons.len());
    }

    // アイコンサイズの統計
    if !rows.is_empty() && !rows[0].input_icons.is_empty() {
        let icon_size = (
            rows[0].input_icons[0].width(),
            rows[0].input_icons[0].height()
        );
        println!("\n入力アイコンサイズ: {}x{}", icon_size.0, icon_size.1);
    }

    println!("\n出力ファイル:");
    println!("  デバッグ画像: {}", debug_path);
    println!("  インジケータ領域: {}", indicator_path);
    println!("  入力行データ: {}/", rows_dir);

    println!("\n✓ 解析が完了しました");

    Ok(())
}
