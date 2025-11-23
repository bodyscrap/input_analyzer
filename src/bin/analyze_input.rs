use input_analyzer::input_analyzer::{InputAnalyzer, InputIndicatorRegion};

fn main() -> anyhow::Result<()> {
    println!("=== ゲーム入力解析 - 入力インジケータ抽出 ===\n");

    // サンプル画像のパス
    let image_path = "output/frames/frame_000630.png";

    // 入力インジケータ領域の設定
    let region = InputIndicatorRegion::default();
    println!("入力インジケータ領域設定:");
    println!("  位置: ({}, {})", region.x, region.y);
    println!("  サイズ: {}x{}", region.width, region.height);
    println!("  グリッド: {}行 x {}列", region.rows, region.cols);
    println!("  セルサイズ: {}x{}\n", region.cell_width(), region.cell_height());

    // 入力解析器を作成
    let analyzer = InputAnalyzer::new(region);

    // デバッグ画像を作成（グリッド線付き）
    println!("デバッグ画像を作成中...");
    analyzer.save_debug_image(
        image_path,
        "output/analysis/debug_grid.png",
    )?;

    // 入力インジケータ領域全体を抽出
    println!("入力インジケータ領域を抽出中...");
    analyzer.extract_and_save_indicator(
        image_path,
        "output/analysis/indicator_region.png",
    )?;

    // すべての入力行を抽出して保存
    println!("\nすべての入力行を抽出中...");
    let rows = analyzer.extract_and_save_all_rows(
        image_path,
        "output/analysis/input_rows",
    )?;

    // 結果のサマリーを表示
    println!("\n--- 抽出結果サマリー ---");
    println!("総行数: {}", rows.len());
    println!("\n各行の詳細:");
    for (i, row) in rows.iter().enumerate() {
        println!("  行{:2}: フレームカウント画像 + {}個の入力アイコン",
                 i, row.input_icons.len());
    }

    println!("\n出力ファイル:");
    println!("  デバッグ画像: output/analysis/debug_grid.png");
    println!("  インジケータ領域: output/analysis/indicator_region.png");
    println!("  入力行データ: output/analysis/input_rows/");

    Ok(())
}
