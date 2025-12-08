//! 既存のmodel.mpkをtar.gz形式に変換するユーティリティ

#[cfg(feature = "ml")]
use anyhow::{Context, Result};
#[cfg(feature = "ml")]
use input_analyzer::model_metadata::ModelMetadata;
#[cfg(feature = "ml")]
use input_analyzer::model_storage;

#[cfg(feature = "ml")]
fn main() -> Result<()> {
    use input_analyzer::config::AppConfig;
    
    println!("=== 既存モデルをtar.gz形式に変換 ===\n");

    // 設定ファイルを読み込み
    let config = AppConfig::load_or_default();

    // 既存のmodel.mpkを読み込み
    let model_binary = std::fs::read("models/model.mpk")
        .context("Failed to read models/model.mpk")?;
    
    println!("✓ models/model.mpk を読み込みました ({} bytes)", model_binary.len());

    // メタデータを作成（設定から値を使用）
    let button_labels = vec![
        "A1".to_string(),
        "A2".to_string(),
        "B".to_string(),
        "Start".to_string(),
        "W".to_string(),
    ];

    let metadata = ModelMetadata::new(
        button_labels,
        640,  // image_width (固定値、学習時に検出された値を使用することを推奨)
        480,  // image_height (固定値、学習時に検出された値を使用することを推奨)
        config.button_tile.x,
        config.button_tile.y,
        config.button_tile.width,
        config.button_tile.height,
        config.button_tile.columns_per_row,
        48,   // model_input_size
        1,    // num_epochs
    );

    println!("\nメタデータ:");
    model_storage::print_metadata_info(&metadata);

    // tar.gz形式で保存
    let output_path = std::path::PathBuf::from("models/icon_classifier");
    model_storage::save_model_with_metadata(&output_path, &metadata, &model_binary)
        .context("Failed to save model with metadata")?;

    let tar_gz_path = output_path.with_extension("tar.gz");
    println!("\n✓ tar.gz形式で保存しました: {}", tar_gz_path.display());

    // 保存したファイルを検証
    let file_size = std::fs::metadata(&tar_gz_path)?.len();
    println!("  ファイルサイズ: {} bytes ({:.2} MB)", file_size, file_size as f64 / 1024.0 / 1024.0);

    // メタデータを読み込んで確認
    println!("\n=== 保存したtar.gzの検証 ===");
    let loaded_metadata = model_storage::load_metadata(&tar_gz_path)
        .context("Failed to load metadata from tar.gz")?;
    
    println!("✓ メタデータの読み込み成功");
    model_storage::print_metadata_info(&loaded_metadata);

    // モデルバイナリのサイズを確認
    let loaded_binary = model_storage::load_model_binary(&tar_gz_path)
        .context("Failed to load model binary from tar.gz")?;
    println!("\n✓ モデルバイナリの読み込み成功 ({} bytes)", loaded_binary.len());

    if loaded_binary.len() == model_binary.len() {
        println!("✓ バイナリサイズが一致しました");
    } else {
        println!("⚠ バイナリサイズが一致しません (元: {}, 読込: {})", 
            model_binary.len(), loaded_binary.len());
    }

    println!("\n✓ 変換完了!");

    Ok(())
}

#[cfg(not(feature = "ml"))]
fn main() {
    eprintln!("このバイナリは 'ml' フィーチャーを有効にしてビルドする必要があります。");
    eprintln!("実行方法: cargo run --bin convert_model_to_targz --features ml");
}
