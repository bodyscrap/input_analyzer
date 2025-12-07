//! デフォルト設定ファイルを生成するユーティリティ
//!
//! # 使用方法
//! ```bash
//! cargo run --bin create_default_config
//! ```

use input_analyzer::config::AppConfig;

fn main() -> anyhow::Result<()> {
    println!("=== デフォルト設定ファイル生成 ===\n");

    let config = AppConfig::default();

    // 設定内容を表示
    config.display();

    // 設定ファイルを保存
    let path = AppConfig::default_path();
    config.save(&path)?;

    println!("✓ デフォルト設定ファイルを生成しました: {}", path.display());
    println!();
    println!("設定ファイルを編集して、アプリケーションの動作をカスタマイズできます。");
    println!("次回の実行時から、この設定が自動的に使用されます。");

    Ok(())
}
