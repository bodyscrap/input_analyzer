mod frame_extractor;

use frame_extractor::{FrameExtractor, FrameExtractorConfig};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    println!("=== ゲーム入力解析アプリ - フレーム抽出 ===\n");

    // サンプル動画のパス
    let video_path = "sample_data/input_sample.mp4";

    // 動画情報を取得して表示
    println!("動画情報を取得中...");
    match FrameExtractor::get_video_info(video_path) {
        Ok(info) => {
            println!("動画情報:");
            println!("  解像度: {}x{}", info.width, info.height);
            println!("  FPS: {:.2}", info.fps);
            println!("  再生時間: {:.2}秒\n", info.duration_sec);
        }
        Err(e) => {
            eprintln!("エラー: 動画情報の取得に失敗しました: {}", e);
            return Err(e);
        }
    }

    // フレーム抽出の設定
    let config = FrameExtractorConfig {
        frame_interval: 30, // 30フレームごとに抽出（約1秒ごと if FPS=30）
        output_dir: PathBuf::from("output/frames"),
        image_format: "png".to_string(),
        jpeg_quality: 95,
    };

    // フレーム抽出器を作成
    let extractor = FrameExtractor::new(config);

    // フレームを抽出
    println!("フレーム抽出を開始します...\n");
    match extractor.extract_frames(video_path) {
        Ok(paths) => {
            println!("\n✓ フレーム抽出が完了しました");
            println!("  抽出されたフレーム数: {}", paths.len());
            if !paths.is_empty() {
                println!("  出力ディレクトリ: {}", paths[0].parent().unwrap().display());
            }
        }
        Err(e) => {
            eprintln!("\nエラー: フレーム抽出に失敗しました: {}", e);
            return Err(e);
        }
    }

    // 特定のフレームを抽出する例（コメントアウト）
    // println!("\n特定のフレームを抽出中...");
    // let specific_config = FrameExtractorConfig {
    //     output_dir: PathBuf::from("output/specific_frames"),
    //     ..Default::default()
    // };
    // let specific_extractor = FrameExtractor::new(specific_config);
    // specific_extractor.extract_frame_at(video_path, 0)?; // 最初のフレーム
    // specific_extractor.extract_frame_at_time(video_path, 1.0)?; // 1秒時点のフレーム

    Ok(())
}
