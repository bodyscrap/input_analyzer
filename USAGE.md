# 使用方法ガイド

## 基本的な使い方

### 1. 動画からフレームを抽出する

最も基本的な使い方は、サンプル動画からフレームを抽出することです。

```bash
cargo run
```

デフォルトでは、`sample_data/input_sample.mp4` から30フレームごと（約0.5秒ごと）にフレームを抽出し、`output/frames/` ディレクトリにPNG形式で保存します。

### 2. 抽出されたフレームを確認する

Windows:
```bash
explorer output\frames
```

Linux/macOS:
```bash
ls -lh output/frames/
```

## カスタマイズ例

### 全フレームを抽出する

`src/main.rs` の設定を以下のように変更します：

```rust
let config = FrameExtractorConfig {
    frame_interval: 1,  // 全フレームを抽出
    output_dir: PathBuf::from("output/all_frames"),
    image_format: "png".to_string(),
    jpeg_quality: 95,
};
```

### JPEG形式で保存する（ファイルサイズを小さくする）

```rust
let config = FrameExtractorConfig {
    frame_interval: 30,
    output_dir: PathBuf::from("output/frames_jpg"),
    image_format: "jpg".to_string(),  // JPEG形式
    jpeg_quality: 85,  // 品質: 0-100 (85は高品質)
};
```

### 1秒ごとにフレームを抽出する

動画のFPSが60の場合：

```rust
let config = FrameExtractorConfig {
    frame_interval: 60,  // 60フレームごと = 1秒ごと
    output_dir: PathBuf::from("output/frames_1sec"),
    image_format: "png".to_string(),
    jpeg_quality: 95,
};
```

### 独自の動画ファイルを処理する

```rust
fn main() -> anyhow::Result<()> {
    let video_path = "path/to/your/video.mp4";
    
    let config = FrameExtractorConfig {
        frame_interval: 30,
        output_dir: PathBuf::from("output/my_video_frames"),
        image_format: "png".to_string(),
        jpeg_quality: 95,
    };
    
    let extractor = FrameExtractor::new(config);
    extractor.extract_frames(video_path)?;
    
    Ok(())
}
```

## 高度な使い方

### 特定のフレーム番号を抽出する

```rust
use frame_extractor::{FrameExtractor, FrameExtractorConfig};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let config = FrameExtractorConfig {
        frame_interval: 1,
        output_dir: PathBuf::from("output/specific_frames"),
        image_format: "png".to_string(),
        jpeg_quality: 95,
    };
    
    let extractor = FrameExtractor::new(config);
    
    // フレーム番号で指定（0が最初のフレーム）
    extractor.extract_frame_at("sample_data/input_sample.mp4", 0)?;     // 最初のフレーム
    extractor.extract_frame_at("sample_data/input_sample.mp4", 100)?;   // 100番目のフレーム
    extractor.extract_frame_at("sample_data/input_sample.mp4", 1000)?;  // 1000番目のフレーム
    
    println!("特定のフレームを抽出しました");
    Ok(())
}
```

### 時間指定でフレームを抽出する

```rust
fn main() -> anyhow::Result<()> {
    let config = FrameExtractorConfig {
        frame_interval: 1,
        output_dir: PathBuf::from("output/time_based_frames"),
        image_format: "png".to_string(),
        jpeg_quality: 95,
    };
    
    let extractor = FrameExtractor::new(config);
    
    // 秒単位で時間を指定
    extractor.extract_frame_at_time("sample_data/input_sample.mp4", 0.0)?;    // 開始時点
    extractor.extract_frame_at_time("sample_data/input_sample.mp4", 1.5)?;    // 1.5秒時点
    extractor.extract_frame_at_time("sample_data/input_sample.mp4", 5.0)?;    // 5秒時点
    extractor.extract_frame_at_time("sample_data/input_sample.mp4", 10.5)?;   // 10.5秒時点
    
    println!("時間指定でフレームを抽出しました");
    Ok(())
}
```

### 動画情報を取得する

```rust
use frame_extractor::FrameExtractor;

fn main() -> anyhow::Result<()> {
    let video_path = "sample_data/input_sample.mp4";
    
    // 動画情報を取得
    let info = FrameExtractor::get_video_info(video_path)?;
    
    println!("動画ファイル: {}", video_path);
    println!("解像度: {}x{}", info.width, info.height);
    println!("FPS: {:.2}", info.fps);
    println!("再生時間: {:.2}秒", info.duration_sec);
    
    // 総フレーム数を計算
    let total_frames = (info.fps * info.duration_sec) as u32;
    println!("推定総フレーム数: {}", total_frames);
    
    Ok(())
}
```

### 複数の動画を一括処理する

```rust
use frame_extractor::{FrameExtractor, FrameExtractorConfig};
use std::path::PathBuf;
use std::fs;

fn main() -> anyhow::Result<()> {
    // 動画ファイルのリスト
    let video_files = vec![
        "sample_data/video1.mp4",
        "sample_data/video2.mp4",
        "sample_data/video3.mp4",
    ];
    
    for (index, video_path) in video_files.iter().enumerate() {
        println!("\n処理中: {} ({}/{})", video_path, index + 1, video_files.len());
        
        let config = FrameExtractorConfig {
            frame_interval: 30,
            output_dir: PathBuf::from(format!("output/video_{}", index + 1)),
            image_format: "png".to_string(),
            jpeg_quality: 95,
        };
        
        let extractor = FrameExtractor::new(config);
        
        match extractor.extract_frames(video_path) {
            Ok(paths) => {
                println!("✓ 完了: {}フレーム抽出", paths.len());
            }
            Err(e) => {
                eprintln!("✗ エラー: {}", e);
                continue;
            }
        }
    }
    
    println!("\n全ての動画の処理が完了しました");
    Ok(())
}
```

### 条件付きフレーム抽出

```rust
use frame_extractor::{FrameExtractor, FrameExtractorConfig};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let video_path = "sample_data/input_sample.mp4";
    
    // 動画情報を取得
    let info = FrameExtractor::get_video_info(video_path)?;
    
    // 動画の長さに応じて抽出間隔を調整
    let frame_interval = if info.duration_sec < 10.0 {
        10  // 短い動画は10フレームごと
    } else if info.duration_sec < 60.0 {
        30  // 中程度の動画は30フレームごと
    } else {
        60  // 長い動画は60フレームごと
    };
    
    println!("動画の長さ: {:.2}秒", info.duration_sec);
    println!("抽出間隔: {}フレームごと", frame_interval);
    
    let config = FrameExtractorConfig {
        frame_interval,
        output_dir: PathBuf::from("output/adaptive_frames"),
        image_format: "png".to_string(),
        jpeg_quality: 95,
    };
    
    let extractor = FrameExtractor::new(config);
    extractor.extract_frames(video_path)?;
    
    Ok(())
}
```

## パフォーマンスのヒント

### メモリ使用量を抑える

高解像度の動画や長時間の動画を処理する場合：

1. **抽出間隔を大きくする**: `frame_interval` を大きくすると、メモリ使用量が減ります
2. **JPEG形式を使用する**: PNG よりもファイルサイズが小さくなります
3. **出力ディレクトリを定期的にクリア**: 古いフレームを削除してディスク容量を確保します

### 処理速度を上げる

1. **SSDに出力する**: HDDよりもSSDの方が書き込み速度が速いです
2. **JPEG品質を下げる**: `jpeg_quality` を70-80程度にすると保存が速くなります
3. **不要なフレームを抽出しない**: `frame_interval` を適切に設定します

## トラブルシューティング

### メモリ不足エラー

```
Error: memory allocation failed
```

→ `frame_interval` を大きくして、一度に処理するフレーム数を減らしてください。

### ディスク容量不足

```
Error: No space left on device
```

→ 出力ディレクトリのファイルを削除するか、JPEG形式を使用してください。

### 処理が遅い

→ 以下を試してください：
- JPEG形式を使用する
- `jpeg_quality` を下げる（85 → 75など）
- `frame_interval` を大きくする

### 動画が開けない

```
Error: 動画ファイルを開けませんでした
```

→ 以下を確認してください：
- ファイルパスが正しいか
- ファイルが破損していないか
- GStreamerのプラグインが正しくインストールされているか

## ライブラリとして使用する

他のRustプロジェクトから使用する場合：

```toml
# Cargo.toml
[dependencies]
input_analyzer = { path = "../input_analyzer" }
```

```rust
// main.rs
use input_analyzer::frame_extractor::{FrameExtractor, FrameExtractorConfig};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let config = FrameExtractorConfig {
        frame_interval: 30,
        output_dir: PathBuf::from("frames"),
        image_format: "png".to_string(),
        jpeg_quality: 95,
    };
    
    let extractor = FrameExtractor::new(config);
    extractor.extract_frames("video.mp4")?;
    
    Ok(())
}
```

## よくある質問

**Q: サポートされている動画形式は？**
A: GStreamerがサポートする全ての形式（MP4, AVI, MOV, MKV, WebM, FLVなど）が使用できます。

**Q: 4K動画も処理できますか？**
A: はい、できます。ただし、メモリとディスク容量に注意してください。

**Q: フレームレートを変更できますか？**
A: このツールはフレームレートを変更せず、元の動画からフレームを抽出するだけです。

**Q: 音声も抽出できますか？**
A: 現在のバージョンでは画像フレームのみを抽出します。音声抽出は今後のバージョンで対応予定です。

**Q: GPUアクセラレーションは使用できますか？**
A: GStreamerがGPUデコーダーをサポートしている場合、自動的に使用されます。