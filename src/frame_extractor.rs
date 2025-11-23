use anyhow::{Context, Result};
use gstreamer::prelude::*;
use gstreamer::{self as gst, ElementFactory};
use gstreamer_app::AppSink;
use image::{ImageBuffer, Rgb};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// フレーム抽出の設定
#[derive(Debug, Clone)]
pub struct FrameExtractorConfig {
    /// フレーム抽出間隔（フレーム数）。1なら全フレーム、30なら30フレームごと
    pub frame_interval: u32,
    /// 出力ディレクトリ
    pub output_dir: PathBuf,
    /// 出力画像のフォーマット（例: "png", "jpg"）
    pub image_format: String,
    /// JPEGの品質（0-100、jpgの場合のみ有効）
    pub jpeg_quality: u8,
}

impl Default for FrameExtractorConfig {
    fn default() -> Self {
        Self {
            frame_interval: 1,
            output_dir: PathBuf::from("output/frames"),
            image_format: "png".to_string(),
            jpeg_quality: 95,
        }
    }
}

/// 動画情報
#[derive(Debug, Clone)]
pub struct CustomVideoInfo {
    pub width: i32,
    pub height: i32,
    pub fps: f64,
    pub duration_sec: f64,
}

/// フレーム抽出器
pub struct FrameExtractor {
    config: FrameExtractorConfig,
}

impl FrameExtractor {
    /// 新しいフレーム抽出器を作成
    pub fn new(config: FrameExtractorConfig) -> Self {
        Self { config }
    }

    /// デフォルト設定でフレーム抽出器を作成
    pub fn default() -> Self {
        Self {
            config: FrameExtractorConfig::default(),
        }
    }

    /// GStreamerを初期化
    fn init_gstreamer() -> Result<()> {
        gst::init().context("GStreamerの初期化に失敗しました")?;
        Ok(())
    }

    /// 動画ファイルの情報を取得
    pub fn get_video_info<P: AsRef<Path>>(video_path: P) -> Result<CustomVideoInfo> {
        Self::init_gstreamer()?;

        let video_path = video_path.as_ref();
        let uri = format!(
            "file:///{}",
            video_path
                .canonicalize()
                .context("動画ファイルのパスを解決できませんでした")?
                .to_str()
                .unwrap()
                .replace("\\", "/")
                .trim_start_matches("\\\\?\\")
        );

        // Discovererを使って動画情報を取得
        let discoverer = gstreamer_pbutils::Discoverer::new(gst::ClockTime::from_seconds(10))
            .context("Discovererの作成に失敗しました")?;

        let info = discoverer
            .discover_uri(&uri)
            .context("動画の解析に失敗しました")?;

        let video_streams = info.video_streams();
        if video_streams.is_empty() {
            anyhow::bail!("動画ストリームが見つかりません");
        }

        let video_stream = &video_streams[0];
        let width = video_stream.width() as i32;
        let height = video_stream.height() as i32;
        let fps_num = video_stream.framerate().numer() as f64;
        let fps_den = video_stream.framerate().denom() as f64;
        let fps = fps_num / fps_den;

        let duration = info.duration();
        let duration_sec = if let Some(dur) = duration {
            dur.seconds() as f64
        } else {
            0.0
        };

        Ok(CustomVideoInfo {
            width,
            height,
            fps,
            duration_sec,
        })
    }

    /// 動画からフレームを抽出
    pub fn extract_frames<P: AsRef<Path>>(&self, video_path: P) -> Result<Vec<PathBuf>> {
        Self::init_gstreamer()?;

        let video_path = video_path.as_ref();
        println!("動画ファイルを開いています: {}", video_path.display());

        // 出力ディレクトリを作成
        std::fs::create_dir_all(&self.config.output_dir)
            .context("出力ディレクトリの作成に失敗しました")?;

        // 動画情報を取得
        let info = Self::get_video_info(video_path)?;
        println!("動画情報:");
        println!("  解像度: {}x{}", info.width, info.height);
        println!("  FPS: {:.2}", info.fps);
        println!("  再生時間: {:.2}秒", info.duration_sec);

        let _uri = format!(
            "file:///{}",
            video_path
                .canonicalize()?
                .to_str()
                .unwrap()
                .replace("\\", "/")
                .trim_start_matches("\\\\?\\")
        );

        // GStreamerパイプラインを構築
        let pipeline = gst::Pipeline::new();

        // エレメントを作成
        let source = ElementFactory::make("filesrc")
            .name("source")
            .build()
            .context("filesrcの作成に失敗しました")?;

        let decodebin = ElementFactory::make("decodebin")
            .name("decoder")
            .build()
            .context("decodebinの作成に失敗しました")?;

        let videoconvert = ElementFactory::make("videoconvert")
            .name("converter")
            .build()
            .context("videoconvertの作成に失敗しました")?;

        let appsink = ElementFactory::make("appsink")
            .name("sink")
            .build()
            .context("appsinkの作成に失敗しました")?;

        let appsink = appsink
            .dynamic_cast::<AppSink>()
            .map_err(|_| anyhow::anyhow!("appsinkへのキャストに失敗しました"))?;

        // AppSinkの設定
        appsink.set_caps(Some(
            &gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .build(),
        ));
        appsink.set_property("emit-signals", false);
        appsink.set_property("sync", false);

        // ファイルパスを設定
        source.set_property("location", video_path.to_str().unwrap());

        // パイプラインにエレメントを追加
        pipeline
            .add_many(&[&source, &decodebin, &videoconvert, appsink.upcast_ref::<gst::Element>()])
            .context("エレメントの追加に失敗しました")?;

        // sourceとdecodebinをリンク
        source
            .link(&decodebin)
            .context("sourceとdecoderのリンクに失敗しました")?;

        // videoconvertとappsinkをリンク
        videoconvert
            .link(appsink.upcast_ref::<gst::Element>())
            .context("converterとsinkのリンクに失敗しました")?;

        // decodebinの動的パッドをリンク
        let videoconvert_clone = videoconvert.clone();
        decodebin.connect_pad_added(move |_src, src_pad| {
            let sink_pad = videoconvert_clone
                .static_pad("sink")
                .expect("videoconvertのsinkパッドが見つかりません");

            if !sink_pad.is_linked() {
                if let Err(e) = src_pad.link(&sink_pad) {
                    eprintln!("パッドのリンクに失敗: {:?}", e);
                }
            }
        });

        println!("\nフレーム抽出中...");
        println!("  抽出間隔: {}フレームごと", self.config.frame_interval);
        println!("  出力先: {}", self.config.output_dir.display());

        let output_paths = Arc::new(Mutex::new(Vec::new()));
        let frame_count = Arc::new(Mutex::new(0u32));
        let extracted_count = Arc::new(Mutex::new(0u32));

        let output_paths_clone = output_paths.clone();
        let frame_count_clone = frame_count.clone();
        let extracted_count_clone = extracted_count.clone();
        let config = self.config.clone();

        // サンプルコールバックを設定
        appsink.set_callbacks(
            gstreamer_app::AppSinkCallbacks::builder()
                .new_sample(move |appsink| {
                    let sample = appsink.pull_sample().map_err(|_| gst::FlowError::Error)?;
                    let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;
                    let caps = sample.caps().ok_or(gst::FlowError::Error)?;

                    let video_info = gstreamer_video::VideoInfo::from_caps(caps)
                        .map_err(|_| gst::FlowError::Error)?;

                    let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;

                    let mut frame_num = frame_count_clone.lock().unwrap();
                    let current_frame = *frame_num;
                    *frame_num += 1;

                    // 指定された間隔でフレームを保存
                    if current_frame % config.frame_interval == 0 {
                        let width = video_info.width() as u32;
                        let height = video_info.height() as u32;

                        // RGB画像として保存
                        if let Some(img_buffer) =
                            ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, map.as_slice())
                        {
                            let filename = format!("frame_{:06}.{}", current_frame, config.image_format);
                            let output_path = config.output_dir.join(&filename);

                            if let Err(e) = if config.image_format == "jpg" || config.image_format == "jpeg" {
                                let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                                    std::fs::File::create(&output_path).unwrap(),
                                    config.jpeg_quality,
                                );
                                img_buffer.write_with_encoder(encoder)
                            } else {
                                img_buffer.save(&output_path)
                            } {
                                eprintln!("フレームの保存に失敗: {}", e);
                            } else {
                                let mut paths = output_paths_clone.lock().unwrap();
                                paths.push(output_path);

                                let mut extracted = extracted_count_clone.lock().unwrap();
                                *extracted += 1;

                                if *extracted % 10 == 0 {
                                    println!("  {}フレーム抽出完了", *extracted);
                                }
                            }
                        }
                    }

                    Ok(gst::FlowSuccess::Ok)
                })
                .build(),
        );

        // パイプラインを開始
        pipeline
            .set_state(gst::State::Playing)
            .context("パイプラインの開始に失敗しました")?;

        // バスメッセージを処理
        let bus = pipeline
            .bus()
            .expect("パイプラインにバスがありません");

        for msg in bus.iter_timed(gst::ClockTime::NONE) {
            use gst::MessageView;

            match msg.view() {
                MessageView::Eos(..) => {
                    println!("\n動画の終わりに到達しました");
                    break;
                }
                MessageView::Error(err) => {
                    pipeline.set_state(gst::State::Null).ok();
                    anyhow::bail!(
                        "エラーが発生しました: {} (デバッグ情報: {:?})",
                        err.error(),
                        err.debug()
                    );
                }
                _ => (),
            }
        }

        // パイプラインを停止
        pipeline
            .set_state(gst::State::Null)
            .context("パイプラインの停止に失敗しました")?;

        let final_frame_count = *frame_count.lock().unwrap();
        let final_extracted_count = *extracted_count.lock().unwrap();

        println!("\n抽出完了!");
        println!("  処理フレーム数: {}", final_frame_count);
        println!("  抽出フレーム数: {}", final_extracted_count);

        let paths = Arc::try_unwrap(output_paths)
            .map(|m| m.into_inner().unwrap())
            .unwrap_or_else(|arc| arc.lock().unwrap().clone());

        Ok(paths)
    }

    /// 特定のフレーム番号のフレームを抽出
    pub fn extract_frame_at<P: AsRef<Path>>(
        &self,
        video_path: P,
        frame_number: u32,
    ) -> Result<PathBuf> {
        // 一時的に設定を変更して特定のフレームのみを抽出
        let mut temp_config = self.config.clone();
        temp_config.frame_interval = frame_number + 1;

        let temp_extractor = FrameExtractor::new(temp_config);
        let paths = temp_extractor.extract_frames(video_path)?;

        paths
            .into_iter()
            .find(|p| {
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.strip_prefix("frame_"))
                    .and_then(|s| s.parse::<u32>().ok())
                    .map(|n| n == frame_number)
                    .unwrap_or(false)
            })
            .ok_or_else(|| anyhow::anyhow!("指定されたフレームが見つかりませんでした"))
    }

    /// 時間指定でフレームを抽出（秒単位）
    pub fn extract_frame_at_time<P: AsRef<Path>>(
        &self,
        video_path: P,
        time_sec: f64,
    ) -> Result<PathBuf> {
        let info = Self::get_video_info(&video_path)?;
        let frame_number = (time_sec * info.fps) as u32;
        self.extract_frame_at(video_path, frame_number)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_extractor_config_default() {
        let config = FrameExtractorConfig::default();
        assert_eq!(config.frame_interval, 1);
        assert_eq!(config.image_format, "png");
        assert_eq!(config.jpeg_quality, 95);
    }
}
