//! 入力履歴CSV編集GUIアプリケーション
//!
//! # 機能
//! - 動画から入力履歴を抽出してCSVとして開く
//! - 既存のCSVファイルを開く
//! - 入力履歴の編集（追加・削除・変更）
//! - CSVファイルとして保存
//!
//! # 使用方法
//! ```bash
//! cargo run --release --features gui,ml --bin input_editor_gui
//! ```

#[cfg(all(feature = "gui", feature = "ml"))]
use eframe::egui;
#[cfg(all(feature = "gui", feature = "ml"))]
use rfd;
#[cfg(all(feature = "gui", feature = "ml"))]
use std::path::PathBuf;
#[cfg(all(feature = "gui", feature = "ml"))]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[cfg(all(feature = "gui", feature = "ml"))]
use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::Tensor,
};
#[cfg(all(feature = "gui", feature = "ml"))]
use input_analyzer::config::{AppConfig, DeviceType};
#[cfg(all(feature = "gui", feature = "ml"))]
use input_analyzer::frame_extractor::FrameExtractor;
#[cfg(all(feature = "gui", feature = "ml"))]
use input_analyzer::input_history_extractor::{
    extract_bottom_row_icons, update_input_state, InputState,
};
#[cfg(all(feature = "gui", feature = "ml"))]
use input_analyzer::ml_model::{
    load_and_normalize_image, IconClassifier, ModelConfig, CLASS_NAMES,
};

#[cfg(all(feature = "gui", feature = "ml"))]
type WgpuBackend = burn_wgpu::Wgpu;
#[cfg(all(feature = "gui", feature = "ml"))]
type NdArrayBackend = burn_ndarray::NdArray<f32>;

#[cfg(all(feature = "gui", feature = "ml"))]
#[derive(Debug, Clone, Copy, PartialEq)]
enum BackendType {
    Gpu,
    Cpu,
}

/// 入力レコード（1行分）
#[cfg(all(feature = "gui", feature = "ml"))]
#[derive(Debug, Clone, PartialEq)]
struct InputRecord {
    duration: u32,
    direction: u8,
    btn_a1: bool,
    btn_a2: bool,
    btn_b: bool,
    btn_w: bool,
    btn_start: bool,
}

#[cfg(all(feature = "gui", feature = "ml"))]
impl InputRecord {
    fn new() -> Self {
        Self {
            duration: 1,
            direction: 5,
            btn_a1: false,
            btn_a2: false,
            btn_b: false,
            btn_w: false,
            btn_start: false,
        }
    }

    fn from_csv_line(line: &str) -> Option<Self> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 7 {
            return None;
        }

        Some(Self {
            duration: parts[0].parse().ok()?,
            direction: parts[1].parse().ok()?,
            btn_a1: parts[2] == "1",
            btn_a2: parts[3] == "1",
            btn_b: parts[4] == "1",
            btn_w: parts[5] == "1",
            btn_start: parts[6] == "1",
        })
    }

    fn to_csv_line(&self) -> String {
        format!(
            "{},{},{},{},{},{},{}",
            self.duration,
            self.direction,
            if self.btn_a1 { 1 } else { 0 },
            if self.btn_a2 { 1 } else { 0 },
            if self.btn_b { 1 } else { 0 },
            if self.btn_w { 1 } else { 0 },
            if self.btn_start { 1 } else { 0 }
        )
    }
}

/// 抽出結果
#[cfg(all(feature = "gui", feature = "ml"))]
enum ExtractionResult {
    Progress(usize, usize),
    Complete(Vec<InputRecord>),
    Error(String),
}

/// アプリケーション状態
#[cfg(all(feature = "gui", feature = "ml"))]
struct InputEditorApp {
    config: AppConfig,
    records: Vec<InputRecord>,
    current_file: Option<PathBuf>,
    selected_row: Option<usize>,
    selected_rows: std::collections::HashSet<usize>,
    clipboard: Vec<InputRecord>,
    clipboard_marker: Option<usize>,
    status_message: String,
    backend_type: BackendType,
    model_path: Option<PathBuf>,
    max_video_duration_secs: u64,
    extraction_progress: Option<(usize, usize)>,
    show_duration_warning: bool,
    show_model_warning: bool,
    extraction_receiver: Option<std::sync::mpsc::Receiver<ExtractionResult>>,
    cancel_flag: Option<Arc<AtomicBool>>,
    
    // 解析領域設定ウィンドウ用フィールド
    show_region_settings: bool,
    region_preview_video: Option<PathBuf>,
    region_preview_frame: Option<egui::ColorImage>,
    region_preview_video_width: Option<u32>,
    region_preview_video_height: Option<u32>,
    region_preview_zoom_mode: bool,
    region_preview_zoom_scale: f32,
    region_preview_frame_number: u32,
    frame_preview_receiver: Option<std::sync::mpsc::Receiver<egui::ColorImage>>,
    
    // 学習データ生成ウィンドウ用フィールド
    show_training_data_generator: bool,
    training_video_path: Option<PathBuf>,
    training_output_dir: Option<PathBuf>,
    training_frame_interval: u32,  // フレーム間引き設定（デフォルト=1）
    training_progress: Option<(usize, usize)>,
    training_cancel_flag: Option<Arc<AtomicBool>>,
    training_progress_rx: Option<std::sync::mpsc::Receiver<(usize, usize)>>,
}

#[cfg(all(feature = "gui", feature = "ml"))]
impl Default for InputEditorApp {
    fn default() -> Self {
        let config = AppConfig::load_or_default();
        let backend_type = match config.device_type {
            DeviceType::Wgpu => BackendType::Gpu,
            DeviceType::Cpu => BackendType::Cpu,
        };
        let model_path = if std::path::Path::new(&config.model.model_path).exists() {
            Some(PathBuf::from(&config.model.model_path))
        } else {
            None
        };
        let training_output_dir = config.training_output_dir.as_ref().map(|s| PathBuf::from(s));

        Self {
            config,
            records: Vec::new(),
            current_file: None,
            selected_row: None,
            status_message: "モデルファイルを選択してください".to_string(),
            backend_type,
            model_path,
            max_video_duration_secs: 120, // デフォルト2分
            extraction_progress: None,
            show_duration_warning: false,
            show_model_warning: false,
            extraction_receiver: None,
            cancel_flag: None,
            selected_rows: std::collections::HashSet::new(),
            clipboard: Vec::new(),
            clipboard_marker: None,
            
            // 解析領域設定フィールド初期化
            show_region_settings: false,
            region_preview_video: None,
            region_preview_frame: None,
            region_preview_video_width: None,
            region_preview_video_height: None,
            region_preview_zoom_mode: false,
            region_preview_zoom_scale: 1.0,
            region_preview_frame_number: 0,
            frame_preview_receiver: None,
            
            // 学習データ生成フィールド初期化
            show_training_data_generator: false,
            training_video_path: None,
            training_output_dir,
            training_frame_interval: 1,  // デフォルト：全フレーム（間引きなし）
            training_progress: None,
            training_cancel_flag: None,
            training_progress_rx: None,
        }
    }
}

#[cfg(all(feature = "gui", feature = "ml"))]
impl InputEditorApp {
    fn save_config(&self) -> Result<(), String> {
        let config_path = "config.json";
        let json = serde_json::to_string_pretty(&self.config)
            .map_err(|e| format!("Configのシリアライズに失敗: {}", e))?;
        std::fs::write(config_path, json)
            .map_err(|e| format!("Configの保存に失敗: {}", e))?;
        Ok(())
    }
    
    fn load_csv(&mut self, path: PathBuf) -> Result<(), String> {
        let content =
            std::fs::read_to_string(&path).map_err(|e| format!("ファイル読み込みエラー: {}", e))?;

        let mut records = Vec::new();
        for (i, line) in content.lines().enumerate() {
            if i == 0 {
                continue; // ヘッダー行をスキップ
            }
            if let Some(record) = InputRecord::from_csv_line(line) {
                records.push(record);
            }
        }

        self.records = records;
        self.current_file = Some(path.clone());
        self.selected_row = None;
        self.selected_rows.clear();
        self.status_message = format!(
            "読み込み完了: {} ({} レコード)",
            path.display(),
            self.records.len()
        );

        self.config
            .update_last_output_dir(path.parent().unwrap_or(std::path::Path::new(".")));
        if let Err(e) = self.config.save_default() {
            eprintln!("警告: 設定ファイルの保存に失敗しました: {}", e);
        }
        Ok(())
    }

    fn save_csv(&mut self, path: &PathBuf) -> Result<(), String> {
        let mut content = String::from("duration,direction,A1,A2,B,W,Start\n");
        for record in &self.records {
            content.push_str(&record.to_csv_line());
            content.push('\n');
        }

        std::fs::write(path, content).map_err(|e| format!("ファイル保存エラー: {}", e))?;

        Ok(())
    }

    fn add_record(&mut self, index: Option<usize>) {
        let new_record = InputRecord::new();
        // 複数行選択時は一番下の選択行の下に追加
        let insert_idx = if !self.selected_rows.is_empty() {
            let max_idx = *self.selected_rows.iter().max().unwrap();
            max_idx + 1
        } else if let Some(idx) = index {
            idx + 1
        } else {
            self.records.len()
        };

        self.records.insert(insert_idx, new_record);
        self.selected_row = Some(insert_idx);
        self.selected_rows.clear();
        self.selected_rows.insert(insert_idx);
        self.status_message = "新しいレコードを追加しました".to_string();
    }

    fn delete_record(&mut self, index: usize) {
        if self.records.len() <= 1 {
            self.status_message = "入力履歴は最低1行必要です".to_string();
            return;
        }
        if index < self.records.len() {
            self.records.remove(index);
            self.selected_row = None;
            self.status_message = "レコードを削除しました".to_string();
        }
    }

    fn delete_selected(&mut self) {
        if self.selected_rows.is_empty() {
            self.status_message = "削除する行を選択してください".to_string();
            return;
        }

        if self.records.len() - self.selected_rows.len() < 1 {
            self.status_message = "最低1行は残す必要があります".to_string();
            return;
        }

        let mut indices: Vec<usize> = self.selected_rows.iter().copied().collect();
        indices.sort();
        indices.reverse();

        let count = indices.len();

        for idx in indices {
            if idx < self.records.len() {
                self.records.remove(idx);
            }
        }

        self.selected_rows.clear();
        self.selected_row = None;
        self.status_message = format!("{}行を削除しました", count);
    }

    fn new_document(&mut self) {
        self.records = vec![InputRecord::new()];
        self.current_file = None;
        self.selected_row = None;
        self.selected_rows.clear();
        self.status_message = "新規作成しました".to_string();
    }

    fn copy_selected(&mut self) {
        if self.selected_rows.is_empty() {
            self.status_message = "コピーする行を選択してください".to_string();
            return;
        }

        let mut indices: Vec<usize> = self.selected_rows.iter().copied().collect();
        indices.sort();

        self.clipboard.clear();
        for &idx in &indices {
            if idx < self.records.len() {
                self.clipboard.push(self.records[idx].clone());
            }
        }

        // 内部クリップボードのマーカーを保存（次回のupdateでシステムクリップボードに書き込む）
        self.clipboard_marker = Some(self.clipboard.len());

        self.status_message = format!("{}行をコピーしました", self.clipboard.len());
    }

    fn cut_selected(&mut self) {
        if self.selected_rows.is_empty() {
            self.status_message = "切り取る行を選択してください".to_string();
            return;
        }

        if self.records.len() - self.selected_rows.len() < 1 {
            self.status_message = "最低1行は残す必要があります".to_string();
            return;
        }

        let mut indices: Vec<usize> = self.selected_rows.iter().copied().collect();
        indices.sort();

        self.clipboard.clear();
        for &idx in &indices {
            if idx < self.records.len() {
                self.clipboard.push(self.records[idx].clone());
            }
        }

        // 逆順で削除
        for &idx in indices.iter().rev() {
            if idx < self.records.len() {
                self.records.remove(idx);
            }
        }

        // 内部クリップボードのマーカーを保存
        self.clipboard_marker = Some(self.clipboard.len());

        self.selected_rows.clear();
        self.selected_row = None;
        self.status_message = format!("{}行を切り取りました", self.clipboard.len());
    }

    fn paste(&mut self) {
        if self.clipboard.is_empty() {
            self.status_message = "クリップボードが空です".to_string();
            return;
        }

        let insert_pos = self
            .selected_row
            .map(|r| r + 1)
            .unwrap_or(self.records.len());

        for (i, record) in self.clipboard.iter().enumerate() {
            self.records.insert(insert_pos + i, record.clone());
        }

        self.status_message = format!("{}行を貼り付けました", self.clipboard.len());
    }

    fn select_all(&mut self) {
        self.selected_rows.clear();
        for i in 0..self.records.len() {
            self.selected_rows.insert(i);
        }
        self.status_message = format!("全{}行を選択しました", self.records.len());
    }

    fn extract_from_video(&mut self, video_path: PathBuf) -> Result<(), String> {
        // モデルが選択されているかチェック
        if self.model_path.is_none() {
            self.show_model_warning = true;
            return Err("モデルファイルを選択してください（設定メニュー）".to_string());
        }

        // 動画の長さをチェック
        let duration_secs = self.get_video_duration(&video_path)?;
        if duration_secs > self.max_video_duration_secs {
            self.show_duration_warning = true;
            return Err(format!(
                "動画が長すぎます: {}秒 (上限: {}秒)",
                duration_secs, self.max_video_duration_secs
            ));
        }

        self.status_message = format!("動画から抽出中: {}", video_path.display());
        self.extraction_progress = Some((0, 0));

        // キャンセルフラグを作成
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.cancel_flag = Some(cancel_flag.clone());

        // チャネルを作成
        let (tx, rx) = std::sync::mpsc::channel();
        self.extraction_receiver = Some(rx);

        let backend_type = self.backend_type;
        let model_path = self.model_path.clone().unwrap();

        // バックグラウンドスレッドで抽出処理を実行
        std::thread::spawn(move || {
            let result = match backend_type {
                BackendType::Gpu => Self::extract_from_video_impl_thread::<WgpuBackend>(
                    video_path,
                    model_path,
                    tx.clone(),
                    cancel_flag,
                ),
                BackendType::Cpu => Self::extract_from_video_impl_thread::<NdArrayBackend>(
                    video_path,
                    model_path,
                    tx.clone(),
                    cancel_flag,
                ),
            };

            // 結果を送信
            match result {
                Ok(records) => {
                    let _ = tx.send(ExtractionResult::Complete(records));
                }
                Err(e) => {
                    let _ = tx.send(ExtractionResult::Error(e));
                }
            }
        });

        Ok(())
    }

    fn cancel_extraction(&mut self) {
        if let Some(flag) = &self.cancel_flag {
            flag.store(true, Ordering::Relaxed);
        }
        self.extraction_progress = None;
        self.extraction_receiver = None;
        self.cancel_flag = None;
        self.status_message = "抽出をキャンセルしました".to_string();
    }

    fn extract_from_video_impl_thread<B: burn::tensor::backend::Backend>(
        video_path: PathBuf,
        model_path: PathBuf,
        tx: std::sync::mpsc::Sender<ExtractionResult>,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<Vec<InputRecord>, String> {
        use std::fs;

        let device = B::Device::default();

        // モデル読み込み
        let record = CompactRecorder::new()
            .load(model_path, &device)
            .map_err(|e| format!("モデル読み込みエラー: {}", e))?;

        let model = ModelConfig::new(CLASS_NAMES.len())
            .init::<B>(&device)
            .load_record(record);

        // 一時ディレクトリ
        let temp_dir = std::path::PathBuf::from("temp_extract_gui");
        let temp_frames_dir = std::path::PathBuf::from("temp_frames_gui");
        fs::create_dir_all(&temp_dir).map_err(|e| format!("ディレクトリ作成エラー: {}", e))?;
        fs::create_dir_all(&temp_frames_dir)
            .map_err(|e| format!("ディレクトリ作成エラー: {}", e))?;

        // フレーム抽出
        let config = input_analyzer::frame_extractor::FrameExtractorConfig {
            frame_interval: 1,
            output_dir: temp_frames_dir.clone(),
            image_format: "png".to_string(),
            jpeg_quality: 95,
        };

        let extractor = FrameExtractor::new(config);
        let frame_paths = extractor
            .extract_frames(&video_path)
            .map_err(|e| format!("フレーム抽出エラー: {}", e))?;

        // 入力履歴抽出
        let mut records = Vec::new();
        let mut current_state: Option<InputState> = None;
        let mut duration = 0u32;
        let total_frames = frame_paths.len();

        for (frame_idx, frame_path) in frame_paths.iter().enumerate() {
            // キャンセルチェック
            if cancel_flag.load(Ordering::Relaxed) {
                fs::remove_dir_all(&temp_dir).ok();
                fs::remove_dir_all(&temp_frames_dir).ok();
                return Err("キャンセルされました".to_string());
            }

            // 進捗を送信
            let _ = tx.send(ExtractionResult::Progress(frame_idx + 1, total_frames));

            let state =
                Self::extract_state_from_frame_static::<B>(frame_path, &model, &device, &temp_dir)
                    .map_err(|e| format!("フレーム処理エラー: {}", e))?;

            if let Some(ref prev_state) = current_state {
                if &state == prev_state {
                    duration += 1;
                } else {
                    records.push(Self::state_to_record_static(prev_state, duration));
                    current_state = Some(state);
                    duration = 1;
                }
            } else {
                current_state = Some(state);
                duration = 1;
            }
        }

        // 最後の入力を記録
        if let Some(ref state) = current_state {
            records.push(Self::state_to_record_static(state, duration));
        }

        // 一時ディレクトリを削除
        fs::remove_dir_all(&temp_dir).ok();
        fs::remove_dir_all(&temp_frames_dir).ok();

        Ok(records)
    }

    fn get_video_duration(&self, video_path: &std::path::Path) -> Result<u64, String> {
        // GStreamerを使って動画の長さを取得
        let video_info = FrameExtractor::get_video_info(video_path)
            .map_err(|e| format!("動画情報の取得エラー: {}", e))?;

        Ok(video_info.duration_sec.ceil() as u64)
    }

    fn extract_state_from_frame_static<B: burn::tensor::backend::Backend>(
        frame_path: &std::path::Path,
        model: &IconClassifier<B>,
        device: &B::Device,
        temp_dir: &std::path::Path,
    ) -> anyhow::Result<InputState> {
        use std::fs;

        let mut state = InputState::new();
        let icons = extract_bottom_row_icons(frame_path)?;

        for (icon_idx, icon_img) in icons.iter().enumerate() {
            let temp_icon_path = temp_dir.join(format!("temp_icon_{}.png", icon_idx));
            icon_img.save(&temp_icon_path)?;

            // 分類
            let image_data = load_and_normalize_image(&temp_icon_path)?;
            let tensor =
                Tensor::<B, 1>::from_floats(image_data.as_slice(), device).reshape([1, 3, 48, 48]);
            let (predictions, _) = model.predict(tensor);
            let class_id = predictions.into_data().to_vec::<i32>().unwrap()[0] as usize;
            let class_name = CLASS_NAMES[class_id];

            update_input_state(&mut state, class_name);
            fs::remove_file(&temp_icon_path)?;
        }

        Ok(state)
    }

    fn state_to_record_static(state: &InputState, duration: u32) -> InputRecord {
        InputRecord {
            duration,
            direction: state.direction,
            btn_a1: state.btn_a1 == 1,
            btn_a2: state.btn_a2 == 1,
            btn_b: state.btn_b == 1,
            btn_w: state.btn_w == 1,
            btn_start: state.btn_start == 1,
        }
    }
}

/// 方向値を矢印文字列に変換
#[cfg(all(feature = "gui", feature = "ml"))]
fn direction_to_arrow(direction: u8) -> &'static str {
    match direction {
        1 => "↙", // 左下
        2 => "↓", // 下
        3 => "↘", // 右下
        4 => "←", // 左
        5 => "N", // ニュートラル
        6 => "→", // 右
        7 => "↖", // 左上
        8 => "↑", // 上
        9 => "↗", // 右上
        _ => "?",
    }
}

#[cfg(all(feature = "gui", feature = "ml"))]
impl eframe::App for InputEditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 学習データ生成のプログレスを更新
        if let Some(rx) = &self.training_progress_rx {
            if let Ok((current, total)) = rx.try_recv() {
                self.training_progress = Some((current, total));
                
                // 完了判定
                if current >= total {
                    // 完了したのでクリーンアップ
                    self.training_progress = None;
                    self.training_cancel_flag = None;
                    self.training_progress_rx = None;
                }
            }
        }
        
        // 抽出結果をチェック
        if let Some(ref rx) = self.extraction_receiver {
            if let Ok(result) = rx.try_recv() {
                match result {
                    ExtractionResult::Progress(current, total) => {
                        self.extraction_progress = Some((current, total));
                        ctx.request_repaint();
                    }
                    ExtractionResult::Complete(records) => {
                        self.records = records;
                        self.extraction_progress = None;
                        self.extraction_receiver = None;
                        self.cancel_flag = None;
                        self.selected_row = None;
                        self.selected_rows.clear();
                        self.status_message = format!("抽出完了: {} レコード", self.records.len());
                    }
                    ExtractionResult::Error(e) => {
                        self.extraction_progress = None;
                        self.extraction_receiver = None;
                        self.cancel_flag = None;
                        self.status_message = format!("抽出エラー: {}", e);
                    }
                }
            }
        }

        // キーボードショートカット
        let wants_keyboard = ctx.wants_keyboard_input();

        // Copy/Cut/Pasteイベントを処理
        let events = ctx.input(|i| i.events.clone());

        if !wants_keyboard {
            // フォーカスがない場合のみアプリケーションのショートカットを有効化
            for event in &events {
                match event {
                    egui::Event::Copy => {
                        self.copy_selected();
                    }
                    egui::Event::Cut => {
                        self.cut_selected();
                    }
                    egui::Event::Paste(_) => {
                        self.paste();
                    }
                    egui::Event::Key {
                        key,
                        pressed,
                        modifiers,
                        ..
                    } => {
                        if *pressed {
                            // Ctrl+A (Select All)
                            if *key == egui::Key::A
                                && modifiers.ctrl
                                && !modifiers.shift
                                && !modifiers.alt
                            {
                                self.select_all();
                            }
                            // Delete
                            else if *key == egui::Key::Delete
                                && !modifiers.ctrl
                                && !modifiers.shift
                                && !modifiers.alt
                            {
                                self.delete_selected();
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // clipboard_markerが設定されていたらシステムクリップボードに書き込む
        if let Some(count) = self.clipboard_marker.take() {
            ctx.output_mut(|o| {
                o.copied_text = format!("__INTERNAL_CLIPBOARD__{}", count);
            });
        }

        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("ファイル", |ui| {
                    if ui.button("新規作成").clicked() {
                        self.new_document();
                        ui.close_menu();
                    }

                    ui.separator();

                    if ui.button("開く (CSV/動画)").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("対応ファイル", &["csv", "mp4", "avi", "mov", "mkv"])
                            .add_filter("CSV", &["csv"])
                            .add_filter("動画", &["mp4", "avi", "mov", "mkv"])
                            .pick_file()
                        {
                            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                            match ext.to_lowercase().as_str() {
                                "csv" => {
                                    if let Err(e) = self.load_csv(path) {
                                        self.status_message = e;
                                    }
                                }
                                "mp4" | "avi" | "mov" | "mkv" => {
                                    if let Err(e) = self.extract_from_video(path) {
                                        self.status_message = format!("抽出エラー: {}", e);
                                    }
                                }
                                _ => {
                                    self.status_message =
                                        "対応していないファイル形式です".to_string();
                                }
                            }
                        }
                        ui.close_menu();
                    }

                    ui.separator();

                    if ui.button("保存").clicked() {
                        if let Some(path) = self.current_file.clone() {
                            if let Err(e) = self.save_csv(&path) {
                                self.status_message = e;
                            } else {
                                self.status_message = format!("保存しました: {}", path.display());
                            }
                        } else {
                            self.status_message =
                                "保存先を指定してください（名前を付けて保存）".to_string();
                        }
                        ui.close_menu();
                    }

                    if ui.button("名前を付けて保存").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("CSV", &["csv"])
                            .save_file()
                        {
                            if let Err(e) = self.save_csv(&path) {
                                self.status_message = e;
                            } else {
                                self.current_file = Some(path.clone());
                                self.status_message = format!("保存しました: {}", path.display());
                            }
                        }
                        ui.close_menu();
                    }

                    ui.separator();

                    if ui.button("終了").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("編集", |ui| {
                    if ui.button("コピー (Ctrl+C)").clicked() {
                        self.copy_selected();
                        ui.close_menu();
                    }

                    if ui.button("切り取り (Ctrl+X)").clicked() {
                        self.cut_selected();
                        ui.close_menu();
                    }

                    if ui.button("貼り付け (Ctrl+V)").clicked() {
                        self.paste();
                        ui.close_menu();
                    }

                    ui.separator();

                    if ui.button("すべて選択 (Ctrl+A)").clicked() {
                        self.select_all();
                        ui.close_menu();
                    }

                    ui.separator();

                    if ui.button("新規レコード追加").clicked() {
                        self.add_record(self.selected_row);
                        ui.close_menu();
                    }

                    let can_delete = self.records.len() > 1
                        && (self.records.len() - self.selected_rows.len() >= 1);
                    ui.add_enabled_ui(can_delete, |ui| {
                        if ui.button("選択レコード削除 (Del)").clicked() {
                            self.delete_selected();
                            ui.close_menu();
                        }
                    });
                    if !can_delete && self.records.len() <= 1 {
                        ui.label("（最低1行必要）");
                    }
                });

                ui.menu_button("設定", |ui| {
                    if ui.button("解析領域設定").clicked() {
                        self.show_region_settings = true;
                        ui.close_menu();
                    }
                    
                    if ui.button("学習データ生成").clicked() {
                        self.show_training_data_generator = true;
                        ui.close_menu();
                    }

                    ui.separator();

                    ui.label("推論バックエンド:");
                    if ui
                        .radio_value(&mut self.backend_type, BackendType::Gpu, "GPU (WGPU)")
                        .clicked()
                    {
                        self.config.set_device_type(DeviceType::Wgpu);
                        let _ = self.config.save_default();
                    }
                    if ui
                        .radio_value(&mut self.backend_type, BackendType::Cpu, "CPU (NdArray)")
                        .clicked()
                    {
                        self.config.set_device_type(DeviceType::Cpu);
                        let _ = self.config.save_default();
                    }

                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.label("モデルファイル:");
                        if ui.button("選択...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("モデル", &["mpk"])
                                .pick_file()
                            {
                                self.model_path = Some(path);
                                self.status_message = "モデルを読み込みました".to_string();
                            }
                        }
                    });
                    if let Some(ref path) = self.model_path {
                        ui.label(format!("現在: {}", path.display()));
                    } else {
                        ui.colored_label(egui::Color32::RED, "未選択（動画抽出不可）");
                    }

                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.label("動画長さ上限 (秒):");
                        ui.add(
                            egui::DragValue::new(&mut self.max_video_duration_secs)
                                .speed(1.0)
                                .range(10..=600),
                        );
                    });
                    ui.label(format!(
                        "現在: {}秒 ({}:{:02})",
                        self.max_video_duration_secs,
                        self.max_video_duration_secs / 60,
                        self.max_video_duration_secs % 60
                    ));
                });
            });
        });

        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status_message);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("総レコード数: {}", self.records.len()));
                    ui.separator();
                    let backend_text = match self.backend_type {
                        BackendType::Gpu => "GPU",
                        BackendType::Cpu => "CPU",
                    };
                    ui.label(format!("バックエンド: {}", backend_text));
                });
            });
        });

        // モデル未選択警告ダイアログ
        if self.show_model_warning {
            egui::Window::new("警告")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(10.0);
                        ui.label(
                            egui::RichText::new("⚠ モデルファイルが選択されていません")
                                .size(16.0)
                                .color(egui::Color32::from_rgb(255, 150, 0)),
                        );
                        ui.add_space(10.0);
                        ui.label("動画から入力履歴を抽出するには、");
                        ui.label("機械学習モデルファイルを選択する必要があります。");
                        ui.add_space(10.0);
                        ui.label("「設定」メニュー → 「モデルファイルを選択」");
                        ui.label("からモデルファイルを指定してください。");
                        ui.add_space(15.0);
                        if ui.button("OK").clicked() {
                            self.show_model_warning = false;
                        }
                    });
                });
        }

        // 動画長すぎ警告ダイアログ
        if self.show_duration_warning {
            egui::Window::new("警告")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(10.0);
                        ui.label("動画が長すぎます");
                        ui.add_space(5.0);
                        ui.label(format!(
                            "上限: {}秒 ({}:{:02})",
                            self.max_video_duration_secs,
                            self.max_video_duration_secs / 60,
                            self.max_video_duration_secs % 60
                        ));
                        ui.add_space(5.0);
                        ui.label("設定メニューから上限を変更できます。");
                        ui.add_space(10.0);
                        if ui.button("OK").clicked() {
                            self.show_duration_warning = false;
                        }
                    });
                });
        }

        // プログレスバー
        let mut should_cancel = false;
        if let Some((current, total)) = self.extraction_progress {
            egui::Window::new("抽出中")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(10.0);
                        ui.label(format!("フレーム処理中: {} / {}", current, total));
                        ui.add_space(5.0);
                        let progress = current as f32 / total as f32;
                        ui.add(
                            egui::ProgressBar::new(progress)
                                .show_percentage()
                                .animate(true),
                        );
                        ui.add_space(10.0);
                        if ui.button("キャンセル").clicked() {
                            should_cancel = true;
                        }
                        ui.add_space(5.0);
                    });
                });
            ctx.request_repaint();
        }

        if should_cancel {
            self.cancel_extraction();
        }

        // 解析領域設定ウィンドウ
        if self.show_region_settings {
            let mut is_open = true;
            egui::Window::new("解析領域設定")
                .open(&mut is_open)
                .resizable(true)
                .vscroll(true)
                .id(egui::Id::new("region_settings_window"))
                .show(ctx, |ui| {
                    ui.heading("ゲーム画面解析領域の設定");
                    
                    // ビデオファイル選択
                    ui.label("📹 ビデオファイル:");
                    ui.horizontal(|ui| {
                        if let Some(path) = &self.region_preview_video {
                            ui.label(format!("選択: {}", path.display()));
                        } else {
                            ui.label("未選択");
                        }
                        
                        if ui.button("ファイルを選択...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("動画", &["mp4", "avi", "mov"])
                                .pick_file()
                            {
                                self.region_preview_video = Some(path.clone());
                                // フレーム0を抽出
                                if let Ok(info) = FrameExtractor::get_video_info(&path) {
                                    self.region_preview_video_width = Some(info.width as u32);
                                    self.region_preview_video_height = Some(info.height as u32);
                                    
                                    let (tx, rx) = std::sync::mpsc::channel();
                                    self.frame_preview_receiver = Some(rx);
                                    
                                    let path_clone = path.clone();
                                    std::thread::spawn(move || {
                                        let extractor = FrameExtractor::default();
                                        match extractor.extract_frame_at(&path_clone, 0) {
                                            Ok(frame_path) => {
                                                // フレーム画像ファイルを読み込み
                                                if let Ok(image_buf) = std::fs::read(&frame_path) {
                                                    if let Ok(image) = image::load_from_memory(&image_buf) {
                                                        let rgba_image = image.to_rgba8();
                                                        let width = rgba_image.width() as usize;
                                                        let height = rgba_image.height() as usize;
                                                        
                                                        let color_image = egui::ColorImage {
                                                            size: [width, height],
                                                            pixels: rgba_image
                                                                .pixels()
                                                                .map(|p| egui::Color32::from_rgba_unmultiplied(
                                                                    p[0], p[1], p[2], p[3],
                                                                ))
                                                                .collect(),
                                                        };
                                                        let _ = tx.send(color_image);
                                                    }
                                                }
                                                // 一時ファイルを削除
                                                let _ = std::fs::remove_file(&frame_path);
                                            }
                                            Err(e) => {
                                                eprintln!("フレーム抽出エラー: {}", e);
                                            }
                                        }
                                    });
                                }
                            }
                        }
                    });
                    ui.separator();
                    
                    // フレーム選択UI
                    if self.region_preview_video.is_some() {
                        ui.horizontal(|ui| {
                            ui.label("表示フレーム:");
                            ui.add(
                                egui::DragValue::new(&mut self.region_preview_frame_number)
                                    .range(0..=u32::MAX)
                                    .speed(1.0),
                            );
                            
                            if ui.button("🔄 更新").clicked() {
                                // フレーム再抽出
                                if let Some(ref video_path) = self.region_preview_video.clone() {
                                    let (tx, rx) = std::sync::mpsc::channel();
                                    self.frame_preview_receiver = Some(rx);
                                    // フレームをクリア
                                    self.region_preview_frame = None;
                                    
                                    let path_clone = video_path.clone();
                                    let frame_num = self.region_preview_frame_number;
                                    std::thread::spawn(move || {
                                        let extractor = FrameExtractor::default();
                                        match extractor.extract_frame_at(&path_clone, frame_num) {
                                            Ok(frame_path) => {
                                                if let Ok(image_buf) = std::fs::read(&frame_path) {
                                                    if let Ok(image) = image::load_from_memory(&image_buf) {
                                                        let rgba_image = image.to_rgba8();
                                                        let width = rgba_image.width() as usize;
                                                        let height = rgba_image.height() as usize;
                                                        
                                                        let color_image = egui::ColorImage {
                                                            size: [width, height],
                                                            pixels: rgba_image
                                                                .pixels()
                                                                .map(|p| egui::Color32::from_rgba_unmultiplied(
                                                                    p[0], p[1], p[2], p[3],
                                                                ))
                                                                .collect(),
                                                        };
                                                        let _ = tx.send(color_image);
                                                    }
                                                }
                                                let _ = std::fs::remove_file(&frame_path);
                                            }
                                            Err(e) => {
                                                eprintln!("フレーム抽出エラー: {}", e);
                                            }
                                        }
                                    });
                                }
                            }
                        });
                    }
                    
                    ui.separator();
                    
                    // フレームプレビュー
                    if let Some(ref frame) = self.region_preview_frame {
                        let texture = ctx.load_texture(
                            format!("region_preview_frame_{}", std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_nanos()),
                            egui::ImageData::Color(std::sync::Arc::new(frame.clone())),
                            Default::default(),
                        );
                        
                        // ズーム倍率を反映したサイズを計算
                        let zoom_scale = if self.region_preview_zoom_mode {
                            self.region_preview_zoom_scale
                        } else {
                            1.0
                        };
                        
                        let img_width = frame.width() as f32;
                        let img_height = frame.height() as f32;
                        let max_dim = img_width.max(img_height);
                        let base_scale = 400.0 / max_dim;
                        
                        // ズーム倍率を含めた最終的なスケール
                        let final_scale = base_scale * zoom_scale;
                        let scaled_w = img_width * final_scale;
                        let scaled_h = img_height * final_scale;
                        
                        ui.label("📺 プレビュー（タイル位置表示）:");
                        
                        let image_response = ui.image(egui::load::SizedTexture::new(
                            texture.id(),
                            [scaled_w, scaled_h],
                        ));
                        
                        // タイルオーバーレイ描画
                        if self.region_preview_video_width.is_some() && self.region_preview_video_height.is_some() {
                            let painter = ui.painter_at(image_response.rect);
                            self.paint_tile_overlay(
                                &painter,
                                image_response.rect,
                                scaled_w,
                                scaled_h,
                                zoom_scale,
                            );
                        }
                    } else if self.frame_preview_receiver.is_some() {
                        // フレーム受信待機中
                        if let Some(ref mut rx) = self.frame_preview_receiver {
                            match rx.try_recv() {
                                Ok(frame) => {
                                    self.region_preview_frame = Some(frame);
                                }
                                Err(_) => {
                                    ui.label("フレーム読み込み中...");
                                }
                            }
                        }
                    }
                    
                    ui.separator();
                    
                    // タイル設定
                    ui.collapsing("タイル/ボタン位置設定", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("X座標:");
                            ui.add(
                                egui::DragValue::new(&mut self.config.button_tile.x)
                                    .range(0..=1920)
                                    .speed(1.0),
                            );
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Y座標:");
                            ui.add(
                                egui::DragValue::new(&mut self.config.button_tile.y)
                                    .range(0..=1080)
                                    .speed(1.0),
                            );
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("幅:");
                            ui.add(
                                egui::DragValue::new(&mut self.config.button_tile.width)
                                    .range(1..=1920)
                                    .speed(1.0),
                            );
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("高さ:");
                            ui.add(
                                egui::DragValue::new(&mut self.config.button_tile.height)
                                    .range(1..=1080)
                                    .speed(1.0),
                            );
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("行あたりの列数:");
                            ui.add(
                                egui::DragValue::new(&mut self.config.button_tile.columns_per_row)
                                    .range(1..=16)
                                    .speed(1.0),
                            );
                        });
                    });
                    
                    ui.separator();
                    
                    // ズーム設定
                    ui.checkbox(&mut self.region_preview_zoom_mode, "ズーム表示を有効にする");
                    
                    if self.region_preview_zoom_mode {
                        ui.horizontal(|ui| {
                            ui.label("ズーム倍率:");
                            ui.add(
                                egui::Slider::new(&mut self.region_preview_zoom_scale, 1.0..=4.0)
                                    .show_value(true),
                            );
                        });
                    }
                    
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("保存して閉じる").clicked() {
                            // AppConfigを保存
                            if let Ok(json_str) = serde_json::to_string_pretty(&self.config) {
                                let _ = std::fs::write("config.json", json_str);
                            }
                            self.show_region_settings = false;
                        }
                        
                        if ui.button("キャンセル").clicked() {
                            self.show_region_settings = false;
                        }
                    });
                });
            
            if !is_open {
                self.show_region_settings = false;
            }
        }

        // 学習データ生成ウィンドウ
        if self.show_training_data_generator {
            let mut is_open = true;
            egui::Window::new("学習データ生成")
                .open(&mut is_open)
                .resizable(true)
                .vscroll(true)
                .id(egui::Id::new("training_data_generator_window"))
                .show(ctx, |ui| {
                    ui.heading("動画からタイル画像を抽出");
                    
                    ui.label("📹 ビデオファイル:");
                    ui.horizontal(|ui| {
                        if let Some(path) = &self.training_video_path {
                            ui.label(format!("選択: {}", path.display()));
                        } else {
                            ui.label("未選択");
                        }
                        
                        if ui.button("ファイルを選択...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("動画", &["mp4", "avi", "mov"])
                                .pick_file()
                            {
                                self.training_video_path = Some(path.clone());
                            }
                        }
                    });
                    
                    ui.separator();
                    
                    ui.label("📁 出力フォルダ:");
                    ui.horizontal(|ui| {
                        if let Some(path) = &self.training_output_dir {
                            ui.label(format!("選択: {}", path.display()));
                        } else {
                            ui.label("未選択");
                        }
                        
                        if ui.button("フォルダを選択...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .pick_folder()
                            {
                                self.training_output_dir = Some(path);
                            }
                        }
                    });
                    
                    ui.separator();
                    
                    ui.label("フレーム間引き設定:");
                    ui.horizontal(|ui| {
                        ui.label("n フレームおきに抽出:");
                        ui.add(
                            egui::DragValue::new(&mut self.training_frame_interval)
                                .range(1..=120)
                                .speed(1.0),
                        );
                    });
                    ui.label(format!("💡 ヒント: {}フレーム間隔で抽出します (1=全フレーム, 2=2フレームごと)", self.training_frame_interval));
                    
                    ui.separator();
                    
                    // 進捗表示
                    if let Some((current, total)) = self.training_progress {
                        ui.label(format!("処理中: {} / {} フレーム", current, total));
                        let progress = current as f32 / total as f32;
                        ui.add(
                            egui::ProgressBar::new(progress)
                                .show_percentage()
                                .animate(true),
                        );
                        ui.add_space(10.0);
                        if ui.button("キャンセル").clicked() {
                            if let Some(flag) = &self.training_cancel_flag {
                                flag.store(true, Ordering::Relaxed);
                            }
                        }
                    } else {
                        // 処理開始ボタン
                        if ui.button("🚀 タイル画像を抽出開始").clicked() {
                            if self.training_video_path.is_some() && self.training_output_dir.is_some() {
                                let video_path = self.training_video_path.clone().unwrap();
                                let output_dir = self.training_output_dir.clone().unwrap();
                                let config = self.config.clone();
                                let frame_interval = self.training_frame_interval;
                                
                                // 出力フォルダをconfigに保存
                                self.config.training_output_dir = Some(output_dir.to_string_lossy().to_string());
                                let _ = self.save_config();
                                
                                let cancel_flag = Arc::new(AtomicBool::new(false));
                                self.training_cancel_flag = Some(cancel_flag.clone());
                                
                                // プログレス更新用チャンネル作成
                                let (progress_tx, progress_rx) = std::sync::mpsc::channel::<(usize, usize)>();
                                self.training_progress_rx = Some(progress_rx);
                                
                                self.training_progress = Some((0, 1));
                                
                                std::thread::spawn(move || {
                                    eprintln!("🟢 学習データ生成スレッド開始");
                                    extract_tile_images(
                                        &video_path,
                                        &output_dir,
                                        &config,
                                        frame_interval,
                                        cancel_flag,
                                        progress_tx,
                                    );
                                    eprintln!("🟢 学習データ生成スレッド終了");
                                });
                            }
                        }
                    }
                    
                    ui.separator();
                    if ui.button("閉じる").clicked() {
                        self.show_training_data_generator = false;
                    }
                });
            
            if !is_open {
                self.show_training_data_generator = false;
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("サイバーボッツ入力履歴エディタ");
            ui.separator();

            if self.records.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(100.0);
                    ui.label("データがありません");
                    ui.label("「ファイル」メニューからCSVを開くか、動画から抽出してください");
                });
                return;
            }

            // ヘッダー（固定表示）
            egui::Grid::new("input_grid_header")
                .num_columns(10)
                .show(ui, |ui| {
                    ui.label("選択");
                    ui.label("持続F");
                    ui.label("方向");
                    ui.label("A1");
                    ui.label("A2");
                    ui.label("B");
                    ui.label("W");
                    ui.label("Start");
                    ui.label("挿入");
                    ui.label("削除");
                    ui.end_row();
                });

            ui.separator();

            // スクロール可能なデータ領域
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    egui::Grid::new("input_grid")
                        .striped(true)
                        .num_columns(10)
                        .show(ui, |ui| {
                            // レコード
                            let mut action: Option<(usize, &str)> = None;
                            let total_records = self.records.len();
                            let can_delete = total_records > 1;

                            for (i, record) in self.records.iter_mut().enumerate() {
                                let is_selected = self.selected_rows.contains(&i);

                                let response =
                                    ui.selectable_label(is_selected, format!("{}", i + 1));

                                if response.clicked() {
                                    let modifiers = ui.input(|i| i.modifiers);
                                    if modifiers.ctrl {
                                        // Ctrl+クリック: トグル選択
                                        if self.selected_rows.contains(&i) {
                                            self.selected_rows.remove(&i);
                                        } else {
                                            self.selected_rows.insert(i);
                                        }
                                    } else if modifiers.shift && self.selected_row.is_some() {
                                        // Shift+クリック: 範囲選択
                                        let start = self.selected_row.unwrap().min(i);
                                        let end = self.selected_row.unwrap().max(i);
                                        for idx in start..=end {
                                            self.selected_rows.insert(idx);
                                        }
                                    } else {
                                        // 通常クリック: 単一選択
                                        self.selected_rows.clear();
                                        self.selected_rows.insert(i);
                                    }
                                    self.selected_row = Some(i);
                                }

                                ui.add(
                                    egui::DragValue::new(&mut record.duration).range(1..=u32::MAX),
                                );

                                egui::ComboBox::from_id_salt(format!("dir_{}", i))
                                    .selected_text(direction_to_arrow(record.direction))
                                    .width(40.0)
                                    .show_ui(ui, |ui| {
                                        ui.style_mut().spacing.item_spacing.x = 2.0;
                                        for dir in 1..=9 {
                                            let arrow = direction_to_arrow(dir);
                                            ui.selectable_value(&mut record.direction, dir, arrow);
                                        }
                                    });

                                ui.checkbox(&mut record.btn_a1, "");
                                ui.checkbox(&mut record.btn_a2, "");
                                ui.checkbox(&mut record.btn_b, "");
                                ui.checkbox(&mut record.btn_w, "");
                                ui.checkbox(&mut record.btn_start, "");

                                // 挿入ボタン
                                if ui.button("➕").on_hover_text("この行の後に挿入").clicked()
                                {
                                    action = Some((i, "insert"));
                                }

                                // 削除ボタン
                                ui.add_enabled_ui(can_delete, |ui| {
                                    if ui
                                        .button("❌")
                                        .on_hover_text(if can_delete {
                                            "この行を削除"
                                        } else {
                                            "最低1行必要"
                                        })
                                        .clicked()
                                    {
                                        action = Some((i, "delete"));
                                    }
                                });

                                ui.end_row();
                            }

                            // 処理を実行
                            if let Some((idx, act)) = action {
                                match act {
                                    "insert" => self.add_record(Some(idx)),
                                    "delete" => self.delete_record(idx),
                                    _ => {}
                                }
                            }
                        });
                });
        });
    }
}

#[cfg(all(feature = "gui", feature = "ml"))]
impl InputEditorApp {
    /// タイル（ボタン位置）をプレビュー上に描画
    fn paint_tile_overlay(&self, painter: &egui::Painter, rect: egui::Rect, display_w: f32, display_h: f32, zoom_scale: f32) {
        if let (Some(orig_w), Some(orig_h)) = (self.region_preview_video_width, self.region_preview_video_height) {
            let orig_w = orig_w as f32;
            let orig_h = orig_h as f32;
            
            // 元画像座標から表示座標へのスケーリング係数
            let scale_x = display_w / orig_w;
            let scale_y = display_h / orig_h;
            
            // タイル位置とサイズを計算
            let tile_x = rect.left() + (self.config.button_tile.x as f32 * scale_x);
            let tile_y = rect.top() + (self.config.button_tile.y as f32 * scale_y);
            let tile_w = self.config.button_tile.width as f32 * scale_x;
            let tile_h = self.config.button_tile.height as f32 * scale_y;
            
            // タイルを描画（columns_per_row個）
            for i in 0..self.config.button_tile.columns_per_row {
                let x = tile_x + (tile_w * i as f32);
                let tile_rect = egui::Rect::from_min_size(
                    egui::pos2(x, tile_y),
                    egui::vec2(tile_w, tile_h)
                );
                
                // 枠を描画
                painter.rect_stroke(
                    tile_rect,
                    0.0,
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 0, 0))
                );
                
                // タイル番号を表示
                painter.text(
                    tile_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    format!("{}", i + 1),
                    egui::FontId::proportional(12.0),
                    egui::Color32::YELLOW
                );
            }
        }
    }
}

#[cfg(all(feature = "gui", feature = "ml"))]
fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([720.0, 800.0])
            .with_title("サイバーボッツ入力履歴エディタ"),
        ..Default::default()
    };

    eframe::run_native(
        "サイバーボッツ入力履歴エディタ",
        options,
        Box::new(|cc| {
            // 日本語フォント設定
            setup_japanese_fonts(&cc.egui_ctx);

            Ok(Box::new(InputEditorApp::default()))
        }),
    )
}

#[cfg(all(feature = "gui", feature = "ml"))]
fn setup_japanese_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    // Windowsシステムフォントを試行
    #[cfg(target_os = "windows")]
    {
        let font_paths = vec![
            "C:\\Windows\\Fonts\\meiryo.ttc",   // メイリオ
            "C:\\Windows\\Fonts\\msgothic.ttc", // MS ゴシック
            "C:\\Windows\\Fonts\\msmincho.ttc", // MS 明朝
            "C:\\Windows\\Fonts\\yugothic.ttf", // 游ゴシック
        ];

        for (i, font_path) in font_paths.iter().enumerate() {
            if let Ok(font_data) = std::fs::read(font_path) {
                let font_name = format!("japanese_font_{}", i);
                fonts.font_data.insert(
                    font_name.clone(),
                    egui::FontData::from_owned(font_data).into(),
                );

                fonts
                    .families
                    .entry(egui::FontFamily::Proportional)
                    .or_default()
                    .insert(0, font_name.clone());

                fonts
                    .families
                    .entry(egui::FontFamily::Monospace)
                    .or_default()
                    .push(font_name);

                break; // 最初に見つかったフォントを使用
            }
        }
    }

    ctx.set_fonts(fonts);
}

#[cfg(all(feature = "gui", feature = "ml"))]
fn extract_and_process_tiles_streaming(
    video_path: &PathBuf,
    output_dir: &PathBuf,
    video_name: &str,
    tile_pos_x: u32,
    tile_pos_y: u32,
    tile_width: u32,
    tile_height: u32,
    columns: u32,
    frame_interval: u32,
    cancel_flag: Arc<AtomicBool>,
    progress_sender: std::sync::mpsc::Sender<(usize, usize)>,
) -> Result<(), String> {
    use gstreamer::prelude::*;
    use gstreamer::{self as gst, ElementFactory};
    use gstreamer_app::AppSink;
    
    // GStreamer初期化
    gst::init().map_err(|e| format!("GStreamer初期化失敗: {}", e))?;
    
    // 動画情報取得
    let info = FrameExtractor::get_video_info(video_path)
        .map_err(|e| format!("動画情報の取得に失敗: {}", e))?;
    
    let total_frames = (info.duration_sec * info.fps) as usize;
    let estimated_extracts = (total_frames / frame_interval as usize).max(1);
    
    eprintln!("動画情報: {}x{}, {:.2}fps, {:.2}秒", info.width, info.height, info.fps, info.duration_sec);
    eprintln!("推定フレーム数: {}, 推定抽出数: {}", total_frames, estimated_extracts);
    
    let _ = progress_sender.send((0, estimated_extracts));
    
    // GStreamerパイプライン構築
    let pipeline = gst::Pipeline::new();
    
    let source = ElementFactory::make("filesrc")
        .name("source")
        .build()
        .map_err(|e| format!("filesrc作成失敗: {}", e))?;
    
    let decodebin = ElementFactory::make("decodebin")
        .name("decoder")
        .build()
        .map_err(|e| format!("decodebin作成失敗: {}", e))?;
    
    let videoconvert = ElementFactory::make("videoconvert")
        .name("converter")
        .build()
        .map_err(|e| format!("videoconvert作成失敗: {}", e))?;
    
    let appsink = ElementFactory::make("appsink")
        .name("sink")
        .build()
        .map_err(|e| format!("appsink作成失敗: {}", e))?;
    
    let appsink = appsink
        .dynamic_cast::<AppSink>()
        .map_err(|_| "appsinkへのキャスト失敗".to_string())?;
    
    // AppSink設定
    appsink.set_caps(Some(
        &gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .build(),
    ));
    appsink.set_property("emit-signals", false);
    appsink.set_property("sync", false);
    
    // ファイルパス設定
    source.set_property("location", video_path.to_str().unwrap());
    
    // パイプライン構築
    pipeline
        .add_many(&[&source, &decodebin, &videoconvert, appsink.upcast_ref::<gst::Element>()])
        .map_err(|e| format!("エレメント追加失敗: {}", e))?;
    
    source
        .link(&decodebin)
        .map_err(|e| format!("sourceとdecoderのリンク失敗: {}", e))?;
    
    videoconvert
        .link(appsink.upcast_ref::<gst::Element>())
        .map_err(|e| format!("converterとsinkのリンク失敗: {}", e))?;
    
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
    
    // フレームカウンタと抽出カウンタ
    let frame_count = Arc::new(std::sync::Mutex::new(0u32));
    let extracted_count = Arc::new(std::sync::Mutex::new(0usize));
    
    let frame_count_clone = frame_count.clone();
    let extracted_count_clone = extracted_count.clone();
    let output_dir = output_dir.clone();
    let video_name = video_name.to_string();
    let cancel_flag_clone = cancel_flag.clone();
    
    // サンプルコールバック設定（ストリーミング処理）
    appsink.set_callbacks(
        gstreamer_app::AppSinkCallbacks::builder()
            .new_sample(move |appsink| {
                // キャンセルチェック
                if cancel_flag_clone.load(Ordering::Relaxed) {
                    eprintln!("⚠️ キャンセルされました");
                    return Err(gst::FlowError::Eos);
                }
                
                let sample = appsink.pull_sample().map_err(|_| gst::FlowError::Error)?;
                let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;
                let caps = sample.caps().ok_or(gst::FlowError::Error)?;
                
                let video_info = gstreamer_video::VideoInfo::from_caps(caps)
                    .map_err(|_| gst::FlowError::Error)?;
                
                let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
                
                let mut frame_num = frame_count_clone.lock().unwrap();
                let current_frame = *frame_num;
                *frame_num += 1;
                
                // 指定された間隔でフレームを処理
                if current_frame % frame_interval == 0 {
                    let width = video_info.width();
                    let height = video_info.height();
                    
                    // RGB画像バッファから直接ImageBufferを作成
                    let img_rgb8 = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(
                        width,
                        height,
                        map.as_slice().to_vec(),
                    ).ok_or(gst::FlowError::Error)?;
                    
                    // タイルを抽出して保存
                    for col in 0..columns {
                        let crop_x = tile_pos_x + (col * tile_width);
                        let crop_y = tile_pos_y;
                        
                        // 境界チェック
                        if crop_x + tile_width > width || crop_y + tile_height > height {
                            continue;
                        }
                        
                        // タイル画像をクロップ
                        let tile_img = image::ImageBuffer::from_fn(
                            tile_width,
                            tile_height,
                            |x, y| {
                                let px = crop_x + x;
                                let py = crop_y + y;
                                *img_rgb8.get_pixel(px, py)
                            },
                        );
                        
                        // ファイル保存
                        let tile_id = col + 1;
                        let filename = format!("{}_frame={}_tile={}.png", video_name, current_frame, tile_id);
                        let output_file = output_dir.join(&filename);
                        
                        if let Err(e) = tile_img.save(&output_file) {
                            eprintln!("⚠️ タイル保存失敗: {} - {}", output_file.display(), e);
                        }
                    }
                    
                    // プログレス更新
                    let mut extracted = extracted_count_clone.lock().unwrap();
                    *extracted += 1;
                    let _ = progress_sender.send((*extracted, estimated_extracts));
                    
                    if *extracted % 10 == 0 {
                        eprintln!("  処理済み: {} / {} フレーム", *extracted, estimated_extracts);
                    }
                }
                
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );
    
    // パイプライン実行
    pipeline
        .set_state(gst::State::Playing)
        .map_err(|e| format!("パイプライン開始失敗: {:?}", e))?;
    
    let bus = pipeline.bus().ok_or("busの取得失敗")?;
    
    // メッセージループ
    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;
        
        match msg.view() {
            MessageView::Eos(..) => {
                eprintln!("動画の終わりに到達しました");
                break;
            }
            MessageView::Error(err) => {
                pipeline.set_state(gst::State::Null).ok();
                return Err(format!("エラー: {} ({:?})", err.error(), err.debug()));
            }
            _ => {}
        }
        
        // キャンセルチェック
        if cancel_flag.load(Ordering::Relaxed) {
            eprintln!("⚠️ キャンセルされました");
            pipeline.set_state(gst::State::Null).ok();
            return Err("キャンセルされました".to_string());
        }
    }
    
    pipeline
        .set_state(gst::State::Null)
        .map_err(|e| format!("パイプライン停止失敗: {:?}", e))?;
    
    Ok(())
}

#[cfg(all(feature = "gui", feature = "ml"))]
fn extract_tile_images(
    video_path: &PathBuf,
    output_dir: &PathBuf,
    config: &AppConfig,
    frame_interval: u32,
    cancel_flag: Arc<AtomicBool>,
    progress_sender: std::sync::mpsc::Sender<(usize, usize)>,
) {
    use std::fs;
    
    eprintln!("========================================");
    eprintln!("🎯 extract_tile_images 関数が呼び出されました");
    eprintln!("========================================");
    
    // 出力ディレクトリ作成
    if let Err(_) = fs::create_dir_all(output_dir) {
        eprintln!("出力フォルダ作成失敗");
        return;
    }
    
    eprintln!("📁 タイル画像出力先: {}", output_dir.display());
    
    let tile_pos_x = config.button_tile.x as u32;
    let tile_pos_y = config.button_tile.y as u32;
    let tile_width = config.button_tile.width as u32;
    let tile_height = config.button_tile.height as u32;
    let columns = config.button_tile.columns_per_row as u32;
    
    // ビデオ名を取得
    let video_name = video_path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("video")
        .to_string();
    
    eprintln!("ビデオ名: {}", video_name);
    eprintln!("タイル設定: pos=({}, {}), size={}x{}, columns={}", 
        tile_pos_x, tile_pos_y, tile_width, tile_height, columns);
    
    // 一時的なフレーム抽出用の設定
    let temp_dir = std::env::temp_dir().join(format!("input_analyzer_temp_{}", 
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()));
    
    let mut extractor_config = input_analyzer::frame_extractor::FrameExtractorConfig::default();
    extractor_config.output_dir = temp_dir.clone();
    extractor_config.frame_interval = frame_interval;
    
    eprintln!("🎬 動画からフレームを抽出・タイル化開始（{}フレーム間隔）", frame_interval);
    
    // FrameExtractorを使わずに直接GStreamerでストリーミング処理
    // これによりメモリ効率が大幅に向上
    match extract_and_process_tiles_streaming(
        video_path,
        output_dir,
        &video_name,
        tile_pos_x,
        tile_pos_y,
        tile_width,
        tile_height,
        columns,
        frame_interval,
        cancel_flag,
        progress_sender,
    ) {
        Ok(_) => {
            eprintln!("✅ タイル画像抽出完了");
        }
        Err(e) => {
            eprintln!("❌ フレーム抽出エラー: {}", e);
        }
    }
}

#[cfg(not(all(feature = "gui", feature = "ml")))]
fn main() {
    eprintln!("エラー: このプログラムはgui機能とml機能を有効にしてビルドする必要があります。");
    eprintln!();
    eprintln!("ビルドコマンド:");
    eprintln!("  cargo build --bin input_editor_gui --features gui,ml --release");
    eprintln!();
    std::process::exit(1);
}
