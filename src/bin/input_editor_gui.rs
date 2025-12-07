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
        }
    }
}

#[cfg(all(feature = "gui", feature = "ml"))]
impl InputEditorApp {
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

#[cfg(not(all(feature = "gui", feature = "ml")))]
fn main() {
    eprintln!("エラー: このプログラムはgui機能とml機能を有効にしてビルドする必要があります。");
    eprintln!();
    eprintln!("ビルドコマンド:");
    eprintln!("  cargo build --bin input_editor_gui --features gui,ml --release");
    eprintln!();
    std::process::exit(1);
}
