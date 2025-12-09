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
    Arc, Mutex,
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
    load_and_normalize_image, IconClassifier, ModelConfig,
};
#[cfg(all(feature = "gui", feature = "ml"))]
use input_analyzer::model_metadata::ModelMetadata;
#[cfg(all(feature = "gui", feature = "ml"))]
use input_analyzer::model_storage;

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
    
    // 学習ウィンドウ用フィールド
    show_training_window: bool,
    train_data_dir: Option<PathBuf>,
    train_button_labels: Vec<String>,
    train_button_labels_edit: String,
    train_epochs: usize,
    train_batch_size: usize,
    train_learning_rate: f64,
    train_val_ratio: f32,
    train_output_path: String,
    train_progress_message: String,
    training_running: bool,
    training_result_rx: Option<std::sync::mpsc::Receiver<Result<String, String>>>,
    
    // 分類ウィンドウ用フィールド
    show_classification_window: bool,
    classify_model_path: Option<PathBuf>,
    classify_video_path: Option<PathBuf>,
    classify_output_dir: Option<PathBuf>,
    classify_progress: Option<(usize, usize)>,
    classify_cancel_flag: Option<Arc<AtomicBool>>,
    classify_progress_rx: Option<std::sync::mpsc::Receiver<(usize, usize)>>,
    classify_result_rx: Option<std::sync::mpsc::Receiver<Result<String, String>>>,
    classify_status_message: String,
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
            
            // 学習ウィンドウフィールド初期化
            show_training_window: false,
            train_data_dir: None,
            train_button_labels: Vec::new(),
            train_button_labels_edit: String::new(),
            train_epochs: 50,
            train_batch_size: 8,
            train_learning_rate: 0.001,
            train_val_ratio: 0.2,
            train_output_path: "models/icon_classifier".to_string(),
            train_progress_message: String::new(),
            training_running: false,
            training_result_rx: None,
            
            // 分類ウィンドウフィールド初期化
            show_classification_window: false,
            classify_model_path: None,
            classify_video_path: None,
            classify_output_dir: None,
            classify_progress: None,
            classify_cancel_flag: None,
            classify_progress_rx: None,
            classify_result_rx: None,
            classify_status_message: String::new(),
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
    
    /// buttons.txtを生成または読み込み
    fn load_or_generate_button_labels(&mut self, data_dir: &PathBuf) -> Result<(), String> {
        let buttons_file = data_dir.join("buttons.txt");
        
        if buttons_file.exists() {
            // 既存のbuttons.txtを読み込み
            let content = std::fs::read_to_string(&buttons_file)
                .map_err(|e| format!("buttons.txt読み込みエラー: {}", e))?;
            self.train_button_labels = content
                .trim()
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        } else {
            // サブフォルダ名から自動生成
            let entries: Vec<_> = std::fs::read_dir(data_dir)
                .map_err(|e| format!("ディレクトリ読み込みエラー: {}", e))?
                .filter_map(Result::ok)
                .filter_map(|e| {
                    let path = e.path();
                    if path.is_dir() {
                        if let Some(name) = path.file_name() {
                            if let Some(name_str) = name.to_str() {
                                // dir_*とothersを除外
                                if !name_str.starts_with("dir_") && name_str != "others" && name_str != "empty" {
                                    return Some(name_str.to_string());
                                }
                            }
                        }
                    }
                    None
                })
                .collect();
            
            if entries.is_empty() {
                return Err("ボタンフォルダが見つかりません".to_string());
            }
            
            // アルファベット順でソート
            let mut sorted = entries;
            sorted.sort();
            self.train_button_labels = sorted.clone();
            
            // buttons.txtに保存
            let content = sorted.join(",");
            std::fs::write(&buttons_file, content)
                .map_err(|e| format!("buttons.txt保存エラー: {}", e))?;
        }
        
        self.train_button_labels_edit = self.train_button_labels.join(",");
        Ok(())
    }
    
    /// 学習開始
    fn start_training(&mut self) {
        self.training_running = true;
        self.train_progress_message = "学習を開始しています...".to_string();
        
        let data_dir = self.train_data_dir.clone().unwrap();
        let button_labels = self.train_button_labels.clone();
        let epochs = self.train_epochs;
        let batch_size = self.train_batch_size;
        let learning_rate = self.train_learning_rate;
        let val_ratio = self.train_val_ratio;
        let output_path = self.train_output_path.clone();
        
        // 結果通知用チャンネル
        let (result_tx, result_rx) = std::sync::mpsc::channel::<Result<String, String>>();
        self.training_result_rx = Some(result_rx);
        
        // 別スレッドで学習実行
        std::thread::spawn(move || {
            eprintln!("🚀 学習スレッド開始");
            eprintln!("データディレクトリ: {:?}", data_dir);
            eprintln!("ボタンラベル: {:?}", button_labels);
            
            // 学習データを増強（10枚未満のクラスを10枚以上に）
            eprintln!("学習データを増強中...");
            if let Err(e) = augment_training_data(&data_dir) {
                eprintln!("❌ データ増強エラー: {}", e);
                let _ = result_tx.send(Err(format!("データ増強エラー: {}", e)));
                return;
            }
            eprintln!("✓ データ増強完了");
            
            eprintln!("学習パラメータ:");
            eprintln!("  エポック数: {}", epochs);
            eprintln!("  バッチサイズ: {}", batch_size);
            eprintln!("  学習率: {}", learning_rate);
            eprintln!("  検証データ割合: {}", val_ratio);
            eprintln!("  出力パス: {}", output_path);
            
            // train_modelバイナリを呼び出す
            let args = vec![
                "--data-dir".to_string(),
                data_dir.to_string_lossy().to_string(),
                "--output".to_string(),
                output_path.clone(),
                "--buttons".to_string(),
                button_labels.join(","),
                "--epochs".to_string(),
                epochs.to_string(),
                "--batch-size".to_string(),
                batch_size.to_string(),
                "--learning-rate".to_string(),
                learning_rate.to_string(),
                "--val-ratio".to_string(),
                val_ratio.to_string(),
            ];
            
            eprintln!("train_modelを実行中...");
            let child = std::process::Command::new("target/release/train_model.exe")
                .args(&args)
                .spawn();
            
            match child {
                Ok(mut child) => {
                    // プロセスの終了を待つ
                    match child.wait() {
                        Ok(status) => {
                            if status.success() {
                                eprintln!("✅ 学習完了");
                                let _ = result_tx.send(Ok(format!("学習完了: {}.tar.gz", output_path)));
                            } else {
                                eprintln!("❌ 学習失敗（終了コード: {:?}）", status.code());
                                let _ = result_tx.send(Err(format!("学習失敗（終了コード: {:?}）", status.code())));
                            }
                        }
                        Err(e) => {
                            eprintln!("❌ プロセス待機エラー: {}", e);
                            let _ = result_tx.send(Err(format!("プロセス待機エラー: {}", e)));
                        }
                    }
                }
                Err(e) => {
                    eprintln!("❌ train_model実行エラー: {}", e);
                    eprintln!("ヒント: 先に `cargo build --bin train_model --features ml --release` を実行してください");
                    let _ = result_tx.send(Err(format!("train_model実行エラー: {}. train_modelをビルドしてください", e)));
                }
            }
            
            eprintln!("✅ 学習スレッド完了");
        });
    }
    
    /// 分類開始
    fn start_classification(&mut self) {
        let model_path = self.classify_model_path.clone().unwrap();
        let video_path = self.classify_video_path.clone().unwrap();
        let output_dir = self.classify_output_dir.clone().unwrap();
        let config = self.config.clone();
        
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.classify_cancel_flag = Some(cancel_flag.clone());
        
        let (progress_tx, progress_rx) = std::sync::mpsc::channel::<(usize, usize)>();
        self.classify_progress_rx = Some(progress_rx);
        self.classify_progress = Some((0, 1));
        
        let (result_tx, result_rx) = std::sync::mpsc::channel::<Result<String, String>>();
        self.classify_result_rx = Some(result_rx);
        self.classify_status_message = "分類を開始しています...".to_string();
        
        std::thread::spawn(move || {
            eprintln!("🚀 分類スレッド開始");
            eprintln!("モデル: {:?}", model_path);
            eprintln!("動画: {:?}", video_path);
            eprintln!("出力: {:?}", output_dir);
            
            let video_name = video_path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("video");
            
            let tile_pos_x = config.button_tile.x;
            let tile_pos_y = config.button_tile.y;
            let tile_size = config.button_tile.tile_size;
            let columns = config.button_tile.columns_per_row;
            
            eprintln!("タイル抽出と分類を実行中...");
            eprintln!("  タイル位置: ({}, {})", tile_pos_x, tile_pos_y);
            eprintln!("  タイルサイズ: {}x{}", tile_size, tile_size);
            eprintln!("  列数: {}", columns);
            
            // 出力ディレクトリを作成
            if let Err(e) = std::fs::create_dir_all(&output_dir) {
                let _ = result_tx.send(Err(format!("出力ディレクトリ作成エラー: {}", e)));
                return;
            }
            
            // 動画名フォルダを作成
            let video_output_dir = output_dir.join(video_name);
            if let Err(e) = std::fs::create_dir_all(&video_output_dir) {
                let _ = result_tx.send(Err(format!("動画出力フォルダ作成エラー: {}", e)));
                return;
            }
            
            // Step 1: タイル画像を一時フォルダに抽出
            eprintln!("Step 1: タイル画像を抽出中...");
            let temp_tiles_dir = video_output_dir.join("temp_tiles");
            if let Err(e) = std::fs::create_dir_all(&temp_tiles_dir) {
                let _ = result_tx.send(Err(format!("一時フォルダ作成エラー: {}", e)));
                return;
            }
            
            match extract_and_process_tiles_streaming(
                &video_path,
                &temp_tiles_dir,
                video_name,
                tile_pos_x,
                tile_pos_y,
                tile_size,
                tile_size,
                columns,
                1, // 全フレーム処理
                cancel_flag.clone(),
                progress_tx.clone(),
            ) {
                Ok(_) => {
                    eprintln!("✅ タイル抽出完了");
                }
                Err(e) => {
                    eprintln!("❌ タイル抽出エラー: {}", e);
                    let _ = result_tx.send(Err(format!("タイル抽出エラー: {}", e)));
                    return;
                }
            }
            
            // Step 2: 抽出したタイルを分類
            eprintln!("Step 2: タイルを分類中...");
            match classify_extracted_tiles(
                &model_path,
                &temp_tiles_dir,
                &video_output_dir,
                cancel_flag,
                progress_tx.clone(),
            ) {
                Ok(stats) => {
                    eprintln!("✅ タイル分類完了");
                    // 一時フォルダを削除
                    let _ = std::fs::remove_dir_all(&temp_tiles_dir);
                    
                    let _ = result_tx.send(Ok(format!(
                        "分類完了: {} 枚のタイルを分類しました\n処理済み: {} / 未分類: {}",
                        stats.total,
                        stats.classified,
                        stats.unclassified
                    )));
                }
                Err(e) => {
                    eprintln!("❌ タイル分類エラー: {}", e);
                    let _ = result_tx.send(Err(format!("タイル分類エラー: {}", e)));
                }
            }
            
            eprintln!("✅ 分類スレッド完了");
        });
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

        // モデル読み込み（tar.gz形式）
        let (metadata, model_binary) = model_storage::load_model_with_metadata(&model_path)
            .map_err(|e| format!("モデル読み込みエラー: {}", e))?;
        
        eprintln!("モデル情報:");
        eprintln!("  ボタンラベル: {:?}", metadata.button_labels);
        eprintln!("  クラス数: {}", 8 + metadata.button_labels.len());
        
        // クラス順序: dir_1~9 (ニュートラルの5を除く), button_labelsの順
        let mut class_names: Vec<String> = vec![
            "dir_1".to_string(), "dir_2".to_string(), "dir_3".to_string(),
            "dir_4".to_string(), "dir_6".to_string(), "dir_7".to_string(),
            "dir_8".to_string(), "dir_9".to_string(),
        ];
        class_names.extend(metadata.button_labels.clone());
        
        let num_classes = class_names.len();
        
        // 一時ファイルに保存してロード
        let temp_model_file = std::env::temp_dir().join("temp_model_gui.mpk");
        std::fs::write(&temp_model_file, &model_binary)
            .map_err(|e| format!("一時ファイル作成エラー: {}", e))?;
        
        let record = CompactRecorder::new()
            .load(temp_model_file.clone(), &device)
            .map_err(|e| format!("モデルレコード読み込みエラー: {}", e))?;
        
        std::fs::remove_file(&temp_model_file).ok();

        let model = ModelConfig::new(num_classes)
            .init::<B>(&device)
            .load_record(record);

        // 一時ディレクトリ
        let temp_dir = std::path::PathBuf::from("temp_extract_gui");
        let temp_frames_dir = std::path::PathBuf::from("temp_frames_gui");
        fs::create_dir_all(&temp_dir).map_err(|e| format!("ディレクトリ作成エラー: {}", e))?;
        fs::create_dir_all(&temp_frames_dir)
            .map_err(|e| format!("ディレクトリ作成エラー: {}", e))?;

        // GStreamerで動画情報を取得
        let video_info = FrameExtractor::get_video_info(&video_path)
            .map_err(|e| format!("動画情報取得エラー: {}", e))?;
        
        let video_width = video_info.width as u32;
        let video_height = video_info.height as u32;
        
        // 動画解像度を検証
        if video_width != metadata.video_width || video_height != metadata.video_height {
            return Err(format!(
                "動画解像度が学習時と異なります。\n  学習時: {}x{}\n  入力動画: {}x{}\n学習時と同じ解像度の動画を使用してください。",
                metadata.video_width, metadata.video_height,
                video_width, video_height
            ));
        }
        eprintln!("✓ 動画解像度を検証: {}x{}", video_width, video_height);

        // 総フレーム数を推定
        let total_frames = (video_info.duration_sec * video_info.fps).ceil() as usize;
        eprintln!("推定フレーム数: {}", total_frames);
        
        // フレーム抽出設定
        let config = input_analyzer::frame_extractor::FrameExtractorConfig {
            frame_interval: 1,
            output_dir: temp_frames_dir.clone(),
            image_format: "png".to_string(),
            jpeg_quality: 95,
        };

        let extractor = FrameExtractor::new(config);
        
        // 入力履歴抽出（フレームごとに随時処理）
        let records = Arc::new(Mutex::new(Vec::new()));
        let current_state = Arc::new(Mutex::new(None::<InputState>));
        let duration = Arc::new(Mutex::new(0u32));
        let frame_count = Arc::new(Mutex::new(0usize));

        eprintln!("フレーム抽出と解析を開始（1フレームずつ処理）...");
        
        let records_clone = records.clone();
        let current_state_clone = current_state.clone();
        let duration_clone = duration.clone();
        let frame_count_clone = frame_count.clone();
        let cancel_flag_clone = cancel_flag.clone();
        let tx_clone = tx.clone();
        let temp_dir_clone = temp_dir.clone();
        let metadata_clone = metadata.clone();
        let class_names_clone = class_names.clone();
        
        extractor.extract_frames_with_callback(&video_path, move |frame_path| {
            // キャンセルチェック
            if cancel_flag_clone.load(Ordering::Relaxed) {
                return Err(anyhow::anyhow!("キャンセルされました"));
            }

            let mut count = frame_count_clone.lock().unwrap();
            *count += 1;
            let current_count = *count;
            drop(count);
            
            // 進捗を送信
            let _ = tx_clone.send(ExtractionResult::Progress(current_count, total_frames));

            // フレームから入力状態を抽出
            let state = Self::extract_state_from_frame_static::<B>(
                &frame_path,
                &model,
                &device,
                &temp_dir_clone,
                &metadata_clone,
                &class_names_clone,
            ).map_err(|e| anyhow::anyhow!("フレーム処理エラー: {}", e))?;

            // 状態の変化を記録
            let mut current = current_state_clone.lock().unwrap();
            let mut dur = duration_clone.lock().unwrap();
            let mut recs = records_clone.lock().unwrap();
            
            if let Some(ref prev_state) = *current {
                if &state == prev_state {
                    *dur += 1;
                } else {
                    recs.push(Self::state_to_record_static(prev_state, *dur, &metadata_clone.button_labels));
                    *current = Some(state);
                    *dur = 1;
                }
            } else {
                *current = Some(state);
                *dur = 1;
            }
            
            drop(current);
            drop(dur);
            drop(recs);
            
            // フレームファイルを即座に削除（メモリ節約）
            fs::remove_file(&frame_path).ok();
            
            Ok(())
        }).map_err(|e| format!("フレーム処理エラー: {}", e))?;
        
        // 結果を取り出す
        let mut records = Arc::try_unwrap(records)
            .map(|m| m.into_inner().unwrap())
            .unwrap_or_else(|arc| arc.lock().unwrap().clone());

        // 最後の入力を記録
        let final_state = Arc::try_unwrap(current_state)
            .map(|m| m.into_inner().unwrap())
            .unwrap_or_else(|arc| arc.lock().unwrap().clone());
        let final_duration = Arc::try_unwrap(duration)
            .map(|m| m.into_inner().unwrap())
            .unwrap_or_else(|arc| *arc.lock().unwrap());
            
        if let Some(ref state) = final_state {
            records.push(Self::state_to_record_static(state, final_duration, &metadata.button_labels));
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
        metadata: &ModelMetadata,
        class_names: &[String],
    ) -> anyhow::Result<InputState> {
        use std::fs;

        let mut state = InputState::new();
        
        // メタデータから解析領域を取得
        // tile_x, tile_y = 解析対象の左上座標（継続フレーム数列を除く）
        // tile_width/height = 1セルのサイズ（正方形）
        // columns_per_row = 解析対象列数（方向1 + ボタン5 = 6）
        use input_analyzer::input_analyzer::InputIndicatorRegion;
        let region = InputIndicatorRegion {
            x: metadata.tile_x,
            y: metadata.tile_y,
            width: metadata.tile_width * metadata.columns_per_row,
            height: metadata.tile_height,
            rows: 1,
            cols: metadata.columns_per_row,
        };
        
        let icons = extract_bottom_row_icons(frame_path, &region)?;

        // 各列を分類
        // - 1列目（icon_idx=0）: 方向キー、ボタン、その他すべてが入る可能性
        // - 2列目以降: ボタンまたはその他のみ（方向キーは最左列のみに出現）
        for (icon_idx, icon_img) in icons.iter().enumerate() {
            let temp_icon_path = temp_dir.join(format!("temp_icon_{}.png", icon_idx));
            icon_img.save(&temp_icon_path)?;

            // 分類
            let image_data = load_and_normalize_image(&temp_icon_path)?;
            let tensor =
                Tensor::<B, 1>::from_floats(image_data.as_slice(), device).reshape([1, 3, 48, 48]);
            let (predictions, _) = model.predict(tensor);
            let class_id = predictions.into_data().to_vec::<i32>().unwrap()[0] as usize;
            let class_name = if class_id < class_names.len() {
                &class_names[class_id]
            } else {
                "others"
            };

            // 方向キーは最左列（icon_idx=0）のみで有効
            // 2列目以降で方向キーが検出された場合は無視（学習データが正しければ発生しない）
            if icon_idx > 0 && class_name.starts_with("dir_") {
                // 2列目以降で方向キーが検出された場合は警告のみ（ボタンとしては扱わない）
                eprintln!("警告: {}列目で方向キー {} が検出されました（無視）", icon_idx + 1, class_name);
            } else {
                update_input_state(&mut state, class_name);
            }
            
            fs::remove_file(&temp_icon_path)?;
        }

        Ok(state)
    }

    fn state_to_record_static(state: &InputState, duration: u32, button_labels: &[String]) -> InputRecord {
        // メタデータのボタン順に取得（旧フォーマットとの互換性のため固定フィールドを使用）
        let btn_a1 = button_labels.iter().position(|l| l == "A1")
            .and_then(|_| state.buttons.get("A1").copied()).unwrap_or(0) == 1;
        let btn_a2 = button_labels.iter().position(|l| l == "A2")
            .and_then(|_| state.buttons.get("A2").copied()).unwrap_or(0) == 1;
        let btn_b = button_labels.iter().position(|l| l == "B")
            .and_then(|_| state.buttons.get("B").copied()).unwrap_or(0) == 1;
        let btn_w = button_labels.iter().position(|l| l == "W")
            .and_then(|_| state.buttons.get("W").copied()).unwrap_or(0) == 1;
        let btn_start = button_labels.iter().position(|l| l == "Start")
            .and_then(|_| state.buttons.get("Start").copied()).unwrap_or(0) == 1;
        
        InputRecord {
            duration,
            direction: state.direction,
            btn_a1,
            btn_a2,
            btn_b,
            btn_w,
            btn_start,
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
        
        // 学習結果を確認
        if let Some(rx) = &self.training_result_rx {
            if let Ok(result) = rx.try_recv() {
                self.training_running = false;
                match result {
                    Ok(msg) => {
                        self.train_progress_message = format!("✅ {}", msg);
                    }
                    Err(err) => {
                        self.train_progress_message = format!("❌ エラー: {}", err);
                    }
                }
                self.training_result_rx = None;
            }
        }
        
        // 分類のプログレスを更新
        if let Some(rx) = &self.classify_progress_rx {
            if let Ok((current, total)) = rx.try_recv() {
                self.classify_progress = Some((current, total));
                
                // 完了判定
                if current >= total {
                    self.classify_progress = None;
                    self.classify_cancel_flag = None;
                    self.classify_progress_rx = None;
                }
            }
        }
        
        // 分類結果を確認
        if let Some(rx) = &self.classify_result_rx {
            if let Ok(result) = rx.try_recv() {
                match result {
                    Ok(msg) => {
                        self.classify_status_message = format!("✅ {}", msg);
                    }
                    Err(err) => {
                        self.classify_status_message = format!("❌ エラー: {}", err);
                    }
                }
                self.classify_result_rx = None;
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
                    
                    if ui.button("モデル学習").clicked() {
                        self.show_training_window = true;
                        ui.close_menu();
                    }
                    
                    if ui.button("タイル分類").clicked() {
                        self.show_classification_window = true;
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
                                .add_filter("モデル (tar.gz)", &["tar.gz"])
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
                            ui.label("タイルサイズ (正方形):");
                            ui.add(
                                egui::DragValue::new(&mut self.config.button_tile.tile_size)
                                    .range(1..=512)
                                    .speed(1.0),
                            );
                            ui.label("px");
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
        
        // モデル学習ウィンドウ
        if self.show_training_window {
            let mut is_open = true;
            egui::Window::new("モデル学習")
                .open(&mut is_open)
                .resizable(true)
                .vscroll(true)
                .default_width(600.0)
                .id(egui::Id::new("training_window"))
                .show(ctx, |ui| {
                    ui.heading("入力アイコン分類モデルの学習");
                    
                    ui.label("📁 学習データフォルダ:");
                    ui.horizontal(|ui| {
                        if let Some(path) = &self.train_data_dir {
                            ui.label(format!("選択: {}", path.display()));
                        } else {
                            ui.label("未選択");
                        }
                        
                        if ui.button("フォルダを選択...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .pick_folder()
                            {
                                // buttons.txtを生成または読み込み
                                if let Err(e) = self.load_or_generate_button_labels(&path) {
                                    self.train_progress_message = format!("エラー: {}", e);
                                } else {
                                    self.train_data_dir = Some(path);
                                }
                            }
                        }
                    });
                    
                    ui.separator();
                    
                    // ボタンラベル編集
                    if !self.train_button_labels.is_empty() {
                        ui.label("🎮 ボタン順序:");
                        ui.label("カンマ区切りで編集できます（方向キーとothersは自動除外）");
                        
                        ui.horizontal(|ui| {
                            if ui.text_edit_singleline(&mut self.train_button_labels_edit).changed() {
                                // 編集内容をリストに反映
                                self.train_button_labels = self.train_button_labels_edit
                                    .split(',')
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| !s.is_empty())
                                    .collect();
                            }
                        });
                        
                        ui.label(format!("現在のボタン: {}", self.train_button_labels.join(", ")));
                    }
                    
                    ui.separator();
                    
                    ui.label("⚙️ 学習パラメータ:");
                    ui.horizontal(|ui| {
                        ui.label("エポック数:");
                        ui.add(egui::DragValue::new(&mut self.train_epochs).range(1..=500).speed(1.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("バッチサイズ:");
                        ui.add(egui::DragValue::new(&mut self.train_batch_size).range(1..=64).speed(1.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("学習率:");
                        ui.add(egui::DragValue::new(&mut self.train_learning_rate).range(0.0001..=0.1).speed(0.0001));
                    });
                    ui.horizontal(|ui| {
                        ui.label("検証データ割合:");
                        ui.add(egui::DragValue::new(&mut self.train_val_ratio).range(0.1..=0.5).speed(0.01));
                    });
                    
                    ui.separator();
                    
                    ui.label("💾 出力パス:");
                    ui.text_edit_singleline(&mut self.train_output_path);
                    ui.label("(.tar.gz が自動追加されます)");
                    
                    ui.separator();
                    
                    if !self.train_progress_message.is_empty() {
                        ui.colored_label(egui::Color32::LIGHT_BLUE, &self.train_progress_message);
                    }
                    
                    ui.horizontal(|ui| {
                        if !self.training_running {
                            if ui.button("🚀 学習開始").clicked() {
                                if self.train_data_dir.is_some() && !self.train_button_labels.is_empty() {
                                    self.start_training();
                                } else {
                                    self.train_progress_message = "学習データフォルダとボタンラベルを設定してください".to_string();
                                }
                            }
                        } else {
                            ui.label("学習中...");
                        }
                        
                        if ui.button("閉じる").clicked() {
                            self.show_training_window = false;
                        }
                    });
                });
            
            if !is_open {
                self.show_training_window = false;
            }
        }
        
        // タイル分類ウィンドウ
        if self.show_classification_window {
            let mut is_open = true;
            egui::Window::new("タイル分類")
                .open(&mut is_open)
                .resizable(true)
                .vscroll(true)
                .default_width(600.0)
                .id(egui::Id::new("classification_window"))
                .show(ctx, |ui| {
                    ui.heading("動画からタイルを抽出して分類");
                    
                    ui.label("🤖 学習済みモデル:");
                    ui.horizontal(|ui| {
                        if let Some(path) = &self.classify_model_path {
                            ui.label(format!("選択: {}", path.display()));
                        } else {
                            ui.label("未選択");
                        }
                        
                        if ui.button("モデルを選択...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("モデル (tar.gz)", &["tar.gz"])
                                .pick_file()
                            {
                                self.classify_model_path = Some(path);
                            }
                        }
                    });
                    
                    ui.separator();
                    
                    ui.label("📹 ビデオファイル:");
                    ui.horizontal(|ui| {
                        if let Some(path) = &self.classify_video_path {
                            ui.label(format!("選択: {}", path.display()));
                        } else {
                            ui.label("未選択");
                        }
                        
                        if ui.button("ビデオを選択...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("動画", &["mp4", "avi", "mov"])
                                .pick_file()
                            {
                                self.classify_video_path = Some(path);
                            }
                        }
                    });
                    
                    ui.separator();
                    
                    ui.label("📁 出力フォルダ:");
                    ui.horizontal(|ui| {
                        if let Some(path) = &self.classify_output_dir {
                            ui.label(format!("選択: {}", path.display()));
                        } else {
                            ui.label("未選択");
                        }
                        
                        if ui.button("フォルダを選択...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .pick_folder()
                            {
                                self.classify_output_dir = Some(path);
                            }
                        }
                    });
                    
                    ui.separator();
                    
                    // ステータスメッセージ表示
                    if !self.classify_status_message.is_empty() {
                        ui.colored_label(egui::Color32::LIGHT_BLUE, &self.classify_status_message);
                    }
                    
                    // 進捗表示
                    if let Some((current, total)) = self.classify_progress {
                        ui.label(format!("処理中: {} / {} フレーム", current, total));
                        let progress = current as f32 / total as f32;
                        ui.add(
                            egui::ProgressBar::new(progress)
                                .show_percentage()
                                .animate(true),
                        );
                        ui.add_space(10.0);
                        if ui.button("キャンセル").clicked() {
                            if let Some(flag) = &self.classify_cancel_flag {
                                flag.store(true, Ordering::Relaxed);
                            }
                        }
                    } else {
                        if ui.button("🚀 分類開始").clicked() {
                            if self.classify_model_path.is_some() 
                                && self.classify_video_path.is_some() 
                                && self.classify_output_dir.is_some() 
                            {
                                self.start_classification();
                            }
                        }
                    }
                    
                    ui.separator();
                    if ui.button("閉じる").clicked() {
                        self.show_classification_window = false;
                    }
                });
            
            if !is_open {
                self.show_classification_window = false;
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
            let tile_w = self.config.button_tile.tile_size as f32 * scale_x;
            let tile_h = self.config.button_tile.tile_size as f32 * scale_y;
            
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

/// 学習データを増強（10枚未満のクラスを10枚以上にコピーで増やす）
#[cfg(all(feature = "gui", feature = "ml"))]
fn augment_training_data(data_dir: &PathBuf) -> Result<(), String> {
    const MIN_IMAGES: usize = 10;
    
    // 各クラスディレクトリを走査
    let entries = std::fs::read_dir(data_dir)
        .map_err(|e| format!("データディレクトリ読み込みエラー: {}", e))?;
    
    for entry in entries {
        let entry = entry.map_err(|e| format!("エントリ読み込みエラー: {}", e))?;
        let path = entry.path();
        
        if !path.is_dir() {
            continue;
        }
        
        let class_name = path.file_name().unwrap().to_string_lossy().to_string();
        
        // 画像ファイルを収集
        let image_files: Vec<PathBuf> = std::fs::read_dir(&path)
            .map_err(|e| format!("クラスディレクトリ読み込みエラー: {}", e))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file() && 
                path.extension().and_then(|s| s.to_str()).map(|ext| {
                    ext == "png" || ext == "jpg" || ext == "jpeg"
                }).unwrap_or(false)
            })
            .collect();
        
        let current_count = image_files.len();
        
        if current_count == 0 {
            eprintln!("  ⚠️  {}: 画像なし", class_name);
            continue;
        }
        
        if current_count >= MIN_IMAGES {
            eprintln!("  ✓ {}: {} 枚（十分）", class_name, current_count);
            continue;
        }
        
        // 何枚コピーが必要か計算
        let copies_needed_per_image = (MIN_IMAGES + current_count - 1) / current_count;
        let total_copies = copies_needed_per_image - 1; // 元の画像は既にあるので -1
        
        eprintln!("  📦 {}: {} 枚 -> 各画像を{}回コピーして{}枚に増やします", 
            class_name, current_count, total_copies, current_count * copies_needed_per_image);
        
        // 各画像をコピー
        for (idx, image_file) in image_files.iter().enumerate() {
            let stem = image_file.file_stem().unwrap().to_string_lossy();
            let ext = image_file.extension().unwrap().to_string_lossy();
            
            for copy_num in 1..=total_copies {
                let new_filename = format!("{}_copy{}.{}", stem, copy_num, ext);
                let dest_path = path.join(new_filename);
                
                if let Err(e) = std::fs::copy(image_file, &dest_path) {
                    eprintln!("    ⚠️  コピー失敗: {} -> {} - {}", 
                        image_file.display(), dest_path.display(), e);
                }
            }
        }
        
        let final_count = current_count * copies_needed_per_image;
        eprintln!("    ✓ {}: 増強完了（{} 枚）", class_name, final_count);
    }
    
    Ok(())
}

#[cfg(all(feature = "gui", feature = "ml"))]
#[derive(Debug, Clone)]
struct ClassificationStats {
    total: usize,
    classified: usize,
    unclassified: usize,
}

/// 抽出済みタイルを分類
#[cfg(all(feature = "gui", feature = "ml"))]
fn classify_extracted_tiles(
    model_path: &PathBuf,
    tiles_dir: &PathBuf,
    output_dir: &PathBuf,
    cancel_flag: Arc<AtomicBool>,
    progress_sender: std::sync::mpsc::Sender<(usize, usize)>,
) -> Result<ClassificationStats, String> {
    use burn::tensor::Tensor;
    use input_analyzer::model_storage;
    use input_analyzer::ml_model::NUM_CLASSES;
    
    type MyBackend = burn_wgpu::Wgpu;
    type MyDevice = burn_wgpu::WgpuDevice;
    
    // モデルをロード
    eprintln!("モデルをロード中: {:?}", model_path);
    let device = MyDevice::default();
    
    // メタデータとモデルバイナリをロード
    let (metadata, model_data) = model_storage::load_model_with_metadata(model_path)
        .map_err(|e| format!("モデルメタデータの読み込みエラー: {}", e))?;
    
    eprintln!("メタデータ: ボタン={:?}", metadata.button_labels);
    
    // モデルバイナリを一時ファイルに保存
    let temp_model_path = std::env::temp_dir().join("temp_model.mpk");
    std::fs::write(&temp_model_path, &model_data)
        .map_err(|e| format!("一時ファイル書き込みエラー: {}", e))?;
    
    // モデルを初期化
    let config = ModelConfig::new(NUM_CLASSES);
    let model = config.init::<MyBackend>(&device);
    
    // モデルの重みをロード
    let record = CompactRecorder::new()
        .load(temp_model_path.clone(), &device)
        .map_err(|e| format!("モデルの読み込みエラー: {}", e))?;
    
    let model = model.load_record(record);
    eprintln!("✓ モデルをロードしました");
    
    // 一時ファイルを削除
    let _ = std::fs::remove_file(&temp_model_path);
    
    // クラス名のリスト（buttons.txtのボタン + dir_x + others）
    let mut class_names: Vec<String> = metadata.button_labels.clone();
    class_names.extend_from_slice(&[
        "dir_1".to_string(), "dir_2".to_string(), "dir_3".to_string(), "dir_4".to_string(),
        "dir_6".to_string(), "dir_7".to_string(), "dir_8".to_string(), "dir_9".to_string(),
        "others".to_string(),
    ]);
    
    // 分類フォルダを作成
    for class_name in &class_names {
        let class_dir = output_dir.join(class_name);
        std::fs::create_dir_all(&class_dir)
            .map_err(|e| format!("分類フォルダ作成エラー {}: {}", class_name, e))?;
    }
    
    eprintln!("分類フォルダを作成しました: {:?}", class_names);
    
    // タイル画像ファイルを収集
    let tile_files: Vec<PathBuf> = std::fs::read_dir(tiles_dir)
        .map_err(|e| format!("タイルディレクトリ読み込みエラー: {}", e))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("png")
        })
        .collect();
    
    let total_tiles = tile_files.len();
    eprintln!("タイル数: {}", total_tiles);
    
    let mut stats = ClassificationStats {
        total: 0,
        classified: 0,
        unclassified: 0,
    };
    
    // 各タイルを分類
    for (idx, tile_path) in tile_files.iter().enumerate() {
        if cancel_flag.load(Ordering::Relaxed) {
            eprintln!("⚠️ キャンセルされました");
            return Err("キャンセルされました".to_string());
        }
        
        // 画像を正規化
        let normalized = match load_and_normalize_image(tile_path) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("⚠️ 画像読み込み失敗: {:?} - {}", tile_path, e);
                continue;
            }
        };
        
        // テンソルに変換して予測
        let tile_height = metadata.tile_height as usize;
        let tile_width = metadata.tile_width as usize;
        let tensor = Tensor::<MyBackend, 1>::from_floats(normalized.as_slice(), &device)
            .reshape([1, 3, tile_height, tile_width]);
        
        let (predictions, _) = model.predict(tensor);
        let predicted_class = predictions.to_data().to_vec::<i32>().unwrap()[0] as usize;
        
        // クラス名を取得（範囲外は全てothersに分類）
        let class_name = if predicted_class < class_names.len() {
            class_names[predicted_class].as_str()
        } else {
            "others"
        };
        
        // ファイルを分類フォルダに移動
        let filename = tile_path.file_name().unwrap();
        let class_dir = output_dir.join(class_name);
        let dest_path = class_dir.join(filename);
        
        if let Err(e) = std::fs::copy(tile_path, &dest_path) {
            eprintln!("⚠️ ファイルコピー失敗: {} -> {} - {}", 
                tile_path.display(), dest_path.display(), e);
        }
        
        // 統計更新
        stats.total += 1;
        if class_name == "others" {
            stats.unclassified += 1;
        } else {
            stats.classified += 1;
        }
        
        // プログレス更新
        if (idx + 1) % 10 == 0 || idx == total_tiles - 1 {
            let _ = progress_sender.send((idx + 1, total_tiles));
            eprintln!("  分類進行: {} / {} (分類: {}, 未分類: {})",
                idx + 1, total_tiles, stats.classified, stats.unclassified);
        }
    }
    
    Ok(stats)
}

/*
// この関数は使用されていません。classify_extracted_tiles関数を使用してください。
#[cfg(all(feature = "gui", feature = "ml"))]
fn extract_and_classify_tiles_streaming(
    model_path: &PathBuf,
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
) -> Result<ClassificationStats, String> {
    use burn::tensor::Tensor;
    use input_analyzer::model_storage;
    use input_analyzer::ml_model::NUM_CLASSES;
    
    type MyBackend = burn_wgpu::Wgpu;
    type MyDevice = burn_wgpu::WgpuDevice;
    
    // モデルをロード
    eprintln!("モデルをロード中: {:?}", model_path);
    let device = MyDevice::default();
    
    // メタデータとモデルバイナリをロード
    let (metadata, model_data) = model_storage::load_model_with_metadata(model_path)
        .map_err(|e| format!("モデルメタデータの読み込みエラー: {}", e))?;
    
    eprintln!("メタデータ: ボタン={:?}", metadata.button_labels);
    
    // モデルバイナリを一時ファイルに保存
    let temp_model_path = std::env::temp_dir().join("temp_model.mpk");
    std::fs::write(&temp_model_path, &model_data)
        .map_err(|e| format!("一時ファイル書き込みエラー: {}", e))?;
    
    // モデルを初期化
    let config = ModelConfig::new(NUM_CLASSES);
    let model = config.init::<MyBackend>(&device);
    
    // モデルの重みをロード
    let record = CompactRecorder::new()
        .load(temp_model_path.clone(), &device)
        .map_err(|e| format!("モデルの読み込みエラー: {}", e))?;
    
    let model = model.load_record(record);
    eprintln!("✓ モデルをロードしました");
    
    // 一時ファイルを削除
    let _ = std::fs::remove_file(&temp_model_path);
    
    // クラス名のリスト（buttons.txtのボタン + dir_x + others + empty）
    let mut class_names: Vec<String> = metadata.button_labels.clone();
    class_names.extend_from_slice(&[
        "dir_1".to_string(), "dir_2".to_string(), "dir_3".to_string(), "dir_4".to_string(),
        "dir_6".to_string(), "dir_7".to_string(), "dir_8".to_string(), "dir_9".to_string(),
        "others".to_string(), "empty".to_string(),
    ]);
    
    // 分類フォルダを作成
    for class_name in &class_names {
        let class_dir = output_dir.join(class_name);
        std::fs::create_dir_all(&class_dir)
            .map_err(|e| format!("分類フォルダ作成エラー {}: {}", class_name, e))?;
    }
    let unclassified_dir = output_dir.join("unclassified");
    std::fs::create_dir_all(&unclassified_dir)
        .map_err(|e| format!("未分類フォルダ作成エラー: {}", e))?;
    
    eprintln!("分類フォルダを作成しました: {:?}", class_names);
    
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
    
    // 統計カウンタ
    let stats = Arc::new(std::sync::Mutex::new(ClassificationStats {
        total: 0,
        classified: 0,
        unclassified: 0,
    }));
    
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
    let stats_clone = stats.clone();
    let class_names_arc = Arc::new(class_names.clone());
    let model_arc = Arc::new(model);
    let device_arc = Arc::new(device);
    
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
                    
                    // タイルを抽出して分類
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
                        
                        // 一時ファイルとして保存して画像を正規化
                        let temp_path = std::env::temp_dir().join(format!("tile_temp_{}.png", col));
                        if let Err(e) = tile_img.save(&temp_path) {
                            eprintln!("⚠️ 一時タイル保存失敗: {}", e);
                            continue;
                        }
                        
                        // 画像を正規化
                        let normalized = match load_and_normalize_image(&temp_path) {
                            Ok(data) => data,
                            Err(e) => {
                                eprintln!("⚠️ 画像正規化失敗: {}", e);
                                let _ = std::fs::remove_file(&temp_path);
                                continue;
                            }
                        };
                        
                        let _ = std::fs::remove_file(&temp_path);
                        
                        // テンソルに変換して予測
                        let tensor = Tensor::<MyBackend, 1>::from_floats(normalized.as_slice(), &device_arc)
                            .reshape([1, 3, tile_height as usize, tile_width as usize]);
                        
                        let (predictions, _) = model_arc.predict(tensor);
                        let predicted_class = predictions.to_data().to_vec::<i32>().unwrap()[0] as usize;
                        
                        // クラス名を取得
                        let class_name = if predicted_class < class_names_arc.len() {
                            class_names_arc[predicted_class].as_str()
                        } else {
                            "unclassified"
                        };
                        
                        // ファイル保存
                        let tile_id = col + 1;
                        let filename = format!("{}_frame={}_tile={}.png", video_name, current_frame, tile_id);
                        let class_dir = output_dir.join(class_name);
                        let output_file = class_dir.join(&filename);
                        
                        if let Err(e) = tile_img.save(&output_file) {
                            eprintln!("⚠️ タイル保存失敗: {} - {}", output_file.display(), e);
                        }
                        
                        // 統計更新
                        let mut stats = stats_clone.lock().unwrap();
                        stats.total += 1;
                        if class_name == "unclassified" {
                            stats.unclassified += 1;
                        } else {
                            stats.classified += 1;
                        }
                    }
                    
                    // プログレス更新
                    let mut extracted = extracted_count_clone.lock().unwrap();
                    *extracted += 1;
                    let _ = progress_sender.send((*extracted, estimated_extracts));
                    
                    if *extracted % 10 == 0 {
                        let stats = stats_clone.lock().unwrap();
                        eprintln!("  処理済み: {} / {} フレーム (分類: {}, 未分類: {})",
                            *extracted, estimated_extracts, stats.classified, stats.unclassified);
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
    
    let final_stats = stats.lock().unwrap().clone();
    Ok(final_stats)
}
*/

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
    let tile_size = config.button_tile.tile_size as u32;
    let columns = config.button_tile.columns_per_row as u32;
    
    // ビデオ名を取得
    let video_name = video_path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("video")
        .to_string();
    
    eprintln!("ビデオ名: {}", video_name);
    eprintln!("タイル設定: pos=({}, {}), size={}x{}, columns={}", 
        tile_pos_x, tile_pos_y, tile_size, tile_size, columns);
    
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
        tile_size,
        tile_size,
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
