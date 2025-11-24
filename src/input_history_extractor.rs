//! 入力履歴抽出の共通機能

#[cfg(feature = "ml")]
use anyhow::Result;
#[cfg(feature = "ml")]
use std::path::Path;

/// 入力状態（各ボタンの状態）
#[cfg(feature = "ml")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputState {
    pub direction: u8,     // 1-9 (テンキー配列で8方向を表現。5がニュートラル)
    pub btn_a1: u8,        // 0(OFF) or 1(ON)
    pub btn_a2: u8,        // 0(OFF) or 1(ON)
    pub btn_b: u8,         // 0(OFF) or 1(ON)
    pub btn_w: u8,         // 0(OFF) or 1(ON)
    pub btn_start: u8,     // 0(OFF) or 1(ON)
}

#[cfg(feature = "ml")]
impl InputState {
    pub fn new() -> Self {
        Self {
            direction: 5,
            btn_a1: 0,
            btn_a2: 0,
            btn_b: 0,
            btn_w: 0,
            btn_start: 0,
        }
    }

    pub fn to_csv_line(&self, duration: u32) -> String {
        format!(
            "{},{},{},{},{},{},{}",
            duration, self.direction, self.btn_a1, self.btn_a2, self.btn_b, self.btn_w, self.btn_start
        )
    }
}

/// クラス名から入力状態を更新
#[cfg(feature = "ml")]
pub fn update_input_state(state: &mut InputState, class_name: &str) {
    match class_name {
        "dir_1" => state.direction = 1,
        "dir_2" => state.direction = 2,
        "dir_3" => state.direction = 3,
        "dir_4" => state.direction = 4,
        "dir_6" => state.direction = 6,
        "dir_7" => state.direction = 7,
        "dir_8" => state.direction = 8,
        "dir_9" => state.direction = 9,
        "btn_a1" => state.btn_a1 = 1,
        "btn_a2" => state.btn_a2 = 1,
        "btn_b" => state.btn_b = 1,
        "btn_w" => state.btn_w = 1,
        "btn_start" => state.btn_start = 1,
        "empty" => {} // 何もしない
        _ => eprintln!("警告: 未知のクラス名: {}", class_name),
    }
}

/// 最下行のアイコンを抽出
#[cfg(feature = "ml")]
pub fn extract_bottom_row_icons(frame_path: &Path) -> Result<Vec<image::RgbImage>> {
    use crate::input_analyzer::{InputAnalyzer, InputIndicatorRegion};
    
    let img = image::open(frame_path)?;
    
    // デフォルトの領域設定を使用（右下の入力履歴表示）
    let region = InputIndicatorRegion::default();
    let analyzer = InputAnalyzer::new(region.clone());
    
    // 最下行（行番号は0から始まるので、rows-1が最下行）
    let last_row = region.rows - 1;
    let input_row = analyzer.extract_input_row(&img, last_row)?;
    
    // 入力アイコンのみを返す（フレームカウントは除外）
    Ok(input_row.input_icons)
}
