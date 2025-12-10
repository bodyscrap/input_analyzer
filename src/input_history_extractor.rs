//! 入力履歴抽出の共通機能

#[cfg(feature = "ml")]
use anyhow::Result;
#[cfg(feature = "ml")]
use std::path::Path;

/// 入力インジケータ領域の設定
#[cfg(feature = "ml")]
#[derive(Debug, Clone)]
pub struct InputIndicatorRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub rows: u32,
    pub cols: u32,
}

/// 入力状態（各ボタンの状態）
#[cfg(feature = "ml")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputState {
    pub direction: u8,     // 1-9 (テンキー配列で8方向を表現。5がニュートラル)
    pub buttons: std::collections::HashMap<String, u8>, // ボタン名 -> 状態 (0 or 1)
}

#[cfg(feature = "ml")]
impl InputState {
    pub fn new() -> Self {
        Self {
            direction: 5,
            buttons: std::collections::HashMap::new(),
        }
    }

    pub fn to_csv_line(&self, duration: u32, button_labels: &[String]) -> String {
        let mut parts = vec![duration.to_string(), self.direction.to_string()];
        for label in button_labels {
            let state = self.buttons.get(label).copied().unwrap_or(0);
            parts.push(state.to_string());
        }
        parts.join(",")
    }
}

/// クラス名から入力状態を更新
#[cfg(feature = "ml")]
pub fn update_input_state(state: &mut InputState, class_name: &str) {
    if class_name.starts_with("dir_") {
        // 方向入力
        match class_name {
            "dir_1" => state.direction = 1,
            "dir_2" => state.direction = 2,
            "dir_3" => state.direction = 3,
            "dir_4" => state.direction = 4,
            "dir_6" => state.direction = 6,
            "dir_7" => state.direction = 7,
            "dir_8" => state.direction = 8,
            "dir_9" => state.direction = 9,
            _ => eprintln!("警告: 未知の方向: {}", class_name),
        }
    } else if class_name != "empty" && class_name != "others" {
        // ボタン入力（empty/others以外）
        state.buttons.insert(class_name.to_string(), 1);
    }
}

/// 最下行のアイコンを抽出
/// 
/// region には継続フレーム数列を含めない（解析対象のみ）
#[cfg(feature = "ml")]
pub fn extract_bottom_row_icons(frame_path: &Path, region: &InputIndicatorRegion) -> Result<Vec<image::RgbImage>> {
    let img = image::open(frame_path)?;
    
    // 各セルを直接抽出（継続フレーム数列は領域に含まれていない）
    let mut icons = Vec::new();
    let row = 0; // rows=1 なので row=0 が対象
    
    for col in 0..region.cols {
        let cell_x = region.x + (col * region.width / region.cols);
        let cell_y = region.y + (row * region.height / region.rows);
        let cell_width = region.width / region.cols;
        let cell_height = region.height / region.rows;
        
        let cell_image = img.crop_imm(cell_x, cell_y, cell_width, cell_height);
        icons.push(cell_image.to_rgb8());
    }
    
    Ok(icons)
}
