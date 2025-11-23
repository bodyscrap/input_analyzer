use anyhow::{Context, Result};
use image::RgbImage;
use std::collections::HashMap;
use std::path::Path;

/// 入力の種類
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputType {
    /// 方向入力
    DirectionUp,
    DirectionUpRight,
    DirectionRight,
    DirectionDownRight,
    DirectionDown,
    DirectionDownLeft,
    DirectionLeft,
    DirectionUpLeft,
    /// ボタン入力
    ButtonA1,
    ButtonA2,
    ButtonB,
    ButtonW,
    ButtonStart,
    /// 空（入力なし）
    Empty,
}

impl InputType {
    /// すべての入力タイプを取得
    pub fn all() -> Vec<Self> {
        vec![
            Self::DirectionUp,
            Self::DirectionUpRight,
            Self::DirectionRight,
            Self::DirectionDownRight,
            Self::DirectionDown,
            Self::DirectionDownLeft,
            Self::DirectionLeft,
            Self::DirectionUpLeft,
            Self::ButtonA1,
            Self::ButtonA2,
            Self::ButtonB,
            Self::ButtonW,
            Self::ButtonStart,
            Self::Empty,
        ]
    }

    /// 方向入力かどうか
    pub fn is_direction(&self) -> bool {
        matches!(
            self,
            Self::DirectionUp
                | Self::DirectionUpRight
                | Self::DirectionRight
                | Self::DirectionDownRight
                | Self::DirectionDown
                | Self::DirectionDownLeft
                | Self::DirectionLeft
                | Self::DirectionUpLeft
        )
    }

    /// ボタン入力かどうか
    pub fn is_button(&self) -> bool {
        matches!(
            self,
            Self::ButtonA1 | Self::ButtonA2 | Self::ButtonB | Self::ButtonW | Self::ButtonStart
        )
    }

    /// 文字列表現
    pub fn to_string(&self) -> &str {
        match self {
            Self::DirectionUp => "↑",
            Self::DirectionUpRight => "↗",
            Self::DirectionRight => "→",
            Self::DirectionDownRight => "↘",
            Self::DirectionDown => "↓",
            Self::DirectionDownLeft => "↙",
            Self::DirectionLeft => "←",
            Self::DirectionUpLeft => "↖",
            Self::ButtonA1 => "A_1",
            Self::ButtonA2 => "A_2",
            Self::ButtonB => "B",
            Self::ButtonW => "W",
            Self::ButtonStart => "Start",
            Self::Empty => "(空)",
        }
    }

    /// 名前（ファイル名用）
    pub fn name(&self) -> &str {
        match self {
            Self::DirectionUp => "dir_up",
            Self::DirectionUpRight => "dir_up_right",
            Self::DirectionRight => "dir_right",
            Self::DirectionDownRight => "dir_down_right",
            Self::DirectionDown => "dir_down",
            Self::DirectionDownLeft => "dir_down_left",
            Self::DirectionLeft => "dir_left",
            Self::DirectionUpLeft => "dir_up_left",
            Self::ButtonA1 => "button_a1",
            Self::ButtonA2 => "button_a2",
            Self::ButtonB => "button_b",
            Self::ButtonW => "button_w",
            Self::ButtonStart => "button_start",
            Self::Empty => "empty",
        }
    }
}

/// テンプレートの種類（行位置を考慮）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemplateVariant {
    /// 通常のテンプレート（3-15行目用）
    Normal,
    /// インジケータ映り込み用テンプレート（0-2行目用）
    WithIndicator,
}

/// テンプレート画像
#[derive(Debug, Clone)]
pub struct Template {
    pub input_type: InputType,
    pub image: RgbImage,
    pub variant: TemplateVariant,
}

impl Template {
    /// 新しいテンプレートを作成
    pub fn new(input_type: InputType, image: RgbImage, variant: TemplateVariant) -> Self {
        Self {
            input_type,
            image,
            variant,
        }
    }

    /// テンプレート画像を読み込み
    pub fn load<P: AsRef<Path>>(input_type: InputType, path: P, variant: TemplateVariant) -> Result<Self> {
        let img = image::open(path.as_ref())
            .context("テンプレート画像の読み込みに失敗しました")?;
        Ok(Self::new(input_type, img.to_rgb8(), variant))
    }
}

/// 認識結果
#[derive(Debug, Clone)]
pub struct RecognitionResult {
    /// 認識された入力タイプ
    pub input_type: InputType,
    /// 信頼度（0.0-1.0）
    pub confidence: f64,
}

impl RecognitionResult {
    pub fn new(input_type: InputType, confidence: f64) -> Self {
        Self {
            input_type,
            confidence,
        }
    }
}

/// 1行の入力状態
#[derive(Debug, Clone)]
pub struct RowInputState {
    /// 行番号
    pub row_index: u32,
    /// フレームカウント（認識結果）
    pub frame_count: Option<u32>,
    /// 各列の入力（列1-6）
    pub inputs: Vec<RecognitionResult>,
}

impl RowInputState {
    pub fn new(row_index: u32) -> Self {
        Self {
            row_index,
            frame_count: None,
            inputs: Vec::new(),
        }
    }

    /// 方向入力を取得
    pub fn get_direction(&self) -> Option<InputType> {
        self.inputs
            .iter()
            .find(|r| r.input_type.is_direction())
            .map(|r| r.input_type)
    }

    /// ボタン入力を取得
    pub fn get_buttons(&self) -> Vec<InputType> {
        self.inputs
            .iter()
            .filter(|r| r.input_type.is_button())
            .map(|r| r.input_type)
            .collect()
    }

    /// 文字列表現
    pub fn to_string(&self) -> String {
        let frame_count_str = self
            .frame_count
            .map(|c| format!("{:02}", c))
            .unwrap_or_else(|| "??".to_string());

        let inputs_str: Vec<String> = self
            .inputs
            .iter()
            .filter(|r| r.input_type != InputType::Empty)
            .map(|r| format!("{}", r.input_type.to_string()))
            .collect();

        format!("[{}] {}", frame_count_str, inputs_str.join(" + "))
    }
}

/// 入力認識器
pub struct InputRecognizer {
    /// 通常テンプレート（3-15行目用）
    templates_normal: HashMap<InputType, Vec<RgbImage>>,
    /// インジケータ映り込み用テンプレート（0-2行目用）
    templates_with_indicator: HashMap<InputType, Vec<RgbImage>>,
}

impl InputRecognizer {
    /// 新しい認識器を作成
    pub fn new() -> Self {
        Self {
            templates_normal: HashMap::new(),
            templates_with_indicator: HashMap::new(),
        }
    }

    /// テンプレートを追加
    pub fn add_template(&mut self, template: Template) {
        let templates = match template.variant {
            TemplateVariant::Normal => &mut self.templates_normal,
            TemplateVariant::WithIndicator => &mut self.templates_with_indicator,
        };

        templates
            .entry(template.input_type)
            .or_insert_with(Vec::new)
            .push(template.image);
    }

    /// テンプレートディレクトリから読み込み
    pub fn load_templates<P: AsRef<Path>>(&mut self, template_dir: P) -> Result<()> {
        let template_dir = template_dir.as_ref();

        println!("テンプレートを読み込んでいます: {}", template_dir.display());

        for input_type in InputType::all() {
            // 通常テンプレート
            let type_dir = template_dir.join(input_type.name());
            let mut normal_count = 0;

            if type_dir.exists() {
                for entry in std::fs::read_dir(&type_dir)? {
                    let entry = entry?;
                    let path = entry.path();

                    if path.extension().and_then(|s| s.to_str()) == Some("png") {
                        match Template::load(input_type, &path, TemplateVariant::Normal) {
                            Ok(template) => {
                                self.add_template(template);
                                normal_count += 1;
                            }
                            Err(e) => {
                                eprintln!("  警告: テンプレート読み込みエラー: {}: {}", path.display(), e);
                            }
                        }
                    }
                }
            }

            // インジケータ映り込み用テンプレート
            let indicator_dir = template_dir.join(format!("{}_indicator", input_type.name()));
            let mut indicator_count = 0;

            if indicator_dir.exists() {
                for entry in std::fs::read_dir(&indicator_dir)? {
                    let entry = entry?;
                    let path = entry.path();

                    if path.extension().and_then(|s| s.to_str()) == Some("png") {
                        match Template::load(input_type, &path, TemplateVariant::WithIndicator) {
                            Ok(template) => {
                                self.add_template(template);
                                indicator_count += 1;
                            }
                            Err(e) => {
                                eprintln!("  警告: テンプレート読み込みエラー: {}: {}", path.display(), e);
                            }
                        }
                    }
                }
            }

            if normal_count > 0 || indicator_count > 0 {
                let msg = if indicator_count > 0 {
                    format!("{}: {}個 (通常: {}, インジケータ付: {})",
                           input_type.to_string(), normal_count + indicator_count, normal_count, indicator_count)
                } else {
                    format!("{}: {}個のテンプレートを読み込み", input_type.to_string(), normal_count)
                };
                println!("  {}", msg);
            } else {
                println!("  警告: {}のテンプレートが見つかりません", input_type.to_string());
            }
        }

        Ok(())
    }

    /// 画像との類似度を計算（正規化相互相関）
    fn calculate_similarity(&self, image: &RgbImage, template: &RgbImage) -> f64 {
        if image.dimensions() != template.dimensions() {
            return 0.0;
        }

        let (width, height) = image.dimensions();
        let mut sum_image = 0.0;
        let mut sum_template = 0.0;
        let mut sum_product = 0.0;
        let mut sum_image_sq = 0.0;
        let mut sum_template_sq = 0.0;
        let n = (width * height * 3) as f64;

        for y in 0..height {
            for x in 0..width {
                let img_pixel = image.get_pixel(x, y);
                let tpl_pixel = template.get_pixel(x, y);

                for i in 0..3 {
                    let img_val = img_pixel[i] as f64;
                    let tpl_val = tpl_pixel[i] as f64;

                    sum_image += img_val;
                    sum_template += tpl_val;
                    sum_product += img_val * tpl_val;
                    sum_image_sq += img_val * img_val;
                    sum_template_sq += tpl_val * tpl_val;
                }
            }
        }

        let mean_image = sum_image / n;
        let mean_template = sum_template / n;

        let numerator = sum_product - n * mean_image * mean_template;
        let denominator = ((sum_image_sq - n * mean_image * mean_image)
            * (sum_template_sq - n * mean_template * mean_template))
            .sqrt();

        if denominator == 0.0 {
            return 0.0;
        }

        (numerator / denominator).max(0.0).min(1.0)
    }

    /// セル画像を認識（行番号を考慮）
    pub fn recognize(&self, image: &RgbImage, row_index: u32) -> Result<RecognitionResult> {
        // 0-2行目はインジケータ映り込み用テンプレートを優先
        let use_indicator_templates = row_index <= 2;

        let mut best_match = RecognitionResult::new(InputType::Empty, 0.0);

        // インジケータ映り込み用テンプレートがある場合はそれを優先
        if use_indicator_templates && !self.templates_with_indicator.is_empty() {
            for (input_type, templates) in &self.templates_with_indicator {
                let mut max_similarity = 0.0;

                for template in templates {
                    let similarity = self.calculate_similarity(image, template);
                    if similarity > max_similarity {
                        max_similarity = similarity;
                    }
                }

                if max_similarity > best_match.confidence {
                    best_match = RecognitionResult::new(*input_type, max_similarity);
                }
            }

            // インジケータ用テンプレートで十分な信頼度が得られた場合は終了
            if best_match.confidence > 0.7 {
                return Ok(best_match);
            }
        }

        // 通常テンプレートでも試す
        for (input_type, templates) in &self.templates_normal {
            let mut max_similarity = 0.0;

            for template in templates {
                let similarity = self.calculate_similarity(image, template);
                if similarity > max_similarity {
                    max_similarity = similarity;
                }
            }

            if max_similarity > best_match.confidence {
                best_match = RecognitionResult::new(*input_type, max_similarity);
            }
        }

        Ok(best_match)
    }

    /// 入力行全体を認識
    pub fn recognize_row(
        &self,
        row_index: u32,
        cell_images: &[RgbImage],
    ) -> Result<RowInputState> {
        let mut state = RowInputState::new(row_index);

        // 各セルを認識（列1-6の入力アイコン）
        // 行番号を考慮してテンプレートを選択
        for cell_image in cell_images {
            let result = self.recognize(cell_image, row_index)?;
            state.inputs.push(result);
        }

        Ok(state)
    }

    /// テンプレート数を取得
    pub fn template_count(&self) -> usize {
        let normal_count: usize = self.templates_normal.values().map(|v| v.len()).sum();
        let indicator_count: usize = self.templates_with_indicator.values().map(|v| v.len()).sum();
        normal_count + indicator_count
    }

    /// 各入力タイプのテンプレート数を取得
    pub fn template_count_by_type(&self) -> HashMap<InputType, usize> {
        let mut counts = HashMap::new();

        for (input_type, templates) in &self.templates_normal {
            *counts.entry(*input_type).or_insert(0) += templates.len();
        }

        for (input_type, templates) in &self.templates_with_indicator {
            *counts.entry(*input_type).or_insert(0) += templates.len();
        }

        counts
    }

    /// インジケータ映り込み用テンプレートがあるか確認
    pub fn has_indicator_templates(&self) -> bool {
        !self.templates_with_indicator.is_empty()
    }
}

impl Default for InputRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_type_is_direction() {
        assert!(InputType::DirectionUp.is_direction());
        assert!(InputType::DirectionRight.is_direction());
        assert!(!InputType::ButtonA1.is_direction());
        assert!(!InputType::Empty.is_direction());
    }

    #[test]
    fn test_input_type_is_button() {
        assert!(InputType::ButtonA1.is_button());
        assert!(InputType::ButtonB.is_button());
        assert!(!InputType::DirectionUp.is_button());
        assert!(!InputType::Empty.is_button());
    }

    #[test]
    fn test_row_input_state() {
        let mut state = RowInputState::new(0);
        state.frame_count = Some(42);
        state.inputs.push(RecognitionResult::new(InputType::DirectionRight, 0.95));
        state.inputs.push(RecognitionResult::new(InputType::ButtonA1, 0.92));

        assert_eq!(state.get_direction(), Some(InputType::DirectionRight));
        assert_eq!(state.get_buttons(), vec![InputType::ButtonA1]);
    }
}
