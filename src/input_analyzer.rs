use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, ImageEncoder, Rgb, RgbImage};
use image::codecs::png::{PngEncoder, CompressionType, FilterType};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// PNG画像を非圧縮で保存（DynamicImage用）
#[allow(dead_code)]
fn save_png_uncompressed(img: &DynamicImage, path: &Path) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("ファイルの作成に失敗: {}", path.display()))?;
    let writer = BufWriter::new(file);

    let encoder = PngEncoder::new_with_quality(
        writer,
        CompressionType::Fast,  // 非圧縮（最速）
        FilterType::NoFilter,   // フィルタなし
    );

    let color_type = match img.color() {
        image::ColorType::Rgb8 => image::ExtendedColorType::Rgb8,
        image::ColorType::Rgba8 => image::ExtendedColorType::Rgba8,
        image::ColorType::L8 => image::ExtendedColorType::L8,
        image::ColorType::La8 => image::ExtendedColorType::La8,
        _ => image::ExtendedColorType::Rgba8,
    };

    encoder.write_image(
        img.as_bytes(),
        img.width(),
        img.height(),
        color_type,
    ).context("PNG画像の書き込みに失敗しました")?;

    Ok(())
}

/// PNG画像を非圧縮で保存（RgbImage用）
fn save_png_uncompressed_rgb(img: &RgbImage, path: &Path) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("ファイルの作成に失敗: {}", path.display()))?;
    let writer = BufWriter::new(file);

    let encoder = PngEncoder::new_with_quality(
        writer,
        CompressionType::Fast,  // 非圧縮（最速）
        FilterType::NoFilter,   // フィルタなし
    );

    encoder.write_image(
        img.as_raw(),
        img.width(),
        img.height(),
        image::ExtendedColorType::Rgb8,
    ).context("PNG画像の書き込みに失敗しました")?;

    Ok(())
}

/// セルサイズの定数
pub const CELL_WIDTH: u32 = 48;
pub const CELL_HEIGHT: u32 = 48;

/// 入力インジケータの領域設定
#[derive(Debug, Clone)]
pub struct InputIndicatorRegion {
    /// 開始X座標
    pub x: u32,
    /// 開始Y座標
    pub y: u32,
    /// 幅
    pub width: u32,
    /// 高さ
    pub height: u32,
    /// 行数
    pub rows: u32,
    /// 列数
    pub cols: u32,
}

impl Default for InputIndicatorRegion {
    fn default() -> Self {
        Self {
            x: 204,
            y: 182,
            width: 336,  // 48 * 7
            height: 768, // 48 * 16
            rows: 16,
            cols: 7,
        }
    }
}

impl InputIndicatorRegion {
    /// 新しい入力インジケータ領域を作成
    pub fn new(x: u32, y: u32, width: u32, height: u32, rows: u32, cols: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            rows,
            cols,
        }
    }

    /// 1セルの幅を取得
    pub fn cell_width(&self) -> u32 {
        CELL_WIDTH
    }

    /// 1セルの高さを取得
    pub fn cell_height(&self) -> u32 {
        CELL_HEIGHT
    }

    /// 指定された行と列のセル座標を取得（画像全体での座標）
    pub fn get_cell_position(&self, row: u32, col: u32) -> (u32, u32, u32, u32) {
        let cell_width = self.cell_width();
        let cell_height = self.cell_height();

        let cell_x = self.x + col * cell_width;
        let cell_y = self.y + row * cell_height;

        (cell_x, cell_y, cell_width, cell_height)
    }
}

/// 入力行データ（1行分の入力情報）
#[derive(Debug, Clone)]
pub struct InputRow {
    /// 行番号（0が最新、rows-1が最古）
    pub row_index: u32,
    /// フレームカウント（1列目）
    pub frame_count_image: RgbImage,
    /// 入力アイコン（2～7列目）
    pub input_icons: Vec<RgbImage>,
}

impl InputRow {
    /// 新しい入力行を作成
    pub fn new(row_index: u32, frame_count_image: RgbImage, input_icons: Vec<RgbImage>) -> Self {
        Self {
            row_index,
            frame_count_image,
            input_icons,
        }
    }

    /// フレームカウント画像を保存（非圧縮PNG）
    pub fn save_frame_count<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        save_png_uncompressed_rgb(&self.frame_count_image, output_path.as_ref())
            .context("フレームカウント画像の保存に失敗しました")?;
        Ok(())
    }

    /// 入力アイコンを保存（非圧縮PNG）
    pub fn save_input_icon<P: AsRef<Path>>(&self, col_index: usize, output_path: P) -> Result<()> {
        if col_index >= self.input_icons.len() {
            anyhow::bail!("列インデックスが範囲外です: {}", col_index);
        }

        save_png_uncompressed_rgb(&self.input_icons[col_index], output_path.as_ref())
            .context("入力アイコンの保存に失敗しました")?;
        Ok(())
    }

    /// すべての入力アイコンを指定ディレクトリに保存
    pub fn save_all_icons<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)
            .context("出力ディレクトリの作成に失敗しました")?;

        // フレームカウント画像を保存
        let frame_count_path = output_dir.join(format!("row{:02}_col00_frame_count.png", self.row_index));
        self.save_frame_count(&frame_count_path)?;

        // 入力アイコンを保存
        for (i, _icon) in self.input_icons.iter().enumerate() {
            let icon_path = output_dir.join(format!("row{:02}_col{:02}_input.png", self.row_index, i + 1));
            self.save_input_icon(i, &icon_path)?;
        }

        Ok(())
    }
}

/// 入力インジケータ解析器
pub struct InputAnalyzer {
    region: InputIndicatorRegion,
}

impl InputAnalyzer {
    /// 新しい入力解析器を作成
    pub fn new(region: InputIndicatorRegion) -> Self {
        Self { region }
    }

    /// デフォルト設定で入力解析器を作成
    pub fn default() -> Self {
        Self {
            region: InputIndicatorRegion::default(),
        }
    }

    /// 画像から入力インジケータ領域全体を抽出
    pub fn extract_indicator_region(&self, image: &DynamicImage) -> Result<RgbImage> {
        let (img_width, img_height) = image.dimensions();

        // 範囲チェック
        if self.region.x + self.region.width > img_width
            || self.region.y + self.region.height > img_height {
            anyhow::bail!(
                "入力インジケータ領域が画像範囲外です: 画像サイズ={}x{}, 領域=({},{})から({},{})",
                img_width, img_height,
                self.region.x, self.region.y,
                self.region.x + self.region.width,
                self.region.y + self.region.height
            );
        }

        let cropped = image.crop_imm(
            self.region.x,
            self.region.y,
            self.region.width,
            self.region.height,
        );

        Ok(cropped.to_rgb8())
    }

    /// 画像から指定された行の入力データを抽出
    pub fn extract_input_row(&self, image: &DynamicImage, row: u32) -> Result<InputRow> {
        if row >= self.region.rows {
            anyhow::bail!("行番号が範囲外です: {}/{}", row, self.region.rows);
        }

        let mut input_icons = Vec::new();

        // 各列のセルを抽出
        for col in 0..self.region.cols {
            let (cell_x, cell_y, cell_width, cell_height) = self.region.get_cell_position(row, col);

            // セル画像を抽出
            let cell_image = image.crop_imm(cell_x, cell_y, cell_width, cell_height);
            let rgb_image = cell_image.to_rgb8();

            if col == 0 {
                // 1列目はフレームカウント
                // 次のイテレーションで input_icons に追加するのではなく、
                // InputRow の frame_count_image として保持
                continue;
            } else {
                // 2～7列目は入力アイコン
                input_icons.push(rgb_image);
            }
        }

        // フレームカウント画像を取得（1列目）
        let (cell_x, cell_y, cell_width, cell_height) = self.region.get_cell_position(row, 0);
        let frame_count_cell = image.crop_imm(cell_x, cell_y, cell_width, cell_height);
        let frame_count_image = frame_count_cell.to_rgb8();

        Ok(InputRow::new(row, frame_count_image, input_icons))
    }

    /// 画像からすべての入力行を抽出
    pub fn extract_all_rows(&self, image: &DynamicImage) -> Result<Vec<InputRow>> {
        let mut rows = Vec::new();

        for row in 0..self.region.rows {
            let input_row = self.extract_input_row(image, row)?;
            rows.push(input_row);
        }

        Ok(rows)
    }

    /// 画像ファイルから入力インジケータ領域を抽出して保存
    pub fn extract_and_save_indicator<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input_path: P,
        output_path: Q,
    ) -> Result<()> {
        let image = image::open(input_path.as_ref())
            .context("画像ファイルを開けませんでした")?;

        // 出力ディレクトリを作成
        if let Some(parent) = output_path.as_ref().parent() {
            std::fs::create_dir_all(parent)
                .context("出力ディレクトリの作成に失敗しました")?;
        }

        let indicator = self.extract_indicator_region(&image)?;
        save_png_uncompressed_rgb(&indicator, output_path.as_ref())
            .context("入力インジケータ画像の保存に失敗しました")?;

        println!("入力インジケータ領域を保存しました: {}", output_path.as_ref().display());
        Ok(())
    }

    /// 画像ファイルからすべての入力行を抽出して保存
    pub fn extract_and_save_all_rows<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input_path: P,
        output_dir: Q,
    ) -> Result<Vec<InputRow>> {
        let input_path = input_path.as_ref();
        let output_dir = output_dir.as_ref();

        println!("画像を読み込んでいます: {}", input_path.display());
        let image = image::open(input_path)
            .context("画像ファイルを開けませんでした")?;

        println!("入力行を抽出中...");
        let rows = self.extract_all_rows(&image)?;

        // 出力ディレクトリを作成
        std::fs::create_dir_all(output_dir)
            .context("出力ディレクトリの作成に失敗しました")?;

        println!("抽出した入力行を保存中...");
        for row in &rows {
            row.save_all_icons(output_dir)?;
        }

        println!("\n✓ {}行の入力データを抽出しました", rows.len());
        println!("  出力先: {}", output_dir.display());

        Ok(rows)
    }

    /// 入力インジケータ領域にグリッド線を描画したデバッグ画像を作成
    pub fn create_debug_image(&self, image: &DynamicImage) -> Result<RgbImage> {
        let mut debug_image = image.to_rgb8();
        let (img_width, img_height) = image.dimensions();

        // 領域の外枠を描画（赤）
        let red = Rgb([255, 0, 0]);
        self.draw_rectangle(
            &mut debug_image,
            self.region.x,
            self.region.y,
            self.region.width,
            self.region.height,
            red,
        );

        // グリッド線を描画（緑）
        let green = Rgb([0, 255, 0]);
        let cell_width = self.region.cell_width();
        let cell_height = self.region.cell_height();

        // 縦線
        for col in 0..=self.region.cols {
            let x = self.region.x + col * cell_width;
            if x < img_width {
                self.draw_vertical_line(&mut debug_image, x, self.region.y, self.region.height, green);
            }
        }

        // 横線
        for row in 0..=self.region.rows {
            let y = self.region.y + row * cell_height;
            if y < img_height {
                self.draw_horizontal_line(&mut debug_image, self.region.x, y, self.region.width, green);
            }
        }

        Ok(debug_image)
    }

    /// 矩形を描画
    fn draw_rectangle(&self, image: &mut RgbImage, x: u32, y: u32, width: u32, height: u32, color: Rgb<u8>) {
        let (img_width, img_height) = image.dimensions();

        // 上下の水平線
        for dx in 0..width {
            let px = x + dx;
            if px < img_width {
                if y < img_height {
                    image.put_pixel(px, y, color);
                }
                if y + height < img_height {
                    image.put_pixel(px, y + height, color);
                }
            }
        }

        // 左右の垂直線
        for dy in 0..height {
            let py = y + dy;
            if py < img_height {
                if x < img_width {
                    image.put_pixel(x, py, color);
                }
                if x + width < img_width {
                    image.put_pixel(x + width, py, color);
                }
            }
        }
    }

    /// 垂直線を描画
    fn draw_vertical_line(&self, image: &mut RgbImage, x: u32, y: u32, height: u32, color: Rgb<u8>) {
        let (img_width, img_height) = image.dimensions();
        if x >= img_width {
            return;
        }

        for dy in 0..height {
            let py = y + dy;
            if py < img_height {
                image.put_pixel(x, py, color);
            }
        }
    }

    /// 水平線を描画
    fn draw_horizontal_line(&self, image: &mut RgbImage, x: u32, y: u32, width: u32, color: Rgb<u8>) {
        let (img_width, img_height) = image.dimensions();
        if y >= img_height {
            return;
        }

        for dx in 0..width {
            let px = x + dx;
            if px < img_width {
                image.put_pixel(px, y, color);
            }
        }
    }

    /// デバッグ画像を保存
    pub fn save_debug_image<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input_path: P,
        output_path: Q,
    ) -> Result<()> {
        let image = image::open(input_path.as_ref())
            .context("画像ファイルを開けませんでした")?;

        // 出力ディレクトリを作成
        if let Some(parent) = output_path.as_ref().parent() {
            std::fs::create_dir_all(parent)
                .context("出力ディレクトリの作成に失敗しました")?;
        }

        let debug_image = self.create_debug_image(&image)?;
        save_png_uncompressed_rgb(&debug_image, output_path.as_ref())
            .context("デバッグ画像の保存に失敗しました")?;

        println!("デバッグ画像を保存しました: {}", output_path.as_ref().display());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_indicator_region_default() {
        let region = InputIndicatorRegion::default();
        assert_eq!(region.x, 204);
        assert_eq!(region.y, 182);
        assert_eq!(region.width, 336);
        assert_eq!(region.height, 768);
        assert_eq!(region.rows, 16);
        assert_eq!(region.cols, 7);
    }

    #[test]
    fn test_cell_dimensions() {
        let region = InputIndicatorRegion::default();
        let cell_width = region.cell_width();
        let cell_height = region.cell_height();

        assert_eq!(cell_width, 48);
        assert_eq!(cell_height, 48);
    }

    #[test]
    fn test_get_cell_position() {
        let region = InputIndicatorRegion::default();
        let (x, y, w, h) = region.get_cell_position(0, 0);

        assert_eq!(x, 204);
        assert_eq!(y, 182);
        assert_eq!(w, 48);
        assert_eq!(h, 48);
    }
}
