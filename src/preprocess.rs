//! SIMD image resizing: antialiased bilinear for classification/cells,
//! fixed-kernel bilinear for YOLO letterboxing.
use anyhow::{Context, Result, ensure};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{GrayImage, ImageDecoder, ImageReader, Luma, RgbImage};
use std::io::Cursor;

pub const MAX_PIXELS: u64 = 24_000_000;

pub fn decode(bytes: &[u8]) -> Result<RgbImage> {
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(12_000);
    limits.max_image_height = Some(12_000);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let mut decoder = reader
        .into_decoder()
        .context("无法解码图片，请上传 JPG、PNG、WebP 或 BMP")?;
    let (w, h) = decoder.dimensions();
    ensure!(
        w > 0 && h > 0 && u64::from(w) * u64::from(h) <= MAX_PIXELS,
        "图片不能超过 2400 万像素，且宽高均不能超过 12000"
    );
    let orientation = decoder.orientation()?;
    let mut image = image::DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image.into_rgb8())
}

#[derive(Clone, Copy)]
pub struct Letterbox {
    pub sx: f32,
    pub sy: f32,
    pub left: f32,
    pub top: f32,
}

/// Reuses SIMD scratch space, the resized image and the NCHW input buffer.
/// Each pipeline owns one instance; no shared global buffers are used.
pub struct Preprocessor {
    resizer: Resizer,
    rgb: RgbImage,
    tensor: Vec<f32>,
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self {
            resizer: Resizer::new(),
            rgb: RgbImage::new(1, 1),
            tensor: Vec::with_capacity(3 * 640 * 640),
        }
    }
}

impl Preprocessor {
    fn resize_rgb(&mut self, image: &RgbImage, width: u32, height: u32, algorithm: ResizeAlg) {
        if self.rgb.dimensions() != (width, height) {
            self.rgb = RgbImage::new(width, height);
        }
        self.resizer
            .resize(
                image,
                &mut self.rgb,
                &ResizeOptions::new().resize_alg(algorithm),
            )
            .expect("nonempty RGB images with matching pixel types");
    }

    // Fuse padding, RGB -> NCHW and normalization. Do not create an RGB canvas
    // or make a second normalization pass over the completed tensor.
    fn write_tensor(&mut self, size: usize, left: usize, top: usize, classifier: bool) -> &[f32] {
        self.tensor.resize(3 * size * size, 0.0);
        let pad = if classifier { 128 } else { 114 };
        let width = self.rgb.width() as usize;
        let height = self.rgb.height() as usize;
        for (c, plane) in self.tensor.chunks_exact_mut(size * size).enumerate() {
            let lut: [f32; 256] = std::array::from_fn(|value| {
                let value = value as f32 / 255.0;
                if classifier {
                    (value - [0.485, 0.456, 0.406][c]) / [0.229, 0.224, 0.225][c]
                } else {
                    value
                }
            });
            plane.fill(lut[pad]);
            for (source, destination) in self
                .rgb
                .as_raw()
                .chunks_exact(width * 3)
                .zip(plane[top * size..(top + height) * size].chunks_exact_mut(size))
            {
                for (pixel, value) in source
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .zip(&mut destination[left..left + width])
                {
                    *value = lut[pixel[c] as usize];
                }
            }
        }
        &self.tensor
    }

    pub fn classifier(&mut self, image: &RgbImage) -> &[f32] {
        let ratio = f64::from(image.width()) / f64::from(image.height());
        let (w, h) = if ratio > 1.0 {
            (256, (256.0 / ratio).round_ties_even().max(1.0) as u32)
        } else {
            ((256.0 * ratio).round_ties_even().max(1.0) as u32, 256)
        };
        self.resize_rgb(image, w, h, ResizeAlg::Convolution(FilterType::Bilinear));
        let left = (f64::from(256 - w) * 0.5).round_ties_even() as usize;
        let top = (f64::from(256 - h) * 0.5).round_ties_even() as usize;
        self.write_tensor(256, left, top, true)
    }

    pub fn detector(&mut self, image: &RgbImage) -> (&[f32], Letterbox) {
        let r = (640.0 / f64::from(image.width())).min(640.0 / f64::from(image.height()));
        let w = (f64::from(image.width()) * r).round_ties_even().max(1.0) as u32;
        let h = (f64::from(image.height()) * r).round_ties_even().max(1.0) as u32;
        let left = ((640 - w) as f64 / 2.0 - 0.1).round_ties_even() as usize;
        let top = ((640 - h) as f64 / 2.0 - 0.1).round_ties_even() as usize;
        self.resize_rgb(image, w, h, ResizeAlg::Interpolation(FilterType::Bilinear));
        let transform = Letterbox {
            sx: w as f32 / image.width() as f32,
            sy: h as f32 / image.height() as f32,
            left: left as f32,
            top: top as f32,
        };
        (self.write_tensor(640, left, top, false), transform)
    }
}

pub fn cell(image: &RgbImage) -> Vec<f32> {
    let gray = GrayImage::from_fn(image.width(), image.height(), |x, y| {
        let v = *image.get_pixel(x, y).0.iter().min().unwrap();
        Luma([v])
    });
    let mut resized = GrayImage::new(32, 48);
    Resizer::new()
        .resize(
            &gray,
            &mut resized,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
        )
        .expect("nonempty grayscale images with matching pixel types");
    let mut hist = [0usize; 256];
    for p in resized.pixels() {
        hist[p[0] as usize] += 1;
    }
    let cut = 32 * 48 / 100;
    let mut low_hist = hist;
    let mut remaining = cut;
    for v in &mut low_hist {
        let n = remaining.min(*v);
        *v -= n;
        remaining -= n;
    }
    remaining = cut;
    for v in low_hist.iter_mut().rev() {
        let n = remaining.min(*v);
        *v -= n;
        remaining -= n;
    }
    let low = low_hist.iter().position(|&n| n > 0).unwrap_or(0);
    let high = low_hist.iter().rposition(|&n| n > 0).unwrap_or(255);
    let scale = 255.0 / (high as f64 - low as f64);
    resized
        .pixels()
        .map(|p| {
            let v = if high <= low {
                p[0]
            } else {
                (f64::from(p[0]) * scale - low as f64 * scale).clamp(0.0, 255.0) as u8
            };
            f32::from(v) / 255.0
        })
        .collect()
}

pub fn crop_bounds(b: [f32; 4], w: u32, h: u32, mx: f32, my: f32) -> [u32; 4] {
    let px = (b[2] - b[0]) * mx;
    let py = (b[3] - b[1]) * my;
    [
        (b[0] - px).floor().clamp(0.0, w as f32) as u32,
        (b[1] - py).floor().clamp(0.0, h as f32) as u32,
        (b[2] + px).ceil().clamp(0.0, w as f32) as u32,
        (b[3] + py).ceil().clamp(0.0, h as f32) as u32,
    ]
}

pub fn crop(image: &RgbImage, b: [u32; 4]) -> RgbImage {
    image::imageops::crop_imm(image, b[0], b[1], b[2] - b[0], b[3] - b[1]).to_image()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    #[test]
    fn letterbox_and_darkest_cell() {
        let image = RgbImage::from_pixel(200, 100, Rgb([128, 64, 192]));
        let mut preprocessor = Preprocessor::default();
        let (t, s) = preprocessor.detector(&image);
        assert_eq!((s.left, s.top, s.sx, s.sy), (0.0, 160.0, 3.2, 3.2));
        assert_eq!(t[0], 114.0 / 255.0);
        assert_eq!(t[160 * 640], 128.0 / 255.0);
        assert!(cell(&image).iter().all(|&v| v == 64.0 / 255.0));
    }
    #[test]
    fn crop_rounds_outward_and_clamps() {
        assert_eq!(
            crop_bounds([0.3, 1.2, 9.1, 8.3], 10, 10, 0.1, 0.1),
            [0, 0, 10, 10]
        );
    }

    #[test]
    fn reused_tensor_clears_padding_and_keeps_channel_normalization() {
        let mut prep = Preprocessor::default();
        let wide = RgbImage::from_pixel(2, 1, Rgb([255, 0, 0]));
        let tall = RgbImage::from_pixel(1, 2, Rgb([0, 255, 0]));
        prep.detector(&wide);
        let tensor = prep.classifier(&tall);
        assert_eq!(tensor.len(), 3 * 256 * 256);
        assert_eq!(tensor[0], (128.0 / 255.0 - 0.485) / 0.229);
        assert_eq!(tensor[64], (0.0 - 0.485) / 0.229);
        assert_eq!(tensor[256 * 256 + 64], (1.0 - 0.456) / 0.224);
        let (tensor, _) = prep.detector(&wide);
        assert_eq!(tensor[0], 114.0 / 255.0);
        assert_eq!(tensor[160 * 640], 1.0);
        assert_eq!(tensor[640 * 640 + 160 * 640], 0.0);
    }
}
