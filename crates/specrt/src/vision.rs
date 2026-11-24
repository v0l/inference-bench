use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::path::Path;
use std::process::Command;

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct SizeBounds {
    pub shortest_edge: usize,
    pub longest_edge: usize,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Preprocessor {
    pub size: SizeBounds,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    #[serde(default = "default_fps")]
    pub fps: f64,
    #[serde(default = "default_min_frames")]
    pub min_frames: usize,
    #[serde(default = "default_max_frames")]
    pub max_frames: usize,
}

fn default_fps() -> f64 {
    2.0
}
fn default_min_frames() -> usize {
    4
}
fn default_max_frames() -> usize {
    768
}

impl Preprocessor {
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("preprocessor config")
    }

    pub fn unused() -> Self {
        Preprocessor {
            size: SizeBounds { shortest_edge: 0, longest_edge: 0 },
            patch_size: 1,
            temporal_patch_size: 1,
            merge_size: 1,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
            fps: default_fps(),
            min_frames: default_min_frames(),
            max_frames: default_max_frames(),
        }
    }

    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    pub fn patch_dim(&self) -> usize {
        3 * self.temporal_patch_size * self.patch_size * self.patch_size
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Patches {
    pub grid: [usize; 3],
    pub data: Vec<f32>,
    pub layout: [usize; 2],
}

impl Patches {
    pub fn tokens(&self, merge_size: usize) -> usize {
        self.grid.iter().product::<usize>() / (merge_size * merge_size)
    }
}

#[derive(Clone)]
pub struct Frame {
    pub height: usize,
    pub width: usize,
    pub rgb: Vec<u8>,
}

fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - x.signum() } else { r }
}

pub fn smart_resize(frames: usize, temporal: usize, height: usize, width: usize, factor: usize, min_pixels: usize, max_pixels: usize) -> Result<(usize, usize)> {
    let (mut height, mut width) = (height as f64, width as f64);
    let f = factor as f64;
    if height < f || width < f {
        let scale = (f / height).max(f / width);
        height = (height * scale).trunc();
        width = (width * scale).trunc();
    }
    ensure!(height.max(width) / height.min(width) <= 200.0, "aspect ratio above 200");
    let mut h_bar = round_half_even(height / f) * f;
    let mut w_bar = round_half_even(width / f) * f;
    let t_bar = round_half_even(frames as f64 / temporal as f64) * temporal as f64;
    let (frames, max_pixels, min_pixels) = (frames as f64, max_pixels as f64, min_pixels as f64);
    if t_bar * h_bar * w_bar > max_pixels {
        let beta = (frames * height * width / max_pixels).sqrt();
        h_bar = f.max((height / beta / f).floor() * f);
        w_bar = f.max((width / beta / f).floor() * f);
    } else if t_bar * h_bar * w_bar < min_pixels {
        let beta = (min_pixels / (frames * height * width)).sqrt();
        h_bar = (height * beta / f).ceil() * f;
        w_bar = (width * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    Cubic,
    Triangle,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Cubic => 2.0,
            Filter::Triangle => 1.0,
        }
    }

    fn weight(self, x: f64) -> f64 {
        match self {
            Filter::Cubic => cubic(x),
            Filter::Triangle => (1.0 - x.abs()).max(0.0),
        }
    }
}

fn cubic(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

struct Coefficients {
    bounds: Vec<(usize, usize)>,
    kernel: Vec<i32>,
    size: usize,
    precision: u32,
}

fn coefficients(input: usize, output: usize, filter: Filter) -> Coefficients {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = filter.support() * filter_scale;
    let size = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(output);
    let mut weights = vec![0f64; output * size];
    let mut max_weight = 0f64;
    for xx in 0..output {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = (((center + support + 0.5) as i64).min(input as i64) as usize).saturating_sub(xmin);
        let mut total = 0.0;
        for x in 0..xmax {
            let v = filter.weight((x as f64 + xmin as f64 - center + 0.5) / filter_scale);
            weights[xx * size + x] = v;
            total += v;
        }
        for x in 0..xmax {
            weights[xx * size + x] /= total;
            max_weight = max_weight.max(weights[xx * size + x]);
        }
        bounds.push((xmin, xmax));
    }
    let mut precision = 0;
    while precision < 22 && ((0.5 + max_weight * (1u64 << (precision + 1)) as f64) as i64) < (1 << 15) {
        precision += 1;
    }
    let kernel = weights
        .iter()
        .map(|&w| {
            let n = w * (1u64 << precision) as f64;
            if n < 0.0 { (-0.5 + n) as i32 } else { (0.5 + n) as i32 }
        })
        .collect();
    Coefficients { bounds, kernel, size, precision }
}

fn clip(v: i64, precision: u32) -> u8 {
    (v >> precision).clamp(0, 255) as u8
}

pub fn resize_bicubic(src: &[u8], h: usize, w: usize, new_h: usize, new_w: usize) -> Vec<u8> {
    resize(src, h, w, new_h, new_w, Filter::Cubic)
}

pub fn resize_bilinear(src: &[u8], h: usize, w: usize, new_h: usize, new_w: usize) -> Vec<u8> {
    resize(src, h, w, new_h, new_w, Filter::Triangle)
}

fn resize(src: &[u8], h: usize, w: usize, new_h: usize, new_w: usize, filter: Filter) -> Vec<u8> {
    let horizontal = if new_w == w {
        src.to_vec()
    } else {
        let c = coefficients(w, new_w, filter);
        let init = 1i64 << (c.precision.max(1) - 1);
        let mut out = vec![0u8; h * new_w * 3];
        for y in 0..h {
            for x in 0..new_w {
                let (xmin, n) = c.bounds[x];
                let k = &c.kernel[x * c.size..];
                for ch in 0..3 {
                    let mut acc = init;
                    for i in 0..n {
                        acc += src[(y * w + xmin + i) * 3 + ch] as i64 * k[i] as i64;
                    }
                    out[(y * new_w + x) * 3 + ch] = clip(acc, c.precision);
                }
            }
        }
        out
    };
    if new_h == h {
        return horizontal;
    }
    let c = coefficients(h, new_h, filter);
    let init = 1i64 << (c.precision.max(1) - 1);
    let mut out = vec![0u8; new_h * new_w * 3];
    for y in 0..new_h {
        let (ymin, n) = c.bounds[y];
        let k = &c.kernel[y * c.size..];
        for x in 0..new_w {
            for ch in 0..3 {
                let mut acc = init;
                for i in 0..n {
                    acc += horizontal[((ymin + i) * new_w + x) * 3 + ch] as i64 * k[i] as i64;
                }
                out[(y * new_w + x) * 3 + ch] = clip(acc, c.precision);
            }
        }
    }
    out
}

pub fn patchify(cfg: &Preprocessor, frames: &[Vec<u8>], height: usize, width: usize) -> Patches {
    let (p, m, tp) = (cfg.patch_size, cfg.merge_size, cfg.temporal_patch_size);
    let n = frames.len();
    let padded = n.div_ceil(tp) * tp;
    let frame = |i: usize| &frames[i.min(n - 1)];
    let (gt, gh, gw) = (padded / tp, height / p, width / p);
    let scale: Vec<(f32, f32)> = (0..3).map(|c| (cfg.image_mean[c] * 255.0, cfg.image_std[c] * 255.0)).collect();
    let mut data = Vec::with_capacity(gt * gh * gw * cfg.patch_dim());
    for t in 0..gt {
        for bh in 0..gh / m {
            for bw in 0..gw / m {
                for mh in 0..m {
                    for mw in 0..m {
                        for c in 0..3 {
                            for dt in 0..tp {
                                let f = frame(t * tp + dt);
                                for py in 0..p {
                                    for px in 0..p {
                                        let y = (bh * m + mh) * p + py;
                                        let x = (bw * m + mw) * p + px;
                                        let v = f[(y * width + x) * 3 + c] as f32;
                                        data.push((v - scale[c].0) / scale[c].1);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Patches { grid: [gt, gh, gw], data, layout: [0, 0] }
}

pub fn preprocess_image(cfg: &Preprocessor, bytes: &[u8]) -> Result<Patches> {
    let format = image::guess_format(bytes).map_err(|_| anyhow::anyhow!("the bytes are not a PNG, JPEG, GIF, BMP, TIFF or WebP image: send videos as video_url"))?;
    let img = image::load_from_memory_with_format(bytes, format).context("decode image")?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let frame = Frame { height: h, width: w, rgb: img.into_raw() };
    preprocess_frames(cfg, &[frame], 1)
}

fn preprocess_frames(cfg: &Preprocessor, frames: &[Frame], temporal: usize) -> Result<Patches> {
    ensure!(!frames.is_empty(), "no frames");
    let (h, w) = (frames[0].height, frames[0].width);
    let (nh, nw) = smart_resize(frames.len().max(temporal), temporal, h, w, cfg.factor(), cfg.size.shortest_edge, cfg.size.longest_edge)?;
    let resized: Vec<Vec<u8>> = frames
        .iter()
        .map(|f| if (nh, nw) == (h, w) { f.rgb.clone() } else { resize_bicubic(&f.rgb, h, w, nh, nw) })
        .collect();
    Ok(patchify(cfg, &resized, nh, nw))
}

#[derive(Clone, Debug, PartialEq)]
pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub frames: usize,
}

pub fn sample_indices(cfg: &Preprocessor, total: usize, fps: f64) -> Vec<usize> {
    let wanted = (total as f64 / fps * cfg.fps) as usize;
    let count = wanted.max(cfg.min_frames).min(cfg.max_frames).min(total);
    if count == 1 {
        return vec![0];
    }
    (0..count).map(|i| round_half_even(i as f64 * (total - 1) as f64 / (count - 1) as f64) as usize).collect()
}

pub fn timestamps(indices: &[usize], fps: f64, temporal: usize) -> Vec<f64> {
    let mut idx = indices.to_vec();
    while !idx.len().is_multiple_of(temporal) {
        idx.push(*idx.last().unwrap());
    }
    idx.chunks(temporal).map(|c| (c[0] as f64 / fps + c[temporal - 1] as f64 / fps) / 2.0).collect()
}

pub fn probe_video(path: &Path) -> Result<VideoInfo> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json"])
        .arg(path)
        .output()
        .context("run ffprobe")?;
    ensure!(out.status.success(), "ffprobe failed: {}", String::from_utf8_lossy(&out.stderr));
    let json: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    let s = json["streams"].get(0).context("no video stream")?;
    let rate = s["r_frame_rate"].as_str().unwrap_or("0/1");
    let (num, den) = rate.split_once('/').unwrap_or((rate, "1"));
    let fps = num.parse::<f64>()? / den.parse::<f64>()?;
    ensure!(fps > 0.0, "video has no frame rate");
    let int = |k: &str| s[k].as_u64().map(|v| v as usize).or_else(|| s[k].as_str().and_then(|v| v.parse().ok()));
    Ok(VideoInfo {
        width: int("width").context("width")?,
        height: int("height").context("height")?,
        fps,
        frames: int("nb_read_frames").context("frame count")?,
    })
}

pub fn decode_frames(path: &Path, info: &VideoInfo, indices: &[usize]) -> Result<Vec<Frame>> {
    let mut unique: Vec<usize> = indices.to_vec();
    unique.dedup();
    let select: Vec<String> = unique.iter().map(|i| format!("eq(n\\,{i})")).collect();
    let filter = format!("select='{}'", select.join("+"));
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-vf", &filter, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .context("run ffmpeg")?;
    ensure!(out.status.success(), "ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr));
    let size = info.height * info.width * 3;
    ensure!(!out.stdout.is_empty() && out.stdout.len() % size == 0, "ffmpeg returned {} bytes", out.stdout.len());
    let decoded: Vec<&[u8]> = out.stdout.chunks(size).collect();
    ensure!(decoded.len() == unique.len(), "decoded {} of {} frames", decoded.len(), unique.len());
    let mut ordered = Vec::with_capacity(indices.len());
    for i in indices {
        let pos = unique.binary_search(i).map_err(|_| anyhow::anyhow!("frame {i} missing"))?;
        ordered.push(Frame { height: info.height, width: info.width, rgb: decoded[pos].to_vec() });
    }
    Ok(ordered)
}

#[derive(Clone, Debug, PartialEq)]
pub struct VideoPatches {
    pub patches: Patches,
    pub timestamps: Vec<f64>,
}

pub fn preprocess_video_file(cfg: &Preprocessor, path: &Path) -> Result<VideoPatches> {
    let info = probe_video(path)?;
    if info.frames == 0 {
        bail!("video has no frames");
    }
    let indices = sample_indices(cfg, info.frames, info.fps);
    let frames = decode_frames(path, &info, &indices)?;
    let patches = preprocess_frames(cfg, &frames, cfg.temporal_patch_size)?;
    Ok(VideoPatches { patches, timestamps: timestamps(&indices, info.fps, cfg.temporal_patch_size) })
}

pub fn preprocess_video(cfg: &Preprocessor, bytes: &[u8]) -> Result<VideoPatches> {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let path = std::env::temp_dir().join(format!("specrt-video-{}-{nanos}.bin", std::process::id()));
    std::fs::write(&path, bytes)?;
    let out = preprocess_video_file(cfg, &path);
    let _ = std::fs::remove_file(&path);
    out
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaTokens {
    pub image: String,
    pub video: String,
    pub start: String,
    pub end: String,
    pub placeholder: Option<String>,
    pub wrap: String,
    pub tile_tokens: usize,
    pub grid: Option<GridTokens>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GridTokens {
    pub tile_x: String,
    pub tile_y: String,
    pub global: String,
}

impl Default for MediaTokens {
    fn default() -> Self {
        MediaTokens {
            image: "<|image_pad|>".into(),
            video: "<|video_pad|>".into(),
            start: "<|vision_start|>".into(),
            end: "<|vision_end|>".into(),
            placeholder: None,
            wrap: String::new(),
            tile_tokens: 0,
            grid: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanAndScan {
    pub min_crop: usize,
    pub max_crops: usize,
    pub min_ratio: f64,
}

impl Default for PanAndScan {
    fn default() -> Self {
        PanAndScan { min_crop: 256, max_crops: 4, min_ratio: 1.2 }
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct TileSize {
    pub height: usize,
    pub width: usize,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct TileConfig {
    pub size: TileSize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    #[serde(default)]
    pub pan_and_scan_min_crop_size: Option<usize>,
    #[serde(default)]
    pub pan_and_scan_max_num_crops: Option<usize>,
    #[serde(default)]
    pub pan_and_scan_min_ratio_to_activate: Option<f64>,
    #[serde(default)]
    pub max_patches: Option<usize>,
    #[serde(default)]
    pub resize_to_max_canvas: bool,
}

impl TileConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("preprocessor config")
    }

    pub fn pan_and_scan(&self) -> PanAndScan {
        let d = PanAndScan::default();
        PanAndScan {
            min_crop: self.pan_and_scan_min_crop_size.unwrap_or(d.min_crop),
            max_crops: self.pan_and_scan_max_num_crops.unwrap_or(d.max_crops),
            min_ratio: self.pan_and_scan_min_ratio_to_activate.unwrap_or(d.min_ratio),
        }
    }
}

pub fn pan_and_scan_boxes(height: usize, width: usize, p: PanAndScan) -> Vec<[usize; 4]> {
    let landscape = width >= height;
    let (long, short) = if landscape { (width, height) } else { (height, width) };
    if (long as f64) / (short as f64) < p.min_ratio {
        return Vec::new();
    }
    let ideal = ((long as f64 / short as f64) + 0.5).floor() as usize;
    let crops = ideal.min(long / p.min_crop).max(2).min(p.max_crops);
    let (nw, nh) = if landscape { (crops, 1) } else { (1, crops) };
    let (cw, ch) = (width.div_ceil(nw), height.div_ceil(nh));
    if cw.min(ch) < p.min_crop {
        return Vec::new();
    }
    let mut out = Vec::new();
    for iy in 0..nh {
        for ix in 0..nw {
            let (y, x) = (iy * ch, ix * cw);
            out.push([y, x, (y + ch).min(height), (x + cw).min(width)]);
        }
    }
    out
}

fn crop_rgb(frame: &Frame, b: [usize; 4]) -> Frame {
    let (h, w) = (b[2] - b[0], b[3] - b[1]);
    let mut rgb = Vec::with_capacity(h * w * 3);
    for y in b[0]..b[2] {
        rgb.extend_from_slice(&frame.rgb[(y * frame.width + b[1]) * 3..(y * frame.width + b[3]) * 3]);
    }
    Frame { height: h, width: w, rgb }
}

pub fn tile_patches(cfg: &TileConfig, patch: usize, tiles: &[Vec<u8>]) -> Patches {
    let (h, w) = (cfg.size.height, cfg.size.width);
    let (gh, gw) = (h / patch, w / patch);
    let scale: Vec<(f32, f32)> = (0..3).map(|c| (cfg.image_mean[c] * 255.0, cfg.image_std[c] * 255.0)).collect();
    let mut data = Vec::with_capacity(tiles.len() * gh * gw * 3 * patch * patch);
    for tile in tiles {
        for by in 0..gh {
            for bx in 0..gw {
                for c in 0..3 {
                    for py in 0..patch {
                        for px in 0..patch {
                            let v = tile[((by * patch + py) * w + bx * patch + px) * 3 + c] as f32;
                            data.push((v - scale[c].0) / scale[c].1);
                        }
                    }
                }
            }
        }
    }
    Patches { grid: [tiles.len(), gh, gw], data, layout: [0, 0] }
}

pub fn preprocess_tiles(cfg: &TileConfig, patch: usize, bytes: &[u8], pan: Option<PanAndScan>) -> Result<Patches> {
    let format = image::guess_format(bytes).map_err(|_| anyhow::anyhow!("the bytes are not a PNG, JPEG, GIF, BMP, TIFF or WebP image"))?;
    let img = image::load_from_memory_with_format(bytes, format).context("decode image")?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let whole = Frame { height: h, width: w, rgb: img.into_raw() };
    let mut sources = vec![whole.clone()];
    if let Some(p) = pan {
        sources.extend(pan_and_scan_boxes(h, w, p).into_iter().map(|b| crop_rgb(&whole, b)));
    }
    let (th, tw) = (cfg.size.height, cfg.size.width);
    let tiles: Vec<Vec<u8>> = sources
        .iter()
        .map(|f| if (f.height, f.width) == (th, tw) { f.rgb.clone() } else { resize_bilinear(&f.rgb, f.height, f.width, th, tw) })
        .collect();
    Ok(tile_patches(cfg, patch, &tiles))
}

pub fn canvas_resolutions(max_chunks: usize, tile: usize) -> Vec<(usize, usize)> {
    let mut groups: Vec<(u64, Vec<(usize, usize)>)> = Vec::new();
    for chunk in (1..=max_chunks).rev() {
        let mut factors: Vec<usize> = (1..=chunk).filter(|f| chunk % f == 0).collect();
        factors.sort();
        for f in factors {
            let ratio = (f as f64 / (chunk / f) as f64).to_bits();
            let entry = (f * tile, (chunk / f) * tile);
            match groups.iter_mut().find(|g| g.0 == ratio) {
                Some(g) => g.1.push(entry),
                None => groups.push((ratio, vec![entry])),
            }
        }
    }
    groups.into_iter().flat_map(|g| g.1).collect()
}

pub fn best_canvas(height: usize, width: usize, options: &[(usize, usize)], max_canvas: bool) -> (usize, usize) {
    let scale = |&(th, tw): &(usize, usize)| {
        let (sh, sw) = (th as f32 / height as f32, tw as f32 / width as f32);
        if sh > sw { sw } else { sh }
    };
    let scales: Vec<f32> = options.iter().map(scale).collect();
    let up = scales.iter().copied().filter(|s| *s >= 1.0);
    let chosen = if scales.iter().any(|s| *s >= 1.0) {
        if max_canvas { up.fold(f32::MIN, f32::max) } else { up.fold(f32::MAX, f32::min) }
    } else {
        scales.iter().copied().filter(|s| *s < 1.0).fold(f32::MIN, f32::max)
    };
    options.iter().zip(&scales).filter(|(_, s)| **s == chosen).map(|(o, _)| *o).min_by_key(|(h, w)| h * w).expect("a canvas matches")
}

fn fit_without_distortion(height: usize, width: usize, target: (usize, usize)) -> (usize, usize) {
    let (sw, sh) = (target.1 as f64 / width as f64, target.0 as f64 / height as f64);
    if sw < sh {
        (((height as f64 * sw).floor() as usize).min(target.0).max(1), target.1.max(1))
    } else {
        (target.0.max(1), ((width as f64 * sh).floor() as usize).min(target.1).max(1))
    }
}

pub fn preprocess_canvas(cfg: &TileConfig, patch: usize, bytes: &[u8]) -> Result<Patches> {
    let format = image::guess_format(bytes).map_err(|_| anyhow::anyhow!("the bytes are not a PNG, JPEG, GIF, BMP, TIFF or WebP image"))?;
    let img = image::load_from_memory_with_format(bytes, format).context("decode image")?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let tile = cfg.size.height;
    ensure!(tile == cfg.size.width, "the tile size must be square");
    let options = canvas_resolutions(cfg.max_patches.unwrap_or(16), tile);
    let canvas = best_canvas(h, w, &options, cfg.resize_to_max_canvas);
    let limit = if cfg.resize_to_max_canvas { canvas } else { (h.max(tile).min(canvas.0), w.max(tile).min(canvas.1)) };
    let (nh, nw) = fit_without_distortion(h, w, limit);
    let raw = img.into_raw();
    let resized = if (nh, nw) == (h, w) { raw.clone() } else { resize_bilinear(&raw, h, w, nh, nw) };
    let mut padded = vec![0u8; canvas.0 * canvas.1 * 3];
    for y in 0..nh {
        padded[y * canvas.1 * 3..y * canvas.1 * 3 + nw * 3].copy_from_slice(&resized[y * nw * 3..(y + 1) * nw * 3]);
    }
    let (rh, rw) = (canvas.0 / tile, canvas.1 / tile);
    let mut tiles = Vec::new();
    for ty in 0..rh {
        for tx in 0..rw {
            let mut t = Vec::with_capacity(tile * tile * 3);
            for y in 0..tile {
                let at = ((ty * tile + y) * canvas.1 + tx * tile) * 3;
                t.extend_from_slice(&padded[at..at + tile * 3]);
            }
            tiles.push(t);
        }
    }
    if rh * rw > 1 {
        tiles.push(if (h, w) == (tile, tile) { raw } else { resize_bilinear(&raw, h, w, tile, tile) });
    }
    let mut out = tile_patches(cfg, patch, &tiles);
    out.layout = [rh, rw];
    Ok(out)
}

pub enum Media<'a> {
    Image(&'a Patches),
    Video(&'a VideoPatches),
}

fn tile_sequence(tokens: &MediaTokens) -> String {
    format!("{}{}{}{}{}", tokens.wrap, tokens.start, tokens.image.repeat(tokens.tile_tokens), tokens.end, tokens.wrap)
}

pub fn expand_tiles(prompt: &str, tokens: &MediaTokens, media: &[Media]) -> Result<String> {
    let slot = tokens.placeholder.as_deref().context("the model has no image placeholder")?;
    let sequence = tile_sequence(tokens);
    let mut out = String::with_capacity(prompt.len());
    let mut rest = prompt;
    for item in media {
        let Media::Image(p) = item else { bail!("the model takes images only") };
        let crops = p.grid[0] - 1;
        let expansion = if crops == 0 {
            sequence.clone()
        } else {
            format!("Here is the original image {sequence} and here are some crops to help you see better {}", vec![sequence.as_str(); crops].join(" "))
        };
        let at = rest.find(slot).context("prompt has fewer media placeholders than media items")?;
        out.push_str(&rest[..at]);
        out.push_str(&expansion);
        rest = &rest[at + slot.len()..];
    }
    out.push_str(rest);
    Ok(out)
}

fn grid_sequence(tokens: &MediaTokens, grid: &GridTokens, layout: [usize; 2]) -> String {
    let patches = tokens.image.repeat(tokens.tile_tokens);
    let mut s = tokens.start.clone();
    if layout[0] * layout[1] > 1 {
        for _ in 0..layout[0] {
            for x in 0..layout[1] {
                s.push_str(&patches);
                if x + 1 < layout[1] {
                    s.push_str(&grid.tile_x);
                }
            }
            s.push_str(&grid.tile_y);
        }
    }
    s.push_str(&grid.global);
    s.push_str(&patches);
    s.push_str(&tokens.end);
    s
}

pub fn expand_grid(prompt: &str, tokens: &MediaTokens, media: &[Media]) -> Result<String> {
    let slot = tokens.placeholder.as_deref().context("the model has no image placeholder")?;
    let grid = tokens.grid.as_ref().context("the model has no tile grid tokens")?;
    let mut out = String::with_capacity(prompt.len());
    let mut rest = prompt;
    for item in media {
        let Media::Image(p) = item else { bail!("the model takes images only") };
        let at = rest.find(slot).context("prompt has fewer media placeholders than media items")?;
        out.push_str(&rest[..at]);
        out.push_str(&grid_sequence(tokens, grid, p.layout));
        rest = &rest[at + slot.len()..];
    }
    out.push_str(rest);
    Ok(out)
}

pub fn expand_placeholders(prompt: &str, tokens: &MediaTokens, merge_size: usize, media: &[Media]) -> Result<String> {
    if tokens.grid.is_some() {
        return expand_grid(prompt, tokens, media);
    }
    if tokens.placeholder.is_some() {
        return expand_tiles(prompt, tokens, media);
    }
    let image_slot = format!("{}{}{}", tokens.start, tokens.image, tokens.end);
    let video_slot = format!("{}{}{}", tokens.start, tokens.video, tokens.end);
    let mut out = String::with_capacity(prompt.len());
    let mut rest = prompt;
    for item in media {
        let (slot, expansion) = match item {
            Media::Image(p) => (&image_slot, format!("{}{}{}", tokens.start, tokens.image.repeat(p.tokens(merge_size)), tokens.end)),
            Media::Video(v) => {
                let [t, h, w] = v.patches.grid;
                let per_frame = h * w / (merge_size * merge_size);
                let mut s = String::new();
                for ts in v.timestamps.iter().take(t) {
                    s.push_str(&format!("<{ts:.1} seconds>{}{}{}", tokens.start, tokens.video.repeat(per_frame), tokens.end));
                }
                (&video_slot, format!("{}{s}{}", tokens.start, tokens.end))
            }
        };
        let at = rest.find(slot.as_str()).context("prompt has fewer media placeholders than media items")?;
        out.push_str(&rest[..at]);
        out.push_str(&expansion);
        rest = &rest[at + slot.len()..];
    }
    out.push_str(rest);
    Ok(out)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tower {
    #[default]
    Merged,
    Pooled,
    Tiled,
}

fn one() -> usize {
    1
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionConfig {
    #[serde(alias = "num_hidden_layers")]
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(alias = "num_attention_heads")]
    pub num_heads: usize,
    pub patch_size: usize,
    #[serde(default = "one")]
    pub spatial_merge_size: usize,
    #[serde(default = "one")]
    pub temporal_patch_size: usize,
    #[serde(default)]
    pub out_hidden_size: usize,
    #[serde(default)]
    pub num_position_embeddings: usize,
    #[serde(default, alias = "num_channels", skip_serializing_if = "Option::is_none")]
    pub in_channels: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_act: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deepstack_visual_indexes: Vec<usize>,
    #[serde(default, skip_serializing_if = "is_default_tower")]
    pub tower: Tower,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mm_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "norm_eps")]
    pub layer_norm_eps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_theta: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projector_input_dim: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projector_output_dim: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pixel_shuffle_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub bidirectional: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placeholder: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub wrap: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grid_tokens: Vec<String>,
}

fn is_default_tower(t: &Tower) -> bool {
    *t == Tower::Merged
}

impl VisionConfig {
    pub fn pooled(&self) -> bool {
        self.tower == Tower::Pooled
    }

    pub fn tiled(&self) -> bool {
        self.tower == Tower::Tiled
    }

    pub fn tile_grid(&self) -> bool {
        self.tower != Tower::Merged
    }

    pub fn eps(&self) -> f32 {
        self.layer_norm_eps.unwrap_or(1e-6) as f32
    }

    pub fn side(&self) -> usize {
        match self.image_size {
            Some(size) if self.tower != Tower::Merged => size / self.patch_size,
            _ => (self.num_position_embeddings as f64).sqrt() as usize,
        }
    }

    pub fn tokens_per_tile(&self) -> usize {
        self.mm_tokens.unwrap_or(0)
    }

    pub fn pool_kernel(&self) -> usize {
        let per_side = (self.tokens_per_tile() as f64).sqrt() as usize;
        self.side().checked_div(per_side).unwrap_or(1)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionTokens {
    pub image: u32,
    pub video: u32,
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vision {
    pub config: VisionConfig,
    pub tokens: VisionTokens,
}

impl Vision {
    pub fn from_hf_config(json: &str) -> Result<Option<Self>> {
        let family = crate::family::active();
        Self::from_hf_with(json, &family)
    }

    pub fn from_hf_with(json: &str, registry: &crate::family::Registry) -> Result<Option<Self>> {
        let top: serde_json::Value = serde_json::from_str(json).context("config.json")?;
        let Some(vc) = top.get("vision_config").filter(|v| v.is_object()) else { return Ok(None) };
        if top.get("language_model_only").and_then(|v| v.as_bool()) == Some(true) {
            return Ok(None);
        }
        let family = top.get("model_type").and_then(|t| t.as_str()).and_then(|t| registry.for_model_type(t));
        let spec = family.map(|f| f.vision.clone()).unwrap_or_default();
        if spec.ignore {
            return Ok(None);
        }
        let mut config: VisionConfig = serde_json::from_value(vc.clone()).context("vision_config")?;
        if spec.tiled {
            let text = top.get("text_config").unwrap_or(&top);
            config.tower = Tower::Tiled;
            config.placeholder = spec.placeholder.clone();
            if let Some(g) = &spec.grid_tokens {
                config.grid_tokens = vec![g.tile_x.clone(), g.tile_y.clone(), g.global.clone()];
            }
            config.out_hidden_size = text.get("hidden_size").and_then(|v| v.as_u64()).context("text_config has no hidden_size")? as usize;
            if config.rope_theta.is_none() {
                config.rope_theta = vc.get("rope_parameters").and_then(|r| r.get("rope_theta")).and_then(|v| v.as_f64());
            }
            let image = config.image_size.context("vision_config has no image_size")?;
            let side = image / config.patch_size;
            ensure!(config.pixel_shuffle_ratio == Some(0.5), "only a pixel shuffle ratio of 0.5 is supported");
            ensure!(side.is_multiple_of(2), "the patch grid of {side} does not shuffle by 2");
            config.num_position_embeddings = side * side + 1;
            config.mm_tokens = Some(side * side / 4);
            config.hidden_act.get_or_insert_with(|| "gelu".into());
            ensure!(config.projector_input_dim.is_some() && config.projector_output_dim.is_some(), "vision_config has no projector dims");
        }
        if spec.pooled {
            let text = top.get("text_config").unwrap_or(&top);
            config.tower = Tower::Pooled;
            config.bidirectional = spec.bidirectional;
            config.placeholder = spec.placeholder.clone();
            config.wrap = spec.wrap.clone();
            config.out_hidden_size = text.get("hidden_size").and_then(|v| v.as_u64()).context("text_config has no hidden_size")? as usize;
            config.mm_tokens = top.get(spec.soft_tokens_key.as_str()).and_then(|v| v.as_u64()).map(|v| v as usize);
            let image = config.image_size.context("vision_config has no image_size")?;
            config.num_position_embeddings = (image / config.patch_size).pow(2);
            ensure!(config.mm_tokens.is_some_and(|n| n > 0), "config.json has no {}", spec.soft_tokens_key);
        }
        let id = |k: &str| top.get(k).and_then(|v| v.as_u64()).map(|v| v as u32).with_context(|| format!("config.json has no {k}"));
        let keys = &spec.token_keys;
        let tokens = VisionTokens {
            image: id(&keys.image)?,
            video: if keys.video.is_empty() { u32::MAX } else { id(&keys.video)? },
            start: id(&keys.start)?,
            end: id(&keys.end)?,
        };
        Ok(Some(Vision { config, tokens }))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaKind {
    Image,
    Video,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaItem {
    pub kind: MediaKind,
    pub bytes: Arc<Vec<u8>>,
    pub hash: [u8; 32],
}

impl MediaItem {
    pub fn new(kind: MediaKind, bytes: Vec<u8>) -> Self {
        let mut h = Sha256::new();
        h.update([kind as u8]);
        h.update(&bytes);
        Self { kind, hash: h.finalize().into(), bytes: Arc::new(bytes) }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Processed {
    Image(Patches),
    Video(VideoPatches),
}

impl Processed {
    pub fn patches(&self) -> &Patches {
        match self {
            Processed::Image(p) => p,
            Processed::Video(v) => &v.patches,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MediaAssets {
    pub image: Preprocessor,
    pub video: Preprocessor,
    pub tokens: MediaTokens,
    pub ids: VisionTokens,
    pub tiles: Option<TileAssets>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TileAssets {
    pub config: TileConfig,
    pub patch: usize,
    pub pan_and_scan: Option<PanAndScan>,
    pub canvas: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MediaBudget {
    pub max_pixels: Option<usize>,
    pub max_video_secs: Option<f64>,
    pub max_video_frames: Option<usize>,
    pub max_patches: Option<usize>,
}

fn check_patches(p: &Patches, budget: &MediaBudget) -> Result<()> {
    let patches: usize = p.grid.iter().product();
    ensure!(patches > 0, "media produced an empty patch grid");
    if let Some(limit) = budget.max_patches {
        ensure!(patches <= limit, "media needs {patches} vision patches (grid {}x{}x{}), the limit is {limit}: send a smaller image or lower media.max_pixels", p.grid[0], p.grid[1], p.grid[2]);
    }
    Ok(())
}

impl MediaAssets {
    pub fn merge(&self) -> usize {
        self.image.merge_size
    }

    pub fn process(&self, items: &[MediaItem], budget: MediaBudget) -> Result<Vec<Processed>> {
        if let Some(t) = &self.tiles {
            return items
                .iter()
                .map(|item| match item.kind {
                    MediaKind::Image => {
                        let p = if t.canvas { preprocess_canvas(&t.config, t.patch, &item.bytes)? } else { preprocess_tiles(&t.config, t.patch, &item.bytes, t.pan_and_scan)? };
                        check_patches(&p, &budget)?;
                        Ok(Processed::Image(p))
                    }
                    MediaKind::Video => bail!("the model takes images only"),
                })
                .collect();
        }
        items
            .iter()
            .map(|item| match item.kind {
                MediaKind::Image => {
                    let mut cfg = self.image.clone();
                    let cap = budget.max_patches.map(|n| n.saturating_mul(cfg.patch_size * cfg.patch_size));
                    if let Some(px) = budget.max_pixels.into_iter().chain(cap).min() {
                        cfg.size.longest_edge = cfg.size.longest_edge.min(px);
                        cfg.size.shortest_edge = cfg.size.shortest_edge.min(cfg.size.longest_edge);
                    }
                    let p = preprocess_image(&cfg, &item.bytes)?;
                    check_patches(&p, &budget)?;
                    Ok(Processed::Image(p))
                }
                MediaKind::Video => {
                    let mut cfg = self.video.clone();
                    let cap = budget.max_patches.map(|n| n.saturating_mul(cfg.patch_size * cfg.patch_size * cfg.temporal_patch_size));
                    if let Some(px) = budget.max_pixels.into_iter().chain(cap).min() {
                        cfg.size.longest_edge = cfg.size.longest_edge.min(px);
                        cfg.size.shortest_edge = cfg.size.shortest_edge.min(cfg.size.longest_edge);
                    }
                    let v = preprocess_video_limited(&cfg, &item.bytes, budget)?;
                    check_patches(&v.patches, &budget)?;
                    Ok(Processed::Video(v))
                }
            })
            .collect()
    }

    pub fn expand(&self, prompt: &str, processed: &[Processed]) -> Result<String> {
        let media: Vec<Media> = processed
            .iter()
            .map(|p| match p {
                Processed::Image(i) => Media::Image(i),
                Processed::Video(v) => Media::Video(v),
            })
            .collect();
        expand_placeholders(prompt, &self.tokens, self.merge(), &media)
    }
}

pub fn preprocess_video_limited(cfg: &Preprocessor, bytes: &[u8], budget: MediaBudget) -> Result<VideoPatches> {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let path = std::env::temp_dir().join(format!("specrt-video-{}-{nanos}.bin", std::process::id()));
    std::fs::write(&path, bytes)?;
    let out = (|| {
        let info = probe_video(&path)?;
        if let Some(limit) = budget.max_video_secs {
            let secs = info.frames as f64 / info.fps;
            ensure!(secs <= limit, "video is {secs:.1} s long, the limit is {limit} s");
        }
        if let Some(limit) = budget.max_video_frames {
            let frames = sample_indices(cfg, info.frames, info.fps).len();
            ensure!(frames <= limit, "video needs {frames} sampled frames, the limit is {limit}");
        }
        preprocess_video_file(cfg, &path)
    })();
    let _ = std::fs::remove_file(&path);
    out
}

#[derive(Clone, Debug, PartialEq)]
pub struct MediaPlan {
    pub positions: Vec<[i32; 4]>,
    pub delta: i32,
    pub tags: Vec<u32>,
    pub embeds: Arc<Vec<f32>>,
    pub deepstack: Vec<Arc<Vec<f32>>>,
    pub spans: Vec<i32>,
}

impl MediaPlan {
    pub fn extend_chunk(&self, start: usize, end: usize, limit: usize) -> usize {
        if self.spans.is_empty() || end == 0 || end > self.spans.len() {
            return end;
        }
        let last = end - 1;
        if self.spans[last] as usize <= last {
            return end;
        }
        let mut block_start = last;
        while block_start > 0 && self.spans[block_start - 1] == self.spans[last] {
            block_start -= 1;
        }
        if block_start > start {
            return block_start;
        }
        (self.spans[last] as usize + 1).min(start + limit)
    }
}

fn tag(hash: &[u8; 32], row: usize) -> u32 {
    let mut h = Sha256::new();
    h.update(hash);
    h.update((row as u64).to_le_bytes());
    let d = h.finalize();
    u32::from_le_bytes([d[0], d[1], d[2], d[3]]) | 0x8000_0000
}

pub fn plan_media(ids: &[u32], vis: &VisionTokens, merge: usize, processed: &[Processed], hashes: &[[u8; 32]]) -> Result<MediaPlan> {
    ensure!(processed.len() == hashes.len(), "one hash per media item");
    let modality = |t: u32| if t == vis.image { 1 } else if t == vis.video { 2 } else { 0 };
    let mut positions = vec![[0i32; 4]; ids.len()];
    let mut tags = vec![0u32; ids.len()];
    let mut cur = 0i32;
    let mut next_item = 0usize;
    let mut frame = 0usize;
    let mut embed_row = 0i32;
    let mut item_row = 0usize;
    let mut i = 0;
    while i < ids.len() {
        let m = modality(ids[i]);
        let mut j = i;
        while j < ids.len() && modality(ids[j]) == m {
            j += 1;
        }
        if m == 0 {
            for (k, p) in positions[i..j].iter_mut().enumerate() {
                let v = cur + k as i32;
                *p = [v, v, v, -1];
            }
            cur += (j - i) as i32;
        } else {
            let item = processed.get(next_item).context("more media placeholders than media items")?;
            let [t, h, w] = item.patches().grid;
            let (lh, lw) = (h / merge, w / merge);
            match (m, item) {
                (1, Processed::Image(_)) | (2, Processed::Video(_)) => {}
                _ => bail!("media placeholder kind does not match media item {next_item}"),
            }
            ensure!(j - i == lh * lw, "media run of {} tokens, the grid needs {}", j - i, lh * lw);
            for (k, p) in positions[i..j].iter_mut().enumerate() {
                let (hi, wi) = ((k / lw) as i32, (k % lw) as i32);
                *p = [cur, cur + hi, cur + wi, embed_row];
                tags[i + k] = tag(&hashes[next_item], item_row);
                embed_row += 1;
                item_row += 1;
            }
            cur += lh.max(lw) as i32;
            frame += 1;
            if m == 1 || frame == t {
                next_item += 1;
                frame = 0;
                item_row = 0;
            }
        }
        i = j;
    }
    ensure!(next_item == processed.len(), "fewer media placeholders than media items");
    Ok(MediaPlan { positions, delta: cur - ids.len() as i32, tags, embeds: Arc::new(Vec::new()), deepstack: Vec::new(), spans: Vec::new() })
}

pub fn plan_tiles(ids: &[u32], vis: &VisionTokens, tile_tokens: usize, processed: &[Processed], hashes: &[[u8; 32]]) -> Result<MediaPlan> {
    ensure!(processed.len() == hashes.len(), "one hash per media item");
    let mut positions: Vec<[i32; 4]> = (0..ids.len() as i32).map(|v| [v, v, v, -1]).collect();
    let mut tags = vec![0u32; ids.len()];
    let mut spans: Vec<i32> = (0..ids.len() as i32).collect();
    let mut tiles_left = 0usize;
    let mut next_item = 0usize;
    let mut item_row = 0usize;
    let mut embed_row = 0i32;
    let mut i = 0;
    while i < ids.len() {
        if ids[i] != vis.image {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < ids.len() && ids[j] == vis.image {
            j += 1;
        }
        ensure!(j - i == tile_tokens, "an image run of {} tokens, a tile has {tile_tokens}", j - i);
        if tiles_left == 0 {
            let item = processed.get(next_item).context("more image placeholders than images")?;
            tiles_left = item.patches().grid[0];
            item_row = 0;
        }
        for k in i..j {
            positions[k][3] = embed_row;
            tags[k] = tag(&hashes[next_item], item_row);
            spans[k] = (j - 1) as i32;
            embed_row += 1;
            item_row += 1;
        }
        tiles_left -= 1;
        if tiles_left == 0 {
            next_item += 1;
        }
        i = j;
    }
    ensure!(tiles_left == 0 && next_item == processed.len(), "fewer image placeholders than images");
    Ok(MediaPlan { positions, delta: 0, tags, embeds: Arc::new(Vec::new()), deepstack: Vec::new(), spans })
}
