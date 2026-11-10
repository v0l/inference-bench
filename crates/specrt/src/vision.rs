use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
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

fn coefficients(input: usize, output: usize) -> Coefficients {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
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
            let v = cubic((x as f64 + xmin as f64 - center + 0.5) / filter_scale);
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
    let horizontal = if new_w == w {
        src.to_vec()
    } else {
        let c = coefficients(w, new_w);
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
    let c = coefficients(h, new_h);
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
    Patches { grid: [gt, gh, gw], data }
}

pub fn preprocess_image(cfg: &Preprocessor, bytes: &[u8]) -> Result<Patches> {
    let img = image::load_from_memory(bytes).context("decode image")?.to_rgb8();
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
    while idx.len() % temporal != 0 {
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
}

impl Default for MediaTokens {
    fn default() -> Self {
        MediaTokens { image: "<|image_pad|>".into(), video: "<|video_pad|>".into(), start: "<|vision_start|>".into(), end: "<|vision_end|>".into() }
    }
}

pub enum Media<'a> {
    Image(&'a Patches),
    Video(&'a VideoPatches),
}

pub fn expand_placeholders(prompt: &str, tokens: &MediaTokens, merge_size: usize, media: &[Media]) -> Result<String> {
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
                (&video_slot, s)
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
