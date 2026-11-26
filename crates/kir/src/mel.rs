use crate::conv::ConvShape;
use crate::expr::Expr;
use crate::fnv;
use std::f64::consts::PI;

pub const THREADS: u32 = 256;
pub const KERNELS: &[&str] = &["mel_frames", "mel_power", "mel_log", "mel_max", "mel_log_max", "mel_scale", "mel_act"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    Hann { periodic: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MelScale {
    Slaney,
    Htk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    None,
    Slaney,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pad {
    Reflect,
    Zero,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogMel {
    pub samples: usize,
    pub n_fft: usize,
    pub hop: usize,
    pub frames: usize,
    pub bins: usize,
    pub rate: f64,
    pub fmin: f64,
    pub fmax: f64,
    pub window: Window,
    pub scale: MelScale,
    pub norm: Norm,
    pub pad: Pad,
    pub log: Expr,
    pub floor: Expr,
    pub out: Expr,
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f64;
    (0..n).map(|i| if i == n - 1 { b } else { a + i as f64 * step }).collect()
}

fn names_in(e: &Expr) -> Vec<String> {
    let mut out = Vec::new();
    e.names(&mut out);
    out
}

impl LogMel {
    pub fn freqs(&self) -> usize {
        self.n_fft / 2 + 1
    }

    pub fn validate(&self) -> Result<(), String> {
        let bad = |s: String| Err(format!("log_mel: {s}"));
        if self.n_fft < 2 || self.hop == 0 || self.bins == 0 || self.samples == 0 || self.frames == 0 {
            return bad(format!("needs n_fft, hop, bins, samples and frames, got {} {} {} {} {}", self.n_fft, self.hop, self.bins, self.samples, self.frames));
        }
        if self.pad == Pad::Reflect && self.samples <= self.n_fft / 2 {
            return bad(format!("reflect padding by {} needs more than that many samples, the window has {}", self.n_fft / 2, self.samples));
        }
        if self.frames > self.samples / self.hop + 1 {
            return bad(format!("{} frames of hop {} over {} samples", self.frames, self.hop, self.samples));
        }
        if !(0.0 <= self.fmin && self.fmin < self.fmax && self.fmax <= self.rate / 2.0) {
            return bad(format!("mel range {} to {} Hz at {} Hz", self.fmin, self.fmax, self.rate));
        }
        for (what, e, allowed) in [("log", &self.log, &["m"][..]), ("floor", &self.floor, &["v", "top"][..]), ("out", &self.out, &["v"][..])] {
            if let Some(n) = names_in(e).into_iter().find(|n| !allowed.contains(&n.as_str())) {
                return bad(format!("the {what} expression {e} names {n}, it may name {allowed:?}"));
            }
        }
        Ok(())
    }

    pub fn window_values(&self) -> Vec<f64> {
        let n = self.n_fft;
        match self.window {
            Window::Hann { periodic } => {
                let d = if periodic { n } else { n - 1 } as f64;
                (0..n).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / d).cos()).collect()
            }
        }
    }

    pub fn dft_table(&self) -> Vec<f32> {
        let (n, f) = (self.n_fft, self.freqs());
        let w = self.window_values();
        let mut out = vec![0f32; 2 * f * n];
        for k in 0..f {
            for (t, wt) in w.iter().enumerate() {
                let a = 2.0 * PI * ((k * t) % n) as f64 / n as f64;
                out[(2 * k) * n + t] = (wt * a.cos()) as f32;
                out[(2 * k + 1) * n + t] = (wt * a.sin()) as f32;
            }
        }
        out
    }

    fn hz_to_mel(&self, f: f64) -> f64 {
        match self.scale {
            MelScale::Htk => 2595.0 * (1.0 + f / 700.0).log10(),
            MelScale::Slaney if f >= 1000.0 => 15.0 + (f / 1000.0).ln() * 27.0 / 6.4f64.ln(),
            MelScale::Slaney => 3.0 * f / 200.0,
        }
    }

    fn mel_to_hz(&self, m: f64) -> f64 {
        match self.scale {
            MelScale::Htk => 700.0 * (10f64.powf(m / 2595.0) - 1.0),
            MelScale::Slaney if m >= 15.0 => 1000.0 * (6.4f64.ln() / 27.0 * (m - 15.0)).exp(),
            MelScale::Slaney => 200.0 * m / 3.0,
        }
    }

    pub fn mel_bank(&self) -> Vec<f32> {
        let (f, b) = (self.freqs(), self.bins);
        let mels = linspace(self.hz_to_mel(self.fmin), self.hz_to_mel(self.fmax), b + 2);
        let edges: Vec<f64> = mels.iter().map(|m| self.mel_to_hz(*m)).collect();
        let fft = linspace(0.0, (self.rate / 2.0).floor(), f);
        let mut out = vec![0f32; b * f];
        for m in 0..b {
            let enorm = match self.norm {
                Norm::Slaney => 2.0 / (edges[m + 2] - edges[m]),
                Norm::None => 1.0,
            };
            for (k, hz) in fft.iter().enumerate() {
                let down = -(edges[m] - hz) / (edges[m + 1] - edges[m]);
                let up = (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]);
                out[m * f + k] = (0f64.max(down.min(up)) * enorm) as f32;
            }
        }
        out
    }

    pub fn dft_shape(&self) -> ConvShape {
        ConvShape::line(self.n_fft, 2 * self.freqs(), 1, 1, 1)
    }

    pub fn bank_shape(&self) -> ConvShape {
        ConvShape::line(self.freqs(), self.bins, 1, 1, 1)
    }

    pub fn source(&self, i: isize) -> Option<usize> {
        let n = self.samples as isize;
        let j = match self.pad {
            Pad::Reflect if i < 0 => -i,
            Pad::Reflect if i >= n => 2 * (n - 1) - i,
            Pad::Zero if i < 0 || i >= n => return None,
            _ => i,
        };
        Some(j as usize)
    }

    pub fn describe(&self) -> String {
        format!(
            "log_mel samples={} n_fft={} hop={} frames={} bins={} rate={} fmin={} fmax={} window={:?} scale={:?} norm={:?} pad={:?} log={} floor={} out={}",
            self.samples, self.n_fft, self.hop, self.frames, self.bins, self.rate, self.fmin, self.fmax, self.window, self.scale, self.norm, self.pad, self.log, self.floor, self.out
        )
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }
}
