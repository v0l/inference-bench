use crate::weights::{e4m3_to_f32, f32_to_e4m3, interleave_rows};
use anyhow::{Result, ensure};
use std::borrow::Cow;

pub const GROUP: usize = 16;
pub const SCALE_HEADER: usize = 16;
pub const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

pub fn e2m1_to_f32(nibble: u8) -> f32 {
    let v = E2M1[(nibble & 7) as usize];
    if nibble & 8 != 0 { -v } else { v }
}

pub fn f32_to_e2m1(v: f32) -> u8 {
    let a = v.abs();
    let index = match a {
        _ if a <= 0.25 => 0,
        _ if a < 0.75 => 1,
        _ if a <= 1.25 => 2,
        _ if a < 1.75 => 3,
        _ if a <= 2.5 => 4,
        _ if a < 3.5 => 5,
        _ if a <= 5.0 => 6,
        _ => 7,
    };
    index | if v.is_sign_negative() && index != 0 { 8 } else { 0 }
}

pub struct Fp4<'a> {
    pub packed: Cow<'a, [u8]>,
    pub scales: Cow<'a, [u8]>,
    pub global: f32,
    pub input_global: f32,
    pub cols: usize,
}

impl<'a> Fp4<'a> {
    pub fn new(packed: Cow<'a, [u8]>, scales: Cow<'a, [u8]>, global: f32, rows: usize, cols: usize) -> Result<Self> {
        ensure!(cols.is_multiple_of(GROUP), "nvfp4 rows must be a multiple of {GROUP} wide");
        ensure!(packed.len() == rows * cols / 2, "packed weight has {} bytes, expected {}", packed.len(), rows * cols / 2);
        ensure!(scales.len() == rows * cols / GROUP, "group scales have {} bytes, expected {}", scales.len(), rows * cols / GROUP);
        ensure!(global.is_finite() && global > 0.0, "global scale {global} is not a positive number");
        Ok(Fp4 { packed, scales, global, input_global: 0.0, cols })
    }

    pub fn rows(&self) -> usize {
        self.packed.len() * 2 / self.cols
    }

    pub fn interleave(a: Fp4<'a>, b: Fp4<'a>) -> Result<Fp4<'a>> {
        ensure!(a.rows() == b.rows() && a.cols == b.cols, "gate and up differ in shape");
        ensure!(a.global == b.global, "gate and up have different global scales ({} and {})", a.global, b.global);
        ensure!(a.input_global == b.input_global, "gate and up have different input global scales");
        Ok(Fp4 {
            packed: Cow::Owned(interleave_rows(&a.packed, &b.packed, a.cols / 2)),
            scales: Cow::Owned(interleave_rows(&a.scales, &b.scales, a.cols / GROUP)),
            global: a.global,
            input_global: a.input_global,
            cols: a.cols,
        })
    }

    pub fn dequant_row(&self, row: usize) -> Vec<f32> {
        let half = self.cols / 2;
        let bytes = &self.packed[row * half..(row + 1) * half];
        let scales = &self.scales[row * self.cols / GROUP..(row + 1) * self.cols / GROUP];
        (0..self.cols)
            .map(|c| {
                let b = bytes[c / 2];
                let nibble = if c % 2 == 0 { b & 0xf } else { b >> 4 };
                e2m1_to_f32(nibble) * (e4m3_to_f32(scales[c / GROUP]) / self.global)
            })
            .collect()
    }

    pub fn device_scales(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(SCALE_HEADER + self.scales.len());
        out.extend_from_slice(&(1.0 / self.global).to_le_bytes());
        out.extend_from_slice(&self.input_global.to_le_bytes());
        out.resize(SCALE_HEADER, 0);
        out.extend_from_slice(&self.scales);
        out
    }
}

pub fn quantize(w: &[f32], cols: usize, global: f32) -> (Vec<u8>, Vec<u8>) {
    let mut packed = vec![0u8; w.len() / 2];
    let mut scales = Vec::with_capacity(w.len() / GROUP);
    for (g, group) in w.chunks_exact(GROUP).enumerate() {
        let amax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = f32_to_e4m3((amax / 6.0 * global).min(448.0));
        scales.push(scale);
        let s = e4m3_to_f32(scale) / global;
        for (i, v) in group.iter().enumerate() {
            let q = if s > 0.0 { f32_to_e2m1(v / s) } else { 0 };
            let at = g * GROUP + i;
            packed[at / 2] |= if at % 2 == 0 { q } else { q << 4 };
        }
    }
    debug_assert!(w.len().is_multiple_of(cols));
    (packed, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_round_trips_every_code_and_rounds_to_nearest_even_at_ties() {
        for n in 0..16u8 {
            let v = e2m1_to_f32(n);
            if n != 8 {
                assert_eq!(f32_to_e2m1(v), n, "{n} -> {v}");
            }
        }
        assert_eq!(e2m1_to_f32(0x7), 6.0);
        assert_eq!(e2m1_to_f32(0xf), -6.0);
        assert_eq!(e2m1_to_f32(0x9), -0.5);
        assert_eq!(f32_to_e2m1(0.25), 0);
        assert_eq!(f32_to_e2m1(0.75), 2);
        assert_eq!(f32_to_e2m1(1.25), 2);
        assert_eq!(f32_to_e2m1(1.75), 4);
        assert_eq!(f32_to_e2m1(2.5), 4);
        assert_eq!(f32_to_e2m1(3.5), 6);
        assert_eq!(f32_to_e2m1(5.0), 6);
        assert_eq!(f32_to_e2m1(100.0), 7);
        assert_eq!(f32_to_e2m1(-3.0), 0xd);
    }

    #[test]
    fn low_nibble_is_the_even_column_and_the_global_scale_divides() {
        let packed = vec![0x72u8, 0x0f, 0, 0, 0, 0, 0, 0];
        let scales = vec![0x38u8];
        let m = Fp4::new(Cow::Owned(packed), Cow::Owned(scales), 2.0, 1, 16).unwrap();
        let row = m.dequant_row(0);
        assert_eq!(&row[..4], &[0.5, 3.0, -3.0, 0.0]);
        let header = m.device_scales();
        assert_eq!(f32::from_le_bytes(header[..4].try_into().unwrap()), 0.5);
        assert_eq!(header.len(), SCALE_HEADER + 1);
    }

    #[test]
    fn quantize_then_dequant_stays_within_half_a_step_of_each_group() {
        let cols = 64;
        let w: Vec<f32> = (0..2 * cols).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013).collect();
        let (packed, scales) = quantize(&w, cols, 900.0);
        let m = Fp4::new(Cow::Owned(packed), Cow::Owned(scales), 900.0, 2, cols).unwrap();
        for r in 0..2 {
            let row = m.dequant_row(r);
            for (g, chunk) in w[r * cols..(r + 1) * cols].chunks(GROUP).enumerate() {
                let amax = chunk.iter().fold(0f32, |a, v| a.max(v.abs()));
                for (i, v) in chunk.iter().enumerate() {
                    assert!((row[g * GROUP + i] - v).abs() <= amax / 6.0 * 1.07, "{r} {g} {i}");
                }
            }
        }
    }
}
