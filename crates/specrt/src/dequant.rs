use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Elem {
    E2m1,
    E4m3,
    Int8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScaleKind {
    E8m0,
    E4m3,
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockFormat {
    pub elem: Elem,
    pub scale: ScaleKind,
    pub group: usize,
}

pub const DEVICE_TILE: usize = 128;

pub const MXFP4: BlockFormat = BlockFormat { elem: Elem::E2m1, scale: ScaleKind::E8m0, group: 32 };

impl Elem {
    pub fn bits(self) -> usize {
        match self {
            Elem::E2m1 => 4,
            Elem::E4m3 | Elem::Int8 => 8,
        }
    }

    fn value(self, code: u8) -> f32 {
        match self {
            Elem::E2m1 => crate::nvfp4::e2m1_to_f32(code),
            Elem::E4m3 => crate::weights::e4m3_to_f32(code),
            Elem::Int8 => f32::from(code as i8),
        }
    }
}

impl ScaleKind {
    fn value(self, code: u8) -> f32 {
        match self {
            ScaleKind::E8m0 => f32::from_bits(u32::from(code) << 23),
            ScaleKind::E4m3 => crate::weights::e4m3_to_f32(code),
            ScaleKind::F32 => f32::from(code),
        }
    }
}

impl BlockFormat {
    pub fn check(&self) -> Result<()> {
        ensure!(self.group > 0 && (self.group * self.elem.bits()).is_multiple_of(8), "block group {} does not fill whole bytes", self.group);
        if self.scale == ScaleKind::F32 {
            bail!("a one byte scale cannot be f32");
        }
        Ok(())
    }

    pub fn device_packed(&self) -> bool {
        self.elem == Elem::E2m1 && self.scale == ScaleKind::E8m0 && self.group.is_multiple_of(32)
    }

    pub fn native_mma(&self) -> bool {
        self.device_packed() && self.group == 32
    }

    pub fn padded_cols(&self, cols: usize) -> usize {
        cols.next_multiple_of(DEVICE_TILE)
    }

    pub fn pad_rows(&self, data: &[u8], row_bytes: usize, padded_row_bytes: usize) -> Vec<u8> {
        if row_bytes == padded_row_bytes {
            return data.to_vec();
        }
        let mut out = vec![0u8; data.len() / row_bytes * padded_row_bytes];
        for (src, dst) in data.chunks(row_bytes).zip(out.chunks_mut(padded_row_bytes)) {
            dst[..row_bytes].copy_from_slice(src);
        }
        out
    }

    pub fn packed_len(&self, values: usize) -> usize {
        values * self.elem.bits() / 8
    }

    pub fn decode(&self, packed: &[u8], scales: &[u8], out: &mut Vec<f32>) -> Result<()> {
        self.check()?;
        let per_group = self.group * self.elem.bits() / 8;
        ensure!(packed.len() == scales.len() * per_group, "{} packed bytes do not match {} scales of {} values", packed.len(), scales.len(), self.group);
        for (block, &s) in packed.chunks(per_group).zip(scales) {
            let scale = self.scale.value(s);
            match self.elem.bits() {
                4 => block.iter().for_each(|&b| {
                    out.push(self.elem.value(b & 0xf) * scale);
                    out.push(self.elem.value(b >> 4) * scale);
                }),
                _ => block.iter().for_each(|&b| out.push(self.elem.value(b) * scale)),
            }
        }
        Ok(())
    }

    pub fn decode_bf16(&self, packed: &[u8], scales: &[u8]) -> Result<Vec<u16>> {
        let mut f = Vec::with_capacity(packed.len() * 8 / self.elem.bits());
        self.decode(packed, scales, &mut f)?;
        Ok(f.into_iter().map(crate::weights::f32_to_bf16).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mxfp4_decodes_low_nibble_first_with_a_power_of_two_scale() {
        let packed: Vec<u8> = (0..16).map(|i| (i * 2 + 1) << 4 | (i * 2) & 0xf).collect();
        let mut out = Vec::new();
        MXFP4.decode(&packed, &[127], &mut out).unwrap();
        let want: Vec<f32> = (0..32u8).map(|n| crate::nvfp4::e2m1_to_f32(n & 0xf)).collect();
        assert_eq!(out, want);
        let mut scaled = Vec::new();
        MXFP4.decode(&packed, &[129], &mut scaled).unwrap();
        assert!(scaled.iter().zip(&want).all(|(a, b)| *a == b * 4.0));
        let mut small = Vec::new();
        MXFP4.decode(&packed, &[126], &mut small).unwrap();
        assert!(small.iter().zip(&want).all(|(a, b)| *a == b * 0.5));
    }

    #[test]
    fn a_format_is_data_so_nvfp4_blocks_decode_through_the_same_path() {
        let nv = BlockFormat { elem: Elem::E2m1, scale: ScaleKind::E4m3, group: 16 };
        let mut out = Vec::new();
        nv.decode(&[0x21; 8], &[crate::weights::f32_to_e4m3(2.0)], &mut out).unwrap();
        assert_eq!(&out[..4], &[1.0, 2.0, 1.0, 2.0]);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn a_length_mismatch_is_an_error() {
        assert!(MXFP4.decode(&[0; 15], &[127], &mut Vec::new()).is_err());
        assert!(BlockFormat { elem: Elem::E2m1, scale: ScaleKind::E8m0, group: 3 }.check().is_err());
    }

    #[test]
    fn rows_pad_to_the_device_tile_with_zeros() {
        assert_eq!(MXFP4.padded_cols(2880), 2944);
        assert_eq!(MXFP4.padded_cols(256), 256);
        assert_eq!(MXFP4.pad_rows(&[1, 2, 3, 4], 2, 4), vec![1, 2, 0, 0, 3, 4, 0, 0]);
        assert_eq!(MXFP4.pad_rows(&[1, 2, 3, 4], 2, 2), vec![1, 2, 3, 4]);
    }

    #[test]
    fn the_format_reads_from_toml() {
        let f: BlockFormat = toml::from_str("elem = \"e2m1\"\nscale = \"e8m0\"\ngroup = 32").unwrap();
        assert_eq!(f, MXFP4);
    }
}
