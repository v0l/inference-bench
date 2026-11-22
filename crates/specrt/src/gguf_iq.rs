// CPU dequantisers for the ggml i-quants. Reference: llama.cpp ggml/src/ggml-quants.c,
// dequantize_row_iq* (MIT). Grid and codebook tables are in kir::iq_tables, generated
// from ggml-common.h by crates/kir/scripts/gen_iq_tables.py.
use kir::iq_tables::*;

fn f16(b: &[u8]) -> f32 {
    crate::weights::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
}

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

fn signed(signs: u8, j: usize) -> f32 {
    if signs & (1 << j) != 0 { -1.0 } else { 1.0 }
}

pub fn dequantize_iq2_xxs(block: &[u8], out: &mut [f32]) {
    let d = f16(block);
    for ib in 0..8 {
        let at = 2 + 8 * ib;
        let (idx, aux) = (u32_at(block, at), u32_at(block, at + 4));
        let db = d * (0.5 + (aux >> 28) as f32) * 0.25;
        for l in 0..4 {
            let grid = IQ2XXS_GRID[((idx >> (8 * l)) & 255) as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                out[32 * ib + 8 * l + j] = db * grid[j] as f32 * signed(signs, j);
            }
        }
    }
}

pub fn dequantize_iq2_xs(block: &[u8], out: &mut [f32]) {
    let d = f16(block);
    for ib in 0..8 {
        let sc = block[66 + ib];
        let db = [d * (0.5 + (sc & 0xf) as f32) * 0.25, d * (0.5 + (sc >> 4) as f32) * 0.25];
        for l in 0..4 {
            let q = u16_at(block, 2 + 2 * (4 * ib + l)) as usize;
            let grid = IQ2XS_GRID[q & 511].to_le_bytes();
            let signs = KSIGNS_IQ2XS[q >> 9];
            for j in 0..8 {
                out[32 * ib + 8 * l + j] = db[l / 2] * grid[j] as f32 * signed(signs, j);
            }
        }
    }
}

pub fn dequantize_iq2_s(block: &[u8], out: &mut [f32]) {
    let d = f16(block);
    for ib in 0..8 {
        let sc = block[74 + ib];
        let db = [d * (0.5 + (sc & 0xf) as f32) * 0.25, d * (0.5 + (sc >> 4) as f32) * 0.25];
        let qh = block[66 + ib] as usize;
        for l in 0..4 {
            let idx = block[2 + 4 * ib + l] as usize | ((qh << (8 - 2 * l)) & 0x300);
            let grid = IQ2S_GRID[idx].to_le_bytes();
            let signs = block[34 + 4 * ib + l];
            for j in 0..8 {
                out[32 * ib + 8 * l + j] = db[l / 2] * grid[j] as f32 * signed(signs, j);
            }
        }
    }
}

pub fn dequantize_iq3_xxs(block: &[u8], out: &mut [f32]) {
    let d = f16(block);
    for ib in 0..8 {
        let aux = u32_at(block, 66 + 4 * ib);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[block[2 + 8 * ib + 2 * l] as usize].to_le_bytes();
            let g2 = IQ3XXS_GRID[block[2 + 8 * ib + 2 * l + 1] as usize].to_le_bytes();
            for j in 0..4 {
                out[32 * ib + 8 * l + j] = db * g1[j] as f32 * signed(signs, j);
                out[32 * ib + 8 * l + j + 4] = db * g2[j] as f32 * signed(signs, j + 4);
            }
        }
    }
}

pub fn dequantize_iq3_s(block: &[u8], out: &mut [f32]) {
    let d = f16(block);
    for ib in 0..8 {
        let sc = block[106 + ib / 2];
        let db = d * (1 + 2 * ((sc >> (4 * (ib % 2))) & 0xf) as i32) as f32;
        let qh = block[66 + ib] as usize;
        for l in 0..4 {
            let i1 = block[2 + 8 * ib + 2 * l] as usize | ((qh << (8 - 2 * l)) & 256);
            let i2 = block[2 + 8 * ib + 2 * l + 1] as usize | ((qh << (7 - 2 * l)) & 256);
            let (g1, g2) = (IQ3S_GRID[i1].to_le_bytes(), IQ3S_GRID[i2].to_le_bytes());
            let signs = block[74 + 4 * ib + l];
            for j in 0..4 {
                out[32 * ib + 8 * l + j] = db * g1[j] as f32 * signed(signs, j);
                out[32 * ib + 8 * l + j + 4] = db * g2[j] as f32 * signed(signs, j + 4);
            }
        }
    }
}

pub fn dequantize_iq4_nl(block: &[u8], out: &mut [f32]) {
    for b in 0..8 {
        let blk = &block[18 * b..];
        let d = f16(blk);
        for j in 0..16 {
            out[32 * b + j] = d * KVALUES_IQ4NL[(blk[2 + j] & 0xf) as usize] as f32;
            out[32 * b + j + 16] = d * KVALUES_IQ4NL[(blk[2 + j] >> 4) as usize] as f32;
        }
    }
}

pub fn dequantize_iq4_xs(block: &[u8], out: &mut [f32]) {
    let d = f16(block);
    let scales_h = u16_at(block, 2);
    for ib in 0..8 {
        let ls = ((block[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) as i32 | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32;
        let qs = &block[8 + 16 * ib..];
        for j in 0..16 {
            out[32 * ib + j] = dl * KVALUES_IQ4NL[(qs[j] & 0xf) as usize] as f32;
            out[32 * ib + j + 16] = dl * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
        }
    }
}
