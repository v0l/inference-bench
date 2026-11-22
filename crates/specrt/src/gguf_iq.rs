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
