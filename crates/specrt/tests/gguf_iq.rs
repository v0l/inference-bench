// Bit-exact check of the CPU i-quant dequantisers against ggml's own
// dequantize_row_iq* (llama.cpp ggml-quants.c). Fixtures come from
// scripts/gen_iq_reference.c: 64 random blocks per type, each followed by the
// 256 f32 values libggml produced for it.
use specrt::gguf_iq::*;
use std::path::Path;

fn check(name: &str, block_bytes: usize, f: fn(&[u8], &mut [f32])) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/iq").join(format!("{name}.bin"));
    let data = std::fs::read(path).unwrap();
    let record = block_bytes + 256 * 4;
    assert_eq!(data.len() % record, 0);
    for (n, rec) in data.chunks_exact(record).enumerate() {
        let mut out = [0f32; 256];
        f(&rec[..block_bytes], &mut out);
        for (i, want) in rec[block_bytes..].chunks_exact(4).enumerate() {
            let want = f32::from_le_bytes(want.try_into().unwrap());
            assert_eq!(out[i].to_bits(), want.to_bits(), "{name} block {n} element {i}: {} vs {want}", out[i]);
        }
    }
}

#[test]
fn iq4_nl_matches_ggml() {
    check("iq4_nl", 144, dequantize_iq4_nl);
}

#[test]
fn iq4_xs_matches_ggml() {
    check("iq4_xs", 136, dequantize_iq4_xs);
}
