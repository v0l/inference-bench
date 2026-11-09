use specrt::gguf::{GgmlType, dequantize};

const GOLDEN: &[u8] = include_bytes!("fixtures/ggml-golden.bin");

fn u32_at(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize
}

#[test]
fn every_quant_type_dequantises_bit_exact_against_ggml() {
    let mut at = 0;
    let mut seen = Vec::new();
    while at < GOLDEN.len() {
        let (ty, n, bytes) = (u32_at(GOLDEN, at), u32_at(GOLDEN, at + 4), u32_at(GOLDEN, at + 8));
        at += 12;
        let kind = GgmlType::from_id(ty as u32);
        let quant = &GOLDEN[at..at + bytes];
        at += bytes;
        let want: Vec<u32> = GOLDEN[at..at + 4 * n].chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        at += 4 * n;
        let mut got = vec![0f32; n];
        dequantize(kind, quant, &mut got).unwrap();
        let mismatches: Vec<usize> = (0..n).filter(|&i| got[i].to_bits() != want[i]).collect();
        assert!(mismatches.is_empty(), "{} differs from ggml at {} of {n} values, first {:?}", kind.name(), mismatches.len(), mismatches.first());
        seen.push(kind);
    }
    assert_eq!(seen, [GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K]);
}
