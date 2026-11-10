#![cfg(feature = "wgpu")]

use specrt::backends::wgpu::{Buffer, Wgpu};
use specrt::backends::Backend;
use specrt::gguf::{DeviceRows, GgmlType};
use specrt::spec::Spec;
use specrt::wgsl;
use std::path::Path;
use std::sync::Mutex;

static GPU: Mutex<()> = Mutex::new(());

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn unit(&mut self) -> f32 {
        self.next() as f32 / (1u32 << 31) as f32 * 2.0 - 1.0
    }
}

fn spec() -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/qwen3-0.6b.toml")).unwrap()
}

fn device() -> Wgpu {
    let w = Wgpu::new(None).expect("a real GPU adapter is required; lavapipe does not count");
    eprintln!("adapter {}", specrt::backends::Backend::description(&w));
    w
}

fn setup(spec: &Spec, ctx: usize) -> Wgpu {
    let mut w = device();
    w.compile(&wgsl::generate(spec, ctx).unwrap()).unwrap();
    w
}

fn f32_buf(w: &Wgpu, v: &[f32]) -> Buffer {
    let b = w.alloc(v.len() as u64 * 4).unwrap();
    w.upload(&b, 0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    b
}

fn u32_buf(w: &Wgpu, v: &[u32]) -> Buffer {
    let b = w.alloc(v.len() as u64 * 4).unwrap();
    w.upload(&b, 0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    b
}

fn read_f32(w: &Wgpu, b: &Buffer, n: usize) -> Vec<f32> {
    w.download(b, 0, n as u64 * 4).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn read_u32(w: &Wgpu, b: &Buffer, n: usize) -> Vec<u32> {
    w.download(b, 0, n as u64 * 4).unwrap().chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn half(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let man = ((bits >> 13) & 0x3ff) as u16;
    (sign | ((exp.clamp(1, 30) as u16) << 10) | man).to_le_bytes()
}

fn random_blocks(kind: GgmlType, rows: usize, cols: usize, rng: &mut Lcg) -> Vec<u8> {
    let (elems, bytes) = kind.block().unwrap();
    let blocks = rows * cols / elems;
    let mut out = Vec::new();
    for _ in 0..blocks {
        let mut b: Vec<u8> = (0..bytes).map(|_| rng.next() as u8).collect();
        let scale_at: &[usize] = match kind {
            GgmlType::Q8_0 => &[0],
            GgmlType::Q4K | GgmlType::Q5K => &[0, 2],
            GgmlType::Q6K => &[208],
            _ => &[],
        };
        for &at in scale_at {
            let d = half(rng.unit() * 0.1 + 0.11);
            b[at..at + 2].copy_from_slice(&d);
        }
        if matches!(kind, GgmlType::Q8_0) {
            for chunk in b.chunks_exact_mut(34) {
                chunk[..2].copy_from_slice(&half(rng.unit() * 0.1 + 0.11));
            }
        }
        out.extend(b);
    }
    out
}

fn dequant_on_gpu(w: &mut Wgpu, rows: &DeviceRows) -> Vec<f32> {
    let nrows = rows.rows.len();
    let data = w.alloc(rows.data.len() as u64).unwrap();
    w.upload(&data, 0, &rows.data);
    let table = u32_buf(w, &rows.rows);
    let out = w.alloc((nrows * rows.cols * 4) as u64).unwrap();
    let items = nrows * rows.cols;
    let groups = items.div_ceil(256) as u32;
    let grid = [groups.min(wgsl::ROW_SPAN), groups.div_ceil(wgsl::ROW_SPAN), 1];
    let params = [nrows as u32, rows.cols as u32, 0, 0, 0, 0, 0, 0];
    go(w, "dequant", &[&data, &table, &out], params, grid);
    read_f32(w, &out, items)
}

fn go(w: &mut Wgpu, name: &str, buffers: &[&Buffer], params: [u32; 8], grid: [u32; 3]) {
    w.dispatch(name, buffers, params, grid).unwrap();
    w.submit().unwrap();
}

fn exact_dequant(kind: GgmlType, cols: usize) {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut w = setup(&spec(), 64);
    let rows = 5;
    let mut rng = Lcg(7 + cols as u64);
    let data = random_blocks(kind, rows, cols, &mut rng);
    let packed = DeviceRows::pack(kind, &data, rows, cols).unwrap();
    let got = dequant_on_gpu(&mut w, &packed);
    let mut mismatches = 0;
    let mut worst = 0f32;
    for r in 0..rows {
        let want = packed.row_f32(r);
        for c in 0..cols {
            let g = got[r * cols + c];
            if g.to_bits() != want[c].to_bits() {
                mismatches += 1;
                worst = worst.max((g - want[c]).abs());
            }
        }
    }
    assert_eq!(mismatches, 0, "{} differs from the CPU dequant in {mismatches} values, worst {worst}", kind.name());
}

#[test]
#[ignore = "needs a GPU adapter"]
fn q8_0_dequant_is_bit_exact_with_the_cpu() {
    exact_dequant(GgmlType::Q8_0, 1024);
}

#[test]
#[ignore = "needs a GPU adapter"]
fn q4_k_dequant_is_bit_exact_with_the_cpu() {
    exact_dequant(GgmlType::Q4K, 1024);
}

#[test]
#[ignore = "needs a GPU adapter"]
fn q5_k_dequant_is_bit_exact_with_the_cpu() {
    exact_dequant(GgmlType::Q5K, 1024);
}

#[test]
#[ignore = "needs a GPU adapter"]
fn q6_k_dequant_is_bit_exact_with_the_cpu() {
    exact_dequant(GgmlType::Q6K, 1024);
}

#[test]
#[ignore = "needs a GPU adapter"]
fn bf16_and_f16_rows_decode_exactly() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut w = setup(&spec(), 64);
    let mut rng = Lcg(3);
    let vals: Vec<u16> = (0..4 * 512).map(|_| specrt::weights::f32_to_bf16(rng.unit() * 3.0)).collect();
    let packed = DeviceRows::from_bf16(&vals, 512).unwrap();
    let got = dequant_on_gpu(&mut w, &packed);
    for r in 0..4 {
        assert_eq!(got[r * 512..(r + 1) * 512].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), packed.row_f32(r).iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
    let bytes: Vec<u8> = (0..2 * 512).flat_map(|_| half(rng.unit())).collect();
    let packed = DeviceRows::pack(GgmlType::F16, &bytes, 2, 512).unwrap();
    let got = dequant_on_gpu(&mut w, &packed);
    for r in 0..2 {
        assert_eq!(got[r * 512..(r + 1) * 512].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), packed.row_f32(r).iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
}

fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len());
    let mut worst = 0f32;
    for (g, w) in got.iter().zip(want) {
        worst = worst.max((g - w).abs() / (1.0 + w.abs()));
    }
    assert!(worst <= tol, "{what}: worst relative error {worst} above {tol}");
}

fn run_linear(w: &mut Wgpu, rows: &DeviceRows, x: &[f32], tokens: usize, accumulate: Option<&[f32]>) -> Vec<f32> {
    let nrows = rows.rows.len();
    let data = w.alloc(rows.data.len() as u64).unwrap();
    w.upload(&data, 0, &rows.data);
    let table = u32_buf(w, &rows.rows);
    let xb = f32_buf(w, x);
    let yb = f32_buf(w, &accumulate.map_or_else(|| vec![0.0; tokens * nrows], |a| a.to_vec()));
    let (name, tt) = if tokens == 1 { ("linear", 1) } else { ("linear_t8", wgsl::LINEAR_TOKENS) };
    let grid = [nrows as u32, 1, tokens.div_ceil(tt) as u32];
    let params = [nrows as u32, rows.cols as u32, tokens as u32, 0, 0, accumulate.is_some() as u32, 0, 0];
    go(w, name, &[&xb, &yb, &data, &table], params, grid);
    read_f32(w, &yb, tokens * nrows)
}

fn linear_case(kind: GgmlType, tokens: usize) {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut w = setup(&spec(), 64);
    let (nrows, cols) = (37, 512);
    let mut rng = Lcg(11);
    let packed = if kind == GgmlType::Bf16 {
        let v: Vec<u16> = (0..nrows * cols).map(|_| specrt::weights::f32_to_bf16(rng.unit())).collect();
        DeviceRows::from_bf16(&v, cols).unwrap()
    } else {
        let data = random_blocks(kind, nrows, cols, &mut rng);
        DeviceRows::pack(kind, &data, nrows, cols).unwrap()
    };
    let x: Vec<f32> = (0..tokens * cols).map(|_| rng.unit()).collect();
    let weights: Vec<Vec<f32>> = (0..nrows).map(|r| packed.row_f32(r)).collect();
    let mut want = vec![0f32; tokens * nrows];
    for t in 0..tokens {
        for r in 0..nrows {
            want[t * nrows + r] = (0..cols).map(|c| weights[r][c] as f64 * x[t * cols + c] as f64).sum::<f64>() as f32;
        }
    }
    close(&run_linear(&mut w, &packed, &x, tokens, None), &want, 2e-4, "linear");
    let base: Vec<f32> = (0..tokens * nrows).map(|_| rng.unit()).collect();
    let plus: Vec<f32> = want.iter().zip(&base).map(|(a, b)| a + b).collect();
    close(&run_linear(&mut w, &packed, &x, tokens, Some(&base)), &plus, 2e-4, "linear accumulate");
}

#[test]
#[ignore = "needs a GPU adapter"]
fn linear_matches_a_cpu_matmul_for_bf16_at_one_and_many_tokens() {
    linear_case(GgmlType::Bf16, 1);
    linear_case(GgmlType::Bf16, 11);
}

#[test]
#[ignore = "needs a GPU adapter"]
fn linear_matches_a_cpu_matmul_for_q8_0_and_q4_k() {
    linear_case(GgmlType::Q8_0, 1);
    linear_case(GgmlType::Q8_0, 9);
    linear_case(GgmlType::Q4K, 1);
    linear_case(GgmlType::Q4K, 9);
}

#[test]
#[ignore = "needs a GPU adapter"]
fn rmsnorm_matches_a_cpu_reference() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let s = spec();
    let mut w = setup(&s, 64);
    let h = s.dims.hidden;
    let mut rng = Lcg(5);
    let rows = 3;
    let x: Vec<f32> = (0..rows * h).map(|_| rng.unit() * 4.0).collect();
    let weight: Vec<f32> = (0..h).map(|_| rng.unit() + 1.5).collect();
    let (xb, wb, ob) = (f32_buf(&w, &x), f32_buf(&w, &weight), f32_buf(&w, &vec![0.0; rows * h]));
    go(&mut w, "rmsnorm", &[&xb, &wb, &ob], [0, 0, 0, 0, 0, 0, 0, 0], [rows as u32, 1, 1]);
    let mut want = Vec::new();
    for r in 0..rows {
        let row = &x[r * h..(r + 1) * h];
        let inv = 1.0 / (row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / h as f64 + s.norm.eps).sqrt();
        want.extend(row.iter().zip(&weight).map(|(v, w)| (*w as f64 * (*v as f64 * inv)) as f32));
    }
    close(&read_f32(&w, &ob, rows * h), &want, 1e-5, "rmsnorm");
}

#[test]
#[ignore = "needs a GPU adapter"]
fn swiglu_matches_a_cpu_reference() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let s = spec();
    let mut w = setup(&s, 64);
    let inter = s.dims.intermediate;
    let mut rng = Lcg(9);
    let tokens = 3;
    let gu: Vec<f32> = (0..tokens * 2 * inter).map(|_| rng.unit() * 5.0).collect();
    let (gb, ob) = (f32_buf(&w, &gu), f32_buf(&w, &vec![0.0; tokens * inter]));
    let items = tokens * inter;
    go(&mut w, "swiglu", &[&gb, &ob], [tokens as u32, 0, 0, 0, 0, 0, 0, 0], [items.div_ceil(256) as u32, 1, 1]);
    let mut want = vec![0f32; items];
    for t in 0..tokens {
        for i in 0..inter {
            let g = gu[t * 2 * inter + 2 * i] as f64;
            want[t * inter + i] = (g / (1.0 + (-g).exp()) * gu[t * 2 * inter + 2 * i + 1] as f64) as f32;
        }
    }
    close(&read_f32(&w, &ob, items), &want, 1e-5, "swiglu");
}

fn rope_inv_freq(s: &Spec, i: usize) -> f64 {
    1.0 / s.rope.theta.powf(2.0 * i as f64 / s.rotary_dim() as f64)
}

#[test]
#[ignore = "needs a GPU adapter"]
fn rope_with_qk_norm_and_the_kv_write_match_a_cpu_reference() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let s = spec();
    let ctx = 64;
    let mut w = setup(&s, ctx);
    let a = &s.attention;
    let (hd, rot) = (a.head_dim, s.rotary_dim());
    let rows = 3;
    let pos0 = 5;
    let row_len = s.qkv_rows();
    let mut rng = Lcg(21);
    let qkv: Vec<f32> = (0..rows * row_len).map(|_| rng.unit()).collect();
    let qn: Vec<f32> = (0..hd).map(|_| rng.unit() + 1.5).collect();
    let kn: Vec<f32> = (0..hd).map(|_| rng.unit() + 1.5).collect();
    let kv_len = a.kv_heads * ctx * hd;
    let (qb, qnb, knb) = (f32_buf(&w, &qkv), f32_buf(&w, &qn), f32_buf(&w, &kn));
    let (kc, vc) = (f32_buf(&w, &vec![0.0; kv_len]), f32_buf(&w, &vec![0.0; kv_len]));
    go(&mut w, "rope", &[&qb, &qnb, &knb, &kc, &vc], [pos0 as u32, 0, 0, 0, 0, 0, 0, 0], [(a.heads + 2 * a.kv_heads) as u32, rows as u32, 1]);
    let (gq, gk, gv) = (read_f32(&w, &qb, rows * row_len), read_f32(&w, &kc, kv_len), read_f32(&w, &vc, kv_len));
    let rot_head = |src: &[f32], norm: Option<&Vec<f32>>, pos: usize| -> Vec<f32> {
        let mut v: Vec<f32> = src.to_vec();
        if let Some(n) = norm {
            let inv = 1.0 / (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / hd as f64 + s.norm.eps).sqrt();
            v = v.iter().zip(n).map(|(x, w)| (*w as f64 * (*x as f64 * inv)) as f32).collect();
        }
        let base = v.clone();
        for d in 0..rot {
            let i = if d < rot / 2 { d } else { d - rot / 2 };
            let ang = pos as f64 * rope_inv_freq(&s, i);
            let (c, sn) = (ang.cos(), ang.sin());
            v[d] = if d < rot / 2 { (base[d] as f64 * c - base[d + rot / 2] as f64 * sn) as f32 } else { (base[d] as f64 * c + base[d - rot / 2] as f64 * sn) as f32 };
        }
        v
    };
    for r in 0..rows {
        for h in 0..a.heads {
            let at = r * row_len + h * hd;
            close(&gq[at..at + hd], &rot_head(&qkv[at..at + hd], Some(&qn), pos0 + r), 2e-4, "q rope");
        }
        for h in 0..a.kv_heads {
            let at = r * row_len + (a.heads + h) * hd;
            let slot = (h * ctx + pos0 + r) * hd;
            close(&gk[slot..slot + hd], &rot_head(&qkv[at..at + hd], Some(&kn), pos0 + r), 2e-4, "k cache");
            let at = r * row_len + (a.heads + a.kv_heads + h) * hd;
            close(&gv[slot..slot + hd], &qkv[at..at + hd], 0.0, "v cache");
        }
    }
}

#[test]
#[ignore = "needs a GPU adapter"]
fn causal_attention_matches_a_cpu_softmax_reference() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let s = spec();
    let ctx = 300;
    let mut w = setup(&s, ctx);
    let a = &s.attention;
    let hd = a.head_dim;
    let (pos0, rows) = (200, 3);
    let row_len = s.qkv_rows();
    let mut rng = Lcg(33);
    let qkv: Vec<f32> = (0..rows * row_len).map(|_| rng.unit()).collect();
    let kv_len = a.kv_heads * ctx * hd;
    let kcache: Vec<f32> = (0..kv_len).map(|_| rng.unit()).collect();
    let vcache: Vec<f32> = (0..kv_len).map(|_| rng.unit()).collect();
    let (qb, kc, vc) = (f32_buf(&w, &qkv), f32_buf(&w, &kcache), f32_buf(&w, &vcache));
    let ob = f32_buf(&w, &vec![0.0; rows * a.heads * hd]);
    go(&mut w, "attention", &[&qb, &kc, &vc, &ob], [pos0 as u32, 0, 0, 0, 0, 0, 0, 0], [a.heads as u32, rows as u32, 1]);
    let got = read_f32(&w, &ob, rows * a.heads * hd);
    let mut want = vec![0f32; rows * a.heads * hd];
    for t in 0..rows {
        let len = pos0 + t + 1;
        for h in 0..a.heads {
            let kvh = h / (a.heads / a.kv_heads);
            let q = &qkv[t * row_len + h * hd..t * row_len + (h + 1) * hd];
            let scores: Vec<f64> = (0..len)
                .map(|k| (0..hd).map(|d| q[d] as f64 * kcache[(kvh * ctx + k) * hd + d] as f64).sum::<f64>() / (hd as f64).sqrt())
                .collect();
            let m = scores.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = scores.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for d in 0..hd {
                want[(t * a.heads + h) * hd + d] = ((0..len).map(|k| e[k] * vcache[(kvh * ctx + k) * hd + d] as f64).sum::<f64>() / z) as f32;
            }
        }
    }
    close(&got, &want, 2e-4, "attention");
}

#[test]
#[ignore = "needs a GPU adapter"]
fn argmax_picks_the_greedy_token_and_gumbel_sampling_is_seeded() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let s = spec();
    let mut w = setup(&s, 64);
    let vocab = s.dims.vocab;
    let mut rng = Lcg(77);
    let logits: Vec<f32> = (0..vocab).map(|_| rng.unit() * 3.0).collect();
    let want = logits.iter().enumerate().fold(0, |b, (i, v)| if *v > logits[b] { i } else { b }) as u32;
    let lb = f32_buf(&w, &logits);
    let tb = u32_buf(&w, &[0; 4]);
    go(&mut w, "argmax", &[&lb, &tb], [0, 0, 0, 0, 1, 0, 0, 0], [1, 1, 1]);
    assert_eq!(read_u32(&w, &tb, 4)[1], want);
    let hot = 2.0f32.to_bits();
    let mut sample = |seed: u32| {
        go(&mut w, "argmax", &[&lb, &tb], [0, hot, seed, 3, 2, 0, 0, 0], [1, 1, 1]);
        read_u32(&w, &tb, 4)[2]
    };
    let a = sample(5);
    assert_eq!(a, sample(5));
    assert!((0..8).map(&mut sample).collect::<std::collections::HashSet<_>>().len() > 1);
}

#[test]
fn unsupported_features_are_named() {
    let s = Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b-nvfp4.toml")).unwrap();
    let why = wgsl::unsupported(&s).unwrap();
    assert!(why.contains("not yet on wgpu"), "{why}");
    assert!(wgsl::unsupported(&spec()).is_none());
}

#[test]
fn generated_wgsl_has_no_unfilled_slots() {
    for k in wgsl::generate(&spec(), 128).unwrap() {
        assert!(!k.source.contains("{{"), "{} has an unfilled slot", k.name);
    }
}

const PROMPT: [u32; 18] = [785, 6722, 315, 9625, 374, 12095, 13, 9645, 264, 2805, 32794, 911, 279, 9396, 323, 279, 9759, 13];

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().copied().fold(f32::MIN, f32::max) as f64;
    let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
    row.iter().map(|&v| v as f64 - max - sum.ln()).collect()
}

fn kl(reference: &[f32], ours: &[f32]) -> f64 {
    let (p, q) = (log_softmax(reference), log_softmax(ours));
    p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum()
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
}

fn gguf_dir() -> std::path::PathBuf {
    dirs::home_dir().unwrap().join(".cache/specrt-gguf")
}

fn compare_with_cuda(spec: &Spec, mean_bound: f64, max_bound: f64, greedy: usize, require_equal: bool) {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let vocab = spec.dims.vocab;
    let (cuda_logits, cuda_tokens) = {
        let mut rt = specrt::runtime::Runtime::load(spec, 256).unwrap();
        let l = rt.prompt_logits(&PROMPT).unwrap();
        let g = rt.generate(&PROMPT, greedy, specrt::runtime::Sampling::GREEDY).unwrap().tokens;
        (l, g)
    };
    let be = device();
    let mut dense = specrt::backends::dense::Dense::load(be, spec, 256).unwrap();
    let ours = dense.prompt_logits(&PROMPT).unwrap();
    let rows = PROMPT.len();
    let kls: Vec<f64> = (0..rows).map(|r| kl(&cuda_logits[r * vocab..(r + 1) * vocab], &ours[r * vocab..(r + 1) * vocab])).collect();
    let top1 = (0..rows).filter(|&r| argmax(&cuda_logits[r * vocab..(r + 1) * vocab]) == argmax(&ours[r * vocab..(r + 1) * vocab])).count();
    let (mean, max) = (kls.iter().sum::<f64>() / rows as f64, kls.iter().copied().fold(0.0, f64::max));
    let out = dense.generate(&PROMPT, greedy, specrt::runtime::Sampling::GREEDY).unwrap();
    let same = out.tokens.iter().zip(&cuda_tokens).take_while(|(a, b)| a == b).count();
    eprintln!("{}: mean KL {mean:.3e} max KL {max:.3e} top1 {top1}/{rows} greedy prefix {same}/{greedy} prefill {:.3}s decode {:.3}s", spec.name, out.prefill_secs, out.decode_secs);
    assert!(mean <= mean_bound && max <= max_bound, "mean KL {mean} max KL {max}");
    assert_eq!(top1, rows);
    if require_equal {
        assert_eq!(same, greedy, "greedy tokens differ from CUDA after {same}");
    }
}

#[test]
#[ignore = "needs a GPU adapter, the CUDA backend and Qwen3-0.6B in the HF cache"]
fn qwen3_bf16_logits_match_cuda_and_greedy_64_tokens_are_equal() {
    compare_with_cuda(&spec(), 1e-4, 1e-3, 64, true);
}

#[test]
#[ignore = "needs a GPU adapter, the CUDA backend and the Qwen3-0.6B Q4_K_M GGUF in ~/.cache/specrt-gguf"]
fn qwen3_q4_k_m_gguf_logits_match_cuda_within_the_kl_bound() {
    let s = Spec::from_gguf_file(&gguf_dir().join("Qwen3-0.6B-Q4_K_M.gguf")).unwrap();
    compare_with_cuda(&s, 1e-3, 1e-2, 32, false);
}

#[test]
#[ignore = "needs a GPU adapter, the CUDA backend and the Llama-3.2-1B Q4_K_M GGUF in ~/.cache/specrt-gguf"]
fn llama_3_2_1b_q4_k_m_gguf_logits_match_cuda_within_the_kl_bound() {
    let s = Spec::from_gguf_file(&gguf_dir().join("Llama-3.2-1B-Instruct-Q4_K_M.gguf")).unwrap();
    compare_with_cuda(&s, 1e-3, 1e-2, 32, false);
}
