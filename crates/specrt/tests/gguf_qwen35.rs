use specrt::gguf::Gguf;
use specrt::spec::{Block, Quant, Spec, VHeadOrder};
use specrt::weights::{Checkpoint, Gdn, Layer, Matrix, Mixer, bf16_to_f32};
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn gguf_path(name: &str) -> PathBuf {
    fixtures().join("qwen35-gguf").join(name)
}

fn hf_spec(dir: &str) -> Spec {
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(fixtures().join(dir).join("config.json")).unwrap()).unwrap();
    spec.repo = fixtures().join(dir).display().to_string();
    spec
}

fn gguf_spec(file: &str) -> Spec {
    Spec::from_gguf_file(&gguf_path(file)).unwrap()
}

fn to_f32(m: &Matrix) -> Vec<f32> {
    match m {
        Matrix::Bf16(w) => w.iter().map(|&v| bf16_to_f32(v)).collect(),
        Matrix::Q8 { values, scales } => values.chunks(32).zip(scales).flat_map(|(g, s)| g.iter().map(move |&q| q as f32 * s)).collect(),
        Matrix::Ggml(d) => (0..d.rows.len()).flat_map(|r| d.row_f32(r)).collect(),
        _ => unreachable!("no fp8 or fp4 in these fixtures"),
    }
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn gdn<'a>(l: &'a Layer<'a>) -> &'a Gdn<'a> {
    match &l.mixer {
        Mixer::Linear(g) => g,
        _ => panic!("layer 0 is linear"),
    }
}

#[test]
fn a_qwen35_gguf_gives_the_same_spec_as_the_safetensors_config() {
    let mut hf = hf_spec("tiny-qwen3.5-hd256");
    let spec = gguf_spec("tiny-mtp-Q8_0.gguf");
    assert_eq!(spec.layers.pattern, [Block::Linear, Block::Linear, Block::Full]);
    assert_eq!(spec.layers.count, 3);
    let la = spec.linear_attention.as_ref().unwrap();
    assert_eq!(la.v_heads, VHeadOrder::Tiled);
    assert_eq!((la.key_heads, la.value_heads, la.key_head_dim, la.value_head_dim, la.conv_kernel), (2, 6, 128, 128, 4));
    assert!(spec.attention.output_gate && spec.attention.qk_norm && !spec.attention.qkv_bias);
    assert_eq!(spec.rope.mrope.as_ref().map(|m| (m.section, m.interleaved)), Some(([11, 11, 10], true)));
    assert_eq!(spec.mtp.as_ref().map(|m| m.layers), Some(1));
    hf.name = spec.name.clone();
    hf.repo = spec.repo.clone();
    hf.quant = spec.quant;
    hf.runtime = spec.runtime.clone();
    hf.mtp = spec.mtp.clone();
    hf.norm.unit_offset = false;
    hf.norm.eps = spec.norm.eps;
    hf.rope.mrope = spec.rope.mrope.clone();
    hf.linear_attention.as_mut().unwrap().v_heads = VHeadOrder::Tiled;
    assert_eq!(spec, hf);
}

#[test]
fn every_quant_recipe_of_a_qwen35_gguf_builds_a_runnable_spec() {
    for (file, quant) in [("tiny-mtp-Q8_0.gguf", Quant::Gguf), ("tiny-mtp-Q4_K_M.gguf", Quant::Gguf), ("tiny-mtp-Q5_K_M.gguf", Quant::Gguf), ("tiny-mtp-Q4_0.gguf", Quant::Gguf), ("tiny-mtp-Q5_0.gguf", Quant::Gguf)] {
        let spec = gguf_spec(file);
        assert_eq!(spec.quant, quant, "{file}");
        assert!(spec.runtime.unsplit.contains(&"lin_in".to_string()), "{file}");
        spec.validate().unwrap();
    }
}

fn hf_head(gguf_head: usize, k_heads: usize, v_per_k: usize) -> usize {
    (gguf_head % k_heads) * v_per_k + gguf_head / k_heads
}

fn to_hf_order(file_order: &[f32], per_head: usize, k_heads: usize, v_per_k: usize) -> Vec<f32> {
    let mut out = vec![0f32; file_order.len()];
    for (h, chunk) in file_order.chunks_exact(per_head).enumerate() {
        let at = hf_head(h, k_heads, v_per_k) * per_head;
        out[at..at + per_head].copy_from_slice(chunk);
    }
    out
}

fn columns_to_hf_order(file_order: &[f32], rows: usize, k_heads: usize, v_per_k: usize, head: usize) -> Vec<f32> {
    let cols = file_order.len() / rows;
    let mut out = vec![0f32; file_order.len()];
    for r in 0..rows {
        let row = &file_order[r * cols..(r + 1) * cols];
        out[r * cols..(r + 1) * cols].copy_from_slice(&to_hf_order(row, head, k_heads, v_per_k));
    }
    out
}

#[test]
fn qwen35_gguf_stores_every_value_head_tensor_in_one_consistent_tiled_order() {
    let hf = hf_spec("tiny-qwen3.5-hd256");
    let want = Checkpoint::open(&fixtures().join("tiny-qwen3.5-hd256")).unwrap();
    let want0 = want.layer(&hf, 0).unwrap();
    let want_g = gdn(&want0);
    let spec = gguf_spec("tiny-mtp-Q8_0.gguf");
    let ck = Checkpoint::open(&gguf_path("tiny-mtp-Q8_0.gguf")).unwrap();
    let got0 = ck.layer(&spec, 0).unwrap();
    let got_g = gdn(&got0);
    let (kh, vh, d, k, h) = (2usize, 6usize, 128usize, 4usize, hf.dims.hidden);
    let ratio = vh / kh;
    let (qk, value) = (2 * kh * d, vh * d);
    let (got_in, want_in) = (to_f32(&got_g.in_proj), to_f32(&want_g.in_proj));
    assert!(rel_rms(&got_in[..qk * h], &want_in[..qk * h]) < 0.02, "q and k rows are untouched");
    assert!(rel_rms(&to_hf_order(&got_in[qk * h..(qk + value) * h], d * h, kh, ratio), &want_in[qk * h..(qk + value) * h]) < 0.02, "v rows");
    assert!(rel_rms(&to_hf_order(&got_in[(qk + value) * h..], d * h, kh, ratio), &want_in[(qk + value) * h..]) < 0.02, "z rows");
    let (got_ba, want_ba) = (to_f32(&got_g.ba), to_f32(&want_g.ba));
    assert!(rel_rms(&to_hf_order(&got_ba[..vh * h], h, kh, ratio), &want_ba[..vh * h]) < 0.02, "beta rows");
    assert!(rel_rms(&to_hf_order(&got_ba[vh * h..], h, kh, ratio), &want_ba[vh * h..]) < 0.02, "alpha rows");
    assert!(rel_rms(&columns_to_hf_order(&to_f32(&got_g.out), h, kh, ratio, d), &to_f32(&want_g.out)) < 0.02, "out columns");
    assert!(rel_rms(&to_hf_order(&got_g.dt_bias, 1, kh, ratio), &want_g.dt_bias) < 1e-2, "dt bias");
    assert!(rel_rms(&to_hf_order(&got_g.neg_a, 1, kh, ratio), &want_g.neg_a) < 1e-2, "-exp(A_log)");
    assert!(rel_rms(&got_g.conv[..qk * k], &want_g.conv[..qk * k]) < 1e-3, "conv q and k channels");
    assert!(rel_rms(&to_hf_order(&got_g.conv[qk * k..], d * k, kh, ratio), &want_g.conv[qk * k..]) < 1e-3, "conv v channels");
}

#[test]
fn qwen35_gguf_norms_conv_and_attention_layers_match_the_safetensors() {
    let hf = hf_spec("tiny-qwen3.5-hd256");
    let want = Checkpoint::open(&fixtures().join("tiny-qwen3.5-hd256")).unwrap();
    let spec = gguf_spec("tiny-mtp-Q8_0.gguf");
    let ck = Checkpoint::open(&gguf_path("tiny-mtp-Q8_0.gguf")).unwrap();
    for i in 0..3 {
        let (w, g) = (want.layer(&hf, i).unwrap(), ck.layer(&spec, i).unwrap());
        assert_eq!(w.ln1.len(), g.ln1.len());
        assert!(rel_rms(&g.ln1, &w.ln1) < 1e-2, "ln1 {i}");
        assert!(rel_rms(&g.ln2, &w.ln2) < 1e-2, "ln2 {i} {:?} {:?}", &g.ln2[..4], &w.ln2[..4]);
        assert!(rel_rms(&to_f32(&g.down), &to_f32(&w.down)) < 0.02, "down {i}");
        assert!(rel_rms(&to_f32(&g.gate_up), &to_f32(&w.gate_up)) < 0.02, "gate_up {i}");
    }
    let (w, g) = (want.layer(&hf, 2).unwrap(), ck.layer(&spec, 2).unwrap());
    let (Mixer::Attention(w), Mixer::Attention(g)) = (&w.mixer, &g.mixer) else { panic!("layer 2 is attention") };
    assert!(rel_rms(&to_f32(&g.qkv), &to_f32(&w.qkv)) < 0.02, "qkv");
    assert!(rel_rms(&to_f32(&g.o), &to_f32(&w.o)) < 0.02, "o");
    assert!(rel_rms(g.q_norm.as_ref().unwrap(), w.q_norm.as_ref().unwrap()) < 1e-2, "q_norm");
    assert!(rel_rms(g.k_norm.as_ref().unwrap(), w.k_norm.as_ref().unwrap()) < 1e-2, "k_norm");
    let (w0, g0) = (want.layer(&hf, 0).unwrap(), ck.layer(&spec, 0).unwrap());
    let (kh, vh, d, k) = (2usize, 6usize, 128usize, 4usize);
    let conv_dim = 2 * kh * d + vh * d;
    let (wg, gg) = (gdn(&w0), gdn(&g0));
    assert_eq!(gg.conv.len(), conv_dim * k);
    assert!(rel_rms(&gg.norm, &wg.norm) < 1e-6, "ssm norm");
}

#[test]
fn qwen35_gguf_mtp_block_matches_the_safetensors_mtp() {
    let hf = hf_spec("tiny-qwen3.5-hd256-mtp");
    let want_ck = Checkpoint::open(&fixtures().join("tiny-qwen3.5-hd256-mtp")).unwrap();
    let want = want_ck.mtp(&hf).unwrap();
    let spec = gguf_spec("tiny-mtp-Q8_0.gguf");
    let got_ck = Checkpoint::open(&gguf_path("tiny-mtp-Q8_0.gguf")).unwrap();
    let got = got_ck.mtp(&spec).unwrap();
    assert!(rel_rms(&to_f32(&got.fc), &to_f32(&want.fc)) < 0.02, "fc");
    assert!(rel_rms(&got.norm_embed, &want.norm_embed) < 1e-2, "enorm");
    assert!(rel_rms(&got.norm_hidden, &want.norm_hidden) < 1e-2, "hnorm");
    assert!(rel_rms(&got.norm, &want.norm) < 1e-2, "head norm");
    assert!(rel_rms(&got.layer.ln1, &want.layer.ln1) < 1e-2, "mtp attn norm");
    assert!(rel_rms(&to_f32(&got.layer.gate_up), &to_f32(&want.layer.gate_up)) < 0.02, "mtp gate_up");
    let (Mixer::Attention(w), Mixer::Attention(g)) = (&want.layer.mixer, &got.layer.mixer) else { panic!("the MTP layer is attention") };
    assert!(rel_rms(&to_f32(&g.qkv), &to_f32(&w.qkv)) < 0.02, "mtp qkv");
}

#[test]
fn qwen35_quant_recipes_load_inside_their_quant_bound() {
    let hf = hf_spec("tiny-qwen3.5-hd256");
    let want = Checkpoint::open(&fixtures().join("tiny-qwen3.5-hd256")).unwrap();
    let want_down = to_f32(&want.layer(&hf, 0).unwrap().down);
    let bounds = [("tiny-mtp-Q8_0.gguf", 0.01), ("tiny-mtp-Q5_K_M.gguf", 0.06), ("tiny-mtp-Q5_0.gguf", 0.06), ("tiny-mtp-Q4_K_M.gguf", 0.12), ("tiny-mtp-Q4_0.gguf", 0.14)];
    for (file, bound) in bounds {
        let spec = gguf_spec(file);
        let ck = Checkpoint::open(&gguf_path(file)).unwrap();
        let got = to_f32(&ck.layer(&spec, 0).unwrap().down);
        let err = rel_rms(&got, &want_down);
        assert!(err < bound, "{file}: relative rms error {err} over {bound}");
    }
}

#[test]
fn an_unsupported_architecture_lists_each_supported_one_once() {
    let bytes = std::fs::read(gguf_path("tiny-mtp-Q8_0.gguf")).unwrap();
    let mut patched = bytes.clone();
    let at = patched.windows(6).position(|w| w == b"qwen35").unwrap();
    patched[at..at + 6].copy_from_slice(b"qwen99");
    let g = Gguf::from_bytes(patched).unwrap();
    let e = g.spec().err().unwrap().to_string();
    let listed = e.split_once('(').unwrap().1.split_once(" are)").unwrap().0;
    let names: Vec<&str> = listed.split(", ").collect();
    let mut unique = names.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(names.len(), unique.len(), "{e}");
    for arch in ["llama", "qwen2", "qwen3", "qwen3moe", "qwen35", "qwen35moe"] {
        assert!(names.contains(&arch), "{arch} missing from {e}");
    }
}

#[test]
fn the_qwen35_gguf_tokenizer_uses_the_family_pretokenizer_and_the_nfc_flag() {
    use specrt::gguf::writer::{Kv, build};
    let bytes = build(
        3,
        &[
            ("general.architecture", Kv::Str("qwen35")),
            ("tokenizer.ggml.model", Kv::Str("gpt2")),
            ("tokenizer.ggml.pre", Kv::Str("qwen35")),
            ("tokenizer.ggml.tokens", Kv::Strs(&["a", "b", "ab"])),
            ("tokenizer.ggml.token_type", Kv::U32s(&[1, 1, 1])),
            ("tokenizer.ggml.merges", Kv::Strs(&["a b"])),
            ("tokenizer.ggml.normalizer.nfc", Kv::Bool(true)),
        ],
        &[],
    );
    let json = Gguf::from_bytes(bytes).unwrap().tokenizer_json().unwrap();
    assert!(json.contains("\\\\p{M}"), "the qwen35 pre-tokenizer keeps combining marks in words");
    assert!(json.contains("\"NFC\""));
    let tok: tokenizers::Tokenizer = json.parse().unwrap();
    assert_eq!(tok.encode("ab", false).unwrap().get_ids(), [2]);
}

#[test]
fn the_hybrid_gguf_module_decodes_ggml_rows_in_the_linear_attention_kernels() {
    let spec = gguf_spec("tiny-mtp-Q4_K_M.gguf");
    let module = specrt::codegen::Module::generate(&spec, 256).unwrap();
    let rows = spec.linear_attention.as_ref().unwrap().in_proj_rows();
    assert!(module.source.contains(&format!("linear_norm<DecGgml, {rows}, false>")));
    assert!(module.source.contains("constexpr bool LA_TILED = true;"));
    let grouped = specrt::codegen::Module::generate(&hf_spec("tiny-qwen3.5-hd256"), 256).unwrap();
    assert!(grouped.source.contains("constexpr bool LA_TILED = false;"));
}

#[test]
fn the_hybrid_gguf_module_compiles_with_nvrtc_without_a_device() {
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    for file in ["tiny-mtp-Q4_K_M.gguf", "tiny-mtp-Q8_0.gguf"] {
        let spec = gguf_spec(file);
        let module = specrt::codegen::Module::generate(&spec, 256).unwrap();
        let opts = CompileOptions { arch: Some("compute_120"), options: vec!["--std=c++17".to_string()], ..Default::default() };
        let ptx = compile_ptx_with_opts(&module.source, opts).unwrap_or_else(|e| panic!("{file}: {e:?}")).to_src();
        assert!(ptx.contains("lin_in_norm"), "{file}");
        assert!(ptx.contains("gdn_step"), "{file}");
    }
}

#[test]
#[ignore = "set SPECRT_GGUF_FILE to a real qwen35 GGUF"]
fn a_real_qwen35_gguf_loads_a_linear_layer_an_attention_layer_and_the_mtp_block() {
    let path = std::env::var("SPECRT_GGUF_FILE").unwrap();
    let spec = Spec::from_gguf_file(Path::new(&path)).unwrap();
    eprintln!("{} layers {} pattern {:?} quant {:?} mtp {:?}", spec.name, spec.layers.count, spec.layers.pattern, spec.quant, spec.mtp);
    let ck = Checkpoint::open(Path::new(&path)).unwrap();
    for i in [0, spec.layers.pattern.len() - 1] {
        ck.layer(&spec, i).unwrap();
    }
    if spec.speculative().is_some() {
        ck.mtp(&spec).unwrap();
    }
}
