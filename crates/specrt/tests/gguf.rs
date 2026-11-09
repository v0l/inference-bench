use specrt::gguf::{DeviceRows, GgmlType, dequantize};

const GOLDEN: &[u8] = include_bytes!("fixtures/ggml-golden.bin");
const ROWS: usize = 4;

struct Golden {
    kind: GgmlType,
    quant: &'static [u8],
    want: Vec<u32>,
}

fn u32_at(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize
}

fn golden() -> Vec<Golden> {
    let mut at = 0;
    let mut out = Vec::new();
    while at < GOLDEN.len() {
        let (ty, n, bytes) = (u32_at(GOLDEN, at), u32_at(GOLDEN, at + 4), u32_at(GOLDEN, at + 8));
        at += 12;
        let quant = &GOLDEN[at..at + bytes];
        at += bytes;
        let want = GOLDEN[at..at + 4 * n].chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        at += 4 * n;
        out.push(Golden { kind: GgmlType::from_id(ty as u32), quant, want });
    }
    out
}

fn assert_bits(label: &str, got: &[f32], want: &[u32]) {
    assert_eq!(got.len(), want.len(), "{label}");
    let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i].to_bits() != want[i]).collect();
    assert!(bad.is_empty(), "{label} differs from ggml at {} of {} values, first {:?}", bad.len(), got.len(), bad.first().map(|&i| (i, got[i], f32::from_bits(want[i]))));
}

fn gguf_dir() -> std::path::PathBuf {
    std::env::var_os("SPECRT_GGUF_DIR").filter(|d| !d.is_empty()).map(Into::into).unwrap_or_else(|| dirs::home_dir().unwrap().join(".cache/specrt-gguf"))
}

fn assets(file: &str) -> (specrt::spec::Spec, tokenizers::Tokenizer, specrt::gguf::Gguf) {
    let path = gguf_dir().join(file);
    let g = specrt::gguf::Gguf::open(&path).unwrap();
    let tok: tokenizers::Tokenizer = g.tokenizer_json().unwrap().parse().unwrap();
    (specrt::spec::Spec::from_gguf_file(&path).unwrap(), tok, g)
}

const TOKENIZE: &str = "Hello world! 12345 <|im_start|>user\n  naïve café, isn't it?";

fn ids(tok: &tokenizers::Tokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, false).unwrap().get_ids().to_vec()
}

#[test]
#[ignore = "needs the unsloth Qwen3-0.6B Q8_0 GGUF in ~/.cache/specrt-gguf"]
fn the_qwen3_gguf_spec_and_tokenizer_match_the_hf_spec_and_llama_cpp() {
    let (spec, tok, g) = assets("Qwen3-0.6B-Q8_0.gguf");
    let mut hf = specrt::spec::Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/qwen3-0.6b.toml")).unwrap();
    (hf.name, hf.repo, hf.quant) = (spec.name.clone(), spec.repo.clone(), specrt::spec::Quant::Q8);
    hf.norm.eps = spec.norm.eps;
    assert_eq!(spec, hf);
    assert_eq!(ids(&tok, TOKENIZE), [9707, 1879, 0, 220, 16, 17, 18, 19, 20, 220, 151644, 872, 198, 220, 94880, 586, 51950, 11, 4436, 944, 432, 30]);
    let chat = specrt::chat::ChatTemplate::from_tokenizer_config(&g.tokenizer_config_json().unwrap()).unwrap().unwrap();
    assert!(chat.render_user("hi").unwrap().contains("<|im_start|>user\nhi<|im_end|>"));
    let stop = specrt::engine::stop_ids(Some(&g.generation_config_json()), g.tokenizer_config_json().as_deref(), &|t| tok.token_to_id(t));
    assert!(stop.contains(&151645), "{stop:?}");
    let q4 = specrt::spec::Spec::from_gguf_file(&gguf_dir().join("Qwen3-0.6B-Q4_K_M.gguf")).unwrap();
    assert_eq!(q4.quant, specrt::spec::Quant::Gguf);
}

#[test]
#[ignore = "needs the bartowski Llama-3.2-1B-Instruct Q4_K_M GGUF in ~/.cache/specrt-gguf"]
fn the_llama_gguf_spec_recovers_llama3_rope_scaling_and_tokenizes_like_llama_cpp() {
    let (spec, tok, _) = assets("Llama-3.2-1B-Instruct-Q4_K_M.gguf");
    let mut hf = specrt::spec::Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/llama-3.2-1b.toml")).unwrap();
    (hf.name, hf.repo, hf.quant) = (spec.name.clone(), spec.repo.clone(), specrt::spec::Quant::Gguf);
    hf.norm.eps = spec.norm.eps;
    hf.rope.theta = spec.rope.theta;
    hf.runtime.unsplit = ["qkv", "o", "gate_up", "down"].map(String::from).to_vec();
    assert_eq!(spec, hf);
    assert_eq!(ids(&tok, TOKENIZE), [9906, 1917, 0, 220, 4513, 1774, 83739, 318, 5011, 91, 29, 882, 198, 220, 95980, 588, 53050, 11, 4536, 956, 433, 30]);
}

#[test]
fn every_quant_type_dequantises_bit_exact_against_ggml() {
    let all = golden();
    for g in &all {
        let mut got = vec![0f32; g.want.len()];
        dequantize(g.kind, g.quant, &mut got).unwrap();
        assert_bits(&g.kind.name(), &got, &g.want);
    }
    assert_eq!(all.iter().map(|g| g.kind).collect::<Vec<_>>(), [GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K]);
}

#[test]
fn device_rows_keep_every_type_bit_exact_and_mix_types_by_row() {
    let all = golden();
    let cols = all[0].want.len() / ROWS;
    let mut mixed: Option<DeviceRows> = None;
    for g in &all {
        let rows = DeviceRows::pack(g.kind, g.quant, ROWS, cols).unwrap();
        let got: Vec<f32> = (0..ROWS).flat_map(|r| rows.row_f32(r)).collect();
        assert_bits(&g.kind.name(), &got, &g.want);
        match &mut mixed {
            None => mixed = Some(rows),
            Some(m) => {
                let table = m.append(rows);
                m.rows.extend(table);
            }
        }
    }
    let mixed = mixed.unwrap();
    let want: Vec<u32> = all.iter().flat_map(|g| g.want.iter().copied()).collect();
    let got: Vec<f32> = (0..mixed.rows.len()).flat_map(|r| mixed.row_f32(r)).collect();
    assert_bits("mixed", &got, &want);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_gpu_dequantises_mixed_rows_bit_exact_against_ggml() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    let all = golden();
    let cols = all[0].want.len() / ROWS;
    let mut mixed: Option<DeviceRows> = None;
    for g in all.iter().rev() {
        let rows = DeviceRows::pack(g.kind, g.quant, ROWS, cols).unwrap();
        match &mut mixed {
            None => mixed = Some(rows),
            Some(m) => {
                let table = m.append(rows);
                m.rows.extend(table);
            }
        }
    }
    let mixed = mixed.unwrap();
    let want: Vec<u32> = all.iter().rev().flat_map(|g| g.want.iter().copied()).collect();
    let spec = specrt::spec::Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/qwen3-0.6b.toml")).unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let dir = std::env::temp_dir().join(format!("specrt-gguf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let module = specrt::codegen::Module::generate(&spec, 256).unwrap();
    let loaded = specrt::jit::Jit::new(ctx.clone(), dir).unwrap().load(&module).unwrap();
    let stream = ctx.default_stream();
    let data = stream.clone_htod(&mixed.data).unwrap();
    let table = stream.clone_htod(&mixed.rows).unwrap();
    let rows = mixed.rows.len();
    let out = stream.alloc_zeros::<f32>(rows * cols).unwrap();
    let f = loaded.function("gg_dequant_rows").unwrap();
    let (r, c) = (rows as i32, cols as i32);
    let mut b = stream.launch_builder(&f);
    b.arg(&data).arg(&table).arg(&out).arg(&r).arg(&c);
    let chunks = (rows * cols / 8) as u32;
    unsafe { b.launch(LaunchConfig { grid_dim: (chunks.div_ceil(128), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }) }.unwrap();
    let got = stream.clone_dtoh(&out).unwrap();
    assert_bits("gpu", &got, &want);
}
