use specrt::codegen::Module;
use specrt::runtime::{Runtime, Sampling};
use specrt::spec::{Quant, Spec};
use specrt::weights::{Checkpoint, Stacked, ggml_expert_row};
use std::path::{Path, PathBuf};

const FILES: [&str; 2] = ["tiny-qwen3moe-Q4_K_M.gguf", "tiny-qwen3moe-Q8_0.gguf"];
const PROMPT: [u32; 16] = [11, 400, 23, 999, 5, 77, 301, 88, 640, 2, 19, 250, 512, 33, 701, 90];

fn path(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3moe-gguf").join(file)
}

fn spec(file: &str, quantized: bool) -> Spec {
    let mut s = Spec::from_gguf_file(&path(file)).unwrap();
    s.moe.as_mut().unwrap().quantized_experts = quantized;
    s
}

fn bf16(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

fn stacked_bytes(s: &Stacked) -> usize {
    match s {
        Stacked::Bf16(w) => w.len() * 2,
        Stacked::Ggml { data, tables } => data.len() + tables.len() * 4,
        _ => unreachable!("gguf experts are bf16 or ggml"),
    }
}

#[test]
fn quantised_gguf_experts_hold_the_file_blocks_and_decode_to_the_expanded_bf16() {
    for file in FILES {
        let ck = Checkpoint::open(&path(file)).unwrap();
        let g = specrt::gguf::Gguf::open(&path(file)).unwrap();
        let mut totals = (0usize, 0usize);
        let (plain, kept) = (spec(file, false), spec(file, true));
        let (h, inter, e) = (plain.dims.hidden, plain.moe.as_ref().unwrap().expert_intermediate, plain.moe.as_ref().unwrap().experts);
        for layer in 0..plain.layers.count {
            let a = ck.layer_at(&plain, layer, layer).unwrap().moe.unwrap();
            let b = ck.layer_at(&kept, layer, layer).unwrap().moe.unwrap();
            for (which, x, y, rows, cols) in [("gate", &a.gate, &b.gate, inter, h), ("up", &a.up, &b.up, inter, h), ("down", &a.down, &b.down, h, inter)] {
                let Stacked::Bf16(full) = x else { panic!("{file} {which}: the default path expands to bf16") };
                let Stacked::Ggml { data, tables } = y else { panic!("{file} {which}: quantized_experts keeps ggml blocks") };
                assert_eq!(full.len(), e * rows * cols);
                assert_eq!(tables.len(), e * (rows + 1));
                for expert in 0..e {
                    for row in [0, 1, rows / 2, rows - 1] {
                        let got = ggml_expert_row(data, tables, rows, cols, expert, row);
                        let want = &full[(expert * rows + row) * cols..(expert * rows + row + 1) * cols];
                        for (c, (g, w)) in got.iter().zip(want).enumerate() {
                            assert_eq!(specrt::weights::f32_to_bf16(*g), *w, "{file} layer {layer} {which} expert {expert} row {row} col {c}: {g} against {}", bf16(*w));
                        }
                    }
                }
                let file_bytes = g.tensor(&format!("blk.{layer}.ffn_{which}_exps.weight")).unwrap().bytes;
                let Stacked::Ggml { data, .. } = y else { unreachable!() };
                let padded = (file_bytes as f64 * 224.0 / 210.0).ceil() as usize + 16 * e;
                eprintln!("{file} layer {layer} {which}: bf16 {} bytes, ggml {} bytes ({file_bytes} in the file), {:.2}x smaller", stacked_bytes(x), stacked_bytes(y), stacked_bytes(x) as f64 / stacked_bytes(y) as f64);
                assert!(data.len() >= file_bytes && data.len() <= padded, "{file} {which}: {} bytes kept for {file_bytes} in the file", data.len());
                totals.0 += stacked_bytes(x);
                totals.1 += stacked_bytes(y);
            }
        }
        let ratio = totals.0 as f64 / totals.1 as f64;
        eprintln!("{file}: experts {} bytes expanded, {} bytes kept, {ratio:.2}x", totals.0, totals.1);
        assert!(ratio > if file.contains("Q8_0") { 1.8 } else { 3.0 }, "{file}: {ratio:.2}x");
    }
}

#[test]
fn the_anatomy_counts_expanded_experts_as_bf16_and_kept_ones_at_their_block_size() {
    let file = FILES[0];
    let (plain, kept) = (spec(file, false), spec(file, true));
    let a = specrt::anatomy::ModelAnatomy::new(&plain, 0, None);
    let b = specrt::anatomy::ModelAnatomy::new(&kept, 0, None);
    let routed = |x: &specrt::anatomy::ModelAnatomy| x.layers.iter().map(|l| l.bytes).sum::<u64>();
    assert!(routed(&a) as f64 > 2.5 * routed(&b) as f64, "{} against {}", routed(&a), routed(&b));
}

#[test]
fn quantized_experts_needs_gguf_weights() {
    let mut s = Spec::from_hf_config(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5-moe/config.json")).unwrap()).unwrap();
    s.moe.as_mut().unwrap().quantized_experts = true;
    let err = s.check_moe_runnable().unwrap_err().to_string();
    assert!(err.contains("quantized_experts"), "{err}");
    s.moe.as_mut().unwrap().quantized_experts = false;
    s.check_moe_runnable().unwrap();
}

#[test]
fn the_option_adds_only_the_ggml_expert_kernels_and_leaves_the_rest_of_the_module_alone() {
    for file in FILES {
        let off = Module::generate(&spec(file, false), 256).unwrap();
        let on = Module::generate(&spec(file, true), 256).unwrap();
        let added: Vec<&str> = on.kernels.iter().copied().filter(|k| !off.kernels.contains(k)).collect();
        assert_eq!(added, ["moe_up_gg", "moe_down_gg", "moe_up_t_gg", "moe_down_t_gg", "moe_group_up_gg", "moe_group_down_gg"], "{file}");
        assert!(on.source.as_str().starts_with(off.source.as_str().trim_end()), "{file}: the option must only append the ggml expert family");
        assert_ne!(on.kir_hash, off.kir_hash);
    }
}

#[test]
fn the_ggml_expert_module_compiles_with_nvrtc_without_a_device() {
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    for file in FILES {
        let module = Module::generate(&spec(file, true), 256).unwrap();
        let opts = CompileOptions { arch: Some("compute_120"), options: vec!["--std=c++17".to_string()], ..Default::default() };
        let ptx = compile_ptx_with_opts(module.source.as_str(), opts).unwrap_or_else(|e| panic!("{file}: {e:?}")).to_src();
        for k in ["moe_up_gg", "moe_down_gg", "moe_group_up_gg", "moe_group_down_gg"] {
            assert!(ptx.contains(&format!(".entry {k}(")), "{file}: {k}");
        }
    }
}

#[test]
fn each_expert_of_a_ggml_stack_is_one_fixed_size_slot_for_the_expert_cache() {
    let file = FILES[0];
    let s = spec(file, true);
    let ck = Checkpoint::open(&path(file)).unwrap();
    let m = ck.layer_at(&s, 1, 1).unwrap().moe.unwrap();
    let e = s.moe.as_ref().unwrap().experts;
    let Stacked::Ggml { data, tables } = &m.down else { panic!("ggml") };
    assert_eq!(data.len() % e, 0);
    assert_eq!(tables.len() % e, 0);
    let rows = tables.len() / e - 1;
    let stride = data.len() / e;
    for x in 0..e {
        let t = &tables[x * (rows + 1)..(x + 1) * (rows + 1)];
        assert_eq!(t[rows] as usize * 16, stride);
        assert!(t[..rows].iter().all(|r| ((r >> 3) as usize) * 16 < stride), "expert {x} rows stay inside its slot");
        assert_eq!(t, &tables[..rows + 1], "every expert has the same relative row table, so a cache slot can hold any of them");
    }
}

#[test]
fn the_pointer_table_gets_ggml_table_kernels_that_resolve_slots_and_read_bias_by_expert_id() {
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    let mut s = spec(FILES[0], true);
    s.runtime.offload.pointer_table = true;
    s.runtime.offload.vram_budget = 1 << 30;
    let module = Module::generate(&s, 256).unwrap();
    for k in ["moe_up_t_gg", "moe_down_t_gg", "moe_up_gg", "moe_down_gg"] {
        assert!(module.kernels.contains(&k), "{k}");
    }
    let text = module.source.as_str();
    let body = |name: &str| {
        let at = text.find(&format!(" {name}(")).unwrap_or_else(|| panic!("{name}"));
        &text[at..at + text[at..].find("\n}\n").unwrap()]
    };
    for name in ["moe_up_gg_tbl_impl", "moe_down_gg_tbl_impl"] {
        let b = body(name);
        assert!(b.contains("expert = tbl[expert];") && b.contains("expert -= 1;"), "{name} resolves the slot from the table");
        assert!(b.contains("b[(size_t)eid") && !b.contains("b[(size_t)expert"), "{name} reads the bias by expert id");
        assert!(b.contains("moe_dot_gg<"), "{name} dots ggml rows");
    }
    for name in ["moe_up_gg_impl", "moe_down_gg_impl", "moe_group_up_gg_impl", "moe_group_down_gg_impl"] {
        assert!(body(name).contains("b[(size_t)expert"), "{name} reads the staged bias by slot");
    }
    let opts = CompileOptions { arch: Some("compute_120"), options: vec!["--std=c++17".to_string()], ..Default::default() };
    let ptx = compile_ptx_with_opts(text, opts).unwrap().to_src();
    for k in ["moe_up_t_gg", "moe_down_t_gg", "moe_up_t_bf16"] {
        assert!(ptx.contains(&format!(".entry {k}(")), "{k}");
    }
}

fn cached(file: &str, slots: usize, table: bool) -> Spec {
    let mut s = spec(file, true);
    s.runtime.offload.vram_budget = u64::MAX / 4;
    s.runtime.offload.expert_slots = slots;
    s.runtime.offload.pointer_table = table;
    s
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn kept_experts_through_the_expert_cache_and_the_pointer_table_decode_like_resident_ones() {
    let long: Vec<u32> = (0..40u32).map(|i| (i * 53 + 7) % 1000).collect();
    for file in FILES {
        let mut full = Runtime::load(&spec(file, true), 256).unwrap();
        let want = full.trace_decode(&PROMPT[..6]).unwrap();
        let want_tokens = full.generate(&long, 16, Sampling::GREEDY).unwrap().tokens;
        let want_prefill = full.prompt_logits(&long).unwrap();
        drop(full);
        let e = spec(file, true).moe.unwrap().experts;
        let layers = spec(file, true).layers.count;
        for (slots, table) in [(e / 2, false), (e * layers, true), (e / 2, true)] {
            let mut rt = Runtime::load(&cached(file, slots, table), 256).unwrap();
            assert_eq!(rt.expert_slots(), slots, "{file}");
            let got = rt.trace_decode(&PROMPT[..6]).unwrap();
            assert_eq!(got.logits, want.logits, "{file} slots {slots} table {table}: decode logits");
            assert_eq!(rt.generate(&long, 16, Sampling::GREEDY).unwrap().tokens, want_tokens, "{file} slots {slots} table {table}");
            let pre = rt.prompt_logits(&long).unwrap();
            let worst = pre.iter().zip(&want_prefill).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(worst < 1e-3, "{file} slots {slots} table {table}: prefill differs by {worst}");
            let launches = rt.expert_table_launches();
            eprintln!("{file} slots {slots} table {table}: table launches {launches}, prefill worst {worst:.2e}, {:?}", rt.offload_stats());
            assert_eq!(launches > 0, table && slots == e * layers, "{file} slots {slots} table {table}");
        }
    }
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().copied().fold(f32::MIN, f32::max) as f64;
    let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
    row.iter().map(|&v| v as f64 - max - sum.ln()).collect()
}

fn kl(p: &[f32], q: &[f32]) -> f64 {
    let (p, q) = (log_softmax(p), log_softmax(q));
    p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum()
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn quantised_experts_match_the_bf16_expanded_experts_within_the_rounding_of_bf16() {
    for file in FILES {
        let run = |s: &Spec| {
            let mut rt = Runtime::load(s, 512).unwrap();
            let logits = rt.prompt_logits(&PROMPT).unwrap();
            let greedy = rt.generate(&PROMPT, 32, Sampling::GREEDY).unwrap().tokens;
            let long: Vec<u32> = (0..256u32).map(|i| (i * 37 + 11) % 1000).collect();
            let (mut pre, mut dec) = (f64::MAX, f64::MAX);
            for _ in 0..5 {
                let g = rt.generate(&long, 128, Sampling::GREEDY).unwrap();
                pre = pre.min(g.prefill_secs);
                dec = dec.min(g.decode_secs);
            }
            eprintln!("{file} quantized_experts={}: prefill {:.0} tok/s decode {:.0} tok/s", s.moe.as_ref().unwrap().quantized_experts, long.len() as f64 / pre, 127.0 / dec);
            (logits, greedy, rt.weight_bytes)
        };
        let ((a, ga, wa), (b, gb, wb)) = (run(&spec(file, false)), run(&spec(file, true)));
        let vocab = a.len() / PROMPT.len();
        let kls: Vec<f64> = (0..PROMPT.len()).map(|r| kl(&a[r * vocab..(r + 1) * vocab], &b[r * vocab..(r + 1) * vocab])).collect();
        let (mean, max) = (kls.iter().sum::<f64>() / kls.len() as f64, kls.iter().copied().fold(0.0, f64::max));
        let same = ga.iter().zip(&gb).take_while(|(x, y)| x == y).count();
        eprintln!("{file}: weights {wa} bytes expanded, {wb} bytes kept; mean KL {mean:.3e} max KL {max:.3e} greedy prefix {same}/32");
        assert!(wb < wa, "{file}");
        assert!(max < 1e-3, "{file}: max KL {max:.3e}");
    }
}

#[test]
fn a_quant_q8_spec_is_what_a_q8_moe_gguf_gives() {
    assert_eq!(Spec::from_gguf_file(&path(FILES[1])).unwrap().quant, Quant::Q8);
}
