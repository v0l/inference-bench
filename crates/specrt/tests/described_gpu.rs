use specrt::family::Registry as Families;
use specrt::kir_kernel::Registry;
use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling, SpecRow};
use specrt::spec::{EMBEDDED, Quant, Spec};
use std::path::{Path, PathBuf};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn plain_scan() -> String {
    let base = [
        ("qkv", "f32", "activation", "read"),
        ("k_cache", "bf16", "activation", "read"),
        ("v_cache", "bf16", "activation", "read"),
        ("block_table", "i32", "activation", "read"),
        ("pos", "i32", "activation", "read"),
        ("slots", "i32", "activation", "read"),
        ("out_hi", "bf16", "activation", "write"),
        ("out_lo", "bf16", "activation", "write"),
        ("layer", "i32", "value", "read"),
        ("pages", "i32", "value", "read"),
        ("pos_step", "i32", "value", "read"),
        ("row_step", "i32", "value", "read"),
    ];
    let mut out = String::from("kind = \"kir-kernel\"\nfamily = \"scan_attention\"\nname = \"plain_scan\"\nthreads = 128\n\n");
    for (n, t, r, a) in base {
        out.push_str(&format!("[[param]]\nname = \"{n}\"\ntype = \"{t}\"\nrole = \"{r}\"\naccess = \"{a}\"\n\n"));
    }
    out.push_str("[[stage]]\nop = \"epilogue\"\nname = \"score\"\nexpr = \"s * scale\"\n\n[[stage]]\nop = \"online_softmax\"\nvalue = \"score\"\n");
    out
}

fn install() {
    let texts: Vec<String> = EMBEDDED.iter().map(|(_, t)| t.to_string()).chain([plain_scan()]).collect();
    Registry::install(Registry::from_texts(texts.iter().map(String::as_str)).unwrap());
}

const NODES: &str = r#"
[[attention.node]]
op = "qk_rope"
phase = "prefill"

[[attention.node]]
op = "kernel"
kernel = "plain_scan"
phase = "prefill"
grid = ["heads", "rows"]

[[attention.node]]
op = "decode_rope"
phase = "rows"

[[attention.node]]
op = "kernel"
kernel = "plain_scan"
phase = "rows"
grid = ["heads", "rows"]
"#;

fn mtp_spec(described: bool) -> Spec {
    let dir = root().join("tests/fixtures/tiny-qwen3.5-mtp");
    let mut s = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    s.name = "tiny-qwen3.5-mtp".into();
    s.repo = dir.display().to_string();
    s.runtime.split_activations = Some(true);
    s.mtp.as_mut().unwrap().draft_tokens = 2;
    s.mtp.as_mut().unwrap().graph = true;
    s.mtp.as_mut().unwrap().head_quant = Some(Quant::Bf16);
    if described {
        let text = format!("{}\n{NODES}", s.to_toml().unwrap());
        s = Spec::parse(&text).unwrap_or_else(|e| panic!("{e:#}"));
    }
    s
}

fn layout() -> Layout {
    let pages = 512usize.div_ceil(specrt::codegen::PAGE);
    Layout { max_ctx: 512, slots: 1, kv_blocks: pages, prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: false }
}

fn prompt(i: usize) -> Vec<u32> {
    let len = 5 + (i * 37) % 120;
    (0..len as u32).map(|j| (j * 131 + i as u32 * 977 + 7) % 1024).collect()
}

const NEW: usize = 48;

fn plain(s: &Spec, p: &[u32]) -> Vec<u32> {
    let mut plain = s.clone();
    plain.mtp.as_mut().unwrap().speculate = false;
    let mut rt = Runtime::load_layout(&plain, layout()).unwrap();
    let pages = 512usize.div_ceil(specrt::codegen::PAGE) as u32;
    rt.set_block_table(0, &(0..pages).collect::<Vec<_>>()).unwrap();
    let mut st = rt.stepper().unwrap();
    let mut out = vec![rt.prefill_slot(&mut st, 0, p, 0, Some((Sampling::GREEDY, 0))).unwrap().unwrap()];
    while out.len() < NEW {
        let row = DecodeRow { slot: 0, token: *out.last().unwrap(), pos: p.len() + out.len() - 1, sampling: Sampling::GREEDY, step: out.len() as u32 };
        out.push(rt.decode(&mut st, &[row]).unwrap()[0]);
    }
    out
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn mtp_with_described_attention_speculates_to_the_plain_greedy_tokens() {
    install();
    let s = mtp_spec(true);
    assert!(s.speculative().is_some() && !s.attention.node.is_empty());
    let stock = mtp_spec(false);
    for i in [0usize, 3, 7] {
        let p = prompt(i);
        let want = plain(&stock, &p);
        assert_eq!(plain(&s, &p), want, "prompt {i}: described attention changes the plain tokens");
        let mut rt = Runtime::load_layout(&s, layout()).unwrap();
        let pages = 512usize.div_ceil(specrt::codegen::PAGE) as u32;
        rt.set_block_table(0, &(0..pages).collect::<Vec<_>>()).unwrap();
        let mut st = rt.stepper().unwrap();
        let mut out = vec![rt.prefill_slot(&mut st, 0, &p, 0, Some((Sampling::GREEDY, 0))).unwrap().unwrap()];
        while out.len() < NEW {
            let row = SpecRow { slot: 0, token: *out.last().unwrap(), pos: p.len() + out.len() - 1, sampling: Sampling::GREEDY, step: out.len() as u32, max_accept: 2 };
            out.extend(rt.speculate(&[row]).unwrap().remove(0));
        }
        out.truncate(NEW);
        assert_eq!(out, want, "prompt {i}");
        let stats = rt.spec_stats();
        eprintln!("prompt {i}: {} steps, {} of {} drafts accepted", stats.steps, stats.accepted, stats.drafted);
        assert!(stats.drafted > 0);
        let [prefill, decode, scalar] = rt.scan_launches();
        assert!(prefill + decode + scalar > 0, "no described kernel ran");
    }
}

fn fixture(name: &str) -> PathBuf {
    root().join("tests/fixtures").join(name)
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_variants.py output in SPECRT_REF_DIR"]
fn the_single_stream_step_runs_described_attention_and_matches_transformers() {
    let name = "tiny-sinks";
    let json = std::fs::read_to_string(fixture(name).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config_serving_in(Families::embedded(), &json).unwrap();
    spec.name = name.into();
    spec.repo = fixture(name).display().to_string();
    spec.mlp.act_quant = specrt::spec::ActQuant::None;
    spec.mlp.up_act_planes = 1;
    spec.mlp.down_act_planes = 1;
    let dir = std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap_or_else(|| root().join("out"));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.ref.json"))).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let (tokens, greedy) = (ids("tokens"), ids("greedy"));
    let mut rt = Runtime::load_batched(&spec, 256, 1).unwrap();
    let got = rt.generate(&tokens, greedy.len(), Sampling::GREEDY).unwrap();
    assert_eq!(got.tokens, greedy);
    let [prefill, decode, scalar] = rt.scan_launches();
    assert!(prefill + decode + scalar > 0, "the step path ran no described kernel");
}
