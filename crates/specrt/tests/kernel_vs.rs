use kir::Target;
use serde_json::{Value, json};
use specrt::codegen::Module;
use specrt::gemm::WeightKind;
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

const MAX_CTX: usize = 32768;
const SMS: u32 = 188;

fn target() -> Target {
    Target::cuda(12, 0, SMS, 99 * 1024)
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn one_layer(mut s: Spec, label: &str) -> Spec {
    s.name = label.into();
    s.layers.count = s.layers.pattern.len();
    s.mtp = None;
    s
}

fn hf(name: &str, label: &str) -> Spec {
    one_layer(Spec::from_hf_config_serving(&std::fs::read_to_string(fixture(name)).unwrap()).unwrap(), label)
}

fn with_attention(mut s: Spec, label: &str, heads: usize, kv_heads: usize, head_dim: usize) -> Spec {
    s.name = label.into();
    s.attention.heads = heads;
    s.attention.kv_heads = kv_heads;
    s.attention.head_dim = head_dim;
    s
}

fn specs() -> Vec<Spec> {
    let q27 = one_layer(Spec::load(&fixture("qwen3.8-27b-nvfp4.toml")).unwrap(), "27b-nvfp4");
    vec![
        q27.clone(),
        with_attention(q27.clone(), "hd128-gqa8", 32, 4, 128),
        with_attention(q27, "hd128-gqa4", 32, 8, 128),
        hf("qwen3.5-122b-a10b-nvfp4.config.json", "122b-nvfp4"),
        hf("qwen3.5-4b-base.config.json", "4b-bf16"),
        hf("gpt-oss-20b.config.json", "gpt-oss-20b"),
    ]
}

fn launch(kernel: &str, grid: [usize; 3], block: usize, smem: u32) -> Value {
    json!({ "kernel": kernel, "grid": grid, "block": block, "smem": smem })
}

fn case(row: u32, name: String, module: &str, launches: Vec<Value>, params: Value) -> Value {
    json!({ "row": row, "name": name, "module": module, "launches": launches, "params": params })
}

fn cases_for(spec: &Spec, m: &Module, t: &Target) -> Vec<Value> {
    let a = &spec.attention;
    let has = |k: &str| m.kernels.contains(&k);
    let mut out = Vec::new();
    let s = spec.name.as_str();
    if s == "27b-nvfp4" || s.starts_with("hd128") {
        let splits = specrt::codegen::decode_splits(MAX_CTX);
        let warps = specrt::codegen::decode_warps(a.heads / a.kv_heads, 1);
        let row = if s == "27b-nvfp4" { 1 } else { 2 };
        for ctx in [1024usize, 8192, 32768] {
            let l = vec![
                launch("batch_decode_attention", [a.kv_heads, 8, splits], 32 * warps, m.smem["batch_decode_attention"]),
                launch("batch_decode_combine", [a.heads, 8, 1], a.head_dim / 2, 0),
            ];
            out.push(case(row, format!("{s} decode seqs=8 ctx={ctx}"), s, l, json!({ "kind": "batch_decode", "seqs": 8, "ctx": ctx, "pages": MAX_CTX / specrt::codegen::PAGE, "splits": splits })));
        }
    }
    if s == "122b-nvfp4" {
        for ctx in [1024usize, 8192, 32768] {
            let l = vec![
                launch("attention_split", [a.kv_heads, 64, 1], kir::attention::SPLIT_BLOCK as usize, 0),
                launch("attention_reduce", [a.heads, 1, 1], a.head_dim.max(32), 0),
            ];
            out.push(case(3, format!("{s} attention_split ctx={ctx}"), s, l, json!({ "kind": "split_decode", "seqs": 1, "ctx": ctx, "splits": 64 })));
        }
    }
    if s == "27b-nvfp4" {
        let rows = specrt::codegen::flash_rows(a.head_dim, 1);
        for tokens in [1024usize, 4096, 16384] {
            let l = vec![launch("flash_mma", [tokens.div_ceil(rows), a.heads, 1], rows * 2, m.smem["flash_mma"])];
            out.push(case(6, format!("{s} flash_mma tokens={tokens}"), s, l, json!({ "kind": "prefill", "tokens": tokens })));
        }
        for linear in ["gate_up", "down"] {
            let (n, k, _, _) = specrt::codegen::linear_dims(spec, linear);
            for tokens in [512usize, 2048, 4096] {
                let c = specrt::gemm::select(spec, t, linear, tokens, WeightKind::Fp4).unwrap();
                assert!(has(c.name.as_str()), "{s}: {} not in the module", c.name);
                let l = vec![launch(&c.name, [n.div_ceil(c.tile.bn), tokens.div_ceil(c.tile.bm), 1], c.tile.threads() as usize, m.smem[c.name.as_str()])];
                let p = json!({ "kind": "fp4_gemm", "linear": linear, "m": tokens, "n": n, "k": k, "tile": [c.tile.bm, c.tile.bn, c.tile.wm, c.tile.wn] });
                out.push(case(4, format!("{s} {} m={tokens}", c.name), s, l, p));
            }
        }
        for (linear, normed) in [("gate_up", true), ("down", false)] {
            let g = specrt::gemm::gemv(spec, linear, WeightKind::Fp4, normed).unwrap();
            let (n, k, _, _) = specrt::codegen::linear_dims(spec, linear);
            let l = vec![launch(g.name, [g.grid(SMS) as usize, 1, 1], g.threads as usize, 0)];
            out.push(case(8, format!("{s} {} m=1", g.name), s, l, json!({ "kind": "fp4_gemv", "linear": linear, "n": n, "k": k })));
        }
    }
    if s == "4b-bf16" || s == "gpt-oss-20b" {
        let (n, k, _, _) = specrt::codegen::linear_dims(spec, "gate_up");
        for tokens in [2048usize, 4096] {
            let c = specrt::gemm::select(spec, t, "gate_up", tokens, WeightKind::Bf16).unwrap();
            assert!(has(c.name.as_str()), "{s}: {} not in the module", c.name);
            let l = vec![launch(&c.name, [n.div_ceil(c.tile.bn), tokens.div_ceil(c.tile.bm), 1], c.tile.threads() as usize, m.smem[c.name.as_str()])];
            let p = json!({ "kind": "bf16_gemm", "linear": "gate_up", "m": tokens, "n": n, "k": k, "tile": [c.tile.bm, c.tile.bn, c.tile.wm, c.tile.wn] });
            out.push(case(5, format!("{s} {} m={tokens}", c.name), s, l, p));
        }
    }
    if (s == "27b-nvfp4" || s == "4b-bf16") && has("rmsnorm") {
        for rows in [1usize, 4096] {
            let l = vec![launch("rmsnorm", [rows, 1, 1], 1024, 0)];
            out.push(case(7, format!("rmsnorm hidden={} rows={rows}", spec.dims.hidden), s, l, json!({ "kind": "rmsnorm", "rows": rows, "hidden": spec.dims.hidden })));
        }
    }
    out
}

fn add_rmsnorm_module(t: &Target) -> kir::KirSource {
    let mut ks = Vec::new();
    for hidden in [2560u32, 5120] {
        let mut k = kir::Kernel::add_rmsnorm(t, hidden, 1e-6, 0.0, 1024).unwrap();
        k.name = if hidden == 2560 { "add_rmsnorm_2560" } else { "add_rmsnorm_5120" };
        ks.push(k);
    }
    kir::cuda::emit_module(&ks).unwrap()
}

fn add_rmsnorm_cases() -> Vec<Value> {
    let mut out = Vec::new();
    for hidden in [2560usize, 5120] {
        for rows in [1usize, 4096] {
            let l = vec![launch(&format!("add_rmsnorm_{hidden}"), [rows, 1, 1], 1024, 0)];
            out.push(case(7, format!("add_rmsnorm hidden={hidden} rows={rows}"), "add-rmsnorm", l, json!({ "kind": "add_rmsnorm", "rows": rows, "hidden": hidden })));
        }
    }
    out
}

fn sm_name(arch_features: bool) -> &'static str {
    if arch_features { "sm_120a" } else { "sm_120" }
}

fn write_module(dir: &Path, stem: &str, source: &kir::KirSource, sm: &str, jobs: usize) -> Value {
    std::fs::write(dir.join(format!("{stem}.cu")), source.as_str()).unwrap();
    let bytes = specrt::jit::cubin_for(source, sm, jobs).unwrap_or_else(|e| panic!("{stem} {sm}: {e:#}"));
    std::fs::write(dir.join(format!("{stem}.cubin")), bytes).unwrap();
    json!({ "source": format!("{stem}.cu"), "cubin": format!("{stem}.cubin"), "sm": sm })
}

fn manifest(dir: Option<&Path>, jobs: usize) -> Value {
    let t = target();
    let mut cases = Vec::new();
    let mut modules = serde_json::Map::new();
    for spec in specs() {
        let m = Module::generate_kv(&spec, MAX_CTX, 1, &t).unwrap_or_else(|e| panic!("{}: {e:#}", spec.name));
        let rows = cases_for(&spec, &m, &t);
        if rows.iter().any(|r| r["params"]["kind"].as_str().unwrap().contains("decode") || r["params"]["kind"] == "prefill") {
            assert_eq!(spec.kv_layers(), 1, "{}: attention cases need one KV layer", spec.name);
        }
        if let Some(d) = dir {
            modules.insert(spec.name.clone(), write_module(d, &spec.name, &m.source, sm_name(m.arch_features), jobs));
        }
        cases.extend(rows);
    }
    if let Some(d) = dir {
        modules.insert("add-rmsnorm".into(), write_module(d, "add-rmsnorm", &add_rmsnorm_module(&t), "sm_120", jobs));
    }
    cases.extend(add_rmsnorm_cases());
    json!({ "target": t.describe(), "sms": SMS, "max_ctx": MAX_CTX, "compile_jobs": jobs, "modules": modules, "cases": cases })
}

#[test]
fn every_vs_case_launches_a_kernel_its_module_builds_and_covers_the_eight_rows() {
    let m = manifest(None, 1);
    let cases = m["cases"].as_array().unwrap();
    let mut rows: Vec<u64> = cases.iter().map(|c| c["row"].as_u64().unwrap()).collect();
    rows.sort();
    rows.dedup();
    assert_eq!(rows, (1..=8).collect::<Vec<_>>());
    for c in cases {
        for l in c["launches"].as_array().unwrap() {
            let block = l["block"].as_u64().unwrap();
            assert!(block > 0 && block <= 1024, "{}: block {block}", c["name"]);
            assert!(l["smem"].as_u64().unwrap() <= 99 * 1024, "{}: smem", c["name"]);
            assert!(l["grid"].as_array().unwrap().iter().all(|g| g.as_u64().unwrap() > 0), "{}: empty grid", c["name"]);
        }
    }
}

#[test]
#[ignore = "NVRTC compile of the timed kernels into KERNEL_VS_OUT; no GPU"]
fn dump_vs_kernels() {
    let Some(out) = std::env::var_os("KERNEL_VS_OUT") else { return };
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    let jobs = specrt::jit::default_compile_jobs();
    let mut m = manifest(Some(&out), jobs);
    let (major, minor) = specrt::jit::nvrtc_version().unwrap();
    m["nvrtc"] = json!(format!("{major}.{minor}"));
    std::fs::write(out.join("manifest.json"), serde_json::to_string_pretty(&m).unwrap()).unwrap();
    eprintln!("{} cases", m["cases"].as_array().unwrap().len());
}
