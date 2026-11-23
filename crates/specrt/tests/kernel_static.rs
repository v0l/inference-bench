use kir::Target;
use serde_json::{Value, json};
use specrt::codegen::Module;
use specrt::gemm::WeightKind;
use specrt::spec::{KvDtype, Spec};
use std::path::{Path, PathBuf};

struct Gpu {
    name: &'static str,
    host: &'static str,
    target: Target,
}

fn gpus() -> Vec<Gpu> {
    vec![
        Gpu { name: "sm_120", host: "desktop RTX PRO 6000 Blackwell Max-Q", target: Target::cuda(12, 0, 188, 99 * 1024) },
        Gpu { name: "sm_121", host: "Spark GB10", target: Target::cuda(12, 1, 48, 99 * 1024) },
        Gpu { name: "sm_75", host: "Windows RTX 2060", target: Target::cuda(7, 5, 30, 64 * 1024) },
    ]
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn hf(name: &str, label: &str) -> Spec {
    let mut s = Spec::from_hf_config(&std::fs::read_to_string(fixture(name)).unwrap()).unwrap();
    s.name = label.into();
    s
}

fn with_attention(mut s: Spec, label: &str, heads: usize, kv_heads: usize, head_dim: usize) -> Spec {
    s.name = label.into();
    s.attention.heads = heads;
    s.attention.kv_heads = kv_heads;
    s.attention.head_dim = head_dim;
    s
}

fn specs() -> Vec<Spec> {
    let mut q27 = Spec::load(&fixture("qwen3.8-27b-nvfp4.toml")).unwrap();
    q27.name = "27b-nvfp4".into();
    let mut q27_fp8kv = q27.clone();
    q27_fp8kv.name = "27b-nvfp4-fp8kv".into();
    q27_fp8kv.runtime.kv_dtype = KvDtype::Fp8;
    let mut moe128 = hf("qwen3.5-122b-a10b-nvfp4.config.json", "122b-e128");
    moe128.moe.as_mut().unwrap().experts = 128;
    vec![
        q27.clone(),
        q27_fp8kv,
        hf("qwen3.5-4b-base.config.json", "4b-bf16"),
        hf("qwen3.5-122b-a10b-nvfp4.config.json", "122b-nvfp4"),
        moe128,
        with_attention(q27.clone(), "hd128-gqa8", 32, 4, 128),
        with_attention(q27, "hd128-gqa4", 32, 8, 128),
    ]
}

const DECODE_CTX: usize = 32768;

fn kv_planes(_spec: &Spec) -> usize {
    1
}

fn row(family: &str, spec: &Spec, kernel: &str, grid: [usize; 3], block: usize, smem: u32, params: Value) -> Value {
    json!({ "family": family, "spec": spec.name, "kernel": kernel, "grid": grid, "block": block, "smem": smem, "params": params })
}

fn shape(spec: &Spec) -> Value {
    let a = &spec.attention;
    json!({
        "hidden": spec.dims.hidden, "inter": spec.dims.intermediate, "heads": a.heads, "kv_heads": a.kv_heads,
        "head_dim": a.head_dim, "qkv_row": spec.qkv_rows(), "q_dim": spec.q_dim(), "kv_fp8": spec.runtime.kv_dtype.is_fp8(),
        "experts": spec.moe.as_ref().map_or(0, |m| m.experts), "top_k": spec.moe.as_ref().map_or(0, |m| m.top_k()),
    })
}

fn rows_for(spec: &Spec, module: &Module, t: &Target) -> Vec<Value> {
    let a = &spec.attention;
    let has = |k: &str| module.kernels.contains(&k);
    let mut out = Vec::new();
    let s = shape(spec);
    let with = |extra: Value| {
        let mut v = s.clone();
        for (k, x) in extra.as_object().unwrap() {
            v[k] = x.clone();
        }
        v
    };
    if has("rmsnorm") {
        for rows in [1usize, 4096] {
            out.push(row("rmsnorm", spec, "rmsnorm", [rows, 1, 1], 1024, 0, with(json!({ "rows": rows }))));
        }
    }
    if has("swiglu") {
        for rows in [1usize, 4096] {
            out.push(row("silu_and_mul", spec, "swiglu", [spec.dims.intermediate.div_ceil(256), rows, 1], 256, 0, with(json!({ "rows": rows }))));
        }
    }
    if has("qk_norm_rope") {
        for rows in [1usize, 4096] {
            out.push(row("rotary", spec, "qk_norm_rope", [a.heads + 2 * a.kv_heads, rows, 1], a.head_dim.max(32), 0, with(json!({ "rows": rows }))));
        }
    }
    if has("batch_decode_attention") {
        let splits = specrt::codegen::decode_splits(DECODE_CTX);
        let warps = specrt::codegen::decode_warps(a.heads / a.kv_heads, 1);
        for seqs in [1usize, 8] {
            for ctx in [1024usize, 8192, 32768] {
                let p = with(json!({ "seqs": seqs, "ctx": ctx, "splits": splits }));
                out.push(row("decode_attention", spec, "batch_decode_attention", [a.kv_heads, seqs, splits], 32 * warps, module.smem["batch_decode_attention"], p.clone()));
                out.push(row("decode_attention", spec, "batch_decode_combine", [a.heads, seqs, 1], a.head_dim / 2, 0, p));
            }
        }
    }
    if has("flash_mma") {
        let warps = specrt::codegen::flash_rows(a.head_dim, kv_planes(spec)) / 16;
        for tokens in [1024usize, 4096, 16384] {
            out.push(row("prefill_attention", spec, "flash_mma", [tokens.div_ceil(warps * 16), a.heads, 1], warps * 32, module.smem["flash_mma"], with(json!({ "tokens": tokens }))));
        }
    }
    let fp4 = spec.mlp.quant == Some(specrt::spec::Quant::Nvfp4G16);
    let weight = if fp4 { WeightKind::Fp4 } else { WeightKind::Bf16 };
    for (linear, tokens) in [("gate_up", 4096usize), ("down", 4096)] {
        let Ok(c) = specrt::gemm::select(spec, t, linear, tokens, weight) else { continue };
        if !has(c.name.as_str()) {
            continue;
        }
        let (n, k, _, _) = specrt::codegen::linear_dims(spec, linear);
        let family = if fp4 { "nvfp4_gemm" } else { "bf16_gemm" };
        let smem = module.smem.get(c.name.as_str()).copied().unwrap_or(0);
        out.push(row(family, spec, &c.name, [n.div_ceil(c.tile.bn), tokens.div_ceil(c.tile.bm), 1], c.tile.threads() as usize, smem, with(json!({ "m": tokens, "n": n, "k": k, "linear": linear, "path": format!("{:?}", c.path), "tile": [c.tile.bm, c.tile.bn, c.tile.wm, c.tile.wn] }))));
        if let Some(q) = c.quant.filter(|q| has(q)) {
            out.push(row(family, spec, q, [tokens, 1, 1], 256, 0, with(json!({ "m": tokens, "n": n, "k": k, "linear": linear, "role": "activation quant" }))));
        }
    }
    for (linear, normed) in [("gate_up", true), ("down", false), ("qkv", true)] {
        let Some(g) = specrt::gemm::gemv(spec, linear, weight, normed) else { continue };
        if !has(g.name) {
            continue;
        }
        let (n, k, _, _) = specrt::codegen::linear_dims(spec, linear);
        let family = if fp4 { "nvfp4_gemv" } else { "bf16_gemv" };
        out.push(row(family, spec, g.name, [g.rows.div_ceil(g.rows_per_block), 1, 1], g.threads as usize, 0, with(json!({ "m": 1, "n": n, "k": k, "linear": linear, "rows_per_block": g.rows_per_block, "normed": normed }))));
    }
    if let Some(m) = &spec.moe
        && has("moe_route")
    {
        for tokens in [1usize, 4096] {
            let l = kir::moe::route_logits_launch(m.experts, tokens);
            let p = with(json!({ "tokens": tokens }));
            out.push(row("topk_routing", spec, "moe_route_logits", [l.grid.0 as usize, l.grid.1 as usize, 1], l.threads as usize, 0, p.clone()));
            out.push(row("topk_routing", spec, "moe_route", [tokens, 1, 1], kir::moe::ROUTE_THREADS as usize, 0, p));
        }
    }
    out
}

fn extra_module(t: &Target) -> Result<kir::KirSource, String> {
    let mut ks = Vec::new();
    for hidden in [2560u32, 5120] {
        let mut k = kir::Kernel::add_rmsnorm(t, hidden, 1e-6, 0.0, 1024)?;
        k.name = if hidden == 2560 { "add_rmsnorm_2560" } else { "add_rmsnorm_5120" };
        ks.push(k);
    }
    kir::cuda::emit_module(&ks)
}

fn extra_rows() -> Vec<Value> {
    let mut out = Vec::new();
    for hidden in [2560usize, 5120] {
        for rows in [1usize, 4096] {
            out.push(json!({ "family": "fused_add_rmsnorm", "spec": format!("hidden-{hidden}"), "kernel": format!("add_rmsnorm_{hidden}"), "grid": [rows, 1, 1], "block": 1024, "smem": 0, "params": { "hidden": hidden, "rows": rows } }));
        }
    }
    out
}

fn sm_name(t: &Target, arch_features: bool) -> String {
    let (ma, mi) = t.compute_capability;
    let a = if arch_features && ma == 12 { "a" } else { "" };
    format!("sm_{ma}{mi}{a}")
}

fn compile(dir: &Path, stem: &str, source: &kir::KirSource, sm: &str, entry: &mut serde_json::Map<String, Value>) {
    std::fs::write(dir.join(format!("{stem}.cu")), source.as_str()).unwrap();
    match specrt::jit::cubin_for(source, sm, 8) {
        Ok(bytes) => {
            std::fs::write(dir.join(format!("{stem}.cubin")), bytes).unwrap();
            entry.insert("cubin".into(), json!(format!("{stem}.cubin")));
        }
        Err(e) => {
            entry.insert("error".into(), json!(format!("{e:#}").lines().filter(|l| l.contains("error")).take(6).collect::<Vec<_>>().join("\n")));
        }
    }
}

fn manifest_for(gpu: &Gpu, dir: Option<&Path>) -> Value {
    let t = &gpu.target;
    let mut modules = Vec::new();
    for spec in specs() {
        let mut entry = serde_json::Map::new();
        entry.insert("spec".into(), json!(spec.name));
        match Module::generate_kv(&spec, DECODE_CTX, kv_planes(&spec), t) {
            Ok(m) => {
                let sm = sm_name(t, m.arch_features);
                entry.insert("sm".into(), json!(sm));
                entry.insert("kir_hash".into(), json!(m.kir_hash));
                entry.insert("rows".into(), Value::Array(rows_for(&spec, &m, t)));
                if let Some(d) = dir {
                    compile(d, &spec.name, &m.source, &sm, &mut entry);
                }
            }
            Err(e) => {
                entry.insert("error".into(), json!(format!("{e:#}")));
                entry.insert("rows".into(), json!([]));
            }
        }
        modules.push(Value::Object(entry));
    }
    let mut entry = serde_json::Map::new();
    entry.insert("spec".into(), json!("add-rmsnorm"));
    match extra_module(t) {
        Ok(src) => {
            let sm = sm_name(t, false);
            entry.insert("sm".into(), json!(sm));
            entry.insert("rows".into(), Value::Array(extra_rows()));
            if let Some(d) = dir {
                compile(d, "add-rmsnorm", &src, &sm, &mut entry);
            }
        }
        Err(e) => {
            entry.insert("error".into(), json!(e));
            entry.insert("rows".into(), json!([]));
        }
    }
    modules.push(Value::Object(entry));
    json!({ "gpu": gpu.name, "host": gpu.host, "target": t.describe(), "sms": t.sm_count, "smem_limit": t.smem_limit, "modules": modules })
}

#[test]
fn every_static_row_names_a_kernel_its_module_builds() {
    for gpu in gpus() {
        let m = manifest_for(&gpu, None);
        let mut families: Vec<String> = Vec::new();
        for module in m["modules"].as_array().unwrap() {
            for r in module["rows"].as_array().unwrap() {
                assert!(r["block"].as_u64().unwrap() > 0 && r["block"].as_u64().unwrap() <= 1024, "{} {}: block {}", gpu.name, r["kernel"], r["block"]);
                assert!(r["smem"].as_u64().unwrap() <= gpu.target.smem_limit as u64, "{} {}: smem over the target", gpu.name, r["kernel"]);
                assert!(r["grid"].as_array().unwrap().iter().all(|g| g.as_u64().unwrap() > 0), "{} {}: empty grid", gpu.name, r["kernel"]);
                let f = r["family"].as_str().unwrap().to_string();
                if !families.contains(&f) {
                    families.push(f);
                }
            }
        }
        families.sort();
        if gpu.target.compute_capability.0 == 12 {
            for f in ["bf16_gemm", "decode_attention", "fused_add_rmsnorm", "nvfp4_gemm", "nvfp4_gemv", "prefill_attention", "rmsnorm", "rotary", "silu_and_mul", "topk_routing"] {
                assert!(families.iter().any(|x| x == f), "{}: no {f} row in {families:?}", gpu.name);
            }
        }
    }
}

#[test]
#[ignore = "NVRTC compile of every static row's module for every target into KERNEL_STATIC_OUT; no GPU"]
fn dump_static_kernels() {
    let Some(out) = std::env::var_os("KERNEL_STATIC_OUT") else { return };
    let only = std::env::var("KERNEL_STATIC_GPUS").ok();
    let out = PathBuf::from(out);
    let (major, minor) = specrt::jit::nvrtc_version().unwrap();
    for gpu in gpus() {
        if only.as_deref().is_some_and(|o| !o.split(',').any(|g| g == gpu.name)) {
            continue;
        }
        let dir = out.join(gpu.name);
        std::fs::create_dir_all(&dir).unwrap();
        let mut m = manifest_for(&gpu, Some(&dir));
        m["nvrtc"] = json!(format!("{major}.{minor}"));
        m["options"] = json!(["--std=c++17", "--split-compile=8"]);
        std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&m).unwrap()).unwrap();
        eprintln!("{}: {} modules", gpu.name, m["modules"].as_array().unwrap().len());
    }
}
