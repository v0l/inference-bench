use kir::Target;
use kir::desc::{Description, Shape};
use kir::int::{IntExpr, floor_rem, prime_after, prime_sum, splitmix64};
use std::path::Path;

fn specs() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../specrt/specs"))
}

fn desc(name: &str) -> Description {
    Description::parse(&std::fs::read_to_string(specs().join(format!("{name}.toml"))).unwrap()).unwrap()
}

fn cases() -> Vec<serde_json::Value> {
    let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../specrt/tests/fixtures/tiny-qwen4-exp/reference/tiny-qwen4-exp.ngram.json"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn shape_of(case: &serde_json::Value) -> Shape {
    let p = &case["params"];
    let n = |k: &str| p[k].as_u64().unwrap() as f64;
    let heads = (n("ngram_size") - 1.0) * n("heads_per_ngram");
    Shape::new()
        .num("ngram_size", n("ngram_size"))
        .num("ngram_heads", heads)
        .num("vocab", n("vocab_size"))
        .num("ngram_base", n("ngram_vocab_size_base"))
        .num("ngram_seed", n("seed"))
        .num("ngram_eos", n("eos_token_id"))
        .num("ple_index", case["ple_index"].as_u64().unwrap() as f64)
        .num("ple_layers", 1.0)
}

#[test]
fn integer_expressions_wrap_like_int64_and_fold_build_time_calls() {
    let e = IntExpr::parse("(a * 0x9E3779B97F4A7C15) ^ (b << 3)").unwrap();
    let got = e.eval(&|n| match n { "a" => Some(7), "b" => Some(5), _ => None }).unwrap();
    assert_eq!(got, 7u64.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 40);
    assert_eq!(floor_rem((-7i64) as u64, 5).unwrap(), 3);
    assert_eq!(floor_rem(i64::MIN as u64, 1009).unwrap() as i64, i64::MIN.rem_euclid(1009));
    assert_eq!(IntExpr::parse("rem(0 - 7, 5)").unwrap().eval(&|_| None).unwrap(), 3);
    assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
    assert_eq!((prime_after(999, 1).unwrap(), prime_after(999, 4).unwrap()), (1009, 1021));
    assert_eq!(prime_sum(999, 1, 3).unwrap(), 1009 + 1013 + 1019);
    assert_eq!(IntExpr::parse("splitmix64(x) + 1").unwrap().fold().unwrap(), IntExpr::parse("splitmix64(x) + 1").unwrap());
    assert_eq!(IntExpr::parse("prime_after(999, 2) + t0").unwrap().fold().unwrap(), IntExpr::parse("1013 + t0").unwrap());
    assert!(IntExpr::parse("splitmix64(t0)").unwrap().lower_cuda(&|n| Some(n.into())).is_err());
    assert!(IntExpr::parse("a ** b").is_err());
    assert!(IntExpr::parse("rem(a)").is_err());
}

#[test]
fn the_ngram_hash_description_matches_the_numpy_reference_bit_for_bit() {
    let d = desc("ple_ngram_hash");
    let all = cases();
    assert_eq!(all.len(), 5);
    for case in &all {
        let name = case["name"].as_str().unwrap();
        let h = d.token_hash(&shape_of(case)).unwrap_or_else(|e| panic!("{name}: {e}"));
        let sizes: Vec<u64> = case["sizes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let offsets: Vec<u64> = case["offsets"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert_eq!(h.per_head.iter().find(|(n, _)| n == "size").unwrap().1, sizes, "{name}");
        assert_eq!(h.per_head.iter().find(|(n, _)| n == "offset").unwrap().1, offsets, "{name}");
        let tokens: Vec<i64> = case["tokens"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        let want: Vec<i64> = case["ids"].as_array().unwrap().iter().flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_i64().unwrap())).collect();
        let eos = case["params"]["eos_token_id"].as_i64().unwrap();
        let mut history = vec![eos; 2];
        assert_eq!(h.eval(&tokens, &mut history).unwrap(), want, "{name} whole");
        let mut history = vec![eos; 2];
        let mut chunked = Vec::new();
        for part in tokens.chunks(7) {
            chunked.extend(h.eval(part, &mut history).unwrap());
        }
        assert_eq!(chunked, want, "{name} in chunks of 7");
        let mut history = vec![eos; 2];
        let mut stepped = Vec::new();
        for t in &tokens {
            stepped.extend(h.eval(std::slice::from_ref(t), &mut history).unwrap());
        }
        assert_eq!(stepped, want, "{name} one token at a time");
        let k = d.build(&Target::sm_120(), &shape_of(case)).unwrap();
        let src = kir::cuda::emit_module(std::slice::from_ref(&k)).unwrap().to_string();
        assert!(src.contains("ple_ngram_hash") && src.contains("unsigned long long") && src.contains("long long)tok[j]"), "{name}");
    }
}

#[test]
fn the_ngram_hash_refuses_an_ngram_size_it_does_not_spell() {
    let shape = shape_of(&cases()[0]).num("ngram_size", 4.0);
    assert!(desc("ple_ngram_hash").token_hash(&shape).unwrap_err().contains("division by zero"));
}

fn ple_shape() -> Shape {
    Shape::new()
        .num("hidden", 128.0)
        .num("streams", 4.0)
        .num("norm_eps", 1e-6)
        .num("ple_gate_scale", 1.0 / 128f64.sqrt())
        .num("ple_conv_taps", 4.0)
        .num("ngram_size", 3.0)
        .num("ple_layers", 1.0)
        .num("ngram_heads", 4.0)
        .num("ple_head_dim", 32.0)
        .num("ple_embed_dim", 128.0)
}

#[test]
fn the_ple_descriptions_build_and_emit() {
    let t = Target::sm_120();
    let mut kernels = Vec::new();
    for name in ["ple_embed_gather", "ple_key_proj", "ple_value_proj", "ple_gate", "ple_conv", "ple_add"] {
        kernels.push(desc(name).build(&t, &ple_shape()).unwrap_or_else(|e| panic!("{name}: {e}")));
    }
    let src = kir::cuda::emit_module(&kernels).unwrap().to_string();
    assert!(src.contains("kir_block_sum(float v"));
    assert!(src.contains("constexpr int DIL = 3;") && src.contains("win[j * DIL]"));
    assert!(src.contains("(size_t)idx[(size_t)row * 4 + grp] * 32"));
    assert!(desc("ple_gate").build(&Target::wgpu_baseline(), &ple_shape()).is_err());
}

#[test]
fn an_undilated_seq_conv_emits_the_same_text_as_before() {
    let text = std::fs::read_to_string(specs().join("mamba2_conv.toml")).unwrap();
    let shape = Shape::new().num("ssm_in", 300.0).num("ssm_inner", 64.0).num("ssm_conv", 96.0).num("ssm_taps", 4.0).num("linear_layers", 2.0);
    let plain = Description::parse(&text).unwrap().build(&Target::sm_120(), &shape).unwrap();
    let one = Description::parse(&text.replace("taps = \"ssm_taps\"", "taps = \"ssm_taps\"\ndilation = 1")).unwrap().build(&Target::sm_120(), &shape).unwrap();
    let a = kir::cuda::emit_module(std::slice::from_ref(&plain)).unwrap().to_string();
    assert_eq!(a, kir::cuda::emit_module(std::slice::from_ref(&one)).unwrap().to_string());
    assert!(a.contains("win[TAPS - 1] = x[") && !a.contains("DIL"));
}
