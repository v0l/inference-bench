use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use specrt::codegen::Module;
use specrt::spec::{ActQuant, Quant, Spec};
use std::path::Path;

fn spec(fixture: &str, quant: Quant, act: ActQuant, sigmoid: bool) -> Spec {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(fixture);
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut s = Spec::from_hf_config(&json).unwrap();
    s.vision = None;
    s.mlp.quant = Some(quant);
    s.mlp.act_quant = act;
    if sigmoid {
        s.moe.as_mut().unwrap().router = specrt::spec::RouterAct::Sigmoid;
        s.moe.as_mut().unwrap().shared_gate = true;
    }
    s
}

fn variants() -> Vec<(&'static str, Spec)> {
    vec![
        ("bf16", spec("tiny-qwen3.5-moe", Quant::Bf16, ActQuant::None, false)),
        ("bf16_sigmoid", spec("tiny-qwen3.5-moe", Quant::Bf16, ActQuant::None, true)),
        ("fp8", spec("tiny-qwen3.5-moe-fp8", Quant::Fp8Block128, ActQuant::None, false)),
        ("fp4", spec("tiny-qwen3.5-moe", Quant::Nvfp4G16, ActQuant::None, false)),
        ("fp4_mma", spec("tiny-qwen3.5-moe", Quant::Nvfp4G16, ActQuant::Fp4, false)),
    ]
}

#[test]
#[ignore = "writes PTX for every MoE variant into MOE_PTX_OUT; needs nvrtc, no GPU"]
fn dump_moe_ptx() {
    let out = std::env::var("MOE_PTX_OUT").expect("MOE_PTX_OUT");
    std::fs::create_dir_all(&out).unwrap();
    for (cc, arch) in [((12, 0), "compute_120a"), ((12, 1), "compute_121a")] {
        for (name, s) in variants() {
            let target = kir::Target::cuda(cc.0, cc.1, 188, 99 * 1024);
            let m = Module::generate_kv(&s, 256, 2, &target).unwrap();
            let opts = CompileOptions { arch: Some(arch), options: vec!["--std=c++17".to_string()], ..Default::default() };
            let ptx = compile_ptx_with_opts(&m.source, opts).unwrap().to_src();
            std::fs::write(Path::new(&out).join(format!("{name}_{arch}.ptx")), ptx).unwrap();
        }
    }
}
