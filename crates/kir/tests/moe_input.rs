use kir::moe::{Family, MoeKernel, MoeShape};
use kir::Target;

fn shape(input_norm: bool) -> MoeShape {
    MoeShape {
        hidden: 128,
        experts: 8,
        local: 8,
        top_k: 2,
        pairs: 3,
        inter: 32,
        shared: 1,
        norm_topk: true,
        routed_scale: 1.0,
        out_scale: 1.0,
        shared_gate: true,
        sigmoid: false,
        groups: 0,
        topk_group: 0,
        score_bias: false,
        router_bias: false,
        expert_bias: false,
        hash: false,
        block_group: 0,
        mapped: false,
        input_scaled: false,
        input_norm,
        split_input: false,
        expert_scale: false,
    }
}

fn source(input_norm: bool) -> String {
    let k = MoeKernel::new(&Target::sm_120(), Family::Experts, shape(input_norm)).unwrap();
    kir::cuda::emit_moe(&[k]).unwrap().into_string()
}

#[test]
fn the_route_norms_its_input_only_when_the_shape_asks() {
    let normed = source(true);
    assert!(normed.contains("constexpr bool MOE_INPUT_NORM = true;"), "{normed}");
    let raw = source(false);
    assert!(raw.contains("constexpr bool MOE_INPUT_NORM = false;"), "{raw}");
    assert!(raw.contains("float inv = MOE_INPUT_NORM ? rsqrtf(total / HIDDEN + EPS) : 1.0f;"), "{raw}");
    assert!(raw.contains("float v = MOE_INPUT_NORM ? xr[i] * inv * ln[i] : xr[i];"), "{raw}");
    assert_ne!(MoeKernel::new(&Target::sm_120(), Family::Experts, shape(true)).unwrap().hash(), MoeKernel::new(&Target::sm_120(), Family::Experts, shape(false)).unwrap().hash());
}
