use specrt::plan::{Buf, NormKind, Op, Phase, Plan, QuantSource, Weight};
use specrt::spec::{Block, Spec};
use std::path::Path;

struct Ref {
    head: Vec<Op>,
    attention: Vec<Op>,
    linear: Vec<Op>,
    attention_moe: Vec<Op>,
    linear_moe: Vec<Op>,
    tail: Vec<Op>,
}

    fn quant_of(weight: Weight) -> QuantSource {
        match weight {
            Weight::GateUp | Weight::Down => QuantSource::Mlp,
            Weight::LinBa => QuantSource::Bf16,
            Weight::LmHead => QuantSource::Head,
            _ => QuantSource::Attn,
        }
    }

    fn reference(spec: &Spec, phase: Phase, tensor_core: bool) -> Ref {
        let norm = |which, into| Op::Norm { which, into };
        let lin = |weight, input, output| Op::Linear { weight, quant: quant_of(weight), input, output };
        if !tensor_core {
            return Ref {
                    head: vec![Op::Embed],
                attention: vec![
                    norm(NormKind::Input, Buf::Xn),
                    lin(Weight::Qkv, Buf::Xn, Buf::Qkv),
                    Op::QkRope,
                    Op::FlashLegacy,
                    lin(Weight::O, Buf::Attn, Buf::X),
                    norm(NormKind::PostAttention, Buf::Xn),
                    lin(Weight::GateUp, Buf::Xn, Buf::GateUp),
                    Op::Swiglu,
                    lin(Weight::Down, Buf::Act, Buf::X),
                ],
                linear: Vec::new(),
                attention_moe: Vec::new(),
                linear_moe: Vec::new(),
                tail: vec![norm(NormKind::Final, Buf::Xn), lin(Weight::LmHead, Buf::Xn, Buf::Logits)],
            };
        }
        let mlp = [
            norm(NormKind::PostAttention, Buf::XnPlanes),
            lin(Weight::GateUp, Buf::XnPlanes, Buf::ActPlanes),
            lin(Weight::Down, Buf::ActPlanes, Buf::X),
        ];
        let prefill = phase == Phase::Prefill;
        let mut attention = vec![norm(NormKind::Input, Buf::XnPlanes), lin(Weight::Qkv, Buf::XnPlanes, Buf::Qkv)];
        if prefill {
            attention.extend([Op::KvRope, Op::Flash]);
        } else {
            attention.push(Op::DecodeAttention);
        }
        attention.push(lin(Weight::O, Buf::AttnPlanes, Buf::X));
        attention.extend(mlp);
        let mut linear = vec![
            norm(NormKind::Input, Buf::XnPlanes),
            lin(Weight::LinIn, Buf::XnPlanes, Buf::PreLin),
            lin(Weight::LinBa, Buf::XnPlanes, Buf::PreBa),
            if prefill { Op::GdnPrefill } else { Op::GdnDecode },
            lin(Weight::LinOut, Buf::GdnPlanes, Buf::X),
        ];
        linear.extend(mlp);
        let moe = [Op::MoeRoute, Op::MoeExperts, Op::MoeCombine];
        let swap = |ops: &[Op]| -> Vec<Op> { ops[..ops.len() - mlp.len()].iter().copied().chain(moe).collect() };
        let (attention_moe, linear_moe) = if spec.moe.is_some() { (swap(&attention), swap(&linear)) } else { (Vec::new(), Vec::new()) };
        Ref {
            head: vec![Op::Embed],
            attention,
            linear: if spec.linear_attention.is_some() { linear } else { Vec::new() },
            attention_moe,
            linear_moe: if spec.linear_attention.is_some() { linear_moe } else { Vec::new() },
            tail: vec![norm(NormKind::Final, Buf::XnPlanes), lin(Weight::LmHead, Buf::XnPlanes, Buf::Logits)],
        }
    }


fn configs() -> Vec<std::path::PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut out = Vec::new();
    for e in std::fs::read_dir(&root).unwrap().flatten() {
        let p = e.path();
        if p.file_name().unwrap().to_string_lossy().ends_with(".config.json") {
            out.push(p);
        } else if p.is_dir() && p.join("config.json").exists() {
            out.push(p.join("config.json"));
        }
    }
    out.sort();
    out
}

#[test]
fn the_graph_lowers_to_the_reference_plan_for_every_config() {
    let mut checked = 0;
    for path in configs() {
        let json = std::fs::read_to_string(&path).unwrap();
        let Ok(spec) = Spec::from_hf_config_serving(&json).or_else(|_| Spec::describe_hf_config(&json)) else { continue };
        if spec.norm.sandwich {
            sandwich_plan_is_well_formed(&spec);
            checked += 1;
            continue;
        }
        for tc in [true, false] {
            for phase in [Phase::Prefill, Phase::Decode] {
                let mut want = reference(&spec, phase, tc);
                if tc && spec.dims.logit_softcap.is_some() {
                    want.tail.push(Op::LogitSoftcap);
                }
                let got = Plan::build(&spec, phase, tc);
                let what = format!("{} tc={tc} {phase:?}", path.display());
                assert_eq!(got.head, want.head, "{what}");
                assert_eq!(got.tail, want.tail, "{what}");
                for (block, sparse, ops) in [
                    (Block::Full, false, &want.attention),
                    (Block::Dense, false, &want.attention),
                    (Block::Sliding, false, &want.attention),
                    (Block::Linear, false, &want.linear),
                    (Block::Full, true, &want.attention_moe),
                    (Block::Linear, true, &want.linear_moe),
                ] {
                    if ops.is_empty() {
                        continue;
                    }
                    assert_eq!(got.layer_of(block, sparse), ops.as_slice(), "{what} {block:?} sparse={sparse}");
                }
            }
        }
        checked += 1;
    }
    assert!(checked >= 8, "only {checked} configs were checked");
}

#[test]
fn plan_rs_has_no_architecture_or_op_sequence_knowledge() {
    let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/plan.rs")).unwrap();
    let body = src.split("#[cfg(test)]").next().unwrap();
    for banned in ["is_linear", "Block::Linear", "Block::Full", "Block::Dense", "spec.moe", "tensor_core {", "if prefill", "qwen", "llama", "gemma"] {
        assert!(!body.contains(banned), "plan.rs mentions {banned}");
    }
}

#[test]
fn runtime_and_kir_have_no_architecture_names() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = vec![root.join("src/runtime.rs")];
    for dir in [root.join("src/runtime"), root.join("../kir/src")] {
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            files.push(e.path());
        }
    }
    for f in files.into_iter().filter(|f| f.extension().is_some_and(|e| e == "rs")) {
        let text = std::fs::read_to_string(&f).unwrap().to_lowercase();
        for banned in ["qwen", "llama", "gemma", "mistral", "deepseek"] {
            assert!(!text.contains(banned), "{} mentions {banned}", f.display());
        }
    }
}

fn sandwich_plan_is_well_formed(spec: &Spec) {
    for phase in [Phase::Prefill, Phase::Decode] {
        let plan = Plan::build(spec, phase, true);
        let ops = plan.layer(Block::Full);
        let adds: Vec<_> = ops.iter().filter(|o| matches!(o, Op::NormAdd { .. })).collect();
        assert_eq!(adds.len(), 2, "{phase:?}");
        assert!(ops.contains(&Op::Norm { which: NormKind::PreMlp, into: Buf::XnPlanes }));
        assert_eq!(ops.iter().filter(|o| matches!(o, Op::Clear { .. })).count(), 2);
        assert_eq!(plan.layer(Block::Sliding), ops);
        assert_eq!(plan.tail.contains(&Op::LogitSoftcap), spec.dims.logit_softcap.is_some());
    }
}
