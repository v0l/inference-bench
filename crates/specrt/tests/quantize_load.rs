use specrt::quantize::{fp8_channel_search, nvfp4_search};
use specrt::spec::{Quant, QuantizeChoice, Spec};
use specrt::weights::{Checkpoint, Matrix, Mixer, interleave_rows};
use std::path::{Path, PathBuf};

fn fixture_dir(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir)
}

fn spec_for(dir: &str, quantize: QuantizeChoice, keep_edges: usize) -> Spec {
    let json = std::fs::read_to_string(fixture_dir(dir).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = dir.into();
    spec.repo = fixture_dir(dir).display().to_string();
    spec.runtime.quantize = quantize;
    spec.quant_keep.first_layers = keep_edges;
    spec.quant_keep.last_layers = keep_edges;
    spec.apply_quantize().unwrap();
    spec
}

fn checkpoint(spec: &Spec) -> Checkpoint {
    Checkpoint::for_repo(&spec.repo).unwrap().renamed(&spec.tensors)
}

fn bf16_of(ck: &Checkpoint, name: &str, rows: usize, cols: usize) -> Vec<u16> {
    ck.bf16(name, &[rows, cols]).unwrap().into_owned()
}

fn assert_fp4(m: &Matrix, w: &[u16], cols: usize, what: &str) {
    let Matrix::Fp4(got) = m else { panic!("{what} is not nvfp4") };
    let want = nvfp4_search(w, cols);
    assert_eq!(got.packed.as_ref(), want.packed.as_ref(), "{what} packed");
    assert_eq!(got.scales.as_ref(), want.scales.as_ref(), "{what} scales");
    assert_eq!(got.global, want.global, "{what} global");
}

#[test]
fn nvfp4_quantize_streams_each_linear_through_the_scale_search_and_keeps_the_rest_bf16() {
    let spec = spec_for("tiny-qwen3.5", QuantizeChoice::Nvfp4G16, 0);
    let ck = checkpoint(&spec);
    let (h, inter) = (spec.dims.hidden, spec.dims.intermediate);
    let mut seen = 0;
    for i in 0..spec.layers.count {
        let p = format!("model.layers.{i}");
        let layer = ck.layer(&spec, i).unwrap();
        let gate = bf16_of(&ck, &format!("{p}.mlp.gate_proj.weight"), inter, h);
        let up = bf16_of(&ck, &format!("{p}.mlp.up_proj.weight"), inter, h);
        assert_fp4(&layer.gate_up, &interleave_rows(&gate, &up, h), h, "gate_up");
        assert_fp4(&layer.down, &bf16_of(&ck, &format!("{p}.mlp.down_proj.weight"), h, inter), inter, "down");
        match &layer.mixer {
            Mixer::Attention(a) => {
                assert_fp4(&a.o, &bf16_of(&ck, &format!("{p}.self_attn.o_proj.weight"), h, spec.o_dim()), spec.o_dim(), "o");
                assert!(matches!(a.qkv, Matrix::Fp4(_)));
            }
            Mixer::Linear(g) => {
                let value = spec.linear_attention.as_ref().unwrap().value_dim();
                assert_fp4(&g.out, &bf16_of(&ck, &format!("{p}.linear_attn.out_proj.weight"), h, value), value, "lin_out");
                assert!(matches!(g.in_proj, Matrix::Fp4(_)));
                assert!(matches!(g.ba, Matrix::Bf16(_)), "the delta-net gates stay bf16");
            }
            _ => panic!("layer {i} has no mixer"),
        }
        seen += 1;
    }
    assert_eq!(seen, spec.layers.count);
    let head = ck.outside(&spec).unwrap();
    assert!(matches!(head.lm_head, Matrix::Bf16(_)), "the output head stays bf16");
    let q = ck.quantized();
    assert_eq!(q.tensors, 4 * spec.layers.count as u64);
    assert!(q.bf16_bytes > 0 && q.secs > 0.0);
}

#[test]
fn the_keep_list_holds_the_edge_layers_and_named_tensors_in_bf16() {
    let spec = spec_for("tiny-qwen3.5", QuantizeChoice::Nvfp4G16, 1);
    let ck = checkpoint(&spec);
    let formats: Vec<bool> = (0..spec.layers.count).map(|i| matches!(ck.layer(&spec, i).unwrap().gate_up, Matrix::Fp4(_))).collect();
    assert_eq!(formats, [false, true, false]);
    assert_eq!(spec.served_mlp_quant(0), Quant::Bf16);
    assert_eq!(spec.served_mlp_quant(1), Quant::Nvfp4G16);
    let mut marked = spec_for("tiny-qwen3.5", QuantizeChoice::Nvfp4G16, 0);
    marked.quant_keep.sensitive.push("self_attn.o_proj".into());
    let ck = checkpoint(&marked);
    let Mixer::Attention(a) = ck.layer(&marked, 2).unwrap().mixer else { panic!("layer 2 is attention") };
    assert!(matches!(a.o, Matrix::Bf16(_)), "a sensitive tensor stays bf16");
    assert!(matches!(a.qkv, Matrix::Fp4(_)));
    assert_eq!(marked.linear_quant("o", 2), Quant::Bf16);
    assert_eq!(marked.linear_quant("qkv", 2), Quant::Nvfp4G16);
}

#[test]
fn fp8_quantize_serves_the_checkpoint_as_fp8_channel_with_searched_row_scales() {
    let spec = spec_for("tiny-qwen3.5", QuantizeChoice::Fp8Channel, 2);
    assert_eq!(spec.quant, Quant::Fp8Channel);
    let ck = checkpoint(&spec);
    let (h, inter) = (spec.dims.hidden, spec.dims.intermediate);
    let layer = ck.layer(&spec, 1).unwrap();
    let Matrix::Fp8 { values, scales } = &layer.down else { panic!("down is not fp8") };
    let (want_values, want_scales) = fp8_channel_search(&bf16_of(&ck, "model.layers.1.mlp.down_proj.weight", h, inter), inter);
    assert_eq!(values.as_ref(), want_values.as_slice());
    assert_eq!(scales, &want_scales);
}

#[test]
fn quantize_off_leaves_the_spec_and_every_matrix_as_it_was() {
    let plain = spec_for("tiny-qwen3.5", QuantizeChoice::Off, 2);
    let json = std::fs::read_to_string(fixture_dir("tiny-qwen3.5").join("config.json")).unwrap();
    let mut untouched = Spec::from_hf_config(&json).unwrap();
    untouched.name = plain.name.clone();
    untouched.repo = plain.repo.clone();
    assert_eq!(plain, untouched);
    assert!(plain.requanted_linears().is_empty());
    let ck = checkpoint(&plain);
    for i in 0..plain.layers.count {
        assert!(matches!(ck.layer(&plain, i).unwrap().gate_up, Matrix::Bf16(_)));
        for linear in ["qkv", "o", "lin_in", "lin_out", "gate_up", "down"] {
            assert_eq!(plain.linear_quant(linear, i), Quant::Bf16);
        }
    }
    assert_eq!(ck.quantized().tensors, 0);
}

#[test]
fn nvfp4_quantize_reaches_mixture_of_experts_stacks_but_not_their_routers() {
    let spec = spec_for("tiny-qwen3.5-moe", QuantizeChoice::Nvfp4G16, 0);
    let ck = checkpoint(&spec);
    let layer = ck.layer(&spec, 1).unwrap();
    let moe = layer.moe.expect("layer 1 has experts");
    assert!(matches!(moe.gate, specrt::weights::Stacked::Fp4 { .. }));
    assert_eq!(moe.router.len(), spec.moe.as_ref().unwrap().experts * spec.dims.hidden);
    assert_eq!(spec.expert_quant(1), Quant::Nvfp4G16);
    assert!(spec_for("tiny-qwen3.5-moe", QuantizeChoice::Off, 0).expert_quant(1) == Quant::Bf16);
}

#[test]
fn a_stage_keeps_the_keep_list_in_whole_model_layers() {
    let json = std::fs::read_to_string(fixture_dir("qwen3-0.6b.config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.runtime.quantize = QuantizeChoice::Nvfp4G16;
    let stage = spec.stage(14, 28);
    assert_eq!(stage.linear_quant("qkv", 0), Quant::Nvfp4G16, "whole model layer 14");
    assert_eq!(stage.linear_quant("qkv", 11), Quant::Nvfp4G16, "whole model layer 25");
    assert_eq!(stage.linear_quant("qkv", 12), Quant::Bf16, "whole model layer 26");
    let front = spec.stage(0, 14);
    assert_eq!(front.linear_quant("qkv", 1), Quant::Bf16);
    assert_eq!(front.linear_quant("qkv", 13), Quant::Nvfp4G16);
}
