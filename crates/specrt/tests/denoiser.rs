use specrt::denoiser::{self, DenoiserGraph, DenoiserSpec};
use specrt::denoiser_host::{HostDenoiser, Inputs, Trace, grid_ids, timestep_embedding};
use specrt::plan::{DitOp, Op};
use specrt::weights::Checkpoint;
use std::path::{Path, PathBuf};

const VELOCITY_TOLERANCE: f32 = 1e-4;
const HIDDEN_TOLERANCE: f32 = 1e-4;
const BF16_WEIGHT_BOUND: f32 = 5e-3;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-flux2-klein")
}

fn json(file: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixture().join("reference").join(file)).unwrap()).unwrap()
}

fn floats_in(index: &serde_json::Value, key: &str) -> Vec<f32> {
    let file = index["files"][key]["file"].as_str().unwrap_or_else(|| panic!("no reference {key}"));
    std::fs::read(fixture().join("reference").join(file)).unwrap().chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn pipeline(key: &str) -> Vec<f32> {
    floats_in(&json("tiny-flux2-klein.json"), key)
}

fn reference(key: &str) -> Vec<f32> {
    floats_in(&json("tiny-flux2-klein.denoiser.json"), key)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn spec() -> DenoiserSpec {
    DenoiserSpec::from_pipeline(&fixture()).unwrap()
}

fn step_input(step: usize) -> Vec<f32> {
    if step == 0 { pipeline("noise") } else { pipeline(&format!("step{}.latent", step - 1)) }
}

fn sigma(step: usize) -> f32 {
    json("tiny-flux2-klein.json")["sigmas"][step].as_f64().unwrap() as f32
}

fn ids(spec: &DenoiserSpec) -> Vec<i32> {
    let meta = json("tiny-flux2-klein.json");
    let side = (meta["image_tokens"].as_u64().unwrap() as f64).sqrt() as usize;
    grid_ids(meta["text_tokens"].as_u64().unwrap() as usize, side, side, spec.axes.len())
}

fn run_host(bf16: bool, step: usize, plan: &specrt::plan::Plan) -> (Vec<f32>, Trace) {
    let graph = DenoiserGraph::embedded();
    let spec = spec();
    let ck = Checkpoint::open(&fixture().join("transformer")).unwrap();
    let host = HostDenoiser::load(graph, &spec, plan, &ck, bf16).unwrap();
    let latent = step_input(step);
    let text = pipeline("text.embeds");
    let ids = ids(&spec);
    let mut trace = Trace::default();
    let grid = host.forward(&Inputs { latent: &latent, text: &text, timestep: sigma(step), ids: &ids, image: 64, text_rows: 16, patch_h: 8, patch_w: 8 }, &mut trace).unwrap();
    let v = grid;
    (v, trace)
}

#[test]
fn the_tiny_flux2_config_binds_to_the_generic_denoiser_graph() {
    let s = spec();
    assert_eq!(s.hidden(), 32);
    assert_eq!(s.size("fused").unwrap(), 288);
    assert_eq!(s.size("cat").unwrap(), 128);
    assert_eq!(s.size("mod2").unwrap(), 192);
    assert_eq!(s.size("double").unwrap(), 1);
    assert_eq!(s.size("single").unwrap(), 1);
    assert_eq!(s.axes, vec![4, 4, 4, 4]);
    assert_eq!(s.get("time_scale").unwrap(), 1000.0);
}

#[test]
fn the_denoiser_plan_validates_from_its_signatures() {
    let graph = DenoiserGraph::embedded();
    let plan = graph.plan().unwrap();
    assert_eq!(plan.phase, specrt::plan::Phase::Forward);
    assert_eq!(plan.blocks.iter().map(|b| b.layers[0].as_str()).collect::<Vec<_>>(), ["double", "single"]);
    denoiser::validate(graph, &spec(), &plan).unwrap();
}

#[test]
fn a_plan_that_reads_the_attention_output_before_attention_is_refused() {
    let graph = DenoiserGraph::embedded();
    let mut plan = graph.plan().unwrap();
    let ops = &mut plan.blocks[0].ops;
    let at = ops.iter().position(|o| matches!(o, Op::Dit(DitOp::JointAttention { .. }))).unwrap();
    let attention = ops.remove(at);
    ops.push(attention);
    let e = denoiser::validate(graph, &spec(), &plan).unwrap_err().to_string();
    assert!(e.contains("before any node writes it") || e.contains("shape mismatch"), "{e}");
}

#[test]
fn the_timestep_embedding_matches_numpy() {
    let s = spec();
    for step in 0..3 {
        let t = sigma(step) * s.get("time_scale").unwrap() as f32;
        let got = timestep_embedding(t, s.size("freq").unwrap());
        let want = reference(&format!("step{step}.timestep"));
        let d = max_diff(&got, &want);
        assert!(d < 2e-4, "step {step}: {d}");
    }
}

fn sampling_plan(guided: bool) -> specrt::plan::Plan {
    let p = specrt::sampler::Pipeline::from_dir(&fixture()).unwrap();
    DenoiserGraph::embedded().sampling_plan(&p.step_rule(guided).unwrap(), guided).unwrap()
}

#[test]
fn every_denoiser_kernel_builds_from_a_description_and_emits() {
    let graph = DenoiserGraph::embedded();
    let s = spec();
    let shapes = denoiser::Shapes { image: 64, text: 16, patch_h: 8, patch_w: 8 };
    let mut src = String::new();
    for guided in [false, true] {
        let kernels = denoiser::described_kernels(graph, &s, &sampling_plan(guided), &shapes, &kir::Target::sm_120()).unwrap();
        let names: Vec<&str> = kernels.iter().map(|k| k.name).collect();
        let step = if guided { "dit_step_cfg" } else { "dit_step" };
        for want in ["dit_modulate", "dit_gated_add", "dit_timestep", step, "dit_glu_192_0", "dit_glu_288_96", "dit_split_96", "dit_split_16", "dit_silu_split_32", "dit_copy_32"] {
            assert!(names.contains(&want), "{want} missing from {names:?}");
        }
        src.push_str(&kir::cuda::emit_module(&kernels).unwrap());
    }
    if std::env::var_os("DIT_DUMP").is_some() {
        eprintln!("{src}");
    }
    assert!(src.contains("void __launch_bounds__(256) dit_modulate("), "{src}");
    assert!(src.contains("cosf(") && src.contains("sinf("), "{src}");
    assert!(src.contains("dit_step(") && src.contains("dit_step_cfg("), "{src}");
}

#[test]
fn the_step_rule_lowers_into_the_step_kernel_and_a_foreign_name_is_refused() {
    let graph = DenoiserGraph::embedded();
    let p = specrt::sampler::Pipeline::from_dir(&fixture()).unwrap();
    let guided = p.step_rule(true).unwrap().to_string();
    assert_eq!(guided, "x + (s1 - s0) * (u + g * (c - u))");
    assert_eq!(kir::expr::Expr::parse(&guided).unwrap(), p.step_rule(true).unwrap());
    let plan = sampling_plan(true);
    let step = plan.tail.iter().find_map(|o| if let Op::Dit(d @ DitOp::Step { .. }) = o { Some(*d) } else { None }).unwrap();
    let text = denoiser::step_description(&step, graph).unwrap();
    assert!(text.contains(&format!("expr = \"{guided}\"")), "{text}");
    let bad = kir::expr::Expr::parse("x + w * v").unwrap();
    let e = graph.sampling_plan(&bad, false).unwrap_err().to_string();
    assert!(e.contains("\"w\""), "{e}");
    denoiser::validate(graph, &spec(), &plan).unwrap();
}

#[test]
fn the_euler_step_fuses_into_proj_out_only_without_guidance() {
    let graph = DenoiserGraph::embedded();
    let s = klein();
    let costs = specrt::fusion::Costs::for_target(&kir::Target::sm_120()).with_compute(400.0, 1600.0);
    let tail = |guided: bool| {
        let count = if guided { denoiser::guided_count(4096, 512) } else { denoiser::count(4096, 512) };
        let f = specrt::fusion::denoiser_fusion(graph, &s, &sampling_plan(guided), count, &costs, false).unwrap();
        f.into_iter().find(|x| x.name == "tail").unwrap()
    };
    let plain = tail(false);
    for g in &plain.groups {
        eprintln!("no guidance: [{}]", g.units.join(" + "));
    }
    assert!(grouped(&plain, "proj_out (image)", "scheduler step"), "the Euler update joins the proj_out epilogue");
    let cfg = tail(true);
    for d in &cfg.rejected {
        eprintln!("guided kept apart {}", d.describe());
    }
    assert!(!grouped(&cfg, "proj_out (image)", "guided scheduler step"), "with guidance the step reads both sequences' rows");
    assert!(cfg.rejected.iter().any(|d| d.involves("proj_out (image)", "guided scheduler step") && d.verdict == Err(specrt::fusion::Reason::Rows)));
    assert_eq!(plain.launches_after + 1, plain.launches_before);
}

#[test]
fn the_patchify_layout_matches_the_pipeline_unpatchify() {
    let noise = pipeline("noise");
    let grid = reference("noise.grid");
    let tokens = specrt::denoiser_host::patchify(&grid, 4, 2, 8, 8);
    assert_eq!(max_diff(&tokens, &noise), 0.0);
    assert_eq!(max_diff(&specrt::denoiser_host::unpatchify(&noise, 4, 2, 8, 8), &grid), 0.0);
}

#[test]
fn the_denoiser_modules_compile_with_nvrtc_for_sm_120() {
    let graph = DenoiserGraph::embedded();
    let s = spec();
    let plan = sampling_plan(true);
    let shapes = denoiser::Shapes { image: 64, text: 16, patch_h: 8, patch_w: 8 };
    let t = kir::Target::sm_120();
    let kernels = denoiser::described_kernels(graph, &s, &plan, &shapes, &t).unwrap();
    let compose = kir::cuda::emit_module(&kernels).unwrap();
    let vk = kir::vision::VisionKernel::new(&t, s.size("head_dim").unwrap(), true).unwrap().joint();
    let rule = specrt::sampler::Pipeline::from_dir(&fixture()).unwrap().step_rule(false).unwrap();
    let vision = kir::cuda::emit_vision_step(&vk, &rule).unwrap();
    assert!(vision.contains("vis_mma_step(") && vision.contains("return st_x + (st_s1 - st_s0) * st_v;"), "the Euler rule is the step GEMM's epilogue");
    for (name, src) in [("compose", compose), ("vision", vision)] {
        let cubin = specrt::jit::cubin_for(&src, "sm_120", 1).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        assert!(!cubin.is_empty(), "{name}");
    }
}

fn klein() -> DenoiserSpec {
    let json = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-flux2-klein/klein-4b-transformer.json")).unwrap();
    DenoiserSpec::from_transformer_config(&json, 2).unwrap()
}

fn fusion(compute_prologue: bool, image: usize, text: usize) -> Vec<specrt::fusion::SectionFusion> {
    let graph = DenoiserGraph::embedded();
    let s = klein();
    let plan = graph.plan().unwrap();
    let costs = specrt::fusion::Costs::for_target(&kir::Target::sm_120()).with_compute(400.0, 1600.0);
    specrt::fusion::denoiser_fusion(graph, &s, &plan, denoiser::count(image, text), &costs, compute_prologue).unwrap()
}

fn grouped(f: &specrt::fusion::SectionFusion, a: &str, b: &str) -> bool {
    f.groups.iter().any(|g| g.units.iter().any(|u| u == a) && g.units.iter().any(|u| u == b))
}

#[test]
fn the_fusion_pass_rediscovers_the_denoiser_groups() {
    let sections = fusion(false, 4096, 512);
    let double = sections.iter().find(|s| s.name == "layer double").unwrap();
    let single = sections.iter().find(|s| s.name == "layer single").unwrap();
    for s in [double, single] {
        for g in &s.groups {
            eprintln!("{}: [{}]", s.name, g.units.join(" + "));
        }
        for d in &s.rejected {
            eprintln!("{} kept apart {}", s.name, d.describe());
        }
    }
    assert!(grouped(single, "attn.to_out[hidden:cat] (joint) residual", "joint gated add branch mod_single[2]") || grouped(single, "attn.to_out[0:hidden] (joint)", "joint gated add branch mod_single[2]") || grouped(single, "attn.to_out[hidden:cat] (joint)", "joint gated add branch mod_single[2]"), "the gated residual joins the to_out epilogue");
    assert!(grouped(double, "ff.linear_in (image)", "image swiglu gate_up"), "SwiGLU joins the image linear_in epilogue");
    assert!(grouped(double, "ff_context.linear_in (text)", "text swiglu gate_up"), "SwiGLU joins the text linear_in epilogue");
    assert!(grouped(double, "attn.to_out.0 (image)", "image gated add branch mod_image[2]"), "the image gated residual joins to_out");
    assert!(grouped(double, "ff.linear_out (image)", "image gated add branch mod_image[5]"), "the image gated residual joins linear_out");
    let refused = double.rejected.iter().any(|d| d.verdict == Err(specrt::fusion::Reason::ComputeBound));
    assert!(refused, "the modulate prologue on a compute bound GEMM is refused");
    assert!(double.launches_after < double.launches_before, "{} -> {}", double.launches_before, double.launches_after);
}

#[test]
fn the_compute_prologue_switch_lets_the_modulate_prologue_through() {
    let off = fusion(false, 4096, 512);
    let on = fusion(true, 4096, 512);
    let count = |s: &[specrt::fusion::SectionFusion]| s.iter().find(|x| x.name == "layer double").unwrap().rejected.iter().filter(|d| d.verdict == Err(specrt::fusion::Reason::ComputeBound)).count();
    assert!(count(&off) > 0);
    assert_eq!(count(&on), 0);
}

#[test]
fn decoder_costs_have_no_compute_term() {
    let c = specrt::fusion::Costs::for_target(&kir::Target::sm_120());
    assert_eq!(c.peak_flops_per_byte, None);
    assert!(!c.compute_bound(1, u64::MAX));
    assert_eq!(c.time_bytes(10, 1 << 40), 10);
}

#[test]
fn the_host_plan_with_bf16_weights_matches_the_bf16_weight_reference() {
    let plan = DenoiserGraph::embedded().plan().unwrap();
    for step in 0..3 {
        let (v, trace) = run_host(true, step, &plan);
        let d = max_diff(&v, &reference(&format!("bf16.step{step}.velocity")));
        let to_f32 = max_diff(&v, &pipeline(&format!("step{step}.velocity")));
        eprintln!("step {step}: bf16 weights, {d:e} to the bf16 weight reference, {to_f32:e} to the f32 pipeline");
        assert!(d < VELOCITY_TOLERANCE, "step {step}: {d}");
        assert!(to_f32 < BF16_WEIGHT_BOUND, "step {step}: {to_f32}");
        for (name, x) in &trace.blocks {
            let d = max_diff(x, &reference(&format!("bf16.step{step}.{name}")));
            assert!(d < HIDDEN_TOLERANCE, "step {step} {name}: {d}");
        }
    }
}

#[test]
fn the_host_plan_matches_diffusers_in_f32() {
    let plan = DenoiserGraph::embedded().plan().unwrap();
    for step in 0..3 {
        let (v, trace) = run_host(false, step, &plan);
        let d = max_diff(&v, &pipeline(&format!("step{step}.velocity")));
        eprintln!("step {step}: velocity max diff {d:e}");
        assert!(d < VELOCITY_TOLERANCE, "step {step}: velocity max diff {d}");
        for (name, x) in &trace.blocks {
            let want = reference(&format!("step{step}.{name}"));
            let d = max_diff(x, &want);
            assert!(d < HIDDEN_TOLERANCE, "step {step} {name}: hidden max diff {d}");
        }
    }
}

#[test]
fn a_wrong_modulation_order_fails_the_reference() {
    let mut plan = DenoiserGraph::embedded().plan().unwrap();
    for op in plan.blocks.iter_mut().flat_map(|b| b.ops.iter_mut()).chain(plan.tail.iter_mut()) {
        if let Op::Dit(DitOp::Modulate { shift, scale, .. }) = op {
            std::mem::swap(shift, scale);
        }
    }
    let (v, trace) = run_host(false, 0, &plan);
    let d = max_diff(&v, &pipeline("step0.velocity"));
    let hidden = max_diff(&trace.blocks[0].1, &reference("step0.double0"));
    eprintln!("swapped shift and scale: velocity {d:e}, double0 hidden {hidden:e}");
    assert!(d > VELOCITY_TOLERANCE, "swapping shift and scale must fail the velocity check, max diff {d}");
    assert!(hidden > HIDDEN_TOLERANCE, "swapping shift and scale must fail the hidden check, max diff {hidden}");
}

#[test]
fn the_step_invariant_ops_are_the_text_projection_only() {
    let graph = DenoiserGraph::embedded();
    let s = spec();
    let plan = graph.plan().unwrap();
    let inv = denoiser::hoisted(graph, &s, &plan).unwrap();
    let hoisted: Vec<String> = plan.head.iter().zip(&inv).filter(|(_, i)| **i).map(|(o, _)| o.label()).collect();
    assert_eq!(hoisted, ["split text", "context_embedder (text)"], "{hoisted:?}");
    let naive = denoiser::invariant_ops(graph, &s, &plan.head, &["latent", "timestep"]).unwrap();
    assert!(naive[3], "without the later writes the copy of the context into x looks invariant");
    assert!(!inv[3], "the blocks rewrite x every step, so the copy stays in the step");
}

#[test]
fn the_plan_report_counts_bytes_flops_and_launches() {
    let graph = DenoiserGraph::embedded();
    let s = spec();
    let plan = graph.plan().unwrap();
    let r = denoiser::report(graph, &s, &plan, denoiser::count(64, 16)).unwrap();
    let (_, _, flops, launches) = r.total();
    let ops: usize = plan.head.len() + plan.blocks.iter().map(|b| b.ops.len()).sum::<usize>() + plan.tail.len();
    assert_eq!(launches, ops);
    let attention = 2 * 4 * 80 * 80 * 32;
    assert!(flops > attention as u64, "{flops}");
    let text = r.render();
    assert!(text.contains("step invariant"), "{text}");
}
