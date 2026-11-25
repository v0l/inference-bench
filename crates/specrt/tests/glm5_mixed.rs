use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use specrt::spec::{Quant, Spec};
use specrt::weights::{Checkpoint, Matrix, Mixer, Stacked, e4m3_to_f32, f32_to_bf16, f32_to_e4m3};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const P: &str = "model.language_model.layers.";

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-glm5-next")
}

fn floats(t: &TensorView) -> Vec<f32> {
    match t.dtype() {
        Dtype::F32 => t.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
        Dtype::BF16 => t.data().as_chunks::<2>().0.iter().map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16)).collect(),
        other => panic!("{other:?}"),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Mxfp8,
    Fp8Block,
    Nvfp4,
}

impl Format {
    fn algo(self) -> serde_json::Value {
        match self {
            Format::Mxfp8 => serde_json::json!({ "quant_algo": "MXFP8", "weight_block_size": [1, 32] }),
            Format::Fp8Block => serde_json::json!({ "quant_algo": "FP8_PB_WO", "weight_block_size": [128, 128] }),
            Format::Nvfp4 => serde_json::json!({ "quant_algo": "NVFP4", "group_size": 16 }),
        }
    }
}

struct Out {
    tensors: BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
}

impl Out {
    fn put(&mut self, name: &str, dtype: Dtype, shape: &[usize], data: Vec<u8>) {
        self.tensors.insert(name.to_string(), (dtype, shape.to_vec(), data));
    }

    fn bf16(&mut self, name: &str, shape: &[usize], v: &[f32]) {
        self.put(name, Dtype::BF16, shape, v.iter().flat_map(|x| f32_to_bf16(*x).to_le_bytes()).collect());
    }

    fn write(&self, dir: &Path) {
        let views: Vec<(String, TensorView)> = self.tensors.iter().map(|(n, (d, s, b))| (n.clone(), TensorView::new(*d, s.clone(), b).unwrap())).collect();
        safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    }
}

fn mxfp8(w: &[f32], cols: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    let (mut codes, mut scales, mut exact) = (Vec::new(), Vec::new(), Vec::new());
    for g in w.chunks(32) {
        let amax = g.iter().fold(0f32, |m, v| m.max(v.abs()));
        let e = if amax == 0.0 { 0 } else { (amax / 448.0).log2().ceil() as i32 };
        let code = (e + 127) as u8;
        let s = f32::from_bits(u32::from(code) << 23);
        scales.push(code);
        for v in g {
            let c = f32_to_e4m3(v / s);
            codes.push(c);
            exact.push(e4m3_to_f32(c) * s);
        }
    }
    assert_eq!(scales.len(), w.len() / cols * (cols / 32));
    (codes, scales, exact)
}

fn fp8_block(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let (rb, cb) = (rows.div_ceil(128), cols.div_ceil(128));
    let mut grid = vec![0f32; rb * cb];
    for r in 0..rows {
        for c in 0..cols {
            let b = r / 128 * cb + c / 128;
            grid[b] = grid[b].max(w[r * cols + c].abs() / 448.0);
        }
    }
    grid.iter_mut().filter(|g| **g == 0.0).for_each(|g| *g = 1.0);
    let mut codes = Vec::new();
    let mut exact = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let s = grid[r / 128 * cb + c / 128];
            let code = f32_to_e4m3(w[r * cols + c] / s);
            codes.push(code);
            exact.push(e4m3_to_f32(code) * s);
        }
    }
    (codes, grid, exact)
}

fn quantised(layer_formats: &[(String, Format)], dir: &Path, twin: &Path) {
    let bytes = std::fs::read(fixture().join("model.safetensors")).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut mixed = Out { tensors: BTreeMap::new() };
    let mut plain = Out { tensors: BTreeMap::new() };
    for name in st.names() {
        let t = st.tensor(name).unwrap();
        let module = name.strip_suffix(".weight");
        let format = module.and_then(|m| layer_formats.iter().find(|(k, _)| m == k || m.starts_with(&format!("{k}."))).map(|(_, f)| *f));
        let Some(format) = format.filter(|_| t.shape().len() == 2) else {
            mixed.put(name, t.dtype(), t.shape(), t.data().to_vec());
            plain.put(name, t.dtype(), t.shape(), t.data().to_vec());
            continue;
        };
        let (rows, cols) = (t.shape()[0], t.shape()[1]);
        let w = floats(&t);
        let m = module.unwrap();
        match format {
            Format::Mxfp8 => {
                let (codes, scales, exact) = mxfp8(&w, cols);
                mixed.put(name, Dtype::F8_E4M3, &[rows, cols], codes);
                mixed.put(&format!("{m}.weight_scale"), Dtype::U8, &[rows, cols / 32], scales);
                plain.bf16(name, &[rows, cols], &exact);
            }
            Format::Fp8Block => {
                let (codes, grid, exact) = fp8_block(&w, rows, cols);
                mixed.put(name, Dtype::F8_E4M3, &[rows, cols], codes);
                mixed.put(&format!("{m}.weight_scale"), Dtype::F32, &[rows.div_ceil(128), 1, cols.div_ceil(128), 1], grid.iter().flat_map(|g| g.to_le_bytes()).collect());
                plain.bf16(name, &[rows, cols], &exact);
            }
            Format::Nvfp4 => {
                let bf: Vec<u16> = w.iter().map(|v| f32_to_bf16(*v)).collect();
                let f = specrt::weights::quantize_nvfp4(&bf, cols);
                mixed.put(name, Dtype::U8, &[rows, cols / 2], f.packed.to_vec());
                mixed.put(&format!("{m}.weight_scale"), Dtype::F8_E4M3, &[rows, cols / 16], f.scales.to_vec());
                mixed.put(&format!("{m}.weight_scale_2"), Dtype::F32, &[], (1.0 / f.global).to_le_bytes().to_vec());
                let exact: Vec<f32> = (0..rows).flat_map(|r| f.dequant_row(r)).collect();
                plain.bf16(name, &[rows, cols], &exact);
            }
        }
    }
    let layers: serde_json::Map<String, serde_json::Value> = layer_formats.iter().map(|(k, f)| (k.clone(), f.algo())).collect();
    let quant = serde_json::json!({ "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "exclude_modules": [], "quantized_layers": layers });
    let mut config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixture().join("config.json")).unwrap()).unwrap();
    for (d, out, q) in [(dir, &mixed, Some(&quant)), (twin, &plain, None)] {
        std::fs::create_dir_all(d).unwrap();
        out.write(d);
        let mut c = config.clone();
        if let Some(q) = q {
            c["quantization_config"] = q.clone();
            std::fs::write(d.join("hf_quant_config.json"), q.to_string()).unwrap();
        }
        std::fs::write(d.join("config.json"), c.to_string()).unwrap();
    }
    config["quantization_config"] = quant;
}

fn glm5_layout(experts: Option<Format>) -> Vec<(String, Format)> {
    let mut out = vec![("lm_head".to_string(), Format::Mxfp8)];
    for m in ["gate_proj", "up_proj", "down_proj"] {
        out.push((format!("{P}0.mlp.{m}"), Format::Mxfp8));
    }
    for l in 0..3 {
        for m in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            out.push((format!("{P}{l}.self_attn.{m}"), Format::Mxfp8));
        }
    }
    for m in ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"] {
        out.push((format!("{P}3.self_attn.{m}"), Format::Fp8Block));
    }
    for l in 1..4 {
        for m in ["gate_proj", "up_proj", "down_proj"] {
            out.push((format!("{P}{l}.mlp.shared_experts.{m}"), Format::Fp8Block));
        }
        if let Some(f) = experts {
            out.push((format!("{P}{l}.mlp.experts"), f));
        }
    }
    out
}

fn dirs(tag: &str) -> (PathBuf, PathBuf) {
    let base = std::env::temp_dir().join(format!("specrt-glm5-mixed-{tag}-{}", std::process::id()));
    (base.join("mixed"), base.join("twin"))
}

fn spec_for(dir: &Path) -> Spec {
    let mut spec = Spec::from_hf_config_serving(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap_or_else(|e| panic!("{e:#}"));
    spec.repo = dir.display().to_string();
    spec
}

fn open(dir: &Path, spec: &Spec) -> Checkpoint {
    let ck = Checkpoint::open(dir).unwrap().renamed(&spec.tensors);
    ck.check_quant_layout(&ck.quant_map().unwrap().renamed(&spec.tensors)).unwrap_or_else(|e| panic!("{e:#}"));
    ck
}

fn bf16_of<'a>(m: &'a Matrix) -> &'a [u16] {
    match m {
        Matrix::Bf16(w) => w,
        _ => panic!("expected a bf16 matrix"),
    }
}

#[test]
fn a_mixed_precision_glm5_next_checkpoint_loads_its_projections_as_the_decoded_bf16() {
    let (mixed, twin) = dirs("bf16");
    quantised(&glm5_layout(None), &mixed, &twin);
    let spec = spec_for(&mixed);
    spec.runnable().unwrap_or_else(|e| panic!("{e:#}"));
    assert_eq!((spec.quant, spec.mlp.quant), (Quant::Bf16, None), "bf16 experts keep the stack bf16");
    assert_eq!(spec.quant_map.0.len(), glm5_layout(None).len());
    let twin_spec = spec_for(&twin);
    let a = open(&mixed, &spec);
    let b = open(&twin, &twin_spec);
    for i in 0..4 {
        let (la, lb) = (a.layer(&spec, i).unwrap(), b.layer(&twin_spec, i).unwrap());
        match (&la.mixer, &lb.mixer) {
            (Mixer::Described(x), Mixer::Described(y)) => {
                assert_eq!(bf16_of(&x.in_proj), bf16_of(&y.in_proj), "layer {i} KDA in_proj");
                assert_eq!(bf16_of(&x.out), bf16_of(&y.out), "layer {i} KDA o_proj");
            }
            (Mixer::Attention(x), Mixer::Attention(y)) => {
                assert_eq!(bf16_of(&x.qkv), bf16_of(&y.qkv), "layer {i} DSA projections");
                assert_eq!(bf16_of(&x.o), bf16_of(&y.o), "layer {i} DSA o_proj");
                let names: Vec<&str> = x.extra.iter().map(|(n, _)| n.as_str()).collect();
                assert!(names.contains(&"q_b_proj.weight"), "{names:?}");
                assert_eq!(x.extra, y.extra, "layer {i} node weights, q_b_proj from FP8 blocks");
            }
            _ => panic!("layer {i} mixers differ"),
        }
        assert_eq!(bf16_of(&la.gate_up), bf16_of(&lb.gate_up), "layer {i} dense gate_up");
        assert_eq!(bf16_of(&la.down), bf16_of(&lb.down), "layer {i} dense down");
        match (la.moe.as_ref().map(|m| (&m.gate, &m.down)), lb.moe.as_ref().map(|m| (&m.gate, &m.down))) {
            (Some((Stacked::Bf16(g), Stacked::Bf16(d))), Some((Stacked::Bf16(g2), Stacked::Bf16(d2)))) => assert!(g == g2 && d == d2, "layer {i} experts and FP8 shared expert"),
            (None, None) => assert_eq!(i, 0),
            _ => panic!("layer {i} moe differs"),
        }
    }
    let (_, ha) = a.head(&spec).unwrap();
    let (_, hb) = b.head(&twin_spec).unwrap();
    assert_eq!(bf16_of(&ha), bf16_of(&hb), "MXFP8 lm_head");
    std::fs::remove_dir_all(mixed.parent().unwrap()).ok();
}

#[test]
fn nvfp4_routed_experts_stack_as_fp4_and_take_a_requantised_fp8_shared_expert() {
    let (mixed, twin) = dirs("fp4");
    quantised(&glm5_layout(Some(Format::Nvfp4)), &mixed, &twin);
    let config = std::fs::read_to_string(mixed.join("config.json")).unwrap();
    let refusal = Spec::from_hf_config_serving(&config).unwrap_err().to_string();
    assert!(refusal.contains("nvfp4_g16 needs hidden to be a multiple of 256"), "the tiny expert width is too narrow for the fp4 expert kernels: {refusal}");
    let twin_spec = spec_for(&twin);
    let mut spec = twin_spec.clone();
    spec.repo = mixed.display().to_string();
    spec.mlp.quant = Some(Quant::Nvfp4G16);
    spec.quant_map = Spec::describe_hf_config(&config).unwrap().quant_map;
    assert_eq!(spec.mlp_quant(0), Quant::Bf16, "the MXFP8 dense MLP is not fp4");
    assert!((1..4).all(|l| spec.mlp_quant(l) == Quant::Nvfp4G16));
    let a = open(&mixed, &spec);
    let layer = a.layer(&spec, 3).unwrap();
    let moe = layer.moe.as_ref().unwrap();
    let Stacked::Fp4 { packed, scales, .. } = &moe.gate else { panic!("fp4 expert stack") };
    let (inter, h, experts) = (32, 128, 8);
    assert_eq!(packed.len(), (experts + 1) * inter * h / 2, "8 routed experts and the shared expert");
    assert_eq!(scales.len(), (experts + 1) * inter * h / 16);
    let f = a.fp4(&format!("{P}3.mlp.experts.5.gate_proj"), inter, h).unwrap();
    assert_eq!(&packed[5 * inter * h / 2..6 * inter * h / 2], &f.packed[..], "routed expert bytes are kept as stored");
    let b = open(&twin, &twin_spec);
    let shared: Vec<f32> = b.layer(&twin_spec, 3).unwrap().moe.map(|m| match m.gate {
        Stacked::Bf16(w) => w[experts * inter * h..].iter().map(|v| specrt::weights::bf16_to_f32(*v)).collect(),
        _ => panic!(),
    }).unwrap();
    let requant = specrt::weights::quantize_nvfp4(&shared.iter().map(|v| f32_to_bf16(*v)).collect::<Vec<_>>(), h);
    assert_eq!(&packed[experts * inter * h / 2..], &requant.packed[..], "the FP8 shared expert is decoded, then requantised to fp4");
    std::fs::remove_dir_all(mixed.parent().unwrap()).ok();
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_engine_on_a_mixed_precision_checkpoint_matches_its_decoded_bf16_twin() {
    use specrt::runtime::Runtime;
    let (mixed, twin) = dirs("gpu");
    quantised(&glm5_layout(None), &mixed, &twin);
    let tokens: Vec<u32> = (0..24).map(|i| (i * 37 + 11) % 512).collect();
    let run = |dir: &Path| {
        let mut rt = Runtime::load_batched(&spec_for(dir), 256, 1).unwrap_or_else(|e| panic!("{e:#}"));
        rt.trace_prefill(&tokens).unwrap().logits
    };
    let (a, b) = (run(&mixed), run(&twin));
    let gap = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    eprintln!("mixed precision against its bf16 twin: max abs {gap}");
    assert!(a.iter().all(|v| v.is_finite()));
    assert_eq!(gap, 0.0, "the same decoded weights give the same logits");
    std::fs::remove_dir_all(mixed.parent().unwrap()).ok();
}
