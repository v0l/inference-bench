use crate::spec::{Block, Quant, Spec};

pub const DEFAULT_MAX_CONTEXT: usize = 262_144;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    Attention,
    SlidingAttention,
    DeltaNet,
}

impl LayerKind {
    pub fn label(self) -> &'static str {
        match self {
            LayerKind::Attention => "full attention",
            LayerKind::SlidingAttention => "sliding attention",
            LayerKind::DeltaNet => "gated delta-net",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerInfo {
    pub index: usize,
    pub kind: LayerKind,
    pub mixer_quant: Quant,
    pub mlp_quant: Quant,
    pub mixer_params: u64,
    pub mlp_params: u64,
    pub moe: bool,
    pub router_params: u64,
    pub routed_params: u64,
    pub shared_params: u64,
    pub active_params: u64,
    pub active_bytes: u64,
    pub norm_params: u64,
    pub params: u64,
    pub bytes: u64,
}

impl LayerInfo {
    pub fn dense_params(&self) -> u64 {
        self.mlp_params - self.router_params - self.routed_params - self.shared_params
    }

    pub fn routed_bytes(&self) -> u64 {
        quant_bytes(self.routed_params, self.mlp_quant)
    }

    pub fn shared_bytes(&self) -> u64 {
        quant_bytes(self.shared_params, self.mlp_quant)
    }

    pub fn dense_bytes(&self) -> u64 {
        quant_bytes(self.dense_params(), self.mlp_quant)
    }

    pub fn router_bytes(&self) -> u64 {
        2 * self.router_params
    }

    pub fn mlp_bytes(&self) -> u64 {
        self.dense_bytes() + self.routed_bytes() + self.shared_bytes() + self.router_bytes()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoeSummary {
    pub experts: usize,
    pub active: Option<usize>,
    pub expert_intermediate: usize,
    pub shared_intermediate: usize,
    pub moe_layers: usize,
    pub routed_params: u64,
    pub routed_bytes: u64,
    pub shared_params: u64,
    pub router_params: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Component {
    pub name: &'static str,
    pub params: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelAnatomy {
    pub name: String,
    pub layers: Vec<LayerInfo>,
    pub components: Vec<Component>,
    pub embedding_params: u64,
    pub head_params: u64,
    pub head_quant: Quant,
    pub head_tied: bool,
    pub mtp_layers: usize,
    pub mtp_params: u64,
    pub mtp_bytes: u64,
    pub moe: Option<MoeSummary>,
    pub total_params: u64,
    pub total_bytes: u64,
    pub decode_params: u64,
    pub decode_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub state_bytes_per_seq: u64,
    pub max_context: usize,
}

pub fn quant_tag(q: Quant) -> &'static str {
    match q {
        Quant::Bf16 => "bf16",
        Quant::Q8 => "q8",
        Quant::Fp8Block128 => "fp8 block-128",
        Quant::Fp8Channel => "fp8 channel",
        Quant::Nvfp4G16 => "nvfp4 g16",
        Quant::Gguf => "gguf",
    }
}

pub fn quant_bytes(params: u64, q: Quant) -> u64 {
    let (num, den) = match q {
        Quant::Bf16 => (2, 1),
        Quant::Q8 | Quant::Fp8Channel => (1, 1),
        Quant::Fp8Block128 => (16386, 16384),
        Quant::Nvfp4G16 | Quant::Gguf => (9, 16),
    };
    (params * num).div_ceil(den)
}

struct Counts {
    quantised: u64,
    plain: u64,
}

fn attention_counts(spec: &Spec) -> Counts {
    let h = spec.dims.hidden as u64;
    let a = &spec.attention;
    let bias = if a.qkv_bias { (spec.qkv_rows() + spec.dims.hidden) as u64 } else { 0 };
    let norms = if a.qk_norm { 2 * a.head_dim as u64 } else { 0 };
    Counts { quantised: spec.qkv_rows() as u64 * h + h * spec.q_dim() as u64, plain: bias + norms }
}

fn delta_counts(spec: &Spec) -> Counts {
    let h = spec.dims.hidden as u64;
    let Some(la) = &spec.linear_attention else { return Counts { quantised: 0, plain: 0 } };
    let value = la.value_dim() as u64;
    let conv = la.conv_dim() as u64;
    let heads = la.value_heads as u64;
    Counts {
        quantised: conv * h + value * h + h * value,
        plain: 2 * heads * h + conv * la.conv_kernel as u64 + 2 * heads + la.value_head_dim as u64,
    }
}

struct MlpParts {
    dense: u64,
    routed: u64,
    shared: u64,
    router: u64,
    active_routed: u64,
}

impl MlpParts {
    fn total(&self) -> u64 {
        self.dense + self.routed + self.shared + self.router
    }

    fn bytes(&self, q: Quant) -> u64 {
        quant_bytes(self.dense, q) + quant_bytes(self.routed, q) + quant_bytes(self.shared, q) + 2 * self.router
    }

    fn active(&self) -> u64 {
        self.dense + self.active_routed + self.shared + self.router
    }

    fn active_bytes(&self, q: Quant) -> u64 {
        quant_bytes(self.dense, q) + quant_bytes(self.active_routed, q) + quant_bytes(self.shared, q) + 2 * self.router
    }
}

fn mlp_parts(spec: &Spec, index: usize, force_moe: bool) -> MlpParts {
    let h = spec.dims.hidden as u64;
    match &spec.moe {
        Some(m) if force_moe || m.is_sparse(index) => {
            let routed = 3 * h * m.expert_intermediate as u64 * m.experts as u64;
            let per_token = m.active.unwrap_or(m.experts).min(m.experts) as u64;
            MlpParts {
                dense: 0,
                routed,
                shared: 3 * h * m.shared_intermediate as u64,
                router: h * m.experts as u64 + if m.shared_intermediate > 0 { h } else { 0 },
                active_routed: routed / m.experts.max(1) as u64 * per_token,
            }
        }
        _ => MlpParts { dense: 3 * h * spec.dims.intermediate as u64, routed: 0, shared: 0, router: 0, active_routed: 0 },
    }
}

fn layer(spec: &Spec, index: usize) -> LayerInfo {
    let kind = match spec.block(index) {
        Block::Linear => LayerKind::DeltaNet,
        Block::Sliding => LayerKind::SlidingAttention,
        _ => LayerKind::Attention,
    };
    let mixer = match kind {
        LayerKind::Attention | LayerKind::SlidingAttention => attention_counts(spec),
        LayerKind::DeltaNet => delta_counts(spec),
    };
    let mixer_quant = spec.quant;
    let mlp_quant = spec.mlp_quant(index);
    let parts = mlp_parts(spec, index, false);
    let mlp = parts.total();
    let norm_params = 2 * spec.dims.hidden as u64;
    let mixer_params = mixer.quantised + mixer.plain;
    let fixed = quant_bytes(mixer.quantised, mixer_quant) + 2 * mixer.plain + 2 * norm_params;
    LayerInfo {
        index,
        kind,
        mixer_quant,
        mlp_quant,
        mixer_params,
        mlp_params: mlp,
        moe: parts.routed > 0,
        router_params: parts.router,
        routed_params: parts.routed,
        shared_params: parts.shared,
        active_params: mixer_params + parts.active() + norm_params,
        active_bytes: fixed + parts.active_bytes(mlp_quant),
        norm_params,
        params: mixer_params + mlp + norm_params,
        bytes: fixed + parts.bytes(mlp_quant),
    }
}

fn mtp_counts(spec: &Spec, count: usize) -> (u64, u64) {
    if count == 0 {
        return (0, 0);
    }
    let h = spec.dims.hidden as u64;
    let attn = attention_counts(spec);
    let parts = mlp_parts(spec, 0, true);
    let mlp = parts.total();
    let norms = 5 * h;
    let fc = 2 * h * h;
    let params = attn.quantised + attn.plain + mlp + norms + fc;
    let bytes = quant_bytes(attn.quantised, spec.quant) + 2 * attn.plain + parts.bytes(spec.quant) + 2 * norms + 2 * fc;
    (params * count as u64, bytes * count as u64)
}

impl ModelAnatomy {
    pub fn new(spec: &Spec, mtp_layers: usize, max_context: Option<usize>) -> Self {
        let layers: Vec<LayerInfo> = (0..spec.layers.count).map(|i| layer(spec, i)).collect();
        let hidden = spec.dims.hidden as u64;
        let matrix = spec.dims.vocab as u64 * hidden;
        let head_quant = spec.lm_head_quant();
        let head_tied = spec.dims.tie_embeddings;
        let embedding_params = matrix;
        let head_params = if head_tied { 0 } else { matrix };
        let head_bytes = if head_tied { 0 } else { quant_bytes(head_params, head_quant) };
        let embedding_bytes = 2 * embedding_params;
        let (mtp_params, mtp_bytes) = mtp_counts(spec, mtp_layers);

        let sum = |kind: LayerKind, f: fn(&LayerInfo) -> u64| layers.iter().filter(|l| l.kind == kind).map(f).sum::<u64>();
        let attn_params = layers.iter().filter(|l| l.kind != LayerKind::DeltaNet).map(|l| l.mixer_params).sum::<u64>();
        let delta_params = sum(LayerKind::DeltaNet, |l| l.mixer_params);
        let dense_total: u64 = layers.iter().map(|l| l.dense_params()).sum();
        let routed_total: u64 = layers.iter().map(|l| l.routed_params).sum();
        let shared_total: u64 = layers.iter().map(|l| l.shared_params).sum();
        let router_total: u64 = layers.iter().map(|l| l.router_params).sum();
        let norm_total: u64 = layers.iter().map(|l| l.norm_params).sum::<u64>() + hidden;
        let mixer_bytes = |delta: bool| -> u64 {
            layers
                .iter()
                .filter(|l| (l.kind == LayerKind::DeltaNet) == delta)
                .map(|l| {
                    let c = if delta { delta_counts(spec) } else { attention_counts(spec) };
                    quant_bytes(c.quantised, l.mixer_quant) + 2 * c.plain
                })
                .sum()
        };
        let dense_bytes: u64 = layers.iter().map(|l| l.dense_bytes()).sum();
        let routed_bytes: u64 = layers.iter().map(|l| l.routed_bytes()).sum();
        let shared_bytes: u64 = layers.iter().map(|l| l.shared_bytes()).sum();
        let router_bytes: u64 = layers.iter().map(|l| l.router_bytes()).sum();
        let norm_bytes = 2 * norm_total;

        let mut components = vec![Component { name: "embedding", params: embedding_params, bytes: embedding_bytes }];
        components.push(Component { name: "attention", params: attn_params, bytes: mixer_bytes(false) });
        if delta_params > 0 {
            components.push(Component { name: "delta-net", params: delta_params, bytes: mixer_bytes(true) });
        }
        if dense_total > 0 || routed_total == 0 {
            components.push(Component { name: "mlp", params: dense_total, bytes: dense_bytes });
        }
        if routed_total > 0 {
            components.push(Component { name: "experts", params: routed_total, bytes: routed_bytes });
            components.push(Component { name: "shared expert", params: shared_total, bytes: shared_bytes });
            components.push(Component { name: "router", params: router_total, bytes: router_bytes });
        }
        components.push(Component { name: "norms", params: norm_total, bytes: norm_bytes });
        components.push(Component { name: "lm_head", params: head_params, bytes: head_bytes });
        if mtp_params > 0 {
            components.push(Component { name: "mtp", params: mtp_params, bytes: mtp_bytes });
        }

        let total_params = components.iter().map(|c| c.params).sum();
        let total_bytes = components.iter().map(|c| c.bytes).sum();
        let layer_params: u64 = layers.iter().map(|l| l.active_params).sum();
        let layer_bytes: u64 = layers.iter().map(|l| l.active_bytes).sum();
        let decode_params = layer_params + hidden + matrix;
        let decode_bytes = layer_bytes + 2 * hidden + quant_bytes(matrix, head_quant);
        let moe = spec.moe.as_ref().map(|m| MoeSummary {
            experts: m.experts,
            active: m.active,
            expert_intermediate: m.expert_intermediate,
            shared_intermediate: m.shared_intermediate,
            moe_layers: layers.iter().filter(|l| l.moe).count(),
            routed_params: routed_total,
            routed_bytes,
            shared_params: shared_total,
            router_params: router_total,
        });

        let kv_bytes_per_token = (2 * spec.kv_layers() * spec.attention.kv_heads * spec.attention.head_dim * 2) as u64;
        let state_bytes_per_seq = match &spec.linear_attention {
            Some(la) => {
                let ssm = la.value_heads * la.key_head_dim * la.key_head_dim;
                let conv = la.conv_dim() * (la.conv_kernel - 1);
                (spec.linear_layers() * (ssm + conv) * 4) as u64
            }
            None => 0,
        };

        ModelAnatomy {
            name: spec.name.clone(),
            layers,
            components,
            embedding_params,
            head_params,
            head_quant,
            head_tied,
            mtp_layers,
            mtp_params,
            mtp_bytes,
            moe,
            total_params,
            total_bytes,
            decode_params,
            decode_bytes,
            kv_bytes_per_token,
            state_bytes_per_seq,
            max_context: max_context.unwrap_or(DEFAULT_MAX_CONTEXT),
        }
    }

    pub fn count(&self, kind: LayerKind) -> usize {
        self.layers.iter().filter(|l| l.kind == kind).count()
    }

    pub fn memory_at(&self, context: usize) -> u64 {
        self.total_bytes + self.kv_bytes_per_token * context as u64 + self.state_bytes_per_seq
    }

    pub fn decode_tokens_per_second(&self, bandwidth_bytes_per_second: f64) -> f64 {
        bandwidth_bytes_per_second / self.decode_bytes as f64
    }

    pub fn quant_mix(&self) -> Vec<(Quant, u64)> {
        let mut mix: Vec<(Quant, u64)> = Vec::new();
        let mut add = |q: Quant, bytes: u64| match mix.iter_mut().find(|(k, _)| *k == q) {
            Some(e) => e.1 += bytes,
            None => mix.push((q, bytes)),
        };
        for l in &self.layers {
            add(l.mixer_quant, quant_bytes(l.mixer_params, l.mixer_quant));
            add(l.mlp_quant, l.mlp_bytes() - l.router_bytes());
            if l.router_bytes() > 0 {
                add(Quant::Bf16, l.router_bytes());
            }
        }
        mix.sort_by_key(|(_, b)| std::cmp::Reverse(*b));
        mix
    }

    pub fn pattern(&self) -> String {
        self.layers
            .iter()
            .map(|l| match l.kind {
                LayerKind::Attention => 'A',
                LayerKind::SlidingAttention => 'S',
                LayerKind::DeltaNet => 'D',
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::PAGE;
    use crate::runtime::kv_block_bytes;
    use std::path::Path;

    fn fixture(name: &str) -> Spec {
        Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)).unwrap()
    }

    fn qwen3() -> Spec {
        Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/qwen3-0.6b.toml")).unwrap()
    }

    fn moe_35b() -> Spec {
        let json = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.6-35b-a3b-fp8.config.json")).unwrap();
        let mut spec = Spec::describe_hf_config(&json).unwrap();
        let rows = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/headers/qwen3.6-35b-a3b-fp8.json")).unwrap();
        spec.apply_structure(&crate::structure::Structure::from_json(&rows).unwrap());
        spec
    }

    fn component(a: &ModelAnatomy, name: &str) -> u64 {
        a.components.iter().find(|c| c.name == name).unwrap().params
    }

    #[test]
    fn qwen3_0_6b_total_matches_the_checkpoint_header() {
        let a = ModelAnatomy::new(&qwen3(), 0, None);
        let header_total: u64 = 751_632_384 - 151_936 * 1024;
        let off = a.total_params.abs_diff(header_total) as f64 / header_total as f64;
        assert!(off < 0.01, "{} vs {header_total}", a.total_params);
        assert_eq!(a.total_params, header_total);
        assert!(a.total_params.abs_diff(600_000_000) < 6_000_000);
        assert!(a.head_tied);
        assert_eq!(a.layers.len(), 28);
        assert!(a.layers.iter().all(|l| l.kind == LayerKind::Attention));
    }

    #[test]
    fn qwen3_8_27b_has_sixteen_attention_layers_in_a_three_to_one_rhythm() {
        let a = ModelAnatomy::new(&fixture("qwen3.8-27b-exact.toml"), 1, None);
        assert_eq!(a.layers.len(), 64);
        assert_eq!(a.count(LayerKind::Attention), 16);
        assert_eq!(a.count(LayerKind::DeltaNet), 48);
        for (i, l) in a.layers.iter().enumerate() {
            let expect = if i % 4 == 3 { LayerKind::Attention } else { LayerKind::DeltaNet };
            assert_eq!(l.kind, expect, "layer {i}");
        }
        assert_eq!(a.pattern().len(), 64);
        assert!(a.pattern().starts_with("DDDA"));
    }

    #[test]
    fn qwen3_8_27b_parameters_match_the_safetensors_headers() {
        let a = ModelAnatomy::new(&fixture("qwen3.8-27b-exact.toml"), 1, None);
        let layers: u64 = a.layers.iter().map(|l| l.params).sum();
        assert_eq!(layers, 24_353_196_544);
        assert_eq!(component(&a, "embedding"), 1_271_398_400);
        assert_eq!(component(&a, "lm_head"), 1_271_398_400);
        assert_eq!(component(&a, "mtp"), 424_699_392);
        assert_eq!(a.total_params, 27_320_697_856);
    }

    #[test]
    fn nvfp4_shrinks_the_mlp_only_below_quant_until() {
        let fp8 = ModelAnatomy::new(&fixture("qwen3.8-27b-exact.toml"), 0, None);
        let fp4 = ModelAnatomy::new(&fixture("qwen3.8-27b-nvfp4-exact.toml"), 0, None);
        assert_eq!(fp4.total_params, fp8.total_params);
        assert!(fp4.total_bytes < fp8.total_bytes);
        assert_eq!(fp4.layers[0].mlp_quant, Quant::Nvfp4G16);
        assert_eq!(fp4.layers[63].mlp_quant, Quant::Fp8Channel);
        assert_eq!(fp4.layers[55].mlp_quant, Quant::Nvfp4G16);
        assert_eq!(fp4.layers[56].mlp_quant, Quant::Fp8Channel);
        assert_eq!(fp4.head_quant, Quant::Bf16);
    }

    #[test]
    fn kv_and_state_bytes_follow_the_runtime_allocation() {
        for name in ["qwen3.8-27b-exact.toml", "qwen3.8-27b-nvfp4-exact.toml"] {
            let spec = fixture(name);
            let a = ModelAnatomy::new(&spec, 0, None);
            assert_eq!(a.kv_bytes_per_token as usize * PAGE, kv_block_bytes(&spec, 1), "{name}");
            assert_eq!(a.state_bytes_per_seq, crate::serve::slot_state_bytes(&spec), "{name}");
        }
        let small = qwen3();
        let a = ModelAnatomy::new(&small, 0, None);
        assert_eq!(a.kv_bytes_per_token as usize * PAGE, kv_block_bytes(&small, 1));
        assert_eq!(a.kv_bytes_per_token, 2 * 28 * 8 * 128 * 2);
        assert_eq!(a.state_bytes_per_seq, 0);
    }

    #[test]
    fn bytes_follow_the_quant_and_decode_reads_the_head_matrix() {
        let bf16 = ModelAnatomy::new(&qwen3(), 0, None);
        assert_eq!(bf16.total_bytes, bf16.total_params * 2);
        assert_eq!(bf16.decode_params, bf16.total_params);
        let fp8 = ModelAnatomy::new(&fixture("qwen3.8-27b-exact.toml"), 0, None);
        assert!(fp8.total_bytes < fp8.total_params * 2 * 6 / 10);
        assert!(fp8.decode_bytes <= fp8.total_bytes);
        assert_eq!(quant_bytes(32, Quant::Nvfp4G16), 18);
        assert_eq!(quant_bytes(10, Quant::Bf16), 20);
    }

    #[test]
    fn the_quant_mix_lists_every_format_by_weight_bytes() {
        let mixed = ModelAnatomy::new(&fixture("qwen3.8-27b-nvfp4-exact.toml"), 0, None).quant_mix();
        assert_eq!(mixed.iter().map(|m| m.0).collect::<Vec<_>>(), vec![Quant::Fp8Channel, Quant::Nvfp4G16]);
        let plain = ModelAnatomy::new(&fixture("qwen3.8-27b-exact.toml"), 0, None).quant_mix();
        assert_eq!(plain.len(), 1);
    }

    #[test]
    fn qwen3_6_35b_a3b_reads_as_a_moe_with_the_experts_of_its_config() {
        let spec = moe_35b();
        let m = spec.moe.as_ref().unwrap();
        assert_eq!((m.experts, m.active, m.expert_intermediate, m.shared_intermediate), (256, Some(8), 512, 512));
        assert_eq!(spec.dims.hidden, 2048);
        assert_eq!(spec.layers.count, 40);
        assert_eq!(spec.attention_layers(), 10);
        assert_eq!(spec.quant, Quant::Fp8Block128);
    }

    #[test]
    fn qwen3_6_35b_a3b_total_and_active_parameters_match_the_safetensors_headers() {
        let a = ModelAnatomy::new(&moe_35b(), 1, None);
        let layers: u64 = a.layers.iter().map(|l| l.params).sum();
        assert_eq!(layers, 33_643_489_920);
        assert_eq!(component(&a, "embedding"), 508_559_360);
        assert_eq!(component(&a, "lm_head"), 508_559_360);
        assert_eq!(component(&a, "experts"), 32_212_254_720);
        assert_eq!(component(&a, "mtp"), 844_640_768);
        assert_eq!(a.total_params, 34_660_610_688 + 844_640_768);
        let moe = a.moe.as_ref().unwrap();
        assert_eq!(moe.moe_layers, 40);
        assert_eq!(moe.routed_params, 32_212_254_720);
        let routed_active = 32_212_254_720 / 256 * 8;
        assert_eq!(a.decode_params, 34_660_610_688 - 508_559_360 - 32_212_254_720 + routed_active);
        assert!(a.decode_params.abs_diff(3_000_000_000) < 600_000_000, "{}", a.decode_params);
    }

    #[test]
    fn qwen3_6_35b_a3b_bytes_match_the_checkpoint_and_decode_reads_only_active_experts() {
        let a = ModelAnatomy::new(&moe_35b(), 0, None);
        let header_bytes: u64 = 35_707_978_496;
        assert!(a.total_bytes.abs_diff(header_bytes) * 1000 < header_bytes, "{} vs {header_bytes}", a.total_bytes);
        assert_eq!(a.moe.as_ref().unwrap().routed_bytes, 32_216_186_880);
        assert!(a.decode_bytes * 8 < a.total_bytes, "{} of {}", a.decode_bytes, a.total_bytes);
    }

    #[test]
    fn dense_models_have_no_moe_summary_and_decode_all_their_weights() {
        let a = ModelAnatomy::new(&fixture("qwen3.8-27b-exact.toml"), 0, None);
        assert!(a.moe.is_none());
        assert!(a.layers.iter().all(|l| !l.moe && l.active_params == l.params));
    }

    #[test]
    fn memory_grows_with_context_and_the_ceiling_follows_bandwidth() {
        let a = ModelAnatomy::new(&fixture("qwen3.8-27b-exact.toml"), 0, Some(65536));
        assert_eq!(a.max_context, 65536);
        assert_eq!(a.memory_at(2048) - a.memory_at(1024), a.kv_bytes_per_token * 1024);
        let tps = a.decode_tokens_per_second(a.decode_bytes as f64 * 40.0);
        assert!((tps - 40.0).abs() < 1e-9);
    }
}
