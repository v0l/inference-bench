use serde_json::Value;
use specrt::spec::{Block, Spec};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/haru")
}

fn json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn real_config() -> Value {
    json(&dir().join("haru-4b.config.json"))
}

fn tiny_config() -> Value {
    json(&dir().join("tiny/config.json"))
}

fn header() -> Vec<(String, Vec<usize>)> {
    let rows: Vec<(String, String, Vec<usize>)> = serde_json::from_str(&std::fs::read_to_string(dir().join("haru-4b.header.json")).unwrap()).unwrap();
    rows.into_iter().map(|(n, dtype, s)| {
        assert_eq!(dtype, "BF16", "{n}");
        (n, s)
    }).collect()
}

fn reference(key: &str) -> (Vec<f32>, Vec<usize>) {
    let meta = json(&dir().join("tiny/reference.json"));
    let entry = &meta["files"][key];
    let shape: Vec<usize> = entry["shape"].as_array().unwrap_or_else(|| panic!("{key}")).iter().map(|d| d.as_u64().unwrap() as usize).collect();
    let bytes = std::fs::read(dir().join("tiny/reference").join(entry["file"].as_str().unwrap())).unwrap();
    let values: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    assert_eq!(values.len(), shape.iter().product::<usize>(), "{key}");
    (values, shape)
}

struct Weights {
    tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Weights {
    fn load() -> Weights {
        let bytes = std::fs::read(dir().join("tiny/model.safetensors")).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let mut tensors = BTreeMap::new();
        for (name, view) in st.tensors() {
            assert_eq!(view.dtype(), safetensors::Dtype::BF16, "{name}");
            let data = view.data().as_chunks::<2>().0.iter().map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16)).collect();
            tensors.insert(name.to_string(), (view.shape().to_vec(), data));
        }
        Weights { tensors }
    }

    fn get(&self, name: &str) -> &[f32] {
        &self.tensors.get(name).unwrap_or_else(|| panic!("{name} missing")).1
    }

    fn shape(&self, name: &str) -> &[usize] {
        &self.tensors.get(name).unwrap_or_else(|| panic!("{name} missing")).0
    }

    fn linear(&self, prefix: &str, x: &[f32]) -> Vec<f32> {
        let shape = self.shape(&format!("{prefix}.weight"));
        let (out, inp) = (shape[0], shape[1]);
        assert_eq!(x.len(), inp, "{prefix}");
        let w = self.get(&format!("{prefix}.weight"));
        let bias = self.tensors.get(&format!("{prefix}.bias")).map(|t| t.1.as_slice());
        (0..out).map(|o| w[o * inp..(o + 1) * inp].iter().zip(x).map(|(a, b)| a * b).sum::<f32>() + bias.map_or(0.0, |b| b[o])).collect()
    }
}

fn rms_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter().map(|v| v * r).collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) / scale
}

fn normalised(name: &str) -> String {
    name.split('.').map(|p| if p.parse::<usize>().is_ok() { "N" } else { p }).collect::<Vec<_>>().join(".")
}

fn embed(w: &Weights, config: &Value, token: usize, model: Option<&[f32]>, user: Option<&[f32]>) -> Vec<f32> {
    let hidden = config["lm"]["hidden_size"].as_u64().unwrap() as usize;
    let scale = w.get("latent_scale")[0];
    let bias = w.get("latent_bias")[0];
    let mut x = w.get("lm.embed_tokens.weight")[token * hidden..(token + 1) * hidden].to_vec();
    for (proj, latent) in [("model_audio_proj", model), ("user_audio_proj", user)] {
        let Some(latent) = latent else { continue };
        let scaled: Vec<f32> = latent.iter().map(|v| (v + bias) * scale).collect();
        let h = w.linear(&format!("{proj}.fc1"), &scaled);
        let norm = w.get(&format!("{proj}.norm.weight"));
        let h: Vec<f32> = rms_norm(&h, 1e-6).iter().zip(norm).map(|(a, g)| a * g).collect();
        for (xi, yi) in x.iter_mut().zip(w.linear(&format!("{proj}.fc2"), &h)) {
            *xi += yi;
        }
    }
    x
}

fn scalar_embed(w: &Weights, prefix: &str, value: f32) -> Vec<f32> {
    let half = 128;
    let features: Vec<f32> = (0..2 * half)
        .map(|i| {
            let f = (-(10000f32.ln()) * (i % half) as f32 / half as f32).exp();
            if i < half { (value * f).cos() } else { (value * f).sin() }
        })
        .collect();
    let h: Vec<f32> = w.linear(&format!("{prefix}.fc1"), &features).into_iter().map(silu).collect();
    w.linear(&format!("{prefix}.fc2"), &h)
}

fn adaptive(x: &[f32], eps: f32, weight: Option<&[f32]>, shift: &[f32], scale: &[f32]) -> Vec<f32> {
    rms_norm(x, eps).iter().enumerate().map(|(i, v)| v * weight.map_or(1.0, |g| g[i]) * (1.0 + scale[i]) + shift[i]).collect()
}

fn meanflow(w: &Weights, config: &Value, hidden: &[f32], noise: &[f32], guidance: f32, steps: usize) -> Vec<f32> {
    let head = &config["head"];
    let eps = head["norm_eps"].as_f64().unwrap() as f32;
    let interval = head["interval_embed"].as_f64().unwrap() as f32;
    let layers = head["num_layers"].as_u64().unwrap() as usize;
    let h = 1.0 / steps as f32;
    let cond = w.linear("head.cond_proj", hidden);
    let mut z = noise.to_vec();
    for step in 0..steps {
        let mut c = cond.clone();
        for (name, value) in [("t_embed", step as f32 * h), ("dt_embed", interval), ("h_embed", h), ("omega_embed", 1.0 - 1.0 / guidance)] {
            for (ci, e) in c.iter_mut().zip(scalar_embed(w, &format!("head.{name}"), value)) {
                *ci += e;
            }
        }
        let c: Vec<f32> = c.into_iter().map(silu).collect();
        let mut x = w.linear("head.latent_proj", &z);
        let n = x.len();
        for l in 0..layers {
            let ada = w.linear(&format!("head.layers.{l}.ada"), &c);
            let (shift, scale, gate) = (&ada[..n], &ada[n..2 * n], &ada[2 * n..]);
            let y = adaptive(&x, eps, Some(w.get(&format!("head.layers.{l}.norm.weight"))), shift, scale);
            let g = w.linear(&format!("head.layers.{l}.ffn.gate_proj"), &y);
            let u = w.linear(&format!("head.layers.{l}.ffn.up_proj"), &y);
            let act: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();
            let down = w.linear(&format!("head.layers.{l}.ffn.down_proj"), &act);
            for i in 0..n {
                x[i] += gate[i] * down[i];
            }
        }
        let ada = w.linear("head.final.ada", &c);
        let y = adaptive(&x, eps, None, &ada[..n], &ada[n..]);
        let v = w.linear("head.final.proj", &y);
        for (zi, vi) in z.iter_mut().zip(v) {
            *zi += h * vi;
        }
    }
    let scale = w.get("latent_scale")[0];
    let bias = w.get("latent_bias")[0];
    z.iter().map(|v| v / scale - bias).collect()
}

#[test]
fn the_real_checkpoint_has_the_five_parts_and_the_card_sizes() {
    let rows = header();
    let mut parts: BTreeMap<String, u64> = BTreeMap::new();
    for (name, shape) in &rows {
        let part = if name.starts_with("codec.") { name.split('.').take(2).collect::<Vec<_>>().join(".") } else { name.split('.').next().unwrap().to_string() };
        *parts.entry(part).or_default() += shape.iter().product::<usize>() as u64;
    }
    assert_eq!(parts["lm"], 4_205_751_296);
    assert_eq!(parts["head"], 363_407_360);
    assert_eq!(parts["codec.encoder"], 343_696_032);
    assert_eq!(parts["codec.decoder"], 343_695_969);
    assert_eq!(parts["model_audio_proj"], 6_725_120);
    assert_eq!(parts["user_audio_proj"], 6_725_120);
    assert_eq!(parts["latent_scale"] + parts["latent_bias"], 2);
    assert_eq!(parts.values().sum::<u64>(), 5_270_000_899);
    let shape = |n: &str| rows.iter().find(|r| r.0 == n).unwrap_or_else(|| panic!("{n}")).1.clone();
    assert_eq!(shape("lm.embed_tokens.weight"), [248320, 2560]);
    assert!(!rows.iter().any(|r| r.0.starts_with("lm.lm_head") || r.0 == "lm_head.weight"), "the text head is tied");
    assert_eq!(shape("user_audio_proj.fc1.weight"), [2560, 64]);
    assert_eq!(shape("model_audio_proj.fc2.weight"), [2560, 2560]);
    assert_eq!(shape("head.layers.3.ada.weight"), [7680, 2560]);
    assert_eq!(shape("head.layers.0.ffn.gate_proj.weight"), [7680, 2560]);
    assert_eq!(shape("head.final.ada.weight"), [5120, 2560]);
    assert_eq!(shape("head.final.proj.weight"), [64, 2560]);
    assert_eq!(shape("codec.encoder.stem.weight"), [32, 1, 7]);
    assert_eq!(shape("codec.encoder.head.weight"), [64, 2048, 7]);
    assert_eq!(shape("codec.decoder.stem.weight"), [2048, 64, 7]);
    assert_eq!(shape("codec.decoder.up.0.weight"), [2048, 1024, 16]);
    assert_eq!(shape("codec.decoder.head.weight"), [1, 32, 7]);
    assert_eq!(shape("codec.encoder.stages.6.7.fc1.weight"), [8192, 2048]);
    assert_eq!(shape("lm.layers.0.linear_attn.in_proj_qkv.weight"), [8192, 2560]);
    assert_eq!(shape("lm.layers.3.self_attn.q_proj.weight"), [8192, 2560]);
}

#[test]
fn the_backbone_tensors_are_the_qwen3_5_text_layout_under_lm() {
    let rows = header();
    let lm: BTreeSet<String> = rows.iter().filter_map(|(n, _)| n.strip_prefix("lm.")).map(normalised).collect();
    let expected: BTreeSet<String> = [
        "embed_tokens.weight",
        "norm.weight",
        "layers.N.input_layernorm.weight",
        "layers.N.post_attention_layernorm.weight",
        "layers.N.mlp.gate_proj.weight",
        "layers.N.mlp.up_proj.weight",
        "layers.N.mlp.down_proj.weight",
        "layers.N.linear_attn.A_log",
        "layers.N.linear_attn.conv1d.weight",
        "layers.N.linear_attn.dt_bias",
        "layers.N.linear_attn.in_proj_a.weight",
        "layers.N.linear_attn.in_proj_b.weight",
        "layers.N.linear_attn.in_proj_qkv.weight",
        "layers.N.linear_attn.in_proj_z.weight",
        "layers.N.linear_attn.norm.weight",
        "layers.N.linear_attn.out_proj.weight",
        "layers.N.self_attn.q_proj.weight",
        "layers.N.self_attn.k_proj.weight",
        "layers.N.self_attn.v_proj.weight",
        "layers.N.self_attn.o_proj.weight",
        "layers.N.self_attn.q_norm.weight",
        "layers.N.self_attn.k_norm.weight",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    assert_eq!(lm, expected);
    let full = rows.iter().filter(|(n, _)| n.ends_with("self_attn.q_proj.weight")).count();
    let linear = rows.iter().filter(|(n, _)| n.ends_with("linear_attn.out_proj.weight")).count();
    assert_eq!((full, linear), (8, 24));
}

#[test]
fn the_lm_sub_config_is_a_qwen3_5_spec_the_engine_accepts_and_the_wrapper_is_refused() {
    for config in [real_config(), tiny_config()] {
        let lm = &config["lm"];
        let spec = Spec::from_hf_config(&lm.to_string()).unwrap_or_else(|e| panic!("{e:#}"));
        assert_eq!(spec.dims.hidden, lm["hidden_size"].as_u64().unwrap() as usize);
        assert_eq!(spec.layers.count, lm["num_hidden_layers"].as_u64().unwrap() as usize);
        assert!(spec.dims.tie_embeddings);
        assert!(spec.linear_attention.is_some());
        let full = spec.layers.pattern.iter().filter(|b| !b.is_linear()).count();
        assert_eq!(full * 4, spec.layers.pattern.len());
        assert_eq!(*spec.layers.pattern.last().unwrap(), Block::Full);
        assert!(spec.layers.pattern.iter().all(|b| b.is_linear() || *b == Block::Full));
        assert!(Spec::from_hf_config(&config.to_string()).is_err(), "model_type haru has no family yet");
    }
    let real = Spec::from_hf_config(&real_config()["lm"].to_string()).unwrap();
    assert_eq!((real.dims.hidden, real.dims.vocab, real.layers.count, real.attention.head_dim), (2560, 248320, 32, 256));
}

#[test]
fn the_tiny_fixture_has_every_real_tensor_kind_and_no_other() {
    let real: BTreeSet<String> = header().into_iter().map(|(n, _)| normalised(&n)).collect();
    let w = Weights::load();
    let tiny: BTreeSet<String> = w.tensors.keys().map(|n| normalised(n)).collect();
    assert_eq!(tiny, real);
}

#[test]
fn the_three_streams_sum_into_one_position() {
    let config = tiny_config();
    let w = Weights::load();
    let (tokens, _) = reference("prompt.tokens");
    let (model, _) = reference("prompt.model_latents");
    let (user, _) = reference("prompt.user_latents");
    let (model_mask, _) = reference("prompt.model_mask");
    let (user_mask, _) = reference("prompt.user_mask");
    let (want, shape) = reference("prompt.embeds");
    let d = 64;
    let mut got = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        let m = (model_mask[i] > 0.5).then(|| &model[i * d..(i + 1) * d]);
        let u = (user_mask[i] > 0.5).then(|| &user[i * d..(i + 1) * d]);
        got.extend(embed(&w, &config, *t as usize, m, u));
    }
    assert_eq!(shape, [tokens.len(), 64]);
    assert!(max_rel(&got, &want) < 1e-5, "{}", max_rel(&got, &want));

    let (session_tokens, _) = reference("session.token");
    let (latents, _) = reference("session.latent");
    let (users, _) = reference("session.user_latent");
    let (means, _) = reference("session.user_mean");
    let (noise, _) = reference("session.user_noise");
    let (scales, _) = reference("session.user_scale");
    let latent_std = config["codec"]["latent_std"].as_f64().unwrap() as f32;
    let drawn: Vec<f32> = means.iter().zip(&noise).enumerate().map(|(i, (m, n))| m + scales[i / d] * (latent_std / 0.8) * n).collect();
    assert!(max_rel(&drawn, &users) < 1e-6, "user latents are the encoder mean plus scaled noise");
    let (want, _) = reference("session.embeds");
    let mut got = Vec::new();
    for i in 0..users.len() / d {
        got.extend(embed(&w, &config, session_tokens[i] as usize, Some(&latents[i * d..(i + 1) * d]), Some(&users[i * d..(i + 1) * d])));
    }
    assert!(max_rel(&got, &want) < 1e-5, "{}", max_rel(&got, &want));
}

#[test]
fn text_logits_are_the_hidden_state_against_the_tied_embedding() {
    let w = Weights::load();
    let (hidden, shape) = reference("session.hidden");
    let (want, _) = reference("session.logits");
    let (tokens, _) = reference("session.token");
    let (h, vocab) = (shape[1], 512);
    let table = w.get("lm.embed_tokens.weight");
    let mut got = Vec::new();
    for row in hidden.chunks(h) {
        got.extend((0..vocab).map(|v| table[v * h..(v + 1) * h].iter().zip(row).map(|(a, b)| a * b).sum::<f32>()));
    }
    assert!(max_rel(&got, &want) < 1e-5, "{}", max_rel(&got, &want));
    for (i, row) in want.chunks(vocab).enumerate() {
        let best = row.iter().enumerate().fold((0, f32::MIN), |b, (j, v)| if *v > b.1 { (j, *v) } else { b }).0;
        assert_eq!(best, tokens[i] as usize, "greedy token of frame {i}");
    }
}

#[test]
fn the_meanflow_head_is_one_mlp_pass_per_frame() {
    let config = tiny_config();
    let w = Weights::load();
    let (hidden, shape) = reference("session.hidden");
    let (noise, _) = reference("session.head_noise");
    let (h, d) = (shape[1], 64);
    let frames = shape[0];
    let run = |guidance: f32, steps: usize| -> Vec<f32> {
        (0..frames).flat_map(|i| meanflow(&w, &config, &hidden[i * h..(i + 1) * h], &noise[i * d..(i + 1) * d], guidance, steps)).collect()
    };
    let (want, _) = reference("session.latent");
    let got = run(1.5, 1);
    assert!(max_rel(&got, &want) < 1e-4, "{}", max_rel(&got, &want));
    for (key, guidance, steps) in [("head.guidance1.0", 1.0, 1), ("head.guidance2.0", 2.0, 1), ("head.steps2", 1.5, 2)] {
        let (want, _) = reference(key);
        let got = run(guidance, steps);
        assert!(max_rel(&got, &want) < 1e-4, "{key} {}", max_rel(&got, &want));
    }
    let (g1, _) = reference("head.guidance1.0");
    let (g2, _) = reference("head.guidance2.0");
    assert!(max_rel(&g1, &g2) > 1e-4, "the guidance scale is an input of the head");
}

fn codec_state_floats(codec: &Value) -> (usize, usize) {
    let c = codec["base_channels"].as_u64().unwrap() as usize;
    let latent = codec["latent_dim"].as_u64().unwrap() as usize;
    let ratios: Vec<usize> = codec["ratios"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let depths = |k: &str| -> Vec<usize> { codec[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect() };
    let n = ratios.len();
    let block = |ch: usize| ch * 6;
    let mut encoder = 6;
    for (i, d) in depths("encoder_depths").iter().enumerate() {
        if i > 0 {
            let s = ratios[n - i];
            encoder += (c << (i - 1)) * s;
        }
        encoder += d * block(c << i);
    }
    encoder += (c << n) * 6;
    let mut decoder = latent * 6;
    for (i, d) in depths("decoder_depths").iter().enumerate() {
        if i > 0 {
            decoder += (c << (n - i)) * ratios[i - 1];
        }
        decoder += d * block(c << (n - i));
    }
    decoder += c * 6;
    (encoder, decoder)
}

#[test]
fn the_codec_streaming_state_is_a_fixed_tail_per_layer() {
    let meta = json(&dir().join("tiny/reference.json"));
    let (enc, dec) = codec_state_floats(&tiny_config()["codec"]);
    assert_eq!(meta["codec_state_floats"]["encoder"].as_u64().unwrap() as usize, enc);
    assert_eq!(meta["codec_state_floats"]["decoder"].as_u64().unwrap() as usize, dec);
    let (enc, dec) = codec_state_floats(&real_config()["codec"]);
    assert_eq!((enc, dec), (159_622, 147_904));
    for (k, v) in meta["checks"].as_object().unwrap() {
        assert!(v.as_f64().unwrap() < 1e-4, "{k} {v}");
    }
    let frame: usize = real_config()["codec"]["ratios"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).product();
    assert_eq!(frame, real_config()["frame_samples"].as_u64().unwrap() as usize);
}

#[test]
fn the_fixture_stays_under_3_mb() {
    fn size(p: &Path) -> u64 {
        std::fs::read_dir(p).unwrap().flatten().map(|e| if e.path().is_dir() { size(&e.path()) } else { e.metadata().unwrap().len() }).sum()
    }
    let total = size(&dir());
    assert!(total < 3 * 1024 * 1024, "{total}");
}
