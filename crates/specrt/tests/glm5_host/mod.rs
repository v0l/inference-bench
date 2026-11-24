use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

pub struct Weights(HashMap<String, Vec<f32>>);

impl Weights {
    pub fn load(path: &Path) -> Weights {
        let bytes = std::fs::read(path).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let mut map = HashMap::new();
        for name in st.names() {
            let t = st.tensor(name).unwrap();
            let v: Vec<f32> = match t.dtype() {
                Dtype::F32 => t.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
                Dtype::BF16 => t.data().as_chunks::<2>().0.iter().map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16)).collect(),
                other => panic!("{name} is {other:?}"),
            };
            map.insert(name.to_string(), v);
        }
        Weights(map)
    }

    pub fn get(&self, name: &str) -> &[f32] {
        self.0.get(name).unwrap_or_else(|| panic!("{name} missing"))
    }
}

#[derive(Clone, Copy)]
pub struct Knobs {
    pub sinkhorn_iters: usize,
    pub tail: bool,
}

pub fn planes(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| {
        let hi = bf16(v);
        hi + bf16(v - hi)
    }).collect()
}

pub fn bf16(v: f32) -> f32 {
    let u = v.to_bits();
    f32::from_bits(((u + 0x7fff + ((u >> 16) & 1)) >> 16) << 16)
}

pub struct Config {
    pub hidden: usize,
    pub vocab: usize,
    pub layers: usize,
    pub streams: usize,
    pub hc_eps: f32,
    pub eps: f32,
    pub kinds: Vec<bool>,
    pub dense: Vec<bool>,
    pub kda_heads: usize,
    pub kda_dim: usize,
    pub taps: usize,
    pub lower: f32,
    pub heads: usize,
    pub nope: usize,
    pub v_dim: usize,
    pub kv_lora: usize,
    pub q_lora: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub index_topk: usize,
    pub kpool: usize,
    pub experts: usize,
    pub top_k: usize,
    pub routed_scale: f32,
    pub limit: f32,
    pub inter: usize,
    pub expert_inter: usize,
    pub knobs: Knobs,
}

impl Config {
    pub fn read(json: &str) -> Config {
        let v: Value = serde_json::from_str(json).unwrap();
        let t = &v["text_config"];
        let n = |k: &str| t[k].as_f64().unwrap_or_else(|| panic!("{k}"));
        let u = |k: &str| n(k) as usize;
        let la = &t["linear_attn_config"];
        let layers = u("num_hidden_layers");
        Config {
            hidden: u("hidden_size"),
            vocab: u("vocab_size"),
            layers,
            streams: u("hc_mult"),
            hc_eps: n("hc_eps") as f32,
            eps: n("rms_norm_eps") as f32,
            kinds: (0..layers).map(|i| t["layer_types"][i] == "linear_attention").collect(),
            dense: (0..layers).map(|i| t["mlp_layer_types"][i] == "dense").collect(),
            kda_heads: la["num_heads"].as_u64().unwrap() as usize,
            kda_dim: la["head_dim"].as_u64().unwrap() as usize,
            taps: la["short_conv_kernel_size"].as_u64().unwrap() as usize,
            lower: la["gate_lower_bound"].as_f64().unwrap() as f32,
            heads: u("num_attention_heads"),
            nope: u("qk_nope_head_dim"),
            v_dim: u("v_head_dim"),
            kv_lora: u("kv_lora_rank"),
            q_lora: u("q_lora_rank"),
            index_heads: u("index_n_heads"),
            index_dim: u("index_head_dim"),
            index_topk: u("index_topk"),
            kpool: u("index_kpool"),
            experts: u("n_routed_experts"),
            top_k: u("num_experts_per_tok"),
            routed_scale: n("routed_scaling_factor") as f32,
            limit: n("swiglu_limit") as f32,
            inter: u("intermediate_size"),
            expert_inter: u("moe_intermediate_size"),
            knobs: Knobs { sinkhorn_iters: u("hc_sinkhorn_iters"), tail: t["index_kpool_always_select_tail"].as_bool().unwrap() },
        }
    }
}

thread_local! {
    pub static PLANES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn matvec(w: &[f32], rows: usize, x: &[f32]) -> Vec<f32> {
    let rounded;
    let x = if PLANES.get() {
        rounded = planes(x);
        &rounded[..]
    } else {
        x
    };
    let cols = x.len();
    assert_eq!(w.len(), rows * cols);
    (0..rows).map(|r| w[r * cols..(r + 1) * cols].iter().zip(x).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>() as f32).collect()
}

pub fn rms(x: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len() as f64;
    let inv = (1.0 / (ms + f64::from(eps)).sqrt()) as f32;
    x.iter().enumerate().map(|(i, v)| v * inv * w.map_or(1.0, |w| w[i])).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

pub struct Mix {
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
    pub comb: Vec<f32>,
}

pub fn hyper_weights(c: &Config, fn_w: &[f32], base: &[f32], scale: &[f32], streams: &[Vec<f32>]) -> Mix {
    let n = c.streams;
    let flat: Vec<f32> = streams.iter().flatten().copied().collect();
    let m = matvec(fn_w, (2 + n) * n, &rms(&flat, None, c.eps));
    let pre = (0..n).map(|s| sigmoid(m[s] * scale[0] + base[s]) + c.hc_eps).collect();
    let post = (0..n).map(|s| 2.0 * sigmoid(m[n + s] * scale[1] + base[n + s])).collect();
    let mut comb = vec![0f32; n * n];
    for i in 0..n {
        let row: Vec<f32> = (0..n).map(|j| m[2 * n + i * n + j] * scale[2] + base[2 * n + i * n + j]).collect();
        let top = row.iter().copied().fold(f32::MIN, f32::max);
        let e: Vec<f32> = row.iter().map(|v| (v - top).exp()).collect();
        let s: f32 = e.iter().sum();
        for j in 0..n {
            comb[i * n + j] = e[j] / s + c.hc_eps;
        }
    }
    let cols = |comb: &mut Vec<f32>| {
        for j in 0..n {
            let s: f32 = (0..n).map(|i| comb[i * n + j]).sum::<f32>() + c.hc_eps;
            (0..n).for_each(|i| comb[i * n + j] /= s);
        }
    };
    let rows = |comb: &mut Vec<f32>| {
        for i in 0..n {
            let s: f32 = comb[i * n..(i + 1) * n].iter().sum::<f32>() + c.hc_eps;
            comb[i * n..(i + 1) * n].iter_mut().for_each(|v| *v /= s);
        }
    };
    cols(&mut comb);
    for _ in 1..c.knobs.sinkhorn_iters {
        rows(&mut comb);
        cols(&mut comb);
    }
    Mix { pre, post, comb }
}

pub fn collapse(mix: &Mix, streams: &[Vec<f32>]) -> Vec<f32> {
    let mut out = vec![0f32; streams[0].len()];
    for (s, st) in streams.iter().enumerate() {
        out.iter_mut().zip(st).for_each(|(o, v)| *o += mix.pre[s] * v);
    }
    out
}

pub fn expand(mix: &Mix, y: &[f32], streams: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = streams.len();
    (0..n)
        .map(|o| (0..y.len()).map(|d| mix.post[o] * y[d] + (0..n).map(|i| mix.comb[i * n + o] * streams[i][d]).sum::<f32>()).collect())
        .collect()
}

struct KdaState {
    history: Vec<Vec<f32>>,
    s: Vec<f32>,
}

struct DsaState {
    latent: Vec<Vec<f32>>,
    key: Vec<Vec<f32>>,
    gate: Vec<Vec<f32>>,
}

enum LayerState {
    Kda(KdaState),
    Dsa(DsaState),
}

pub struct Host<'a> {
    pub c: Config,
    w: &'a Weights,
    state: Vec<LayerState>,
    pos: usize,
}

const P: &str = "model.language_model";

impl<'a> Host<'a> {
    pub fn new(c: Config, w: &'a Weights) -> Host<'a> {
        let state = (0..c.layers)
            .map(|i| {
                if c.kinds[i] {
                    let qkv = c.kda_heads * c.kda_dim;
                    LayerState::Kda(KdaState { history: vec![vec![0.0; 3 * qkv]; c.taps - 1], s: vec![0.0; c.kda_heads * c.kda_dim * c.kda_dim] })
                } else {
                    LayerState::Dsa(DsaState { latent: vec![], key: vec![], gate: vec![] })
                }
            })
            .collect();
        Host { c, w, state, pos: 0 }
    }

    fn lw(&self, layer: usize, name: &str) -> &'a [f32] {
        self.w.get(&format!("{P}.layers.{layer}.{name}"))
    }

    pub fn step(&mut self, token: u32, trace: &mut Vec<Vec<Vec<f32>>>) -> Vec<f32> {
        let c = &self.c;
        let h = c.hidden;
        let x = self.w.get(&format!("{P}.embed_tokens.weight"))[token as usize * h..(token as usize + 1) * h].to_vec();
        let mut streams = vec![x; c.streams];
        for l in 0..c.layers {
            let mix = hyper_weights(&self.c, self.lw(l, "hc_attn_fn"), self.lw(l, "hc_attn_base"), self.lw(l, "hc_attn_scale"), &streams);
            let xn = rms(&collapse(&mix, &streams), Some(self.lw(l, "input_layernorm.weight")), self.c.eps);
            let y = if self.c.kinds[l] { self.kda(l, &xn) } else { self.dsa(l, &xn) };
            streams = expand(&mix, &y, &streams);
            let mix = hyper_weights(&self.c, self.lw(l, "hc_ffn_fn"), self.lw(l, "hc_ffn_base"), self.lw(l, "hc_ffn_scale"), &streams);
            let xn = rms(&collapse(&mix, &streams), Some(self.lw(l, "post_attention_layernorm.weight")), self.c.eps);
            let y = if self.c.dense[l] { self.mlp(l, "mlp", self.c.inter, &xn) } else { self.moe(l, &xn) };
            streams = expand(&mix, &y, &streams);
            trace.push(streams.clone());
        }
        self.pos += 1;
        let c = &self.c;
        let mean: Vec<f32> = (0..h).map(|d| streams.iter().map(|s| s[d]).sum::<f32>() / c.streams as f32).collect();
        let out = rms(&mean, Some(self.w.get(&format!("{P}.norm.weight"))), c.eps);
        matvec(self.w.get("lm_head.weight"), c.vocab, &out)
    }

    fn mlp(&self, l: usize, module: &str, inter: usize, x: &[f32]) -> Vec<f32> {
        let lim = self.c.limit;
        let g = matvec(self.lw(l, &format!("{module}.gate_proj.weight")), inter, x);
        let u = matvec(self.lw(l, &format!("{module}.up_proj.weight")), inter, x);
        let a: Vec<f32> = g.iter().zip(&u).map(|(g, u)| silu(g.min(lim)) * u.clamp(-lim, lim)).collect();
        matvec(self.lw(l, &format!("{module}.down_proj.weight")), self.c.hidden, &a)
    }

    fn moe(&self, l: usize, x: &[f32]) -> Vec<f32> {
        let c = &self.c;
        let logits = matvec(self.lw(l, "mlp.gate.weight"), c.experts, x);
        let scores: Vec<f32> = logits.iter().map(|v| sigmoid(*v)).collect();
        let bias = self.lw(l, "mlp.gate.e_score_correction_bias");
        let mut order: Vec<usize> = (0..c.experts).collect();
        order.sort_by(|a, b| (scores[*b] + bias[*b]).total_cmp(&(scores[*a] + bias[*a])));
        let chosen = &order[..c.top_k];
        let total: f32 = chosen.iter().map(|e| scores[*e]).sum::<f32>() + 1e-20;
        let mut out = self.mlp(l, "mlp.shared_experts", c.expert_inter, x);
        for e in chosen {
            let y = self.mlp(l, &format!("mlp.experts.{e}"), c.expert_inter, x);
            let wt = scores[*e] / total * c.routed_scale;
            out.iter_mut().zip(&y).for_each(|(o, v)| *o += wt * v);
        }
        out
    }

    fn kda(&mut self, l: usize, x: &[f32]) -> Vec<f32> {
        let (heads, d, taps, lower, eps, hidden) = (self.c.kda_heads, self.c.kda_dim, self.c.taps, self.c.lower, self.c.eps, self.c.hidden);
        let qkv = heads * d;
        let a = "self_attn";
        let mut pre = matvec(self.lw(l, &format!("{a}.q_proj.weight")), qkv, x);
        pre.extend(matvec(self.lw(l, &format!("{a}.k_proj.weight")), qkv, x));
        pre.extend(matvec(self.lw(l, &format!("{a}.v_proj.weight")), qkv, x));
        let convs: Vec<&[f32]> = ["q", "k", "v"].iter().map(|p| self.lw(l, &format!("{a}.{p}_conv1d.weight"))).collect();
        let f = matvec(self.lw(l, &format!("{a}.f_b_proj.weight")), qkv, &matvec(self.lw(l, &format!("{a}.f_a_proj.weight")), d, x));
        let gate = matvec(self.lw(l, &format!("{a}.g_b_proj.weight")), qkv, &matvec(self.lw(l, &format!("{a}.g_a_proj.weight")), d, x));
        let b = matvec(self.lw(l, &format!("{a}.b_proj.weight")), heads, x);
        let (dtb, a_log, onorm) = (self.lw(l, &format!("{a}.dt_bias")), self.lw(l, &format!("{a}.A_log")), self.lw(l, &format!("{a}.o_norm.weight")));
        let LayerState::Kda(st) = &mut self.state[l] else { unreachable!() };
        let mut mixed = vec![0f32; 3 * qkv];
        for ch in 0..3 * qkv {
            let w = &convs[ch / qkv][(ch % qkv) * taps..(ch % qkv + 1) * taps];
            let mut acc = w[taps - 1] * pre[ch];
            for j in 0..taps - 1 {
                acc += w[j] * st.history[j][ch];
            }
            mixed[ch] = silu(acc);
        }
        st.history.remove(0);
        st.history.push(pre);
        let l2 = |v: &[f32]| {
            let n = (v.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
            v.iter().map(|x| x / n).collect::<Vec<f32>>()
        };
        let mut out = vec![0f32; qkv];
        for hh in 0..heads {
            let seg = |o: usize| &mixed[o * qkv + hh * d..o * qkv + (hh + 1) * d];
            let q: Vec<f32> = l2(seg(0)).iter().map(|v| v / (d as f32).sqrt()).collect();
            let k = l2(seg(1));
            let v = seg(2);
            let beta = sigmoid(b[hh]);
            let s = &mut st.s[hh * d * d..(hh + 1) * d * d];
            for i in 0..d {
                let ch = hh * d + i;
                let g = (lower * sigmoid(a_log[hh].exp() * (f[ch] + dtb[ch]))).exp();
                s[i * d..(i + 1) * d].iter_mut().for_each(|v| *v *= g);
            }
            for j in 0..d {
                let mem: f32 = (0..d).map(|i| s[i * d + j] * k[i]).sum();
                let delta = (v[j] - mem) * beta;
                (0..d).for_each(|i| s[i * d + j] += k[i] * delta);
            }
            let o: Vec<f32> = (0..d).map(|j| (0..d).map(|i| s[i * d + j] * q[i]).sum()).collect();
            let n = rms(&o, Some(onorm), eps);
            for j in 0..d {
                out[hh * d + j] = n[j] * sigmoid(gate[hh * d + j]);
            }
        }
        matvec(self.lw(l, &format!("{a}.o_proj.weight")), hidden, &out)
    }

    fn dsa(&mut self, l: usize, x: &[f32]) -> Vec<f32> {
        let c = &self.c;
        let (heads, nope, vd, lora, eps) = (c.heads, c.nope, c.v_dim, c.kv_lora, c.eps);
        let (ih, id, kpool) = (c.index_heads, c.index_dim, c.kpool);
        let a = "self_attn";
        let qr = rms(&matvec(self.lw(l, &format!("{a}.q_a_proj.weight")), c.q_lora, x), Some(self.lw(l, &format!("{a}.q_a_layernorm.weight"))), eps);
        let q = matvec(self.lw(l, &format!("{a}.q_b_proj.weight")), heads * nope, &qr);
        let lat = rms(&matvec(self.lw(l, &format!("{a}.kv_a_proj_with_mqa.weight")), lora, x), Some(self.lw(l, &format!("{a}.kv_a_layernorm.weight"))), eps);
        let iq = matvec(self.lw(l, &format!("{a}.indexer.wq_b.weight")), ih * id, &qr);
        let raw = matvec(self.lw(l, &format!("{a}.indexer.wk.weight")), id, x);
        let mean = raw.iter().sum::<f32>() / id as f32;
        let var = raw.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / id as f32;
        let (kw, kb) = (self.lw(l, &format!("{a}.indexer.k_norm.weight")), self.lw(l, &format!("{a}.indexer.k_norm.bias")));
        let key: Vec<f32> = (0..id).map(|i| (raw[i] - mean) / (var + 1e-6).sqrt() * kw[i] + kb[i]).collect();
        let gate = matvec(self.lw(l, &format!("{a}.indexer.index_kpool_compress_gate")), id, x);
        let wts: Vec<f32> = matvec(self.lw(l, &format!("{a}.indexer.weights_proj.weight")), ih, x).iter().map(|v| v * (ih as f32).powf(-0.5)).collect();
        let ape = self.lw(l, &format!("{a}.indexer.index_kpool_compress_ape"));
        let kv_b = self.lw(l, &format!("{a}.kv_b_proj.weight"));
        let o_w = self.lw(l, &format!("{a}.o_proj.weight"));
        let (topk, tail, hidden) = (c.index_topk, c.knobs.tail, c.hidden);
        let LayerState::Dsa(st) = &mut self.state[l] else { unreachable!() };
        st.latent.push(lat);
        st.key.push(key);
        st.gate.push(gate);
        let n = st.latent.len();
        let pools = n / kpool;
        let scale = (id as f32).powf(-0.5);
        let mut scored: Vec<(f32, usize)> = (0..pools)
            .map(|p| {
                let mut pk = vec![0f32; id];
                for (ch, slot) in pk.iter_mut().enumerate() {
                    let logit: Vec<f32> = (0..kpool).map(|j| st.gate[p * kpool + j][ch] + ape[j * id + ch]).collect();
                    let top = logit.iter().copied().fold(f32::MIN, f32::max);
                    let e: Vec<f32> = logit.iter().map(|v| (v - top).exp()).collect();
                    let s: f32 = e.iter().sum();
                    *slot = (0..kpool).map(|j| e[j] / s * st.key[p * kpool + j][ch]).sum();
                }
                let score: f32 = (0..ih).map(|hh| wts[hh] * (scale * iq[hh * id..(hh + 1) * id].iter().zip(&pk).map(|(a, b)| a * b).sum::<f32>()).max(0.0)).sum();
                (score, p)
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut chosen: Vec<usize> = scored.iter().take(topk / kpool).flat_map(|(_, p)| (p * kpool..(p + 1) * kpool).collect::<Vec<_>>()).collect();
        if tail {
            chosen.extend(pools * kpool..n);
        }
        let mut out = vec![0f32; heads * vd];
        let sm = (nope as f32).powf(-0.5);
        for hh in 0..heads {
            let kw = &kv_b[hh * (nope + vd) * lora..(hh * (nope + vd) + nope) * lora];
            let vw = &kv_b[(hh * (nope + vd) + nope) * lora..(hh + 1) * (nope + vd) * lora];
            let qh = &q[hh * nope..(hh + 1) * nope];
            let logits: Vec<f32> = chosen.iter().map(|t| matvec(kw, nope, &st.latent[*t]).iter().zip(qh).map(|(a, b)| a * b).sum::<f32>() * sm).collect();
            let top = logits.iter().copied().fold(f32::MIN, f32::max);
            let e: Vec<f32> = logits.iter().map(|v| (v - top).exp()).collect();
            let s: f32 = e.iter().sum();
            for (i, t) in chosen.iter().enumerate() {
                let v = matvec(vw, vd, &st.latent[*t]);
                (0..vd).for_each(|j| out[hh * vd + j] += e[i] / s * v[j]);
            }
        }
        matvec(o_w, hidden, &out)
    }
}
