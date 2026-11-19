use crate::{Access, Kernel, Language, Param, Role, Scalar, Target};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeStyle {
    Halves,
    Pairs,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RowBody {
    SpanRope { row: u32, head: u32, offset: u32, rot: u32, style: RopeStyle, invert: bool },
    HeadRms { row: u32, head: u32, offset: u32, width: u32, eps: f32 },
    BlockPool { dim: u32, rot: u32, layers: u32, raw_width: u32, gate_offset: u32, eps: f32 },
    StateWrite { src_row: u32, src_offset: u32, layers: u32, rows: u32, width: u32, stride: u32 },
    IndexTopk { heads: u32, dim: u32, topk: u32, max_keys: u32, q_row: u32, q_offset: u32, w_row: u32, w_offset: u32, out_row: u32, layers: u32 },
    LayerNorm { width: u32, row: u32, head: u32, offset: u32, eps: f32 },
    HeadMatvec { x_row: u32, x_head: u32, x_offset: u32, inner: u32, w_head: u32, w_inner: u32, w_out: u32, w_offset: u32, out: u32, y_row: u32, y_head: u32, y_offset: u32 },
}

const POSITION: [(&str, Scalar, Role, Access); 4] = [
    ("pos", Scalar::I32, Role::Activation, Access::Read),
    ("layer", Scalar::I32, Role::Value, Access::Read),
    ("pos_step", Scalar::I32, Role::Value, Access::Read),
    ("row_step", Scalar::I32, Role::Value, Access::Read),
];

const PAGED: [(&str, Scalar, Role, Access); 3] = [
    ("block_table", Scalar::I32, Role::Activation, Access::Read),
    ("slots", Scalar::I32, Role::Activation, Access::Read),
    ("pages", Scalar::I32, Role::Value, Access::Read),
];

impl RowBody {
    pub fn family(&self) -> &'static str {
        match self {
            RowBody::SpanRope { .. } => "span_rope",
            RowBody::StateWrite { .. } => "state_write",
            RowBody::HeadMatvec { .. } => "head_matvec",
            RowBody::IndexTopk { .. } => "index_topk",
            RowBody::LayerNorm { .. } => "layernorm_segment",
            RowBody::HeadRms { .. } => "head_rms",
            RowBody::BlockPool { .. } => "block_pool",
        }
    }

    pub fn describe(&self) -> String {
        format!("{self:?}")
    }

    fn planes(params: &[Param]) -> bool {
        params.iter().any(|p| p.name == "out_hi")
    }

    fn required(&self, params: &[Param]) -> Vec<(&'static str, Scalar, Role, Access)> {
        let mut out = Vec::new();
        match self {
            RowBody::SpanRope { .. } => {
                out.push(("x", Scalar::F32, Role::Activation, Access::Write));
                out.extend(POSITION);
            }
            RowBody::StateWrite { .. } => {
                out.push(("src", Scalar::F32, Role::Activation, Access::Read));
                out.push(("dst", Scalar::Bf16, Role::Activation, Access::Write));
                out.extend(POSITION);
                out.extend(PAGED);
            }
            RowBody::IndexTopk { .. } => {
                out.push(("q", Scalar::F32, Role::Activation, Access::Read));
                out.push(("w", Scalar::F32, Role::Activation, Access::Read));
                out.push(("keys", Scalar::Bf16, Role::Activation, Access::Read));
                out.push(("idx", Scalar::I32, Role::Activation, Access::Write));
                out.extend(POSITION);
                out.extend(PAGED);
                out.push(("scale", Scalar::F32, Role::Value, Access::Read));
                out.push(("w_scale", Scalar::F32, Role::Value, Access::Read));
                if params.iter().any(|p| p.name == "ratio") {
                    out.push(("ratio", Scalar::I32, Role::Value, Access::Read));
                }
            }
            RowBody::HeadRms { .. } => {
                out.push(("x", Scalar::F32, Role::Activation, Access::Write));
            }
            RowBody::BlockPool { .. } => {
                out.push(("raw", Scalar::Bf16, Role::Activation, Access::Read));
                out.push(("bias", Scalar::F32, Role::Weight, Access::Read));
                out.push(("norm", Scalar::F32, Role::Weight, Access::Read));
                out.push(("dst", Scalar::Bf16, Role::Activation, Access::Write));
                out.extend(POSITION);
                out.extend(PAGED);
                out.push(("ratio", Scalar::I32, Role::Value, Access::Read));
                out.push(("overlap", Scalar::I32, Role::Value, Access::Read));
            }
            RowBody::LayerNorm { .. } => {
                out.push(("x", Scalar::F32, Role::Activation, Access::Write));
                out.push(("w", Scalar::F32, Role::Weight, Access::Read));
                out.push(("b", Scalar::F32, Role::Weight, Access::Read));
            }
            RowBody::HeadMatvec { .. } => {
                out.push(("x", Scalar::F32, Role::Activation, Access::Read));
                out.push(("w", Scalar::F32, Role::Weight, Access::Read));
                if Self::planes(params) {
                    out.push(("out_hi", Scalar::Bf16, Role::Activation, Access::Write));
                    out.push(("out_lo", Scalar::Bf16, Role::Activation, Access::Write));
                } else {
                    out.push(("y", Scalar::F32, Role::Activation, Access::Write));
                }
            }
        }
        out
    }

    pub fn check(&self, t: &Target, params: &[Param]) -> Result<(), String> {
        if t.language != Language::Cuda {
            return Err(format!("a {} body has only a CUDA lowering", self.family()));
        }
        let want = self.required(params);
        for (name, ty, role, access) in &want {
            match params.iter().find(|p| p.name == *name) {
                Some(p) if p.ty == *ty && p.role == *role && p.access == *access => {}
                Some(p) => return Err(format!("{} param {name:?} must be {ty:?} {role:?} {access:?}, not {:?} {:?} {:?}", self.family(), p.ty, p.role, p.access)),
                None => return Err(format!("{} needs a param named {name:?}", self.family())),
            }
        }
        if params.len() != want.len() {
            return Err(format!("{} takes exactly the params {:?}", self.family(), want.iter().map(|w| w.0).collect::<Vec<_>>()));
        }
        match self {
            RowBody::SpanRope { rot, head, .. } if *rot == 0 || !rot.is_multiple_of(2) || rot > head => Err("span_rope needs an even rot no wider than the head".into()),
            RowBody::StateWrite { width, rows, layers, stride, .. } if *width == 0 || *rows == 0 || *layers == 0 || stride < width => {
                Err("state_write needs a positive width, rows and layers and a stride of at least the width".into())
            }
            RowBody::HeadRms { width, eps, .. } if *width == 0 || !eps.is_finite() || *eps < 0.0 => Err("head_rms needs a positive width and a finite eps".into()),
            RowBody::BlockPool { dim, rot, layers, eps, .. } if *dim == 0 || *layers == 0 || *rot == 0 || !rot.is_multiple_of(2) || rot > dim || !eps.is_finite() || *eps < 0.0 => {
                Err("block_pool needs a positive dim and layers, an even rot no wider than dim and a finite eps".into())
            }
            RowBody::BlockPool { dim, .. } if *dim > 1024 => Err("block_pool rows live in shared memory, at most 1024 wide".into()),
            RowBody::IndexTopk { heads, dim, topk, max_keys, .. } if *heads == 0 || *dim == 0 || *topk == 0 || *max_keys == 0 => Err("index_topk needs positive heads, dim, topk and max_keys".into()),
            RowBody::IndexTopk { topk, max_keys, .. } if topk > max_keys => Err("index_topk cannot select more keys than it scores".into()),
            RowBody::IndexTopk { max_keys, .. } if *max_keys > 12288 => Err("index_topk scores live in shared memory, at most 12288 keys".into()),
            RowBody::LayerNorm { width, eps, .. } if *width == 0 || !eps.is_finite() || *eps < 0.0 => Err("layernorm_segment needs a positive width and a finite eps".into()),
            RowBody::HeadMatvec { inner, out, .. } if *inner == 0 || *out == 0 => Err("head_matvec needs positive inner and out sizes".into()),
            _ => Ok(()),
        }
    }

    pub fn emit_cuda(&self, k: &Kernel) -> Result<String, String> {
        let sig: Vec<String> = k
            .params
            .iter()
            .map(|p| {
                let ty = match p.ty {
                    Scalar::F32 => "float",
                    Scalar::U32 => "unsigned",
                    Scalar::Bf16 => "bf16_t",
                    Scalar::I32 => "int",
                };
                match (p.role, p.access) {
                    (Role::Value, _) => format!("{ty} {}", p.name),
                    (_, Access::Read) => format!("const {ty}* {}", p.name),
                    (_, Access::Write) => format!("{ty}* {}", p.name),
                }
            })
            .collect();
        let head = format!("extern \"C\" __global__ void __launch_bounds__({}) {}({}) {{\n", k.threads, k.name, sig.join(", "));
        let body = match self {
            RowBody::SpanRope { row, head: stride, offset, rot, style, invert } => {
                let (a, b) = match style {
                    RopeStyle::Halves => ("i", "i + ROT_SPAN / 2"),
                    RopeStyle::Pairs => ("2 * i", "2 * i + 1"),
                };
                format!(
                    "    constexpr int ROT_SPAN = {rot};
    const int row = blockIdx.y;
    const int p = pos[row * pos_step] + row * row_step;
    float* v = x + (size_t)row * {row} + (size_t)blockIdx.x * {stride} + {offset};
    for (int i = threadIdx.x; i < ROT_SPAN / 2; i += blockDim.x) {{
        float angle = (float)p * rope_inv_freq(i, layer);
        float c = cosf(angle) * ROPE_MSCALE;
        float s = {sign}sinf(angle) * ROPE_MSCALE;
        float a = v[{a}];
        float b = v[{b}];
        v[{a}] = a * c - b * s;
        v[{b}] = b * c + a * s;
    }}
",
                    sign = if *invert { "-" } else { "" }
                )
            }
            RowBody::StateWrite { src_row, src_offset, layers, rows, width, stride } => format!(
                "    const int row = blockIdx.y;
    const int r = blockIdx.x;
    const int p = pos[row * pos_step] + row * row_step;
    const int* table = seq_table(block_table, slots, row, pages);
    size_t slot = (((size_t)table[p / PAGE] * {layers} + layer) * {rows} + r) * PAGE + p % PAGE;
    bf16_t* out = dst + slot * {stride} * KV_PLANES;
    const float* in = src + (size_t)row * {src_row} + {src_offset} + (size_t)r * {width};
    for (int d = threadIdx.x; d < {width}; d += blockDim.x) {{
        unsigned hi, lo;
        attn_split(in[d], hi, lo);
        out[d] = (bf16_t)hi;
        if (KV_PLANES == 2) out[{stride} + d] = (bf16_t)lo;
    }}
"
            ),
            RowBody::LayerNorm { width, row, head, offset, eps } => format!(
                "    constexpr int WIDTH = {width};
    __shared__ float scratch[32];
    float* v = x + (size_t)blockIdx.y * {row} + (size_t)blockIdx.x * {head} + {offset};
    float sum = 0.0f;
    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) sum += v[i];
    float mean = kir_block_sum(sum, scratch) / WIDTH;
    float sq = 0.0f;
    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) sq += (v[i] - mean) * (v[i] - mean);
    float inv = rsqrtf(kir_block_sum(sq, scratch) / WIDTH + {eps}f);
    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) v[i] = w[i] * ((v[i] - mean) * inv) + b[i];
",
                eps = crate::lit(*eps),
            ),
            RowBody::IndexTopk { heads, dim, topk, max_keys, q_row, q_offset, w_row, w_offset, out_row, layers } => {
                let compressed = k.params.iter().any(|p| p.name == "ratio");
                let (count, entry) = if compressed { ("(pos[row * pos_step] + row * row_step + 1) / ratio", "t * ratio + ratio - 1") } else { ("pos[row * pos_step] + row * row_step + 1", "t") };
                format!(
                "    constexpr int HEADS = {heads};
    constexpr int DIM = {dim};
    constexpr int TOPK = {topk};
    constexpr int MAX_KEYS = {max_keys};
    __shared__ float score[MAX_KEYS];
    __shared__ float best_v[32];
    __shared__ int best_i[32];
    const int row = blockIdx.y;
    const int n = min({count}, MAX_KEYS);
    const float* qr = q + (size_t)row * {q_row} + {q_offset};
    const float* wr = w + (size_t)row * {w_row} + {w_offset};
    const int* table = seq_table(block_table, slots, row, pages);
    for (int t = threadIdx.x; t < MAX_KEYS; t += blockDim.x) {{
        float s = -3.0e30f;
        if (t < n) {{
            const int tp = {entry};
            size_t at = (((size_t)table[tp / PAGE] * {layers} + layer)) * PAGE + tp % PAGE;
            const bf16_t* kr = keys + at * DIM * KV_PLANES;
            s = 0.0f;
            for (int h = 0; h < HEADS; h++) {{
                float d = 0.0f;
                for (int j = 0; j < DIM; j++) d += qr[h * DIM + j] * (attn_float(kr[j]) + (KV_PLANES == 2 ? attn_float(kr[DIM + j]) : 0.0f));
                s += wr[h] * w_scale * fmaxf(d * scale, 0.0f);
            }}
        }}
        score[t] = s;
    }}
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    for (int r = 0; r < TOPK; r++) {{
        float bv = -3.0e30f;
        int bi = MAX_KEYS;
        for (int t = threadIdx.x; t < MAX_KEYS; t += blockDim.x) {{
            if (score[t] > bv) {{ bv = score[t]; bi = t; }}
        }}
        for (int o = 16; o > 0; o >>= 1) {{
            float ov = __shfl_xor_sync(0xffffffff, bv, o);
            int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {{ bv = ov; bi = oi; }}
        }}
        if (lane == 0) {{ best_v[warp] = bv; best_i[warp] = bi; }}
        __syncthreads();
        if (threadIdx.x == 0) {{
            float v = best_v[0];
            int at = best_i[0];
            for (int i = 1; i < warps; i++) {{
                if (best_v[i] > v || (best_v[i] == v && best_i[i] < at)) {{ v = best_v[i]; at = best_i[i]; }}
            }}
            bool live = r < n && at < MAX_KEYS;
            idx[(size_t)row * {out_row} + r] = live ? at : -1;
            if (live) score[at] = -3.0e30f;
        }}
        __syncthreads();
    }}
"
                )
            }
            RowBody::HeadRms { row, head, offset, width, eps } => format!(
                "    __shared__ float part[32];
    const int r = blockIdx.y;
    float* v = x + (size_t)r * {row} + (size_t)blockIdx.x * {head} + {offset};
    float ss = 0.0f;
    for (int i = threadIdx.x; i < {width}; i += blockDim.x) ss += v[i] * v[i];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < ((int)blockDim.x + 31) / 32; w++) total += part[w];
    const float inv = rsqrtf(total / {width}.0f + {eps}f);
    for (int i = threadIdx.x; i < {width}; i += blockDim.x) v[i] *= inv;
",
                eps = crate::lit(*eps),
            ),
            RowBody::BlockPool { dim, rot, layers, raw_width, gate_offset, eps } => format!(
                "    constexpr int DIM = {dim};
    constexpr int ROT = {rot};
    constexpr int RAW = {raw_width};
    __shared__ float pooled[DIM];
    __shared__ float part[32];
    const int row = blockIdx.y;
    const int p = pos[row * pos_step] + row * row_step;
    if (ratio <= 0 || (p + 1) % ratio != 0) return;
    const int e = p / ratio;
    const int* table = seq_table(block_table, slots, row, pages);
    const int coff = overlap ? 2 : 1;
    const int stride = coff * DIM;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < DIM; d += blockDim.x) {{
        float top = -3.0e30f;
        for (int k = 0; k < coff * ratio; k++) {{
            const bool previous = overlap && k < ratio;
            if (previous && e == 0) continue;
            const int s = previous ? (e - 1) * ratio + k : e * ratio + (k - (overlap ? ratio : 0));
            const int col = (overlap && !previous ? DIM : 0) + d;
            const int j = previous ? k : k - (overlap ? ratio : 0);
            const bf16_t* rr = raw + (((size_t)table[s / PAGE] * {layers} + layer) * PAGE + s % PAGE) * RAW * KV_PLANES;
            float g = attn_float(rr[{gate_offset} + col]) + (KV_PLANES == 2 ? attn_float(rr[RAW + {gate_offset} + col]) : 0.0f) + bias[(size_t)j * stride + col];
            top = fmaxf(top, g);
        }}
        float denom = 0.0f;
        float acc = 0.0f;
        for (int k = 0; k < coff * ratio; k++) {{
            const bool previous = overlap && k < ratio;
            if (previous && e == 0) continue;
            const int s = previous ? (e - 1) * ratio + k : e * ratio + (k - (overlap ? ratio : 0));
            const int col = (overlap && !previous ? DIM : 0) + d;
            const int j = previous ? k : k - (overlap ? ratio : 0);
            const bf16_t* rr = raw + (((size_t)table[s / PAGE] * {layers} + layer) * PAGE + s % PAGE) * RAW * KV_PLANES;
            float g = attn_float(rr[{gate_offset} + col]) + (KV_PLANES == 2 ? attn_float(rr[RAW + {gate_offset} + col]) : 0.0f) + bias[(size_t)j * stride + col];
            float w = expf(g - top);
            float v = attn_float(rr[col]) + (KV_PLANES == 2 ? attn_float(rr[RAW + col]) : 0.0f);
            denom += w;
            acc += w * v;
        }}
        float v = acc / denom;
        pooled[d] = v;
        ss += v * v;
    }}
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < ((int)blockDim.x + 31) / 32; w++) total += part[w];
    const float inv = rsqrtf(total / (float)DIM + {eps}f);
    for (int d = threadIdx.x; d < DIM; d += blockDim.x) pooled[d] = norm[d] * (pooled[d] * inv);
    __syncthreads();
    bf16_t* out = dst + (((size_t)table[p / PAGE] * {layers} + layer) * PAGE + p % PAGE) * DIM * KV_PLANES;
    for (int d = threadIdx.x; d < DIM - ROT; d += blockDim.x) {{
        unsigned hi, lo;
        attn_split(pooled[d], hi, lo);
        out[d] = (bf16_t)hi;
        if (KV_PLANES == 2) out[DIM + d] = (bf16_t)lo;
    }}
    for (int i = threadIdx.x; i < ROT / 2; i += blockDim.x) {{
        float angle = (float)(e * ratio) * rope_inv_freq(i, layer);
        float c = cosf(angle) * ROPE_MSCALE;
        float s = sinf(angle) * ROPE_MSCALE;
        float a = pooled[DIM - ROT + 2 * i];
        float b = pooled[DIM - ROT + 2 * i + 1];
        float ra = a * c - b * s;
        float rb = b * c + a * s;
        unsigned hi, lo;
        attn_split(ra, hi, lo);
        out[DIM - ROT + 2 * i] = (bf16_t)hi;
        if (KV_PLANES == 2) out[DIM + DIM - ROT + 2 * i] = (bf16_t)lo;
        attn_split(rb, hi, lo);
        out[DIM - ROT + 2 * i + 1] = (bf16_t)hi;
        if (KV_PLANES == 2) out[DIM + DIM - ROT + 2 * i + 1] = (bf16_t)lo;
    }}
",
                eps = crate::lit(*eps),
            ),
            RowBody::HeadMatvec { x_row, x_head, x_offset, inner, w_head, w_inner, w_out, w_offset, out, y_row, y_head, y_offset } => {
                let store = if Self::planes(&k.params) {
                    format!(
                        "        unsigned hi, lo;
        split_hi_lo(acc, hi, lo);
        size_t at = (size_t)row * {y_row} + (size_t)h * {y_head} + {y_offset} + o;
        out_hi[at] = (bf16_t)hi;
        out_lo[at] = (bf16_t)lo;
"
                    )
                } else {
                    format!("        y[(size_t)row * {y_row} + (size_t)h * {y_head} + {y_offset} + o] = acc;\n")
                };
                format!(
                    "    const int h = blockIdx.x;
    const int row = blockIdx.y;
    const float* xv = x + (size_t)row * {x_row} + (size_t)h * {x_head} + {x_offset};
    const float* wv = w + (size_t)h * {w_head} + {w_offset};
    for (int o = threadIdx.x; o < {out}; o += blockDim.x) {{
        float acc = 0.0f;
        for (int i = 0; i < {inner}; i++) acc += xv[i] * wv[(size_t)i * {w_inner} + (size_t)o * {w_out}];
{store}    }}
"
                )
            }
        };
        Ok(format!("{head}{body}}}\n\n"))
    }
}
