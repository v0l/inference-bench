use crate::{Access, Kernel, Language, Param, Role, Scalar, Target};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeStyle {
    Halves,
    Pairs,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RowBody {
    SpanRope { row: u32, head: u32, offset: u32, rot: u32, style: RopeStyle },
    StateWrite { src_row: u32, src_offset: u32, layers: u32, rows: u32, width: u32 },
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
            RowBody::StateWrite { width, rows, layers, .. } if *width == 0 || *rows == 0 || *layers == 0 => Err("state_write needs a positive width, rows and layers".into()),
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
            RowBody::SpanRope { row, head: stride, offset, rot, style } => {
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
        float s = sinf(angle) * ROPE_MSCALE;
        float a = v[{a}];
        float b = v[{b}];
        v[{a}] = a * c - b * s;
        v[{b}] = b * c + a * s;
    }}
"
                )
            }
            RowBody::StateWrite { src_row, src_offset, layers, rows, width } => format!(
                "    const int row = blockIdx.y;
    const int r = blockIdx.x;
    const int p = pos[row * pos_step] + row * row_step;
    const int* table = seq_table(block_table, slots, row, pages);
    size_t slot = (((size_t)table[p / PAGE] * {layers} + layer) * {rows} + r) * PAGE + p % PAGE;
    bf16_t* out = dst + slot * {width} * KV_PLANES;
    const float* in = src + (size_t)row * {src_row} + {src_offset} + (size_t)r * {width};
    for (int d = threadIdx.x; d < {width}; d += blockDim.x) {{
        unsigned hi, lo;
        attn_split(in[d], hi, lo);
        out[d] = (bf16_t)hi;
        if (KV_PLANES == 2) out[{width} + d] = (bf16_t)lo;
    }}
"
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
