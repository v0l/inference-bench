use crate::{Language, Target, fnv};

pub const BA_ROWS_MAX_TOKENS: usize = 32;
pub const BA_ROWS_THREADS: u32 = 256;
pub const GEMV_THREADS: u32 = 256;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Gemv {
    Linear { decoder: String, n: usize, k: usize, bias: bool, residual: bool },
    Norm { decoder: String, n: usize, bias: bool },
    GateUpAct { decoder: String },
    Tiled { decoder: String, n: usize, k: usize, bias: bool, residual: bool },
    Bf16Rows { n: usize, k: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GemvKernel {
    pub name: String,
    pub target: Target,
    pub op: Gemv,
}

impl GemvKernel {
    pub fn new(t: &Target, name: &str, op: Gemv) -> Result<GemvKernel, String> {
        let k = GemvKernel { name: name.to_string(), target: *t, op };
        k.validate()?;
        Ok(k)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.target.language != Language::Cuda {
            return Err(format!("{}: gemv has no {:?} lowering", self.name, self.target.language));
        }
        if self.name.is_empty() || !self.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!("kernel name {:?} is not an identifier", self.name));
        }
        let ok = match &self.op {
            Gemv::Linear { decoder, n, k, .. } | Gemv::Tiled { decoder, n, k, .. } => !decoder.is_empty() && *n > 0 && *k > 0,
            Gemv::Norm { decoder, n, .. } => !decoder.is_empty() && *n > 0,
            Gemv::GateUpAct { decoder } => !decoder.is_empty(),
            Gemv::Bf16Rows { n, k } => *n > 0 && k.is_multiple_of(8),
        };
        if !ok {
            return Err(format!("{}: bad gemv shape {:?}", self.name, self.op));
        }
        Ok(())
    }

    pub fn threads(&self) -> u32 {
        match self.op {
            Gemv::Bf16Rows { .. } => BA_ROWS_THREADS,
            _ => GEMV_THREADS,
        }
    }

    pub fn describe(&self) -> String {
        format!("{}\ngemv {} threads={}\n  {:?}", self.target.describe(), self.name, self.threads(), self.op)
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }

    pub fn emit(&self) -> Result<String, String> {
        self.validate()?;
        let name = &self.name;
        Ok(match &self.op {
            Gemv::Linear { decoder, n, k, bias, residual } => format!(
                "extern \"C\" __global__ void {name}(const void* w, const float* s, const float* b, const float* x, float* out) {{\n    linear_w<{decoder}, {n}, {k}, {bias}, {residual}>(w, s, b, x, out);\n}}\n\n"
            ),
            Gemv::Tiled { decoder, n, k, bias, residual } => format!(
                "extern \"C\" __global__ void {name}(const void* w, const float* s, const float* b, const float* x, float* out, int tokens) {{\n    gemm<{n}, {k}, {bias}, {residual}, {decoder}>(w, s, b, x, out, tokens);\n}}\n\n"
            ),
            Gemv::Norm { decoder, n, bias } => format!(
                "extern \"C\" __global__ void {name}(const void* w, const float* s, const float* b, const float* x, const float* nw, float* out) {{\n    linear_norm<{decoder}, {n}, {bias}>(w, s, b, x, nw, out);\n}}\n\n"
            ),
            Gemv::GateUpAct { decoder } => format!(
                "extern \"C\" __global__ void {name}(const void* w, const float* s, const float* x, const float* nw, float* out) {{\n    gate_up_act_norm<{decoder}>(w, s, x, nw, out);\n}}\n\n"
            ),
            Gemv::Bf16Rows { n, k } => bf16_rows(name, *n, *k),
        })
    }
}

pub fn gemv_hash(kernels: &[GemvKernel]) -> String {
    fnv(kernels.iter().flat_map(|k| k.hash().into_bytes().into_iter().chain(std::iter::once(0))))
}

pub fn emit_gemvs(kernels: &[GemvKernel]) -> Result<String, String> {
    let mut out = String::new();
    for k in kernels {
        out.push_str(&k.emit()?);
    }
    Ok(out)
}

fn bf16_rows(name: &str, n: usize, k: usize) -> String {
    let m = BA_ROWS_MAX_TOKENS;
    let t = BA_ROWS_THREADS;
    let warps = t / 32;
    let chunks = k / 8;
    format!(
        "extern \"C\" __global__ void {name}(const bf16_t* w, const float* s, const float* b, const bf16_t* xh, const bf16_t* xl,
                                       float* out, const bf16_t* dummy_hi, const bf16_t* dummy_lo, int tokens) {{
    __shared__ float part[{warps}][{m}];
    int row = blockIdx.x;
    int m0 = blockIdx.y * {m};
    int chunk = min(tokens - m0, {m});
    int tid = threadIdx.x;
    float acc[{m}];
#pragma unroll
    for (int m = 0; m < {m}; m++) acc[m] = 0.0f;
    const uint4* wr = (const uint4*)(w + (size_t)row * {k});
    for (int i = tid; i < {chunks}; i += {t}) {{
        uint4 q = wr[i];
        float wf[8];
        unsigned words[4] = {{q.x, q.y, q.z, q.w}};
#pragma unroll
        for (int j = 0; j < 4; j++) {{
            wf[2 * j] = __uint_as_float(words[j] << 16);
            wf[2 * j + 1] = __uint_as_float(words[j] & 0xffff0000u);
        }}
#pragma unroll
        for (int m = 0; m < {m}; m++) {{
            if (m < chunk) {{
                uint4 h = ((const uint4*)(xh + (size_t)(m0 + m) * {k}))[i];
                uint4 l = ((const uint4*)(xl + (size_t)(m0 + m) * {k}))[i];
                unsigned hw[4] = {{h.x, h.y, h.z, h.w}};
                unsigned lw[4] = {{l.x, l.y, l.z, l.w}};
                float a = acc[m];
#pragma unroll
                for (int j = 0; j < 4; j++) {{
                    a += wf[2 * j] * (__uint_as_float(hw[j] << 16) + __uint_as_float(lw[j] << 16));
                    a += wf[2 * j + 1] * (__uint_as_float(hw[j] & 0xffff0000u) + __uint_as_float(lw[j] & 0xffff0000u));
                }}
                acc[m] = a;
            }}
        }}
    }}
#pragma unroll
    for (int m = 0; m < {m}; m++) {{
        float v = acc[m];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if ((tid & 31) == 0) part[tid >> 5][m] = v;
    }}
    __syncthreads();
    if (tid < chunk) {{
        float v = 0.0f;
#pragma unroll
        for (int wp = 0; wp < {warps}; wp++) v += part[wp][tid];
        out[(size_t)(m0 + tid) * {n} + row] = v;
    }}
}}

"
    )
}
