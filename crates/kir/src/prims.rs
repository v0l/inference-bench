use crate::layout::Dialect;
use crate::{Language, Target, fnv};
use crate::ops::f32_literal;
use std::fmt::Write as _;

pub const ARGMAX_THREADS: u32 = 1024;
pub const WGSL_ARGMAX_THREADS: u32 = 256;
pub const SAMPLE_PARAMS: usize = 6;
pub const TOPK_BINS: usize = 2048;
pub const TOPK_CHUNK: usize = 32;
pub const TOPP_CANDIDATES: usize = 1024;
pub const TOPP_STEPS: usize = 48;

const HASH_SEED: [u32; 3] = [747796405, 2891336453, 2246822519];
const HASH_MIX: [(bool, u32); 5] = [(false, 16), (true, 2246822507), (false, 13), (true, 3266489909), (false, 16)];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowSource {
    Bf16,
    Ggml,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Prim {
    Embed { width: usize, scale: f64, source: RowSource },
    SpliceRows { width: usize },
    AddRows { width: usize },
    CopyRows,
    SplitRows,
    Bump,
    RmsSplit { width: usize, eps: f64 },
    ArgmaxRows { vocab: usize },
    SampleRows { vocab: usize },
    ArgmaxAdvance { vocab: usize },
    ApplyMask { vocab: usize },
    Sample { vocab: usize },
    AddBias,
    DequantRows,
}

impl Prim {
    pub fn languages(&self) -> &'static [Language] {
        match self {
            Prim::Embed { source: RowSource::Bf16, .. } => &[Language::Cuda],
            Prim::Embed { source: RowSource::Ggml, .. } | Prim::Sample { .. } | Prim::DequantRows | Prim::AddBias => &[Language::Wgsl],
            _ => &[Language::Cuda],
        }
    }

    fn tree(&self) -> bool {
        matches!(self, Prim::ArgmaxRows { .. } | Prim::SampleRows { .. } | Prim::ArgmaxAdvance { .. } | Prim::Sample { .. })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrimKernel {
    pub name: &'static str,
    pub target: Target,
    pub prim: Prim,
    pub threads: u32,
}

impl PrimKernel {
    pub fn new(t: &Target, name: &'static str, prim: Prim) -> Result<PrimKernel, String> {
        let threads = match (t.language, prim) {
            (Language::Wgsl, _) => WGSL_ARGMAX_THREADS,
            (_, p) if p.tree() => ARGMAX_THREADS,
            _ => 256,
        };
        let k = PrimKernel { name, target: *t, prim, threads };
        k.validate()?;
        Ok(k)
    }

    pub fn validate(&self) -> Result<(), String> {
        let t = &self.target;
        if self.name.is_empty() || !self.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!("kernel name {:?} is not an identifier", self.name));
        }
        if !self.prim.languages().contains(&t.language) {
            return Err(format!("{}: {:?} has no {:?} lowering", self.name, self.prim, t.language));
        }
        if self.threads > t.max_threads || !self.threads.is_power_of_two() {
            return Err(format!("{}: {} threads do not fit the target", self.name, self.threads));
        }
        if self.smem_bytes() > t.smem_limit {
            return Err(format!("{}: {} bytes of shared memory exceeds the target", self.name, self.smem_bytes()));
        }
        match self.prim {
            Prim::Embed { width, scale, .. } if width == 0 || !scale.is_finite() => Err(format!("{}: bad embed width or scale", self.name)),
            Prim::SpliceRows { width: 0 } | Prim::AddRows { width: 0 } => Err(format!("{}: width must be positive", self.name)),
            Prim::RmsSplit { width, eps } if width == 0 || !width.is_multiple_of(4) || !eps.is_finite() || eps < 0.0 => {
                Err(format!("{}: rms_split needs a width that is a multiple of 4 and a finite eps", self.name))
            }
            Prim::ArgmaxRows { vocab } | Prim::SampleRows { vocab } | Prim::ArgmaxAdvance { vocab } | Prim::ApplyMask { vocab } | Prim::Sample { vocab } if vocab == 0 => {
                Err(format!("{}: vocab must be positive", self.name))
            }
            _ => Ok(()),
        }
    }

    pub fn smem_bytes(&self) -> u32 {
        let tree = if self.prim.tree() { self.threads * 8 } else { 0 };
        let sample = match self.prim {
            Prim::SampleRows { .. } | Prim::ArgmaxAdvance { .. } => (TOPK_BINS * 4 + 64 * 4 + 32 * 4 + 8 + TOPP_CANDIDATES * 4 + 4) as u32,
            _ => 0,
        };
        tree + sample
    }

    pub fn describe(&self) -> String {
        format!("{}\nprim {} threads={} smem={}\n  {:?}", self.target.describe(), self.name, self.threads, self.smem_bytes(), self.prim)
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }

    pub fn emit(&self) -> Result<String, String> {
        self.validate()?;
        let d = match self.target.language {
            Language::Cuda => Dialect::Cuda,
            Language::Wgsl => Dialect::Wgsl,
        };
        Ok(match (d, self.prim) {
            (Dialect::Cuda, Prim::RmsSplit { width, eps }) => cuda_rms_split(self.name, width, eps),
            (Dialect::Cuda, Prim::Embed { width, scale, .. }) => cuda_embed(self.name, width, scale),
            (Dialect::Cuda, Prim::SpliceRows { width }) => cuda_splice(self.name, width),
            (Dialect::Cuda, Prim::AddRows { width }) => cuda_add_rows(self.name, width),
            (Dialect::Cuda, Prim::ArgmaxRows { vocab }) => cuda_argmax_rows(self.name, vocab),
            (Dialect::Cuda, Prim::ApplyMask { vocab }) => cuda_apply_mask(self.name, vocab),
            (Dialect::Cuda, Prim::SampleRows { vocab }) => cuda_sample_rows(self.name, vocab),
            (Dialect::Cuda, Prim::ArgmaxAdvance { vocab }) => cuda_argmax_advance(self.name, vocab),
            (Dialect::Cuda, Prim::Bump) => format!("extern \"C\" __global__ void {}(int* pos, int* count) {{\n    pos[threadIdx.x] += 1;\n    if (threadIdx.x == 0) count[0] += 1;\n}}\n\n", self.name),
            (Dialect::Cuda, Prim::CopyRows) => format!(
                "extern \"C\" __global__ void {}(const float* src, float* dst, int n) {{\n    {} dst[i] = src[i];\n}}\n\n",
                self.name,
                grid_stride("n", 1)
            ),
            (Dialect::Cuda, Prim::SplitRows) => format!(
                "extern \"C\" __global__ void {}(const float* src, bf16_t* hi, bf16_t* lo, int n) {{\n    {}\n        split_store2(hi, lo, i, src[i], src[i + 1]);\n}}\n\n",
                self.name,
                grid_stride("n", 2)
            ),
            (Dialect::Wgsl, Prim::AddBias) => wgsl_bias(),
            (Dialect::Wgsl, Prim::Embed { width, scale, .. }) => wgsl_embed(width, scale),
            (Dialect::Wgsl, Prim::Sample { vocab }) => wgsl_sample(vocab, self.threads),
            (Dialect::Wgsl, Prim::DequantRows) => wgsl_dequant_rows(),
            (_, p) => return Err(format!("{}: no {d:?} lowering for {p:?}", self.name)),
        })
    }
}

pub fn prims_hash(kernels: &[PrimKernel]) -> String {
    fnv(kernels.iter().flat_map(|k| k.hash().into_bytes().into_iter().chain(std::iter::once(0))))
}

pub fn cuda_prims(t: &Target, hidden: usize, vocab: usize, eps: f64, embed_scale: f64) -> Result<Vec<PrimKernel>, String> {
    [
        ("rms_split", Prim::RmsSplit { width: hidden, eps }),
        ("embed", Prim::Embed { width: hidden, scale: embed_scale, source: RowSource::Bf16 }),
        ("splice_rows", Prim::SpliceRows { width: hidden }),
        ("add_rows", Prim::AddRows { width: hidden }),
        ("argmax_rows", Prim::ArgmaxRows { vocab }),
        ("apply_mask", Prim::ApplyMask { vocab }),
        ("sample_rows", Prim::SampleRows { vocab }),
        ("argmax_advance", Prim::ArgmaxAdvance { vocab }),
        ("bump", Prim::Bump),
        ("copy_rows", Prim::CopyRows),
        ("split_rows", Prim::SplitRows),
    ]
    .into_iter()
    .map(|(n, p)| PrimKernel::new(t, n, p))
    .collect()
}

pub fn emit_cuda(kernels: &[PrimKernel]) -> Result<crate::KirSource, String> {
    let mut out = String::new();
    let vocab = kernels.iter().find_map(|k| match k.prim {
        Prim::SampleRows { vocab } | Prim::ArgmaxAdvance { vocab } => Some(vocab),
        _ => None,
    });
    let mut library = false;
    for k in kernels {
        if k.target.language != Language::Cuda {
            return Err(format!("{}: not a CUDA kernel", k.name));
        }
        if !library && matches!(k.prim, Prim::ArgmaxRows { .. } | Prim::SampleRows { .. } | Prim::ArgmaxAdvance { .. }) {
            out.push_str(&gumbel(Dialect::Cuda));
            out.push('\n');
            if let Some(v) = vocab {
                out.push_str(&cuda_sample_threshold(v));
            }
            library = true;
        }
        out.push_str(&k.emit()?);
    }
    Ok(crate::KirSource::emitted(out))
}

fn grid_stride(n: &str, step: usize) -> String {
    match step {
        1 => format!("for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < {n}; i += gridDim.x * blockDim.x)"),
        s => format!("for (int i = {s} * (blockIdx.x * blockDim.x + threadIdx.x); i < {n}; i += {s} * gridDim.x * blockDim.x)"),
    }
}

pub fn gumbel(d: Dialect) -> String {
    let [a, b, c] = HASH_SEED;
    let mut s = String::new();
    match d {
        Dialect::Cuda => {
            let _ = writeln!(s, "__device__ float gumbel(unsigned seed, unsigned step, unsigned i) {{");
            let _ = writeln!(s, "    unsigned x = seed * {a}u + step * {b}u + i * {c}u + 1u;");
            let mix: Vec<String> = HASH_MIX.iter().map(|&(mul, v)| if mul { format!("x *= {v}u;") } else { format!("x ^= x >> {v};") }).collect();
            let _ = writeln!(s, "    {}", mix.join(" "));
            let _ = writeln!(s, "    float u = ((float)(x >> 8) + 0.5f) / 16777216.0f;");
            let _ = writeln!(s, "    return -logf(-logf(u));\n}}");
        }
        Dialect::Wgsl => {
            let _ = writeln!(s, "fn gumbel(seed: u32, step: u32, i: u32) -> f32 {{");
            let _ = writeln!(s, "    var x = seed * {a}u + step * {b}u + i * {c}u + 1u;");
            for &(mul, v) in &HASH_MIX {
                let _ = writeln!(s, "    {}", if mul { format!("x *= {v}u;") } else { format!("x ^= x >> {v}u;") });
            }
            let _ = writeln!(s, "    let u = (f32(x >> 8u) + 0.5) / 16777216.0;");
            let _ = writeln!(s, "    return -log(-log(u));\n}}");
        }
    }
    s
}

fn argmax_tree(d: Dialect, half: &str) -> String {
    let (tid, int, bar, zero, shift, decl_v, decl_i) = match d {
        Dialect::Cuda => ("threadIdx.x", "int s", "__syncthreads();", "0", "1", "float ov", "int oi"),
        Dialect::Wgsl => ("li", "var s", "workgroupBarrier();", "0u", "1u", "let ov", "let oi"),
    };
    format!(
        "    best_v[{tid}] = bv;
    best_i[{tid}] = bi;
    {bar}
    for ({int} = {half}; s > {zero}; s >>= {shift}) {{
        if ({tid} < s) {{
            {decl_v} = best_v[{tid} + s];
            {decl_i} = best_i[{tid} + s];
            if (ov > best_v[{tid}] || (ov == best_v[{tid}] && oi < best_i[{tid}])) {{
                best_v[{tid}] = ov;
                best_i[{tid}] = oi;
            }}
        }}
        {bar}
    }}
"
    )
}

fn cuda_scan(vocab: usize, sample: Option<&str>) -> String {
    let body = match sample {
        None => String::new(),
        Some(step) => format!(
            "        if (inv_temp > 0.0f) {{
            v *= inv_temp;
            if (v < thr) continue;
            v += gumbel(seed, {step}, (unsigned)i);
        }}
"
        ),
    };
    format!(
        "    float bv = NEG_INF;
    int bi = 0;
    for (int i = threadIdx.x; i < {vocab}; i += blockDim.x) {{
        float v = logits[i];
{body}        if (v > bv) {{
            bv = v;
            bi = i;
        }}
    }}
"
    )
}

const CUDA_TREE_SMEM: &str = "    __shared__ float best_v[1024];\n    __shared__ int best_i[1024];\n";

fn cuda_threshold(args: &str) -> String {
    format!(
        "    float thr = NEG_INF;
    if (inv_temp > 0.0f) {{
        float m;
        thr = sample_threshold({args}, &m);
    }}
"
    )
}

fn cuda_argmax_rows(name: &str, vocab: usize) -> String {
    format!(
        "extern \"C\" __global__ void {name}(const float* logits, int* tokens, int* history, const int* count, int stride) {{
{CUDA_TREE_SMEM}    logits += (size_t)blockIdx.x * {vocab};
{scan}{tree}    if (threadIdx.x == 0) {{
        tokens[blockIdx.x] = best_i[0];
        history[(size_t)blockIdx.x * stride + count[0]] = best_i[0];
    }}
}}

",
        scan = cuda_scan(vocab, None),
        tree = argmax_tree(Dialect::Cuda, "blockDim.x / 2"),
    )
}

fn cuda_sample_rows(name: &str, vocab: usize) -> String {
    format!(
        "extern \"C\" __global__ void {name}(const float* logits, int* tokens, const int* params) {{
{CUDA_TREE_SMEM}    logits += (size_t)blockIdx.x * {vocab};
    const int* p = params + {SAMPLE_PARAMS} * blockIdx.x;
    float inv_temp = __int_as_float(p[0]);
    unsigned seed = (unsigned)p[1];
    unsigned step = (unsigned)p[2];
{thr}{scan}{tree}    if (threadIdx.x == 0) tokens[blockIdx.x] = best_i[0];
}}

",
        thr = cuda_threshold("logits, inv_temp, p[3], __int_as_float(p[4]), __int_as_float(p[5])"),
        scan = cuda_scan(vocab, Some("step")),
        tree = argmax_tree(Dialect::Cuda, "blockDim.x / 2"),
    )
}

fn cuda_argmax_advance(name: &str, vocab: usize) -> String {
    format!(
        "extern \"C\" __global__ void {name}(const float* logits, int* token, int* pos, int* history, int* count, float inv_temp, unsigned seed, int top_k, float top_p, float min_p) {{
{CUDA_TREE_SMEM}{thr}{scan}{tree}    if (threadIdx.x == 0) {{
        token[0] = best_i[0];
        history[count[0]] = best_i[0];
        count[0] += 1;
        pos[0] += 1;
    }}
}}

",
        thr = cuda_threshold("logits, inv_temp, top_k, top_p, min_p"),
        scan = cuda_scan(vocab, Some("(unsigned)count[0]")),
        tree = argmax_tree(Dialect::Cuda, "blockDim.x / 2"),
    )
}

fn cuda_apply_mask(name: &str, vocab: usize) -> String {
    format!(
        "extern \"C\" __global__ void {name}(float* logits, const unsigned* mask, const int* flags, int words) {{
    int row = blockIdx.x;
    if (!flags[row]) return;
    logits += (size_t)row * {vocab};
    mask += (size_t)row * words;
    for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < {vocab}; i += gridDim.y * blockDim.x) {{
        if (!((mask[i >> 5] >> (i & 31)) & 1u)) logits[i] = NEG_INF;
    }}
}}

"
    )
}

fn cuda_rms_split(name: &str, width: usize, eps: f64) -> String {
    let q = width / 4;
    format!(
        "extern \"C\" __global__ void {name}(const float* x, const float* weight, bf16_t* hi, bf16_t* lo) {{
    __shared__ float scratch[32];
    size_t row = (size_t)blockIdx.x * {width};
    const float4* xv = (const float4*)(x + row);
    float ss = 0.0f;
    for (int i = threadIdx.x; i < {q}; i += blockDim.x) {{
        float4 v = xv[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }}
    float inv = rsqrtf(block_reduce<false>(ss, scratch) / {width} + {eps});
    for (int i = threadIdx.x; i < {q}; i += blockDim.x) {{
        float4 v = xv[i];
        float4 nv = ((const float4*)weight)[i];
        split_store2(hi, lo, row + 4 * i, nv.x * (v.x * inv), nv.y * (v.y * inv));
        split_store2(hi, lo, row + 4 * i + 2, nv.z * (v.z * inv), nv.w * (v.w * inv));
    }}
}}

",
        eps = f32_literal(eps),
    )
}

fn cuda_embed(name: &str, width: usize, scale: f64) -> String {
    format!(
        "extern \"C\" __global__ void {name}(const bf16_t* table, const int* token, float* x) {{
    const bf16_t* row = table + (size_t)token[blockIdx.y] * {width};
    x += (size_t)blockIdx.y * {width};
    {loop}
        x[i] = bf16_to_f32(row[i]) * {scale};
}}

",
        loop = grid_stride(&width.to_string(), 1),
        scale = f32_literal(scale),
    )
}

fn cuda_splice(name: &str, width: usize) -> String {
    format!(
        "extern \"C\" __global__ void {name}(float* x, const int* rows, const float* embeds) {{
    int row = rows[blockIdx.y];
    if (row < 0) return;
    float* dst = x + (size_t)blockIdx.y * {width};
    const float* src = embeds + (size_t)row * {width};
    {loop} dst[i] = src[i];
}}

",
        loop = grid_stride(&width.to_string(), 1),
    )
}

fn cuda_add_rows(name: &str, width: usize) -> String {
    format!(
        "extern \"C\" __global__ void {name}(float* x, const int* rows, const float* embeds) {{
    int row = rows[blockIdx.y];
    if (row < 0) return;
    float* dst = x + (size_t)blockIdx.y * {width};
    const float* src = embeds + (size_t)row * {width};
    {loop} dst[i] += src[i];
}}

",
        loop = grid_stride(&width.to_string(), 1),
    )
}

fn cuda_sample_threshold(vocab: usize) -> String {
    let bins = TOPK_BINS;
    let chunks = TOPK_BINS / TOPK_CHUNK;
    let cand = TOPP_CANDIDATES;
    let steps = TOPP_STEPS;
    format!(
        "
__device__ __forceinline__ unsigned order_key(float v) {{
    unsigned b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}}

__device__ __forceinline__ float order_value(unsigned k) {{
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}}

__device__ float sample_threshold(const float* row, float inv_temp, int top_k, float top_p, float min_p, float* row_max) {{
    __shared__ unsigned hist[{bins}];
    __shared__ unsigned chunk[{chunks}];
    __shared__ float scratch[32];
    __shared__ unsigned sh_prefix;
    __shared__ int sh_need;
    float m = NEG_INF;
    for (int i = threadIdx.x; i < {vocab}; i += blockDim.x) m = fmaxf(m, row[i] * inv_temp);
    m = block_reduce<true>(m, scratch);
    *row_max = m;
    bool use_k = top_k > 0 && top_k < {vocab};
    bool use_p = top_p < 1.0f;
    float thr = NEG_INF;
    if (use_k) {{
        unsigned prefix = 0, mask = 0;
        int need = top_k;
        for (int pass = 0; pass < 3; pass++) {{
            int shift = pass == 0 ? 21 : (pass == 1 ? 10 : 0);
            unsigned width_mask = pass == 2 ? 1023u : 2047u;
            for (int i = threadIdx.x; i < {bins}; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int i = threadIdx.x; i < {vocab}; i += blockDim.x) {{
                unsigned key = order_key(row[i] * inv_temp);
                if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & width_mask], 1u);
            }}
            __syncthreads();
            if (threadIdx.x < {chunks}) {{
                unsigned c = 0;
                for (int j = 0; j < {per}; j++) c += hist[threadIdx.x * {per} + j];
                chunk[threadIdx.x] = c;
            }}
            __syncthreads();
            if (threadIdx.x == 0) {{
                int c = need;
                int ch = {last_chunk};
                for (; ch > 0; ch--) {{
                    if ((int)chunk[ch] >= c) break;
                    c -= (int)chunk[ch];
                }}
                int b = ch * {per} + {last_in_chunk};
                for (; b > ch * {per}; b--) {{
                    if ((int)hist[b] >= c) break;
                    c -= (int)hist[b];
                }}
                sh_need = c;
                sh_prefix = prefix | ((unsigned)b << shift);
            }}
            __syncthreads();
            need = sh_need;
            prefix = sh_prefix;
            mask |= width_mask << shift;
        }}
        thr = order_value(prefix);
    }}
    if (use_p) {{
        __shared__ float cand[{cand}];
        __shared__ int ncand;
        if (threadIdx.x == 0) ncand = 0;
        __syncthreads();
        if (use_k) {{
            for (int i = threadIdx.x; i < {vocab}; i += blockDim.x) {{
                float v = row[i] * inv_temp;
                if (v >= thr) {{
                    int at = atomicAdd(&ncand, 1);
                    if (at < {cand}) cand[at] = v;
                }}
            }}
            __syncthreads();
        }}
        bool cached = use_k && ncand <= {cand};
        int count = cached ? ncand : 0;
        float z = 0.0f;
        if (cached) {{
            for (int i = threadIdx.x; i < count; i += blockDim.x) z += expf(cand[i] - m);
        }} else {{
            for (int i = threadIdx.x; i < {vocab}; i += blockDim.x) {{
                float v = row[i] * inv_temp;
                if (v >= thr) z += expf(v - m);
            }}
        }}
        z = block_reduce<false>(z, scratch);
        float target = fmaxf(top_p, 1e-9f) * z;
        float lo = use_k ? thr : m - 88.0f;
        float hi = nextafterf(m, -NEG_INF);
        for (int it = 0; it < {steps}; it++) {{
            float mid = 0.5f * (lo + hi);
            if (mid <= lo || mid >= hi) break;
            float mass = 0.0f;
            if (cached) {{
                for (int i = threadIdx.x; i < count; i += blockDim.x) if (cand[i] >= mid) mass += expf(cand[i] - m);
            }} else {{
                for (int i = threadIdx.x; i < {vocab}; i += blockDim.x) {{
                    float v = row[i] * inv_temp;
                    if (v >= mid) mass += expf(v - m);
                }}
            }}
            mass = block_reduce<false>(mass, scratch);
            if (mass >= target) lo = mid; else hi = mid;
        }}
        float low = NEG_INF;
        if (cached) {{
            for (int i = threadIdx.x; i < count; i += blockDim.x) if (cand[i] >= lo) low = fmaxf(low, -cand[i]);
        }} else {{
            for (int i = threadIdx.x; i < {vocab}; i += blockDim.x) {{
                float v = row[i] * inv_temp;
                if (v >= lo) low = fmaxf(low, -v);
            }}
        }}
        thr = fmaxf(thr, -block_reduce<true>(low, scratch));
    }}
    if (min_p > 0.0f) thr = fmaxf(thr, m + logf(min_p));
    return thr;
}}

",
        per = TOPK_CHUNK,
        last_chunk = chunks - 1,
        last_in_chunk = TOPK_CHUNK - 1,
    )
}

const WGSL_PARAMS: &str = "struct P { a: vec4<u32>, b: vec4<u32> };\n";
pub const WGSL_ELEMENT_SPAN: u32 = 8388608;
pub const WGSL_NEG_INF: &str = "-3.0e38";

fn wgsl_bindings(list: &[(&str, &str, &str)]) -> String {
    let mut s = String::new();
    for (i, (name, mode, ty)) in list.iter().enumerate() {
        let _ = writeln!(s, "@group(0) @binding({i}) var<storage, {mode}> {name}: array<{ty}>;");
    }
    s.push_str(WGSL_PARAMS);
    let _ = writeln!(s, "@group(0) @binding({}) var<uniform> p: P;", list.len());
    s
}

fn wgsl_bias() -> String {
    format!(
        "{}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x + gid.y * {WGSL_ELEMENT_SPAN}u;
    if (idx >= p.a.x * p.a.y) {{
        return;
    }}
    y[idx] = y[idx] + bias[idx % p.a.y];
}}
",
        wgsl_bindings(&[("y", "read_write", "f32"), ("bias", "read", "f32")])
    )
}

fn wgsl_embed(width: usize, scale: f64) -> String {
    let scaled = if scale == 1.0 { String::new() } else { format!(" * {}", crate::ops::wgsl_float(scale)) };
    format!(
        "{}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let t = wid.x;
    let entry = rows[tokens[p.a.y + t]];
    let base = (entry >> 3u) * 16u;
    let code = entry & 7u;
    for (var c = li; c < {width}u; c += 256u) {{
        x[(p.a.x + t) * {width}u + c] = dequant_at(base, code, c){scaled};
    }}
}}
{}",
        wgsl_bindings(&[("tokens", "read", "u32"), ("wdata", "read", "u32"), ("rows", "read", "u32"), ("x", "read_write", "f32")]),
        crate::ggml::wgsl_decoder(),
    )
}

fn wgsl_dequant_rows() -> String {
    format!(
        "{}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x + gid.y * {WGSL_ELEMENT_SPAN}u;
    let nrows = p.a.x;
    let cols = p.a.y;
    if (idx >= nrows * cols) {{
        return;
    }}
    let r = idx / cols;
    let c = idx % cols;
    let entry = rows[r];
    out[idx] = dequant_at((entry >> 3u) * 16u, entry & 7u, c);
}}
{}",
        wgsl_bindings(&[("wdata", "read", "u32"), ("rows", "read", "u32"), ("out", "read_write", "f32")]),
        crate::ggml::wgsl_decoder(),
    )
}

fn wgsl_sample(vocab: usize, threads: u32) -> String {
    format!(
        "{}
var<workgroup> best_v: array<f32, {threads}>;
var<workgroup> best_i: array<u32, {threads}>;

{}
@compute @workgroup_size({threads})
fn main(@builtin(local_invocation_index) li: u32) {{
    let row = p.a.x * {vocab}u;
    let inv_temp = bitcast<f32>(p.a.y);
    var bv = {WGSL_NEG_INF};
    var bi = 0u;
    for (var i = li; i < {vocab}u; i += {threads}u) {{
        var v = logits[row + i];
        if (inv_temp > 0.0) {{
            v = v * inv_temp + gumbel(p.a.z, p.a.w, i);
        }}
        if (v > bv) {{
            bv = v;
            bi = i;
        }}
    }}
{}    if (li == 0u) {{
        tokens[p.b.x] = best_i[0];
    }}
}}
",
        wgsl_bindings(&[("logits", "read", "f32"), ("tokens", "read_write", "u32")]),
        gumbel(Dialect::Wgsl),
        argmax_tree(Dialect::Wgsl, &format!("{}u", threads / 2)),
    )
}
