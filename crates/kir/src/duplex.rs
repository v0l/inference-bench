use crate::fnv;
use std::fmt::Write as _;

pub const KERNELS: &[&str] = &["duplex_join", "duplex_exact_logits", "duplex_edit_logits", "duplex_allow", "duplex_user_latent", "duplex_state_move"];

pub const THREADS: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DuplexShape {
    pub hidden: usize,
    pub vocab: usize,
    pub latent: usize,
}

impl DuplexShape {
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden == 0 || self.vocab == 0 || self.latent == 0 {
            return Err(format!("duplex kernels need a hidden, vocab and latent width, got {self:?}"));
        }
        Ok(())
    }

    pub fn hash(&self) -> String {
        fnv(format!("duplex {} {} {}", self.hidden, self.vocab, self.latent).bytes())
    }
}

pub fn emit_cuda(s: &DuplexShape) -> Result<crate::KirSource, String> {
    s.validate()?;
    let mut o = String::new();
    let _ = writeln!(o, "constexpr int DUPLEX_HIDDEN = {};\nconstexpr int DUPLEX_VOCAB = {};\nconstexpr int DUPLEX_LATENT = {};\n", s.hidden, s.vocab, s.latent);
    o.push_str(BODY);
    Ok(crate::KirSource::emitted(o))
}

const BODY: &str = r#"__device__ __forceinline__ float duplex_bf16(unsigned short v) {
    return __uint_as_float(((unsigned)v) << 16);
}

extern "C" __global__ void duplex_join(const unsigned short* hi, const unsigned short* lo, float* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] = duplex_bf16(hi[i]) + duplex_bf16(lo[i]);
}

extern "C" __global__ void duplex_exact_logits(const unsigned short* hi, const unsigned short* lo, const unsigned short* table, const int* ids, float* logits) {
    __shared__ float part[32];
    int row = blockIdx.x;
    int id = ids[blockIdx.y];
    const unsigned short* h = hi + (size_t)row * DUPLEX_HIDDEN;
    const unsigned short* l = lo + (size_t)row * DUPLEX_HIDDEN;
    const unsigned short* w = table + (size_t)id * DUPLEX_HIDDEN;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < DUPLEX_HIDDEN; i += blockDim.x) acc = fmaf(duplex_bf16(h[i]) + duplex_bf16(l[i]), duplex_bf16(w[i]), acc);
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (threadIdx.x == 0) logits[(size_t)row * DUPLEX_VOCAB + id] = v;
    }
}

extern "C" __global__ void duplex_edit_logits(float* logits, const int* tok, const float* add, const int* count, int stride) {
    int row = blockIdx.x;
    float* lr = logits + (size_t)row * DUPLEX_VOCAB;
    for (int e = threadIdx.x; e < count[row]; e += blockDim.x) {
        int t = tok[row * stride + e];
        float a = add[row * stride + e];
        lr[t] = isinf(a) && a < 0.0f ? a : lr[t] + a;
    }
}

extern "C" __global__ void duplex_allow(float* logits, const int* allow, const int* count, int stride) {
    int row = blockIdx.x;
    int n = count[row];
    if (n == 0) return;
    float* lr = logits + (size_t)row * DUPLEX_VOCAB;
    const int* list = allow + row * stride;
    for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < DUPLEX_VOCAB; i += gridDim.y * blockDim.x) {
        bool keep = false;
        for (int j = 0; j < n; j++) keep |= list[j] == i;
        if (!keep) lr[i] = __int_as_float(0xff800000);
    }
}

extern "C" __global__ void duplex_user_latent(const float* mean, const float* noise, const float* scale, float* out, int rows, float k) {
    int n = rows * DUPLEX_LATENT;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] = mean[i] + scale[i / DUPLEX_LATENT] * k * noise[i];
}

extern "C" __global__ void duplex_state_move(float* pool, float* batch, const int* slots, const int* seg_off, const int* seg_len, int total, int rows, int to_batch) {
    int seg = blockIdx.x, row = blockIdx.y;
    int off = seg_off[seg], len = seg_len[seg];
    float* b = batch + (size_t)rows * off + (size_t)row * len;
    float* p = pool + (size_t)slots[row] * total + off;
    for (int j = threadIdx.x; j < len; j += blockDim.x) {
        if (to_batch) b[j] = p[j];
        else p[j] = b[j];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kernel_is_emitted_with_the_shape_constants() {
        let s = DuplexShape { hidden: 64, vocab: 512, latent: 64 };
        let src = emit_cuda(&s).unwrap();
        for k in KERNELS {
            assert!(src.contains(&format!("void {k}(")), "{k}");
        }
        assert!(src.contains("DUPLEX_VOCAB = 512"));
        assert!(emit_cuda(&DuplexShape { hidden: 0, vocab: 1, latent: 1 }).is_err());
        assert_ne!(s.hash(), DuplexShape { vocab: 513, ..s }.hash());
    }
}
