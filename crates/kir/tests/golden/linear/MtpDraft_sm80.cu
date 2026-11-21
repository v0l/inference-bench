// target Cuda cc=8.0 sms=108 warp=32 threads=1024 smem=166912 graphs=true pdl=false features=[WarpShuffle,CpAsync,Mma16816,ShaderF16]
// linear MtpDraft threads=256 smem=0 chunk=64 ksplit=4 cols=16
//   LinearShape { key_heads: 16, value_heads: 48, head_dim: 128, conv_kernel: 4, tiled_value_heads: false }
extern "C" __global__ void mtp_fc_in(const bf16_t* table, const int* tokens, const float* xsrc, const float* psrc,
                                     const int* hidx, const float* pre, const float* ne, const float* nh, bf16_t* hi,
                                     bf16_t* lo) {
    __shared__ float scratch[32];
    int r = blockIdx.x;
    const bf16_t* e = table + (size_t)tokens[r] * HIDDEN;
    int at = hidx[r];
    const float* h = at >= 0 ? xsrc + (size_t)at * HIDDEN : psrc + (size_t)(-1 - at) * HIDDEN;
    float se = 0.0f, sx = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        float a = bf16_to_f32(e[i]);
        se += a * a;
        sx += h[i] * h[i];
    }
    float ie = rsqrtf(block_reduce<false>(se, scratch) / HIDDEN + EPS);
    float ix = rsqrtf(block_reduce<false>(sx, scratch) / HIDDEN + EPS);
    float sh = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        float g = pre[i] * (h[i] * ix);
        sh += g * g;
    }
    float ih = rsqrtf(block_reduce<false>(sh, scratch) / HIDDEN + EPS);
    size_t row = (size_t)r * 2 * HIDDEN;
    for (int i = 2 * threadIdx.x; i < HIDDEN; i += 2 * blockDim.x) {
        float g0 = pre[i] * (h[i] * ix), g1 = pre[i + 1] * (h[i + 1] * ix);
        split_store2(hi, lo, row + i, ne[i] * (bf16_to_f32(e[i]) * ie), ne[i + 1] * (bf16_to_f32(e[i + 1]) * ie));
        split_store2(hi, lo, row + HIDDEN + i, nh[i] * (g0 * ih), nh[i + 1] * (g1 * ih));
    }
}

extern "C" __global__ void mtp_draft(const float* logits, int* drafts, int col, int k, int* ctok, const int* dslots) {
    __shared__ float best_v[1024];
    __shared__ int best_i[1024];
    int s = blockIdx.x;
    const float* row = logits + (size_t)s * VOCAB;
    float bv = NEG_INF;
    int bi = 0;
    for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
        if (row[i] > bv) {
            bv = row[i];
            bi = i;
        }
    }
    best_v[threadIdx.x] = bv;
    best_i[threadIdx.x] = bi;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            float ov = best_v[threadIdx.x + w];
            int oi = best_i[threadIdx.x + w];
            if (ov > best_v[threadIdx.x] || (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x])) {
                best_v[threadIdx.x] = ov;
                best_i[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        drafts[dslots[s] * k + col] = best_i[0];
        ctok[s] = best_i[0];
    }
}

__device__ __forceinline__ float spec_uniform(unsigned seed, unsigned step) {
    unsigned x = seed * 747796405u + step * 2891336453u + 0x68e31da4u;
    x ^= x >> 16; x *= 2246822507u; x ^= x >> 13; x *= 3266489909u; x ^= x >> 16;
    return ((float)(x >> 8) + 0.5f) / 16777216.0f;
}

extern "C" __global__ void spec_accept(const float* logits, const int* vtok, const int* params, const int* max_accept,
                                       int k, int* out, int* accepted) {
    __shared__ float best_v[1024];
    __shared__ int best_i[1024];
    __shared__ float scratch[32];
    int s = blockIdx.x;
    const int* p = params + 6 * s;
    float inv_temp = __int_as_float(p[0]);
    unsigned seed = (unsigned)p[1];
    unsigned step = (unsigned)p[2];
    int a = 0;
    for (int j = 0; j <= k; j++) {
        const float* row = logits + (size_t)(s * (k + 1) + j) * VOCAB;
        int d = j < k ? vtok[s * (k + 1) + 1 + j] : -1;
        bool may = j < k && j < max_accept[s];
        bool sampled = inv_temp > 0.0f;
        bool take = false;
        float thr = NEG_INF;
        if (sampled) {
            float m;
            thr = sample_threshold(row, inv_temp, p[3], __int_as_float(p[4]), __int_as_float(p[5]), &m);
            if (may) {
                float sum = 0.0f;
                for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
                    float v = row[i] * inv_temp;
                    if (v >= thr) sum += expf(v - m);
                }
                sum = block_reduce<false>(sum, scratch);
                float vd = row[d] * inv_temp;
                take = vd >= thr && spec_uniform(seed, step + j) < expf(vd - m) / sum;
            }
        }
        int tok = d;
        if (!take) {
            int excl = sampled && may ? d : -1;
            float bv = NEG_INF;
            int bi = 0;
            for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
                if (i == excl) continue;
                float v = row[i];
                if (sampled) {
                    v *= inv_temp;
                    if (v < thr) continue;
                    v += gumbel(seed, step + j, (unsigned)i);
                }
                if (v > bv) {
                    bv = v;
                    bi = i;
                }
            }
            __syncthreads();
            best_v[threadIdx.x] = bv;
            best_i[threadIdx.x] = bi;
            __syncthreads();
            for (int w = blockDim.x / 2; w > 0; w >>= 1) {
                if (threadIdx.x < w) {
                    float ov = best_v[threadIdx.x + w];
                    int oi = best_i[threadIdx.x + w];
                    if (ov > best_v[threadIdx.x] || (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x])) {
                        best_v[threadIdx.x] = ov;
                        best_i[threadIdx.x] = oi;
                    }
                }
                __syncthreads();
            }
            tok = best_i[0];
            take = !sampled && may && tok == d;
        }
        if (threadIdx.x == 0) out[s * (k + 1) + j] = tok;
        __syncthreads();
        if (!take) break;
        a++;
    }
    if (threadIdx.x == 0) {
        accepted[s] = a;
        for (int j = a + 1; j <= k; j++) out[s * (k + 1) + j] = 0;
    }
}

extern "C" __global__ void mtp_fill(const int* drafts, int* vtok, int k) {
    int s = blockIdx.x;
    int j = threadIdx.x;
    if (j < k) vtok[s * (k + 1) + 1 + j] = drafts[s * k + j];
}

extern "C" __global__ void mtp_round_in(const int* rin, const int* drafts, int n, int k, int* vtok, int* vpos,
                                        int* vslots) {
    int s = blockIdx.x;
    int j = threadIdx.x;
    int slot = rin[2 * n + s];
    vtok[s * (k + 1) + j] = j == 0 ? rin[s] : drafts[slot * k + j - 1];
    vpos[s * (k + 1) + j] = rin[n + s] + j;
    vslots[s * (k + 1) + j] = slot;
}

extern "C" __global__ void mtp_chain_rows(const int* rin, int n, const int* accepted, int step, int* vpos,
                                          int* vslots) {
    int s = threadIdx.x;
    if (s < n) {
        vpos[s] = rin[n + s] + accepted[s] + step;
        vslots[s] = rin[2 * n + s];
    }
}

extern "C" __global__ void mtp_pick_rows(const int* accepted, int k, int* sidx) {
    int s = threadIdx.x;
    sidx[s] = s * (k + 1) + accepted[s];
}

extern "C" __global__ void mtp_copy(const float* src, const int* sidx, float* dst, const int* didx) {
    int s = blockIdx.y;
    const float* a = src + (size_t)sidx[s] * HIDDEN;
    float* b = dst + (size_t)didx[s] * HIDDEN;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < HIDDEN; i += gridDim.x * blockDim.x) b[i] = a[i];
}
