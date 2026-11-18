extern "C" __global__ void rms_split(const float* x, const float* weight, bf16_t* hi, bf16_t* lo) {
    __shared__ float scratch[32];
    size_t row = (size_t)blockIdx.x * 1024;
    const float4* xv = (const float4*)(x + row);
    float ss = 0.0f;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        float4 v = xv[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    float inv = rsqrtf(block_reduce<false>(ss, scratch) / 1024 + 1e-6f);
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        float4 v = xv[i];
        float4 nv = ((const float4*)weight)[i];
        split_store2(hi, lo, row + 4 * i, nv.x * (v.x * inv), nv.y * (v.y * inv));
        split_store2(hi, lo, row + 4 * i + 2, nv.z * (v.z * inv), nv.w * (v.w * inv));
    }
}

extern "C" __global__ void embed(const bf16_t* table, const int* token, float* x) {
    const bf16_t* row = table + (size_t)token[blockIdx.y] * 1024;
    x += (size_t)blockIdx.y * 1024;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 1024; i += gridDim.x * blockDim.x)
        x[i] = bf16_to_f32(row[i]) * 1e0f;
}

extern "C" __global__ void splice_rows(float* x, const int* rows, const float* embeds) {
    int row = rows[blockIdx.y];
    if (row < 0) return;
    float* dst = x + (size_t)blockIdx.y * 1024;
    const float* src = embeds + (size_t)row * 1024;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 1024; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

__device__ float gumbel(unsigned seed, unsigned step, unsigned i) {
    unsigned x = seed * 747796405u + step * 2891336453u + i * 2246822519u + 1u;
    x ^= x >> 16; x *= 2246822507u; x ^= x >> 13; x *= 3266489909u; x ^= x >> 16;
    float u = ((float)(x >> 8) + 0.5f) / 16777216.0f;
    return -logf(-logf(u));
}


__device__ __forceinline__ unsigned order_key(float v) {
    unsigned b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ float order_value(unsigned k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

__device__ float sample_threshold(const float* row, float inv_temp, int top_k, float top_p, float min_p, float* row_max) {
    __shared__ unsigned hist[2048];
    __shared__ unsigned chunk[64];
    __shared__ float scratch[32];
    __shared__ unsigned sh_prefix;
    __shared__ int sh_need;
    float m = NEG_INF;
    for (int i = threadIdx.x; i < 151936; i += blockDim.x) m = fmaxf(m, row[i] * inv_temp);
    m = block_reduce<true>(m, scratch);
    *row_max = m;
    bool use_k = top_k > 0 && top_k < 151936;
    bool use_p = top_p < 1.0f;
    float thr = NEG_INF;
    if (use_k) {
        unsigned prefix = 0, mask = 0;
        int need = top_k;
        for (int pass = 0; pass < 3; pass++) {
            int shift = pass == 0 ? 21 : (pass == 1 ? 10 : 0);
            unsigned width_mask = pass == 2 ? 1023u : 2047u;
            for (int i = threadIdx.x; i < 2048; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int i = threadIdx.x; i < 151936; i += blockDim.x) {
                unsigned key = order_key(row[i] * inv_temp);
                if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & width_mask], 1u);
            }
            __syncthreads();
            if (threadIdx.x < 64) {
                unsigned c = 0;
                for (int j = 0; j < 32; j++) c += hist[threadIdx.x * 32 + j];
                chunk[threadIdx.x] = c;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                int c = need;
                int ch = 63;
                for (; ch > 0; ch--) {
                    if ((int)chunk[ch] >= c) break;
                    c -= (int)chunk[ch];
                }
                int b = ch * 32 + 31;
                for (; b > ch * 32; b--) {
                    if ((int)hist[b] >= c) break;
                    c -= (int)hist[b];
                }
                sh_need = c;
                sh_prefix = prefix | ((unsigned)b << shift);
            }
            __syncthreads();
            need = sh_need;
            prefix = sh_prefix;
            mask |= width_mask << shift;
        }
        thr = order_value(prefix);
    }
    if (use_p) {
        __shared__ float cand[1024];
        __shared__ int ncand;
        if (threadIdx.x == 0) ncand = 0;
        __syncthreads();
        if (use_k) {
            for (int i = threadIdx.x; i < 151936; i += blockDim.x) {
                float v = row[i] * inv_temp;
                if (v >= thr) {
                    int at = atomicAdd(&ncand, 1);
                    if (at < 1024) cand[at] = v;
                }
            }
            __syncthreads();
        }
        bool cached = use_k && ncand <= 1024;
        int count = cached ? ncand : 0;
        float z = 0.0f;
        if (cached) {
            for (int i = threadIdx.x; i < count; i += blockDim.x) z += expf(cand[i] - m);
        } else {
            for (int i = threadIdx.x; i < 151936; i += blockDim.x) {
                float v = row[i] * inv_temp;
                if (v >= thr) z += expf(v - m);
            }
        }
        z = block_reduce<false>(z, scratch);
        float target = fmaxf(top_p, 1e-9f) * z;
        float lo = use_k ? thr : m - 88.0f;
        float hi = nextafterf(m, -NEG_INF);
        for (int it = 0; it < 48; it++) {
            float mid = 0.5f * (lo + hi);
            if (mid <= lo || mid >= hi) break;
            float mass = 0.0f;
            if (cached) {
                for (int i = threadIdx.x; i < count; i += blockDim.x) if (cand[i] >= mid) mass += expf(cand[i] - m);
            } else {
                for (int i = threadIdx.x; i < 151936; i += blockDim.x) {
                    float v = row[i] * inv_temp;
                    if (v >= mid) mass += expf(v - m);
                }
            }
            mass = block_reduce<false>(mass, scratch);
            if (mass >= target) lo = mid; else hi = mid;
        }
        float low = NEG_INF;
        if (cached) {
            for (int i = threadIdx.x; i < count; i += blockDim.x) if (cand[i] >= lo) low = fmaxf(low, -cand[i]);
        } else {
            for (int i = threadIdx.x; i < 151936; i += blockDim.x) {
                float v = row[i] * inv_temp;
                if (v >= lo) low = fmaxf(low, -v);
            }
        }
        thr = fmaxf(thr, -block_reduce<true>(low, scratch));
    }
    if (min_p > 0.0f) thr = fmaxf(thr, m + logf(min_p));
    return thr;
}

extern "C" __global__ void argmax_rows(const float* logits, int* tokens, int* history, const int* count, int stride) {
    __shared__ float best_v[1024];
    __shared__ int best_i[1024];
    logits += (size_t)blockIdx.x * 151936;
    float bv = NEG_INF;
    int bi = 0;
    for (int i = threadIdx.x; i < 151936; i += blockDim.x) {
        float v = logits[i];
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    best_v[threadIdx.x] = bv;
    best_i[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float ov = best_v[threadIdx.x + s];
            int oi = best_i[threadIdx.x + s];
            if (ov > best_v[threadIdx.x] || (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x])) {
                best_v[threadIdx.x] = ov;
                best_i[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        tokens[blockIdx.x] = best_i[0];
        history[(size_t)blockIdx.x * stride + count[0]] = best_i[0];
    }
}

extern "C" __global__ void apply_mask(float* logits, const unsigned* mask, const int* flags, int words) {
    int row = blockIdx.x;
    if (!flags[row]) return;
    logits += (size_t)row * 151936;
    mask += (size_t)row * words;
    for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < 151936; i += gridDim.y * blockDim.x) {
        if (!((mask[i >> 5] >> (i & 31)) & 1u)) logits[i] = NEG_INF;
    }
}

extern "C" __global__ void sample_rows(const float* logits, int* tokens, const int* params) {
    __shared__ float best_v[1024];
    __shared__ int best_i[1024];
    logits += (size_t)blockIdx.x * 151936;
    const int* p = params + 6 * blockIdx.x;
    float inv_temp = __int_as_float(p[0]);
    unsigned seed = (unsigned)p[1];
    unsigned step = (unsigned)p[2];
    float thr = NEG_INF;
    if (inv_temp > 0.0f) {
        float m;
        thr = sample_threshold(logits, inv_temp, p[3], __int_as_float(p[4]), __int_as_float(p[5]), &m);
    }
    float bv = NEG_INF;
    int bi = 0;
    for (int i = threadIdx.x; i < 151936; i += blockDim.x) {
        float v = logits[i];
        if (inv_temp > 0.0f) {
            v *= inv_temp;
            if (v < thr) continue;
            v += gumbel(seed, step, (unsigned)i);
        }
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    best_v[threadIdx.x] = bv;
    best_i[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float ov = best_v[threadIdx.x + s];
            int oi = best_i[threadIdx.x + s];
            if (ov > best_v[threadIdx.x] || (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x])) {
                best_v[threadIdx.x] = ov;
                best_i[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) tokens[blockIdx.x] = best_i[0];
}

extern "C" __global__ void argmax_advance(const float* logits, int* token, int* pos, int* history, int* count, float inv_temp, unsigned seed, int top_k, float top_p, float min_p) {
    __shared__ float best_v[1024];
    __shared__ int best_i[1024];
    float thr = NEG_INF;
    if (inv_temp > 0.0f) {
        float m;
        thr = sample_threshold(logits, inv_temp, top_k, top_p, min_p, &m);
    }
    float bv = NEG_INF;
    int bi = 0;
    for (int i = threadIdx.x; i < 151936; i += blockDim.x) {
        float v = logits[i];
        if (inv_temp > 0.0f) {
            v *= inv_temp;
            if (v < thr) continue;
            v += gumbel(seed, (unsigned)count[0], (unsigned)i);
        }
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    best_v[threadIdx.x] = bv;
    best_i[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float ov = best_v[threadIdx.x + s];
            int oi = best_i[threadIdx.x + s];
            if (ov > best_v[threadIdx.x] || (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x])) {
                best_v[threadIdx.x] = ov;
                best_i[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        token[0] = best_i[0];
        history[count[0]] = best_i[0];
        count[0] += 1;
        pos[0] += 1;
    }
}

extern "C" __global__ void bump(int* pos, int* count) {
    pos[threadIdx.x] += 1;
    if (threadIdx.x == 0) count[0] += 1;
}

extern "C" __global__ void copy_rows(const float* src, float* dst, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

extern "C" __global__ void split_rows(const float* src, bf16_t* hi, bf16_t* lo, int n) {
    for (int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x); i < n; i += 2 * gridDim.x * blockDim.x)
        split_store2(hi, lo, i, src[i], src[i + 1]);
}

