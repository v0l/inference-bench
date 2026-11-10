typedef unsigned short bf16_t;

__device__ __forceinline__ float bf(bf16_t v) { return __uint_as_float(((unsigned)v) << 16); }

__device__ __forceinline__ float activate(float x, int kind) {
    if (kind == 1) {
        float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + tanhf(u));
    }
    if (kind == 2) {
        return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
    }
    return x;
}

extern "C" __global__ void vis_gemm(const float* A, const bf16_t* W, const bf16_t* bias, const float* resid, float* C,
                                    int N, int M, int K, int act) {
    __shared__ float As[16][65];
    __shared__ float Ws[16][65];
    int row0 = blockIdx.y * 64;
    int col0 = blockIdx.x * 64;
    int tid = threadIdx.x;
    int ty = tid / 16;
    int tx = tid % 16;
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = 0; i < 4; i++) {
            int e = tid + i * 256;
            int r = e / 16;
            int k = e % 16;
            int gk = k0 + k;
            As[k][r] = (row0 + r < N && gk < K) ? A[(size_t)(row0 + r) * K + gk] : 0.0f;
            Ws[k][r] = (col0 + r < M && gk < K) ? bf(W[(size_t)(col0 + r) * K + gk]) : 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < 16; k++) {
            float a[4], b[4];
            for (int i = 0; i < 4; i++) a[i] = As[k][ty * 4 + i];
            for (int j = 0; j < 4; j++) b[j] = Ws[k][tx * 4 + j];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) {
        int r = row0 + ty * 4 + i;
        if (r >= N) continue;
        for (int j = 0; j < 4; j++) {
            int c = col0 + tx * 4 + j;
            if (c >= M) continue;
            float v = acc[i][j];
            if (bias) v += bf(bias[c]);
            v = activate(v, act);
            if (resid) v += resid[(size_t)r * M + c];
            C[(size_t)r * M + c] = v;
        }
    }
}

extern "C" __global__ void vis_layernorm(const float* x, const bf16_t* w, const bf16_t* b, float* out, int width, float eps) {
    __shared__ float red[2][32];
    const float* row = x + (size_t)blockIdx.x * width;
    float s = 0.0f, s2 = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = row[i];
        s += v;
        s2 += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffff, s, o);
        s2 += __shfl_xor_sync(0xffffffff, s2, o);
    }
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) {
        red[0][warp] = s;
        red[1][warp] = s2;
    }
    __syncthreads();
    s = 0.0f;
    s2 = 0.0f;
    for (int i = 0; i < blockDim.x / 32; i++) {
        s += red[0][i];
        s2 += red[1][i];
    }
    float mean = s / width;
    float var = s2 / width - mean * mean;
    float inv = rsqrtf(fmaxf(var, 0.0f) + eps);
    float* o = out + (size_t)blockIdx.x * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        o[i] = (row[i] - mean) * inv * bf(w[i]) + bf(b[i]);
    }
}

extern "C" __global__ void vis_rope(float* qkv, const float* cosv, const float* sinv, int heads, int head_dim) {
    int tok = blockIdx.x;
    int h = blockIdx.y;
    int half = head_dim / 2;
    size_t width = (size_t)3 * heads * head_dim;
    for (int part = 0; part < 2; part++) {
        float* base = qkv + tok * width + (size_t)part * heads * head_dim + (size_t)h * head_dim;
        for (int i = threadIdx.x; i < half; i += blockDim.x) {
            float x1 = base[i], x2 = base[i + half];
            float c1 = cosv[(size_t)tok * head_dim + i], s1 = sinv[(size_t)tok * head_dim + i];
            float c2 = cosv[(size_t)tok * head_dim + i + half], s2 = sinv[(size_t)tok * head_dim + i + half];
            base[i] = x1 * c1 - x2 * s1;
            base[i + half] = x2 * c2 + x1 * s2;
        }
    }
}

constexpr int ATT_CHUNK = 2048;

extern "C" __global__ void vis_attention(const float* qkv, float* out, const int* seg_lo, const int* seg_hi, int heads, int head_dim, float scale) {
    __shared__ float p[ATT_CHUNK];
    __shared__ float red[32];
    __shared__ float stat[2];
    int tok = blockIdx.x;
    int h = blockIdx.y;
    int lo = seg_lo[tok], hi = seg_hi[tok];
    size_t width = (size_t)3 * heads * head_dim;
    const float* q = qkv + tok * width + (size_t)h * head_dim;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int warps = blockDim.x / 32;
    float acc = 0.0f;
    if (threadIdx.x == 0) {
        stat[0] = -3.0e38f;
        stat[1] = 0.0f;
    }
    __syncthreads();
    for (int c0 = lo; c0 < hi; c0 += ATT_CHUNK) {
        int n = min(ATT_CHUNK, hi - c0);
        float lmax = -3.0e38f;
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            const float* k = qkv + (size_t)(c0 + j) * width + (size_t)heads * head_dim + (size_t)h * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; d++) s = fmaf(q[d], k[d], s);
            s *= scale;
            p[j] = s;
            lmax = fmaxf(lmax, s);
        }
        for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, o));
        if (lane == 0) red[warp] = lmax;
        __syncthreads();
        float cmax = -3.0e38f;
        for (int i = 0; i < warps; i++) cmax = fmaxf(cmax, red[i]);
        float old_max = stat[0];
        float new_max = fmaxf(old_max, cmax);
        float rescale = old_max == -3.0e38f ? 0.0f : __expf(old_max - new_max);
        __syncthreads();
        float lsum = 0.0f;
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            float e = __expf(p[j] - new_max);
            p[j] = e;
            lsum += e;
        }
        for (int o = 16; o > 0; o >>= 1) lsum += __shfl_xor_sync(0xffffffff, lsum, o);
        if (lane == 0) red[warp] = lsum;
        __syncthreads();
        float csum = 0.0f;
        for (int i = 0; i < warps; i++) csum += red[i];
        acc *= rescale;
        if (threadIdx.x < head_dim) {
            const float* v = qkv + (size_t)c0 * width + (size_t)2 * heads * head_dim + (size_t)h * head_dim + threadIdx.x;
            float a = 0.0f;
            for (int j = 0; j < n; j++) a = fmaf(p[j], v[(size_t)j * width], a);
            acc += a;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            stat[1] = stat[1] * rescale + csum;
            stat[0] = new_max;
        }
        __syncthreads();
    }
    if (threadIdx.x < head_dim) {
        out[(size_t)tok * heads * head_dim + (size_t)h * head_dim + threadIdx.x] = acc / stat[1];
    }
}
