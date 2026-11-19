use super::prim;

fn logits_norm() -> &'static str {
    r##"    __shared__ __align__(16) float xs[HIDDEN];
    __shared__ float red[32];
    int t = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    const float* xr = x + (size_t)t * HIDDEN;
    float ss = 0.0f;
    for (int i = tid * 4; i < HIDDEN; i += 1024) {
        float4 v = *(const float4*)(xr + i);
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < MOE_WARPS; w++) total += red[w];
    float inv = rsqrtf(total / HIDDEN + EPS);
    for (int i = tid; i < HIDDEN; i += 256) {
        float v = xr[i] * inv * ln[i];
        xs[i] = v;
        if (blockIdx.x == 0) xn[(size_t)t * HIDDEN + i] = v;
    }
    __syncthreads();
"##
}

fn logits_dot() -> &'static str {
    r##"    int e = blockIdx.x * MOE_LOGIT_ROWS + warp;
    if (e >= MOE_E + 1) return;
    float a = 0.0f;
    if (e < MOE_E) {
        const uint4* r = (const uint4*)(router + (size_t)e * HIDDEN);
        for (int c = lane; c < HIDDEN / 8; c += 32) {
            uint4 q = __ldg(r + c);
            const float* xv = xs + c * 8;
            a += bf16_to_f32(q.x & 0xffffu) * xv[0] + bf16_to_f32(q.x >> 16) * xv[1] + bf16_to_f32(q.y & 0xffffu) * xv[2] + bf16_to_f32(q.y >> 16) * xv[3] +
                 bf16_to_f32(q.z & 0xffffu) * xv[4] + bf16_to_f32(q.z >> 16) * xv[5] + bf16_to_f32(q.w & 0xffffu) * xv[6] + bf16_to_f32(q.w >> 16) * xv[7];
        }
    } else if (MOE_SHARED_GATE) {
        for (int i = lane; i < HIDDEN; i += 32) a += shared_gate[i] * xs[i];
    }
    a = warp_sum(a);
    if (lane == 0) logits[(size_t)t * (MOE_E + 1) + e] = a;
"##
}

pub fn route_logits() -> String {
    format!(
        "constexpr int MOE_LOGIT_ROWS = 8;\n\n{}",
        prim::global(Some(256), "moe_route_logits", "const float* x, const float* ln, const bf16_t* router, const float* shared_gate, float* xn, float* logits", &format!("{}{}", logits_norm(), logits_dot()))
    )
}

fn probabilities() -> &'static str {
    r##"    __shared__ float probs[MOE_E];
    __shared__ float sel[MOE_E];
    __shared__ float group_score[MOE_GROUPS];
    __shared__ bool group_kept[MOE_GROUPS];
    __shared__ float top_v[MOE_K];
    __shared__ int top_i[MOE_K];
    int t = blockIdx.x;
    int lane = threadIdx.x;
    const float* lg = logits + (size_t)t * (MOE_E + 1);
    float gate_logit = lg[MOE_E];
    if (MOE_SIGMOID) {
        for (int e = lane; e < MOE_E; e += 32) probs[e] = 1.0f / (1.0f + expf(-lg[e]));
    } else {
        float m = NEG_INF;
        for (int e = lane; e < MOE_E; e += 32) m = fmaxf(m, lg[e]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        float s = 0.0f;
        for (int e = lane; e < MOE_E; e += 32) {
            float p = expf(lg[e] - m);
            probs[e] = p;
            s += p;
        }
        s = warp_sum(s);
        for (int e = lane; e < MOE_E; e += 32) probs[e] /= s;
    }
    __syncwarp();
    for (int e = lane; e < MOE_E; e += 32) sel[e] = MOE_BIAS ? probs[e] + bias[e] : probs[e];
    __syncwarp();
    if (MOE_GROUPS > 1) {
        constexpr int PER = MOE_E / MOE_GROUPS;
        for (int g = lane; g < MOE_GROUPS; g += 32) {
            float a = -3.0e30f, b = -3.0e30f;
            for (int j = 0; j < PER; j++) {
                float v = sel[g * PER + j];
                if (v > a) { b = a; a = v; } else if (v > b) { b = v; }
            }
            group_score[g] = a + b;
            group_kept[g] = false;
        }
        __syncwarp();
        if (lane == 0) {
            for (int r = 0; r < MOE_TOPK_GROUP; r++) {
                int best = -1;
                for (int g = 0; g < MOE_GROUPS; g++) {
                    if (!group_kept[g] && (best < 0 || group_score[g] > group_score[best])) best = g;
                }
                group_kept[best] = true;
            }
        }
        __syncwarp();
        for (int e = lane; e < MOE_E; e += 32) {
            if (!group_kept[e / PER]) sel[e] = -1.0e30f;
        }
        __syncwarp();
    }
"##
}

fn top_k() -> &'static str {
    r##"    float picked = 0.0f;
    for (int k = 0; k < MOE_K; k++) {
        float bv = -3.0e30f;
        int bi = MOE_E;
        for (int e = lane; e < MOE_E; e += 32) {
            if (sel[e] > bv) { bv = sel[e]; bi = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, bv, o);
            int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { top_v[k] = probs[bi]; top_i[k] = bi; sel[bi] = -2.0e30f; }
        picked += probs[bi];
        __syncwarp();
    }
"##
}

fn emit_pairs() -> &'static str {
    r##"    for (int k = lane; k < MOE_K; k += 32) {
        int local = top_i[k] - first;
        bool mine = local >= 0 && local < MOE_LOCAL;
        idx[t * MOE_PAIRS + k] = mine ? local : -1;
        coef[t * MOE_PAIRS + k] = mine ? (MOE_NORM_TOPK ? top_v[k] / picked : top_v[k]) * MOE_ROUTED_SCALE : 0.0f;
    }
    if (MOE_SHARED == 1 && lane == 0) {
        bool mine = !MOE_PARALLEL || first == 0;
        idx[t * MOE_PAIRS + MOE_K] = mine ? MOE_LOCAL : -1;
        coef[t * MOE_PAIRS + MOE_K] = !mine ? 0.0f : MOE_SHARED_GATE ? 1.0f / (1.0f + expf(-gate_logit)) : 1.0f;
    }
    if (MOE_SHARED > 1) {
        for (int s = lane; s < MOE_SHARED; s += 32) {
            bool mine = !MOE_PARALLEL || first == 0;
            idx[t * MOE_PAIRS + MOE_K + s] = mine ? MOE_LOCAL + s : -1;
            coef[t * MOE_PAIRS + MOE_K + s] = !mine ? 0.0f : MOE_SHARED_GATE ? 1.0f / (1.0f + expf(-gate_logit)) : 1.0f;
        }
    }
"##
}

pub fn route() -> String {
    format!(
        "constexpr bool MOE_PARALLEL = MOE_LOCAL != MOE_E;\n\n{}",
        prim::global(Some(32), "moe_route", "const float* logits, const float* bias, int* idx, float* coef, int first", &format!("{}{}{}", probabilities(), top_k(), emit_pairs()))
    )
}

pub fn combine() -> String {
    prim::global(
        None,
        "moe_combine",
        "float* x, const float* out, const float* coef",
        r##"    int t = blockIdx.y;
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= HIDDEN) return;
    float acc = 0.0f;
    for (int s = 0; s < MOE_PAIRS; s++) {
        float c = coef[t * MOE_PAIRS + s];
        if (c != 0.0f) acc += c * out[((size_t)t * MOE_PAIRS + s) * HIDDEN + i];
    }
    x[(size_t)t * HIDDEN + i] += acc;
"##,
    )
}

pub fn permute() -> String {
    format!(
        "constexpr int MOE_TJ = 2;\nconstexpr int MOE_SLOTS = MOE_LOCAL + MOE_SHARED;\n\n{}",
        prim::global(
            Some(1024),
            "moe_permute",
            "const int* idx, int* offsets, int* sorted, int pairs",
            r##"    __shared__ int count[MOE_SLOTS + 1];
    __shared__ int cursor[MOE_SLOTS];
    for (int e = threadIdx.x; e <= MOE_SLOTS; e += blockDim.x) count[e] = 0;
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) if (idx[p] >= 0) atomicAdd(&count[idx[p]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int run = 0;
        for (int e = 0; e < MOE_SLOTS; e++) {
            int c = count[e];
            offsets[e] = run;
            cursor[e] = run;
            run += c;
        }
        offsets[MOE_SLOTS] = run;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        if (idx[p] < 0) continue;
        int at = atomicAdd(&cursor[idx[p]], 1);
        sorted[at] = p;
    }
"##
        )
    )
}
