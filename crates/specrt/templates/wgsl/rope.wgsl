@group(0) @binding(0) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> q_norm: array<f32>;
@group(0) @binding(2) var<storage, read> k_norm: array<f32>;
@group(0) @binding(3) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> v_cache: array<f32>;
struct P { a: vec4<u32>, b: vec4<u32> };
@group(0) @binding(5) var<uniform> p: P;

var<workgroup> red: array<f32, 128>;
var<workgroup> vec_s: array<f32, HEAD_DIM>;

fn inv_freq(i: u32) -> f32 {
    var f = 1.0 / pow(THETA, f32(2u * i) / f32(ROT));
    if (ROPE_KIND == 1u) {
        f = f / ROPE_FACTOR;
    } else if (ROPE_KIND == 2u) {
        let wavelen = 6.283185307179586 / f;
        let low = ROPE_ORIG_POS / ROPE_LOW_FREQ;
        let high = ROPE_ORIG_POS / ROPE_HIGH_FREQ;
        if (wavelen > low) {
            f = f / ROPE_FACTOR;
        } else if (wavelen >= high) {
            let sm = (ROPE_ORIG_POS / wavelen - ROPE_LOW_FREQ) / (ROPE_HIGH_FREQ - ROPE_LOW_FREQ);
            f = (1.0 - sm) * f / ROPE_FACTOR + sm * f;
        }
    }
    return f;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let head = wid.x;
    let row = wid.y;
    let pos = p.a.x + row;
    
    let is_q = head < HEADS;
    let is_k = !is_q && head < HEADS + KV_HEADS;
    let at = row * QKV_ROW + head * HEAD_DIM;
    var ss = 0.0;
    for (var d = li; d < HEAD_DIM; d += 128u) {
        let v = qkv[at + d];
        vec_s[d] = v;
        ss += v * v;
    }
    red[li] = ss;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s >>= 1u) {
        if (li < s) {
            red[li] += red[li + s];
        }
        workgroupBarrier();
    }
    let inv = inverseSqrt(red[0] / f32(HEAD_DIM) + EPS);
    workgroupBarrier();
    if (QK_NORM && (is_q || is_k)) {
        for (var d = li; d < HEAD_DIM; d += 128u) {
            let w = select(k_norm[d], q_norm[d], is_q);
            vec_s[d] = w * (vec_s[d] * inv);
        }
        workgroupBarrier();
    }
    let hr = ROT / 2u;
    for (var d = li; d < HEAD_DIM; d += 128u) {
        var v = vec_s[d];
        if ((is_q || is_k) && d < ROT) {
            let i = select(d - hr, d, d < hr);
            let angle = f32(pos) * inv_freq(i);
            let c = cos(angle);
            let s = sin(angle);
            if (d < hr) {
                v = vec_s[d] * c - vec_s[d + hr] * s;
            } else {
                v = vec_s[d] * c + vec_s[d - hr] * s;
            }
        }
        if (is_q) {
            qkv[at + d] = v;
        } else {
            let kvh = select(head - HEADS - KV_HEADS, head - HEADS, is_k);
            let slot = (kvh * MAX_CTX + pos) * HEAD_DIM + d;
            if (is_k) {
                k_cache[slot] = v;
            } else {
                v_cache[slot] = v;
            }
        }
    }
}
