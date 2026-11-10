@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
struct P { a: vec4<u32>, b: vec4<u32> };
@group(0) @binding(4) var<uniform> p: P;

const TILE: u32 = 128u;
var<workgroup> qs: array<f32, HEAD_DIM>;
var<workgroup> ps: array<f32, 128>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let head = wid.x;
    let t = wid.y;
    let kvh = head / GROUP;
    let len = p.a.x + t + 1u;
    
    let head_base = kvh * MAX_CTX * HEAD_DIM;
    for (var d = li; d < HEAD_DIM; d += TILE) {
        qs[d] = qkv[t * QKV_ROW + head * HEAD_DIM + d];
    }
    workgroupBarrier();
    let scale = inverseSqrt(f32(HEAD_DIM));
    var m = NEG_INF;
    var l = 0.0;
    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var t0 = 0u; t0 < len; t0 += TILE) {
        let key = t0 + li;
        var s = NEG_INF;
        if (key < len) {
            var dot = 0.0;
            let kb = head_base + key * HEAD_DIM;
            for (var d = 0u; d < HEAD_DIM; d++) {
                dot += qs[d] * k_cache[kb + d];
            }
            s = dot * scale;
        }
        red[li] = s;
        workgroupBarrier();
        for (var h = 64u; h > 0u; h >>= 1u) {
            if (li < h) {
                red[li] = max(red[li], red[li + h]);
            }
            workgroupBarrier();
        }
        let m_new = max(m, red[0]);
        let corr = exp(m - m_new);
        ps[li] = select(0.0, exp(s - m_new), key < len);
        workgroupBarrier();
        let n = min(TILE, len - t0);
        var psum = 0.0;
        var a0 = 0.0;
        var a1 = 0.0;
        for (var i = 0u; i < n; i++) {
            let pi = ps[i];
            psum += pi;
            let vb = head_base + (t0 + i) * HEAD_DIM;
            a0 += pi * v_cache[vb + min(li, HEAD_DIM - 1u)];
            if (HEAD_DIM > TILE) {
                a1 += pi * v_cache[vb + min(li + TILE, HEAD_DIM - 1u)];
            }
        }
        l = l * corr + psum;
        acc0 = acc0 * corr + a0;
        acc1 = acc1 * corr + a1;
        m = m_new;
        workgroupBarrier();
    }
    let ob = t * HEADS * HEAD_DIM + head * HEAD_DIM;
    if (li < HEAD_DIM) {
        out[ob + li] = acc0 / l;
    }
    if (HEAD_DIM > TILE && li + TILE < HEAD_DIM) {
        out[ob + li + TILE] = acc1 / l;
    }
}
