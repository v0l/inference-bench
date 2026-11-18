const HIDDEN: u32 = 1024u;
const INTER: u32 = 3072u;
const VOCAB: u32 = 151936u;
const HEADS: u32 = 16u;
const KV_HEADS: u32 = 8u;
const HEAD_DIM: u32 = 128u;
const GROUP: u32 = 2u;
const QK_NORM: bool = true;
const EPS: f32 = 1e-6;
const THETA: f32 = 1e6;
const ROPE_KIND: u32 = 0u;
const ROPE_FACTOR: f32 = 4e0;
const ROPE_LOW_FREQ: f32 = 1e0;
const ROPE_HIGH_FREQ: f32 = 4e0;
const ROPE_ORIG_POS: f32 = 8.192e3;
const ROPE_MSCALE: f32 = 1e0;
const ACT_KIND: u32 = 0u;
const MAX_CTX: u32 = 4096u;
const Q_DIM: u32 = 2048u;
const QKV_ROW: u32 = 4096u;
const ROT: u32 = 128u;
const NEG_INF: f32 = -3.0e38;
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> tokens: array<u32>;
struct P { a: vec4<u32>, b: vec4<u32> };
@group(0) @binding(2) var<uniform> p: P;

var<workgroup> best_v: array<f32, 256>;
var<workgroup> best_i: array<u32, 256>;

fn gumbel(seed: u32, step: u32, i: u32) -> f32 {
    var x = seed * 747796405u + step * 2891336453u + i * 2246822519u + 1u;
    x ^= x >> 16u;
    x *= 2246822507u;
    x ^= x >> 13u;
    x *= 3266489909u;
    x ^= x >> 16u;
    let u = (f32(x >> 8u) + 0.5) / 16777216.0;
    return -log(-log(u));
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    let row = p.a.x * 151936u;
    let inv_temp = bitcast<f32>(p.a.y);
    var bv = -3.0e38;
    var bi = 0u;
    for (var i = li; i < 151936u; i += 256u) {
        var v = logits[row + i];
        if (inv_temp > 0.0) {
            v = v * inv_temp + gumbel(p.a.z, p.a.w, i);
        }
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    best_v[li] = bv;
    best_i[li] = bi;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s) {
            let ov = best_v[li + s];
            let oi = best_i[li + s];
            if (ov > best_v[li] || (ov == best_v[li] && oi < best_i[li])) {
                best_v[li] = ov;
                best_i[li] = oi;
            }
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        tokens[p.b.x] = best_i[0];
    }
}
