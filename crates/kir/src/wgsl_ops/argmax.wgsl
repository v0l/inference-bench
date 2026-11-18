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
    let row = p.a.x * VOCAB;
    let inv_temp = bitcast<f32>(p.a.y);
    var bv = NEG_INF;
    var bi = 0u;
    for (var i = li; i < VOCAB; i += 256u) {
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
