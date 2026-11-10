@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
struct P { a: vec4<u32>, b: vec4<u32> };
@group(0) @binding(3) var<uniform> p: P;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wid.x;
    let src = (p.a.x + t) * HIDDEN;
    let dst = (p.a.y + t) * HIDDEN;
    var ss = 0.0;
    for (var i = li; i < HIDDEN; i += 256u) {
        let v = x[src + i];
        ss += v * v;
    }
    red[li] = ss;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s) {
            red[li] += red[li + s];
        }
        workgroupBarrier();
    }
    let inv = inverseSqrt(red[0] / f32(HIDDEN) + EPS);
    for (var i = li; i < HIDDEN; i += 256u) {
        out[dst + i] = w[i] * (x[src + i] * inv);
    }
}
