enable f16;
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
struct P { a: vec4<u32>, b: vec4<u32> };
@group(0) @binding(5) var<uniform> p: P;

fn e4m3(b: u32) -> f32 {
    return bitcast<f32>(((b & 0x80u) << 24u) | ((b & 0x7fu) << 20u)) * bitcast<f32>(0x7b800000u);
}
fn byte_at(i: u32) -> u32 {
    return (w[i / 4u] >> (8u * (i & 3u))) & 0xffu;
}
fn wv(n: u32, c: u32) -> f32 {
    let base = s[n];
    return e4m3(byte_at(n * 17408u + c)) * bitcast<f32>(s[base + c / 128u]);
}

var<workgroup> xs: array<array<f16, 16>, 16>;
var<workgroup> ws: array<array<f16, 16>, 16>;
var<workgroup> res: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tokens = p.a.x;
    let lx = lid.x;
    let ly = lid.y;
    let col = wid.x * 16u + lx;
    let m = wid.y * 16u + ly;
    let wrow = min(wid.x * 16u + ly, 5119u);
    let xrow = min(m, max(tokens, 1u) - 1u);
    var acc = 0.0;
    for (var k0 = 0u; k0 < 17408u; k0 += 16u) {
        xs[ly][lx] = f16(x[xrow * 17408u + k0 + lx]);
        ws[ly][lx] = f16(wv(wrow, k0 + lx));
        workgroupBarrier();
        for (var i = 0u; i < 16u; i++) {
            acc += f32(xs[ly][i]) * f32(ws[lx][i]);
        }
        workgroupBarrier();
    }
    if (m < tokens && col < 5120u) {
        var v = acc;
        let at = m * 5120u + col;
        v += out[at];
        out[at] = v;
    }
}
