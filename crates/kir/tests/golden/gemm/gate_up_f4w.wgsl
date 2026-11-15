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
fn e2m1(v: u32) -> f32 {
    var mag = array<f32, 8>(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    return select(mag[v & 7u], -mag[v & 7u], (v & 8u) != 0u);
}
fn sbyte(i: u32) -> u32 {
    return (s[i / 4u] >> (8u * (i & 3u))) & 0xffu;
}
fn wv(n: u32, c: u32) -> f32 {
    let i = n * 5120u + c;
    let byte = (w[i / 8u] >> (8u * ((i / 2u) & 3u))) & 0xffu;
    let nib = select(byte & 15u, byte >> 4u, (i & 1u) == 1u);
    let sc = e4m3(sbyte(16u + n * 320u + c / 16u));
    return e2m1(nib) * sc * bitcast<f32>(s[0]);
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
    let wrow = min(wid.x * 16u + ly, 34815u);
    let xrow = min(m, max(tokens, 1u) - 1u);
    var acc = 0.0;
    for (var k0 = 0u; k0 < 5120u; k0 += 16u) {
        xs[ly][lx] = f16(x[xrow * 5120u + k0 + lx]);
        ws[ly][lx] = f16(wv(wrow, k0 + lx));
        workgroupBarrier();
        for (var i = 0u; i < 16u; i++) {
            acc += f32(xs[ly][i]) * f32(ws[lx][i]);
        }
        workgroupBarrier();
    }
    res[ly][lx] = acc;
    workgroupBarrier();
    if (m < tokens && col < 34816u && (lx & 1u) == 0u) {
        let g = res[ly][lx];
        out[m * 17408u + col / 2u] = (g / (1.0 + exp(-g))) * res[ly][lx + 1u];
    }
}
