enable f16;
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
struct P { a: vec4<u32>, b: vec4<u32> };
@group(0) @binding(5) var<uniform> p: P;

fn wv(n: u32, c: u32) -> f32 {
    let i = n * 5120u + c;
    let word = w[i / 2u];
    return bitcast<f32>(select(word << 16u, word & 0xffff0000u, (i & 1u) == 1u));
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
    let wrow = min(wid.x * 16u + ly, 14335u);
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
    if (m < tokens && col < 14336u) {
        var v = acc + bias[col];
        let at = m * 14336u + col;
        out[at] = v;
    }
}
