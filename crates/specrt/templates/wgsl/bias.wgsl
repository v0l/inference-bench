@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
struct P { a: vec4<u32>, b: vec4<u32> };
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 8388608u;
    if (idx >= p.a.x * p.a.y) {
        return;
    }
    y[idx] = y[idx] + bias[idx % p.a.y];
}
