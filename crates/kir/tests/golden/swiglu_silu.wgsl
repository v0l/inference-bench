@group(0) @binding(0) var<storage, read> gate_up: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
struct P { a: vec4<u32>, b: vec4<u32> };
@group(0) @binding(2) var<uniform> p: P;

const WIDTH: u32 = 3072u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 8388608u;
    if (idx >= p.a.x * WIDTH) {
        return;
    }
    let t = idx / WIDTH;
    let i = idx % WIDTH;
    let g = gate_up[t * 2u * WIDTH + i * 2u];
    let a = g / (1.0 + exp(-g));
    out[idx] = a * gate_up[t * 2u * WIDTH + i * 2u + 1u];
}
