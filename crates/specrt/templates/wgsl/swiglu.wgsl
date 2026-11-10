@group(0) @binding(0) var<storage, read> gate_up: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
struct P { a: vec4<u32>, b: vec4<u32> };
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 8388608u;
    if (idx >= p.a.x * INTER) {
        return;
    }
    let t = idx / INTER;
    let i = idx % INTER;
    let g = gate_up[t * 2u * INTER + 2u * i];
    var a = g / (1.0 + exp(-g));
    if (ACT_KIND == 1u) {
        a = 0.5 * g * (1.0 + tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)));
    }
    out[idx] = a * gate_up[t * 2u * INTER + 2u * i + 1u];
}
