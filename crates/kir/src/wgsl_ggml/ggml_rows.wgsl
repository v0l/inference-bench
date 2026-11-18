@group(0) @binding(0) var<storage, read> wdata: array<u32>;
@group(0) @binding(1) var<storage, read> rows: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
struct P { a: vec4<u32>, b: vec4<u32> };
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 8388608u;
    let nrows = p.a.x;
    let cols = p.a.y;
    if (idx >= nrows * cols) {
        return;
    }
    let r = idx / cols;
    let c = idx % cols;
    let entry = rows[r];
    out[idx] = dequant_at((entry >> 3u) * 16u, entry & 7u, c);
}
