@group(0) @binding(0) var<storage, read> tokens: array<u32>;
@group(0) @binding(1) var<storage, read> wdata: array<u32>;
@group(0) @binding(2) var<storage, read> rows: array<u32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
struct P { a: vec4<u32>, b: vec4<u32> };
@group(0) @binding(4) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wid.x;
    let entry = rows[tokens[p.a.y + t]];
    let base = (entry >> 3u) * 16u;
    let code = entry & 7u;
    for (var c = li; c < HIDDEN; c += 256u) {
        x[(p.a.x + t) * HIDDEN + c] = dequant_at(base, code, c);
    }
}
