@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> wdata: array<u32>;
@group(0) @binding(3) var<storage, read> rows: array<u32>;
struct P { a: vec4<u32>, b: vec4<u32> };
@group(0) @binding(4) var<uniform> p: P;

const TT: u32 = {{TT}}u;
const WG: u32 = 64u;
var<workgroup> red: array<f32, {{RED}}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let nrows = p.a.x;
    let cols = p.a.y;
    let tokens = p.a.z;
    let x_row0 = p.a.w;
    let y_row0 = p.b.x;
    let accumulate = p.b.y;
    let r = min(wid.x + wid.y * 32768u, nrows - 1u);
    let valid = (wid.x + wid.y * 32768u) < nrows;
    let t0 = wid.z * TT;
    let entry = rows[r];
    let base = (entry >> 3u) * 16u;
    let code = entry & 7u;
    var acc: array<f32, {{TT}}>;
    for (var c = li; c < cols; c += WG) {
        let w = dequant_at(base, code, c);
        for (var j = 0u; j < TT; j++) {
            let t = min(t0 + j, tokens - 1u);
            acc[j] += w * x[(x_row0 + t) * cols + c];
        }
    }
    for (var j = 0u; j < TT; j++) {
        red[j * WG + li] = acc[j];
    }
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        if (li < s) {
            for (var j = 0u; j < TT; j++) {
                red[j * WG + li] += red[j * WG + li + s];
            }
        }
        workgroupBarrier();
    }
    if (li < TT && valid && t0 + li < tokens) {
        let at = (y_row0 + t0 + li) * nrows + r;
        let v = red[li * WG];
        if (accumulate == 1u) {
            y[at] = y[at] + v;
        } else {
            y[at] = v;
        }
    }
}
