const HIDDEN: u32 = 1024u;
const INTER: u32 = 3072u;
const VOCAB: u32 = 151936u;
const HEADS: u32 = 16u;
const KV_HEADS: u32 = 8u;
const HEAD_DIM: u32 = 128u;
const GROUP: u32 = 2u;
const QK_NORM: bool = true;
const EPS: f32 = 1e-6;
const THETA: f32 = 1e6;
const ROPE_KIND: u32 = 0u;
const ROPE_FACTOR: f32 = 4e0;
const ROPE_LOW_FREQ: f32 = 1e0;
const ROPE_HIGH_FREQ: f32 = 4e0;
const ROPE_ORIG_POS: f32 = 8.192e3;
const ROPE_MSCALE: f32 = 1e0;
const ACT_KIND: u32 = 0u;
const MAX_CTX: u32 = 4096u;
const Q_DIM: u32 = 2048u;
const QKV_ROW: u32 = 4096u;
const ROT: u32 = 128u;
const NEG_INF: f32 = -3.0e38;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> wdata: array<u32>;
@group(0) @binding(3) var<storage, read> rows: array<u32>;
struct P { a: vec4<u32>, b: vec4<u32> };
@group(0) @binding(4) var<uniform> p: P;

const TT: u32 = 1u;
const WG: u32 = 64u;
var<workgroup> red: array<f32, 64>;

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
    var acc: array<f32, 1>;
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
fn wbyte(i: u32) -> u32 {
    return (wdata[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
}

fn wi8(i: u32) -> f32 {
    return f32((i32(wbyte(i)) ^ 128) - 128);
}

fn whalf(i: u32) -> u32 {
    let w = wdata[i >> 2u];
    return select(w & 0xffffu, w >> 16u, (i & 2u) != 0u);
}

fn wf16(i: u32) -> f32 {
    return unpack2x16float(whalf(i)).x;
}

fn scale_min_k4(j: u32, sb: u32) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>(wbyte(sb + 4u + j) & 63u, wbyte(sb + 8u + j) & 63u);
    }
    let a = wbyte(sb + 4u + j + 4u);
    let lo = wbyte(sb + 4u + j - 4u);
    let hi = wbyte(sb + 4u + j);
    return vec2<u32>((a & 0xfu) | ((lo >> 6u) << 4u), (a >> 4u) | ((hi >> 6u) << 4u));
}

fn block_bytes(code: u32) -> u32 {
    switch (code) {
        case 0u: { return 144u; }
        case 1u: { return 176u; }
        case 2u: { return 224u; }
        case 3u: { return 272u; }
        default: { return 512u; }
    }
}

fn dequant_at(row_base: u32, code: u32, e: u32) -> f32 {
    let sb = row_base + (e / 256u) * block_bytes(code);
    let i = e % 256u;
    switch (code) {
        case 0u: {
            let d = wf16(sb);
            let dmin = wf16(sb + 2u);
            let j = i / 64u;
            let within = i % 64u;
            let l = within % 32u;
            let hi = within >= 32u;
            let sm = scale_min_k4(2u * j + select(0u, 1u, hi), sb);
            let q = wbyte(sb + 16u + 32u * j + l);
            let nib = select(q & 0xfu, q >> 4u, hi);
            let d1 = d * f32(sm.x);
            let mm = dmin * f32(sm.y);
            return d1 * f32(nib) - mm;
        }
        case 1u: {
            let d = wf16(sb);
            let dmin = wf16(sb + 2u);
            let j = i / 64u;
            let within = i % 64u;
            let l = within % 32u;
            let hi = within >= 32u;
            let sm = scale_min_k4(2u * j + select(0u, 1u, hi), sb);
            let q = wbyte(sb + 48u + 32u * j + l);
            let nib = select(q & 0xfu, q >> 4u, hi);
            let bit = select(1u << (2u * j), 2u << (2u * j), hi);
            let high = select(0u, 16u, (wbyte(sb + 16u + l) & bit) != 0u);
            let d1 = d * f32(sm.x);
            let mm = dmin * f32(sm.y);
            return d1 * f32(nib + high) - mm;
        }
        case 2u: {
            let d = wf16(sb + 208u);
            let hr = i / 128u;
            let r = i % 128u;
            let seg = r / 32u;
            let l = r % 32u;
            let ql = sb + 64u * hr;
            let qh = sb + 128u + 32u * hr;
            let sc = sb + 192u + 8u * hr;
            let isc = l / 16u;
            let lo_byte = wbyte(ql + l + select(0u, 32u, (seg & 1u) == 1u));
            let hi_byte = wbyte(qh + l);
            let nib = select(lo_byte & 0xfu, lo_byte >> 4u, seg >= 2u);
            let top = (hi_byte >> (2u * seg)) & 3u;
            let q = i32(nib | (top << 4u)) - 32;
            return d * wi8(sc + isc + 2u * seg) * f32(q);
        }
        case 3u: {
            let b = sb + (i / 32u) * 34u;
            let d = wf16(b);
            return d * wi8(b + 2u + (i % 32u));
        }
        case 4u: {
            return bitcast<f32>(whalf(sb + 2u * i) << 16u);
        }
        default: {
            return wf16(sb + 2u * i);
        }
    }
}
