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
