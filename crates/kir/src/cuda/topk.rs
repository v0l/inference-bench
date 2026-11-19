pub struct TopK<'a> {
    pub sel: &'a str,
    pub vals: &'a str,
    pub count: &'a str,
    pub k: &'a str,
    pub out_v: &'a str,
    pub out_i: &'a str,
    pub sum: &'a str,
    pub floor: &'a str,
    pub taken: &'a str,
}

pub fn warp_top_k(t: &TopK) -> String {
    let TopK { sel, vals, count, k, out_v, out_i, sum, floor, taken } = t;
    format!(
        r##"    float {sum} = 0.0f;
    for (int k = 0; k < {k}; k++) {{
        float bv = {floor};
        int bi = {count};
        for (int e = lane; e < {count}; e += 32) {{
            if ({sel}[e] > bv) {{ bv = {sel}[e]; bi = e; }}
        }}
        for (int o = 16; o > 0; o >>= 1) {{
            float ov = __shfl_xor_sync(0xffffffff, bv, o);
            int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {{ bv = ov; bi = oi; }}
        }}
        if (lane == 0) {{ {out_v}[k] = {vals}[bi]; {out_i}[k] = bi; {sel}[bi] = {taken}; }}
        {sum} += {vals}[bi];
        __syncwarp();
    }}
"##
    )
}
