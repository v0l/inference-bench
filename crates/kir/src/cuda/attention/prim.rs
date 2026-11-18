pub struct Mask<'a> {
    pub pos: &'a str,
    pub live: &'a str,
}

pub fn zero_scores() -> &'static str {
    "        float s[SB][4];
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[nb][e] = 0.0f;
"
}

pub fn online_softmax(mask: &Mask) -> String {
    format!(
        "        float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {{
                int key = k0 + nb * 8 + 2 * t + (e & 1);
                int {pos} = e < 2 ? pos0 : pos1;
                bool live = {live};
                float v = live ? (SOFTCAP > 0.0f ? cap_score(s[nb][e] * (scale_log2 / LOG2E)) * LOG2E : s[nb][e] * scale_log2) : NEG_INF;
                s[nb][e] = v;
                if (e < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
            }}
{reduce}",
        pos = mask.pos,
        live = mask.live,
        reduce = softmax_update(),
    )
}

pub fn softmax_update() -> String {
    let quad_max = |m: &str| format!("        {m} = fmaxf({m}, __shfl_xor_sync(0xffffffff, {m}, 1));\n        {m} = fmaxf({m}, __shfl_xor_sync(0xffffffff, {m}, 2));\n");
    let quad_sum = |m: &str| format!("        {m} += __shfl_xor_sync(0xffffffff, {m}, 1);\n        {m} += __shfl_xor_sync(0xffffffff, {m}, 2);\n");
    let mut o = String::new();
    o.push_str(&quad_max("mx0"));
    o.push_str(&quad_max("mx1"));
    o.push_str(
        "        float n0 = fmaxf(m0, mx0);
        float n1 = fmaxf(m1, mx1);
        float a0 = m0 == NEG_INF ? 0.0f : fast_exp2(m0 - n0);
        float a1 = m1 == NEG_INF ? 0.0f : fast_exp2(m1 - n1);
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float nm = e < 2 ? n0 : n1;
                float p = s[nb][e] == NEG_INF ? 0.0f : fast_exp2(s[nb][e] - nm);
                s[nb][e] = p;
                if (e < 2) sum0 += p; else sum1 += p;
            }
",
    );
    o.push_str(&quad_sum("sum0"));
    o.push_str(&quad_sum("sum1"));
    o.push_str("        l0 = l0 * a0 + sum0;\n        l1 = l1 * a1 + sum1;\n        m0 = n0;\n        m1 = n1;\n");
    o
}

pub fn rescale_acc(blocks: &str) -> String {
    format!(
        "#pragma unroll
        for (int nb = 0; nb < {blocks}; nb++) {{
            o[nb][0] *= a0;
            o[nb][1] *= a0;
            o[nb][2] *= a1;
            o[nb][3] *= a1;
        }}
"
    )
}

pub fn pack_probs(low_plane: &str) -> String {
    format!(
        "            unsigned ph[4], pl[4];
            ph[0] = pack_hi(s[2 * kb][0], s[2 * kb][1]);
            ph[1] = pack_hi(s[2 * kb][2], s[2 * kb][3]);
            ph[2] = pack_hi(s[2 * kb + 1][0], s[2 * kb + 1][1]);
            ph[3] = pack_hi(s[2 * kb + 1][2], s[2 * kb + 1][3]);
            if ({low_plane}) {{
                pl[0] = pack_lo(s[2 * kb][0], s[2 * kb][1]);
                pl[1] = pack_lo(s[2 * kb][2], s[2 * kb][3]);
                pl[2] = pack_lo(s[2 * kb + 1][0], s[2 * kb + 1][1]);
                pl[3] = pack_lo(s[2 * kb + 1][2], s[2 * kb + 1][3]);
            }}
"
    )
}
