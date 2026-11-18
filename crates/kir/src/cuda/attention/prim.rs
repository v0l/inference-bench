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

pub const SCORE_SOFTCAP: &str = "(SOFTCAP > 0.0f ? cap_score(s[nb][e] * (scale_log2 / LOG2E)) * LOG2E : s[nb][e] * scale_log2)";

pub struct Scores<'a> {
    pub neg: &'a str,
    pub binds: &'a str,
    pub live: &'a str,
    pub score: &'a str,
}

pub fn online_softmax(mask: &Mask) -> String {
    let binds = format!("                int {pos} = e < 2 ? pos0 : pos1;\n                bool live = {live};\n", pos = mask.pos, live = mask.live);
    online_softmax_with(&Scores { neg: "NEG_INF", binds: &binds, live: "live", score: SCORE_SOFTCAP })
}

pub fn online_softmax_with(sc: &Scores) -> String {
    format!(
        "        float mx0 = {neg}, mx1 = {neg};
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {{
                int key = k0 + nb * 8 + 2 * t + (e & 1);
{binds}                float v = {live} ? {score} : {neg};
                s[nb][e] = v;
                if (e < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
            }}
{reduce}",
        neg = sc.neg,
        binds = sc.binds,
        live = sc.live,
        score = sc.score,
        reduce = softmax_update_with(sc.neg),
    )
}

pub fn softmax_update_with(neg: &str) -> String {
    let quad_max = |m: &str| xor_reduce(Reduce::Max, m, &[1, 2]);
    let quad_sum = |m: &str| xor_reduce(Reduce::Sum, m, &[1, 2]);
    let mut o = String::new();
    o.push_str(&quad_max("mx0"));
    o.push_str(&quad_max("mx1"));
    o.push_str(&format!(
        "        float n0 = fmaxf(m0, mx0);
        float n1 = fmaxf(m1, mx1);
        float a0 = m0 == {neg} ? 0.0f : fast_exp2(m0 - n0);
        float a1 = m1 == {neg} ? 0.0f : fast_exp2(m1 - n1);
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {{
                float nm = e < 2 ? n0 : n1;
                float p = s[nb][e] == {neg} ? 0.0f : fast_exp2(s[nb][e] - nm);
                s[nb][e] = p;
                if (e < 2) sum0 += p; else sum1 += p;
            }}
"
    ));
    o.push_str(&quad_sum("sum0"));
    o.push_str(&quad_sum("sum1"));
    o.push_str("        l0 = l0 * a0 + sum0;\n        l1 = l1 * a1 + sum1;\n        m0 = n0;\n        m1 = n1;\n");
    o
}

pub fn rescale_acc(var: &str, blocks: &str) -> String {
    format!(
        "#pragma unroll
        for (int {var} = 0; {var} < {blocks}; {var}++) {{
            o[{var}][0] *= a0;
            o[{var}][1] *= a0;
            o[{var}][2] *= a1;
            o[{var}][3] *= a1;
        }}
"
    )
}

pub fn pack_probs(low_plane: &str) -> String {
    let hi = pack_rows("            ", "ph", "pack_hi");
    let lo = pack_rows("                ", "pl", "pack_lo");
    format!("            unsigned ph[4], pl[4];\n{hi}            if ({low_plane}) {{\n{lo}            }}\n")
}

pub fn pack_rows(indent: &str, dst: &str, f: &str) -> String {
    let mut s = String::new();
    for (i, (row, c)) in [("2 * kb", 0), ("2 * kb", 2), ("2 * kb + 1", 0), ("2 * kb + 1", 2)].iter().enumerate() {
        s.push_str(&format!("{indent}{dst}[{i}] = {f}(s[{row}][{c}], s[{row}][{}]);\n", c + 1));
    }
    s
}

pub struct TileLoad<'a> {
    pub table: &'a str,
    pub bound: &'a str,
    pub stride: &'a str,
    pub unroll: bool,
}

pub fn load_kv_planes(l: &TileLoad) -> String {
    let unroll = if l.unroll { "#pragma unroll\n" } else { "" };
    format!(
        "        constexpr int ROW_CHUNKS = KV_PLANES * CPR;
        constexpr int TOTAL = 2 * {bk} * ROW_CHUNKS;
{unroll}        for (int i = tid; i < TOTAL; i += {stride}) {{
            int which = i / ({bk} * ROW_CHUNKS);
            int rem = i % ({bk} * ROW_CHUNKS);
            int r = rem / ROW_CHUNKS;
            int c = rem % ROW_CHUNKS;
            int tk = k0 + r;
            {src}
            bf16_t* dst = st + (which * KV_PLANES + c / CPR) * PLANE + swz<CPR>(r, c % CPR) * 8;
            cp_async16_zfill(dst, src + c * 8, tk < {bound});
        }}
",
        bk = if l.stride.starts_with("32") { "FM_BK" } else { "DM_BK" },
        stride = l.stride,
        bound = l.bound,
        src = if l.stride.starts_with("32") {
            format!("const bf16_t* src = kv_row(which ? v_cache : k_cache, {}, layer, kv_head, min(tk, {} - 1));", l.table, l.bound)
        } else {
            format!("int at = min(tk, {} - 1);\n            const bf16_t* src = kv_row(which ? v_cache : k_cache, {}, layer, kv_head, at);", l.bound, l.table)
        },
    )
}

pub fn qk_scores(key_blocks: &str, mma: &str, k_frag: &str) -> String {
    format!(
        "#pragma unroll
        for (int np = 0; np < SB / 2; np++)
#pragma unroll
            for (int kb = 0; kb < {key_blocks}; kb++) {{
                unsigned bh[4], bl[4] = {{0, 0, 0, 0}};
                int kr = np * 16 + (lane >> 4) * 8 + (lane & 7);
                int at = swz<CPR>(kr, kb * 2 + ((lane >> 3) & 1)) * 8;
{k_frag}                if (KV_PLANES == 2) ldsm_x4(bl, k_lo + at);
                {mma}(s[2 * np], qh[kb], ql[kb], bh[0], bh[1], bl[0], bl[1]);
                {mma}(s[2 * np + 1], qh[kb], ql[kb], bh[2], bh[3], bl[2], bl[3]);
            }}
"
    )
}

pub fn pv_accumulate(out_blocks: &str, mma: &str, v_frag: &str) -> String {
    format!(
        "#pragma unroll
            for (int np = 0; np < {out_blocks} / 2; np++) {{
                unsigned bh[4], bl[4] = {{0, 0, 0, 0}};
                int kr = kb * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                int at = swz<CPR>(kr, np * 2 + (lane >> 4)) * 8;
{v_frag}                if (KV_PLANES == 2) ldsm_x4_t(bl, v_lo + at);
                {mma}(o[2 * np], ph, pl, bh[0], bh[1], bl[0], bl[1]);
                {mma}(o[2 * np + 1], ph, pl, bh[2], bh[3], bl[2], bl[3]);
            }}
"
    )
}

#[derive(Clone, Copy)]
pub enum Reduce {
    Max,
    Sum,
}

pub fn xor_reduce(op: Reduce, var: &str, masks: &[u32]) -> String {
    masks
        .iter()
        .map(|m| match op {
            Reduce::Max => format!("        {var} = fmaxf({var}, __shfl_xor_sync(0xffffffff, {var}, {m}));\n"),
            Reduce::Sum => format!("        {var} += __shfl_xor_sync(0xffffffff, {var}, {m});\n"),
        })
        .collect()
}

pub fn warp_reduce(op: Reduce, expr: &str) -> String {
    match op {
        Reduce::Max => format!("warp_max({expr})"),
        Reduce::Sum => format!("warp_sum({expr})"),
    }
}

pub fn block_reduce(op: Reduce, expr: &str, scratch: &str) -> String {
    let max = matches!(op, Reduce::Max);
    format!("block_reduce<{max}>({expr}, {scratch})")
}

pub fn max_over(indent: usize, var: &str, idx: &str, count: &str, expr: &str) -> String {
    let pad = " ".repeat(indent);
    format!("{pad}float {var} = NEG_INF;\n{pad}for (int {idx} = 0; {idx} < {count}; {idx}++) {var} = fmaxf({var}, {expr});\n")
}
