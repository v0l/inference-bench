use std::fmt::Write as _;

pub fn forceinline(ret: &str, name: &str, args: &str, body: &str) -> String {
    format!("__device__ __forceinline__ {ret} {name}({args}) {{\n{body}}}\n\n")
}

pub fn constexpr_ints(items: &[(&str, String)]) -> String {
    let mut s = String::new();
    for (n, v) in items {
        let _ = writeln!(s, "constexpr int {n} = {v};");
    }
    s
}

pub fn scalar_prelude() -> String {
    let mut s = String::from("typedef unsigned short bf16_t;\n\n");
    s += "__device__ __forceinline__ float bf(bf16_t v) { return __uint_as_float(((unsigned)v) << 16); }\n\n";
    s += &forceinline(
        "float",
        "activate",
        "float x, int kind",
        &format!(
            "    if (kind == 1) {{\n        float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);\n        return 0.5f * x * (1.0f + tanhf(u));\n    }}\n    if (kind == 2) {{\n        return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));\n    }}\n    return x;\n"
        ),
    );
    s += &forceinline("unsigned", "bf16_bits_rn", "float v", "    unsigned u = __float_as_uint(v);\n    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;\n");
    s += &forceinline("void", "split_bf16", "float v, unsigned& hi, unsigned& lo", "    hi = bf16_bits_rn(v);\n    lo = bf16_bits_rn(v - __uint_as_float(hi << 16));\n");
    s += &forceinline(
        "void",
        "split_store2",
        "bf16_t* hi, bf16_t* lo, size_t at, float a, float b",
        "    unsigned ha, la, hb, lb;\n    split_bf16(a, ha, la);\n    split_bf16(b, hb, lb);\n    *(unsigned*)(hi + at) = ha | (hb << 16);\n    *(unsigned*)(lo + at) = la | (lb << 16);\n",
    );
    s += &forceinline("unsigned", "f16_bits", "float v", "    unsigned short h;\n    asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(h) : \"f\"(v));\n    return h;\n");
    s += &forceinline("float", "f16_float", "unsigned bits", "    float f;\n    asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(f) : \"h\"((unsigned short)bits));\n    return f;\n");
    s += "__device__ __forceinline__ unsigned pack_f16_hi(float a, float b) { return f16_bits(a) | (f16_bits(b) << 16); }\n\n";
    s += &forceinline(
        "unsigned",
        "pack_f16_lo",
        "float a, float b",
        "    return f16_bits(a - f16_float(f16_bits(a))) | (f16_bits(b - f16_float(f16_bits(b))) << 16);\n",
    );
    s
}

pub fn store_act2(split: &str) -> String {
    format!("\n{}", forceinline(
        "void",
        "store_act2",
        "bf16_t* hi, bf16_t* lo, size_t at, float a, float b",
        &format!("    if ({split}) {{\n        split_store2(hi, lo, at, a, b);\n    }} else {{\n        *(unsigned*)(hi + at) = pack_f16_hi(a, b);\n    }}\n"),
    ))
}

pub fn to_f16_kernel() -> String {
    "extern \"C\" __global__ void vis_to_f16(bf16_t* w, int n) {\n    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < n) w[i] = (bf16_t)f16_bits(bf(w[i]));\n}\n\n".to_string()
}

pub fn cp_async() -> String {
    let mut s = forceinline(
        "void",
        "cp_async16_zfill",
        "void* smem, const void* gmem, bool valid",
        "    unsigned s = (unsigned)__cvta_generic_to_shared(smem);\n    int n = valid ? 16 : 0;\n    asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16, %2;\" :: \"r\"(s), \"l\"(gmem), \"r\"(n));\n",
    );
    s += "__device__ __forceinline__ void cp_async_commit() { asm volatile(\"cp.async.commit_group;\"); }\n\n";
    s += "template <int PENDING>\n";
    s += &forceinline("void", "cp_async_wait", "", "    asm volatile(\"cp.async.wait_group %0;\" :: \"n\"(PENDING));\n");
    s
}

pub fn ldsm_x4(trans: bool) -> String {
    let (name, op) = if trans { ("ldsm_x4_t", "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16") } else { ("ldsm_x4", "ldmatrix.sync.aligned.m8n8.x4.shared.b16") };
    forceinline(
        "void",
        name,
        "unsigned* r, const void* p",
        &format!("    unsigned s = (unsigned)__cvta_generic_to_shared(p);\n    asm volatile(\"{op} {{%0,%1,%2,%3}}, [%4];\"\n                 : \"=r\"(r[0]), \"=r\"(r[1]), \"=r\"(r[2]), \"=r\"(r[3]) : \"r\"(s));\n"),
    )
}

pub fn mma(name: &str, elem: &str) -> String {
    forceinline(
        "void",
        name,
        "float* c, const unsigned* a, unsigned b0, unsigned b1",
        &format!("#if __CUDA_ARCH__ >= 800\n    asm(\"mma.sync.aligned.m16n8k16.row.col.f32.{elem}.{elem}.f32 {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};\"\n        : \"+f\"(c[0]), \"+f\"(c[1]), \"+f\"(c[2]), \"+f\"(c[3])\n        : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"r\"(b0), \"r\"(b1));\n#endif\n"),
    )
}

pub fn mma_f16_hi_lo() -> String {
    "__device__ __forceinline__ void mma_f16_hi_lo(float* c, const unsigned* ah, const unsigned* al, unsigned bh0, unsigned bh1,\n                                              unsigned bl0, unsigned bl1) {\n    mma_f16(c, al, bh0, bh1);\n    mma_f16(c, ah, bl0, bl1);\n    mma_f16(c, ah, bh0, bh1);\n}\n\n".to_string()
}

pub fn fast_exp2() -> String {
    forceinline("float", "fast_exp2", "float x", "    float y;\n    asm(\"ex2.approx.ftz.f32 %0, %1;\" : \"=f\"(y) : \"f\"(x));\n    return y;\n")
}

pub fn swz4() -> String {
    "__device__ __forceinline__ int swz4(int r, int c) { return r * 4 + (c ^ ((r >> 1) & 3)); }\n\n".to_string()
}

pub fn block_sum() -> String {
    format!("\n{}", forceinline(
        "float",
        "block_sum",
        "float v, float* red",
        "    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);\n    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;\n    __syncthreads();\n    if (lane == 0) red[warp] = v;\n    __syncthreads();\n    float t = 0.0f;\n    for (int i = 0; i < blockDim.x / 32; i++) t += red[i];\n    return t;\n",
    ))
}

pub fn fill(template: &str, slots: &[(&str, String)]) -> String {
    let mut out = template.to_string();
    for (k, v) in slots {
        out = out.replace(&format!("@{k}@"), v);
    }
    out
}

pub fn zero_tiles(indent: &str, name: &str, dims: &[&str]) -> String {
    let vars: &[&str] = if dims.len() == 3 { &["i", "j", "e"] } else if name == "s" { &["nb", "e"] } else { &["i", "e"] };
    let mut s = String::new();
    for (n, (v, d)) in vars.iter().zip(dims).enumerate() {
        let pad = if n == 0 { indent.to_string() } else { format!("{indent}    {}", "    ".repeat(n - 1)) };
        let _ = write!(s, "#pragma unroll\n{pad}for (int {v} = 0; {v} < {d}; {v}++)");
        s.push(if n + 1 < dims.len() { '\n' } else { ' ' });
    }
    let idx: String = vars.iter().map(|v| format!("[{v}]")).collect();
    let _ = writeln!(s, "{name}{idx} = 0.0f;");
    s
}

pub fn pack_probs() -> String {
    let mut s = String::new();
    for (name, f) in [("ph", "pack_f16_hi"), ("pl", "pack_f16_lo")] {
        for (i, (row, c)) in [("2 * kb", 0), ("2 * kb", 2), ("2 * kb + 1", 0), ("2 * kb + 1", 2)].iter().enumerate() {
            let _ = writeln!(s, "            {name}[{i}] = {f}(s[{row}][{c}], s[{row}][{}]);", c + 1);
        }
    }
    s
}

pub fn exp_sum() -> String {
    "        float sum0 = 0.0f, sum1 = 0.0f;\n#pragma unroll\n        for (int nb = 0; nb < SB; nb++)\n#pragma unroll\n            for (int e = 0; e < 4; e++) {\n                float nm = e < 2 ? n0 : n1;\n                float p = s[nb][e] == NEG ? 0.0f : fast_exp2(s[nb][e] - nm);\n                s[nb][e] = p;\n                if (e < 2) sum0 += p; else sum1 += p;\n            }\n".to_string()
}

pub fn inv_l() -> String {
    ["0", "1"].iter().map(|i| format!("    float i{i} = l{i} > 0.0f ? 1.0f / l{i} : 0.0f;\n")).collect()
}

pub fn gemm_consts() -> String {
    let ints: Vec<(&str, String)> = vec![
        ("GM_BM", "128".into()),
        ("GM_BN", "128".into()),
        ("GM_BK", "32".into()),
        ("GM_CPR", "GM_BK / 8".into()),
        ("GM_PLANES", "GEMM_SPLIT ? 2 : 1".into()),
        ("GM_WM", "2".into()),
        ("GM_WN", "4".into()),
        ("GM_STAGES", "3".into()),
        ("GM_THREADS", "GM_WM * GM_WN * 32".into()),
        ("GM_TM", "GM_BM / GM_WM".into()),
        ("GM_TN", "GM_BN / GM_WN".into()),
        ("GM_MI", "GM_TM / 16".into()),
        ("GM_NJ", "GM_TN / 8".into()),
        ("GM_A_TILE", "GM_BM * GM_BK".into()),
        ("GM_A_BYTES", "GM_A_TILE * 2 * GM_PLANES".into()),
        ("GM_STAGE_BYTES", "GM_A_BYTES + GM_BN * GM_BK * 2".into()),
    ];
    constexpr_ints(&ints)
}

pub fn attention_consts(_precise: bool) -> String {
    let ints: Vec<(&str, String)> = vec![
        ("AT_STRIDE", "AT_HDP + 8".into()),
        ("AT_WARPS", "8".into()),
        ("AT_BQ", "16 * AT_WARPS".into()),
        ("AT_BK", "ATT_SPLIT ? 32 : 64".into()),
        ("AT_KB", "AT_HDP / 16".into()),
        ("AT_NB", "AT_HDP / 8".into()),
        ("AT_CHUNKS", "AT_HDP / 8".into()),
        ("AT_TILE", "AT_BK * AT_STRIDE".into()),
        ("AT_STAGE", "(ATT_SPLIT ? 4 : 2) * AT_TILE".into()),
    ];
    constexpr_ints(&ints)
}
