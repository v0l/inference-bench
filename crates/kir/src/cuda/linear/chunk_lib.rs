use super::prim::{at, device_fn, unroll_block};

fn pack_pair(dst_hi: &str, dst_lo: &str, a: &str, b: &str) -> String {
    format!(
        "unsigned h0, l0, h1, l1;
split_hi_lo({a}, h0, l0);
split_hi_lo({b}, h1, l1);
{dst_hi} = h0 | (h1 << 16);
{dst_lo} = l0 | (l1 << 16);
"
    )
}

fn split_tile() -> String {
    let a_loop = unroll_block(0, "int i = 0; i < 4; i++", &pack_pair("ah[i]", "al[i]", "av[2 * i]", "av[2 * i + 1]"));
    let b_loop = unroll_block(0, "int i = 0; i < 2; i++", &pack_pair("bh[i]", "bl[i]", "bv[2 * i]", "bv[2 * i + 1]"));
    let step = format!(
        "int k = kk * 16 + 2 * t;
float av[8] = {{a_at(m0 + g, k),     a_at(m0 + g, k + 1),     a_at(m0 + g + 8, k),     a_at(m0 + g + 8, k + 1),
               a_at(m0 + g, k + 8), a_at(m0 + g, k + 9), a_at(m0 + g + 8, k + 8), a_at(m0 + g + 8, k + 9)}};
float bv[4] = {{b_at(k, n0 + g), b_at(k + 1, n0 + g), b_at(k + 8, n0 + g), b_at(k + 9, n0 + g)}};
unsigned ah[4], al[4], bh[2], bl[2];
{a_loop}{b_loop}if (LO) {{
    mma_bf16(c, al[0], al[1], al[2], al[3], bh[0], bh[1]);
    mma_bf16(c, ah[0], ah[1], ah[2], ah[3], bl[0], bl[1]);
}}
mma_bf16(c, ah[0], ah[1], ah[2], ah[3], bh[0], bh[1]);
"
    );
    let body = format!(
        "int g = lane >> 2;\nint t = lane & 3;\n#pragma unroll 2\nfor (int kk = 0; kk < KSTEPS; kk++) {{\n{}}}\n",
        at(4, &step)
    );
    device_fn(
        "int KSTEPS, bool LO = true, typename FA, typename FB",
        "void mma_split_tile(float* c, int m0, int n0, int lane, FA a_at, FB b_at)",
        &body,
    )
}

pub fn library() -> String {
    let pack = device_fn("", "void gdn_pack2(float a, float b, unsigned& hi, unsigned& lo)", &pack_pair("hi", "lo", "a", "b"));
    let put = device_fn(
        "int KSTEPS, bool LO = true",
        "void gdn_frag_put(unsigned* base, int m, int k, float v0, float v1)",
        "unsigned hi, lo;
gdn_pack2(v0, v1, hi, lo);
int rec = ((m >> 4) * KSTEPS + (k >> 4)) * 32 + (m & 7) * 4 + ((k & 7) >> 1);
int reg = ((m >> 3) & 1) + 2 * ((k >> 3) & 1);
base[rec * 8 + reg] = hi;
if (LO) base[rec * 8 + 4 + reg] = lo;
",
    );
    [split_tile(), pack, put].join("\n")
}

pub fn planes() -> String {
    let inner = "const unsigned* rh = (const unsigned*)(bh + (n * 8 + g) * pitch);
unsigned bh0 = rh[kk * 8 + t], bh1 = rh[kk * 8 + t + 4];
if (LO) {
    const unsigned* rl = (const unsigned*)(bl + (n * 8 + g) * pitch);
    unsigned bl0 = rl[kk * 8 + t], bl1 = rl[kk * 8 + t + 4];
    mma_bf16(acc[n], al.x, al.y, al.z, al.w, bh0, bh1);
    mma_bf16(acc[n], ah.x, ah.y, ah.z, ah.w, bl0, bl1);
}
mma_bf16(acc[n], ah.x, ah.y, ah.z, ah.w, bh0, bh1);
";
    let step = format!(
        "uint4 ah = ap[kk * 64];\nuint4 al = LO ? ap[kk * 64 + 1] : ah;\n{}",
        unroll_block(0, "int n = 0; n < NT; n++", inner)
    );
    let body = unroll_block(0, "int kk = 0; kk < KS; kk++", &step);
    device_fn(
        "int KS, int NT, bool LO = true",
        "void gdn_mma_planes(float (*acc)[4], const uint4* ap, const unsigned short* bh,\n                                               const unsigned short* bl, int pitch, int g, int t)",
        &body,
    )
}
