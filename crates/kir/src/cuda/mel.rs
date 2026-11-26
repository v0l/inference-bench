use crate::expr::Expr;
use crate::layout::Dialect;
use crate::mel::{LogMel, Pad, THREADS};

fn device(e: &Expr, names: &[&str]) -> Result<String, String> {
    e.subst(&|n| names.contains(&n).then(|| Expr::Raw(n.to_string())))?.lower(Dialect::Cuda, &[])
}

pub fn emit_mel(m: &LogMel) -> Result<crate::KirSource, String> {
    m.validate()?;
    let log = device(&m.log, &["m"])?;
    let floor = device(&m.floor, &["v", "top"])?;
    let out = device(&m.out, &["v"])?;
    let (n_fft, hop, frames, freqs, bins, samples) = (m.n_fft, m.hop, m.frames, m.freqs(), m.bins, m.samples);
    let half = n_fft / 2;
    let reflect = m.pad == Pad::Reflect;
    let source = if reflect {
        "    if (i < 0) i = -i;\n    if (i >= SAMPLES) i = 2 * (SAMPLES - 1) - i;\n"
    } else {
        "    if (i < 0 || i >= SAMPLES) { y[idx] = 0.0f; return; }\n"
    };
    let [fr, pw, lg, mx, sc] = [0, 1, 2, 3, 4].map(|i| m.name(crate::mel::STAGES[i]));
    let mut s = String::new();
    s += &format!("namespace {fr}_c {{\nconstexpr int N_FFT = {n_fft};\nconstexpr int HOP = {hop};\nconstexpr int FRAMES = {frames};\nconstexpr int FREQS = {freqs};\nconstexpr int BINS = {bins};\nconstexpr int SAMPLES = {samples};\nconstexpr int HALF = {half};\n}}\n\n");
    s += &format!("extern \"C\" __global__ void __launch_bounds__({THREADS}) {fr}(const float* pcm, int len, float* y) {{\n    using namespace {fr}_c;\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx >= FRAMES * N_FFT) return;\n    int f = idx / N_FFT, t = idx % N_FFT;\n    int i = f * HOP + t - HALF;\n{source}    y[idx] = i < len ? pcm[i] : 0.0f;\n}}\n\n");
    s += &format!("extern \"C\" __global__ void __launch_bounds__({THREADS}) {pw}(const float* x, float* y) {{\n    using namespace {fr}_c;\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx >= FRAMES * FREQS) return;\n    int f = idx / FREQS, k = idx % FREQS;\n    float re = x[(size_t)f * 2 * FREQS + 2 * k], im = x[(size_t)f * 2 * FREQS + 2 * k + 1];\n    y[idx] = re * re + im * im;\n}}\n\n");
    s += &format!("extern \"C\" __global__ void __launch_bounds__({THREADS}) {lg}(const float* x, float* y) {{\n    using namespace {fr}_c;\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx >= FRAMES * BINS) return;\n    float m = x[idx];\n    y[idx] = {log};\n}}\n\n");
    s += &format!("extern \"C\" __global__ void __launch_bounds__({THREADS}) {mx}(const float* x, float* top) {{\n    using namespace {fr}_c;\n    __shared__ float red[{w}];\n    float v = -3.0e38f;\n    for (int i = threadIdx.x; i < FRAMES * BINS; i += blockDim.x) v = fmaxf(v, x[i]);\n    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));\n    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = v;\n    __syncthreads();\n    if (threadIdx.x == 0) {{\n        float t = red[0];\n        for (int i = 1; i < {w}; i++) t = fmaxf(t, red[i]);\n        top[0] = t;\n    }}\n}}\n\n", w = THREADS / 32);
    s += &format!("extern \"C\" __global__ void __launch_bounds__({THREADS}) {sc}(const float* x, const float* t, float* y) {{\n    using namespace {fr}_c;\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx >= FRAMES * BINS) return;\n    float top = t[0];\n    float v = x[idx];\n    v = {floor};\n    y[idx] = {out};\n}}\n\n");
    Ok(crate::KirSource::emitted(s))
}
