use crate::bandwidth::{Bandwidth, COPY, READ};

fn grid_stride(body: &str) -> String {
    format!("    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {body}\n")
}

pub fn emit_bandwidth(_b: &Bandwidth) -> crate::KirSource {
    let read = grid_stride(
        "{
        uint4 v;
        asm volatile(\"ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\" : \"=r\"(v.x), \"=r\"(v.y), \"=r\"(v.z), \"=r\"(v.w) : \"l\"(src + i));
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }",
    );
    let copy = grid_stride("dst[i] = src[i];");
    crate::KirSource::emitted(format!(
        "extern \"C\" __global__ void {READ}(const uint4* __restrict__ src, size_t n, unsigned* out) {{
    unsigned acc = 0;
{read}    if (acc == 0x12345678u) out[0] = acc;
}}

extern \"C\" __global__ void {COPY}(const uint4* __restrict__ src, uint4* dst, size_t n) {{
{copy}}}
"
    ))
}
