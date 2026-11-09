import csv
import re
import sys


def short(name):
    name = re.sub(r"\s+", " ", name)
    m = re.search(r"(triton_[a-z_0-9]{0,40})", name)
    if m:
        return m.group(1)
    for key in ("gemvx", "BatchPrefill", "BatchDecode", "chunk_gated", "fused_sigmoid", "causal_conv1d", "cvt_fp16_to_fp4", "reshape_and_cache", "elementwise", "Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_64x1", "GemmUniversal"):
        if key in name:
            return name[:110]
    return name[:110]


def main():
    path = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    rows = list(csv.DictReader(open(path)))
    total = sum(float(r["Total Time (ns)"]) for r in rows)
    print(f"{path} kernel_sum_ms={total / 1e6:.2f} kernels={sum(int(r['Instances']) for r in rows)}")
    for r in rows[:top]:
        print(f"  {float(r['Time (%)']):5.1f}% n={int(r['Instances']):6d} med={float(r['Med (ns)']) / 1e3:9.1f}us tot={float(r['Total Time (ns)']) / 1e6:8.2f}ms  {short(r['Name'])}")


main()
