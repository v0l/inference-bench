import json
import sys
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def rmsnorm(X, W, Y, eps, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    o = tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(X + row * N + o, mask=m, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + o, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * N + o, (x * inv * w).to(Y.dtype.element_ty), mask=m)


@triton.jit
def add_rmsnorm(X, R, W, Y, eps, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    o = tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(X + row * N + o, mask=m, other=0.0).to(tl.float32) + tl.load(R + row * N + o, mask=m, other=0.0).to(tl.float32)
    tl.store(R + row * N + o, x.to(R.dtype.element_ty), mask=m)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + o, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * N + o, (x * inv * w).to(Y.dtype.element_ty), mask=m)


@triton.jit
def silu_and_mul(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    g = tl.load(X + row * 2 * N + o, mask=m, other=0.0).to(tl.float32)
    u = tl.load(X + row * 2 * N + N + o, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * N + o, (g * tl.sigmoid(g) * u).to(Y.dtype.element_ty), mask=m)


@triton.jit
def rotary(Q, Cos, Sin, pos, HEADS: tl.constexpr, D: tl.constexpr):
    tok = tl.program_id(0)
    h = tl.program_id(1)
    half = tl.arange(0, D // 2)
    p = tl.load(pos + tok)
    c = tl.load(Cos + p * (D // 2) + half)
    s = tl.load(Sin + p * (D // 2) + half)
    base = Q + (tok * HEADS + h) * D
    x = tl.load(base + half).to(tl.float32)
    y = tl.load(base + D // 2 + half).to(tl.float32)
    tl.store(base + half, (x * c - y * s).to(Q.dtype.element_ty))
    tl.store(base + D // 2 + half, (y * c + x * s).to(Q.dtype.element_ty))


@triton.jit
def softmax_topk(L, P, I, E: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    o = tl.arange(0, E)
    x = tl.load(L + row * E + o).to(tl.float32)
    x = tl.exp(x - tl.max(x, axis=0))
    x = x / tl.sum(x, axis=0)
    for k in tl.static_range(K):
        v = tl.max(x, axis=0)
        i = tl.argmax(x, axis=0)
        tl.store(P + row * K + k, v)
        tl.store(I + row * K + k, i)
        x = tl.where(o == i, -1.0, x)


@triton.jit
def matmul(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BN)
    width = GROUP * grid_n
    gid = pid // width
    first = gid * GROUP
    size = min(tl.cdiv(M, BM) - first, GROUP)
    pm = first + (pid % width) % size
    pn = (pid % width) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + rm[:, None] * K + (k + rk)[None, :])
        b = tl.load(B + (k + rk)[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(C.dtype.element_ty))


MATMUL_CONFIGS = [(128, 256, 64, 3, 8), (64, 256, 32, 4, 4), (128, 128, 32, 4, 4), (128, 64, 32, 4, 4), (64, 128, 32, 4, 4), (128, 128, 64, 3, 8)]


def kernels():
    for hidden in (2560, 3072, 5120):
        block = triton.next_power_of_2(hidden)
        yield "rmsnorm", f"triton_rmsnorm_{hidden}", rmsnorm, {"X": "*bf16", "W": "*bf16", "Y": "*bf16", "eps": "fp32", "N": "constexpr", "BLOCK": "constexpr"}, {"N": hidden, "BLOCK": block}, {"num_warps": 8}, {"hidden": hidden}
        yield "fused_add_rmsnorm", f"triton_add_rmsnorm_{hidden}", add_rmsnorm, {"X": "*bf16", "R": "*bf16", "W": "*bf16", "Y": "*bf16", "eps": "fp32", "N": "constexpr", "BLOCK": "constexpr"}, {"N": hidden, "BLOCK": block}, {"num_warps": 8}, {"hidden": hidden}
    for inter in (9216, 17408, 1024):
        yield "silu_and_mul", f"triton_silu_and_mul_{inter}", silu_and_mul, {"X": "*bf16", "Y": "*bf16", "N": "constexpr", "BLOCK": "constexpr"}, {"N": inter, "BLOCK": 1024}, {"num_warps": 4}, {"inter": inter}
    for heads, d in ((24, 256), (32, 128), (16, 256)):
        yield "rotary", f"triton_rotary_h{heads}_d{d}", rotary, {"Q": "*bf16", "Cos": "*fp32", "Sin": "*fp32", "pos": "*i32", "HEADS": "constexpr", "D": "constexpr"}, {"HEADS": heads, "D": d}, {"num_warps": 4}, {"heads": heads, "head_dim": d}
    for e, k in ((128, 8), (256, 8), (32, 4)):
        yield "topk_routing", f"triton_softmax_topk_e{e}_k{k}", softmax_topk, {"L": "*fp32", "P": "*fp32", "I": "*i32", "E": "constexpr", "K": "constexpr"}, {"E": e, "K": k}, {"num_warps": 4}, {"experts": e, "top_k": k}
    for bm, bn, bk, stages, warps in MATMUL_CONFIGS:
        yield "bf16_gemm", f"triton_matmul_{bm}x{bn}x{bk}_s{stages}_w{warps}", matmul, {"A": "*bf16", "B": "*bf16", "C": "*bf16", "M": "i32", "N": "i32", "K": "i32", "BM": "constexpr", "BN": "constexpr", "BK": "constexpr", "GROUP": "constexpr"}, {"BM": bm, "BN": bn, "BK": bk, "GROUP": 8}, {"num_warps": warps, "num_stages": stages}, {"tile": [bm, bn, bk], "stages": stages, "warps": warps}


def hints(fn, sig):
    return {(fn.arg_names.index(k),): [["tt.divisibility", 16]] for k, t in sig.items() if t.startswith("*") or t == "i32"}


def main():
    out = Path(sys.argv[1])
    archs = sys.argv[2].split(",")
    index = []
    for arch in archs:
        cc = int(arch[3:].rstrip("a"))
        d = out / arch
        d.mkdir(parents=True, exist_ok=True)
        for family, name, fn, sig, consts, opts, params in kernels():
            entry = {"family": family, "name": name, "arch": arch, "params": params}
            try:
                c = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts, attrs=hints(fn, sig)), target=GPUTarget("cuda", cc, 32), options=opts)
                (d / f"{name}.cubin").write_bytes(c.asm["cubin"])
                (d / f"{name}.ptx").write_text(c.asm["ptx"])
                entry.update({"cubin": f"{name}.cubin", "kernel": c.metadata.name, "threads": c.metadata.num_warps * 32, "shared": c.metadata.shared})
            except Exception as e:
                entry["error"] = str(e).splitlines()[0][:300] if str(e) else type(e).__name__
            index.append(entry)
    (out / "triton.json").write_text(json.dumps({"triton": triton.__version__, "kernels": index}, indent=1, sort_keys=True))
    for e in index:
        print(e["arch"], e["name"], e.get("error", "ok"))


main()
