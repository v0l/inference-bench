import argparse
import ctypes
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0f")

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
CARD = {"name": "RTX PRO 6000 Blackwell Max-Q", "bw_gbs": 1792, "bf16_tflops": 439, "fp4_tflops": 1755}
CUTLASS_SO = Path.home() / ".cache/flashinfer/0.7.3/120f/cached_ops/fp4_gemm_cutlass_sm120/fp4_gemm_cutlass_sm120.so"
SPIN = 2_000_000
CHAIN_US = 200.0
FLUSH_BYTES = 512 << 20
CONTROL_TOL = 0.02
SEED = 1158


def consts(path):
    out = {}
    for name, value in re.findall(r"^constexpr (?:int|float|bool|size_t) (\w+) = ([^;]+);", Path(path).read_text(), re.M):
        if name in out:
            continue
        v = value.strip()
        if v in ("true", "false"):
            out[name] = v == "true"
            continue
        try:
            out[name] = int(v)
        except ValueError:
            try:
                out[name] = float(v.rstrip("f"))
            except ValueError:
                if re.fullmatch(r"[\w\s*+/()-]+", v):
                    try:
                        out[name] = int(eval(v.replace("/", "//"), {"__builtins__": {}}, dict(out)))
                    except Exception:
                        pass
    return out


def gated_body(path):
    m = re.search(r"#define GATED_BODY\(g, u\) (.*)", Path(path).read_text())
    return m.group(1) if m else ""


class Cubins:
    def __init__(self, root, manifest):
        from cuda.bindings import driver as cu
        self.cu = cu
        self.root = root
        self.manifest = manifest
        self.mods = {}
        self.keep = []

    def check(self, r):
        if r[0] != self.cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(str(r[0]))
        return r[1] if len(r) == 2 else None

    def function(self, module, name, smem):
        if module not in self.mods:
            data = (self.root / self.manifest["modules"][module]["cubin"]).read_bytes()
            self.keep.append(data)
            self.mods[module] = self.check(self.cu.cuModuleLoadData(data))
        f = self.check(self.cu.cuModuleGetFunction(self.mods[module], name.encode()))
        if smem > 48 * 1024:
            self.check(self.cu.cuFuncSetAttribute(f, self.cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))
        return f

    def launcher(self, module, launch, args):
        f = self.function(module, launch["kernel"], launch["smem"])
        vals = tuple(a.data_ptr() if isinstance(a, torch.Tensor) else a for a in args)
        types = tuple(ctypes.c_void_p if isinstance(a, torch.Tensor) else ctypes.c_int for a in args)
        gx, gy, gz = launch["grid"]
        block = launch["block"]
        smem = launch["smem"]
        cu = self.cu

        def run():
            stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
            r = cu.cuLaunchKernel(f, gx, gy, gz, block, 1, 1, smem, stream, (vals, types), 0)
            if r[0] != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"{launch['kernel']}: {r[0]}")

        return run


def chain(*fns):
    def run():
        for f in fns:
            f()
    return run


def gen():
    g = torch.Generator()
    g.manual_seed(SEED)
    return g


def randn(shape, g, scale=1.0):
    return torch.randn(shape, generator=g) * scale


def planes(hi, lo):
    return hi.view(torch.bfloat16).float() + lo.view(torch.bfloat16).float()


def err(got, want):
    got = got.double().cpu()
    want = want.double().cpu()
    d = (got - want).abs().max().item()
    scale = want.abs().max().item()
    return {"max_abs": d, "rel": d / scale if scale else d}


def flashinfer():
    import flashinfer as fi
    return fi


def patch_cutlass_fp4():
    import tvm_ffi
    import flashinfer.gemm.gemm_base as gb
    if not CUTLASS_SO.exists():
        return f"missing {CUTLASS_SO}"
    wrapped = gb._create_cutlass_fp4_gemm_module(tvm_ffi.load_module(str(CUTLASS_SO)), "flashinfer::cutlass_fp4_gemm_sm120", "cutlass_fp4_gemm_sm120")
    gb.get_gemm_sm120_module_cutlass_fp4 = lambda: wrapped
    return None


def vllm_ops():
    import vllm._C_stable_libtorch  # noqa: F401
    return torch.ops._C


def gather_kv(cache, table, ctx, page):
    pages = table[: ctx // page].long()
    blk = cache[pages]
    return blk.permute(1, 0, 2, 3).reshape(cache.shape[1], ctx, cache.shape[3]).float()


def attention_ref(q, k, v, scale, causal_rows=None):
    s = torch.einsum("hgd,htd->hgt", q, k) * scale
    if causal_rows is not None:
        t = torch.arange(k.shape[1])
        s = s.masked_fill(t[None, None, :] > causal_rows[None, :, None], float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("hgt,htd->hgd", p, v)


class Case:
    def __init__(self, c, cub, root):
        self.c = c
        self.p = c["params"]
        self.cub = cub
        self.module = c["module"]
        src = root / cub.manifest["modules"][self.module]["source"]
        self.k = consts(src)
        self.body = gated_body(src)
        self.sides = []
        self.errors = {}
        self.work = {}
        self.notes = []

    def ours(self, *arg_lists):
        fns = [self.cub.launcher(self.module, l, a) for l, a in zip(self.c["launches"], arg_lists)]
        return chain(*fns)


def act(g, u, body):
    if "1.702" in body:
        g = g.clamp(max=7.0)
        return (u.clamp(-7.0, 7.0) + 1.0) * (g * torch.sigmoid(g * 1.702))
    return torch.nn.functional.silu(g) * u


def setup_batch_decode(case):
    k, p = case.k, case.p
    seqs, ctx, page = p["seqs"], p["ctx"], k["PAGE"]
    heads, kvh, hd = k["HEADS"], k["KV_HEADS"], k["HEAD_DIM"]
    g = gen()
    pool = seqs * ctx // page
    perm = torch.randperm(pool, generator=g).int()
    kc = randn((pool, kvh, page, hd), g).half()
    vc = randn((pool, kvh, page, hd), g).half()
    qkv = torch.zeros(seqs, k["QKV_ROW"])
    qkv[:, : k["Q_DIM"]] = randn((seqs, k["Q_DIM"]), g)
    qkv[:, k["Q_DIM"] + 2 * k["KV_DIM"]:] = 30.0
    table = torch.zeros(seqs, p["pages"], dtype=torch.int32)
    for s in range(seqs):
        table[s, : ctx // page] = perm[s * ctx // page: (s + 1) * ctx // page]
    scale = k["QUERY_SCALE"]
    want = torch.stack([attention_ref(qkv[s, : k["Q_DIM"]].view(kvh, heads // kvh, hd).double(), gather_kv(kc, table[s], ctx, page).double(), gather_kv(vc, table[s], ctx, page).double(), scale) for s in range(seqs)]).reshape(seqs, heads, hd)
    d = {n: t.cuda() for n, t in dict(qkv=qkv, k=kc, v=vc, table=table).items()}
    d["pos"] = torch.full((seqs,), ctx - 1, dtype=torch.int32, device="cuda")
    d["slots"] = torch.arange(seqs, dtype=torch.int32, device="cuda")
    splits = p["splits"]
    d["acc"] = torch.zeros(seqs * heads * splits * hd, device="cuda")
    d["ml"] = torch.zeros(seqs * heads * splits * 2, device="cuda")
    d["hi"] = torch.zeros(seqs * k["Q_DIM"], dtype=torch.int16, device="cuda")
    d["lo"] = torch.zeros_like(d["hi"])
    ours = case.ours(
        [d["qkv"], d["k"], d["v"], d["table"], d["pos"], d["slots"], 0, 1, p["pages"], splits, d["acc"], d["ml"], 0],
        [d["acc"], d["ml"], d["qkv"], d["pos"], 1, splits, d["hi"], d["lo"]],
    )
    ours()
    torch.cuda.synchronize()
    case.errors["ours"] = err(planes(d["hi"], d["lo"]).view(seqs, heads, hd), want)
    fi = flashinfer()
    q = d["qkv"][:, : k["Q_DIM"]].reshape(seqs, heads, hd).half().contiguous()
    indptr = torch.arange(0, seqs + 1, dtype=torch.int32, device="cuda") * (ctx // page)
    indices = d["table"][:, : ctx // page].contiguous().flatten()
    last = torch.full((seqs,), page, dtype=torch.int32, device="cuda")
    ws = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
    sides = [("ours", ours)]
    for label, tc in (("FlashInfer decode (default plan)", False), ("FlashInfer decode (use_tensor_cores)", True)):
        try:
            w = fi.BatchDecodeWithPagedKVCacheWrapper(ws, "HND", use_tensor_cores=tc)
            w.plan(indptr, indices, last, heads, kvh, hd, page, q_data_type=torch.float16, kv_data_type=torch.float16, sm_scale=scale)
            o = torch.empty_like(q)
            run = (lambda w=w, o=o: w.run(q, (d["k"], d["v"]), out=o))
            run()
            torch.cuda.synchronize()
            label = f"{label} [{w.resolved_backend}]"
            case.errors[label] = err(o.float(), want)
            sides.append((label, run))
        except Exception as e:
            case.notes.append(f"{label}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    kv = seqs * ctx * kvh * hd * 2 * 2
    for label, _ in sides:
        case.work[label] = {"bytes": kv}
    case.roof = ("bw", kv)
    return sides


def setup_split_decode(case):
    k, p = case.k, case.p
    ctx, page = p["ctx"], k["PAGE"]
    heads, kvh, hd = k["HEADS"], k["KV_HEADS"], k["HEAD_DIM"]
    g = gen()
    pool = ctx // page
    table = torch.randperm(pool, generator=g).int()
    kc = randn((pool, kvh, page, hd), g).half()
    vc = randn((pool, kvh, page, hd), g).half()
    qkv = torch.zeros(k["QKV_ROW"])
    qkv[: k["Q_DIM"]] = randn((k["Q_DIM"],), g)
    qkv[k["Q_DIM"] + 2 * k["KV_DIM"]:] = 30.0
    scale = k["QUERY_SCALE"]
    want = attention_ref(qkv[: k["Q_DIM"]].view(kvh, heads // kvh, hd).double(), gather_kv(kc, table, ctx, page).double(), gather_kv(vc, table, ctx, page).double(), scale).reshape(heads, hd)
    d = {n: t.cuda() for n, t in dict(qkv=qkv, k=kc, v=vc, table=table).items()}
    d["pos"] = torch.tensor([ctx - 1], dtype=torch.int32, device="cuda")
    d["part"] = torch.zeros(heads * p["splits"] * (hd + 2), device="cuda")
    d["out"] = torch.zeros(k["Q_DIM"], device="cuda")
    ours = case.ours([d["qkv"], d["k"], d["v"], d["table"], d["pos"], 0, d["part"], 0], [d["part"], d["pos"], d["qkv"], d["out"], 0])
    ours()
    torch.cuda.synchronize()
    case.errors["ours"] = err(d["out"].view(heads, hd), want)
    fi = flashinfer()
    q = d["qkv"][: k["Q_DIM"]].view(1, heads, hd).half().contiguous()
    indptr = torch.tensor([0, pool], dtype=torch.int32, device="cuda")
    last = torch.tensor([page], dtype=torch.int32, device="cuda")
    ws = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
    sides = [("ours", ours)]
    for label, tc in (("FlashInfer decode (default plan)", False), ("FlashInfer decode (use_tensor_cores)", True)):
        try:
            w = fi.BatchDecodeWithPagedKVCacheWrapper(ws, "HND", use_tensor_cores=tc)
            w.plan(indptr, d["table"], last, heads, kvh, hd, page, q_data_type=torch.float16, kv_data_type=torch.float16, sm_scale=scale)
            o = torch.empty_like(q)
            run = (lambda w=w, o=o: w.run(q, (d["k"], d["v"]), out=o))
            run()
            torch.cuda.synchronize()
            label = f"{label} [{w.resolved_backend}]"
            case.errors[label] = err(o.float().view(heads, hd), want)
            sides.append((label, run))
        except Exception as e:
            case.notes.append(f"{label}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    kv = ctx * kvh * hd * 2 * 2
    for label, _ in sides:
        case.work[label] = {"bytes": kv}
    case.roof = ("bw", kv)
    return sides


def rope_norm(q, k, eps, tokens):
    hd = k["HEAD_DIM"]
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps)
    half = k["ROT"] // 2
    i = torch.arange(half, dtype=torch.float32)
    freq = 1.0 / torch.pow(torch.tensor(k["THETA"]), 2 * i / k["ROT"])
    ang = torch.arange(tokens, dtype=torch.float32)[:, None] * freq[None, :]
    c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    x, y = q[..., :half].clone(), q[..., half: 2 * half].clone()
    q = q.clone()
    q[..., :half] = x * c - y * s
    q[..., half: 2 * half] = y * c + x * s
    assert hd >= 2 * half
    return q


def setup_prefill(case):
    k, p = case.k, case.p
    assert not k.get("ROPE_PAIRS") and not k.get("MROPE") and k.get("ROPE_KIND") == 0 and k.get("QK_NORM")
    tokens, page = p["tokens"], k["PAGE"]
    heads, kvh, hd = k["HEADS"], k["KV_HEADS"], k["HEAD_DIM"]
    g = gen()
    pool = tokens // page
    table = torch.randperm(pool, generator=g).int()
    kc = randn((pool, kvh, page, hd), g).half()
    vc = randn((pool, kvh, page, hd), g).half()
    qkv = torch.zeros(tokens, k["QKV_ROW"])
    qkv[:, : k["Q_DIM"]] = randn((tokens, k["Q_DIM"]), g)
    qkv[:, k["Q_DIM"] + 2 * k["KV_DIM"]:] = 30.0
    qr = rope_norm(qkv[:, : k["Q_DIM"]].view(tokens, heads, hd), k, k["EPS"], tokens)
    rows = torch.unique(torch.cat([torch.linspace(0, tokens - 1, 63).long(), torch.tensor([tokens - 1])]))
    scale = k["QUERY_SCALE"]
    kf, vf = gather_kv(kc, table, tokens, page).double(), gather_kv(vc, table, tokens, page).double()
    qs = qr[rows].double().permute(1, 0, 2).reshape(kvh, heads // kvh, len(rows), hd)
    want = torch.stack([attention_ref(qs[:, :, i], kf, vf, scale, causal_rows=torch.full((heads // kvh,), r.item())) for i, r in enumerate(rows)]).reshape(len(rows), heads, hd)
    d = {n: t.cuda() for n, t in dict(qkv=qkv, k=kc, v=vc, table=table).items()}
    d["qn"] = torch.ones(hd, device="cuda")
    d["pos"] = torch.zeros(1, dtype=torch.int32, device="cuda")
    d["hi"] = torch.zeros(tokens * k["Q_DIM"], dtype=torch.int16, device="cuda")
    d["lo"] = torch.zeros_like(d["hi"])
    d["meta"] = torch.zeros(4, dtype=torch.int32, device="cuda")
    d["mtab"] = torch.zeros(4, dtype=torch.int32, device="cuda")
    ours = case.ours([d["qkv"], d["qn"], d["k"], d["v"], d["table"], d["pos"], 0, d["hi"], d["lo"], tokens, d["meta"], d["mtab"], 1, 0, 0, d["meta"], 2**31 - 1])
    ours()
    torch.cuda.synchronize()
    case.errors["ours"] = err(planes(d["hi"], d["lo"]).view(tokens, heads, hd)[rows.cuda()], want)
    fi = flashinfer()
    q = qr.half().cuda().contiguous()
    sides = [("ours (fused q norm and rope)", ours)]
    case.errors["ours (fused q norm and rope)"] = case.errors.pop("ours")
    try:
        ws = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
        w = fi.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="fa2")
        w.plan(torch.tensor([0, tokens], dtype=torch.int32, device="cuda"), torch.tensor([0, pool], dtype=torch.int32, device="cuda"), d["table"], torch.tensor([page], dtype=torch.int32, device="cuda"), heads, kvh, hd, page, causal=True, sm_scale=scale, q_data_type=torch.float16, kv_data_type=torch.float16)
        o = torch.empty_like(q)
        run = lambda: w.run(q, (d["k"], d["v"]), out=o)
        run()
        torch.cuda.synchronize()
        case.errors["FlashInfer FA2 prefill"] = err(o.float()[rows.cuda()], want)
        sides.append(("FlashInfer FA2 prefill", run))
    except Exception as e:
        case.notes.append(f"FlashInfer FA2: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    flops = 4 * heads * hd * tokens * (tokens + 1) / 2
    for label, _ in sides:
        case.work[label] = {"flops": flops}
    case.roof = ("bf16", flops)
    return sides


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def e4m3(b):
    return b.view(torch.float8_e4m3fn).float()


def dequant_fp4(q, s):
    lo, hi = E2M1[(q & 15).long()], E2M1[(q >> 4).long()]
    v = torch.stack([lo, hi], -1).reshape(q.shape[0], q.shape[1] * 2)
    return v * e4m3(s).repeat_interleave(16, dim=1)


def fp4_data(rows, cols, g):
    q = torch.randint(0, 256, (rows, cols // 2), generator=g, dtype=torch.uint8)
    s = torch.randint(0x28, 0x48, (rows, cols // 16), generator=g, dtype=torch.uint8)
    return q, s


def header():
    h = torch.zeros(16, dtype=torch.uint8)
    h[:8] = torch.tensor([1.0, 1.0], dtype=torch.float32).view(torch.uint8)
    return h


def setup_fp4_gemm(case):
    p = case.p
    m, n, kk = p["m"], p["n"], p["k"]
    g = gen()
    wq, wsc = fp4_data(n, kk, g)
    xq, xsc = fp4_data(m, kk, g)
    rows = torch.unique(torch.cat([torch.linspace(0, m - 1, 63).long(), torch.tensor([m - 1])]))
    ref = dequant_fp4(xq[rows], xsc[rows]) @ dequant_fp4(wq, wsc).T
    gate_up = p["linear"] == "gate_up"
    d = dict(w=wq.cuda(), ws=torch.cat([header(), wsc.flatten()]).cuda(), xq=xq.cuda(), xs=xsc.cuda())
    d["out"] = torch.zeros(m * n, device="cuda")
    d["oh"] = torch.zeros(m * n // 2 if gate_up else 2, dtype=torch.int16, device="cuda")
    d["ol"] = torch.zeros_like(d["oh"])
    ours = case.ours([d["w"], d["ws"], d["xq"], d["xs"], d["out"], d["oh"], d["ol"], m])
    ours()
    torch.cuda.synchronize()
    if gate_up:
        case.errors["ours (fused act)"] = err(planes(d["oh"], d["ol"]).view(m, n // 2)[rows.cuda()], act(ref[:, 0::2], ref[:, 1::2], case.body))
    else:
        case.errors["ours (fused act)" if gate_up else "ours"] = err(d["out"].view(m, n)[rows.cuda()], ref)
    label_ours = "ours (fused act)" if gate_up else "ours"
    sides = [(label_ours, ours)]
    fi = flashinfer()
    note = patch_cutlass_fp4()
    if note:
        case.notes.append(note)
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
    a_sf = swizzle_blockscale(d["xs"].view(torch.float8_e4m3fn))
    b_sf = swizzle_blockscale(d["ws"][16:].view(n, kk // 16).view(torch.float8_e4m3fn))
    alpha = torch.ones(1, device="cuda")
    for label, backend in (("FlashInfer mm_fp4 cutlass (autotuned)", "cutlass"), ("FlashInfer mm_fp4 b12x", "b12x")):
        try:
            o = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            run = (lambda o=o, backend=backend: fi.mm_fp4(d["xq"], d["w"].T, a_sf, b_sf.T, alpha, torch.bfloat16, o, backend=backend))
            with fi.autotune(True):
                run()
            run()
            torch.cuda.synchronize()
            case.errors[label] = err(o.float()[rows.cuda()], ref)
            sides.append((label, run))
        except Exception as e:
            case.notes.append(f"{label}: {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
    flops = 2 * m * n * kk
    for label, _ in sides:
        case.work[label] = {"flops": flops}
    case.roof = ("fp4", flops)
    return sides


def triton_matmul(a, bt, c, m, n, kk):
    sys.path.insert(0, str(HERE))
    import kernel_static_triton as kst
    import triton
    best = None
    for bm, bn, bk, stages, warps in kst.MATMUL_CONFIGS:
        if m % bm or n % bn or kk % bk:
            continue
        grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
        run = (lambda bm=bm, bn=bn, bk=bk, stages=stages, warps=warps, grid=grid: kst.matmul[grid](a, bt, c, m, n, kk, BM=bm, BN=bn, BK=bk, GROUP=8, num_warps=warps, num_stages=stages))
        run()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(10):
            run()
        e1.record()
        torch.cuda.synchronize()
        t = e0.elapsed_time(e1)
        if best is None or t < best[0]:
            best = (t, run, f"{bm}x{bn}x{bk} s{stages} w{warps}")
    return best


def setup_bf16_gemm(case):
    p = case.p
    m, n, kk = p["m"], p["n"], p["k"]
    g = gen()
    w = randn((n, kk), g, 0.05).bfloat16()
    x = randn((m, kk), g).bfloat16()
    rows = torch.unique(torch.cat([torch.linspace(0, m - 1, 63).long(), torch.tensor([m - 1])]))
    ref = x[rows].float() @ w.float().T
    d = dict(w=w.cuda(), x=x.cuda())
    d["xl"] = torch.zeros_like(d["x"])
    d["dummy"] = torch.zeros(16, device="cuda")
    d["oh"] = torch.zeros(m * n // 2, dtype=torch.int16, device="cuda")
    d["ol"] = torch.zeros_like(d["oh"])
    ours = case.ours([d["w"], d["dummy"], d["dummy"], d["x"], d["xl"], d["dummy"], d["oh"], d["ol"], m])
    ours()
    torch.cuda.synchronize()
    case.errors["ours (fused act)"] = err(planes(d["oh"], d["ol"]).view(m, n // 2)[rows.cuda()], act(ref[:, 0::2], ref[:, 1::2], case.body))
    sides = [("ours (fused act)", ours)]
    o = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    wt = d["w"].t()
    cublas = lambda: torch.matmul(d["x"], wt, out=o)
    cublas()
    torch.cuda.synchronize()
    case.errors["cuBLAS (torch.matmul)"] = err(o.float()[rows.cuda()], ref)
    sides.append(("cuBLAS (torch.matmul)", cublas))
    try:
        bt = wt.contiguous()
        o2 = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        best = triton_matmul(d["x"], bt, o2, m, n, kk)
        o2.zero_()
        best[1]()
        torch.cuda.synchronize()
        label = f"Triton grouped matmul ({best[2]})"
        case.errors[label] = err(o2.float()[rows.cuda()], ref)
        sides.append((label, best[1]))
    except Exception as e:
        case.notes.append(f"Triton: {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
    flops = 2 * m * n * kk
    for label, _ in sides:
        case.work[label] = {"flops": flops}
    case.roof = ("bf16", flops)
    return sides


def setup_norm(case, add):
    p = case.p
    rows, hid = p["rows"], p["hidden"]
    eps = 1e-6
    g = gen()
    x = randn((rows, hid), g)
    delta = randn((rows, hid), g)
    w = 1.0 + randn((hid,), g, 0.1)
    s = x.double() + delta.double() if add else x.double()
    want = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps) * w.double()
    d = dict(x=x.cuda(), delta=delta.cuda(), w=w.cuda())
    d["sum"] = torch.zeros_like(d["x"])
    d["out"] = torch.zeros_like(d["x"])
    if add:
        ours = case.ours([d["x"], d["delta"], d["w"], d["sum"], d["out"]])
    else:
        ours = case.ours([d["x"], d["w"], d["out"]])
    ours()
    torch.cuda.synchronize()
    case.errors["ours (f32)"] = err(d["out"], want)
    sides = [("ours (f32)", ours)]
    xb, db, wb = d["x"].bfloat16(), d["delta"].bfloat16(), d["w"].bfloat16()
    fi = flashinfer()
    ops = vllm_ops()
    if add:
        def fi_run(a=xb.clone(), r=db.clone()):
            fi.fused_add_rmsnorm(a, r, wb, eps)
        def vl_run(a=xb.clone(), r=db.clone()):
            ops.fused_add_rms_norm(a, r, wb, eps)
        for label, fn in (("FlashInfer fused_add_rmsnorm (bf16)", fi.fused_add_rmsnorm), ("vLLM fused_add_rms_norm (bf16)", ops.fused_add_rms_norm)):
            a, r = db.clone(), xb.clone()
            try:
                fn(a, r, wb, eps)
                torch.cuda.synchronize()
                case.errors[label] = err(a.float(), want)
            except Exception as e:
                case.notes.append(f"{label}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        cands = (("FlashInfer fused_add_rmsnorm (bf16)", fi_run), ("vLLM fused_add_rms_norm (bf16)", vl_run))
    else:
        ofi, ovl = torch.empty_like(xb), torch.empty_like(xb)
        fi_run = lambda: fi.rmsnorm(xb, wb, eps, out=ofi)
        vl_run = lambda: ops.rms_norm(ovl, xb, wb, eps)
        cands = (("FlashInfer rmsnorm (bf16)", fi_run), ("vLLM rms_norm (bf16)", vl_run))
        for label, fn in cands:
            try:
                fn()
                torch.cuda.synchronize()
                case.errors[label] = err((ofi if "FlashInfer" in label else ovl).float(), want)
            except Exception as e:
                case.notes.append(f"{label}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    for label, fn in cands:
        if label in case.errors:
            sides.append((label, fn))
    elems = rows * hid
    ours_bytes = elems * (16 if add else 8) + hid * 4
    theirs_bytes = elems * (8 if add else 4) + hid * 2
    for label, _ in sides:
        case.work[label] = {"bytes": ours_bytes if label.startswith("ours") else theirs_bytes}
    case.roof = ("bw", ours_bytes)
    return sides


def setup_fp4_gemv(case):
    p, k = case.p, case.k
    n, kk = p["n"], p["k"]
    g = gen()
    wq, wsc = fp4_data(n, kk, g)
    x = randn((kk,), g)
    wd = dequant_fp4(wq, wsc).double()
    d = dict(w=wq.cuda(), ws=torch.cat([header(), wsc.flatten()]).cuda())
    gate_up = p["linear"] == "gate_up"
    if gate_up:
        xn = x.double() * torch.rsqrt(x.double().pow(2).mean() + k["EPS"])
        y = wd @ xn
        want = act(y[0::2], y[1::2], case.body)
        d["x"], d["nw"] = x.cuda(), torch.ones(kk, device="cuda")
        d["out"] = torch.zeros(n // 2, device="cuda")
        ours = case.ours([d["w"], d["ws"], d["x"], d["nw"], d["out"]])
    else:
        want = wd @ x.double()
        kg = kk // 16
        idx = torch.arange(kk)
        xt = torch.zeros(kk)
        xt[(idx & 15) * kg + (idx >> 4)] = x
        d["xt"], d["b"] = xt.cuda(), torch.zeros(16, device="cuda")
        d["out"] = torch.zeros(n, device="cuda")
        ours = case.ours([d["w"], d["ws"], d["b"], d["xt"], d["out"]])
    ours()
    torch.cuda.synchronize()
    got = d["out"].cpu()
    if gate_up:
        rows = n // 2
        kg = rows // 16
        r = torch.arange(rows)
        got = got[(r & 15) * kg + (r >> 4)]
    case.errors["ours"] = err(got, want)
    bytes_ = n * kk // 2 + n * kk // 16 + kk * 4
    case.work["ours"] = {"bytes": bytes_}
    case.roof = ("bw", bytes_)
    return [("ours", ours)]


SETUP = {
    "batch_decode": setup_batch_decode,
    "split_decode": setup_split_decode,
    "prefill": setup_prefill,
    "fp4_gemm": setup_fp4_gemm,
    "bf16_gemm": setup_bf16_gemm,
    "rmsnorm": lambda c: setup_norm(c, False),
    "add_rmsnorm": lambda c: setup_norm(c, True),
    "fp4_gemv": setup_fp4_gemv,
}


def stats(ts):
    t = torch.tensor(ts, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float64)).tolist()
    inner = t[(t >= q[0]) & (t <= q[4])]
    return {"median_us": inner.mean().item() * 1e3, "p50_us": q[2] * 1e3, "iqr_us": (q[3] - q[1]) * 1e3, "n": len(ts)}


def chain_len(fn):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(SPIN)
    e0.record()
    for _ in range(8):
        fn()
    e1.record()
    torch.cuda.synchronize()
    each = e0.elapsed_time(e1) * 1e3 / 8
    return max(1, min(64, int(CHAIN_US / max(each, 0.5)) + 1))


def measure(sides, iters, warmup):
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    for _ in range(warmup):
        for _, fn in sides:
            fn()
    torch.cuda.synchronize()
    reps = {label: chain_len(fn) for label, fn in sides}
    enqueue = {label: [] for label, _ in sides}
    out = {label: {} for label, _ in sides}
    rng = torch.Generator()
    rng.manual_seed(SEED)
    for mode in ("warm", "cold"):
        times = {label: [] for label, _ in sides}
        for i in range(iters):
            order = [sides[j] for j in torch.randperm(len(sides), generator=rng).tolist()]
            pending = []
            for label, fn in order:
                n = reps[label] if mode == "warm" else 1
                if mode == "cold":
                    flush.zero_()
                else:
                    fn()
                torch.cuda._sleep(SPIN)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t0 = time.perf_counter()
                e0.record()
                for _ in range(n):
                    fn()
                e1.record()
                enqueue[label].append((time.perf_counter() - t0) / n)
                pending.append((label, e0, e1, n))
            torch.cuda.synchronize()
            for label, e0, e1, n in pending:
                times[label].append(e0.elapsed_time(e1) / n)
        for label in times:
            out[label][mode] = stats(times[label])
    for label in out:
        out[label]["warm"]["chain"] = reps[label]
        out[label]["enqueue_us"] = sorted(enqueue[label])[len(enqueue[label]) // 2] * 1e6
    return out


def gpu_state():
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    q = subprocess.run(["nvidia-smi", "-i", dev, "--query-gpu=index,name,clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu,pstate", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    clock = subprocess.run(["nvidia-smi", "-i", dev, "-q", "-d", "CLOCK"], capture_output=True, text=True).stdout
    return {"query": q, "clock": clock}


def verdict(r):
    if r > 1.05:
        return "ahead"
    if r >= 0.95:
        return "on par"
    if r >= 0.8:
        return "close"
    return "behind"


def run(args):
    root = Path(args.dir)
    manifest = json.loads((root / "manifest.json").read_text())
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    cub = Cubins(root, manifest)
    results = {"date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "commit": args.commit, "command": args.command, "card": CARD, "iters": args.iters, "warmup": args.warmup, "spin_cycles": SPIN, "flush_bytes": FLUSH_BYTES, "before": gpu_state(), "versions": {}, "cases": []}
    import flashinfer as fi
    import triton
    import vllm
    results["versions"] = {"torch": torch.__version__, "flashinfer": fi.__version__, "triton": triton.__version__, "vllm": vllm.__version__, "cuda": torch.version.cuda, "nvrtc": manifest.get("nvrtc")}
    refused = []
    for c in manifest["cases"]:
        if args.only and not re.search(args.only, c["name"]):
            continue
        case = Case(c, cub, root)
        t0 = time.time()
        try:
            sides = SETUP[c["params"]["kind"]](case)
        except Exception as e:
            print(f"{c['name']}: setup failed: {type(e).__name__}: {e}", file=sys.stderr)
            results["cases"].append({"name": c["name"], "row": c["row"], "params": c["params"], "error": f"{type(e).__name__}: {e}"})
            continue
        ours_label, ours_fn = sides[0]
        timed = measure(sides + [("control", ours_fn)], args.iters, args.warmup)
        control = {m: timed["control"][m]["median_us"] / timed[ours_label][m]["median_us"] for m in ("warm", "cold")}
        ok = all(abs(v - 1.0) <= CONTROL_TOL for v in control.values())
        if not ok:
            refused.append(c["name"])
        kind, amount = case.roof
        peak = {"bw": CARD["bw_gbs"] * 1e9, "bf16": CARD["bf16_tflops"] * 1e12, "fp4": CARD["fp4_tflops"] * 1e12}[kind]
        floor_us = amount / peak * 1e6
        entry = {"name": c["name"], "row": c["row"], "params": c["params"], "launches": c["launches"], "ours": ours_label, "sides": {}, "control": control, "control_ok": ok, "notes": case.notes, "roof": {"kind": kind, "floor_us": floor_us}}
        for label, _ in sides:
            t = timed[label]
            w = case.work.get(label, {})
            s = {"warm": t["warm"], "cold": t["cold"], "enqueue_us": t["enqueue_us"], "error": case.errors.get(label), "work": w}
            for m in ("warm", "cold"):
                sec = t[m]["median_us"] * 1e-6
                if "bytes" in w:
                    s[m]["gbs"] = w["bytes"] / sec / 1e9
                if "flops" in w:
                    s[m]["tflops"] = w["flops"] / sec / 1e12
                if label != ours_label:
                    s[m]["speed_vs_ours"] = timed[ours_label][m]["median_us"] / t[m]["median_us"]
                    s[m]["ours_vs_theirs"] = t[m]["median_us"] / timed[ours_label][m]["median_us"]
                    s[m]["verdict"] = verdict(s[m]["ours_vs_theirs"])
                else:
                    s[m]["of_roofline"] = floor_us / t[m]["median_us"]
            entry["sides"][label] = s
        results["cases"].append(entry)
        line = " | ".join(f"{label.split(' (')[0]} {timed[label]['warm']['median_us']:.1f}/{timed[label]['cold']['median_us']:.1f} us" for label, _ in sides)
        print(f"{c['name']}: {line} | control {control['warm']:.3f}/{control['cold']:.3f} ({time.time() - t0:.0f} s)", flush=True)
        del sides, case
        torch.cuda.empty_cache()
    results["after"] = gpu_state()
    results["refused"] = refused
    Path(args.out).write_text(json.dumps(results, indent=1))
    if refused:
        print(f"REFUSED: control outside 1.00 +/- {CONTROL_TOL} on {refused}", file=sys.stderr)
        sys.exit(2)


def fmt_us(s):
    return f"{s['median_us']:.1f} ({s['p50_us']:.1f}, {s['iqr_us']:.1f})"


def rate(s, side):
    if "tflops" in s:
        return f"{s['tflops']:.0f} TFLOP/s"
    if "gbs" in s:
        return f"{s['gbs']:.0f} GB/s"
    return ""


def err_text(e):
    return f"{e['max_abs']:.2e} / {e['rel']:.1e}" if e else "n/a"


ROWS = {
    1: "batch_decode_attention, 27B (hd256, group 6), 8 seqs",
    2: "batch_decode_attention, head_dim 128, GQA 8:1 and 4:1, 8 seqs",
    3: "attention_split, 122B (hd256, group 16), 1 seq",
    4: "NVFP4 prefill GEMM, 27B shapes",
    5: "bf16 prefill GEMM (gate_up_mma_big), 4B and gpt-oss",
    6: "flash_mma, 27B hd256 prefill",
    7: "rmsnorm and add_rmsnorm",
    8: "NVFP4 GEMV at batch 1, against the roofline",
}


def report(args):
    r = json.loads(Path(args.json).read_text())
    out = []
    out.append("# Kernel timing tables (#1158, generated)")
    out.append("")
    out.append(f"Generated by `crates/specrt/scripts/kernel_vs.py report` from `{Path(args.json).name}`. The reading is in [kernel-vs.md](kernel-vs.md).")
    out.append("")
    out.append(f"- Date {r['date']}, commit {r['commit']}.")
    out.append(f"- Command: `{r['command']}`")
    v = r["versions"]
    out.append(f"- torch {v['torch']} (CUDA {v['cuda']}), FlashInfer {v['flashinfer']}, Triton {v['triton']}, vLLM {v['vllm']}, NVRTC {v['nvrtc']}.")
    out.append(f"- {r['iters']} timed launches per side and mode after {r['warmup']} warmup launches, sides alternated launch by launch with a rotating start. Warm: the side runs once untimed right before each timed launch. Cold: a {r['flush_bytes'] >> 20} MB buffer is zeroed before each timed launch. A {r['spin_cycles']} cycle GPU spin precedes each timed launch so host enqueue time is not counted.")
    out.append(f"- Card before: `{r['before']['query']}`")
    out.append(f"- Card after: `{r['after']['query']}`")
    if r.get("refused"):
        out.append(f"- REFUSED rows (control outside 1.00 +/- {CONTROL_TOL}): {', '.join(r['refused'])}")
    out.append("")
    out.append("Times are in microseconds: the 10 to 90% trimmed mean, then (median, IQR). The trimmed mean is the centre used for every ratio and the control, because single cold launches are timed by CUDA events that step in about 2 us units, which a median cannot resolve at a 2% control. Warm rows time a chain of launches sized to about 200 us and divide. Error is max abs / max abs relative to the largest reference value, against an f64 host reference. `ours / theirs` is their time divided by ours: above 1 we are faster.")
    for row in sorted({c["row"] for c in r["cases"]}):
        out.append("")
        out.append(f"## {row}. {ROWS[row]}")
        out.append("")
        out.append("| Case | Side | Warm us | Cold us | Warm rate | Cold rate | Error | ours / theirs warm | ours / theirs cold | Verdict warm / cold | Control |")
        out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for c in [c for c in r["cases"] if c["row"] == row]:
            if "error" in c:
                out.append(f"| {c['name']} | setup failed: {c['error'][:150]} | | | | | | | | | |")
                continue
            for label, s in c["sides"].items():
                ours = label == c["ours"]
                if ours:
                    roof = f"floor {c['roof']['floor_us']:.1f} us, {s['warm']['of_roofline']:.0%} / {s['cold']['of_roofline']:.0%} of roofline"
                    out.append(f"| {c['name']} | {label} | {fmt_us(s['warm'])} | {fmt_us(s['cold'])} | {rate(s['warm'], label)} | {rate(s['cold'], label)} | {err_text(s['error'])} | {roof} | | | {c['control']['warm']:.3f} / {c['control']['cold']:.3f} |")
                else:
                    out.append(f"| | {label} | {fmt_us(s['warm'])} | {fmt_us(s['cold'])} | {rate(s['warm'], label)} | {rate(s['cold'], label)} | {err_text(s['error'])} | {s['warm']['ours_vs_theirs']:.2f} | {s['cold']['ours_vs_theirs']:.2f} | {s['warm']['verdict']} / {s['cold']['verdict']} | |")
            for n in c["notes"]:
                out.append(f"| | note: {n} | | | | | | | | | |")
    out.append("")
    out.append("## Card clocks before the run")
    out.append("")
    out.append("```text")
    out.append(r["before"]["clock"].strip())
    out.append("```")
    Path(args.doc).write_text("\n".join(out) + "\n")


def prepare(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from flashinfer.jit.attention import modules as m
    from flashinfer.jit import norm
    specs = []
    f16 = torch.float16
    for hd in (256, 128):
        specs.append(m.gen_batch_decode_module(f16, f16, f16, torch.int32, hd, hd, 0, False, False))
        specs.append(m._gen_batch_prefill_primary_module("fa2", f16, f16, f16, torch.int32, hd, hd, 0, False, False, False))
        specs.append(m._gen_batch_prefill_independent_paged_module("fa2", f16, f16, f16, torch.int32, hd, hd, 0, False, False, False))
    specs.append(norm.gen_norm_module())
    for s in specs:
        t = time.time()
        s.build()
        print(f"built {s.name} in {time.time() - t:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("dir")
    r.add_argument("out")
    r.add_argument("--iters", type=int, default=200)
    r.add_argument("--warmup", type=int, default=20)
    r.add_argument("--only", default="")
    r.add_argument("--commit", default="")
    r.add_argument("--command", default="")
    p = sub.add_parser("report")
    p.add_argument("json")
    p.add_argument("doc")
    args = ap.parse_args()
    {"prepare": prepare, "run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
