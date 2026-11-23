import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

OUT = Path(sys.argv[1])
ARCHS = sys.argv[2].split(",")
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(OUT / "fi-work"))

import torch  # noqa: E402
import flashinfer  # noqa: E402
import triton  # noqa: E402

KERNEL_TUS = {
    "batch_decode": ["batch_decode_kernel.cu"],
    "batch_prefill": ["batch_prefill_paged_kernel_mask_1.cu"],
    "norm": ["norm.cu"],
    "rope": ["rope.cu"],
    "silu_and_mul": ["silu_and_mul.cu"],
}


def arch_flag(arch):
    cc = arch[3:]
    return f"-gencode=arch=compute_{cc},code={arch}"


def nvcc_flags(spec):
    spec.write_ninja()
    text = Path(spec.ninja_path).read_text()
    block = re.search(r"cuda_cflags = (.*?)\ncuda_post_cflags", text, re.S).group(1)
    common = re.search(r"common_cflags = (.*?)\ncflags", text, re.S).group(1)
    flat = lambda s: [t for t in shlex.split(s.replace("$\n", " ")) if t not in ("$common_cflags",)]
    flags = flat(common) + flat(block)
    cuda_home = re.search(r"cuda_home = (\S+)", text).group(1)
    flags = [f.replace("$cuda_home", cuda_home) for f in flags]
    drop = ("--compiler-options=-fPIC", "-Xfatbin=-compress-all", "--compress-mode=size")
    kept = []
    skip = False
    for i, f in enumerate(flags):
        if skip:
            skip = False
            continue
        if f == "-isystem" and "include/python3" in flags[i + 1]:
            skip = True
            continue
        if f.startswith("-gencode") or f in drop:
            continue
        kept.append(f)
    return kept, cuda_home


def compile_tu(name, spec, tus, arch, rows):
    flags, cuda_home = nvcc_flags(spec)
    gen = [Path(s) for s in spec.sources]
    out = []
    for tu in tus:
        src = next((s for s in gen if s.name == tu or s.name.endswith(tu)), None)
        if src is None:
            rows.append({"competitor": name, "arch": arch, "error": f"no source {tu} in {[g.name for g in gen]}"})
            continue
        dst = OUT / arch / f"fi-{name}-{src.stem}.cubin"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [f"{cuda_home}/bin/nvcc", "-cubin", arch_flag(arch), *flags, str(src), "-o", str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            rows.append({"competitor": name, "arch": arch, "error": ("\n".join(l for l in r.stderr.splitlines() if "error" in l) or r.stderr[-600:])[:600], "cmd": " ".join(cmd)})
        else:
            out.append(dst.name)
    return out


def flashinfer_specs():
    from flashinfer.jit.attention import modules as m
    from flashinfer.jit import norm, rope, activation
    specs = {}
    for hd in (128, 256):
        specs[f"batch_decode_hd{hd}"] = (m.gen_batch_decode_module(torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32, hd, hd, 0, False, False), KERNEL_TUS["batch_decode"])
        specs[f"batch_decode_hd{hd}_fp8kv"] = (m.gen_batch_decode_module(torch.bfloat16, torch.float8_e4m3fn, torch.bfloat16, torch.int32, hd, hd, 0, False, False), KERNEL_TUS["batch_decode"])
        specs[f"batch_prefill_hd{hd}"] = (m.gen_batch_prefill_module("fa2", torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32, hd, hd, 0, False, False, False), KERNEL_TUS["batch_prefill"])
    specs["norm"] = (norm.gen_norm_module(), KERNEL_TUS["norm"])
    specs["rope"] = (rope.gen_rope_module(), KERNEL_TUS["rope"])
    specs["silu_and_mul"] = (activation.gen_act_and_mul_module("silu"), KERNEL_TUS["silu_and_mul"])
    return specs


def main():
    from flashinfer.jit import core
    rows = []
    built = {}
    for arch in ARCHS:
        cc = arch[3:].rstrip("a")
        os.environ["FLASHINFER_CUDA_ARCH_LIST"] = f"{cc[:-1]}.{cc[-1]}"
        core.refresh_current_compilation_context()
        for name, (spec, tus) in flashinfer_specs().items():
            cubins = compile_tu(name, spec, tus, arch, rows)
            if cubins:
                built.setdefault(arch, {})[name] = cubins
    info = {"flashinfer": flashinfer.__version__, "triton": triton.__version__, "torch": torch.__version__, "built": built, "errors": rows}
    (OUT / "competitors.json").write_text(json.dumps(info, indent=1, sort_keys=True))
    for r in rows:
        print("error", r, file=sys.stderr)
    for a, b in built.items():
        print(a, sorted(b))


main()
