# vLLM's JIT kernel warmup contract, read for specrt

Source: <https://docs.vllm.ai/en/latest/contributing/jit_kernel_warmup/>, the installed vLLM 0.27.1 in
`~/.venv-vllm` and vLLM main at `8db73f2` (cloned to `/tmp/vllm-src`). Issue #394. Research only, nothing
measured on the GPU for this note.

## The contract in our terms

| vLLM | what it does | specrt equivalent |
|---|---|---|
| `VllmJitKernel` with a frozen `CompileKey` (`model_executor/warmup/jit_warmup.py:651`) | one wrapper per JIT kernel; the key holds only what the backend specialises on (tile, head dim, dtype, alignment class) | one `Module` per (spec, `max_ctx`, `kv_planes`, `[runtime]` options), `codegen.rs` `generate_kv`; every dim is a template constant |
| `dispatch(...)` shared by the runtime call and warmup | the runtime path and the warmup enumerator cannot drift | none. `Runtime::tc` (`runtime.rs:969`) builds a kernel name with `format!("{name}_mma{suffix}")` and looks it up in a `HashMap` that panics on a miss; codegen lists the names separately in `KERNELS`, `GDN_KERNELS`, `FP4_KERNELS` and friends |
| `get_warmup_keys(config)` via `_trace_dispatch`: `WarmupIntRange` (with `advance` for power-of-two steps), cartesian products, `zip_inputs`, `_when`, deduplicated keys | enumerate the key space from `max_num_batched_tokens`, `max_num_seqs` and model dims | not needed for kernels: specrt compiles every variant (`_tiny`, `_narrow`, `_big`, `f4b8`/`f4b16`, `f8b8`/`f8b16`, MTP) into the one module at load |
| `register_warmup()` is metadata only; `JitWarmupRegistry.warmup()` (`jit_warmup.py:1080`) compiles each distinct key once | no side effects at model build; one place materialises | `Runtime::load_layout` (`runtime.rs:451`) generates and loads the whole module before weights upload |
| compile-only materialisation: `TritonWarmupTensor` fake pointers, `kernel.warmup(...)`, no launches, no allocation | warmup never runs a real workload | already true: NVRTC compiles source, nothing launches |
| `compile_triton_warmup` (`jit_warmup_triton_helper.py:237`): one `ThreadPoolExecutor` with `triton.AsyncCompileMode`, `VLLM_TRITON_JIT_WARMUP_NUM_THREADS` (default 4) | parallel compile, bounded jobs | one serial `compile_ptx_with_opts` call (`jit.rs:67`) |
| `jit_monitor` (`utils/jit_monitor.py`): after warmup, every Triton/CuTeDSL/TileLang compile or autotune miss logs a warning, `--jit-monitor-mode=error` raises | "cold compile on the hot path" is visible and can fail CI | kernels cannot compile mid-serve, but decode CUDA graphs are captured lazily (below) and nothing reports it |
| Triton cache in `TRITON_CACHE_DIR` under the torch.compile cache dir, keyed by Triton's own `compute_cache_key` (source, specialization, options, device); torch.compile cache dir is a hash of env, config, traced code and compiler (`compilation/backends.py:1031`) | persistent, shared across processes | `~/.cache/specrt` or `[runtime] cache_dir`, file name `sha256(source, arch, options).ptx` (`jit.rs:28`) |
| CUDA graph sizes `[1, 2, 4] + range(8, 256, 8) + range(256, max, 16)`, max `min(max_num_seqs * (1 + spec_tokens) * 2, 512)`, `max_num_batched_tokens` appended (`config/vllm.py:1778`); all captured at startup in `capture_model()` after `kernel_warmup` | no capture during serving | `GRAPH_SIZES = [1, 2, 4, 8, 16, 32]`, `pad_rows` (`runtime/step.rs:7`); each size captured on first use inside `Runtime::decode` (`step.rs:144`) |

The Qwen3.5 path in vLLM (`qwen_triton_warmup.py`) is not on the new contract yet: it still does small
real launches of the GDN conv, post-conv prep (lengths 1, 2, 16) and the fused sigmoid-gating delta
rule with `torch.empty` tensors. vLLM itself treats that as the legacy shape.

## What specrt already does as well or better

- No kernel ever compiles after load. The whole key space is small and closed (four tile variants,
  two batched-GEMV token buckets, per linear), so the eager "compile everything" choice is the right
  one and the tracer machinery has nothing to buy us.
- Compile-only, no dummy launches, no fake tensors needed.
- The on-disk cache is shared across processes, written atomically (`.tmp` then rename), and keyed on
  the full generated source, so a template edit or a spec change can never hit a stale entry.
- Load logs `jit_key`, `jit_cache_hit`, `jit_secs` (`runtime.rs:712`), and `bench` prints
  `jit_cache_hit` and `load`.

## Gaps

1. **Decode graphs are captured mid-serve.** The first time the scheduler runs 3 rows, 9 rows or 17
   rows, `Runtime::decode` captures and uploads a graph on the request's critical path. Each capture
   records a full forward (64 layers on the 27B). The cost per size is not measured anywhere. vLLM
   captures every size before it accepts requests.
2. **`max_ctx` is in the source.** `MAX_CTX` feeds only `PAGES`, the block-table row stride
   (`templates/ops.cu:27-29`, used once at `ops.cu:927`). Any change to `max_model_len` (server
   config edit, test fixtures at 256 vs 4096, bench at 8192) produces a different 5.5 MB module and a
   cold compile. This is exactly the vLLM rule "exclude runtime-only values from the key".
   `~/.cache/specrt` holds 1,221 PTX files (1.6 GB) after four days, 60 of them over 4 MB.
3. **The cache holds PTX, not CUBIN.** `CudaContext::load_module` then JITs PTX to SASS in the driver,
   cached in `~/.nv/ComputeCache` (1.1 GB here) under the driver's own size limit (256 MiB default per
   NVIDIA). A cold driver cache, a new driver or a fresh container pays that second compile even when
   our PTX is cached. The key also leaves out the NVRTC version (13.2 and 13.5 are both installed), so
   a compiler upgrade keeps serving PTX from the old one. There is no eviction.
4. **Serial compile.** One NVRTC call for the whole module. NVRTC 13.2 accepts `--split-compile=N`
   (present in `libnvrtc.so.13.2.86`). Not tried.
5. **No prebuild.** The Docker image (`packaging/docker/Dockerfile`) ships specs but no kernel cache,
   so every container start is a cold JIT. vLLM has the same problem and solves it with a mounted
   cache dir; we can do better because a module needs only the spec and the layout, no weights.
6. **Runtime names and codegen names can drift.** A spec or option combination that makes `tc` ask
   for a kernel codegen did not emit panics at `self.kernels[name]` (`runtime.rs:732`) mid-request.
   Tests check that some names exist (`tests/codegen.rs`), not that every name the runtime can ask
   for over the config space exists.
7. **Cold-start numbers are not tracked.** See below.

## Cold-start numbers

| what | number | where |
|---|---|---|
| 27B FP8, whole load | 5.8 s with a warm JIT cache, 11.6 s on the first run | `docs/specrt-27b.md` |
| 27B NVFP4, load | 5.0 to 5.5 s upload, 14 s with a cold JIT | `docs/specrt-nvfp4.md` |
| qwen3-0.6b on the Spark, warm | `jit_secs` 0.002 to 0.006 s, `load_secs` 0.8 to 1.06 s | `docs/spark-run/vllm-20261106/run.log` |
| qwen3-0.6b and qwen2.5-0.5b on the Spark, cold | `load` 5.4 to 6.9 s | `docs/spark-run/decode-ctx-20261107/*-r1.txt` |
| vLLM 27B for comparison | `torch.compile` 68.7 s, engine init 153 s with its compile cache warm | `docs/qwen3.8-27b-vllm-baseline.md` |

Missing, filed as a measurement issue: cold `jit_secs` split into NVRTC and driver load (PTX cache cold
vs ComputeCache cold), graph capture seconds per decode size on the 27B, and first-request TTFT right
after load against steady-state TTFT. The 6 to 9 s cold penalty on the 27B is the whole JIT today;
how it splits decides between items 3 and 4.

## Ranked items

| rank | item | expected win | cost | issue |
|---|---|---|---|---|
| 1 | Capture every decode graph size up to `row_capacity(slots)` at load; count and log any capture after startup; serving test asserts zero | removes per-size capture stalls from live requests; makes them visible | small, `step.rs` and `sched.rs` | #395 |
| 2 | Measure cold start: NVRTC vs driver split, capture per size, first-request TTFT | decides items 3 and 4 | small, GPU time | #396 |
| 3 | Take `max_ctx` out of the generated source (pages stride as a kernel argument) | no cold compile when context changes; cache stops growing per `max_ctx` | small to medium, `ops.cu`, `codegen.rs`, launch sites | #397 |
| 4 | Cache CUBIN for the exact `sm_XX`, key on NVRTC version, bound the cache size; `specrt warmup` CLI that prebuilds from spec and layout without weights, used by the Docker image | cold container start loses both compiles; driver cache no longer matters | medium, `jit.rs`, cli, Dockerfile | #398 |
| 5 | One `kernel_for(...)` function shared by `Runtime::tc` and a CPU test that sweeps tokens 1 to `prefill_chunk` over every spec and `[runtime]` option and checks each name is in `Module::kernels` | turns a mid-request panic into a test failure | small, CPU only | #399 |
| 6 | `--split-compile` for NVRTC | cold `jit_secs` down by an unknown factor | one option, keep only if item 2 shows NVRTC dominates | #398 (folded in) |

Not taken: the AST tracer, `WarmupIntRange`, `zip_inputs` and `_when` exist because vLLM's key spaces
are open-ended Python; ours is a closed list baked by codegen. Per-key lazy modules would undo the
property that nothing compiles mid-serve. FlashInfer-style autotune caching overlaps the load-time
tile and split tuner from the jitllm review (#393), which owns it; shape-specialised emission at load
is already what `codegen.rs` does.
