# Engine and runtime

`crates/specrt` (lib `specrt`) runs a model from its spec on generated kernels. A model is a spec TOML (or a family spec over the checkpoint's `config.json`, see [specs.md](specs.md)); the graph lowers to a plan, kir emits the kernels, NVRTC compiles them, and the runtime runs the plan. `cudarc` uses dynamic loading, so a build needs no CUDA toolkit; a run needs `libcuda` and `libnvrtc`.

## Pieces

| Path | What it does |
|---|---|
| `crates/specrt/src/spec.rs`, `family.rs` | `Spec`, `Spec::from_hf_config`, `RuntimeOptions` (`[runtime]`) |
| `crates/specrt/src/graph.rs`, `plan.rs` | graph spec to `Vec<Op>`, per phase (prefill, decode, verify) |
| `crates/specrt/src/compile.rs`, `fusion.rs`, `signature.rs` | plan passes, see [kir-compiler.md](kir-compiler.md) |
| `crates/specrt/src/codegen.rs`, `gemm.rs` | spec to kir kernels and the module (`Module::generate_kv`), GEMM choice, see [gemm.md](gemm.md) |
| `crates/specrt/src/jit.rs` | NVRTC compile and the kernel cache |
| `crates/specrt/src/runtime.rs`, `runtime/` | `Runtime`: load, `step`, batched decode, MTP rounds, snapshots, tuning |
| `crates/specrt/src/loader.rs` | weight loading |
| `crates/specrt/src/sched.rs`, `serve.rs` | scheduler and serving core, see [scheduler.md](scheduler.md) |
| `crates/specrt/src/engine.rs` | `Engine`: tokenizer, chat template, sampling defaults, the request API the server uses |
| `crates/specrt/src/options.rs` | the option schema, profiles |
| `crates/specrt/src/backends/` | the wgpu backend (`wgpu.rs`, `dense.rs`), see [wgpu.md](wgpu.md) |

`[runtime] backend` is `auto` (CUDA when a CUDA device is present, else wgpu), `cuda` or `wgpu`.

Decode steps and batched decode at each captured row count run as CUDA graphs (`[runtime] graphs`, on). Graph capture is 2 to 5 ms per size and happens at load. `[runtime] pdl` (off) launches through programmatic dependent launch on cc 9.0 and up.

## Options and profiles

Every engine option is a field of `ServeOptions`, `SpeculativeOptions`, `RuntimeOptions`, `PrefixCacheOptions` or `LoaderOptions`. `options::schema()` flattens them into key, kind, default, group, label and help; the load dialog and `bench --a-set/--b-set` read it, and a test fails when an option has no label, help or group. The full table with defaults is [../generated/engine-options.md](../generated/engine-options.md), written by `crates/specrt/tests/options_doc.rs`.

| Profile | Sets |
|---|---|
| max performance (default) | the schema defaults, prefix cache on |
| low memory | MTP off, fp8 KV, `requant.mlp_tail = nvfp4_g16`, 2 parallel requests, 2,048 batched tokens, prefill chunk 512, prefix cache off with 0 snapshots, vision cache 64 MB and batch 1, loader host budget 512 MB |
| custom | shown once an advanced field differs from the profile |

The app keeps the choice per model in `settings.json` (`model_engine`) with a default (`default_profile`). `serve.max_num_seqs` defaults to 8 (`DEFAULT_MAX_NUM_SEQS`): on the 27B NVFP4 a lone request decodes at full speed with the cap at 8, aggregate decode still grows at 8 (97, 171, 265, 367 tok/s at 1, 2, 4, 8 requests), and each slot costs one recurrent state (150 MB on the 27B) (#598).

## Kernel cache

`Jit::load` compiles a module with NVRTC to a CUBIN for the exact `sm_XX` (`nvrtcGetCUBIN`) and stores `<key>.cubin` under `[runtime] cache_dir` (default `~/.cache/specrt`), so the driver never JITs PTX; PTX is the fallback when NVRTC cannot emit a CUBIN. The key hashes the source, arch, options, `kir_hash`, and the NVRTC and driver versions. `[runtime] cache_max_bytes` (default 4 GB) evicts the oldest files by mtime. `[runtime] compile_jobs` sets `--split-compile` (default the CPU count, at most 8): a cold 27B FP8 compile takes 5.0 s with 8 jobs against 11.6 to 12.3 s with 1 (#398).

`inference-bench warmup --spec <spec> --max-ctx N --slots N` builds every module without loading weights. It completes the spec from the checkpoint config first (`runtime::warmup_spec`), so the server's later load hits the cache; `max_ctx`, `slots` and `kv_planes` are part of the key, the prefill chunk is not.

## Weight loading

`crates/specrt/src/loader.rs` and `load_weights` in `runtime.rs`. Weights are split into groups (the head with the NVFP4 draft head made from it, the first half of the layers, the embedding, the MTP layer, the rest), each read, converted on the host, copied on its own stream and released. A reader thread does O_DIRECT `pread`s into aligned buffers (buffered reads when the filesystem refuses); `workers` convert and copy; a host budget caps bytes in flight. When 80 percent of the checkpoint is already in the page cache (`mincore`) the reader is skipped and workers convert straight from the mmap.

`[runtime.loader]` options and defaults: `enabled` true, `direct` true, `chunk_mb` 32, `readers` 8, `ahead` 2, `workers` 4, `lanes` 1, `skip_cached` true, `pinned` false, `host_budget_mb` 3584. 3,584 MB is the largest budget that keeps the 27B's peak RSS under the old mmap path; pinned buffers made cold loads slower (#583).

Load progress (compiling, reading weights with bytes, allocating KV) goes through `specrt::progress` to `EngineEvent::LoadProgress`, the app's status bar and the server's `/health` and `/admin/models`.

## Measured state

27B NVFP4 on desktop GPU1 (RTX PRO 6000 Max-Q), `crates/specrt/scripts/coldstart.sh`, #1013, commit 8ac5740:

| State | load | NVRTC | module load |
|---|---|---|---|
| both caches warm | 3.58 s | 0.00 s | 0.00 s |
| specrt cache cold | 4.44 s | 0.35 s | 0.00 s |
| driver cache cold | 3.87 s | 0.00 s | 0.00 s |
| both cold | 8.41 s | 5.29 s | 0.00 s |

27B NVFP4 weights: 2.85 s cold and 2.34 s warm for 24.78 GB, peak host RSS 5.1 GB (#583). First request TTFT after load equals the steady state.

## Tests

`crates/specrt/tests/jit.rs` (cache, warmup key), `loader.rs`, `codegen.rs`, `golden.rs`, `options_doc.rs`, the option tests in `crates/specrt/src/options.rs`. Tiers and commands: [../guides/testing.md](../guides/testing.md).
