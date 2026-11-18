# specrt on wgpu

specrt runs a model from its spec and the op plan (`plan.rs`). The CUDA backend turns that into NVRTC kernels. The wgpu backend turns the same spec and plan into WGSL and runs it on whatever wgpu reaches: Vulkan (NVIDIA, AMD, Intel), Metal (Apple), DX12 (Windows).

## Selecting a backend

```toml
[runtime]
backend = "auto"   # "cuda" | "wgpu" | "auto"
```

`auto` uses CUDA when a driver and device initialise and wgpu otherwise. The same key is accepted in the server config `[runtime]` table. `cargo build --features wgpu` compiles the backend; without the feature a spec that asks for `wgpu` fails to load with a message that says so.

## Layers

| Piece | File | Backend neutral |
|---|---|---|
| Plan (ops, buffers, weights) | `plan.rs` | yes |
| Backend trait (`alloc`, `upload`, `download`, `compile`, `dispatch`, `flush`) | `backends/mod.rs` | yes |
| WGSL generation from the spec | `wgsl.rs`, `templates/wgsl/*.wgsl` | yes (plain strings) |
| Plan executor (`run_plan`, trait `PlanBackend`) | `executor.rs` | yes |
| Dense plan runner on `Backend` | `backends/dense.rs` | yes, generic over `Backend` |
| wgpu device, pipelines, buffers | `backends/wgpu.rs` | feature `wgpu` |
| CUDA runtime | `runtime.rs`, `jit.rs` | CUDA only |

Both backends run the same `Plan` (`plan::Plan::build`) through `executor::run_plan`: `Runtime` (CUDA, `runtime/exec.rs`) and the wgpu `Dense` runner each implement `PlanBackend::run_op`. The wgpu side maps each `Op` to WGSL kernels in `wgsl::kernels_for`; there is no hand-written layer loop. CUDA launches stay in `runtime/`, behaviour and speed unchanged. Speed gate, 27B NVFP4 on the RTX PRO 6000, base `9b27e03` against `bc75dc6`, interleaved base, new three times each under `flock /tmp/gpu27.lock` (each run is the median of 3 inside `bench`; the table is the median of the three):

| Case | Base | New | Delta |
|---|---|---|---|
| Prefill 1,024 (tok/s) | 1490.2 | 1492.6 | +0.16% |
| Prefill 4,096 (tok/s) | 1490.0 | 1494.5 | +0.30% |
| Decode batch 1 (tok/s) | 55.4 | 55.5 | +0.18% |
| Decode batch 8 (tok/s aggregate) | 332.3 | 332.4 | +0.03% |

All within 2%. real27b 10/10; the non-27B suite passes with `SPECRT_REF_DIR=/tmp/ref387`.

What is not moved: the CUDA buffers, graphs, batching, MTP and delta-net still use cudarc directly rather than the byte-level `Backend` trait. The `wgpu` kernel choice also does not go through `kernel_for` yet, because `kernel_for` picks CUDA GEMM tiles and tensor-core paths; the wgpu plan is `Plan::build(spec, phase, false)` and its two linear kernels are chosen by token count.

## Weights and the decoder interface

Every linear is a `DeviceRows`: the packed GGUF-style row table the CUDA GGML path already uses (`gguf.rs`). A row table entry holds the block offset and a format code (Q4_K, Q5_K, Q6_K, Q8_0, bf16, f16). The WGSL `dequant_at(row_base, code, element)` in `decode.wgsl` is the decoder; a GEMV or GEMM reads weights only through it, which is the pluggable-decoder shape #462 is giving the CUDA template. A new format adds one `case`. bf16 safetensors are packed as bf16 rows, so the same kernels serve bf16, Q8_0 and the K-quants. Q4_K, Q5_K, Q6_K and Q8_0 decode is tested bit-exact against `gguf::dequantize_*`.

## Kernels

`embed`, `rmsnorm`, `rope` (QK norm, NeoX rotation, linear and llama3 scaling, KV write), `linear` (row per workgroup, four tokens per workgroup, optional accumulate into the residual), `bias`, `attention` (causal, online softmax, GQA, reads the paged-by-layer KV), `swiglu` (silu and gelu tanh), `argmax` (greedy and Gumbel).

KV is f32 per layer, kv head and position. No subgroup ops are used yet: the reductions are workgroup memory, which every adapter has.

## Limits

The device asks for the adapter's own limits (`max_buffer_size`, `max_storage_buffer_binding_size`, `max_storage_buffers_per_shader_stage`). A weight larger than the binding limit, or fewer storage buffers than a kernel needs, fails the load with the number in the message instead of a panic. WGSL source is cached on disk next to the CUDA cache, keyed like `jit::cache_key` (source plus adapter name plus options). `Features::PIPELINE_CACHE` data is stored there too when the adapter has it.

## Not in the first slice

Each is its own issue. A spec that needs one fails the wgpu check with `<feature> is not yet on wgpu`.

- NVFP4 and FP8 weights (no WGSL tensor-core MMA; a dequant-to-bf16 fallback is possible and would be flagged slow).
- Gated delta-net layers (Qwen3.5/3.8 hybrids), output-gated attention, sliding window attention.
- MTP and speculative decoding.
- CUDA-graph equivalents beyond reusing one command buffer per step.
- Prefix cache snapshots and the batched scheduler (`sched.rs`, `serve.rs`): wgpu runs through `Engine`, one sequence.
- Subgroup-accelerated reductions and a tiled GEMM.
- Moving the CUDA buffers and launches under the byte-level `Backend` trait.

## Tests

`tests/wgsl.rs` (needs `--features wgpu` and a real adapter, `#[ignore]` otherwise): each kernel against a CPU reference, Q8_0 and Q4_K exactness, then model logits and greedy tokens against the stored CUDA/llama.cpp references. Lavapipe does not count; the tests refuse a CPU adapter.

## Results

All 18 tests in `tests/wgsl.rs` pass on both machines (`cargo test --release --features wgpu --test wgsl -- --include-ignored --test-threads=1`). Q8_0, Q4_K, Q5_K, Q6_K, bf16 and f16 decode are bit-exact with the CPU dequant.

Logits against the CUDA backend on an 18 token prompt (KL bound: mean 1e-4 and max 1e-3 for bf16, mean 1e-3 and max 1e-2 for GGUF; top-1 must match on every position):

| Model | Mean KL | Max KL | Greedy |
|---|---|---|---|
| Qwen3-0.6B bf16 | 1.1e-10 | 4.4e-10 | 64/64 equal |
| Qwen3-0.6B Q4_K_M | 7.3e-5 | 2.5e-4 | 32/32 equal |
| Llama-3.2-1B Q4_K_M | 1.1e-4 | 4.0e-4 | 32/32 equal |

Identical on Vulkan and Metal to the printed digits for the quantised models.

Speed, Qwen3-0.6B bf16, release build, no tuning (row-per-workgroup GEMV, scalar attention):

| Adapter | Prefill 1,024 tokens | Decode batch 1 |
|---|---|---|
| RTX PRO 6000 Blackwell (Vulkan) | 0.65 s (1,582 tok/s) | 41.3 tok/s |
| Apple M1 Pro (Metal) | 14.9 s (69 tok/s) | 18.8 tok/s |

Decode is dominated by dispatch count and the scalar kernels, not bandwidth. Speed work (subgroups, tiled GEMM, fused ops, command buffer reuse) is the follow-up. Metal lost the device when eight decode steps were recorded into one command buffer, so decode submits four steps at a time (`DECODE_STEPS_PER_SUBMIT`).

## M1 Pro vs llama.cpp

Measured 2026-11-18 on the M1 Pro (16 GB, macOS 26.5, Metal adapter "Apple M1 Pro"), specrt at `a085c63` (wgpu 30), llama.cpp `61f182f` built from source with `-DGGML_METAL=ON`. Same GGUF file in both engines (`~/models`): Qwen3-0.6B bf16 (converted from the same safetensors), Qwen3-0.6B Q4_K_M, Llama-3.2-1B-Instruct Q4_K_M, Qwen2.5-7B-Instruct Q4_K_M. Three rounds, engine order alternating each round, medians of the rounds. specrt through `bench --wgpu` (median of 3 runs per round, 1 run for the 7B), llama.cpp through `llama-bench -ngl 99 -p 128,1024 -n 128 -r 3`. Rerun everything with `crates/specrt/scripts/mac_vs_llama.sh`.

Correctness first: `cargo test --release --features wgpu --test wgsl -- --include-ignored --test-threads=1` on the Mac: `test result: ok. 23 passed; 0 failed`. Greedy text for "The capital of France is", 32 tokens, `bench --text` against `llama-completion --temp 0` on the same GGUF: identical for Qwen3-0.6B Q4_K_M, Qwen2.5-7B Q4_K_M and Qwen3-0.6B bf16. Llama-3.2-1B differs from the first token after "Paris." because the two engines tokenise the prompt differently (BOS handling), not because of the kernels: its logits match the CUDA reference in the wgsl test (KL 1.1e-4, 32/32 greedy).

| Model | Metric | specrt wgpu | llama.cpp | specrt / llama.cpp |
|---|---|---|---|---|
| Qwen3-0.6B bf16 | prefill 128 (tok/s) | 83 | 2,943 | 0.028 |
| | prefill 1,024 | 69 | 2,966 | 0.023 |
| | decode batch 1 | 22.6 | 114.8 | 0.20 |
| | load (s) | 2.8 | 0.26 | 11x slower |
| | peak footprint (GB) | 5.5 | 0.22 reported | |
| Qwen3-0.6B Q4_K_M | prefill 128 | 79 | 3,010 | 0.026 |
| | prefill 1,024 | 66 | 3,028 | 0.022 |
| | decode | 18.8 | 181.4 | 0.10 |
| | load (s) | 0.80 | 0.20 | 4x slower |
| Llama-3.2-1B Q4_K_M | prefill 128 | 38.9 | 1,731 | 0.022 |
| | prefill 1,024 | 36.8 | 1,754 | 0.021 |
| | decode | 13.8 | 144.4 | 0.095 |
| | load (s) | 1.83 | 0.23 | 8x slower |
| Qwen2.5-7B Q4_K_M | prefill 128 | 4.2 | 271 | 0.016 |
| | decode | 2.1 | 27.9 | 0.075 |
| | load (s) | 11.1 | 0.55 | 20x slower |

The llama.cpp peak footprint column is not comparable: `llama-bench` maps the GGUF and Metal reports the weights outside the process footprint, so only specrt's number is meaningful (peak 3.8 GB for the 0.4 GB Q4_K_M file, 5.5 GB for the 1.5 GB bf16 file, 11.9 GB for the 7B: the host copy of the weights is kept next to the device copy). The 7B run swapped heavily (swap was at 10 GB used before it started, caused by the iOS simulators on the Mac), so its number is a floor.

Load beside each round (all rounds: gcad sim log showed no job in its last 40 lines; the stage host LaunchDaemon idle):

| Round | uptime load averages (1/5/15 min) | swap used |
|---|---|---|
| 1 | 3.60 2.25 4.11 | 10.7 GB |
| 2 | 1.02 1.68 3.18 | 10.4 GB |
| 3 | 1.88 2.01 2.76 | 10.2 GB |
| 7B | 2.81 2.00 2.42 | 10.2 GB |

Load comes from the iOS simulators the Mac runs permanently, so absolute numbers are a little noisy. The gap is 10-50x, far above that.

### Where the gap is

Per-kernel time from a synchronous build (each dispatch submitted and waited on, temporary patch, not committed; Metal System Trace was tried first, but `xctrace record` did not finish writing the trace for the wgpu process in 5 minutes). Qwen3-0.6B bf16, 28 layers:

Prefill 1,024 tokens, 16.2 s in total:

| Kernel | Dispatches | Time | Per dispatch |
|---|---|---|---|
| `linear_t8` (batched GEMM) | 1,807 | 12.4 s (76%) | 6.8 ms |
| `attention` (scalar) | 1,344 | 3.9 s (24%) | 2.9 ms |
| `linear` (GEMV, lm head) | 3,617 | 1.7 s | 0.47 ms |
| `rmsnorm`, `rope`, `swiglu`, `embed`, `argmax` | 6,500 | 1.5 s | 0.2-0.5 ms |

Decode, 32 tokens, 2.7 s in total: `linear` 1.54 s, `rmsnorm` 0.39 s, `attention` 0.36 s, `rope` 0.20 s, `swiglu` 0.19 s. One decode step is about 255 dispatches (113 linear, 57 rmsnorm, 28 each of rope, attention and swiglu, plus embed and argmax). The small element-wise kernels cost 0.2 ms each in the synchronous build, which is the submit round trip, so the real decode cost per dispatch (22.6 tok/s is 44 ms a token for 255 dispatches, 0.17 ms each) is the same order: decode is dispatch bound. The weights are 1.5 GB; at the M1 Pro's roughly 200 GB/s a bandwidth bound decode would be about 130 tok/s, which is what llama.cpp reaches (115 tok/s), so llama.cpp is at the memory limit and specrt is at about a fifth of it. The llama.cpp dispatch count per token was not measured (no Metal trace); its graph fuses norm, rope and the residual adds into fewer kernels than the 255 above.

Biggest gap: prefill (40-60x). The `linear_t8` GEMM has no threadgroup tiling and no simdgroup matrix instructions, so it sustains about 0.1 TFLOP/s against the M1 Pro's roughly 5 TFLOP/s f16, and the attention kernel is one scalar pass per head and row (llama.cpp uses a flash attention kernel with simdgroup matrices). Decode is second (5-10x): dispatch overhead and untiled GEMV.

Speed fixes filed as follow-ups: tiled GEMM with subgroup matrix ops for prefill, flash attention for prefill, fused element-wise ops and fewer dispatches per decode step, a q4_K/q8 GEMV tuned for Apple GPUs, and releasing the host weight copy after upload (peak memory and load time).
