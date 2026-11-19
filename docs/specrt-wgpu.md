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

Correctness first: `cargo test --release --features wgpu --test wgsl -- --include-ignored --test-threads=1` on the Mac: `test result: ok. 23 passed; 0 failed`. Greedy text for "The capital of France is", 32 tokens, `bench --text` against `llama-completion --temp 0` on the same GGUF: identical for Qwen3-0.6B Q4_K_M, Qwen2.5-7B Q4_K_M and Qwen3-0.6B bf16. Llama-3.2-1B differs only because of BOS: `bench --text` tokenises "The capital of France is" to `[791, 6864, 315, 9822, 374]` (no BOS, as `Engine::tokens` never adds special tokens) while `llama-completion` prepends `<|begin_of_text|>` (128000). With the same ids on both sides the first 32 greedy tokens are identical, both ways: with BOS on both (specrt prompt `<|begin_of_text|>The capital of France is`, ids `[128000, 791, ...]`) both give " Paris. The Eiffel Tower is a famous landmark in Paris. The Eiffel Tower is a symbol of France and is one of the most visited tourist", and without BOS on both (`--override-kv tokenizer.ggml.add_bos_token=bool:false`) both give " Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Japan is Tokyo. The capital of South Korea is Seoul. The". Not a kernel bug.

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
| Qwen2.5-7B Q4_K_M | prefill 128 | 4.2 | 274 | 0.015 |
| | prefill 1,024 | 4.1 | 269 | 0.015 |
| | decode | 2.05 | 28.0 | 0.073 |
| | load (s) | 11.8 | 0.49 | 24x slower |

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

Decode, 32 tokens, 2.7 s in total: `linear` 1.54 s, `rmsnorm` 0.39 s, `attention` 0.36 s, `rope` 0.20 s, `swiglu` 0.19 s. One decode step is about 255 dispatches (113 linear, 57 rmsnorm, 28 each of rope, attention and swiglu, plus embed and argmax). The small element-wise kernels cost 0.2 ms each in the synchronous build, which is the submit round trip, so the real decode cost per dispatch (22.6 tok/s is 44 ms a token for 255 dispatches, 0.17 ms each) is the same order: decode is dispatch bound. The weights are 1.5 GB; at the M1 Pro's roughly 200 GB/s a bandwidth bound decode would be about 130 tok/s, which is what llama.cpp reaches (115 tok/s), so llama.cpp is at the memory limit and specrt is at about a fifth of it. llama.cpp's decode graph for the same model (`GGML_METAL_GRAPH_DEBUG=1`, `-v`, Qwen3-0.6B bf16, one token) has 536 launched nodes: 197 MUL_MAT (separate q, k, v, o, gate, up, down), 113 RMS_NORM (each already fused with its MUL), 56 SET_ROWS, 56 ROPE, 56 ADD, 28 FLASH_ATTN_EXT, 28 GLU and 2 GET_ROWS; the fusion stats over the 5 graphs of the run were 565 RMS_NORM+MUL and 54 MUL_MAT+ADD (about 11 per graph, so roughly 525 launches). That is about twice specrt's 255 dispatches, yet 8.7 ms a token against specrt's 44 ms. Dispatch count is therefore not what separates the engines: llama.cpp marks independent nodes `(concurrent)` so Metal overlaps them, and its kernels are far faster each. The per-kernel cost is the first thing to fix, with fewer dispatches second.

Biggest gap: prefill (40-60x). The `linear_t8` GEMM has no threadgroup tiling and no simdgroup matrix instructions, so it sustains about 0.1 TFLOP/s against the M1 Pro's roughly 5 TFLOP/s f16, and the attention kernel is one scalar pass per head and row (llama.cpp uses a flash attention kernel with simdgroup matrices). Decode is second (5-10x): dispatch overhead and untiled GEMV.

Speed fixes filed as follow-ups: tiled GEMM with subgroup matrix ops for prefill, flash attention for prefill, fused element-wise ops and fewer dispatches per decode step, a q4_K/q8 GEMV tuned for Apple GPUs, and releasing the host weight copy after upload (peak memory and load time).

## Mac speed work (#759 steps)

Same method as above (`mac_vs_llama.sh`, 3 alternating rounds, medians, M1 Pro). Each step was measured on Qwen3-0.6B on the Mac and kept only if it helped.

| Step | Change | Result |
|---|---|---|
| #768 | `linear_tile` register-tiled GEMM, then `linear_coop` with 8x8 f32 cooperative matrices (wgpu `EXPERIMENTAL_COOPERATIVE_MATRIX`, `Feature::CoopMatrix`), bf16 pairs read as u32, row entries hoisted, workgroup memory zero-init off | prefill 1,024 bf16: 69 to 231 tok/s. Tile only: 197. f16 matrices: not faster and error 3e-3, dropped. Zero-init off was worth 1.5x on its own. |
| #771 | `linear_sg` simdgroup GEMV (4 rows per workgroup, direct bf16, Q4_K, Q6_K, Q8_0 paths). Metal reports subgroup sizes 4..64 but runs 32, so the adapter check accepts that range on Metal. | decode bf16 27 to 55 tok/s, Q4_K 19 to 42 |
| #770 | One compute pass per submit instead of one per dispatch. Fusing swiglu, rope or norm was measured by skipping them: under 2% of decode each, so no fusion kernels were written. | decode bf16 55 to 82 tok/s, Q4_K 42 to 60 (short context) |
| #769 | `flash` kernel, one simdgroup per query token, one key per lane, shuffle-broadcast probabilities. Used for prefill only: for one token it leaves 16 workgroups and decode drops from 36 to 12 tok/s. Four queries per simdgroup was slower (203 vs 377). | prefill 1,024 bf16: 231 to 377 tok/s |
| #772 | Flush the queue every 128 MB of weight upload so staging copies are released | peak footprint 7B 11.9 to 7.0 GB (9.1 GB in the full run), 0.6B Q4_K_M 3.8 to 2.9 GB. Load time unchanged at 7B (10.5 s): the host dequantised copy is still built, only the staging is bounded. |

After all steps, full `mac_vs_llama.sh` (specrt / llama.cpp, tok/s):

| Model | Prefill 128 | Prefill 1,024 | Decode (128 new) |
|---|---|---|---|
| Qwen3-0.6B bf16 | 680 / 2,940 = 0.23 (was 0.028) | 377 / 2,969 = 0.13 (was 0.023) | 52.8 / 114.4 = 0.46 (was 0.20) |
| Qwen3-0.6B Q4_K_M | 502 / 3,008 = 0.17 (0.026) | 316 / 3,033 = 0.10 (0.022) | 41.5 / 183.5 = 0.23 (0.10) |
| Llama-3.2-1B Q4_K_M | 348 / 1,733 = 0.20 (0.022) | 270 / 1,755 = 0.15 (0.021) | 32.7 / 145.4 = 0.23 (0.095) |
| Qwen2.5-7B Q4_K_M | 62 / 274 = 0.23 (0.015) | 55 / 269 = 0.20 (0.015) | 6.95 / 28.0 = 0.25 (0.073) |

The 0.5x milestone is met only for Qwen3-0.6B bf16 decode. Where the time goes at 1,024 (0.6B, per-dispatch timing): `linear_coop` about 0.95 ms a dispatch (12,544 dispatches), attention about 6 ms a dispatch before `flash`. Decode attention at long context and the GEMM are the next targets.

## Mac speed work 2 (#812 steps)

Same method (`mac_vs_llama.sh`, 3 alternating rounds, medians, M1 Pro). Each step was A/B'd on the Mac and kept only if it helped. Ablations (skipping a stage to see its cost) showed the prefill GEMM cost was dominated by the global-to-workgroup tile loads, not the matrix multiplies.

| Step | Change | Result |
|---|---|---|
| flash | `flash` kernel reads K and V as `vec4` | prefill 1,024 bf16: 377 to 582 tok/s |
| coop loads | `linear_coop` loads x as `vec4` and weights four at a time | prefill 1,024 bf16: 582 to 759, Q4_K 316 to 427 |
| coop Q4_K | Q4_K dequantised straight into the tile with one scale lookup per four weights | prefill 1,024 Q4_K: 427 to 645 |
| sg Q4_K | `linear_sg` reads x as `vec4`, nibbles unpacked into vec4 dots | decode Q4_K 53 to 58 |
| sg Q6_K | Q6_K GEMV reads words, two superblocks per pass | decode Q4_K_M (Q6_K lm head and layers) 58 to 73 |
| decode attention | `flash_decode`: 8 simdgroups split the keys, 8 keys per simdgroup step, partials merged in workgroup memory | decode at 1,024 context: 35 to 70 (Q4_K), 36 to 66 (bf16) |

Tried and dropped: `COOP_K` 32 (no gain), padded tile stride (no gain), two Q4_K superblocks per lane pass (no gain), 8 rows per workgroup in the GEMV (no gain).

Result (specrt / llama.cpp, tok/s), before and after:

| Model | Prefill 128 | Prefill 1,024 | Decode (128 new) |
|---|---|---|---|
| Qwen3-0.6B bf16 | 0.23 to 0.34 | 0.13 to 0.25 | 0.46 to 0.64 |
| Qwen3-0.6B Q4_K_M | 0.17 to 0.28 | 0.10 to 0.21 | 0.23 to 0.43 |
| Llama-3.2-1B Q4_K_M | 0.20 to 0.33 | 0.15 to 0.27 | 0.23 to 0.41 |
| Qwen2.5-7B Q4_K_M | 0.23 to 0.37 | 0.20 to 0.35 | 0.25 to 0.43 |

The 0.5x milestone is met only for Qwen3-0.6B bf16 decode. Load (7B 11.2 s, peak 9.1 GB, #772) and fusion (#770) were not changed. wgsl tests 23/23 on the Mac.

## Mac speed work 3 (#818 steps)

Same method (3 alternating rounds of base and candidate binaries, `bench --wgpu`, M1 Pro). wgsl tests green after every kept step, greedy text for "The capital of France is" (32 tokens) identical to `llama-completion --temp 0` on Qwen3-0.6B bf16, Qwen3-0.6B Q4_K_M and Qwen2.5-7B Q4_K_M.

### Per-dispatch `linear_coop` against llama.cpp `kernel_mul_mm`

`cargo test --release --features wgpu --test wgsl linear_coop_time_per_dispatch -- --ignored --nocapture` times `linear_coop` with 30 back-to-back dispatches; llama.cpp side is `test-backend-ops perf -o MUL_MAT -b MTL0`. Same shape m=4096, n=512, k=14336:

| | time | TFLOP/s | weight GB/s |
|---|---|---|---|
| llama.cpp f16 | 14.3 ms | 4.19 | 8.2 |
| llama.cpp bf16 | 17.0 ms | 3.53 | 6.9 |
| llama.cpp q4_K | 16.9 ms | 3.56 | 1.9 |
| llama.cpp q6_K | 17.3 ms | 3.48 | 2.2 |
| specrt bf16 | 32.8 ms | 1.83 | 3.6 |
| specrt Q4_K | 35.9 ms | 1.67 | 0.9 |
| specrt Q6_K | 64.1 ms | 0.94 | 0.8 |

Other shapes (n=512): bf16 1.88 TFLOP/s at m=4096/k=1024 and 1.94 at m=18944/k=3584; Q4_K 1.65 to 1.70; Q6_K 0.95 to 0.98. The GEMM is about 2x off llama.cpp, and it is compute-flat with k, so it is not a bandwidth limit (weights stream at under 4 GB/s against about 200 GB/s). Ablations on the same shapes: skipping the multiply-adds gives 2.3 TFLOP/s (bf16), skipping the tile loads after the first gives 2.9 TFLOP/s. So loads cost about 40% and the 8x8 f32 matrix multiplies about 20%, and the kernel is bound by issue and workgroup-memory traffic, not by DRAM. Tried and dropped, each paired A/B on prefill tok/s: register prefetch of tile k+1 (bf16 1,010 to 891, 741 to 674), workgroup-memory double buffer (neutral, within 0.5%), 128x64 output tile with 256 threads (1,012 to 850), f16 tiles with f32 accumulate (the adapter offers 8x8 f16 inputs; 1.9 to 1.6 TFLOP/s, slower), 32x64 tile (1.9 to 1.6). Overlapping the Q4_K dequant with the load therefore has no lever left in this kernel shape: the two overlap experiments gave nothing.

### Kept steps

| Step | Change | Result (tok/s, base to candidate) |
|---|---|---|
| host copy (#772) | `DeviceRows` borrows the mmap (`Cow` parts), the head and embedding are read from the GGUF directly, wgpu allocates the buffer `mapped_at_creation` and copies the mmap straight in (`Backend::alloc_filled`) | Qwen2.5-7B Q4_K_M: peak footprint 9.1 to 5.5 GB; load 10.1-11.4 s to 3.5-7.2 s (median of 5: 5.3 s). The load varies with the page cache; the target of under 3 s was not reached on every run |
| prefill chunk | `PREFILL_CHUNK` 64 to 512 tokens per pass (the GEMM tile is 64 tokens wide and each pass re-read every weight) | prefill 128/1,024: Qwen3-0.6B bf16 1,011/742 to 1,407/1,121; Q4_K_M 844/645 to 1,169/992; Llama-3.2-1B Q4_K_M 565/479 to 690/603. 7B at 512 prompt: 98 to 109, peak 8.1 to 5.9 GB |
| flash_coop | prefill attention on 8x8 cooperative matrices: 4 simdgroups, 32 queries per workgroup, QK and PV as matrix multiplies, lazy rescale of the output (only when a row max rises by more than 6) | prefill 1,024/4,096: bf16 1,120/501 to 1,322/679; Q4_K_M 990/474 to 1,143/632; Llama-1B 601/368 to 676/480; 128 unchanged. Attention was 42% of prefill at 1,024 and 74% at 4,096 before |

### Fusion (#770)

Ceilings measured by skipping the op in decode (Qwen3-0.6B bf16 / Q4_K_M, context 128, tok/s): baseline 81.3/87.5; swiglu skipped 83.0/89.5 (+2%); rope skipped 84.2/90.6 (+3%); rmsnorm skipped 87.3/95.3 (+7 to 9%); all three 93.6/101.7. At 1,024 context the same: +2%, +3%, +6 to 7%. Built the biggest one, rmsnorm into the decode GEMV (the workgroup normalises the row into workgroup memory, then runs the same dot products): decode 81.5 to 75.0 (bf16), 87.8 to 81.3 (Q4_K), slower, because each of the 4-row workgroups recomputes the norm over the whole hidden vector. Dropped. rope plus KV write and swiglu into the down input are bounded by 3% and 2%, so they were not built.

### Result

`mac_vs_llama.sh`, 3 alternating rounds, medians (specrt / llama.cpp tok/s = ratio, pass 2 ratio in brackets):

| Model | Prefill 128 | Prefill 1,024 | Decode (128 new) |
|---|---|---|---|
| Qwen3-0.6B bf16 | 1,412 / 2,941 = 0.48 (0.34) | 1,320 / 2,965 = 0.45 (0.25) | 73.7 / 115.8 = 0.64 (0.64) |
| Qwen3-0.6B Q4_K_M | 1,170 / 3,006 = 0.39 (0.28) | 1,143 / 3,033 = 0.38 (0.21) | 78.8 / 196.5 = 0.40 (0.43) |
| Llama-3.2-1B Q4_K_M | 694 / 1,733 = 0.40 (0.33) | 677 / 1,755 = 0.39 (0.27) | 60.3 / 146.9 = 0.41 (0.41) |
| Qwen2.5-7B Q4_K_M | 107.7 / 273.9 = 0.39 (0.37) | 108.1 / 268.9 = 0.40 (0.35) | 12.1 / 28.0 = 0.43 (0.43) |

7B load 8.2 s in this run (3.5 to 7.2 s in the isolated runs), peak 5.95 GB (was 9.1). The 0.5x milestone is met for Qwen3-0.6B bf16 decode only; bf16 prefill is at 0.45 to 0.48. The remaining gap is the GEMM at about 2x off `kernel_mul_mm` (1.7 to 1.9 against 3.5 to 4.2 TFLOP/s) and decode on the quantised models.

## Mac speed work 4 (#823 steps)

### `kernel_mul_mm` against our `linear_coop`

| | llama.cpp `kernel_mul_mm` | ours before |
|---|---|---|
| output tile | 64 rows x 32 tokens, 128 threads, each simdgroup holds 8 accumulators (4 weight x 2 token matrices) | 64 x 64, 16 accumulators per simdgroup |
| k step | 32 per barrier pair | 16 |
| tile layout | 8x8 blocks stored contiguously (64 elements, stride 8), so each `simdgroup_load` reads one contiguous 128 B block | row-major, stride BK, so each 8x8 load strides across rows |
| tile type | half | f32 |
| barriers per k step | 2 | 2 |
| Q4_K/Q6_K | dequantised 16 values per thread into registers, written to the tile | Q4_K inline, Q6_K through the per-element `dequant_at` |
| output | `simdgroup_store` straight to device memory when the tile is in bounds | always through workgroup memory |
| GEMV (`kernel_mul_mv_q4_K_f32`) | 2 simdgroups x 2 rows each, x held in registers (yl, yh) and reused across the rows, nibbles masked in place (no shifts) and the scale applied after the dot | 4 rows per workgroup, one row per simdgroup |

### `linear_coop` steps (M1 Pro, `linear_coop_time_per_dispatch`, m=4096 n=512 k=14336, TFLOP/s)

| Step | bf16 | Q4_K | Q6_K | Kept |
|---|---|---|---|---|
| base | 1.83 | 1.68 | 0.96 | |
| 8x8 block-contiguous tiles | 1.75 | 1.60 | 0.93 | needed for the next step |
| + k step 32 | 1.88 | 1.73 | 1.00 | yes |
| + 2D swizzle of the workgroup order (L2 reuse) | 1.95 | 1.76 | 1.00 | yes |
| + Q6_K decoded four weights at a time | 1.89 | 1.75 | 1.78 | yes |
| + direct `coopStore` to `y` for full tiles | 1.89 | 1.77 | 1.79 | yes |
| f16 tiles with f16 matrices | 1.47 | 1.39 | 0.79 | no, and error 3e-3 |
| k step 64 | 1.12 | 1.10 | 1.16 | no |
| 64x32 tile (8 accumulators per simdgroup, as `kernel_mul_mm`) | 1.40 | 1.29 | 1.31 | no |
| weight and x tile loads in separate loops (Q4_K quad-style) | 1.77 | 1.65 | 1.68 | no |

k step 64 first gave wrong results: the workgroup array was sized for the 64x64 output tile (4,096 floats) while the two k=64 tiles need 8,192. It is now sized for the larger of the two, and k=64 is correct but slower. The Q4_K path was already decoding four weights at a time; splitting it from the x load made it slower. The 3 TFLOP/s target was not reached: every tile shape and k step tried lands at or below 1.9 TFLOP/s, and f16 matrices are slower, so the half tile and half matrix path that gives `kernel_mul_mm` 4.2 TFLOP/s was not reproduced through wgpu's cooperative matrix. The cause is not established.

### GEMV (`linear_sg_time_per_dispatch`, 7B shapes, GB/s of weights)

| Step | bf16 | Q4_K | Q6_K | Kept |
|---|---|---|---|---|
| base (m=18944 k=3584) | 170 | 51 | 89 | |
| Q4_K after `kernel_mul_mv_q4_K_f32`: lanes split 4 superblocks x 2 halves x 4 groups, scales and mins unpacked from the 12 scale bytes with word masks, nibbles taken without shifts, x summed once for the min term | 170 | 79 | 89 | yes |
| 2 rows per simdgroup so x is loaded once (`N_R0`=2) | 94 | 82 | 67 | no, bf16 and Q6_K lose 30-45% for 8% on Q4_K |
| 2 and 8 rows per workgroup | 170 | 77-79 | 85-89 | no |
| Q6_K scales read as packed words | 170 | 79 | 88 | neutral (89 to 88), kept with the per-quant split |
| one kernel per quant type (`linear_sg_bf16`, `linear_sg_q4k`, `linear_sg_q6k`) | 177 | 77 | 96 | yes |

Step 1 alone made the shared GEMV slower for bf16 (decode 81.7 to 68 tok/s on Qwen3-0.6B): one kernel holding every quant branch has the register pressure of the widest one. A paired A/B on the 0.6B and 1B models caught it, and the kernels are now selected per matrix from its row type (`wgsl::sg_kernel_for`). Q4_K decode on the 0.6B model went 88 to 99 tok/s and Llama-3.2-1B 63 to 81 against the pass 3 binary, paired in the same run.

### Load, 7B (`Dense::load` timing)

Per-layer timing (7B Q4_K_M): `layer_at` 0.9-1.1 s, tensor upload 5.0-6.2 s of a 7.1-9.2 s load. Inside the upload, creating the buffers is 0.6 s, the `mapped_at_creation` copy is 5.4 s (0.9 GB/s: the page-fault cost of first touch on the shared-memory buffer, single threaded) and unmap plus poll 0.65 s. 114 buffers of up to 292 MB; fewer, larger buffers would not help because create and unmap are already 1.3 s together. The copy is now split in 8 MB pieces over up to 8 threads (`Wgpu::alloc_filled`). Load, alternating runs of the two binaries: 7.9, 6.2, 6.9, 6.9 s before and 6.0, 5.6, 6.1, 5.3 s after (median 6.9 to 5.8 s). The target of 3 s is not reached; the remaining cost is the page faults on the mapped allocation plus the GGUF being read from disk on the first touch of each mmap page.

### Failing `--lib` tests

Both pass on Linux master (CPU). On macOS: `a_filesystem_that_refuses_direct_io_falls_back_to_buffered_reads` used `/dev/shm`, which does not exist there (it now skips when the directory is missing); `prompts_stay_bounded_behind_each_other_with_full_decoders` measured ttft with wall clock `Instant`s over `thread::sleep`, and macOS sleeps overshoot a 20 us sleep by several times, so ttft was 2-10 s too high. It now measures ttft on the simulated clock (the sum of the step costs), which is deterministic. 259 passed on the Mac in 3 consecutive runs.

### Checks

wgsl tests 26/26 and `--lib` 259/259 on the Mac, `--lib` 259/259 on Linux, clippy clean with and without the wgpu feature. Greedy text for "The capital of France is" (32 tokens) is identical to `llama-completion --temp 0` on Qwen3-0.6B bf16, Qwen3-0.6B Q4_K_M and Qwen2.5-7B Q4_K_M, checked with the final kernels.

### `mac_vs_llama.sh` result (specrt / llama.cpp tok/s = ratio, 3 alternating rounds, medians)

Quiet Mac: no gcad job in any round, load average 1.1-2.9 during the runs (one 11.6 spike at the start of round 1 from a Simulator process), 5 of the 6 round starts below 3. Pass 3 ratio in brackets.

| Model | Prefill 128 | Prefill 1,024 | Decode (128 new) |
|---|---|---|---|
| Qwen3-0.6B bf16 | 1,408 / 2,963 = 0.475 (0.48) | 1,333 / 2,973 = 0.448 (0.45) | 78.5 / 115.1 = 0.68 (0.64) |
| Qwen3-0.6B Q4_K_M | 1,390 / 3,003 = 0.463 (0.39) | 1,270 / 3,028 = 0.419 (0.38) | 87.8 / 211.0 = 0.416 (0.40) |
| Llama-3.2-1B Q4_K_M | 825 / 1,732 = 0.476 (0.40) | 788 / 1,755 = 0.449 (0.39) | 76.3 / 145.4 = 0.525 (0.41) |
| Qwen2.5-7B Q4_K_M | 129.8 / 273.8 = 0.474 (0.39) | 127.0 / 269.1 = 0.472 (0.40) | 16.45 / 28.0 = 0.587 (0.43) |

7B load 6.1 s, peak footprint 5.96 GB. The 0.5x milestone is now met for decode on bf16, Llama-1B and the 7B, and missed by 0.02-0.08 on prefill everywhere (0.42-0.48) and by 0.08 on Qwen3-0.6B Q4_K_M decode (0.416).

## Mac speed work 5 (#825 steps)

### Why `linear_coop` stopped at 1.9 TFLOP/s

Naga's MSL for `linear_coop` (dumped with the ignored test `dump_linear_coop_wgsl` in `crates/kir/tests/prims.rs`, `KIR_DUMP_COOP=path`, then naga 30 `msl::write_string`) is the same instruction mix as llama.cpp's `kernel_mul_mm`: `coopLoad` becomes `simdgroup_load` and `coopMultiplyAdd` becomes `simdgroup_multiply_accumulate`, with no extra copies. The differences are:

| | naga MSL of `linear_coop` | `kernel_mul_mm` |
|---|---|---|
| tile type | `float`, `simdgroup_float8x8` | `half`, `simdgroup_half8x8` |
| 64x64 tile | works | `half` only (f32 64x64 spills: 394 ms) |
| bounds checks | `naga_div`/`naga_mod` selects, `min()` on every storage index, `loop_bound` 2x u32 counters on every loop, workgroup memory zeroed by thread 0 at entry | none |
| workgroup array | `sh = {}` zero-init | none |

Isolated by hand-writing the same GEMM in MSL (`mb.swift`, m=4096 n=512 k=14336, same 8x8 block-contiguous tiles, 128 threads): f32 tiles 64x32 3.24 TFLOP/s, half tiles 64x32 3.56, half tiles 64x64 3.77. So the Metal compiler is not the limit; f32 loads and the 64x32 shape are worth 10 to 15% over half, and the hardware reaches 3.2 with exactly the structure naga emits. The same naga MSL timed by itself in a Swift harness gave 1.44 TFLOP/s with zero-init and 1.82 without it (the `sh = {}` pass over 16 KB per workgroup is an extra 20%). Removing the naga loop-bound counters and the `naga_div` selects by text edit changed nothing (1.44 to 1.44), and a half-tile WGSL variant (f16 workgroup array and f16 `coop_mat8x8`) was slower (1.25 through naga, as in #823).

The real ceiling is therefore the surrounding code, not the matrix path: the generated tile-load phase (per element index maths, per-quad branches on the quant code, the guarded `x` and `wdata` indexing) costs about as much as the multiplies, where `kernel_mul_mm` does one 16-weight dequant per thread. Ablation on the naga MSL: bf16 only and no branches 2.01 TFLOP/s against 1.82 with all quant branches in the same kernel (register pressure), and dropping the `t < tokens` and `c < cols` guards 2.04. Nothing in the naga lowering is wrong, so there is no upstream bug to file.

### Kept: shader runtime checks off

`wgpu` adds `naga_div`/`naga_mod` selects and `min()` clamps on storage accesses plus loop-bound counters in every loop. The shaders are generated by kir and index in bounds (the wgsl suite passes with 21/21 ignored GPU tests green), so the module is now created with `create_shader_module_trusted(.., ShaderRuntimeChecks::unchecked())`. `SPECRT_CHECKED=1` restores the checks for debugging.

`linear_coop` alone (m=4096 n=512 k=14336, 2 rounds, identical both times): bf16 1.89 to 2.17, Q4_K 1.77 to 1.98, Q6_K 1.80 to 2.02 TFLOP/s. Splitting the two flags: bounds checks off alone 1.95, loop bounding off alone 2.02, both 2.17.

Paired A/B on `bench --wgpu` (3 alternating rounds, quiet Mac, medians, tok/s):

| model | prefill 128 | prefill 1,024 | decode |
|---|---|---|---|
| Qwen3-0.6B bf16 | 1,407 to 1,616 | 1,333 to 1,505 | 87.3 to 94.0 |
| Qwen3-0.6B Q4_K_M | 1,391 to 1,599 | 1,270 to 1,425 | 98.8 to 116.4 |
| Llama-3.2-1B Q4_K_M | 825 to 943 | 788 to 901 | 80.6 to 91.4 |

wgsl tests: 21/21 ignored GPU tests plus the 3 normal ones pass on the Mac (greedy text and KL bounds unchanged).

### Tried and dropped

| Change | Result |
|---|---|
| 72-float block stride (bank conflicts) | 1.58 against 1.89, slower |
| k-major tile layout with transposed loads swapped | 1.83 to 1.43 on Q4_K, slower |
| Q4_K/Q6_K branches removed (bf16 only kernel) | 2.0, so the quant branches cost register pressure only 10% |

### `mac_vs_llama.sh` result (specrt / llama.cpp tok/s, 3 alternating rounds, medians, load average 3 to 12 at round starts, 12 only on round 1)

| Model | Prefill 128 | Prefill 1,024 | Decode (128 new) |
|---|---|---|---|
| Qwen3-0.6B bf16 | 1,617 / 2,950 = 0.548 (0.475) | 1,508 / 2,970 = 0.508 (0.448) | 86.7 / 115.4 = 0.751 (0.68) |
| Qwen3-0.6B Q4_K_M | 1,599 / 3,000 = 0.533 (0.463) | 1,425 / 3,031 = 0.470 (0.419) | 105.6 / 187.8 = 0.562 (0.416) |
| Llama-3.2-1B Q4_K_M | 943 / 1,733 = 0.544 (0.476) | 902 / 1,755 = 0.514 (0.449) | 87.5 / 144.0 = 0.607 (0.525) |
| Qwen2.5-7B Q4_K_M | 147.5 / 273.8 = 0.539 (0.474) | 143.4 / 269.0 = 0.533 (0.472) | 17.65 / 28.0 = 0.630 (0.587) |

Pass 4 ratios in brackets. 0.5x is met on every cell except Qwen3-0.6B Q4_K_M prefill 1,024 (0.470). The decode gain is from unchecked bounds in the GEMV and flash kernels. The 7B load stayed 6.1 s; the Q4_K small-decode dispatch count and the `madvise`/`NoCopy` load idea were not profiled in this pass.

## Mac speed work 6 (#827)

### 1. Workgroup zero-fill

Already off: `PipelineCompilationOptions::zero_initialize_workgroup_memory: false` is passed through wgpu-hal 30 to naga's MSL `Options::zero_initialize_workgroup_memory`, so the `sh = {}` pass is not emitted for specrt kernels. Turning it back on (temporary env switch) on Qwen3-0.6B Q4_K_M, 3 alternating rounds: prefill 128 1,596 to 345 tok/s, prefill 1,024 1,426 to 365 tok/s, decode 116 to 114. The 20% from #825 was measured on a standalone naga MSL without that option. Nothing to file upstream.

### 2. Tile-load code

Rewrote `linear_coop` so the per-quad token row, shared-memory offset, x offset and token guard are computed once before the k loop and read from small private arrays inside it. Measured against the unmodified binary, 2 alternating rounds, Qwen3-0.6B Q4_K_M: prefill 128 1,525 against 1,598, prefill 1,024 1,310 against 1,426 tok/s. Slower (the arrays cost registers), reverted. Metal already hoists the index maths; the load phase is bound by the branchy dequant, as noted in work 5.

### 3. Qwen3-0.6B Q4_K_M prefill 1,024

Unchanged at 0.470. Neither 1 nor 2 moves it.

### 4. Small decode and 7B load

Q4_K_M 0.6B decode dispatches (counted with a trace), per token: 57 rmsnorm, 28 rope, 28 swiglu, 28 flash_decode, 63 `linear_sg*`, 1 argmax, about 205. Skipping kernels (timing only): rmsnorm 116.8 to 128.2 tok/s, rope+swiglu+rmsnorm 140.3. So rmsnorm plus rope plus swiglu are a quarter of the token time, and fusing them into the neighbouring linear or attention kernels is the next lever (not done here).

7B load: `madvise(WILLNEED)` on the GGUF mmap, 3 paired rounds, load seconds without to with: 5.66 to 2.77, 3.82 to 3.31, 3.97 to 2.96. Kept (committed). `newBufferWithBytesNoCopy` over the mmap is not reachable: wgpu 30 exposes no hal path to wrap an external pointer in a Metal buffer on the safe API, and the buffer needs page-aligned `mapped_at_creation` anyway.

### `mac_vs_llama.sh` (3 alternating rounds, medians, load average 2 to 10 at round starts)

| Model | Prefill 128 | Prefill 1,024 | Decode (128 new) |
|---|---|---|---|
| Qwen3-0.6B bf16 | 1,615 / 2,941 = 0.549 | 1,506 / 2,966 = 0.508 | 86.7 / 115.3 = 0.752 |
| Qwen3-0.6B Q4_K_M | 1,597 / 2,998 = 0.533 | 1,425 / 3,032 = 0.470 | 105.5 / 183.4 = 0.575 |
| Llama-3.2-1B Q4_K_M | 944 / 1,733 = 0.545 | 902 / 1,755 = 0.514 | 87.5 / 138.8 = 0.630 |
| Qwen2.5-7B Q4_K_M | 147.5 / 273.9 = 0.539 | 143.4 / 269.2 = 0.533 | 17.60 / 27.97 = 0.629 |

7B load median 5.12 s. 0.7x is not reached on any prefill cell.

### Follow-up pass (#827)

**Per-quant `linear_coop`.** `linear_coop_bf16`, `_q4k` and `_q6k` (plus the generic one for mixed weights) replace the runtime type branch in the tile load; `dense.rs` picks one per weight like `sg_kernel`. Paired A/B on Qwen3-0.6B Q4_K_M, 3 alternating rounds, identical each round: prefill 128 1,599 to 1,719 tok/s, prefill 1,024 1,425 to 1,499 tok/s, decode unchanged. This moves the one-miss cell from 0.470 to 0.494.

**Decode fusion, each tried alone (3 paired rounds, decode tok/s) and dropped:**

| Fusion | Qwen3-0.6B Q4_K_M | Llama-3.2-1B Q4_K_M | Qwen3-0.6B bf16 |
|---|---|---|---|
| rmsnorm into the following GEMV input (each workgroup re-reads x and reduces it, then stages `x * inv * w` in workgroup memory) | 117 to 115 | 91.5 to 83.0 | 94.7 to 88.8 |
| swiglu into the down GEMV input (`silu(g) * u` computed on every x load) | 116.7 to 113.1 | not run | not run |
| rope plus KV write | already one kernel (`rope` writes q, k_cache and v_cache) | | |

Each fused GEMV workgroup (4 rows) redoes the norm or activation over the whole hidden or intermediate vector, so with 3,000 to 9,000 workgroups per dispatch the redundant reads cost more than the saved dispatch. The skipped-kernel upper bound (140 tok/s) is the cost of the dispatches themselves, not of their arithmetic, so the lever is fewer dispatches per token (for example a persistent per-layer kernel), not folding the work into the GEMV. The fusion code was dropped. Prefill rows in those runs are the coop gain above.

**Tests.** `cargo test --release -p inference-bench-specrt --features wgpu --test wgsl -- --include-ignored --test-threads=1` on the Mac: 26 passed, 0 failed (the linear test now dispatches the per-quant coop kernels). `cargo clippy -p inference-bench-specrt --features wgpu --all-targets` clean on the Mac and on Linux; `cargo test -p inference-bench-kir` passes.

**`mac_vs_llama.sh`, 3 alternating rounds, medians (specrt / llama.cpp):**

| Model | Prefill 128 | Prefill 1,024 | Decode |
|---|---|---|---|
| Qwen3-0.6B bf16 | 1,809 / 2,943 = 0.615 | 1,638 / 2,964 = 0.553 | 86.8 / 115.3 = 0.752 |
| Qwen3-0.6B Q4_K_M | 1,719 / 3,000 = 0.573 | 1,497 / 3,030 = 0.494 | 105.3 / 184.4 = 0.571 |
| Llama-3.2-1B Q4_K_M | 1,021 / 1,733 = 0.589 | 974 / 1,755 = 0.555 | 87.5 / 144.7 = 0.604 |
| Qwen2.5-7B Q4_K_M | 161.4 / 274.0 = 0.589 | 156.2 / 269.1 = 0.580 | 17.65 / 28.0 = 0.630 |

7B load median in the full script is 5.13 s (5.49, 4.29, 5.13). The `madvise` win showed in back-to-back runs (3.8 to 3.0 s) but not in this script, where each 7B run follows a llama.cpp run that has evicted the page cache; the cold-cache load is the 4.3 to 5.5 s here. 0.5x now holds on every cell; 0.7x holds on none of the prefill cells and on bf16 decode only.
