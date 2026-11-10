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

Both backends run the same `Plan` (`plan::Plan::build`) through `executor::run_plan`: `Runtime` (CUDA, `runtime/exec.rs`) and the wgpu `Dense` runner each implement `PlanBackend::run_op`. The wgpu side maps each `Op` to WGSL kernels in `wgsl::kernels_for`; there is no hand-written layer loop. CUDA launches stay in `runtime/`, behaviour and speed unchanged (27B NVFP4 prefill 1,024 1,514 vs 1,544 tok/s, 4,096 1,507 vs 1,532, decode batch 1 56.5 vs 57.5, batch 8 333 vs 338 tok/s aggregate, base against this branch, within 2%; real27b 10/10; the non-27B suite passes with `SPECRT_REF_DIR=/tmp/ref387`).

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
