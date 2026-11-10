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
| Dense executor walking the plan | `backends/dense.rs` | yes, generic over `Backend` |
| wgpu device, pipelines, buffers | `backends/wgpu.rs` | feature `wgpu` |
| CUDA runtime | `runtime.rs`, `jit.rs` | CUDA only |

The CUDA runtime is not yet behind the `Backend` trait. It is 1,900 lines of cudarc launches, graphs, batching, MTP and delta-net, and #462 is rewriting its GEMM templates. Moving it behind the trait changes nothing a wgpu user sees, so it is the follow-up issue "Move runtime.rs behind Backend" and must be gated by the logits, greedy and real27b tests (27B NVFP4 within 2%). Until then CUDA behaviour and speed are untouched.

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
- Moving the CUDA runtime behind the `Backend` trait (#483 follow-up).

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
| RTX PRO 6000 Blackwell (Vulkan) | 1.45 s (705 tok/s) | 19.8 tok/s |
| Apple M1 Pro (Metal) | 14.9 s (69 tok/s) | 18.8 tok/s |

Decode is dominated by dispatch count and the scalar kernels, not bandwidth. Speed work (subgroups, tiled GEMM, fused ops, command buffer reuse) is the follow-up. Metal lost the device when eight decode steps were recorded into one command buffer, so decode submits four steps at a time (`DECODE_STEPS_PER_SUBMIT`).
