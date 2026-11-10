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
- Moving the CUDA runtime behind the `Backend` trait.

## Tests

`tests/wgsl.rs` (needs `--features wgpu` and a real adapter, `#[ignore]` otherwise): each kernel against a CPU reference, Q8_0 and Q4_K exactness, then model logits and greedy tokens against the stored CUDA/llama.cpp references. Lavapipe does not count; the tests refuse a CPU adapter.

## Results

Filled in below from the runs on the RTX (Vulkan) and the MacBook (Metal).
