# wgpu backend

The wgpu backend runs the same spec and plan as CUDA through WGSL on whatever wgpu reaches: Vulkan, Metal, DX12. It is the Mac's engine today. A direct Metal target is planned in [kir-metal.md](kir-metal.md).

## Selecting it

```toml
[runtime]
backend = "auto"   # "cuda" | "wgpu" | "auto"
```

`auto` uses CUDA when a driver and device initialise and wgpu otherwise. `cargo build --features wgpu` compiles it; without the feature a spec that asks for `wgpu` fails to load and says so.

## Pieces

| Piece | File |
|---|---|
| Backend trait (`alloc`, `alloc_filled`, `upload`, `download`, `compile`, `dispatch`, `flush`) | `crates/specrt/src/backends/mod.rs` |
| Plan runner over `Backend` (`Dense`, `load_stage`) | `crates/specrt/src/backends/dense.rs` |
| wgpu device, pipelines, buffers | `crates/specrt/src/backends/wgpu.rs` |
| WGSL from the spec, kernel choice (`kernels_for`, `sg_kernel_for`, `coop_kernel_for`), the refusal list (`unsupported`) | `crates/specrt/src/wgsl.rs` |
| WGSL emitters | `crates/kir/src/wgsl.rs`, `crates/kir/src/ggml.rs` |

Both backends run `Plan::build` through `executor::run_plan`; the wgpu side maps each `Op` to WGSL kernels. Every linear is a `DeviceRows` row table (block offset and format code: Q4_K, Q5_K, Q6_K, Q8_0, bf16, f16) read through the WGSL `dequant_at` decoder; bf16 safetensors are packed as bf16 rows. The decoders are bit exact against `gguf::dequantize_*`.

## Kernels

| Kernel | Use |
|---|---|
| `embed`, `rmsnorm`, `rope` (qk norm, rotation, linear and llama3 scaling, KV write), `argmax` (greedy and Gumbel), `bias` | glue |
| `linear_sg_bf16`, `_q4k`, `_q6k` | decode GEMV, 4 rows per workgroup, one simdgroup per row; the gate_up variant applies SwiGLU from the two subgroup sums |
| `linear_coop_bf16`, `_q4k`, `_q6k` and a generic one | prefill GEMM on 8x8 f32 cooperative matrices (`Feature::CoopMatrix`, wgpu `EXPERIMENTAL_COOPERATIVE_MATRIX`), 64x64 tile, k step 32, 8x8 block contiguous tiles, 2D swizzled workgroup order, kk loop unrolled, direct store for full tiles |
| `flash_coop` | prefill attention on cooperative matrices, 32 queries per workgroup, K and V tiles of 32 keys in workgroup memory, lazy rescale |
| `flash_decode` | decode attention, 8 simdgroups split the keys |

KV is f32. Runtime settings in `backends/wgpu.rs` and `dense.rs`: shader modules are created with `ShaderRuntimeChecks::unchecked()` (the kernels index in bounds; `SPECRT_CHECKED=1` restores the checks), workgroup memory is not zero filled, bind groups are cached by pipeline and buffer ids with one dynamic offset params buffer written once per submit, decode submits 4 steps per command buffer (`DECODE_STEPS_PER_SUBMIT`; Metal lost the device at 8), prefill runs 512 tokens per pass (`PREFILL_CHUNK`). Weights are copied from the mmap straight into `mapped_at_creation` buffers in 8 MB pieces over up to 8 threads, after `madvise(WILLNEED)` on the GGUF mmap. The device asks for the adapter's own limits and fails the load with the number when a weight or a kernel does not fit. WGSL and pipeline caches live beside the CUDA cache.

## What wgpu refuses

`wgsl::unsupported` names the first missing piece, and the app reports it: IQ quants, FP8 and NVFP4 weights, activation quantisation, quantise at load, delta-net layers, described mixers, hyper connections and input streams, MLP only and non gated MLPs, MTP, described attention kernels, interleaved rope and weightless qk norm, output gated attention, MoE, sliding windows. The batched scheduler and prefix snapshots do not run on wgpu: it serves one sequence through `Engine`. The planner (`DeviceConfig.wgpu`) never gives such layers to a wgpu device.

## Why the current shape

On the M1 Pro, measured against llama.cpp Metal (#768 to #862): the cooperative matrix GEMM reaches 2.74 TFLOP/s against 3.25 for a hand MSL kernel of the same structure. The gap is the tile load phase (3.17 without the dequant) and the f32 64x64 tile; half tiles, double buffering, 64x32 tiles, k step 64 and separate load loops were all slower. Folding rmsnorm or SwiGLU into the GEMV input was slower because each 4 row workgroup redoes the norm over the whole row, which the fusion cost model now predicts ([kir-compiler.md](kir-compiler.md)). naga lowers the matrix phase one to one, so there is no upstream bug.

## Measured state

M1 Pro, `crates/specrt/scripts/mac_vs_llama.sh`, 3 alternating rounds, medians, specrt over llama.cpp tok/s (#829):

| Model | Prefill 128 | Prefill 1,024 | Decode |
|---|---|---|---|
| Qwen3-0.6B bf16 | 0.660 | 0.596 | 0.829 |
| Qwen3-0.6B Q4_K_M | 0.667 | 0.559 | 0.703 |
| Llama-3.2-1B Q4_K_M | 0.722 | 0.661 | 0.749 |
| Qwen2.5-7B Q4_K_M | 0.714 | 0.681 | 0.732 |

At 4,096 tokens `flash_coop` with K and V in workgroup memory is 20 to 40 percent faster than before #835. Logits against CUDA on an 18 token prompt: Qwen3-0.6B bf16 mean KL 1.1e-10, Q4_K_M 7.3e-5, Llama-3.2-1B Q4_K_M 1.1e-4, greedy equal; identical on Vulkan and Metal.

## Tests

`crates/specrt/tests/wgsl.rs` (needs `--features wgpu` and a real adapter, refuses a CPU adapter): every kernel against a CPU reference, decoder exactness, logits and greedy against the CUDA references, `linear_coop_time_per_dispatch` and `linear_sg_time_per_dispatch` timers. On the Mac: `cargo test --release --features wgpu --test wgsl -- --include-ignored --test-threads=1`. `crates/kir/tests/prims.rs` `dump_linear_coop` writes the emitted WGSL (`KIR_DUMP_COOP`, `KIR_DUMP_VARIANT`).
