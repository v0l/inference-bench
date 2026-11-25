# A Metal target for kir

Analysis for #1369, slice 1. No engine code. Everything below was read from the tree at this commit, from the Mac speed work in `docs/specrt-wgpu.md`, and from TensorFold (github.com/ashhart/TensorFold, Apache-2.0) and MLX (github.com/ml-explore/mlx, MIT) read on the web. No code from either project is copied into this repo; what is taken is ideas, re-derived below as kir descriptions.

Today the Mac runs kir's WGSL emitter, wgpu 30 and naga 30, which translates the WGSL to MSL inside the driver call. The plan here adds `Language::Msl`: kir writes MSL itself and a small Metal backend runs it.

## 0. TensorFold, read for this

TensorFold is a Zig server for Apple Silicon and NVIDIA. It has its own checkpoint reader, tokenizer and HTTP server, and hand written kernels. It is not a compiler: `zig/kernels/metal/` holds `.metal` files per model family (`nemotron_*`, `glm_*`, `kimi/`, `flashnext/`, `qwen3_5/`) beside shared `core/`, `decode/`, `prefill/` and `ops/`, and `sources.zig` embeds them. That is the shape #1136 removed from this repo, so none of it is a model for kir's structure.

What is worth taking:

| Piece | Where | What it does | Taken as |
|---|---|---|---|
| Indirect command buffers | `zig/src/metal/icb.zig` | dispatches encoded once on the host (`concurrentDispatchThreadgroups`, `setBarrier` per dependent dispatch), replayed by the GPU; re-encoding one command changes only its next replay | runtime piece R2 |
| Residency sets | `zig/src/metal/residency.zig` | `MTLResidencySet` (macOS 15+) holds weights and KV pages wired for a queue; refuses past 70% of RAM minus a margin | runtime piece R3, and the memory admission gate |
| Keep warm | `zig/src/metal/keepalive.zig` | while idle, a tick commits a tiny blit that touches every residency set so the GPU and the sets stay hot (default 900 s) | runtime piece R4, measured before it is kept |
| M5 tensor ops | `zig/kernels/metal/nax.h` | 16x32x16 `matmul2d` from `MetalPerformancePrimitives` on M5; the same fragment API over 8x8 `simdgroup_matrix` before M5 (`TF_SIMD_FRAGS`) | `Feature::NeuralAccel`, a later slice |
| MLX affine weights | `docs/quantization.md` | 2, 3, 4, 5, 6, 8 bit, groups of 32, 64, 128, scales and biases in bf16, f16 or f32, per module overrides, read as stored | section 3 |
| Qualification gates | `docs/quantization.md`, `CONTRIBUTING.md` | packed value checks, one row against batched, drafted against serial, prefix resume, concurrent streams, memory admission, all before a speed claim | section 5 |

## 1. What lowers to WGSL today, and what naga does to it

The Mac path (`crates/specrt/src/wgsl.rs::generate`, run by `backends/dense.rs` on `backends/wgpu.rs`) ships these kir kernels. Everything else is refused by `wgsl::unsupported` or by `kir::wgsl::emit_kernel` (`RmsNormSegment`, `AddRmsNorm`, `NormAdd`, `Pointwise`, `Add`, `Scan`, `Rows`, `Recurrence`, `Seq`, every GEMV and attention family, MoE): no delta-net, MoE, sliding window, MTP, FP8 or NVFP4 on the Mac.

Translation measured on Linux with naga 30.0.2's MSL backend (the version wgpu-hal 30.0.2 links), Qwen3-0.6B shapes (head_dim 128, hidden 1024). "Engine options" is what `backends/wgpu.rs` asks for since #825 and #827: unchecked bounds, no workgroup zero fill, no loop bounding, no integer division checks. The WGSL for `linear_coop` comes from `KIR_DUMP_COOP=f.wgsl KIR_DUMP_VARIANT=q4k cargo test -p inference-bench-kir --test prims dump_linear_coop -- --ignored`, the rest from the same `kir::wgsl::emit_*` calls `generate` makes.

| kir piece (WGSL kernel) | WGSL lines | naga MSL lines, wgpu defaults | engine options | `naga_div`/`naga_mod` on defaults | `loop_bound` on defaults | simdgroup ops |
|---|---|---|---|---|---|---|
| `Body::RmsNorm` (`rmsnorm`, subgroup sum) | 34 | 145 | 119 | 0 | 12 | 1 |
| `Body::GatedActivation` (`swiglu`) | 19 | 58 | 40 | 4 | 0 | 0 |
| `rope` (rope, qk norm, KV write) | 112 | 326 | 284 | 0 | 16 | 0 |
| `embed` (ggml rows) | 140 | 259 | 236 | 17 | 4 | 0 |
| `argmax` (greedy and gumbel) | 75 | 214 | 184 | 0 | 8 | 0 |
| `flash` (prefill, scalar) | 98 | 337 | 273 | 7 | 28 | 0 |
| `flash_decode` | 115 | 413 | 339 | 7 | 32 | 0 |
| `flash_coop` (prefill, 8x8 matrices) | 380 | 1072 | 865 | 9 | 48 | 272 |
| `linear_sg_bf16` (GEMV) | 189 | 371 | 333 | 18 | 4 | 1 |
| `linear_sg_q4k` | 222 | 422 | 372 | 23 | 4 | 1 |
| `linear_sg_q6k` | 205 | 420 | 374 | 24 | 8 | 1 |
| `linear_coop_bf16` (GEMM, 8x8 matrices) | 361 | 790 | 661 | 67 | 16 | 201 |
| `linear_coop_q4k` | 382 | 821 | 690 | 72 | 16 | 201 |
| `linear_coop_q6k` | 369 | 802 | 673 | 74 | 16 | 201 |

With the engine options the guards are gone (0 `naga_div`, 0 `loop_bound`, no clamps), which is the #825 result: bounds checks off 1.89 to 2.17 TFLOP/s on `linear_coop`. What is left in the engine MSL is structural. Every helper takes the storage arrays and a `constant _mslBufferSizes&` as extra arguments (31 to 34 mentions in each linear kernel), every `coopLoad` becomes a call to a `NagaCooperativeLoad` wrapper (98 per `linear_coop`, 167 in `flash_coop`), and every byte or half read is a `u32` load plus shift and mask, because WGSL storage arrays are `u32` only (`wbyte`, `whalf`, `wf16` in `kir/src/ggml.rs`).

Kernel time, from the Mac docs (M1 Pro, quiet machine, paired runs):

| Kernel | Ours today | Reference | Source |
|---|---|---|---|
| `linear_coop`, m=4096 n=512 k=14336 | bf16 2.81, Q4_K 2.74, Q6_K 2.63 TFLOP/s | `kernel_mul_mm` f16 4.19, bf16 3.53, q4_K 3.56, q6_K 3.48; hand MSL, same tiles, no dequant: f32 64x32 3.25, half 64x32 3.56, half 64x64 3.77 | wgpu work 3 and 8 |
| `linear_coop` ceilings | no dequant 3.17, no dequant and no x load 3.70 | | wgpu work 8, table 2 |
| `linear_sg` GEMV, 7B shapes | bf16 177, Q4_K 77 to 82, Q6_K 96 GB/s of weights | about 200 GB/s DRAM | wgpu work 4 |
| `flash_coop` | 39% of prefill at 1,024 and 79% at 4,096 on Qwen3-0.6B before #835, then 20 to 40% faster at 4,096 | | wgpu work 8 and 9 |
| `rmsnorm`, `rope`, `swiglu` decode glue | skipping all three: 116.8 to 140.3 tok/s (before #828) | | wgpu work 6 |
| whole model, `mac_vs_llama.sh` | prefill 0.56 to 0.72, decode 0.70 to 0.83 of llama.cpp | | wgpu work 8 |

What a direct emitter changes, piece by piece:

| Lever | Through WGSL and naga | Direct MSL |
|---|---|---|
| `Language::Msl` | none: the Target says `Wgsl` and the device is found at run time | kir knows it is emitting for an Apple GPU and can refuse, size and specialise for it |
| `simdgroup_matrix` (`Feature::SimdgroupMatrix`) | `coop_mat8x8<f32>` through `wgpu_cooperative_matrix`, an experimental wgpu feature (`EXPERIMENTAL_COOPERATIVE_MATRIX`), only 8x8 f32 detected (`backends/wgpu.rs`) | `simdgroup_float8x8` and `simdgroup_half8x8` directly, with `simdgroup_load` from device or threadgroup memory at constant offsets, and half A and B with a float accumulator |
| Threadgroup memory, 32 KB | `max_compute_workgroup_storage_size` from wgpu; `var<workgroup>` arrays of one type each | read from `maxThreadgroupMemoryLength`; one raw block reused as half tiles, float partials and staging (`flash_coop` already reuses `opart` by hand) |
| SIMD width 32 | `subgroup_min_size` reports 4 to 64 on Metal, so `backends/wgpu.rs` guesses 32 on Apple | 32 on every Apple GPU, checked per pipeline with `threadExecutionWidth` |
| No runtime checks | off by an option on each pipeline; `zero_initialize_workgroup_memory` off by another | nothing to switch off |
| `half` tiles | `array<f16>` needs `enable f16`; #829 measured parity with f32 tiles (2.74 and 2.74) | the same, so not a lever on its own; it is one when it lets a 64x64 tile fit, as in the 3.77 hand kernel |
| Byte and half loads | `u32` arrays only: `wbyte` is a load, shift and mask; Q4_K read its 4 header words 5 times until #829 | `device const uchar*`, `ushort`, `half4`, and block structs; a Q4_K block is read the way it is stored |
| Function constants | WGSL `override` exists but wgpu bakes it per pipeline; we pass shapes as uniforms (`p.a.x`, `p.a.y`) and loop on them | `[[function_constant(i)]]` per weight shape and alignment (MLX does this with `align_M`, `align_N`, `align_K`), so one library serves every shape and the guards fold away when the shape is aligned |
| Argument buffers | not reachable | available; needed only past 31 buffer arguments or for ICB, see section 4 |
| Grid size | 65,535 workgroups per dimension, hence `gid.x + gid.y * 8388608u` in `swiglu` | `uint3` grid up to 2^32 - 1 per dimension |
| Runtime | wgpu encoder, bind groups, validation, `mapped_at_creation` copies | ICB, residency sets, `newBufferWithBytesNoCopy` over the mmap, concurrent dispatch (section 4) |

What a direct emitter does not change. The #825 and #829 work found nothing wrong in naga's lowering: the matrix phase is one to one (8 loads and 16 multiply adds per k step), and the loss to the hand kernel is the tile load phase (3.17 without the dequant, 2.74 with it) and the tile shape. The direct emitter attacks the load phase with typed loads and a `kernel_mul_mm` style 16 weight dequant per thread into half tiles, which WGSL cannot write cheaply. Whether that clears 5% is measured in slice M5, and if it does not, prefill keeps the WGSL GEMM.

## 2. The emitter plan

Same shape as `docs/specrt-kir-compiler.md`: typed pieces, each lowering refused with a reason when it cannot be emitted, each landed with goldens, nothing named after an architecture.

### Target

`Target` gains no field; the existing ones describe an Apple GPU:

```
Target::apple(family, cores, smem, features)
  language            Msl
  compute_capability  (family, 0)    7 = M1, 8 = M2, 9 = M3 and M4, 10 = M5 (MTLGPUFamilyApple<N>)
  sm_count            GPU cores (16 on the M1 Pro)
  warp_size           32
  max_threads         1024 (a pipeline can allow fewer: checked at compile, see validate)
  smem_limit          maxThreadgroupMemoryLength (32768 on Apple7 to Apple9)
  features            Subgroups | ShaderF16 | SimdgroupMatrix, plus NeuralAccel on M5
  graphs, pdl         false (ICB is a runtime choice, not a codegen one)
  max_regs            u32::MAX (Metal reports spills per pipeline, not a limit kir can use)
```

`Feature` gets `SimdgroupMatrix` and `NeuralAccel`, appended after `CoopMatrix` so every existing bit position and every `describe()` string stays the same. That keeps every CUDA and WGSL golden and `kir_hash` byte identical. `Target::named` gets `apple7` (M1 class, the Mac we have) and `apple10` for goldens. The live Target is built from the device (`supportsFamily`, `maxThreadgroupMemoryLength`, core count from IORegistry or the config), never from a model name. `layout::Dialect` gets `Msl`; `expr.rs`'s function table gets a third column (`rsqrt` is `metal::rsqrt`, `exp` is `metal::exp`, and so on). `BlockSum::Subgroup` lowers to `simd_sum` on MSL, and `BlockSum::language` becomes `BlockSum::fits(Language)`.

### What lowers first

In order, each a slice in section 6:

| Slice | Pieces | Why this order |
|---|---|---|
| M1 | `Body::RmsNorm`, `GatedActivation`, `Compose` (every `kernels/compose_*.toml`), `Pointwise`, `Add`, `NormAdd`, `AddRmsNorm`, `RmsNormSegment` | the description bodies; already lowered for two dialects, so MSL is a third column, and they give the backend something to run |
| M2 | prims: `embed`, `rope` (through `rope::inv_freq_fn`), `argmax` and gumbel sampling (`prims.rs`), `bias`, ggml row decoders as typed byte loads (`ggml.rs`) | everything a dense decode step needs besides the GEMV and attention |
| M3 | GEMV: `heavy::Family::Gemv { wf }` and `GgmlGemv` for bf16, Q4_K, Q6_K, Q8_0, through `Heavy` with the norm prologue and the act epilogue | 63 of the 177 decode dispatches per token, and most of the weight bytes; the slot model gives the Mac the decode fusions from the same IR, judged by the existing wgpu cost constants |
| M4 | decode attention (`attention::Family::SplitDecode` lowering: simdgroups split the keys, partials merged in threadgroup memory, as `flash_decode` does) and the rope plus KV write (`Family::RopeKv`) | completes decode; the first end to end run on the Metal backend |
| M5 | prefill: `gemm::Mainloop::Simdgroup { tile, half }` for `Kind::Tile` (replaces `linear_coop`), then `attention::Family::FlashSimdgroup` (replaces `flash_coop`) | the TFLOP/s claim; needs M3's decoders for the tile load |
| later | `Scan`, `Rows`, `Recurrence` bodies (attention nodes, delta-net, Mamba), MoE, FP8 and NVFP4 decode in registers | what `wgsl::unsupported` refuses today; each its own issue when a Mac model needs it |
| later | `Feature::NeuralAccel`: the `Simdgroup` mainloop's multiply as `matmul2d` 16x32x16 on M5 | no M5 here to measure on |

The tile, splits and rows per block come from formulas over the Target (smem, warp size, cores) like the CUDA ones (`gemm::Tile`, `gemv::gemv_geometry`), never from per model tuning. The CUDA text prims (`__shfl_xor_sync`, `cp.async`, `mma.sync`) are not translated: each family gets an MSL prim library of the same thin kind (`simd_sum`, `simd_shuffle_xor`, `threadgroup_barrier`, `simdgroup_load`), and `Kind::languages()` and `attention::Family::needs()` list which families have one.

### What `validate` must refuse

Each with the kernel and the number in the message, as the CUDA side does:

- threadgroup memory over `smem_limit` (the existing check), counted over every `threadgroup` array the emitter declares;
- threads over 1024, or not a multiple of 32 in a kernel that uses simdgroup functions;
- a simdgroup matrix tile that is not a multiple of 8 on both sides, or `SimdgroupMatrix` missing;
- more than 31 buffer arguments (Metal's argument table), until a family needs argument buffers;
- any family whose `languages()` or `needs()` lacks Msl: `Mma16816`, `CpAsync`, `Tma`, `Fp8Mma`, `BlockScaledFp4` are CUDA features and stay refused by name;
- `NeuralAccel` on anything below the M5 family;
- 64 bit atomics (the allreduce and some MoE counters) on Apple7 and Apple8.

One check is not static: Metal may lower `maxTotalThreadsPerThreadgroup` below the requested threads when a kernel uses many registers. The backend reads it after `newComputePipelineStateWithFunction` and fails naming the kernel, the threads asked and the threads allowed, the same way the CUDA path reports an occupancy failure.

### Where the MSL runs

Two ways to get MSL onto the GPU were read:

- wgpu passthrough (`Features::PASSTHROUGH_SHADERS`, `Device::create_shader_module_passthrough` with `msl` source, wgpu-hal 30 compiles it with `newLibraryWithSource`). wgpu-types documents it as "you shouldn't expect this to work with bindings except on SPIR-V": the buffer indices wgpu assigns are not ours to choose. It also leaves ICB, residency sets and NoCopy buffers out of reach.
- A Metal backend, `crates/specrt/src/backends/metal.rs`, behind the existing `Backend` trait (`alloc`, `upload`, `alloc_filled`, `download`, `compile`, `dispatch`, `submit`), on `objc2-metal` 0.3 (already in `Cargo.lock` through wgpu-hal), macOS only, selected by `[runtime] backend = "metal"` in the spec or server config with `wgpu` the default until the A/B says otherwise.

Decision: the Metal backend. It is about 400 lines against `backends/wgpu.rs`'s 415, it reuses `dense.rs` unchanged, and every runtime piece in section 4 needs it. Compiled libraries are cached under `[runtime] cache_dir` with `MTLBinaryArchive`, keyed by the source hash as the CUDA JIT cache is.

## 3. MLX affine groups as kir data

MLX stores a quantised linear as three tensors (`mlx/backend/metal/kernels/quantized.h`, read on GitHub):

- `weight`: `u32 [n, k * bits / 32]`. Each row is a little endian bitstream: value i of the row sits at bits `[i * bits, i * bits + bits)`. For 2, 4 and 8 bits that is `32 / bits` values per word; for 3 and 5 bits it is 8 values in 3 or 5 bytes, for 6 bits 4 values in 3 bytes (MLX's `get_pack_factor` and `get_bytes_per_pack` say the same thing in pieces).
- `scales`, `biases`: `[n, k / group]` in bf16, f16 or f32, the same dtype for both.
- the value is `w = scale * q + bias`, q unsigned.

`config.json` carries `quantization: { group_size, bits, mode }` (read before the older `quantization_config`), plus per module entries: a mapping uses its own defaults (`{"bits": 8}` means group 64 even when the global group is 128), `true` inherits the global settings and `false` leaves that module unquantised.

As kir data, next to `dequant::BlockDequant`:

```
pub struct Affine {
    pub bits: u8,           2, 3, 4, 5, 6, 8
    pub group: u16,         32, 64, 128
    pub scale: Scalar,      Bf16 | F16 | F32
    pub bias: Option<Scalar>,  Some for MLX; None for scale only formats
    pub order: Pack,        LsbStream (MLX, GPTQ) | Awq (nibbles 0 2 4 6 1 3 5 7 in a word)
}
```

`validate` refuses bits outside the list, a group that is not 32, 64 or 128, a `k` that is not a multiple of the group or of 32, and a scale dtype different from the bias dtype. `Affine::row_bytes(k)`, `group_bytes` and `value(row, i)` (CPU reference, bit exact) are the data every emitter reads.

How it composes:

- `dequant::BlockDequant` today is `elem` (E2m1, E4m3) times `scale` (E8m0, E4m3), a scale only group. `Affine` with `bias: None` and an integer element covers Q4_0 and Q8_0 style groups; with `bias: Some` it covers MLX, and GPTQ and AWQ, whose `scale * (q - zero)` is `scale * q + (-scale * zero)`. That is #1200's int4 description: one data type, three checkpoint spellings.
- The ggml K quants (`ggml.rs`) are two level: a super block of 256 with an f16 `d` and `dmin` over 8 groups of 6 bit scales and mins. They stay as they are; a later change can express them as `Affine` with a quantised scale, but nothing needs it now.
- The GEMV and tile loaders read the description, not a format name. For the GEMV the emitter factors the bias out of the dot, `scale * sum(q * x) + bias * sum(x)` per group, the trick MLX's `qdot` and our `linear_sg_q4k` (`sy`) both use; this changes float rounding against dequantise then dot, so the gate is bit exact dequant plus a tolerance on the dot.
- The loader maps it through the family TOML `[quantization]` rules (config first), per module, on both CUDA and Metal; fused stacks (qkv, gate_up) only when bits, group, width and dtypes agree, otherwise the projections load and run separately.

## 4. Runtime pieces

Counts for Qwen3-0.6B Q4_K_M decode, one token, from the #828 trace: 57 `rmsnorm`, 28 `rope`, 28 `flash_decode`, 63 `linear_sg*` (gate_up carries the SwiGLU), 1 `argmax`: about 177 dispatches. llama.cpp launches about 525 nodes for the same token at 8.7 ms, so the count alone is not the gap; host encode cost and idle bubbles are.

| # | Piece | What it removes | Needs | Expected size |
|---|---|---|---|---|
| R1 | One command buffer per step, committed without waiting; the next step is encoded while this one runs, reading the token id from the device `tokens` buffer that `argmax` writes | the host wait per token before the next encode | Metal backend | the host encode time per token (177 dispatches); #828's bind group cache alone was worth 24%, about 8 us per dispatch |
| R2 | Indirect command buffer for the decode step: the 177 dispatches encoded once per batch shape and slot set, replayed with `executeCommandsInBuffer`; per step only the params buffer (position, lengths) is written | 177 x (pipeline, 4 to 6 buffers, dispatch) encoder calls, about 1,200, become 1 | pipelines made with `supportIndirectCommandBuffers` | host cost only; GPU dispatches stay 177 |
| R2b | Concurrent dispatch with barriers placed from the typed op signatures (`signature.rs` reads and writes), as TensorFold's `setBarrier` and llama.cpp's `(concurrent)` marks | serial waits between independent dispatches | R2 or a concurrent encoder | small at batch 1 on a dense decoder (almost every op reads the previous one); larger with `lin_in` and `lin_ba` style siblings |
| R3 | Weights as `newBufferWithBytesNoCopy` over the page aligned GGUF or safetensors mmap, held in an `MTLResidencySet` | the 5.4 s first touch copy measured in wgpu work 4 (2.5 s with `madvise`) | macOS 15+ (the Mac is on 26) | load time; 0 dispatches |
| R4 | Keep warm: an idle tick that commits a tiny blit touching the residency sets, as `keepalive.zig` does | the clock ramp and residency loss on the first request after idle | R3 | time to first token after idle; 0 dispatches. `[runtime] keep_warm_s`, default 0 until measured |

Fewer GPU dispatches come from the compiler, not the runtime: the M3 slots give `Gemv::Norm` and the act epilogue to the Mac when the cost model accepts them (it rejected the norm fold for wgpu, 8.4 MB of rereads against a 1 MB saving), and `store_epilogue` removes `clear_branch` where a spec has branches.

## 5. Test plan

On Linux, no Mac:

1. MSL goldens: `crates/kir/tests/golden/msl/*.metal`, one per family and per weight format, written with the existing `golden()` helper and `KIR_BLESS=1`. Every existing CUDA and WGSL golden stays byte identical (`emit`, `gemm`, `prims`, `linear`, `attention`) and so do the specrt `golden`, `source_dump` and `linear_unchanged` suites.
2. Static checks on the emitted text: one `kernel void <name>(` per kernel, buffer indices 0 to n contiguous and under 31, threadgroup bytes summed from the declarations equal to `smem_bytes` and under `smem_limit`.
3. Refusals: a test per `validate` rule above, with the message checked.
4. Guards: `no_arch_names`, `codegen` (`every_kernel_comes_from_kir`: no `.metal` file under `crates/`, no MSL text outside `kir/src/msl`), `spec`.

On the Mac (`ssh mac`, M1 Pro):

5. Compile check: `crates/specrt/tests/msl.rs` (macOS, `metal` feature) builds every kernel of every fixture spec the Mac supports with `newLibraryWithSource` and `newComputePipelineStateWithFunction`, and checks `maxTotalThreadsPerThreadgroup` and `staticThreadgroupMemoryLength` against the kernel. That needs only the Metal framework. `xcrun metal -Werror` as a lint is optional: on 25 Nov the Mac answered `cannot execute tool 'metal' due to missing Metal Toolchain; use: xcodebuild -downloadComponent MetalToolchain`.
6. Numerics, the `tests/wgsl.rs` set run on the Metal backend: dequant bit exact against the CPU for Q4_K, Q5_K, Q6_K, Q8_0, bf16 and every MLX affine bits and group; rmsnorm, swiglu, rope with qk norm and KV write, causal attention over several query blocks, linear against a CPU matmul at 1 and many tokens, argmax and seeded gumbel; then the same kernel on the Metal and wgpu backends on one Mac, identical for elementwise ops and within the reduction tolerance otherwise.
7. Model: greedy text equal to CUDA and to llama.cpp on Qwen3-0.6B bf16, Qwen3-0.6B Q4_K_M and Qwen2.5-7B Q4_K_M (the gate wgpu work 3 to 8 used), and KL inside the existing `compare_with_cuda` bounds.

TensorFold's gates, mapped to tests that exist (CUDA) and run on the Metal backend once it serves them:

| Gate | Our test | On the Mac |
|---|---|---|
| packed value checks | item 6, plus `Affine::value` against a numpy unpack of a real MLX tensor | from M2 and the affine slice |
| one row against batched | `serving.rs` `tiny_mixed_length_step_decode_matches_single_sequence_greedy`, `stepped_spec_matches_single` | when `dense.rs` decodes more than one row per step |
| drafted against serial | `mtp.rs` `scheduled_speculation_matches_single_sequence_greedy_and_counts_drafts`, `eager_speculation_matches_plain_greedy_alone_and_in_a_batch` | when speculation runs on the Mac |
| prefix resume | `prefix.rs` `prefill_a_then_b_equals_prefill_a_snapshot_then_b_from_the_snapshot`, `serving.rs` `park_resume_is_bit_identical` | M4 (snapshot of the f32 caches) |
| concurrent streams | `serving.rs` `the_serving_core_streams_concurrent_jobs_and_cancels_within_a_step`, `four_requests_share_a_pool_of_one_and_a_half_max_model_lens_and_match_single_sequence` | with the scheduler on the Mac |
| memory admission | `serving.rs` `auto_sized_pool_admits_and_decodes_more_slots_than_the_old_resident_cap` | admission against `recommendedMaxWorkingSetSize`, refused before allocation, tested with a lowered limit |

Speed, the method of `docs/specrt-wgpu.md`: `mac_vs_llama.sh`, 3 alternating rounds, medians, quiet Mac (load average and gcad job recorded). First a control, wgpu against wgpu, then Metal against wgpu in the same binary (`[runtime] backend`), and the per kernel timers (`linear_coop_time_per_dispatch`, `linear_sg_time_per_dispatch`) for the MSL kernels. A change is kept at 5% or better on its target and no worse than the control's spread elsewhere; llama.cpp Metal stays the reference column.

## 6. Follow-up slices

Filed as issues (goal g4, high model), chained by `depends_on`. No per architecture Rust in any of them; `no_arch_names`, `codegen`, `golden`, `spec` and `linear_unchanged` stay green in each.

| Slice | Issue | Depends on |
|---|---|---|
| M0 Target, `Language::Msl`, `Dialect::Msl`, features, `validate` refusals | #1371 | |
| M1 MSL emitter for the description bodies, goldens | #1372 | #1371 |
| Metal backend behind `Backend`, compile and numerics tests on the Mac | #1373 | #1372 |
| M2 prims and ggml row decoders | #1374 | #1372 |
| M3 GEMV family through `Heavy`, function constants for shapes | #1375 | #1374, #1373 |
| M4 decode attention and rope KV write, first end to end decode, A/B | #1376 | #1375 |
| M5 simdgroup GEMM and flash prefill, the TFLOP/s A/B | #1379 | #1376 |
| MLX affine description, loader, CUDA and MSL GEMV decode | #1377 | #1375 |
| R1 and R2: pipelined step and indirect command buffer | #1380 | #1376 |
| R3 and R4: NoCopy weights, residency sets, keep warm | #1378 | #1373 |
| `NeuralAccel` on M5 (needs an M5 Mac) | #1381 | #1379 |

#869 (find why emitted WGSL `linear_coop` loses to hand MSL) is answered by section 1 and replaced by M5 (#1379).
