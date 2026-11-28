# kir: kernel IR

`crates/kir` (package `inference-bench-kir`) describes every kernel as typed data and emits CUDA or WGSL text from it. It does not depend on specrt, cudarc or wgpu and knows nothing about `Spec`: specrt maps the spec and runtime options to kir values. It builds and tests on CPU (`cargo test -p inference-bench-kir`). The pass pipeline that groups ops into kernels is in [kir-compiler.md](kir-compiler.md); the planned Metal target is in [kir-metal.md](kir-metal.md).

Design: CUTLASS 3 and CuTe style pieces, compiled like FlashInfer. kir does not pick layouts or pipelining on its own. A builder picks them from the `Target`, so the output is predictable and every choice is in the source.

## Target

`kir::Target` (`crates/kir/src/target.rs`) is what a kernel runs on: `Language` (`Cuda`, `Wgsl`), compute capability, SM count, warp or subgroup size, max threads, shared memory limit, max registers, a `Feature` set (`WarpShuffle`, `CpAsync`, `Mma16816`, `Tma`, `Subgroups`, `ShaderF16`, `Fp8Mma`, `BlockScaledFp4`, `CoopMatrix`) and the runtime facts `graphs` and `pdl`.

specrt fills it from the device: `codegen::cuda_target` reads the CUDA context, and the wgpu backend maps the adapter limits. Tests use fixed targets: `Target::sm_120()`, `sm_80()`, `wgpu_baseline()`, `wgpu_baseline().with_subgroups(32)`, and `Target::named` for the CLI. The target is part of every kernel hash.

## Kernel kinds

| Module | Type | What it covers |
|---|---|---|
| `lib.rs` | `Kernel { name, target, params, threads, smem_bytes, body }` with `Body::{RmsNorm, RmsNormSegment, AddRmsNorm, NormAdd, Pointwise, Add, GatedActivation, Compose, Scan, Rows, Recurrence, Seq}` | light row kernels and the description bodies |
| `gemm.rs` | `KernelInstance { name, target, kind, threads, smem_bytes }`, `Kind::{Tile, Fp8Act, Fp4Act, Batch, GgmlBatch, GgmlGemv, BatchReduce, QuantFp8, QuantFp4, Gemv, HeadNormFp4}` | every weight GEMM and batched GEMV, see [gemm.md](gemm.md) |
| `gemv.rs` | `GemvKernel { name, target, op }` | the single row GEMVs (`linear_w`, `qkv_proj_norm`, `gate_up_act`, `lm_head_norm`, `lin_in_norm`, `lin_ba_norm`, ...) |
| `attention.rs` | `AttentionKernel { family, target, shape }`, `Family::{RopeKv, SplitDecode, FlashLegacy, FlashMma, GroupedDecode}` | rope with qk norm and KV write, split decode, scalar flash, tensor core prefill, grouped batch decode |
| `heavy.rs` | `Heavy { family, problem, tile, prologue, epilogue }` | GEMM and GEMV with fusable stage slots, see [kir-compiler.md](kir-compiler.md) |
| `moe.rs`, `prims.rs`, `rope.rs`, `ggml.rs`, `conv.rs`, `vision.rs`, `mel.rs`, `duplex.rs`, `allreduce.rs`, `weight_quant.rs`, `bandwidth.rs` | per family builders | MoE route and experts, embed and sampling prims, rope tables, ggml decoders, VAE convs, vision, log mel, the duplex head, allreduce, the load time quantiser, the bandwidth probe |

Each builder takes the target and a shape, picks threads, tiles, stages and splits from formulas over the target (`MmaShape::new`, `BatchShape::splits`, `attention::decode_splits`, `gemv::gemv_geometry`, `moe::route_logits_launch`), and validates the result. `Kernel::validate` refuses a non identifier name, threads that are not a power of two or exceed the target, shared memory over the limit, a block whose feature the target lacks (the error names it), a block of the other language, and params that do not match the body.

`kir::layout::Layout` is a CuTe style `(shape, stride)` pair with `offset`, `size`, `cosize`, `coalesce`, `compose` and `expr`, which lowers an index expression to CUDA or WGSL. KV addressing (`attention::prim::kv_addr`) and the gate/up interleave come from it.

## Emitters and the cache key

`kir::cuda::emit_module`, `emit_gemms`, `emit_attention`, `emit_weight_quant` and `emit_bandwidth` return a `kir::KirSource`; `kir::wgsl::emit_kernel`, `emit_gemm` and `emit_op` return WGSL. Only kir can build a `KirSource`, and `Jit::load`, `compile_cubin` and `compile_ptx` take nothing else, so every kernel the engine compiles comes from a kir emitter. Tests that patch text use `KirSource::test_text`, behind kir's `test-text` feature, which only specrt's dev dependency enables. NVRTC is called only in `crates/specrt/src/jit.rs`.

`Kernel::hash()` is FNV-1a over `describe()`, target included; `kir::module_hash` folds a module's kernels in order. specrt keeps it as `Module::kir_hash`, and the JIT cache key includes it.

What stays as `.cu` text under `crates/kir/src/cuda` is the prim library: thin wrappers over one PTX instruction or conversion (`mma.sync`, `cp.async`, `ldmatrix`, `cvt` to e4m3, e2m1 and bf16, warp and block reductions) and the `Dec*` weight decoders the GEMM templates are written against. The shared memory swizzles `swz<CPR>` (`prelude.cu`) and `gemm::prim::swizzle` are still hand written.

Guard tests in `crates/specrt/tests/codegen.rs` (`every_kernel_comes_from_kir`, `the_kernel_text_guards_catch_a_new_offender`, `moe_builders_use_shared_prims_unless_listed`, `launch_geometry_comes_from_kir_builders`) fail on an `include_str!` of `.cu`, `.cuh` or `.wgsl` outside kir, a `__global__` outside the emitters, NVRTC outside `jit.rs`, a `KirSource` built from raw text, library code reading `crates/specrt/tools/`, or a changed allowlist. Hand compiled dev benches live in `crates/specrt/tools/cuda/`.

Goldens are under `crates/kir/tests/golden/` per kernel, family, backend and target (WGSL goldens are also validated with naga). `KIR_BLESS=1 cargo test -p inference-bench-kir` rewrites them; review the diff.

## Kernel descriptions

A kernel description is a TOML file with `kind = "kir-kernel"` (`crates/kir/src/desc.rs`). It names a `family` and its `args`; argument values are literals or shape symbols that `codegen::kir_shape` and `attention_symbols` build from the model config and spec (`hidden`, `inter`, `vocab`, `norm_eps`, `q_dim`, `kv_dim`, `qkv_row`, `mlp_act`, `residual_scale`, `logit_scale`, `logit_softcap`, the attention and rope constants). Unknown keys, families or symbols are load errors naming the kernel, then `Kernel::validate` runs against the live target. The hash is of the resolved kernel, not the file.

- Level 1 families name an existing builder: `rmsnorm`, `add_rmsnorm`, `norm_add`, `rmsnorm_segment`, `add`, `pointwise`, `gated_activation` and the eight attention kernels (`build_attention`). Tests assert the description builds the same kernel and hash as the Rust builder on every target (`crates/kir/tests/desc.rs`, `crates/specrt/tests/kernel_desc.rs`).
- `family = "compose"` is a body of stages: `tile_load`, `epilogue`, `block_sum`, `tile_store`, `gather`, `scatter`, `lookup` over the expression language in `expr.rs` (arithmetic, comparisons, `&&`, `||`, `tanh exp log rsqrt sqrt abs min max floor select softplus`, macros `sigmoid silu gelu_tanh`). rmsnorm, norm_add, swiglu, residual add, logit scale and softcap and `clear_branch` are composed bodies (`crates/kir/kernels/compose_*.toml`) whose text equals the builders byte for byte (`crates/kir/tests/compose.rs`).
- `scan_attention` (`scan.rs`): a query row scan over paged KV with stages `epilogue`, `mask`, `online_softmax` (optional `sink`), `select_from` (keep keys named in an index buffer), split QK and V widths through `[geometry]` segments, and compressed `[blocks]` folded into the same softmax. CUDA only, scalar.
- `rows` (`rows.rs`): `span_rope`, `state_write`, `head_matvec`, `layernorm_segment`, `block_pool`, `head_rms`.
- `recurrence` (`recur.rs`): a per head state matrix in registers with stages `key`, `key_sum`, `scalar`, `value`, `decay`, `update`, `read`, `store`; plus `seq_conv` and `group_norm`. One kernel serves prefill and batched decode.
- `index_topk`: index scores and top k per query row into an i32 buffer.

The embedded descriptions are `crates/kir/kernels/*.toml` and `crates/specrt/specs/*.toml` (the attention, MLA, DSA, Mamba, KDA, DiT, Gemma 3n and other kernels). The spec layers ship them like family specs, the auto updated copy wins (`specrt::kir_kernel::Registry`, `Spec::check` returns `Checked::Kernel`), and the binary falls back to the embedded set. See [specs.md](specs.md).

## Graph specs

`crates/specrt/graphs/*.toml` (`kind = "graph"`) describe a computation: a head, one `[[block]]` per layer kind, a tail, and for decoders `[[mtp]]` and `[[mtp_head]]`. `graphs/decoder.toml` is the decoder graph; the others are the encoder, codec, speech, MeanFlow, LSD, denoiser, VAE and image pipeline graphs.

- A node has `op`, optional `weight`, `quant`, `in`, `out` and `phase` (`prefill`, `decode`, `verify` or `rows`). Blocks carry `layers` and `when` conditions (`plain`, `sandwich`, `tensor_core`, `legacy`, `logit_softcap`, `speculative`, `stock`, joined with `+`) and `mlp = "dense" | "sparse"`.
- `Plan::build` lowers the graph through `graph.rs` into a flat `Vec<Op>` (`plan.rs`, which holds no block kind or op sequence code). Unknown ops, buffers and weights fail at lowering. `tests/graph.rs` checks the lowered plans for every fixture.
- A `linear` node's `quant` is `attn`, `mlp`, `bf16`, `head` or `draft_head`; `Op::Linear` resolves it and calls `gemm::select`.
- Spec nodes: `[[attention.node]]`, `[[attention.projection]]`, `[attention.params]`, `[attention.scratch]`, `[attention.int_scratch]` and `[[mixer.node]]` put described kernels into a layer (`Op::Kernel`), with bindings to buffers, `{ state = name }`, weights, numbers and params; `attention_nodes` in the graph expands them and the stock attention ops are guarded by `when = "stock"`.
- State kinds: `[[state]]` tables (`crates/specrt/src/state.rs`) declare a buffer with `per = token | sequence | layer`, `layers`, `shape` over model symbols, `dtype`, `snapshot`, `prefix_share`, `rollback = positional | checkpoint` and `owner`. `family-base.toml` declares the stock `k`, `v`, `conv` and `ssm` states. The pool, snapshots, prefix cache, swap and the memory fit all read the declared states.

The current spec format is `SPEC_FORMAT` in `crates/specrt/src/spec.rs`.

## Tests

| Test | Checks |
|---|---|
| `crates/kir/tests/*.rs` | goldens, desc equals builder, compose, scan, rows, layouts |
| `crates/specrt/tests/codegen.rs` | the kir only guards |
| `crates/specrt/tests/golden.rs`, `linear_unchanged.rs` | specrt module goldens stay byte identical |
| `crates/specrt/tests/kernel_desc.rs`, `graph.rs`, `variants.rs`, `mamba.rs`, `v4_attention.rs` | descriptions, graph lowering, spec only variants against transformers |
| `crates/specrt/tests/no_arch_names.rs` | no architecture names in engine Rust |

```
inference-bench kernels --spec qwen3.8-27b-nvfp4 --target sm_120
inference-bench kernels --spec qwen3.8-27b-nvfp4 --target wgpu
```
