# specrt kernel IR (kir)

`crates/kir` (package `inference-bench-kir`) describes a kernel as typed data and emits source for a backend. It has no dependency on specrt, cudarc or wgpu, and knows nothing about `Spec`. specrt maps its spec and runtime options to kir values. The crate builds and tests on CPU: `cargo test -p inference-bench-kir`.

Cargo features `cuda` and `wgsl` (both default) enable the emitters. They only produce strings.

## Design reference

Modelled on CUTLASS 3 and CuTe, compiled like FlashInfer. kir does not decide layouts or pipelining by itself (not Triton style): a builder picks them explicitly from the `Target`, so the result is predictable and inspectable.

- **Atoms:** single target instructions (mma.sync, cp.async, ldmatrix, TMA, warp shuffle, subgroup ops, a shared-memory fallback). Each declares the target features it needs. Slice 1 has one atom family: the block sum (`BlockSum`).
- **Collectives:** mainloop (loads plus mma over N stages) and epilogue (scale, bias, activation, residual, store). Slice 1 has the elementwise epilogue only (`GatedActivation`) and the norm reduction (`RmsNorm`).
- **Schedule and launch:** thread count and shared memory on `Kernel`, validated against the target.
- **Layout algebra:** `kir::layout::Layout` is a CuTe style `(shape, stride)` pair, nested and congruent. It supports `offset`, `size`, `cosize`, `coalesce`, `compose`, and `expr`, which lowers a layout to an index expression in CUDA or WGSL. Tiles, swizzles and MMA fragments are meant to be layouts, so index maths is written once for both backends. Slice 1 uses it for the gate/up interleave of the gated activation.
- **Variants and JIT:** one compiled module per (family, config, target), keyed by the kir hash. Families needing a host side split (attention) will follow FlashInfer's plan/run: the plan writes a workspace on the host and run launches fixed shapes, so graph capture works.

## Target

`kir::Target` describes what a kernel runs on: language (`Cuda`, `Wgsl`), compute capability, arch-specific flag, SM count, warp or subgroup size, max threads per block, shared memory limit, a feature set (`WarpShuffle`, `CpAsync`, `Mma16816`, `Tma`, `Subgroups`, `ShaderF16`), and runtime facts (`graphs`, `pdl`).

specrt fills it from the live device: `codegen::cuda_target` reads the CUDA context attributes, and `DeviceLimits::target` maps the wgpu limits. Tests use fixed targets: `Target::sm_120()`, `sm_80()`, `wgpu_baseline()`, `wgpu_baseline().with_subgroups(32)`. Subgroups are not yet detected from the live wgpu adapter, so wgpu targets from the device have none.

The target is an input to the builder, and it is part of every kernel's hash.

## IR

```rust
pub struct Kernel {
    pub name: &'static str,
    pub target: Target,
    pub params: Vec<Param>,
    pub threads: u32,
    pub smem_bytes: u32,
    pub body: Body,
}

pub enum Body {
    RmsNorm { width: u32, eps: f32, sum: BlockSum },
    GatedActivation { width: u32, act: Activation },
}
```

Params carry a scalar type, a role (weight, activation, scratch) and an access. A body declares the roles it expects and `validate` checks them.

Builders (`Kernel::rmsnorm(&target, ...)`, `Kernel::gated_activation(&target, ...)`) choose the blocks and sizes from the target:

| Target | `BlockSum` | Shared memory |
|---|---|---|
| CUDA | `WarpShuffle` (`__shfl_xor_sync`, 32 floats of scratch) | 128 B |
| wgpu with subgroups | `Subgroup` (`subgroupAdd`, per-subgroup scratch) | threads * 4 B |
| wgpu without | `SharedTree` (shared memory tree) | threads * 4 B |

Thread counts are rounded down to a power of two within the target's limit.

## Validation

`Kernel::validate` refuses: a non-identifier name, a thread count that is not a power of two or exceeds the target, shared memory above the target limit, a block whose required feature the target lacks (error names the feature), a block of the wrong language, and params that do not match the body. Both emitters call it first and also refuse a kernel of the other language.

## Hash and JIT cache key

`Kernel::hash()` is FNV-1a over `describe()`, which includes the target. `kir::module_hash` folds the kernels in order. specrt stores it in `Module::kir_hash` and `Jit::load` adds it to the cache key options, so the key changes when the IR or the target changes.

## Emitters

`kir::cuda::emit_module(&[Kernel])` returns CUDA C++ (helpers once, then each kernel). `kir::wgsl::emit_kernel(&Kernel)` returns one WGSL module; `wgsl::bindings` gives the binding kinds the wgpu backend needs. Golden output is in `crates/kir/tests/golden/` per kernel, backend and target. Regenerate with `KIR_BLESS=1 cargo test -p inference-bench-kir` and review the diff.

## Worked example: rmsnorm

```rust
let target = Target::sm_120();
let k = Kernel::rmsnorm(&target, 1024, 1e-6, 1024)?;
let cuda = kir::cuda::emit_module(&[k])?;
```

On `Target::wgpu_baseline()` the same call picks `SharedTree` and 256 threads, and `kir::wgsl::emit_kernel` emits the workgroup tree reduction. On `Target::wgpu_baseline().with_subgroups(32)` it picks `Subgroup` and emits `subgroupAdd`.

In specrt, `codegen::kir_kernels(spec, &target)` builds `rmsnorm` and `swiglu` from the spec. `Module::generate_kv` appends the emitted CUDA to the rendered templates, and `wgsl::generate` appends the WGSL kernels. The old `rmsnorm` and `swiglu` bodies were removed from `ops.cu` and `templates/wgsl`.

## GEMV and GEMM families (slice 2)

`kir::gemm` holds every NVFP4, FP8 and bf16 tile GEMM and batched GEMV as a typed `KernelInstance { name, target, kind, threads, smem_bytes }`. The builders take the target and a `GemmShape { n, k, epi, bias, act }`:

| Builder | Block (`Kind::block`) | Needs |
|---|---|---|
| `KernelInstance::gemm` on CUDA | `TileMma`: `gemm_tc` with a weight decoder (`DecBf16`, `DecQ8`, `DecFp8`, `DecGgml`, `DecFp4`), bf16 `mma.m16n8k16` | `Mma16816`, `CpAsync` |
| `KernelInstance::gemm` on wgpu | `DequantF16`: 16x16 workgroup tile, weights dequantised to f16 in shared memory (bf16, fp8 and nvfp4 decoders) | `ShaderF16` |
| `fp8_act` | `Fp8ActMma`: e4m3 x e4m3 `mma.m16n8k32` | `Fp8Mma` (sm_89+) |
| `fp4_act` | `BlockScaledFp4Mma`: `mma ... kind::mxf4nvf4.block_scale` | `BlockScaledFp4` (sm_12x) |
| `batch`, `batch_reduce` | `BatchRowsMma`, `BatchReduce`: `gemv_f4b` / `gemv_f8b` with split-K | `Mma16816`, `CpAsync` |
| `quant_fp8`, `quant_fp4` | `QuantRowsFp8`, `QuantRowsFp4` | as the GEMM they feed |
| `fp4_gemv`, `fp4_head_norm` | `Fp4Gemv`, `Fp4HeadGemv`: one warp per row | `WarpShuffle`, `CpAsync` |

`MmaShape::new(&target, tile, k, format, split)` picks `bk` and stages (NVFP4 stages from `target.smem_limit`), `BatchShape::for_tokens` the batch variant, and `BatchShape::splits(&target, n, k)` the split-K count from `target.sm_count` and `target.smem_per_sm()`. `Tuning` carries the `[runtime]` table: `tile`, `big_tile`, `fp4_tile`, `fp4_stages`. There are no `SPECRT_*` environment knobs left in codegen: the tiny tile width, `bk` and stage counts that `SPECRT_TINY_WN`, `SPECRT_FP8_BK`, `SPECRT_BK` and `SPECRT_STAGES` used to override are fixed or derived from the target.

`kir::gemm::select(&target, &policy, &request)` replaces specrt's old `kernel_for`: it returns a `Choice` with the path, the kernel name, the quantiser and reduce kernel to launch first or after, and the `KernelInstance` it dispatches. specrt's `codegen::kernel_for(spec, &target, linear, tokens, weight)` maps the spec to a `Policy` and calls it. On a target without `BlockScaledFp4` (sm_80, sm_89) an NVFP4 MLP falls back to `f4w`, which dequantises the e2m1 weights to bf16 in shared memory and runs the bf16 mma; without `Fp8Mma` FP8 weights use `DecFp8` through the same tile GEMM.

`specrt::gemm::gemm_kernels(spec, &target)` builds every instance a module needs, so the module list, the shared memory table and dispatch come from one place; a test checks that every `Choice` the runtime can make on sm_120 and sm_80 is an instance in the module. `kir::cuda::emit_gemms` emits the shared device code once (`crates/kir/src/cuda/gemm.cu`, `fp4.cu`, `fp4_act.cu`, `fp8_act.cu`, `fp4_batch.cu`, `fp8_batch.cu`, `fp4_head.cu`, only the parts the instances use) and one `extern "C"` wrapper per instance; `kir::wgsl::emit_gemm` emits one WGSL module per instance. The emitter refuses a block the target cannot run, naming it, for example `qkv_f8b8: block BatchRowsMma has no Wgsl lowering`. On wgpu, `inference-bench kernels` lists the refused blocks after the families. The `{{MMA_KERNELS}}`, `{{F4W_KERNELS}}`, `{{F4A_SOURCE}}`, `{{F4B_KERNELS}}`, `{{F8B_KERNELS}}`, `{{HEAD_F4W_KERNELS}}`, `{{HEAD_F4_ROWS}}`, `{{F4B_KC}}` and `{{F8B_KC}}` slots are gone, with the template files that held them.

Goldens are in `crates/kir/tests/golden/gemm/`, one per family and target (`*_sm120.cu`, `*_sm80.cu`, `*.wgsl`); the WGSL goldens are also parsed and validated with naga.

```
inference-bench kernels --spec qwen3.8-27b-nvfp4 --target sm_120
inference-bench kernels --spec qwen3.8-27b-nvfp4 --target sm_80
inference-bench kernels --spec qwen3.8-27b-nvfp4 --target wgpu
```

## Attention families (slice 4)

`kir::attention` describes the attention kernels as `AttentionKernel { family, target, shape, threads, smem_bytes }` over an `AttentionShape` (heads, KV heads, head_dim, QKV row, output gate, qk-norm, eps, `RopeShape` with yarn and mrope constants, KV layers, KV planes, fp8 KV). `kir::cuda::emit_attention` emits the shared device code once (`crates/kir/src/cuda/attention/*.cu`) and the kernels of the families asked for.

| Family | Kernels | Needs |
|---|---|---|
| `RopeKvWrite` | `qk_norm_rope`, `kv_norm_rope` (qk-norm, rope, bf16 hi/lo or fp8 KV write) | `WarpShuffle` |
| `SplitDecode` | `attention_split`, `attention_reduce` | `WarpShuffle` |
| `FlashScalar` | `flash_attention` | `WarpShuffle` |
| `FlashMma` | `flash_mma` (prefill) | `Mma16816`, `CpAsync` |
| `GroupedDecodeMma` | `batch_decode_attention`, `batch_decode_combine` (KV read once per sequence, splits grow with length) | `Mma16816`, `CpAsync` |

`MAX_CTX`, `PAGES` and `BD_SPLITS` are no longer source constants. The kernels take `pages` (block-table row stride), `mm_stride` (mrope table stride) and `splits` as launch arguments, so one compiled module serves any context length (#397). The tile constants (`FM_*`, `DM_*`) come from the shape. `specrt::codegen::attention_kernels(spec, kv_planes, target)` builds the set; its hash joins the module hash. Goldens are in `crates/kir/tests/golden/attention/`.

## Slices

1. Crate, Target, Layout, rmsnorm and gated activation.
2. NVFP4 and FP8 GEMV and GEMM (#609, this section).
3. Delta-net and MTP (after #418).
4. Attention (#621, this section).
5. GGUF and vision.
6. Remove `render()`, the `{{SLOT}}`s and the env knobs.
