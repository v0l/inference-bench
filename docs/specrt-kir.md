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

On `Target::wgpu_baseline()` the same call picks `SharedTree` and 256 threads, and `kir::wgsl::emit_kernel` emits the workgroup tree reduction. On `Target::wgpu_baseline().with_subgroups(32)` it picks `Subgroup` and emits `subgroupAdd` with `enable subgroups;`.

In specrt, `codegen::kir_kernels(spec, &target)` builds `rmsnorm` and `swiglu` from the spec. `Module::generate_kv` appends the emitted CUDA to the rendered templates, and `wgsl::generate` appends the WGSL kernels. The old `rmsnorm` and `swiglu` bodies were removed from `ops.cu` and `templates/wgsl`.

## Slices

1. This one: crate, Target, Layout, rmsnorm and gated activation.
2. NVFP4 and FP8 GEMV and GEMM (after #604).
3. Delta-net and MTP (after #418).
4. Attention (after #324).
5. GGUF and vision.
6. Remove `render()`, the `{{SLOT}}`s and the env knobs.
