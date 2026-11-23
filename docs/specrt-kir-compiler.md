# kir as a compiler

Analysis for #1052. No code changes. Everything below was read from the tree at this commit; numbers are quoted from the docs that measured them.

## 1. Fusions written by hand today

Graph level: `graphs/decoder.toml` lowers to a flat `Vec<Op>` (`plan.rs`). Kernel level: one kir builder per op. The only rewrite on the op list is the `tp_ops` peephole in `plan.rs` (`Linear` into `Partial` plus `AllReduce`, merged with a following `Norm` into `AllReduceNorm`). Every other fusion is a kernel somebody wrote and a runtime branch that picks it.

| # | Graph pattern | Kernel (files) | Phase | Measured | Numerics |
|---|---|---|---|---|---|
| 1 | `Norm` then `Linear` | `qkv_proj_norm`, `lin_in_norm`, `lin_ba_norm`, `Gemv::Norm` (`kir/src/gemv.rs`, `specrt/src/codegen.rs`, `runtime.rs:1545`) | decode | in the 27B decode survey ("Decode is already fused", `docs/perf/417-pdl.md`); no standalone norm kernel remains | not documented as bit identical |
| 2 | `Norm`, gate_up `Linear`, SwiGLU | `gate_up_act`, `gate_up_act_f4` (`Gemv::GateUpAct`, `fp4.rs`) | decode | same survey; `gate_up_act_f4` 91 to 78.5 us after the #362 pairing (`specrt-nvfp4.md`) | not documented |
| 3 | `Linear` then residual add into `x` | `Epilogue::Residual` / `EPI_RESIDUAL` (`kir/src/gemm.rs:95`, `cuda/gemm/core.rs`, `prim.rs`), `Gemv { residual }`, `down_proj_f4` | decode and prefill | no separate add kernel in decode (417 survey) | bit identical in f32 (`out + v`) |
| 4 | gate_up `Linear` then SwiGLU into bf16 hi/lo planes | `Epilogue::Act` / `EPI_ACT`, `*_f4b_reduce` applies SwiGLU (`specrt-nvfp4.md:474`) | prefill, batch decode | part of the tile and batch GEMM design | plane rounding differs from an f32 epilogue (`specrt-nvfp4.md:482`) |
| 5 | `Norm` plus bf16 hi/lo plane split | `rms_split` (`kir::prims`, `plan.rs:390`) | prefill, MTP | 2.5% of 27B kernel time at b1, 1.8% at b8 (`specrt-graph-spec.md`) | changes the activation representation by design |
| 6 | qk-norm, rope, KV write (bf16 or fp8) | `qk_norm_rope`, `kv_norm_rope` (`kir/src/cuda/attention/rope.rs`, `norm_rope_head`) | decode, prefill | 0.2% of kernel time | fp8 KV changes numerics (#324, #763) |
| 7 | TP: allreduce, residual add, next norm | `ar_add_rmsnorm_f32` via `tp_ops` | TP decode | #800 fix took tp2 b1 to 169.9 against 117.3 tok/s (#921) | f32 sum order differs per rank count |
| 8 | Activation quant scale | per-token scale applied once in the `gemm_fp8` epilogue (#416), `quant_rows_*` stay separate kernels | prefill | #416 per-call table in `specrt-nvfp4.md:768`; `quant_rows_fp4` 1.9% of prefill, fusion "not built" (`perf/603-prefill-suspects.md:308`) | changes numerics (act_quant) |
| 9 | Dense head: norm, `lm_head` | `lm_head_norm`, `HeadNormFp4` | decode | 29% of the GGUF b1 step before #888 | not documented |
| 10 | Sandwich norms | `clear_branch`, `Linear` into `branch`, `norm_add`, `residual_add` (`plan.rs`, `exec.rs`) | all | `add_rmsnorm` emitted and tested, not launched | f32 |
| 11 | Mac WGSL | SwiGLU into gate_up GEMV (`linear_sg*`) kept; rmsnorm and rope folded into GEMV tried and dropped (`specrt-wgpu.md:401`) | decode | glue kernels gain, GEMV folding lost 3 to 9% | |

Fusions tried and dropped, with the measurement: `add_rmsnorm` (`specrt-graph-spec.md:86`, the GEMM residual epilogue already adds, only the norm's read of `x` is saved), attention reduce into `attention_split` (195 against 216 tok/s, `specrt-gguf.md:153`), rmsnorm into a GEMV on wgpu (above), `quant_rows_fp4` into the preceding op (under the 5% bar). PDL (#417) overlaps kernels instead of fusing them.

I could not read #412, #416, #417's comments, #509 or #770 through the tracker (other project); the rows cite the repo docs that carry their numbers. Rows with "not documented" need a bit-identity check when the compiler reproduces them (slice 3).

Pattern of the list: all of 1 to 7 are the same three shapes. Producer epilogue (3, 4, 8), consumer prologue (1, 2, 5, 9) and a short chain of memory-bound row ops (6, 7, 10). A compiler needs exactly those three rules.

## 2. Is kir transformable or emit-only?

Answer: mostly emit-only. A pass cannot take a kernel and rewrite it.

- `kir::Kernel` is `{ name, target, params, threads, smem_bytes, body: Body }` (`kir/src/lib.rs:90`). `Body` is a closed enum of fixed shapes (`RmsNorm`, `AddRmsNorm`, `NormAdd`, `GatedActivation`, `Pointwise`, `Add`) plus four description bodies. Each builder takes parameters and returns a fully formed kernel; `validate` checks it against the `Target` (smem, threads, features, language).
- GEMM is `KernelInstance { kind: Kind, ... }` (`kir/src/gemm.rs:432`). `Kind` is a closed enum of families (`Tile`, `Fp8Act`, `Fp4Act`, `Batch`, `GgmlBatch`, `Gemv`, `HeadNormFp4`, ...). The epilogue is `enum Epilogue { Store, Residual, Act, Map }` mapped to C++ template constants (`EPI_RESIDUAL`). The decode fusions are `Gemv::Norm` and `Gemv::GateUpAct`, wrappers that name a hand-written device template (`linear_norm<...>`, `gate_up_act_norm<...>`). Adding a new combination means a new enum variant and new template text.

Data (rewritable) today:

| Block | Form | Rewritable how |
|---|---|---|
| `Compose` (`compose.rs`): `TileLoad`, `Epilogue{expr}`, `BlockSum`, `TileStore`, `Gather`, `Lookup`, `Scatter` stages over `Expr` | list of stages with an expression AST (`expr.rs`) | yes: stages can be appended, merged, constant folded; used by `rmsnorm`, `norm_add`, `residual_add`, `swiglu`, logit scale and softcap (`kir/kernels/compose_*.toml`) |
| `Scan` (`scan.rs`), `Rows` (`rows.rs`), `Recurrence` (`recur.rs`) | stage lists, serialisable (`desc.rs`) | yes inside the family; attention and Mamba/KDA variants come from spec TOML |
| `Layout` (`layout.rs`) | CuTe style shape/stride algebra with `compose`, `coalesce` | yes |
| `Tile`, `MmaShape`, `BatchShape`, `Tuning` | parameter structs | yes (this is what tile selection already rewrites) |

Opaque (parameters in, text out):

- GEMM mainloops and epilogues, `Fp8Act`, `Fp4Act`, `Batch*`, `Ggml*`, `Gemv*`: built from `prim::` string functions (`cuda/gemm/prim.rs`, `core.rs`). The `VERBATIM_QUANT` allowlist is the prim library itself; `VERBATIM_MOE` still lists `moe/route.rs`, `moe/gemv.rs`, `moe/prim.rs` with reasons (route and expert row dots keep text because ports change PTX). `VERBATIM_LINEAR` and `VERBATIM_ATTENTION` are empty, but "contains `prim::`" is the bar, not "is data": attention and delta-net bodies are text assembled from prims.
- Fixed `Body` variants (`RmsNorm` etc.): data in name only, a pass can read the width but cannot splice a stage.

So the data model has no common op node: there is no place that says what a kernel reads, writes, in what layout, or how many registers it holds. `Kernel::needs` and `smem_bytes` are the only summary, and they are per kernel, not per stage.

Consequence: the first slice is making kir a real two-level IR, with the Rust builders kept as the reference.

## 3. Two levels, and what sits between them

Exists today: graph IR (decoder.toml nodes, `Op` list, `Buf` names; shapes implicit in the `Spec`), kernel descriptions (`Compose`, `Scan`, ...), emitters (CUDA, WGSL).

Missing:

1. **Typed graph IR.** `Op` carries no signature. Needed per node: tensors read and written (`Buf` plus rows, cols), dtype or plane form (f32, bf16 hi/lo, fp8, fp4 with scales), layout (row major, planes, paged), and a class: `Elementwise`, `RowReduce` (norm), `Gemm`, `Attention`, `Scan`, `Gather`. This is a table over the existing `Op` enum, not a rewrite of the runtime.
2. **Kernel IR with fusable slots.** A heavy kernel (GEMM, attention, scan) becomes `Heavy { family, tile, prologue: Vec<Stage>, epilogue: Vec<Stage> }` where the slots reuse the `compose::Stage` and `expr::Expr` types that already exist. A light kernel is a `Compose`. Fusion is then "append stages to a slot", and the existing builders are the reference implementation: they must emit byte-identical text for an IR that spells today's choice.
3. **Fusion legality data** (the minimum model): per op, `reads`, `writes` (tensor, rows, cols, dtype), `class`, `bytes_moved`, `regs_per_thread`, `smem_bytes`, and for heavy ops the accumulator layout (which thread owns which output element, since an epilogue can only consume what the owner sees). Per Target: registers per SM, smem limit, `Feature` set (already in `Target`).

Copy from other systems, read only:

- **TVM Relax/TIR**: two levels with an explicit lowering (graph `FuseOps` pattern fusion, then `FuseTIR`, then scheduling). Copy the split and the rule "fuse by pattern kind (elementwise, injective, reduce, out-ewise-fusable), not by op name". Avoid: building a general tensor expression language; we only need the three shapes above.
- **XLA**: producer-consumer fusion decided by cost ("does fusing duplicate work or add traffic"), multi-output fusion for siblings. Copy the "never duplicate a producer that is heavy" rule. Avoid the HLO instruction set.
- **MLIR linalg**: tile-and-fuse a consumer into the producer's tile loop. Copy the idea that an epilogue fuses into the GEMM's output tile when the tile owns whole rows or the reduction is tile-local. Avoid the dialect machinery.
- **Inductor**: scheduler that groups nodes into kernels with a "can fuse" check plus a score from saved bytes, and a separate codegen. Copy the scheduler shape: candidate pairs, legality check, score, merge, repeat. Avoid generating the heavy kernels (it autotunes templates too).
- **TensorRT-LLM graph rewriting**: pattern matchers that rewrite to fused plugins (`norm+quant`, `gemm+swiglu`). That is what we have by hand. Copy only the per-pattern test; avoid one plugin per pattern.

## 4. Stages and what is checked between them

```
decoder.toml + Spec ──► Plan (Vec<Op>) ──► [G] graph passes ──► Fused plan
   Fused plan ──► [K] kernel selection (family, tile, splits from Target) ──► Heavy/Compose IR
   Heavy/Compose IR ──► [L] kernel passes (epilogue attach, hoist, tile fit) ──► IR
   IR ──► [E] emit (CUDA, WGSL) ──► source ──► NVRTC
```

| Boundary | Verified by | Fails as |
|---|---|---|
| Plan to G | `validate_plan`: every `Buf` read was written, shapes agree op to op, dtype/plane form matches (`xn_planes` only feeds planes consumers) | compile error naming the node |
| G to K | `validate_fused`: each group is a legal shape (one heavy op, light ops only in its slots, nothing else reads the intermediate), no tensor written and then unused | compile error naming the pass and the group |
| K to L | `Kernel::validate` against `Target` (exists), extended: registers and smem of the whole fused kernel, accumulator ownership of the epilogue | compile error naming pass, kernel, budget |
| L to E | structural hash and golden compare for any IR that spells a hand kernel (today's goldens) | golden diff |
| E | `emit` already refuses a block the Target lacks, with the block named | emit error |

A fusion that would change numerics (`act_quant`, plane rounding) is not a legality question. It is a spec option (config-first) that selects which equivalence class a pass may use: `exact` (bit identical in f32) or `numerics = "act"`. The pass reports which it used.

## 5. Pass list

| # | Pass | Input, output | Rule | Test |
|---|---|---|---|---|
| G1 | canonicalise | Plan: symbolic shapes, dead `Split`, `Clear` before a store | fold constants (`unit_offset`, scales), drop an op whose output no one reads | table test per fixture plan |
| G2 | legalise | Plan, Target: which `Op` has a kernel here | refuse early with the op named (replaces late `unsupported:` strings in `Op::kernels`) | all fixtures on sm_120, sm_80, wgpu |
| G3 | layout assign | planes vs f32 vs fp8 per `Buf` | producers write the form the consumer's family wants (`xn_planes` for tile GEMM, f32 for GEMV) | plan equality against today's `Buf` choice |
| G4 | epilogue attach | `Heavy` + following light ops | fuse consumer C into producer P when: C reads only P's output (plus a vector or scalar), C is `Elementwise`, no other reader of P's output, accumulator owner has the data (per-element, or whole row in tile for reductions), regs/smem fit. Covers residual add, SwiGLU, scale, bias, `clear`+residual to store | pattern fixtures; rediscovery test against rows 3, 4 |
| G5 | prologue attach | light `RowReduce` + following `Heavy` | fuse norm into the GEMV/GEMM input load when the heavy op is memory-bound (decode) and the row fits one pass; refuse when each block would re-read the whole row more than once (the wgpu loss) | rediscovery test against rows 1, 2, 9 |
| G6 | light group | adjacent `RowReduce`/`Elementwise` nodes | merge into one `Compose` (rows 6, 7, 10) if traffic after is lower | rediscovery of `norm_add`, `ar_add_rmsnorm` |
| K1 | split and tile selection | Heavy IR, Target | current `f4b_splits`, `Tile`, `variant_for` formulas, unchanged (#408 found no 5% win from measuring) | goldens |
| L1 | slot fit | fused kernel | recompute regs/smem; if over the Target, split the group again (undo the last attach) | budget test with a small fake Target |
| L2 | hoist | load of weights or scales above the dependent wait | enables PDL style overlap; only when the load does not depend on the previous kernel's output | PTX diff and `pdl` test |
| S1 | stream and capture grouping | fused plan | group launches into the CUDA graph sizes the runtime captures; independent branches (`lin_in` and `lin_ba`) onto one launch or two streams | graph test |

G4, G5, G6 are one engine: candidate pairs from the dataflow graph, the legality check from section 3, the score from section 6, merge, repeat to fixed point.

## 6. Cost model

Decode is memory-bound, so first version is bytes moved plus launches:

```
cost(group) = bytes_read + bytes_written + launch_cost * launches
saving(fuse P,C) = bytes(P.out written) + bytes(C.in read) + launch_cost
penalty = redundant_reads (blocks * row_bytes for a prologue norm) + spill_bytes
fuse iff saving - penalty > 0 and legality holds
```

`launch_cost` is a Target constant expressed in bytes (launch gap times memory bandwidth), measured once per device by the existing probe, not per shape. Validation against section 1: it must (a) select 1, 2, 3, 4, 6, 7, 9 on the 27B and the 4B; (b) reject the dropped ones: the `add_rmsnorm` case saves one read of `x` (hidden x 4 B per row, 2.5% of kernel time) and costs the extra pass, and attention reduce into split (serialises 4 reductions, which the model sees as a penalty if the group has one block per KV head). The wgpu GEMV norm loss is the `redundant_reads` term (blocks x row bytes) and is predicted by the model without measurement.

Needs measured feedback: launch gap on a new Target (one probe, cached in `[runtime]`), and any case where the penalty is occupancy (register pressure changing blocks per SM). Does not: bytes, launches, legality, smem. The tuner result of #408 is why tiles stay formula-based.

## 7. Proof plan

1. Decision equality. For the qwen3 dense fixture and the 27B hybrid (`tiny27b`), the pass pipeline's groups equal the hand fusions of section 1, and the kernel IR emits the same text: kir goldens byte identical, `linear_unchanged` green. No GPU.
2. Logits. `logits`, `linear`, `mtp`, `tiny27b` within today's bounds, then `real27b` on GPU1 under `gpu-run 1`.
3. A fusion nobody wrote. Candidate: `clear` plus residual `Linear` into `branch` plus `norm_add` (sandwich and scaled variants, row 10) becomes `Linear` with store epilogue then `norm_add`; the `clear_branch` launch disappears, saving one launch per branch per layer. Verify with logits (f32, bit identical expected) and the launch count.
4. A new architecture by spec only. A spec that combines existing parameters in a way no hand fusion exists for (sandwich norms plus scaled residual plus MoE-free MLP, or `mixer` with `scaled`) gets its fusions from the passes, with a test that counts launches against the unfused plan.
5. Guard. A test greps pass and cost-model files for architecture names (`no_arch_names` pattern list), and asserts that pass inputs are only the typed signatures and the `Target`.

## 8. Slices (landing order)

1. Typed op signatures plus `validate_plan`, bytes and launch report (`inference-bench kernels --plan`). No behaviour change. Landed (#1053): `crates/specrt/src/signature.rs` (`Signer::signature`, one exhaustive match over `Op`; shapes from `gemm::linear_shape`, cache rows from `state`, described nodes from their kir params and binds), `validate_plan`, `plan_report`; tests in `tests/plan_check.rs`. `inference-bench kernels --spec S --plan [--phase prefill|decode|verify] [--tokens N] [--context C] [--target T]` prints per layer kind the ops, class, bytes read and written, launches and a step total. Bytes are what each op touches once (weights, activations, the cache rows read), not cache-line traffic.
2. Cost model and the legality predicate, with the rediscovery test as a decision-only dry run (prints groups, changes nothing).
3. Kernel IR slots: `Heavy { prologue, epilogue }` over `compose::Stage`, with `Epilogue::{Residual, Act, Map}` re-expressed as stages; goldens byte identical.
4. `Gemv::Norm` and `GateUpAct` rebuilt from the same IR (prologue slot), goldens byte identical; bit identity check for rows marked "not documented".
5. Run the plan from the pass output: replace `tp_ops` and the runtime `fp4` kernel-name branches with the fused plan; identical kernels, `no_arch_names` and grep test.
6. First new fusion: eliminate `clear_branch` (store epilogue) and measure launches and decode tok/s on the 4B via `perfpass.sh`.
7. Spec-only proof and the grep test.
8. L1 slot fit and L2 hoist (PDL), then S1 graph capture grouping.

Out of scope: tuning parameters (formulas stay), a general tensor language, fusing attention or scans into neighbours (the measured losses above).
