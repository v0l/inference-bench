# kir as a compiler

Analysis for #1052. No code changes. Everything below was read from the tree at this commit; numbers are quoted from the docs that measured them.

## 1. Fusions written by hand today

Graph level: `graphs/decoder.toml` lowers to a flat `Vec<Op>` (`plan.rs`). Kernel level: one kir builder per op. The only rewrite on the op list is the `tp_ops` peephole in `plan.rs` (`Linear` into `Partial` plus `AllReduce`, merged with a following `Norm` into `AllReduceNorm`). Every other fusion is a kernel somebody wrote and a runtime branch that picks it.

| # | Graph pattern | Kernel (files) | Phase | Measured | Numerics |
|---|---|---|---|---|---|
| 1 | `Norm` then `Linear` | `qkv_proj_norm`, `lin_in_norm`, `lin_ba_norm`, `Gemv::Norm` (`kir/src/gemv.rs`, `specrt/src/codegen.rs`, `runtime.rs:1545`) | decode | in the 27B decode survey ("Decode is already fused", `docs/perf/417-pdl.md`); no standalone norm kernel remains | bit identical to `rmsnorm` then the plain GEMV at the GEMV's block width; also equal to the runtime's 1024 thread norm on the tested inputs (slice 4, `prologue_bits.rs`) |
| 2 | `Norm`, gate_up `Linear`, SwiGLU | `gate_up_act`, `gate_up_act_f4` (`Gemv::GateUpAct`, `fp4.rs`) | decode | same survey; `gate_up_act_f4` 91 to 78.5 us after the #362 pairing (`specrt-nvfp4.md`) | bit identical to `rmsnorm`, the gate_up GEMV and `swiglu` (slice 4) |
| 3 | `Linear` then residual add into `x` | `Epilogue::Residual` / `EPI_RESIDUAL` (`kir/src/gemm.rs:95`, `cuda/gemm/core.rs`, `prim.rs`), `Gemv { residual }`, `down_proj_f4` | decode and prefill | no separate add kernel in decode (417 survey) | bit identical in f32 (`out + v`) |
| 4 | gate_up `Linear` then SwiGLU into bf16 hi/lo planes | `Epilogue::Act` / `EPI_ACT`, `*_f4b_reduce` applies SwiGLU (`specrt-nvfp4.md:474`) | prefill, batch decode | part of the tile and batch GEMM design | plane rounding differs from an f32 epilogue (`specrt-nvfp4.md:482`) |
| 5 | `Norm` plus bf16 hi/lo plane split | `rms_split` (`kir::prims`, `plan.rs:390`) | prefill, MTP | 2.5% of 27B kernel time at b1, 1.8% at b8 (`specrt-graph-spec.md`) | changes the activation representation by design |
| 6 | qk-norm, rope, KV write (bf16 or fp8) | `qk_norm_rope`, `kv_norm_rope` (`kir/src/cuda/attention/rope.rs`, `norm_rope_head`) | decode, prefill | 0.2% of kernel time | fp8 KV changes numerics (#324, #763) |
| 7 | TP: allreduce, residual add, next norm | `ar_add_rmsnorm_f32` via `tp_ops` | TP decode | #800 fix took tp2 b1 to 169.9 against 117.3 tok/s (#921) | f32 sum order differs per rank count |
| 8 | Activation quant scale | per-token scale applied once in the `gemm_fp8` epilogue (#416), `quant_rows_*` stay separate kernels | prefill | #416 per-call table in `specrt-nvfp4.md:768`; `quant_rows_fp4` 1.9% of prefill, fusion "not built" (`perf/603-prefill-suspects.md:308`) | changes numerics (act_quant) |
| 9 | Dense head: norm, `lm_head` | `lm_head_norm`, `HeadNormFp4` | decode | 29% of the GGUF b1 step before #888 | bit identical for `lm_head_norm` (bf16, GGUF) and `lm_head_norm_f4` (slice 4) |
| 10 | Sandwich norms | `clear_branch`, `Linear` into `branch`, `norm_add`, `residual_add` (`plan.rs`, `exec.rs`) | all | `add_rmsnorm` emitted and tested, not launched | f32 |
| 11 | Mac WGSL | SwiGLU into gate_up GEMV (`linear_sg*`) kept; rmsnorm and rope folded into GEMV tried and dropped (`specrt-wgpu.md:401`) | decode | glue kernels gain, GEMV folding lost 3 to 9% | |

Fusions tried and dropped, with the measurement: `add_rmsnorm` (`specrt-graph-spec.md:86`, the GEMM residual epilogue already adds, only the norm's read of `x` is saved), attention reduce into `attention_split` (195 against 216 tok/s, `specrt-gguf.md:153`), rmsnorm into a GEMV on wgpu (above), `quant_rows_fp4` into the preceding op (under the 5% bar). PDL (#417) overlaps kernels instead of fusing them.

I could not read #412, #416, #417's comments, #509 or #770 through the tracker (other project); the rows cite the repo docs that carry their numbers. Rows with "not documented" needed a bit-identity check when the compiler reproduced them; slice 4 did it for rows 1, 2 and 9.

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

- GEMM mainloops and epilogues, `Fp8Act`, `Fp4Act`, `Batch*`, `Ggml*`, `Gemv*`: built from `prim::` string functions (`cuda/gemm/prim.rs`, `core.rs`). The `VERBATIM_QUANT` allowlist is the prim library itself; `VERBATIM_MOE` lists only `moe/prim.rs` (re-exports and launch wrappers) since #1136 built route and the expert row dots from shared prims. `VERBATIM_LINEAR` and `VERBATIM_ATTENTION` are empty, but "contains `prim::`" is the bar, not "is data": attention and delta-net bodies are text assembled from prims.
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

Target list from the static comparison against FlashInfer, Triton and vLLM (#1148, `docs/perf/kernel-static.md`, no GPU). Where kir output is structurally behind, ranked by time at stake, each with the pass that would close it:

1. Grouped decode attention at head_dim 256: 255 registers, 184 to 792 B spill, 64 threads per SM against FlashInfer's 768 to 1152. Register pressure aware decode tiling with spill as a refusal (#1160).
2. NVFP4 prefill GEMM: same `OMMA` as CUTLASS, but cp.async from every warp and 2 stages against TMA, mbarriers, 3 stages and warp specialization. TMA mainloop lowering for `Feature::Tma` targets, stages from the smem budget (#1162).
3. `moe_route_logits` stages tokens in static shared memory over the 48 KB limit at hidden 3072 and up: GLM-4.5-Air does not build. Static smem check in the budget pass for the route kernels (#1159).
4. Norm, add norm, swiglu and rope load 32 bits per instruction; every competitor loads 128. Vectorise pass over `Compose` and row loops (#1161).
5. No sm_75 plan for any Qwen3.5 or 27B spec (scalar flash tile 84 KB, chunk GDN needs mma) (#1163).
6. `--split-compile` makes the spill of kernels at 255 registers depend on unrelated kernels in the module (#1164).

The prefill flash attention and the bf16 GEMM main loop match the competitors structurally. Timed runs: #1158.

## 8. Slices (landing order)

1. Typed op signatures plus `validate_plan`, bytes and launch report (`inference-bench kernels --plan`). No behaviour change. Landed (#1053): `crates/specrt/src/signature.rs` (`Signer::signature`, one exhaustive match over `Op`; shapes from `gemm::linear_shape`, cache rows from `state`, described nodes from their kir params and binds), `validate_plan`, `plan_report`; tests in `tests/plan_check.rs`. `inference-bench kernels --spec S --plan [--phase prefill|decode|verify] [--tokens N] [--context C] [--target T]` prints per layer kind the ops, class, bytes read and written, launches and a step total. Bytes are what each op touches once (weights, activations, the cache rows read), not cache-line traffic.
2. Cost model and the legality predicate, with the rediscovery test as a decision-only dry run (prints groups, changes nothing). Landed (#1054): `crates/specrt/src/fusion.rs`, `inference-bench kernels --spec S --plan --fusion [--phase P] [--tokens N] [--target T]`, tests in `tests/fusion_plan.rs`.
   - Units. Each plan op lowers to units over the slice 1 signatures. A `Linear` becomes its mainloop (block geometry from `gemm::select`: GEMV rows per block, tile `bn`, batch rows and splits), a split reduce when splits > 1, an act quant row op for the f8a/f4a paths, and its epilogue (residual add, gated act) as an elementwise unit. Rope ops become qk norm, rope, KV write; decode attention becomes rope, split (one block per KV head per split) and combine. Everything else is one unit with the launch count from `plan_report`.
   - Engine. Candidates are dataflow edges between groups. Epilogue and light merges run to a fixed point first, then prologues; the best net saving wins each round. A prologue duplicates the producer into every heavy consumer (one norm into `lin_in` and `lin_ba`).
   - Cost. Saving is the launch cost plus the intermediate's write and reads. Penalties: prologue rereads, `(col_blocks - 1) x row bytes x reread_rate`, less what the consumer's own reads hide (`reread_overlap`); an epilogue on a split mainloop pays `(splits + 1) x round_trip_bytes` for the last-block ticket. `Costs::for_target` holds the constants: CUDA 2 us launch, 0.8 us round trip, 1,600 GB/s, reread rate 0.25 with full overlap (L2 serves the rereads); wgpu 5 us launch, 200 GB/s, rate 1 with no overlap (the #827 measurement).
   - Legality. Single launch units only; one unit with block geometry per group; no device barrier (allreduce) in a consumer; attention and scan units fuse only inside their own op; rows must match; no other group on a path between the two; the intermediate has no reader outside the group and is not live after it; an epilogue row op needs the block to own the whole row (`Owner::Tile { cols }` against the row width), and a split owner refuses row ops; staged prologue rows plus the mainloop smem fit `smem_limit`; registers fit `max_regs`.
   - What it finds on qwen3-0.6b and tiny27b at sm_120: rows 1, 2, 3, 4, 6, 7 and 9 (decode norm into qkv, gate_up, lin_in plus lin_ba, lm_head and the MTP head; residual and act epilogues on GEMV, tile and split reduce; qk norm, rope and KV write; allreduce plus the next norm). It keeps apart `add_rmsnorm` (the residual already fused into its producer scores higher, and the norm needs a whole row a GEMV block does not own), attention reduce into the split (ticket penalty 3.84 MB against a 3.2 MB launch saving) and the wgpu norm fold (8.4 MB of rereads against a 1 MB saving). Prefill also groups each norm with the act quant that follows it (`rms_split` plus `quant_rows_*`). That is the `quant_rows_fp4` into the preceding op fusion section 1 lists as not built because it was under the 5% bar, not because it lost; the quant does not fold into the GEMM because the reread penalty exceeds the saving at 512 tokens. Constants are estimates; slice 3 onwards checks them against nsys.
3. Kernel IR slots: `Heavy { prologue, epilogue }` over `compose::Stage`, with `Epilogue::{Residual, Act, Map}` re-expressed as stages; goldens byte identical. Landed (#1055): `crates/kir/src/heavy.rs`, tests in `crates/kir/tests/gemm.rs` and `crates/specrt/tests/fusion_plan.rs`.
   - IR. `Heavy { name, family, problem: { n, k, act }, tile, prologue, epilogue }`. `Family` is the mainloop with its own parameters (tile MMA, fp8 act, fp4 act, batch rows, ggml batch and gemv, split reduce); the quant rows and the fp4 GEMV and head kernels are not lifted, they are slice 4.
   - Epilogue stages. The accumulator is the name `acc`; `out` is the f32 output, `planes` the bf16 hi/lo output, `bias` the bias vector. Store is `store out acc`; bias adds `load b bias`, `v = acc + b`; Residual is `load r out`, `y = acc + r` (`v + r` with bias), `store out y`; Act loads the gate and up columns with the `(n/2,2):(2,1)` layout and stores `gated(g, u)` to `planes`; Map stores `mapped(acc)` to `planes`. `gated` and `mapped` are expression functions that lower to the module's `gated_act` and `mapped_act` on CUDA and are refused on WGSL.
   - Lowering. `Heavy::lower` matches the epilogue against the stage lists the emitters spell (`heavy::spelled`) and calls the existing builder, so the source is the reference text. A list no emitter spells, or any prologue (no emitter yet), is refused. `Heavy::attach` appends stages to a slot: an epilogue attach drops the store and renames `acc` to the stored value, so `store out acc` plus the residual stages becomes the Residual list.
   - Budget. `Heavy::budget` builds the bare mainloop and checks threads, shared memory (mainloop plus the staged prologue rows) and registers (`acc_regs` of the accumulator tile plus one per stage value) against `Target::max_regs` and `smem_limit`. Every failure is `pass P: kernel K needs N what, the target has L`. `gemm::gemm_kernels` lifts every GEMM it builds and emits it through `Heavy::compile` under the pass name `kernel_select`; the fusion cost model takes `max_regs` from the `Target`.
   - Proof. Every GEMM in the kir golden set lifts and lowers to an equal `KernelInstance` and the byte identical golden; every GEMM of qwen3-0.6b, tiny27b and the 27B fixture does the same; `linear_unchanged` and the specrt goldens are unchanged.
4. `Gemv::Norm` and `GateUpAct` rebuilt from the same IR (prologue slot), goldens byte identical; bit identity check for rows marked "not documented". Landed (#1056): `crates/kir/src/heavy.rs`, `crates/specrt/src/gemm.rs` (`normed`, passes `prologue_attach` and `epilogue_attach`), `codegen::gemv_kernels`; tests in `crates/kir/tests/gemm.rs`, `tests/fusion_plan.rs`, `tests/prologue_bits.rs`.
   - IR. Two more families: `Gemv { wf }` (the `linear_w` row GEMV for every decoder) and `Fp4Gemv` (the NVFP4 GEMV and head). `Heavy::lift_gemv` lifts a `GemvKernel`, and `Heavy::lift` now lifts `Kind::Gemv` and `Kind::HeadNormFp4`; `lower_gemv` and `compile_gemv` lower to a `GemvKernel`. `Bf16Rows` and `Tiled` have no slots and stay unlifted.
   - Prologue stages. `heavy::norm_prologue` is the rmsnorm list from `compose_rmsnorm.toml` over the names `x` (the residual row), `nw` (norm weight) and `xs` (the staged row in shared memory): `load v x`, `total = sum(v * v)`, `inv = rsqrt(total / width + eps)`, `load xv x`, `load w nw`, `y = w * (xv * inv)`, `store xs y`. `spelled_prologue` accepts it or none; the emitters spell it as `load_normed` inside `linear_norm`, `gate_up_act_norm` and `head_norm_f4`.
   - Epilogue on a GEMV. The gated act on a GEMV stores to `out`, not `planes` (`epilogue_into(Act, .., OUT)`). `lower_gemv` maps no prologue with Store or Residual to `Linear`, the norm with Store to `Norm`, the norm with Act to `GateUpAct`; `Fp4Gemv` maps the norm with Act to `gate_up_act_f4`, the norm with Store to `lm_head_norm_f4` and Residual to `down_proj_f4`. Any other pairing is refused with both lists named.
   - Passes. `gemm::normed` builds the bare GEMV, attaches `norm_prologue` under `prologue_attach`, then for the gated MLP the act stages under `epilogue_attach`; every attach rechecks the budget. `codegen::gemv_kernels` builds `qkv_proj_norm`, `gate_up_act`, `lm_head_norm`, `lin_in_norm` and `lin_ba_norm` this way, `gemm_kernels` builds `gate_up_act_f4` and `lm_head_norm_f4`. A GEMV budget is 256 threads, the staged row `k x 4` bytes of shared memory and one register per stage value; a target with less shared memory fails as `pass prologue_attach: kernel qkv_proj_norm needs ...`. `HeavyError::Gemm` keeps a `GemmError` so the WGSL path still lists unsupported blocks instead of failing.
   - Proof. Module source and `kir_hash` byte identical for all 270 fixture and spec modules on sm_120 and sm_80 (`tests/source_dump.rs`), NVRTC PTX identical for the 21 27B, 4B and 0.6B modules with one `libnvrtc` (13.2 and 13.5 give different PTX for the same source, so pin `LD_LIBRARY_PATH`); kir goldens, `linear_unchanged` and the specrt goldens unchanged. On GPU1 each fused kernel equals `rmsnorm`, then `linear_w` on the same weights, then `swiglu` for the act, bit for bit: 40 cases on tiny-qwen3.5 (bf16, fp8, nvfp4, nvfp4 head), tiny27b, the tiny Q8_0 GGUF and Qwen3.5-4B NVFP4 with both heads, against both a norm at the GEMV's width (256 or 512 threads) and the runtime's 1024 thread `rmsnorm`. A one ulp change to the reference's norm weight makes it differ, so the check sees single bits. Equality at the same width is structural: `load_normed` and the compose `rmsnorm` use the same strided partial sums and the same warp then block reduction. Against the 1024 thread norm it held on these inputs, but when `HIDDEN` exceeds the block the partial sums group differently, so a last bit difference in `inv` is possible on other inputs.
5. Run the plan from the pass output: replace `tp_ops` and the runtime `fp4` kernel-name branches with the fused plan; identical kernels, `no_arch_names` and grep test. Landed (#1057): `crates/specrt/src/compile.rs`, `runtime/single.rs`, `gemm::gemv`; tests in `tests/compile_plan.rs` and `tests/no_arch_names.rs`.
   - Passes. `compile::compile(spec, plan, target, opts)` runs `partition` (row parallel linears write `Partial` and are followed by an `AllReduce`, the MoE combine too), then for the single row family `gemv_layout` (planes buffers become their f32 row form, a `Split` of x becomes an alias of x, decode attention and the delta rule step become `StepAttention` and `GdnStep`), then `fuse`: the slice 2 engine (`fusion::fuse_ops`) over the section at one row, with a kernel library predicate. A group is legal only when the library has one kernel for its units: the allreduce with the next planes norm (`AllReduceNorm`, when `[runtime] fuse`), or a row norm with a following whole GEMV (`Op::Gemv { norm }`, with or without the act epilogue) whose `gemm::gemv` choice exists for every layer the section covers. Groups inside one op (residual and act epilogues, rope chain) stay that op. A linear left alone in the single row family becomes `Op::Gemv` with no norm, a gated one followed by `Swiglu`. Single rank planes plans are returned as built.
   - Runtime. `Plans::compile` builds prefill, decode and verify through the passes (tensor parallel ranks pass the graph shards), plus `single`, the one row decode plan `Runtime::step` runs through `executor::run_plan` with the `Single` backend. `step_traced`'s hand arrangement (`mlp_decode`, `gdn_decode`, `attention_out_decode`, `linear_norm`), `gemv_cfg` and its `_f4` name branches are gone; the kernel name, threads, rows per block and bias slot come from `gemm::gemv`.
   - Guard. `pass_and_cost_files_name_no_architecture` greps `compile.rs`, `fusion.rs`, `signature.rs` and `kir/src/heavy.rs` for architecture names, and the pass files for spec layer kinds or device types.
   - Proof. For every fixture and spec the TP passes equal the old `tp_ops` rewrite (all phases, fuse on and off) and the single row plan launches exactly the kernels `step_traced` launched (qwen3 dense, tiny27b and every spec the step runs); every planned kernel is in the module. On GPU1 logits, decode and greedy hashes from `logitdump` equal master for Qwen3.5-4B NVFP4, qwen3-0.6b bf16, q8 and Q4_K_M GGUF, tiny27b, tiny-olmo2 and tiny-granite; `tiny27b`, `linear`, `generate`, `logits`, `prologue_bits`, `head_quant`, `gguf_logits`, the tiny `families` and tiny `gemma3` suites pass.
6. First new fusion: eliminate `clear_branch` (store epilogue) and measure launches and decode tok/s on the 4B via `perfpass.sh`. Landed (#1058) behind `[runtime] store_epilogue`, off by default: the speed gain is under the 5% bar. Files: `kir/src/heavy.rs` (`Heavy::fill`), `specrt/src/fusion.rs` (`Rule::Fill`), `compile.rs`, `codegen::store_gemvs`, `gemm::gemv_into`, `Op::GemvStore`; tests in `kir/tests/gemm.rs`, `tests/fusion_plan.rs`, `tests/compile_plan.rs`.
   - Rule. `Rule::Fill`: a group of elementwise units that read nothing (a constant write) merges into the anchored group that reads its output, when only light epilogue units read it, nothing else reads it and nothing sits between them. In the dry run it pairs `clear` with `o` plus `o residual` and `clear 2` with `down` plus `down residual` on tiny-olmo2 and tiny-granite; the norm add stays apart (the GEMV block owns 8 of the row's columns).
   - Kernel pass. `Heavy::fill(t, "epilogue_fill", out, 0.0)` replaces each load of `out` in the epilogue by the constant, folds `v + 0`, and drops the load: the Residual list becomes the Store list, which the emitters spell as `linear_w<.., RESIDUAL = false>`. Any other constant is refused (`no gemv emitter spells ...`). `codegen::store_gemvs` builds `o_proj_store` and `down_proj_store` this way when the option is on and the spec has a branch.
   - Plan. `compile::fuse` realises the group as `Op::GemvStore` (the GEMV writes `branch` instead of adding into it). NVFP4 down stays `clear_branch` plus `down_proj_f4` (no fp4 store emitter). The batched and prefill plans are unchanged.
   - Launches (`inference-bench kernels --spec S --plan --compiled`, the single stream plan the runtime launches): tiny-olmo2 58 to 50, tiny-granite 47 to 39, OLMo-2-0425-1B 226 to 194, granite-3.3-2b-base 443 to 363. The 4B (qwen3_5) has no branch, so its plan and logits are unchanged.
   - Numerics. `logitdump` prompt, decode and greedy hashes equal master for tiny-olmo2, tiny-granite, OLMo-2 1B and granite-3.3 2B with the option on and off, and for tiny-gemma3 and Qwen3.5-4B NVFP4.
   - Speed (GPU1, RTX PRO 6000 Max-Q, `perfpass.sh --ab`, 256 new tokens, median of per-round ratios). Control master against master: 0.999 (OLMo-2), 0.998 (granite). Store against master at a 128 token prompt: decode 1.012 (505 to 511 tok/s) and 1.015 (246 to 250 tok/s), 5 rounds, IQR 0.001; at 1024: 1.012 and 1.014, 3 rounds. The branch with the option off against master: 1.001 and 0.998. The saved time is about 0.7 us per removed launch (OLMo-2: 23 us over 32 launches; granite: 65 us over 80), so the step stays dominated by the weight reads. Prefill does not use the single stream plan, but granite at a 128 token prompt read 0.971 with the option on (5 rounds, IQR 0.002) and 0.997 at 1024; OLMo-2 read 1.021 and 1.023. The module only gains the two store kernels, so that 128 token number is unexplained.
7. Spec-only proof and the grep test. Landed (#1059): three fixtures combine existing parameters in ways no hand fusion was written for, as spec TOMLs only (`tests/fixtures/tiny-sandwich-scaled.toml`: post-norm sandwich, full width qk norm and `residual_scale`; `tiny-sandwich-pre-qknorm.toml`: pre plus sandwich norms with full width qk norm; `tiny-scaled-qknorm.toml`: granite multipliers with full width qk norm). Weights and the torch reference come from `scripts/tiny_families.py` (a plain torch forward, `check-mix` proves it equals transformers on tiny-olmo2 and tiny-granite to 0.0). No Rust names any of them.
   - Launch accounting. `compile::single_traced` returns the library-backed fusion trace the `fuse` pass ran (`kernels --plan --compiled --fusion` prints it). The library predicate now answers `Fit::{Kernel, Part, None}`: a group the library runs only in parts (a gated linear with its act epilogue but no norm, which the GEMV library runs as `gate_up_proj` plus `swiglu`) stays legal so the norm prologue can still join it, but counts the launches of its parts. Before, the engine counted that group as one launch and predicted 4 launches fewer than the plan for tiny-olmo2 (54 against 58) and 16 for OLMo-2 1B (210 against 226). `the_compiled_step_launches_what_the_cost_model_predicts` checks every single step spec: compiled launches equal the prediction and are below the unfused plan.
   - Scaled sandwich. `residual_scale` with sandwich norms used to be dropped silently (the graph's `scaled` condition is `!sandwich`). It is folded at load into the branch norm weights (`post_attention_layernorm` and `post_feedforward_layernorm`), since `x + s * w * norm(b)` is `norm_add` with weight `s * w`; no new op and no new kernel.
   - Launches (single step decode, sm_120): unfused plan, then compiled with `store_epilogue` off and on. tiny-sandwich-scaled 63, 58, 50; tiny-sandwich-pre-qknorm 63, 54, 46; tiny-scaled-qknorm 64, 55, 47. Each equals the engine's prediction.
   - Refusals (`--compiled --fusion`), none fixed because each names a missing kernel, not a wrong rule: qk norm into the qkv GEMV or attention (`Opaque`, the full width qk norm is two launches); the norm add into the residual GEMV epilogue (`Ownership`: a GEMV block owns 8 of the 128 columns); the scaled add into the residual GEMV epilogue (the dry run fuses it, the GEMV library has no scaled residual epilogue, so `NoKernel`); `PostAttention norm and add` into the next GEMV as a prologue (`NoKernel`, no fused norm add GEMV) and into the next norm (`Path`); the gated act into the down GEMV (`NoKernel`). With `store_epilogue` off, `Fill` is refused as `NoKernel` (the store GEMV is not emitted), which is the intended switch.
8. L1 slot fit, S1 capture grouping and L2 hoist (#1060).
   - L1. After the merge rounds, `Engine::split_over_budget` rechecks every group against the Target (`budget`: smem of the mainloop plus staged rows, registers) and re-splits one that no longer fits at the cheapest edge by `cut_cost` (launch bytes plus the intermediate's write and reads), preferring cuts where both halves fit, never cutting an op's units apart. A split shows as `Split a -> b ... slot fit split, the fused group needs ...` in `kernels --plan --compiled --fusion` (`Reason::SlotFit`). Under the real sm_120 budget no plan changes: `dense_decode_selects_the_hand_fusions`, goldens, `linear_unchanged` and the slice 7 launch test are unchanged.
   - S1. `Options::capture_groups` (off in the runtime, on in `single_traced` for the report) marks runs of GEMV launches with no data dependency between them (`compile::capture_groups`, over the typed signatures); on the hybrid fixture `lin_in` and `lin_ba` form one group. It does not change the plan the runtime runs. Test: `tests/capture_plan.rs`.
   - L2. `[runtime] pdl_hoist` (off): with `pdl`, every kernel with a weight pointer prefetches its weights into L2 before `griddepcontrol.wait` (`kir::cuda::with_pdl`). Text only, same numerics: `tests/pdl.rs` `pdl_hoist_on_and_off_are_bit_identical_on_tiny_fixtures` (8 fixtures, graphs on and off, logits bit identical to pdl off). Measured 2026-11-24 on desktop GPU1 (RTX PRO 6000, max clocks 3090/14001 MHz, card idle otherwise), commit b652dfb5: 27B NVFP4 batch 1 decode with `pdl` on in both arms, A/A 1.0003 and 1.0004, B/A 1.0041 and 1.0043 (IQR 0.1%), 99.2 against 99.6 tok/s. That is 0.4 percent, under the 5 percent bar, so it stays off. Command: `~/.agents/bin/gpu-run 1 -- bench --ab --pdl --rounds 6 --b-set runtime.pdl_hoist=true crates/specrt/tests/fixtures/qwen3.8-27b-nvfp4.toml 1 128`.

Out of scope: tuning parameters (formulas stay), a general tensor language, fusing attention or scans into neighbours (the measured losses above).

## 9. kir as the only kernel builder (#1136)

Every CUDA kernel the engine compiles comes out of a kir emitter, and the compiler enforces it.

- `kir::KirSource` wraps emitted text. Only kir constructs one (`pub(crate) fn emitted`); `Module::source` is a `KirSource`, and `Jit::load`, `jit::load_uncached`, `compile_cubin` and `compile_ptx` take only a `KirSource`. Library code that tries to turn a `String` into one does not compile. Tests that patch emitted text (tile sweeps, reference kernels) use `KirSource::test_text` and `edit_text`, behind kir's `test-text` feature, which only specrt's dev-dependency turns on.
- The two runtime kernels that bypassed kir are kir kernels now: the load time NVFP4 quantiser (`kir::weight_quant`, `kir::cuda::emit_weight_quant`, compiled and cached through `Jit`) and the bandwidth probe (`kir::bandwidth`, `kir::cuda::emit_bandwidth`). Both take the `Target` and give their launch geometry; their text is token for token the old `.cu` files (`kir/tests/weight_quant.rs`). The allreduce peer probe goes through `jit::load_uncached` too, so NVRTC is called only in `src/jit.rs`.
- Dev benches that are compiled by hand with nvcc (`fp4_gemv_bench.cu`, `bw_q6k_head.cu`, `p2p658.cu`) live in `crates/specrt/tools/cuda/`. No library code reads that folder.
- MoE: `cuda/moe/gemv.rs` builds every expert row dot from `prim::multi_row_dot` (fp4, mx through the new `RowFormat::Mx`, fp8, bf16); `cuda/moe/route.rs` builds its reductions from `block_reduce` and `warp_reduce`, the router row from `unpack_bf16x8` and the top k from `topk::warp_top_k`. Only `moe_up_mx` and `moe_down_mx` change PTX (the mx row dot now decodes 32 values then dots them, as the grouped path did). Launch geometry for the route, expert and row kernels is `kir::moe::{route_logits_launch, expert_rows_launch, rows_launch}`.
- GPU proofs (#1144, desktop GPU1, 2026-11-24): `nvfp4_quant` 4/4, `moe_route` 2/2, `moe_mma` 10/10 and the five gpt-oss tests in `variants` pass. 122B-A10B NVFP4 against vLLM: top-1 63/64, mean KL 0.0134. 122B batch 8 decode, old (d89f23b8) against new: 361.15 against 361.7 tok/s (medians, old/new/old three rounds), so the new `moe_up_mx` and `moe_down_mx` stay. Rows in `docs/perf/scoreboard.md`.
- Tile and split choice: none comes from env vars or per architecture Rust. Formulas live in kir and take the `Target` and shape: GEMM tiles and splits (`kir::gemm`, `BatchShape::splits`, `ggml_splits`), attention (`decode_splits`, `decode_chunk`, `decode_warps`, `flash_rows`, `flash_keys`, `flash_stages`), GEMV geometry (`kir::gemv::gemv_geometry`), MoE launches (`kir::moe`). `specrt::gemm` maps the spec to those inputs (`linear_shape`, `policy`, `tuning` from `[runtime]`) and picks kernel names.

Guard (`crates/specrt/tests/codegen.rs`, `every_kernel_comes_from_kir`, `the_kernel_text_guards_catch_a_new_offender`, `moe_builders_use_shared_prims_unless_listed`, `launch_geometry_comes_from_kir_builders`): it fails when an `include_str!` of a `.cu`, `.cuh` or `.wgsl` appears outside crates/kir, a `__global__` appears outside `kir/src/cuda` and the listed emitters and test references, NVRTC is called outside `src/jit.rs`, library code builds a `KirSource` from raw text or turns on `test-text`, library code reads `tools/`, an allowlist entry goes stale, or any allowlist changes size (every list is pinned). Each check was seen failing on a planted offender before it passed.

Prim library. Device function text stays text where it is a thin wrapper over one PTX instruction or a numeric conversion, because there is nothing to generate: `mma.sync` variants (`gemm/prim.rs`, `attention/mma.rs`), `cp.async`, `ldmatrix`, `ld.global.nc`, `ex2.approx`, `cvt` to and from e4m3, e2m1 and bf16 (`prelude.cu`, `decode_common.rs`), the warp and block reductions, and the `Dec*` weight format structs that the GEMV and tile templates are written against. Those are the instruction set the builders compose, and a new kernel calls them, it does not rewrite them. What can come from `Target` and `Layout` already does: KV addressing (`attention::prim::kv_addr` from `layout::addr_chain`), tile shapes, stage counts, thread counts and splits. Two pieces remain hand written that a new kernel could need to copy: the shared memory swizzle `swz<CPR>` in `prelude.cu` and the per family 2 bit swizzle `gemm::prim::swizzle`. Neither has a second user waiting, so they stay until one does; when one appears, express both as a `Layout` XOR mode and emit them.

## 10. Greedy versus exhaustive

The G4 to G6 engine is greedy: it merges the candidate with the best net saving and never undoes it. This section asks whether that leaves anything on the table by our own cost model (#1370). It tests the ordering only. The cost model itself is unmeasured against real timings for many shapes (#408 found no 5 percent win from measuring tile choices), so a gap here is a gap in bytes plus launches, not in microseconds.

Reference. `Probe` on `fuse_ops_probed`, `probed_report`, `denoiser_probed` and `compile::single_probed` (test and dev use only; the runtime passes no probe and runs the same engine as before). For each section it walks every grouping reachable from the singletons by merges the engine itself offers: the same `candidates`, the same `pair_legal`, `prologue_legal` and `fill_legal` checks from section 3, the same library predicate when one is given, and `budget` on every state. Candidates refused only for `Cost` stay open, so a merge that loses alone but enables a better one later is still seen. States are deduplicated, so the walk is a memoised search over sets of groups; it stops at 200,000 states per section and reports `capped` instead of guessing. No plan hit the cap.

Score. Both sides are scored on the final grouping with one function, `Engine::cost`: launches times `launch_bytes`, plus every value that still goes through memory (written once, read by each reader outside the writer's group; a split mainloop always writes its parts), plus the prologue reread penalty and the split ticket penalty of section 6. The table reports the saving against the unfused section, summed over layers. The greedy column is the engine's own grouping scored the same way, so two merge orders that reach the same groups score the same.

Plans: every fixture config and spec TOML that `Spec::from_hf_config_serving` or `Spec::load` accepts (tiny families, `tiny-qwen3.5-27b`, the 27B specs, Qwen3.5-4B, the 122B shape, the popular configs), decode at 1 token and prefill at 512 tokens with a 1,024 context, on sm_120 and sm_80; the compiled single step plan (`compile::single`, library predicate on) on both targets; and the klein-4b denoiser at 1,024 and 512 pixels. 436 plans.

Rule. If greedy is within 2 percent of the best on every plan, no search is needed. That does not hold, so each distinct cause has one follow-up issue. None is implemented here.

Result.

- Every compiled single step plan (the one the runtime launches) is at 0.00 on both targets, for every spec. The library predicate removes the groups that differ, because no GEMV kernel takes a rope or kv write epilogue and no kernel takes a hyper connection prologue.
- The klein-4b denoiser is at 0.00 at both sizes.
- `tiny-qwen3.5-27b`, every 27B spec, Qwen3.5-4B, the 122B, qwen3-next-80b, qwen3.6-35b-a3b and qwen3-0.6b are at 0.00 in every phase and target.
- 68 plan and phase pairs of the dry run are above 0, 52 of them above 2 percent on at least one target. Three causes:

| Cause | What greedy builds | What the best builds | Why greedy missed it | Issue |
|---|---|---|---|---|
| A | `[input norm + qkv + rope]` and `[kv write]` | `[input norm + qkv + rope + kv write]` | merge order: `Epilogue qkv -> rope` wins the round by 2 KB, then `Epilogue qkv + rope -> kv write` is refused as `Readers` (the qkv value is held for attention), while the other order, `Light rope -> kv write` first, passes every check. Legality depends on the path | #1386 |
| B | `[PostAttention norm and add + pre mlp norm]` and `[gate_up + gate_up act]`, or `[hyper pre 0 + input norm]` and `[qkv]` | `[pre mlp norm + gate_up + gate_up act]`, `[input norm + qkv]` | merge order: light merges run to a fixed point before any prologue (`PHASES`), so a norm joins its light neighbour and can no longer be a prologue of the next GEMV | #1387 |
| C | on prefill sm_120, `[input norm + qkv + rope]` over the shared memory budget, then split back to singletons | `[qkv + rope + kv write]` | a legality refusal applied late: the budget is checked only in `split_over_budget` after the merge rounds, and no merge round runs after the split | #1388 |

Gap in percent of the best saving, per plan, sm_120 then sm_80 (decode 1 token, prefill 512). Plans not listed are at 0.00. `fusion_exhaustive::greedy_stays_within_the_recorded_gap_of_the_exhaustive_pick` reads this table and fails when any plan's gap exceeds its number here (a plan not listed is held at 0), or when a search hits the cap. `a_worse_pair_order_breaks_the_recorded_gaps` merges the least saving candidate first on `tiny-qwen3.5-27b` and qwen3-0.6b and checks that the table catches it. When a follow-up closes a cause, lower its rows; print fresh rows with `FUSION_TABLE=1 cargo test -p inference-bench-specrt --test fusion_exhaustive print_doc_rows -- --nocapture`.

| Plan | Phase | sm_120 | sm_80 | Cause |
|---|---|---|---|---|
| gemma3-1b | decode | 0.02 | 0.02 | B |
| gemma4-31b | decode | 0.10 | 0.10 | B |
| glm4.5-air | decode | 24.34 | 24.34 | A |
| glm4.5-air | prefill | 27.85 | 27.85 | A |
| glm5.3-flash | decode | 1.66 | 1.66 | B |
| glm5.3-flash | prefill | 0.96 | 0.96 | B |
| gpt-oss-20b | decode | 24.70 | 24.70 | A |
| gpt-oss-20b | prefill | 25.83 | 25.83 | A |
| granite-3.3-2b | decode | 9.01 | 9.01 | A |
| granite-3.3-2b | prefill | 5.75 | 5.75 | A |
| granite-4.0-h-tiny | decode | 1.94 | 1.94 | A |
| granite-4.0-h-tiny | prefill | 2.74 | 2.74 | A |
| llama-3.2-1b | decode | 14.07 | 14.07 | A |
| llama-3.2-1b | prefill | 7.98 | 7.98 | A |
| llama4-scout | prefill | 0.00 | 16.68 | A |
| nemotron-3-nano-30b-a3b | decode | 6.37 | 6.37 | A |
| nemotron-3-nano-30b-a3b | prefill | 12.16 | 12.16 | A |
| nemotron-nano-9b-v2 | decode | 2.30 | 2.30 | A |
| nemotron-nano-9b-v2 | prefill | 2.31 | 2.31 | A |
| olmo2-1b | decode | 0.04 | 0.04 | B |
| phi3-mini | decode | 13.79 | 13.79 | A |
| phi3-mini | prefill | 13.38 | 13.38 | A |
| phi4-mini | decode | 14.14 | 14.14 | A |
| phi4-mini | prefill | 9.79 | 9.79 | A |
| qwen2.5-0.5b | decode | 14.15 | 14.15 | A |
| qwen2.5-0.5b | prefill | 7.90 | 7.90 | A |
| smollm2-135m | decode | 14.21 | 14.21 | A |
| smollm2-135m | prefill | 13.15 | 13.15 | A |
| smollm3-3b | decode | 14.11 | 14.11 | A |
| smollm3-3b | prefill | 6.69 | 6.69 | A |
| tiny-bamba | decode | 7.41 | 7.41 | A |
| tiny-bamba | prefill | 8.62 | 8.48 | A, C |
| tiny-chunked | decode | 13.80 | 13.80 | A |
| tiny-chunked | prefill | 13.19 | 13.19 | A |
| tiny-gemma3 | decode | 0.01 | 0.01 | B |
| tiny-gemma3-vision | decode | 0.01 | 0.01 | B |
| tiny-gemma3n | decode | 0.01 | 0.01 | B |
| tiny-gemma3n | prefill | 0.49 | 0.49 | B |
| tiny-gemma4 | decode | 0.01 | 0.01 | B |
| tiny-gemma4-e | decode | 0.01 | 0.01 | B |
| tiny-gemma4-moe | decode | 0.01 | 0.01 | B |
| tiny-glm5-next | decode | 4.55 | 4.55 | B |
| tiny-glm5-next | prefill | 4.56 | 4.56 | B |
| tiny-gpt-oss | decode | 23.53 | 23.53 | A |
| tiny-gpt-oss | prefill | 24.06 | 24.06 | A |
| tiny-granite | decode | 8.70 | 8.70 | A |
| tiny-granite | prefill | 8.48 | 8.48 | A |
| tiny-granitehybrid | decode | 8.34 | 8.34 | A |
| tiny-granitehybrid | prefill | 10.08 | 9.89 | A, C |
| tiny-mistral | decode | 13.80 | 13.80 | A |
| tiny-mistral | prefill | 14.48 | 14.48 | A |
| tiny-nemotronh | decode | 7.41 | 7.41 | A |
| tiny-nemotronh | prefill | 9.43 | 9.19 | A, C |
| tiny-nemotronh-moe | decode | 11.11 | 11.11 | A |
| tiny-nemotronh-moe | prefill | 16.33 | 16.32 | A, C |
| tiny-nemotronh-moe-mma | decode | 11.11 | 11.11 | A |
| tiny-nemotronh-moe-mma | prefill | 42.57 | 16.41 | A, C |
| tiny-olmo2 | decode | 0.01 | 0.01 | B |
| tiny-phi3 | decode | 13.43 | 13.43 | A |
| tiny-phi3 | prefill | 13.85 | 13.85 | A |
| tiny-phi3-hd96 | decode | 13.43 | 13.43 | A |
| tiny-phi3-hd96 | prefill | 14.41 | 14.41 | A |
| tiny-sandwich-pre-qknorm.toml | decode | 0.01 | 0.01 | B |
| tiny-sandwich-scaled.toml | decode | 0.01 | 0.01 | B |
| tiny-smollm3 | decode | 13.80 | 13.80 | A |
| tiny-smollm3 | prefill | 13.19 | 13.19 | A |
| tinymistral-248m | decode | 14.08 | 14.08 | A |
| tinymistral-248m | prefill | 9.39 | 9.39 | A |

Reading it: cause A is the large one and is the same group in every row (the KV write after the rope epilogue, about one launch plus one qkv write and read per attention layer). It does not show on the Qwen3.5 and 27B lines because their attention keeps a separate qk norm unit, so the kv write joins through a different path. A search would fix all three causes, but each has a cheaper local fix named in its issue: an order independent legality check (A), letting a prologue compete with a light merge (B), and the budget inside candidate legality (C). Until they land, greedy is good enough for every plan the runtime runs today (the compiled single step rows), and not for future prefill and dry run plans of the listed families.

## 11. Pass contract

Model semantics are fixed. A pass chooses how the model runs and never what it computes. When the faster or smaller choice would change the numbers, the pass refuses and the spec has to ask for it.

A pass may change:

- grouping (which ops share one launch, sections 5 and 6),
- kernel family (tile, batch, GEMV, ggml, split reduce),
- tile, split count and launch geometry (`kir::gemm`, `kir::gemv`, attention formulas),
- layout planes, only where a declaration below allows it,
- buffer reuse and aliasing (`Split` of x as an alias),
- launch order between ops with no data dependency (S1 capture groups),
- sharding of a row parallel linear into a rank partial plus an allreduce, when the spec has shards.

A pass may not change:

- the dtype or form of any tensor a model op writes (`Tensor::form`: f32, bf16, planes, fp8 or fp4 with scales, paged KV dtype),
- the accumulator of an op (the tensor an op reads and writes in place, such as the residual add into x or the branch),
- any explicit rounding: every write in a rounding form (bf16, planes, a quantised weight or activation form, a paged cache) must still be there, in the same form, and no new one may appear,
- the order of an authored ordered reduction: the sequence of ops that add into one accumulator (`x`, the branch, the rank partial, the latent, the hyper streams) between two overwrites,
- the set of tensors the model reads from outside the plan (inputs, weights, caches, state) and the live tensors it leaves at the end of a section (x, logits, draft logits, caches, state),
- sharding semantics: the ops whose partials an allreduce sums, and in which order.

Declared representation changes. Each of these changes numbers on purpose, and each is declared, not chosen by a pass:

| Change | Declared in | Who uses it |
|---|---|---|
| bf16 hi/lo planes instead of f32 activations for tensor core GEMMs | the graph (`graphs/decoder.toml` writes `*_planes` buffers) and the signature table (`Form::Planes`); `[runtime] split_activations` and `unsplit` pick one or two planes per linear | every planes plan |
| NVFP4 activation quantisation before an fp4 GEMM | spec `[mlp] act_quant = "fp4"` | `gemm::select` path `F4a`, `quant_rows_fp4` |
| fp8 activation quantisation | spec `act_quant = "fp8"` | path `F8a`, `quant_rows_fp8` |
| split K partials summed by a reduce kernel | the kernel family (`BatchShape::splits`, `ggml_splits`); the sum order inside one output element is the family's and is fixed by `Target` and shape | `F8b`, `F4b`, `Ggr` reduce |
| fp8 KV cache | `[runtime] kv_dtype`, `kv_f16_layers` | the cache `Form::Paged` |
| NVFP4 requant of fp8 checkpoint linears | `[runtime] requant` | `Spec::requanted` |
| fp32 accumulate in the plain path | `[runtime] gemm_fp32` | runtime GEMM choice |
| rank partials and their allreduce | `[tp]` shards from `Graph::shards` | `partition`, `AllReduce`, `AllReduceNorm` |

Enforcement. `signature::contract` reads, for every section of a plan, each op's writes with their forms, its accumulators, the rounding writes (casts), the ordered reduction chains and the section's boundary tensors, all from the typed signatures of slice 1. `signature::check_contract(pass, model, plan, declared)` compares the authored plan's contract against a later plan and fails as `pass P: section S, op O: field: ...`, with the field one of `tensor dtype`, `accumulator`, `cast`, `reduction order into B` or `boundary`. `compile::compile_traced` takes the contract of the plan it was given and checks it after every pass (`partition`, `gemv_layout`, `fuse`), so a pass cannot return a plan that skips it. `Declared` carries the declarations a pass may use: `shards` when the options carry `[tp]` shards, `planes_as_rows` and `no_media` for the single row layout.

Tests (`tests/plan_check.rs`). Each rule has a hand edited plan that `validate_plan` still accepts and the contract refuses, each seen failing with the check switched off before it passed: `the_contract_refuses_a_bf16_tensor_turned_f32` (input norm writes f32 xn instead of planes, `tensor dtype`), `the_contract_refuses_a_dropped_cast` (the rope and KV write op removed, `cast ... paged bf16 ... dropped`), `the_contract_refuses_a_changed_accumulator` (down writes planes instead of adding into x, `accumulator`), `the_contract_refuses_a_reordered_reduction` (a parallel residual whose scaled add moves before down, `reduction order into x`). `every_fixture_plan_keeps_its_contract_through_the_passes` runs `compile` on every fixture and spec in every phase, single rank and both tensor parallel variants; the `compile_plan` suite runs every single step plan and every TP rewrite through the same checks.

Representation changes the spec does not declare today. Each is let through by a `Declared` flag that the pass options set, not the spec:

| Pass | Change | Owner |
|---|---|---|
| `gemv_layout` | every planes buffer becomes its f32 row form in the single row decode plan, so the bf16 plane rounding of the authored plan is dropped (logits equal master, #1057, because the hand step did the same) | #1398 |
| `gemv_layout` | the media splice is dropped from the decode head (a decode step carries no media) | #1398 |
