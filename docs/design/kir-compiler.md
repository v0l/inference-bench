# kir compiler: plan passes and fusion

A model's plan (`Vec<Op>` lowered from a graph spec, see [kir.md](kir.md)) goes through typed passes that decide which ops share a kernel. The passes read only the typed op signatures and the `Target`, never an architecture name. The kernel IR the passes rewrite is `kir::heavy::Heavy`.

```
graph + Spec -> Plan -> validate_plan -> compile passes (partition, gemv_layout, fuse) -> Plan the runtime runs
Plan -> kernel selection (gemm::select, attention formulas) -> Heavy / Compose IR -> slot passes -> emit -> NVRTC
```

## Signatures and plan checks

`crates/specrt/src/signature.rs` gives every `Op` a signature (`Signer::signature`, one exhaustive match): tensors read and written with rows, cols and form (f32, bf16, hi/lo planes, fp8 or fp4 with scales, paged cache), a class (`Elementwise`, `RowReduce`, `Gemm`, `Attention`, `Scan`, `Gather`) and its bytes and launches. `validate_plan` fails naming the op when a buffer is read before it is written, shapes disagree or a plane form does not match. `plan_report` prints per layer kind the ops, bytes and launches:

```
inference-bench kernels --spec S --plan [--phase prefill|decode|verify] [--tokens N] [--context C] [--target T] [--compiled] [--fusion]
```

## Fusion engine

`crates/specrt/src/fusion.rs`. Each plan op lowers to units over the signatures (a `Linear` is its mainloop with geometry from `gemm::select`, a split reduce when splits > 1, an act quant row op on the act quant paths, and its epilogue). Candidates are dataflow edges between groups. Rules (`Rule`):

| Rule | Fuses |
|---|---|
| `Epilogue` | a light consumer into a heavy producer's output tile (residual add, gated act, scale) |
| `Light` | adjacent row and elementwise ops into one `Compose` (qk norm, rope and KV write; allreduce plus norm) |
| `Fill` | a constant write into the group that reads it (`clear_branch` into a store epilogue) |
| `Prologue` | a row norm into the input load of the following GEMV or GEMM, duplicated into every heavy consumer |
| `Pair` | two anchored groups |
| `Split` | re-split a group that no longer fits the target (`split_over_budget`, slot fit) |

The engine is greedy: epilogue, light and fill merges run to a fixed point first, then prologues (`PHASES`); the best net saving wins each round.

Cost (`Costs::for_target`): saving is the launch cost plus the intermediate's write and reads; penalties are prologue rereads (`(col_blocks - 1) x row bytes x reread_rate`, less `reread_overlap`) and, for an epilogue on a split mainloop, `(splits + 1) x round_trip_bytes`.

| Target | launch | round trip | bandwidth | reread rate | overlap |
|---|---|---|---|---|---|
| CUDA | 2 us | 0.8 us | 1,600 GB/s | 0.25 | 1.0 |
| WGSL | 5 us | 2 us | 200 GB/s | 1.0 | 0.0 |

`Costs::with_compute` adds a compute ridge; a compute bound GEMM refuses a norm prologue (`Reason::ComputeBound`) unless `compute_prologue` is set (denoiser plans).

Legality refusals are `Reason` values printed by `--fusion`: `Opaque`, `TwoHeavy`, `Barrier`, `GridBarrier`, `Boundary`, `Rows`, `Schedule`, `Path`, `NotElementwise`, `Ownership` (an epilogue row op needs the block to own the whole row), `Readers` (the intermediate is read outside the group), `Smem`, `Regs`, `Cost`, `Outscored`, `NoKernel` (the library has no kernel for the group), `Layout`, `SlotFit`.

On the dense and hybrid fixtures the engine picks exactly the hand fusions the runtime had: norm into the qkv, gate_up, lin_in, lin_ba and head GEMVs; residual and act epilogues on GEMV, tile and split reduce; qk norm, rope and KV write; allreduce plus the next norm. It keeps apart `add_rmsnorm` (the residual already fused into its producer), attention reduce into the split (ticket penalty larger than the launch saving) and the norm fold into a wgpu GEMV (rereads larger than the saving).

## Compile passes

`crates/specrt/src/compile.rs`, `compile::compile(spec, plan, target, opts)`:

1. `partition`: with `[tp]` shards, row parallel linears and the MoE combine write a rank `Partial` followed by an `AllReduce`.
2. `gemv_layout` (single row family): planes buffers become f32 rows, a `Split` of x becomes an alias, decode attention and the delta rule step become `StepAttention` and `GdnStep`.
3. `fuse`: the engine above at one row with the kernel library predicate (`Fit::{Kernel, Part, None}`). Legal groups are the allreduce with the next planes norm (`AllReduceNorm`, when `[runtime] fuse`), a row norm with a whole GEMV (`Op::Gemv { norm }`, with or without the act epilogue), and with `[runtime] store_epilogue` the branch clear into a store GEMV (`Op::GemvStore`).

`Plans::compile` builds prefill, decode and verify through these passes, plus `single`, the one row decode plan `Runtime::step` runs through `executor::run_plan`. `compile::single_traced` returns the fusion trace; `Options::capture_groups` marks independent GEMV runs (`lin_in` and `lin_ba`) for the report.

Kernel slots (`kir::heavy`): `Heavy::lift` and `lift_gemv` lift a built kernel, `Heavy::attach` appends stages to the prologue or epilogue slot, `Heavy::fill` folds a constant, `Heavy::budget` checks threads, shared memory and registers, and `Heavy::lower` maps the stage lists back onto the builders, so the emitted text is the reference text. `heavy::norm_prologue` is the rmsnorm stage list; `gemm::normed` builds the fused GEMVs under the pass names `prologue_attach` and `epilogue_attach`. A failure reads `pass P: kernel K needs N what, the target has L`.

## Options

| Option | Default | Effect |
|---|---|---|
| `[runtime] fuse` | true | allreduce plus norm fusion |
| `[runtime] store_epilogue` | false | branch clear folded into a store GEMV: 58 to 50 launches on tiny-olmo2, 226 to 194 on OLMo-2 1B, decode 1.2 to 1.5 percent faster (#1058), under the 5 percent bar |
| `[runtime] pdl_hoist` | false | with `pdl`, prefetch weights into L2 before `griddepcontrol.wait`; 0.4 percent on the 27B NVFP4 batch 1 (#1060) |

## Pass contract

A pass chooses how the model runs, never what it computes. `signature::contract` reads per section each op's write forms, its accumulators, the rounding writes, the ordered reduction chains and the boundary tensors; `check_contract` runs after every compile pass and fails as `pass P: section S, op O: field` with field `tensor dtype`, `accumulator`, `cast`, `reduction order into B` or `boundary`.

A pass may change grouping, kernel family, tile, splits and launch geometry, buffer aliasing, the order of independent launches, and, with `[tp]` shards, the sharding into rank partials. It may not change a tensor's form, an accumulator, an explicit rounding, the order of adds into one accumulator, the tensors read from or left outside the plan, or which partials an allreduce sums.

Representation changes are declared, never chosen by a pass:

| Change | Declared in |
|---|---|
| bf16 hi/lo planes instead of f32 activations | the graph (`*_planes` buffers); `[runtime] split_activations` and `unsplit` |
| NVFP4 activation quant | spec `[mlp] act_quant = "fp4"` |
| fp8 activation quant | spec `act_quant = "fp8"` |
| split K partials | the kernel family (`BatchShape::splits`, `ggml_splits`) |
| fp8 KV cache | `[runtime] kv_dtype`, `kv_f16_layers` |
| NVFP4 requant of fp8 linears | `[runtime] requant` |
| fp32 accumulate | `[runtime] gemm_fp32` |
| rank partials and allreduce | `[tp]` shards |

Two changes are let through by a pass flag instead of the spec (#1398): `gemv_layout` drops the plane rounding in the single row plan, and drops the media splice from the decode head. Tests: `crates/specrt/tests/plan_check.rs`, `compile_plan.rs`, `fusion_plan.rs`, `capture_plan.rs`, `prologue_bits.rs`, `spec_mix.rs`.

## Greedy versus exhaustive

`fusion_exhaustive.rs` walks every grouping reachable by the merges the engine itself offers (same candidates, legality, library and budget; `Cost` refusals stay open), deduplicated, capped at 200,000 states per section, and scores both on the final grouping with `Engine::cost`. It covers every fixture config and spec at decode 1 token and prefill 512 on sm_120 and sm_80, the compiled single step plans and the klein-4b denoiser (436 plans, #1370).

Every compiled single step plan, the denoiser, and the Qwen3.5, 27B, 4B, 122B, qwen3-next and qwen3-0.6b plans are at 0. The dry run plans below are not. `greedy_stays_within_the_recorded_gap_of_the_exhaustive_pick` reads this table and fails when a plan's gap goes above its row (an unlisted plan is held at 0) or a search hits the cap; `a_worse_pair_order_breaks_the_recorded_gaps` checks the table catches a bad order. When a cause is fixed, lower its rows; print fresh rows with `FUSION_TABLE=1 cargo test -p inference-bench-specrt --test fusion_exhaustive print_doc_rows -- --nocapture`.

| Cause | What greedy builds | What the best builds | Why greedy missed it | Issue |
|---|---|---|---|---|
| A | `[input norm + qkv + rope]` and `[kv write]` | `[input norm + qkv + rope + kv write]` | merge order: `Epilogue qkv -> rope` wins the round by 2 KB, then `Epilogue qkv + rope -> kv write` is refused as `Readers` (the qkv value is held for attention), while the other order, `Light rope -> kv write` first, passes every check. Legality depends on the path | #1386 |
| B | `[PostAttention norm and add + pre mlp norm]` and `[gate_up + gate_up act]`, or `[hyper pre 0 + input norm]` and `[qkv]` | `[pre mlp norm + gate_up + gate_up act]`, `[input norm + qkv]` | merge order: light merges run to a fixed point before any prologue (`PHASES`), so a norm joins its light neighbour and can no longer be a prologue of the next GEMV | #1387 |
| C | on prefill sm_120, `[input norm + qkv + rope]` over the shared memory budget, then split back to singletons | `[qkv + rope + kv write]` | a legality refusal applied late: the budget is checked only in `split_over_budget` after the merge rounds, and no merge round runs after the split | #1388 |

## Greedy gap table

Gap in percent of the best saving, sm_120 then sm_80 (decode 1 token, prefill 512).

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

## Open structural gaps

From the static and timed kernel comparison against FlashInfer, Triton and vLLM ([../generated/perf/kernel-static-tables.md](../generated/perf/kernel-static-tables.md), [../generated/perf/kernel-vs-tables.md](../generated/perf/kernel-vs-tables.md)): 32 bit loads in norm, add norm, swiglu and rope where competitors load 128 (#1161); NVFP4 prefill GEMM on cp.async with 2 stages against TMA, 3 stages and warp specialisation (#1162); no sm_75 plan for the Qwen3.5 and 27B specs (#1163); `--split-compile` makes spills at 255 registers depend on unrelated kernels (#1164).
