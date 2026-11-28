# Decode weight GEMMs at 2 to 64 rows (#1514)

On the 27B NVFP4 the weight GEMMs of a speculative round at 3 to 60 rows ran at 40 to 60% of the card's read rate, through `f4b`/`f8b` (bf16 MMA on weights decoded in registers) or the act-quant tiles (`f4a`/`f8a`, native block scaled MMA with every warp loading weights through `cp.async`). This page covers what bounded them, the kernel built from that, and what it did end to end.

## Baseline round (master 94f10c79, GPU1, `crates/specrt/scripts/nsys1514.sh`)

`bench --speculative qwen3.8-27b-nvfp4.toml 1024 64 <batch>`, nsys `--cuda-graph-trace=node`, rounds cut at `spec_accept`, median per call over rounds with that many sequences (`crates/specrt/scripts/nsys_rounds.py`):

| sequences | rows | round kernel sum | NVFP4 `gate_up` | NVFP4 `down` | FP8 `lin_in` | FP8 `lin_out` | FP8 `qkv` |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 25.27 ms | 101.0 `f4b8` | 51.5 `f4b8` | 76.0 `f8b8` | 30.5 `f8b8` | 67.9 `f8b8` |
| 2 | 6 | 26.60 | 101.9 `f4b8` | 52.7 `f4b8` | 78.4 `f8b8` | 30.9 | 70.0 |
| 4 | 12 | 28.43 | 99.0 `f4a_narrow` | 57.6 `f4b16` | 64.6 `f8` | 39.0 `f8_narrow` | 57.0 `f8` |
| 5 | 15 | 28.54 | 96.6 `f4a_narrow` | 62.3 `f4a_narrow` | 63.0 `f8` | 35.4 | 55.7 |
| 8 | 24 | 31.86 | 97.1 `f4a_narrow` | 62.7 `f4a_narrow` | 62.5 `f8` | 35.9 | 55.2 |
| 16 | 48 | 39.76 | 113.7 `f4a` | 67.7 `f4a_narrow` | 66.0 `f8` | 37.5 | 58.4 |
| 32 | 96 | 58.17 | 128.3 `f4a_big` | 80.5 `f4a` | 74.4 `f8_big` | 48.1 `f8` | 69.8 `f8_big` |

## What bounded the batched GEMV

Probe on GPU1, card otherwise idle: the generated `gemv_f4b` from the 27B module, rebuilt with nvcc in four variants, weights rotated over 512 MB. Floor is weights plus scales at the measured stream rate (1,606 GB/s for a grid stride `ld.global.nc` loop; 62.4 us `gate_up`, 32.0 us `down`).

| `gate_up` (100 MB) | 3 rows `f4b8` | 12 rows `f4b16` | 24 rows `f4b32` |
|---|---|---|---|
| as built | 94.2 us | 118.8 | 177.5 |
| no fp4 decode (raw bytes as bf16) | 86.6 | 97.2 | 152.3 |
| no MMA | 84.7 | 87.9 | 97.1 |
| no weight load | 30.0 | 56.0 | 111.7 |

With no MMA and no decode the kernel still only reached 74% of the floor at 3 rows. A read-only kernel with the same access pattern (each lane loads 16 bytes of one row, four lanes cover 64 contiguous bytes of a row) runs at 991 GB/s against 1,606 for a stream; with `.L2::128B` on the load, or 32 contiguous bytes per lane, the same pattern reads at 1,532 to 1,588 GB/s. Each row's 64 bytes per step is half a 128 byte line and the second half arrives as a separate DRAM request. So the first bound is the access pattern, then the bf16 decode and the two bf16 MMAs per k step for the hi and lo planes, which grow with the row tile. With L2 cold (weights rotated over 2 GB) the 8 byte per row scale loads of the NVFP4 path cost another 13 us on `gate_up` unless they carry `.L2::256B`.

## The kernel: swapped act-quant rows (`f4s`, `f8s`)

`Kind::Swap` in `kir::gemm`, emitter `crates/kir/src/cuda/gemm/swap.rs`, lifted by the heavy IR as `Family::Swap`. The weights are the A operand of the native MMA (`mxf4nvf4.block_scale` m16n8k64 for NVFP4, `e4m3` m16n8k32 for FP8), loaded straight into registers with `ld.global.nc.L1::no_allocate.L2::128B`, 16 bytes per lane, two chunks in flight per warp; scales come through `.L2::256B`. One lane shuffle turns the 16 byte row load into the m16n8k64 A fragment and a byte permute puts the scales in the `scale_vec::4X` layout. Each warp owns 32 weight rows; activations are the B operand (8 tokens per n tile), quantised by the existing `quant_rows_fp4` / `quant_rows_fp8*` launch and read through L1. No shared memory. K splits over `blockIdx.y` until the grid covers `Target::sm_count` (`SwapShape::splits`), and the partials go through the existing `f4b`/`f8b` split reduce, which also applies the epilogue (residual, gated act, bias). Buckets 8, 16 and 24 rows (`SWAP_VARIANTS`).

It is one MMA per k step on native fp4 or e4m3, against two bf16 MMAs plus a register decode in `f4b`/`f8b`. It is only a candidate on specs that already quantise activations (`act_quant = "fp8"`, `[mlp] act_quant = "fp4"`), behind `[runtime] swap_rows` (default on); exact specs never see it. It computes the same products as the `f4a`/`f8a` tiles on the same quantised rows in another summation order. `tests/swap_rows.rs` checks every swapped variant against the `_narrow` act tile at 2 to 24 rows, within 1e-4 of the output scale and with nothing written past the live rows: the 27B NVFP4 `gate_up` and `down`, its FP8 channel `lin_in`, `qkv`, `lin_out` and `o`, and the 27B FP8 block spec's `lin_in`, `gate_up` and `down`.

With `swap_rows` on, act quant starts at 2 rows instead of 9 (`kir::gemm::act_rows`). One row decode stays weight only (`gemm_select.rs` `single_row_decode_stays_weight_only_on_act_quant_specs`), so plain single sequence decode is unchanged; MTP verify of one sequence (3 rows) and batched decode of 2 to 8 sequences now quantise activations, as these specs already did from 9 rows. `swap_rows = false` restores the old candidate set and the 9 row threshold (`swap_rows_off_keeps_the_previous_candidates_and_the_nine_row_act_threshold`).

The batched GEMV kept for exact specs decodes with `fma.rn.bf16x2` on the bit patterns (`prim::bf16x2_decode`). It is bit identical to the old `fp4_pair_bf16` and `e4m3_pair_bf16` over every code, checked on the GPU, and 2 to 5% faster with L2 cold. The `.L2::128B` hint helps the swap kernel (65 against 114 us with the hint removed) but makes `f4b` slower with L2 cold (`down_f4b8` 47.6 to 55.0 us), so the batched GEMV does not carry it.

## Per kernel times in isolation

`tests/gemm_select_speed.rs` (command in `docs/perf/gemm-select.md`), weights now rotated over 2 GB so L2 is cold, quant launch inside the span, raw table `docs/perf/gemm-select/sweep-sm120.tsv`. Best previous kernel against the swapped kernel at its bucket, in us:

| 27B linear | 3 rows | 12 rows | 24 rows |
|---|---|---|---|
| NVFP4 `gate_up` (100 MB) | 99.8 `f4a_narrow` / **71.3** | 101.3 / **75.1** | 106.1 / **79.7** |
| NVFP4 `down` (50 MB) | 55.4 `f4b8` / **37.5** | 60.2 / **38.5** | 62.3 / **43.3** |
| FP8 `lin_in` (84 MB) | 68.4 `f8` / **64.1** | 69.5 / **66.9** | 70.9 / **70.6** |
| FP8 `qkv` (73 MB) | 59.6 / **55.8** | 60.5 / **58.1** | 61.7 / **61.2** |
| FP8 `lin_out` (31 MB) | 29.2 / **28.5** | 36.1 / **29.6** | 37.3 / **31.1** |

In isolation NVFP4 `gate_up` at 3, 12 and 24 rows is under 80 us and `down` under 50 us; `lin_in` is 64 to 71 us against the 62 us target.

## In the model the card is power capped

The isolated numbers do not hold in a serving round. During a speculative bench on GPU1 the card sits at its 300 W limit with `SW Power Cap` active and the SM clock at 1,500 to 1,700 MHz, against 2,270 to 3,090 in the probes (`nvidia-smi --query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.sw_power_cap -lms 500` alongside the bench). Under that cap the swap kernel loses most of its margin: at 1 sequence `gate_up_f4s8` runs 99 us per call against 64 to 71 in isolation (`f4b8` 102 to 114, `f4a_narrow` 97). At 8 sequences `gate_up_f4s24` runs 119 against `f4a_narrow` at 97. The FP8 swapped kernels keep their gain at 3 rows: `lin_in` 63 against 77, `lin_out` 27.6 against 31.4, `gate_up` FP8 127 against 148. The swap kernel issues more instructions per byte than the cp.async tiles (shuffle, selects and per n tile B loads), and with the SM clock halved that, not DRAM, sets its time. A memory-only interleaved kernel between launches in the probe did not reproduce the slowdown (67.3 to 67.8 us), so it is the clock, not L2 state.

Two changes follow from this. The load-time tuner (`runtime/tune.rs`) now warms each candidate for three chains and takes the median of seven instead of the best of five, so a burst at full clock does not decide the pick. The swapped kernels are limited to 24 rows, where they still win in isolation by the widest margin.

## End to end

`bench --ab --rounds 6 --a-set runtime.swap_rows=false crates/specrt/tests/fixtures/qwen3.8-27b-nvfp4.toml <batch> 128`, GPU1, 20 prompts, MTP k = 2, A = master's kernels (and the 9 row act threshold), B = this branch, measured picks re-timed at load:

| batch | A tok/s | B tok/s | B/A, two passes | A/A | verdict |
|---|---|---|---|---|---|
| 1 | 99.6 | 105.8 | 1.0624, 1.0625 | 1.0005, 1.0001 | B faster |
| 4 | 355.5 | 351.4 | 0.9826, 0.9885 | 0.9997, 1.0001 | no difference outside the A/A band |
| 8 | 546.3 | 541.7 | 0.9909, 0.9982 | 0.9982, 1.0082 | no difference outside the A/A band |

An earlier build (swap up to 64 rows, scales without `.L2::256B`, tuner best of five) read batch 1 at 1.0976, batch 4 at 0.997 and 1.020, and batch 8 at 0.971 (B slower). The cap is why per kernel wins at 12 and 24 rows in isolation did not carry to batches 4 and 8.

## Numerics

GPU1: `swap_rows` 3/3, `fp4_batch` 3/3, `fp8_batch` 6/6, `tiny27b` 9/9, `mtp` 18/19: greedy identity at batches 1, 3, 4, 5 and 8 passes, and `tiny_mtp_draft_logits_match_the_torch_reference` needs torch reference files that are not on this host, the same skip as #1516. `real27b` 12/12 with `SPECRT_REF_DIR=~/.cache/specrt-ref-375` (run on the build with swap up to 64 rows). CPU gates: kir (all), `no_arch_names`, `codegen`, `golden`, `spec`, `linear_unchanged` (adds the `lin_in_f8s*`/`lin_out_f8s*` entry points, no existing text changed), `gemm_select` (sm_120 pin reblessed; the formula cold default is 1.025 of the fastest eligible kernel over the 480 sweep cells, against 1.101 for the fixed thresholds), `fusion_plan`, `compile_plan`, `plan_check`, `capture_plan`, specrt lib.
