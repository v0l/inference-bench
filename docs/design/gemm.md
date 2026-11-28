# GEMM family selection at decode and verify row counts (#1516)

`kir::gemm::select` used fixed row thresholds to pick the kernel family of a weight GEMM: up to 16 rows always took the batched GEMV (`f8b`/`f4b`, act quant off), 17 to 32 the skinny act-quant tiles, and so on. At 9 to 16 rows that took the slower kernel on most 27B linears (`lin_in` 105 us against 62 us at 24 rows). This page has the measured table, why the rule is now measured rather than a formula, and the end-to-end result.

## Sweep

`crates/specrt/tests/gemm_select_speed.rs` times every family `select` can legally pick for each linear of the 27B NVFP4 spec and the Qwen3.5-4B NVFP4 spec at 1, 2, 3, 4, 6, 8, 9, 12, 16, 17, 24, 32, 48, 64 and 96 rows: the single row GEMV at 1 row, `f8b`/`f4b` at every row bucket that holds the rows, the act-quant tiles (`_f8`, `_f8_narrow`, `_f8_big`, `_f8_huge`, `_f4a`, `_f4a_narrow`, `_f4a_skinny`, `_f4a_skinnywide`, `_f4a_big`) with their `quant_rows_*` launch inside the timed span, and the weight-only MMA tiles (`_mma`, `_mma_narrow`, `_mma_big`, `_mma_tiny`, `_f4w*`). Weights rotate over enough copies to pass 512 MB per chain, so L2 does not flatter small shapes. Each cell is the median of 7 chains of 32 launches, CUDA events.

```bash
export CARGO_TARGET_DIR=/mnt/data/cargo-target/inference-bench-1516
cargo test -p inference-bench-specrt --release --test gemm_select_speed --no-run
cd crates/specrt && SPECRT_SWEEP_OUT=/tmp/sweep.tsv ~/.agents/bin/gpu-run 1 --min-free 8000 -- \
  $CARGO_TARGET_DIR/release/deps/gemm_select_speed-<hash> --ignored --nocapture
```

Measured 27 to 28 Nov 2026 on GPU1 (RTX PRO 6000 Blackwell Max-Q, 188 SMs, card otherwise idle), branch base 4b34a462. The raw table (every kernel, every row count) is `docs/perf/gemm-select/sweep-sm120.tsv`; the CPU test `gemm_select.rs` reads it. Fastest eligible kernel per cell, in us:

### Act-quant specs (`act_quant = "fp8"`, `[mlp] act_quant = "fp4"`); act quant only above 8 rows

| model | linear | weight | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 12 | 16 | 17 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 27b | down | fp8 | proj 59.0 | f8b8 80.2 | f8b8 81.4 | f8b8 82.0 | f8b8 82.8 | f8b8 83.6 | f8b16 106.7 | f8_narrow 107.4 | f8_narrow 107.8 | f8_narrow 108.3 | f8_narrow 109.4 | f8 111.0 | f8 112.3 | f8_narrow 116.2 | f8 127.0 |
| 27b | down | nvfp4 | proj_f4 41.7 | f4b8 46.5 | f4b8 47.0 | f4b8 47.6 | f4b8 49.0 | f4b8 49.8 | f4b16 56.8 | f4b16 59.5 | f4a_narrow 60.2 | f4a_narrow 61.3 | f4a_narrow 63.0 | f4a_narrow 64.6 | f4a_narrow 67.0 | f4a_narrow 71.3 | f4a 87.4 |
| 27b | gate_up | fp8 | proj 111.1 | f8b8 153.9 | f8b8 156.3 | f8b8 158.4 | f8b8 162.6 | f8b8 165.9 | f8 140.8 | f8 142.3 | f8 142.1 | f8 142.5 | f8 144.8 | f8 146.9 | f8 151.0 | f8 155.9 | f8_big 197.4 |
| 27b | gate_up | nvfp4 | act_f4 70.7 | f4b8 105.2 | f4b8 107.8 | f4b8 110.4 | f4b8 113.5 | f4b8 115.8 | f4a_narrow 100.6 | f4a_narrow 101.0 | f4a_narrow 101.7 | f4a_narrow 102.0 | f4a_narrow 105.0 | f4a_narrow 109.5 | f4a 112.4 | f4a 115.5 | f4a_big 127.8 |
| 27b | lin_in | fp8 | norm 62.8 | f8b8 77.6 | f8b8 78.4 | f8b8 79.3 | f8b8 80.4 | f8b8 81.5 | f8 69.2 | f8 69.4 | f8 69.9 | f8 70.2 | f8 70.9 | f8 71.7 | f8 74.0 | f8 76.2 | f8_big 87.8 |
| 27b | lin_out | fp8 | proj 22.2 | f8b8 27.9 | f8b8 28.1 | f8b8 28.2 | f8b8 28.5 | f8b8 28.7 | f8b16 35.2 | f8_narrow 36.2 | f8_narrow 36.3 | f8_narrow 36.6 | f8_narrow 36.7 | f8_narrow 37.3 | f8_narrow 37.8 | f8_narrow 38.7 | f8 43.6 |
| 27b | o | fp8 | proj 22.1 | f8b8 27.6 | f8b8 27.6 | f8b8 27.8 | f8b8 28.1 | f8b8 28.3 | f8b16 34.1 | f8b16 35.5 | f8_narrow 36.2 | f8_narrow 36.6 | f8_narrow 36.6 | f8_narrow 37.1 | f8_narrow 37.3 | f8_narrow 37.7 | f8 41.8 |
| 27b | qkv | fp8 | proj 47.4 | f8b8 66.2 | f8b8 67.0 | f8b8 67.8 | f8b8 68.7 | f8b8 69.4 | f8 60.1 | f8 60.3 | f8 60.7 | f8 60.8 | f8 61.4 | f8 62.0 | f8 63.5 | f8 65.2 | f8_big 77.7 |
| 4b | down | fp8 | proj 17.2 | f8b8 20.5 | f8b8 20.7 | f8b8 20.7 | f8b8 20.9 | f8b8 21.0 | f8b16 24.3 | f8b16 24.8 | f8b16 25.2 | f8b32 29.0 | f8b32 30.3 | f8b32 31.5 | f8_narrow 49.7 | f8_narrow 50.1 | f8_narrow 50.5 |
| 4b | down | nvfp4 | proj_f4 13.3 | f4b8 15.9 | f4b8 16.0 | f4b8 16.2 | f4b8 16.3 | f4b8 16.4 | f4b16 17.9 | f4b16 18.4 | f4b16 18.9 | f4b32 23.7 | f4b32 24.9 | f4b32 26.3 | f4a_skinny 30.0 | f4a_skinny 30.3 | f4a_skinnywide 31.6 |
| 4b | gate_up | fp8 | proj 31.9 | f8b8 42.4 | f8b8 43.7 | f8b8 44.2 | f8b8 44.8 | f8b8 45.3 | f8 42.0 | f8 42.1 | f8 42.4 | f8 42.6 | f8 43.1 | f8 43.7 | f8 44.6 | f8 45.5 | f8_big 52.7 |
| 4b | gate_up | nvfp4 | act_f4 25.2 | f4b8 29.4 | f4b8 30.6 | f4b8 31.2 | f4b8 32.0 | f4b8 32.4 | f4a_skinnywide 25.0 | f4a_skinnywide 25.1 | f4a_skinnywide 25.3 | f4a_skinnywide 25.5 | f4a_skinnywide 25.2 | f4a_skinnywide 25.2 | f4a 29.3 | f4a 30.9 | f4a 48.7 |
| 4b | lin_in | fp8 | norm 25.5 | f8b8 29.9 | f8b8 30.7 | f8b8 31.2 | f8b8 31.6 | f8b8 32.0 | f8 27.9 | f8 28.1 | f8 28.2 | f8 28.2 | f8 28.6 | f8 28.9 | f8 30.0 | f8 31.1 | f8 39.9 |
| 4b | lin_out | fp8 | proj 8.6 | f8b8 12.6 | f8b8 12.6 | f8b8 12.6 | f8b8 12.6 | f8b8 12.5 | f8b16 14.2 | f8b16 14.4 | f8b16 14.3 | f8b32 15.9 | f8b32 16.3 | f8b32 16.9 | f8b64 23.4 | f8b64 24.6 | f8_narrow 24.6 |
| 4b | o | fp8 | proj 8.6 | f8b8 12.6 | f8b8 12.6 | f8b8 12.6 | f8b8 12.7 | f8b8 12.5 | f8b16 14.2 | f8b16 14.3 | f8b16 14.3 | f8b32 16.0 | f8b32 16.3 | f8b32 16.9 | f8b64 23.4 | f8b64 24.6 | f8_narrow 24.6 |
| 4b | qkv | fp8 | proj 19.0 | mma_tiny 26.4 | mma_tiny 27.1 | mma_tiny 28.2 | f8b8 29.1 | f8b8 29.3 | f8 24.5 | f8 24.6 | f8 24.7 | f8 24.7 | f8 25.0 | f8 25.2 | f8 25.9 | f8 26.7 | f8 33.6 |

### Exact specs (weight only, no act-quant path is eligible)

| model | linear | weight | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 12 | 16 | 17 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 27b | down | fp8 | proj 59.0 | f8b8 80.2 | f8b8 81.4 | f8b8 82.0 | f8b8 82.8 | f8b8 83.6 | f8b16 106.7 | f8b16 110.1 | f8b16 113.8 | f8b32 112.9 | f8b32 117.5 | f8b32 125.4 | f8b64 186.3 | f8b64 196.7 | mma_narrow 388.4 |
| 27b | down | nvfp4 | proj_f4 41.7 | f4b8 46.5 | f4b8 47.0 | f4b8 47.6 | f4b8 49.0 | f4b8 49.8 | f4b16 56.8 | f4b16 59.5 | f4b16 62.1 | f4b32 89.7 | f4b32 94.2 | f4b32 108.9 | f4b64 171.3 | f4b64 179.3 | f4w 361.2 |
| 27b | gate_up | fp8 | proj 111.1 | f8b8 153.9 | f8b8 156.3 | f8b8 158.4 | f8b8 162.6 | f8b8 165.9 | mma_tiny 191.4 | mma_tiny 200.0 | mma_tiny 212.9 | f8b32 228.7 | f8b32 245.1 | f8b32 263.5 | f8b64 364.4 | f8b64 385.0 | mma_big 624.3 |
| 27b | gate_up | nvfp4 | act_f4 70.7 | f4b8 105.2 | f4b8 107.8 | f4b8 110.4 | f4b8 113.5 | f4b8 115.8 | f4b16 129.2 | f4b16 135.4 | f4b16 143.1 | f4b32 179.6 | f4b32 192.8 | f4b32 208.9 | f4b64 319.1 | f4b64 336.8 | f4w_big 557.8 |
| 27b | lin_in | fp8 | norm 62.8 | f8b8 77.6 | f8b8 78.4 | f8b8 79.3 | f8b8 80.4 | f8b8 81.5 | mma_tiny 93.4 | mma_tiny 97.8 | mma_tiny 104.1 | f8b32 113.4 | f8b32 119.5 | f8b32 125.9 | f8b64 178.9 | f8b64 187.2 | mma_big 296.8 |
| 27b | lin_out | fp8 | proj 22.2 | f8b8 27.9 | f8b8 28.1 | f8b8 28.2 | f8b8 28.5 | f8b8 28.7 | f8b16 35.2 | f8b16 36.3 | f8b16 37.6 | f8b32 40.2 | f8b32 41.5 | f8b32 44.1 | f8b64 67.6 | f8b64 72.6 | mma_narrow 134.9 |
| 27b | o | fp8 | proj 22.1 | f8b8 27.6 | f8b8 27.6 | f8b8 27.8 | f8b8 28.1 | f8b8 28.3 | f8b16 34.1 | f8b16 35.5 | f8b16 36.6 | f8b32 38.7 | f8b32 40.0 | f8b32 41.7 | f8b64 62.7 | f8b64 66.1 | mma_narrow 128.2 |
| 27b | qkv | fp8 | proj 47.4 | f8b8 66.2 | f8b8 67.0 | f8b8 67.8 | f8b8 68.7 | f8b8 69.4 | mma_tiny 78.8 | mma_tiny 83.1 | mma_tiny 89.2 | f8b32 95.0 | f8b32 100.1 | f8b32 106.7 | f8b64 154.9 | f8b64 161.7 | mma_big 274.9 |
| 4b | down | fp8 | proj 17.2 | f8b8 20.5 | f8b8 20.7 | f8b8 20.7 | f8b8 20.9 | f8b8 21.0 | f8b16 24.3 | f8b16 24.8 | f8b16 25.2 | f8b32 29.0 | f8b32 30.3 | f8b32 31.5 | f8b64 52.0 | f8b64 54.9 | mma_tiny 115.7 |
| 4b | down | nvfp4 | proj_f4 13.3 | f4b8 15.9 | f4b8 16.0 | f4b8 16.2 | f4b8 16.3 | f4b8 16.4 | f4b16 17.9 | f4b16 18.4 | f4b16 18.9 | f4b32 23.7 | f4b32 24.9 | f4b32 26.3 | f4b64 47.3 | f4b64 50.1 | f4w_tiny 120.4 |
| 4b | gate_up | fp8 | proj 31.9 | f8b8 42.4 | f8b8 43.7 | f8b8 44.2 | f8b8 44.8 | f8b8 45.3 | f8b16 51.5 | f8b16 53.1 | f8b16 54.2 | f8b32 64.3 | f8b32 67.4 | f8b32 71.9 | f8b64 107.0 | f8b64 113.7 | mma_big 162.0 |
| 4b | gate_up | nvfp4 | act_f4 25.2 | f4b8 29.4 | f4b8 30.6 | f4b8 31.2 | f4b8 32.0 | f4b8 32.4 | f4b16 36.5 | f4b16 38.0 | f4b16 39.5 | f4b32 52.3 | f4b32 54.8 | f4b32 59.5 | f4b64 96.9 | f4b64 103.8 | f4w_big 141.9 |
| 4b | lin_in | fp8 | norm 25.5 | f8b8 29.9 | f8b8 30.7 | f8b8 31.2 | f8b8 31.6 | f8b8 32.0 | f8b16 37.0 | f8b16 37.8 | f8b16 38.7 | f8b32 45.8 | f8b32 47.7 | f8b32 50.4 | f8b64 76.5 | f8b64 80.9 | mma_big 138.6 |
| 4b | lin_out | fp8 | proj 8.6 | f8b8 12.6 | f8b8 12.6 | f8b8 12.6 | f8b8 12.6 | f8b8 12.5 | f8b16 14.2 | f8b16 14.4 | f8b16 14.3 | f8b32 15.9 | f8b32 16.3 | f8b32 16.9 | f8b64 23.4 | f8b64 24.6 | mma_tiny 50.6 |
| 4b | o | fp8 | proj 8.6 | f8b8 12.6 | f8b8 12.6 | f8b8 12.6 | f8b8 12.7 | f8b8 12.5 | f8b16 14.2 | f8b16 14.3 | f8b16 14.3 | f8b32 16.0 | f8b32 16.3 | f8b32 16.9 | f8b64 23.4 | f8b64 24.6 | mma_tiny 50.7 |
| 4b | qkv | fp8 | proj 19.0 | mma_tiny 26.4 | mma_tiny 27.1 | mma_tiny 28.2 | f8b8 29.1 | f8b8 29.3 | mma_tiny 30.4 | mma_tiny 31.8 | mma_tiny 33.5 | f8b32 41.5 | f8b32 43.2 | f8b32 45.2 | f8b64 62.8 | f8b64 66.0 | mma 111.6 |

Crossovers that matter. On the act-quant 27B, `qkv`, `lin_in` and FP8 `gate_up` take the act-quant tile from 9 rows, NVFP4 `gate_up` too (`f4a_narrow`, 101 us against the old `f4b16` at 135), and `down`, `o`, `lin_out` keep `f8b16`/`f4b16` at 9 to 12 rows and move to the narrow tile at 16. On the 4B the small `down`, `o` and `lin_out` stay on the batched GEMV up to 32 or 64 rows. On the exact specs the old rule was already close except for `qkv` and `lin_in` at 9 to 16 rows (`mma_tiny`), and `gate_up` FP8 there.

The act-quant tiles are also faster than `f8b8` at 2 to 8 rows on `qkv`, `lin_in` and FP8 `gate_up` (59 against 66 us on `qkv`), but act quant changes numerics, and below 9 rows it would change single sequence decode and the MTP verify of one or two sequences. So act quant stays off at 8 rows or fewer (`kir::gemm::act_rows`); this keeps those outputs identical to master.

## Rule

`[runtime] gemm_select` (spec TOML, `measured` by default; `formula` and `fixed` for comparison):

- `measured`: at load, for every tunable linear (`qkv`, `o`, `gate_up`, `down`, `lin_in`, `lin_out` with FP8 or NVFP4 weights) and every row bucket of `kir::gemm::TUNE_BUCKETS` (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96) that fits the prefill chunk, the runtime times each eligible candidate on the loaded weights, rotating over every layer's copy, and keeps the fastest. It keeps the formula's pick when that is within 3% of the fastest, so noise does not move a pick. Picks are appended to `gemm-select.tsv` beside the JIT cache (`[runtime] cache_dir`), keyed by the target (compute capability, SM count, shared memory, features), the policy, the shape, the bucket and the hash of the candidate kernel set, so a new card, spec option or kernel change measures again. A cached load does no timing. A request between buckets takes the pick of the bucket above it. Tensor parallel ranks, pipeline stages and streamed (offloaded) loads skip the measurement and use the formula.
- `formula`: the cold default and the pick when nothing is measured. Per candidate, `bytes * work / (fill * tail) + launches * fusion::Costs::launch_bytes`, where `bytes` is the weight bytes, `fill` the blocks against `Target::sm_count`, `tail` the last wave's fill against `blocks_per_sm`, `work` a per family factor (the row bucket of the batched GEMV, the M tiles of a tile GEMM), and one more launch for the quant kernel or the split reduce.
- `fixed`: the old thresholds, unchanged.

Act-quant candidates are only eligible when the spec sets `act_quant` (FP8 linears) or `[mlp] act_quant` (NVFP4 MLP), and only above 8 rows.

No formula fits the table within 5%. Fits tried on the 480 (linear, rows, spec) cells: bytes over bandwidth with occupancy and an MMA term per family (least squares, 9 to 16 free parameters, misses 40 to 100 cells by more than 5%, worst 1.6x to 3.6x), and the hand formula above (misses 43 of 448 cells at 64 rows or fewer, worst 1.65x, mean 1.022 of the fastest against 1.053 for the old thresholds). Picking per bucket from measurement misses none: every cell is within 2.3% of its fastest eligible kernel. So the formula is only the cold default. `tests/gemm_select.rs` checks both claims against the checked-in sweep.

Run to run: a second sweep at 12 and 24 rows on 28 Nov (344 cells) reads a median of 0.996 of the first, with 5% of cells 13% or more faster, mostly the batched GEMV at 12 rows (`lin_in_f8b16` 110.1 to 87.4 us). The fastest kernel changed in 5 of 32 cells, by 1.116x at most (27B FP8 `down` at 12 rows: `f8_narrow` first, `f8b16` second). That is why the pick is measured on the loaded model at load rather than taken from this table: the runtime tuner timed `f8b16` at 93.6 us and `f8_narrow` at 103.1 us for that cell on the 27B and kept `f8b16`. The cache file keeps every candidate's time beside the pick (third column).

## End to end

`bench --ab --rounds 6 --a-set runtime.gemm_select=fixed <spec> <batch> 128`, GPU1, 20 prompts, MTP k = 2, both arms in one process, the measured picks cached before the run.

| model | batch | A fixed tok/s | B measured tok/s | B/A, two passes | verdict |
|---|---|---|---|---|---|
| 27B NVFP4 | 1 | 100.8 | 100.1 | 0.992, 0.993 | no difference outside the A/A band |
| 27B NVFP4 | 4 | 297.7 | 354.0 | 1.182, 1.189 | B faster |
| 27B NVFP4 | 8 | 532.4 | 558.2 | 1.048, 1.048 | B faster |
| 27B NVFP4 | 16 | 763.2 | 790.8 | 1.031, 1.043 | B faster |
| 4B NVFP4 | 4 | 1059.7 | 1234.7 | 1.164, 1.166 | B faster |
| 4B NVFP4 | 8 | 1591.5 | 1614.4 | 1.013, 1.015 | no difference outside the A/A band |

Prefill is 3.6 to 4.5% faster in every run (the act-quant tile at the last partial chunk rows).

Batch 1 sits 0.5 to 0.8% under A, inside the tool's A/A band (verdict no difference) but repeatable. It is not a kernel choice: at 1 to 8 rows both modes pick the same kernels (`select` compared mode against mode for every linear, and the cache file), plain decode without MTP reads 0.998 and 0.999, and under nsys one MTP batch 1 serve run per mode (`crates/specrt/scripts/nsys1516.sh`, 256 tokens) launches the identical kernel set with identical counts, with kernel sums of 2318.7 against 2333.6 ms in one pair and 2339.6 against 2332.5 ms in the second, so the order flips. With the arms swapped (A measured, B fixed) B/A reads 1.005, the measured arm still 0.5% lower. The cause is outside the GEMM kernels and was not found.

Commands:

```bash
bench --ab --rounds 6 --a-set runtime.gemm_select=fixed crates/specrt/tests/fixtures/qwen3.8-27b-nvfp4.toml <1|4|8|16> 128
crates/specrt/scripts/gemm_select_4b.sh
bench --ab --rounds 6 --a-set runtime.gemm_select=fixed /tmp/gemm-select-4b/act.toml <4|8> 128
SEQS=1 crates/specrt/scripts/nsys1516.sh
```

Each under `~/.agents/bin/gpu-run 1 --min-free 80000 -- systemd-run --user --wait --collect -P -q -p MemoryMax=24G -p MemorySwapMax=0 -E CUDA_VISIBLE_DEVICES=1 -E CUDA_DEVICE_ORDER=PCI_BUS_ID -E HOME`. The first measured load of the 27B times 88 (linear, bucket) cells; later loads read `gemm-select.tsv`.

## Numerics

Outputs at 8 rows or fewer are unchanged (same kernels). Above 8 rows the act-quant specs now quantise activations at 9 to 16 rows too, as they already did at 17 rows and more, and the exact specs stay weight only. On GPU1: `fp8_batch` 6/6, `fp4_batch` 3/3, `fp8_gemm` 3/3, `mtp` 18/18 (the torch reference test skipped, its refs are not on this host), `tiny27b` 9/9, `real_27b_nvfp4_agrees_with_vllm_greedy_and_logprobs` (teacher forced top-1 1,219 of 1,280, mean KL 0.0133, p99 0.110; gates 90% and 0.05), `real_27b_mtp_acceptance_on_vllms_own_greedy_tokens` (acceptance length 2.340), `real_27b_nvfp4_first_layers_match_the_dequantised_checkpoint`. `tiny27b` `speculative_greedy_with_act_quant_equals_its_plain_greedy` and `fusion_plan` `batched_decode_puts_the_residual_and_act_in_the_split_reduce` pin `gemm_select = "fixed"`. On the tiny 27B fixture (256 wide) the formula picks the `_mma_tiny` tile for the 3 row verify where the old rule took `f8b8`; both are weight only, but the summation order differs from the 1 row GEMV plain decode uses, and the act spec greedy comparison of speculative against plain decode diverges on one of the four prompts. Under the old rule verify and plain decode also run different kernels (`f8b8` against the GEMV) and happen to agree. The fusion test checks the split reduce of the 8 row GEMV, which the formula no longer picks on the tiny shapes.
