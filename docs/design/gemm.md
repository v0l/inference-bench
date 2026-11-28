# Weight GEMMs and GEMVs

Every linear runs through one of the kir GEMM families (`crates/kir/src/gemm.rs`, device code in `crates/kir/src/cuda/gemm/`). `specrt::gemm` (`crates/specrt/src/gemm.rs`) maps the spec to a `kir::gemm::Policy` (`linear_shape`, `policy`, `tuning` from `[runtime]`) and `kir::gemm::select` returns a `Choice`: the path, the kernel, the quantiser to launch first and the reduce to launch after. `gemm_kernels(spec, target)` builds every instance a module needs, so the module list, shared memory table and dispatch come from one place. Weight formats are in [quantisation.md](quantisation.md).

## Families

| Kind (`Kind::block`) | Kernels | Used for | Needs |
|---|---|---|---|
| `Tile` (`TileMma`) | `gemm_tc` with a weight decoder (`DecBf16`, `DecQ8`, `DecFp8`, `DecGgml`, `DecFp4` as `f4w`), bf16 `mma.m16n8k16` on hi/lo activation planes | prefill and wide batches, weight only | `Mma16816`, `CpAsync` |
| `Tile` (`DequantF16`) | 16x16 workgroup tile, weights dequantised to f16 | wgpu | `ShaderF16` |
| `Fp8Act` | e4m3 x e4m3 `mma.m16n8k32`, per token scale applied once in the epilogue; `_huge` 128x128 tile above 256 rows | `act_quant = "fp8"` | `Fp8Mma` (sm_89 and up) |
| `Fp4Act` | `mma ... kind::mxf4nvf4.block_scale`, ldmatrix from XOR swizzled 72 byte rows, scales beside their row | `[mlp] act_quant = "fp4"` | `BlockScaledFp4` (sm_12x, compiled as `compute_120a`) |
| `Batch`, `BatchReduce` | `gemv_f4b`, `gemv_f8b`: weight rows on the M side of the bf16 mma, tokens on N, every weight byte read once, split K over `blockIdx.y` and a reduce that applies the epilogue; variants 8, 16, 32, 64 rows; fp4 and e4m3 decoded with `fma.rn.bf16x2` on the bit patterns (`prim::bf16x2_decode`) | decode and verify of 2 to 64 rows on weight only specs | `Mma16816`, `CpAsync` |
| `Swap` (`f4s`, `f8s`) | weights as the A operand of the native `mxf4nvf4.block_scale` m16n8k64 or e4m3 m16n8k32 mma, loaded straight to registers with `ld.global.nc.L1::no_allocate.L2::128B` (scales `.L2::256B`), 32 weight rows per warp, quantised activations as B, no shared memory, split K to fill the SMs and the `f4b`/`f8b` reduce; buckets 8, 16, 24 rows (`SWAP_VARIANTS`) | 2 to 24 rows on act quant specs, with `swap_rows` | as `Fp4Act` or `Fp8Act` |
| `GgmlBatch`, `GgmlGemv` | the ggml block decoders | GGUF, see [gguf.md](gguf.md) | |
| `QuantFp8`, `QuantFp4` | `quant_rows_fp8_token`, `quant_rows_fp4` (vLLM `scaled_fp4_quant` rounding) | before the act quant GEMMs | as the GEMM |
| `Gemv`, `HeadNormFp4` | `gate_up_act_f4`, `down_proj_f4`, `lm_head_norm_f4`: one warp per row pair, 16 column groups per lane, 512 threads (`FP4_GEMV_THREADS`) | single row NVFP4 decode | `WarpShuffle`, `CpAsync` |

The single row bf16, FP8 and Q8 GEMVs are `kir::gemv` (`linear_w`), with the norm prologue and act epilogue from the fusion passes ([kir-compiler.md](kir-compiler.md)). Without `BlockScaledFp4` (sm_80, sm_89) an NVFP4 MLP falls back to `f4w`; without `Fp8Mma` FP8 weights use `DecFp8` through the tile GEMM.

Tiles, stages and splits are formulas over the target: `MmaShape::new` (bk and stages from shared memory), `BatchShape::for_tokens`, `BatchShape::splits(target, n, k)` (the fewest splits that fill one wave), `blocks_per_sm`. `Tuning` carries the `[runtime]` overrides:

| Option | Default | Effect |
|---|---|---|
| `tile` | `auto` | `small`, `big`, `narrow` force one tile GEMM shape |
| `big_tile` | none | `[bm, bn, wm, wn]` for the bf16 and FP8 `_big` tile |
| `fp4_tile` | `[128, 256, 4, 4]` (`Tile::FP4_BIG`) | the `Fp4Act` `_big` tile; bn must be a multiple of 16 x wn |
| `fp8_tile` | none | the FP8 act tile |
| `fp4_stages` | 4 (3 on skinny tiles), clamped to what fits | cp.async depth of `Fp4Act`; the 128x256 tile fits 3 in 99 KB |
| `gemm_fp32` | false | the fp32 tile GEMM and scalar attention (also on pre Ampere) |
| `gemm_select` | `measured` | see below |
| `swap_rows` | true | on specs that quantise activations, add the `Swap` kernels as candidates and quantise activations from 2 rows instead of 9; false keeps the previous candidates and the 9 row threshold |

## Choosing a family per row count

`[runtime] gemm_select`:

- `measured` (default): at load, for every tunable linear (`qkv`, `o`, `gate_up`, `down`, `lin_in`, `lin_out` with FP8 or NVFP4 weights) and every bucket of `TUNE_BUCKETS` (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96) that fits the prefill chunk, the runtime times every eligible candidate on the loaded weights, rotating over the layers, and keeps the fastest, or the formula's pick when it is within 3 percent. Picks go to `gemm-select.tsv` beside the JIT cache (`crates/specrt/src/runtime/tune.rs`), keyed by target, policy, shape, bucket and candidate set, so a cached load does no timing. A row count between buckets takes the bucket above. Tensor parallel ranks, pipeline stages and offloaded loads use the formula.
- `formula`: the cold default. Per candidate, `bytes * work / (fill * tail) + launches * fusion::Costs::launch_bytes`.
- `fixed`: the old row thresholds.

Act quant candidates are eligible only when the spec sets `act_quant` or `[mlp] act_quant`, and only above 1 row with `swap_rows` (above 8 rows without it, `kir::gemm::act_rows`), so single sequence decode keeps weight only numerics; MTP verify of one sequence and batches of 2 to 8 quantise activations. The swapped kernels compute the same products as the `f4a`/`f8a` tiles on the same quantised rows in another summation order. Candidates are timed after three warm chains as the median of seven, so a burst at full clock does not decide a pick.

The rule is measured because no formula fits the sweep within 5 percent (the hand formula misses 43 of 448 cells, worst 1.65x) while per bucket measurement is within 2.3 percent of the fastest eligible kernel in every cell, and run to run the fastest kernel changes in 5 of 32 cells by up to 1.12x (#1516). The raw sweep is [../generated/perf/gemm-select-sweep-sm120.tsv](../generated/perf/gemm-select-sweep-sm120.tsv), written by `crates/specrt/tests/gemm_select_speed.rs` (`SPECRT_SWEEP_OUT`); `crates/specrt/tests/gemm_select.rs` checks both claims against it and pins the sm_120 picks in `golden/gemm-select-sm120.txt` (`GOLDEN_UPDATE=1`).

## Measured state

`bench --ab --rounds 6 --a-set runtime.gemm_select=fixed <spec> <batch> 128`, desktop GPU1, MTP k = 2, 28 Nov 2026, base 4b34a462 (#1516):

| Model | Batch | fixed tok/s | measured tok/s | B/A |
|---|---|---|---|---|
| 27B NVFP4 | 1 | 100.8 | 100.1 | 0.99, inside the A/A band |
| 27B NVFP4 | 4 | 297.7 | 354.0 | 1.18 |
| 27B NVFP4 | 8 | 532.4 | 558.2 | 1.05 |
| 27B NVFP4 | 16 | 763.2 | 790.8 | 1.04 |
| 4B NVFP4 | 4 | 1059.7 | 1234.7 | 1.17 |

`swap_rows`, same harness, `--a-set runtime.swap_rows=false`, 27B NVFP4 (#1514): batch 1 1.062 (99.6 to 105.8 tok/s), batch 4 0.983 to 0.989, batch 8 0.991 to 0.998, both inside the A/A band. In isolation with L2 cold the swapped kernels win at 3 to 24 rows (NVFP4 `gate_up` at 3 rows 71.3 against 99.8 us, `down` 37.5 against 55.4), but in a serving round the card sits at its 300 W cap with the SM clock at 1,500 to 1,700 MHz, and the swapped fp4 kernel, which issues more instructions per byte than the cp.async tiles, becomes instruction bound (`gate_up_f4s24` 119 against `f4a_narrow` 97 us at 8 sequences). That is why batch 4 and 8 gain nothing; #1560 tracks it. The FP8 swapped kernels keep their gain at 3 rows under the cap. Profile a round with `crates/specrt/scripts/nsys1514.sh`.

Kernel level on the 27B NVFP4 (quiet GPU1, #441): `gate_up` `_f4a_big` at 1,024 tokens 0.494 ms isolated (738 TFLOP/s), 603 us minimum in the model against vLLM's 504 us. The remaining gap is load issue and the per tile `__syncthreads` inside the mma warps; producer warp and TMA mainloops were slower (#1162 tracks a TMA mainloop with warp specialisation).

## Tests

`crates/specrt/tests/fp4_batch.rs`, `fp8_batch.rs`, `swap_rows.rs` (every swapped variant against the `_narrow` act tile at 2 to 24 rows within 1e-4 of the output scale, nothing written past the live rows), `fp8_gemm.rs`, `fp4_gemv.rs` (random 27B shaped matrices against an f64 product, within 1e-4 of the largest output), `gemm_speed.rs` and `gemm_select_speed.rs` (timing, GPU), `gemm_select.rs` (CPU), `crates/kir/tests/gemm.rs` and the goldens in `crates/kir/tests/golden/gemm/`.
