# Decode attention at head_dim 256: occupancy against bandwidth (#1160)

Result: no change landed. On the Spark the 27B shaped `batch_decode_attention` already runs at 96% of the measured bandwidth ceiling, so a tiling pass cannot reach the 5% bar there.

Method: `batch_decode_attention_speed_on_a_27b_shaped_layer` (`crates/specrt/tests/decode_attention.rs`), release build, GB10 (sm_121), 27B attention shape (24 heads, 4 KV heads, head_dim 256, group 6), qlen 1, best of 8 launches. The kernel is the shipped one: 255 registers, spill, 1 warp per block, 2 blocks per SM (#1148).

Bandwidth ceiling on this GPU: 233 GB/s measured for lm_head streaming (`docs/spark-matrix.md`), 273 GB/s nominal.

| seqs x ctx | min ms | GB/s of KV | of 233 GB/s |
|---|---|---|---|
| 1 x 8k | 0.182 | 185 | 79% (launch bound) |
| 1 x 128k | 2.455 | 219 | 94% |
| 4 x 8k | 0.647 | 207 | 89% |
| 8 x 32k | 4.909 | 219 | 94% |
| 12 x 8k | 1.883 | 214 | 92% |

Occupancy sweep at 8 x 32k, `SPECRT_DECODE_BK` and `SPECRT_DECODE_STAGES` (keys per tile, stages; more keys in flight per SM):

| keys x stages | min ms | vs shipped (16 x 2) |
|---|---|---|
| 16 x 2 | 4.909 | 0% |
| 16 x 3 | 4.902 | 0.1% |
| 16 x 4 | 4.820 | 1.8% |
| 32 x 2 | 4.966 | -1.2% |
| 32 x 3 | 4.838 | 1.4% |
| 8 x 4 | invalid | see below |
| 8 x 6 | invalid | see below |

Quadrupling the keys in flight per SM moves the time by under 2%, which is inside the run to run spread. The register count and the spill sit in the compute part of the loop and the loads (cp.async, 2 stages of 16 keys) already cover the memory latency on a 273 GB/s part. The 12 x 128k row (512 GB/s) is an aliased pool that fits in L2 and says nothing about DRAM.

Decision: keep the kernel and the constants. The pass would only pay on a part with 6.6x the bandwidth, where 64 keys in flight per SM is a real limit (Little's law: 1792 GB/s at about 1 us latency needs about 1.8 MB in flight, 64 keys of 2 x 512 B is 64 KB per SM, 12 MB over 188 SMs). That needs the desktop RTX PRO 6000, which was serving the user's app (both GPUs at 98% and 93 GB used) during this run, so it was not measured.

Hand back for a window on an idle desktop GPU (`~/.agents/bin/gpu-run 1 --`):

```bash
SPECRT_DECODE_SEQS=8 SPECRT_DECODE_CTX=32768 SPECRT_DECODE_QLEN=1 \
SPECRT_DECODE_BK=16 SPECRT_DECODE_STAGES=4 \
cargo test --release -p inference-bench-specrt --test decode_attention batch_decode_attention_speed -- --ignored --nocapture
```

## Desktop RTX PRO 6000, GPU1 idle (second pass)

The same test binary ran on GPU1 (`gpu-run 1`, 5 MiB used before the run), with `SPECRT_DECODE_MAX_CTX=32768` (64 splits) and 8 seqs. Each figure is the min ms over 8 launches. Two repeats agreed within 2%.

| keys x stages | 1k | 8k | 32k |
|---|---|---|---|
| 16 x 2 (shipped) | 0.072 | 0.199 | 0.733 |
| 16 x 3 | 0.069 | 0.229 | 0.841 |
| 16 x 4 | 0.067 | 0.392 | 1.412 |
| 32 x 2 | 0.062 | 0.363 | 1.291 |
| 32 x 3 | 0.062 | 0.371 | 1.330 |

More keys in flight loses badly at 8k and 32k on this card too, so the pass this issue asked for (more keys or stages per SM) is the wrong lever. The `8 x 4` and `8 x 6` rows look fast (0.030 ms at 1k) but they are wrong: with 8 keys `SB / 2` is 0, the QK and PV loops do nothing, and the reference test fails (max abs error 0.72 at `lens=[1]`). The Spark rows for `8 x 4` and `8 x 6` above are invalid for the same reason.

The lever is the split size. `DM_CHUNK` (the smallest keys per split, 256) leaves most SMs idle at short context: 8 seqs x 1k is 4 KV heads x 8 seqs x 4 splits = 128 blocks on 188 SMs. Lowering only the device floor (`SPECRT_DECODE_CHUNK`, splits still 64; the 8 x 32k row ran at max_ctx 32768, the rest at 131072 with an 8 GB pool):

| seqs x ctx | chunk 256 | 128 | 64 | 32 |
|---|---|---|---|---|
| 1 x 1k | 0.071 | 0.038 | 0.024 | 0.017 |
| 1 x 4k | 0.072 | 0.040 | 0.027 | 0.028 |
| 1 x 16k | 0.077 | 0.075 | 0.074 | 0.075 |
| 4 x 2k | 0.072 | 0.039 | 0.026 | 0.031 |
| 8 x 1k | 0.071 | 0.038 | 0.025 | 0.030 |
| 8 x 2k | 0.072 | 0.041 | 0.045 | 0.052 |
| 8 x 4k | 0.086 | 0.089 | 0.102 | 0.104 |
| 8 x 8k | 0.199 | 0.204 | 0.204 | 0.205 |
| 8 x 32k | 0.734 | 0.732 | 0.734 | 0.732 |
| 16 x 2k | 0.087 | 0.091 | 0.103 | 0.129 |

The Spark's `8 x 4` and `8 x 6` rows in the first table are invalid for the same `SB / 2 == 0` reason, not a 1.8% gain.
| 32 x 16k | 1.433 | 1.422 | 1.422 | 1.423 |

Up to 4x at short context (8 x 1k: 71 to 25 us, against FlashInfer's 33 us in `kernel-vs.md`), no change at long context, and a loss when the grid is already full (8 x 4k, 16 x 2k). So the floor has to come from the grid fill: pick the chunk so that kv_heads x groups x active splits covers `Target::sm_count` times the blocks per SM (from smem), and no smaller. With chunk 32 the reference tests `decode_attention_matches_the_reference_across_splits_mixed_lengths_and_verify_groups` and `decode_attention_applies_window_and_softcap_like_the_reference` pass (2 passed).

## Grid-fill split policy: kernel wins, runtime does not (third pass)

Built on branch `issue-1160-split-policy` (commit 46dc9649, not merged): `DecodeGrid` in `crates/kir/src/attention.rs` picks the chunk per launch. It starts at one pipeline of keys (`DM_STAGES * DM_BK` = 32) and doubles while the blocks (kv_heads x groups x splits) exceed one wave (`sm_count` x blocks per SM from smem and registers: 564 on the RTX PRO 6000) or the 64 split cap. It stops at the old 256 floor, so a grid that already fills the card keeps the old split. The device `decode_chunk` (decode, combine, tile decode scan) computes the same chunk from `gridDim.y`. A GPU test checks that the splits the device writes match the host chunk on the tiny, 27B and 122B shapes. The CPU policy tests, the five gates, the re-blessed goldens (only `DM_CHUNK_MIN`, `DM_WAVE` and the `decode_chunk` lines changed) and the reference tests all passed.

Kernel A/B with the control in the same binary (`SPECRT_DECODE_WAVE=0` gives the old split), min ms, old / old again / new / new again:

| seqs x ctx | old | old | new | new |
|---|---|---|---|---|
| 1 x 1k | 0.069 | 0.069 | 0.018 | 0.018 |
| 1 x 8k | 0.071 | 0.071 | 0.044 | 0.044 |
| 8 x 1k | 0.069 | 0.070 | 0.027 | 0.027 |
| 8 x 2k | 0.070 | 0.071 | 0.042 | 0.042 |
| 8 x 4k | 0.081 | 0.083 | 0.082 | 0.083 |
| 8 x 8k | 0.203 | 0.201 | 0.200 | 0.201 |
| 16 x 1k | 0.070 | 0.070 | 0.042 | 0.042 |
| 16 x 2k | 0.083 | 0.083 | 0.085 | 0.085 |
| 8 x 32k | 0.734 | 0.733 | 0.735 | 0.732 |

Through the runtime it gains nothing. Real 27B NVFP4 (`orcarouter/Qwen3.8-27B-Uncensored-NVFP4`), `bench --serve --runs 3 --kv-gb 8 <model> <prompt> 128 <batch>`, base binary (master 3f8584a9) and branch binary alternated A B A B on GPU1, full-batch decode step in ms:

| batch x prompt | A | B | A | B |
|---|---|---|---|---|
| 1 x 1k | 25.94 | 25.87 | 26.31 | 26.15 |
| 1 x 8k | 26.94 | 27.34 | 27.04 | 27.28 |
| 4 x 1k | 32.73 | 32.49 | 32.69 | 32.57 |
| 4 x 8k | 34.89 | 34.50 | 34.39 | 34.34 |
| 8 x 1k | 34.05 | 33.93 | 34.52 | 34.18 |
| 8 x 4k | 35.72 | 35.50 | 35.74 | 35.51 |
| 8 x 8k | 37.94 | 37.68 | 37.82 | 37.63 |
| 16 x 2k | 41.69 | 41.57 | 41.65 | 41.56 |

All within 1.5%, under the 5% bar. The serving layout sets max_ctx near prompt plus new tokens, so at 1k context it launches about 37 splits, not 64, and decode attention is a small share of a 26 to 42 ms step dominated by the weight GEMVs. Saving 40 us on each of the 16 attention layers is at most 0.6 ms, under 2% of the step, which is what the table shows. Decision: the policy is not merged. The branch stays as a reference in case a profile ever shows decode attention as a larger share of the step.
