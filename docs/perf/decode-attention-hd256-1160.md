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
| 8 x 4 | 4.819 | 1.8% |
| 8 x 6 | 4.961 | -1.0% |

Quadrupling the keys in flight per SM moves the time by under 2%, which is inside the run to run spread. The register count and the spill sit in the compute part of the loop and the loads (cp.async, 2 stages of 16 keys) already cover the memory latency on a 273 GB/s part. The 12 x 128k row (512 GB/s) is an aliased pool that fits in L2 and says nothing about DRAM.

Decision: keep the kernel and the constants. The pass would only pay on a part with 6.6x the bandwidth, where 64 keys in flight per SM is a real limit (Little's law: 1792 GB/s at about 1 us latency needs about 1.8 MB in flight, 64 keys of 2 x 512 B is 64 KB per SM, 12 MB over 188 SMs). That needs the desktop RTX PRO 6000, which was serving the user's app (both GPUs at 98% and 93 GB used) during this run, so it was not measured.

Hand back for a window on an idle desktop GPU (`~/.agents/bin/gpu-run 1 --`):

```bash
SPECRT_DECODE_SEQS=8 SPECRT_DECODE_CTX=32768 SPECRT_DECODE_QLEN=1 \
SPECRT_DECODE_BK=16 SPECRT_DECODE_STAGES=4 \
cargo test --release -p inference-bench-specrt --test decode_attention batch_decode_attention_speed -- --ignored --nocapture
```

Run it for the stage and key pairs in the table above. Build a tiling pass (warps per KV head, rows per warp, stages from `Target::max_regs`, `smem_limit` and the group size, refusing a spilling tile through `cuobjdump --dump-resource-usage` as `kernel_static.rs` does) only if some pair gains 5% there.
