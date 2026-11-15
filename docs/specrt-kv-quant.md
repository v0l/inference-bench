# Compressed KV cache

Plan for #324, in order. Each slice is merged and measured before the next starts.

1. fp8 e4m3 KV (this slice, done below).
2. int8 and int4/int2 grouped with an f16 residual window (KIVI style).
3. TurboQuant (randomised Hadamard rotation, Beta-optimal scalar codebooks, QJL residual for keys).
4. Quality harness (perplexity, needle at 8k to 64k, KL against f16 over 20 prompts) and the defaults table.
5. Recurrent state in bf16 or fp8 with a 16k drift test.

## fp8 KV (slice 1)

Set it in the spec or the server config, not an environment variable:

```toml
[runtime]
kv_dtype = "fp8"
```

It also shows in the load dialog as "kv cache type" through `specrt::options` (`runtime.kv_dtype`). Default is `f16` until the 27B quality gate below has run.

Layout. A KV block (16 tokens, all KV layers) holds, per K or V plane, `layers * kv_heads * 16` rows of `head_dim` e4m3 bytes, followed by the same number of f32 scales: one scale per head per token row. `kv_block_bytes` reports `head_dim + 4` bytes per row instead of `2 * head_dim`, so per-token KV on the 27B drops from 64 KiB to about 33 KiB (0.52x; the 4 byte scale is 1.6 percent of a 256 wide row). The anatomy (`kv_bytes_per_token`), the pool sizing in `Layout::serving`, `/metrics` block bytes and the prefix cache namespace all follow it. A prefix cache entry written with one dtype never matches another.

The scale is per row, not per 16 token block: the store path (`norm_rope_head`) quantises each token as it is appended (`amax / 448`), so no old token is ever rewritten and a new token cannot disturb the scale of its neighbours. This is finer than vLLM's single scale and costs 4 bytes per row.

Kernels dequantise in registers: `attention_split` (single stream decode), `batch_decode_attention` (batched decode and MTP verify rows), `flash_attention` (plain prefill), and `flash_mma` (tensor core prefill, which widens e4m3 to f16 while filling shared memory, so the MMA path is unchanged). `kv_planes` is forced to 1 with fp8.

### Verified

| test | result |
| --- | --- |
| `tests/kv_dtype.rs` sizing, TOML parse | pass |
| `tiny_fp8_kv_logits_stay_within_the_documented_kl_of_f16` (300 token prompt) | KL 3e-6 |
| `head_dim_256_fp8_kv_logits_stay_within_the_documented_kl_of_f16` | KL 4e-6 |
| `fp8_kv_step_decode_matches_single_sequence_greedy` (stepped batch equals single stream, tiny and hd256) | pass |
| existing serving suite with f16 KV | 20 of 20 pass |

Documented bound for fp8: KL against the f16 cache under 0.02 and the same arg max on the tiny fixtures. The bf16 1e-3 logit gate stays for f16.

### 27B NVFP4 gate (15 Nov, desktop window, shared GPU, `scripts/kvgate27b.sh`, `src/bin/kvgate.rs`)

Prompts are the repeated `spec_prompts.txt` text; each KL row is the mean over the prefill-final position and 8 teacher-forced decode steps per prompt, fp8 KV against the 1-plane f16 KV.

| context | prompts | mean KL | max KL | top-1 agree |
| --- | --- | --- | --- | --- |
| 512 | 4 | 0.00008 | 0.0004 | 1.00 |
| 2048 | 3 | 0.0008 | 0.020 | 1.00 |
| 4096 | 3 | 0.118 | 3.13 | 0.96 |
| 8192 | 20 | 0.291 | 9.65 | 0.93 |

The noise floor (f16 1 plane against the 2 plane hi/lo layout, 4096 tokens) is 6e-5, so the loss is real fp8 loss. It grows with context and is dominated by a few positions with a huge KL (a flat position flipping to a sharp one), not by a uniform drift. The tiny fixtures stay at 1e-6 at 4096 tokens, so the kernels are not broken; the 27B keys are simply more sensitive to a 3 bit mantissa than the tiny random weights.

Decode at 8192 tokens, batch 8, 27B NVFP4 (GPU shared with other jobs, so indicative only): f16 276.5 tok/s aggregate, fp8 280.8 tok/s (+1.6 percent). The 32k and 128k speed rows, the 32k KL, the 128k needle and MTP acceptance were not measured: each load costs about 40 s and the window shares the lock with other workers.

Decision: the default stays `f16`. fp8 is opt-in and fails the 27B gate as it stands (KL 0.29 at 8k against a 0.02 bound). vLLM runs fp8 KV with no scale at all, so this is the cost of the same format. Next step before more formats: try a finer scale (per row, per 64 channel group) and keep the first 16 tokens (attention sink) in f16, then rerun `scripts/kvgate27b.sh`.

### Rerun the gate

```sh
cd crates/specrt
CARGO_TARGET_DIR=/mnt/data/cargo-target/inference-bench-324 cargo build --release --bin kvgate --bin bench
flock /tmp/gpu27.lock scripts/kvgate27b.sh /tmp/kv324.txt /mnt/data/cargo-target/inference-bench-324/release [kl|needle|speed|mtp|all]
```

## TurboQuant notes

Paper: Zandieh et al., arXiv 2504.19874. No reference code was looked up in this slice. Kernel design for slice 3: rotate q once per step with a fixed per-layer sign-randomised Hadamard, so keys stay rotated in the cache and values rotate the output once; codebooks for 2, 3, 4 bits computed offline; QJL sign sketch of the key residual plus its norm for an unbiased score.

## Snapshot format

The in-memory state snapshots (`StateSnapshot`) hold recurrent state only, not KV. Disk snapshots of KV do not exist yet; when they land the header must carry `kv_dtype` and refuse a mismatch. The prefix cache namespace already does this.
