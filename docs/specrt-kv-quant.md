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

### Not yet measured

These need the real 27B and a quiet GPU window from the parent, so the default stays `f16` and the numbers are open:

- KL against bf16 KV over 20 prompts at 8k and 32k, a 128k needle, MTP acceptance.
- Decode tok/s at 32k and 128k single stream and at batch 8 (nsys `attention_split` and `batch_decode_attention` times).
- KV token capacity of a fixed GB pool (expected about 1.9x f16).

Flip the default in `RuntimeOptions::default` only after those pass.

## TurboQuant notes

Paper: Zandieh et al., arXiv 2504.19874. No reference code was looked up in this slice. Kernel design for slice 3: rotate q once per step with a fixed per-layer sign-randomised Hadamard, so keys stay rotated in the cache and values rotate the output once; codebooks for 2, 3, 4 bits computed offline; QJL sign sketch of the key residual plus its norm for an unbiased score.

## Snapshot format

The in-memory state snapshots (`StateSnapshot`) hold recurrent state only, not KV. Disk snapshots of KV do not exist yet; when they land the header must carry `kv_dtype` and refuse a mismatch. The prefix cache namespace already does this.
