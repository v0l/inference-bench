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

### Where the fp8 loss comes from (15 Nov, bisection)

Not a kernel bug. The loss shows up identically when fp8 is emulated in the store path of the f16 cache (quantise and dequantise at write time, then the unchanged f16 read kernels), so the fp8 read paths in `flash_mma`, `attention_split` and `batch_decode_attention` agree with the store. Prompt 1 at 4096 tokens, KL at the one spiking position (decode step 4, where the f16 reference top-1 probability is only 0.62; everywhere else it is 0.99+):

| fp8 applied to | spike KL |
| --- | --- |
| real fp8 kernels, all layers, K and V | 3.13 |
| emulated, all layers, K and V | 2.82 |
| emulated, K only | 4.11 |
| emulated, V only | 3.01 |
| emulated, KV layers 0 to 7 | 2.63 |
| emulated, KV layers 8 to 15 | 0.066 |
| emulated, KV layers 0 to 3 | 2.88 |
| emulated, KV layers 4 to 7 | 4.22 |
| emulated, KV layers 8 to 11 | 0.020 |
| emulated, KV layers 12 to 15 | 0.028 |
| emulated, scale per 32 channels instead of per row | 1.81 |
| emulated, first 1 or 16 tokens left exact | 3.16, 3.08 |

So the damage sits in the early full-attention layers (0 to 7), both K and V contribute, finer scales help only a little, and the attention sink tokens are not the cause. The rows are not saturating: the scale is `amax / 448` per row, so the largest channel maps to 448 exactly. The error is the 3 bit mantissa of e4m3 itself (about 6 percent relative) on layers whose attention is sharp, which flips a flat position. Checked: K and V use one scale per row in every kernel (`kv_scale` is read with the same row index as `kv_row`), the cache is one plane under fp8 (`load_stage` forces it), and prefill-final positions agree to 4e-4 while the decode steps carry the spikes.

Options, in order of cost: keep KV layers 0 to 7 in f16 and quantise 8 to 15 (about 0.75x bytes instead of 0.52x, spike KL 0.07), or move to a format with more mantissa for the early layers (int8 with a per-row scale, slice 2). Neither is implemented.

Decision: the default stays `f16`. fp8 is opt-in and fails the 27B gate as it stands (KL 0.29 at 8k against a 0.02 bound). vLLM runs fp8 KV with no scale at all, so this is the cost of the same format. Next step before more formats: try a finer scale (per row, per 64 channel group) and keep the first 16 tokens (attention sink) in f16, then rerun `scripts/kvgate27b.sh`.

### Mixed layout (16 Nov)

`kv_dtype = "fp8"` now keeps the first `[runtime] kv_f16_layers` attention layers in f16 (one plane, 2 bytes per value) and stores the rest as e4m3 with a per-row scale. Default is half of the attention layers (8 of 16 on the 27B), from the bisection above. `kv_f16_layers = 0` is the old all-fp8 layout. Each block holds the f16 layers first, then the fp8 layers, so `kv_block_bytes` is 0.75x of f16 on the 27B. Option key `runtime.kv_f16_layers` in the load dialog. Tests: `mixed_kv_keeps_the_early_layers_f16_and_defaults_to_half` (sizes), `mixed_kv_logits_track_f16_at_4096_tokens` (GPU, 0, 1 and 2 f16 layers on the 3-layer 27B-shaped fixture, KL under 0.02), the golden kernels in `crates/kir/tests/golden/attention`.

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
