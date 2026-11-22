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

### Mixed layout on the 27B NVFP4 (16 Nov, `kvgate27b.sh all`, 8 f16 + 8 fp8 layers)

| row | f16 | mixed fp8 |
| --- | --- | --- |
| KL 8192, 20 prompts | 0 | mean 0.0028, max 0.112, top-1 0.989 |
| KL 32768, 10 prompts | 0 | mean 0.00007, max 0.0042, top-1 1.0 |
| needle 32k at 0.1, 0.5, 0.9 | 3 of 3 | 3 of 3 |
| needle 128k at 0.5 | found, decode 37.8 tok/s | found, decode 40.1 tok/s |
| MTP k=2, 20 prompts, mean acceptance length | 2.414 (97.3 tok/s) | 2.405 (93.5 tok/s) |
| batch 8 decode at 8192 | 299.7 tok/s | 262.7 tok/s |
| batch 8 decode at 32768 | 222.9 tok/s (222.1 on rerun) | 158.0 tok/s (158.0 on rerun) |

Quality passes the 0.02 bound on average at 8k and 32k (all-fp8 was 0.29 at 8k); the worst single position at 8k is 0.112, above 0.02. Single stream decode at 128k is faster (+6 percent). But batched decode (the grouped `batch_decode_attention` kernel) is 12 to 29 percent slower, reproducibly: its fp8 tile load does a synchronous global read, convert and shared store per thread instead of the `cp.async` pipeline, so it loses the overlap that the f16 path has. The default therefore stays f16. Fixing that load (cp.async the raw bytes and convert from shared memory) is the next step before fp8 can be the default; `long617.sh` against master was not run for that reason.

### Long context and the 150k loop report (#763, 18 Nov)

The user saw a verbatim paragraph loop at about 150k tokens with `kv_dtype = "fp8"` (8 f16 + 8 fp8 layers) on the 27B NVFP4. Harness: `kvgate loop` builds a 150k-token agentic transcript from the repo files (chat template, think blocks, tool calls, repeated similar narration), samples like the agent (temperature 1.0, top_k 20, top_p 0.95 from `generation_config.json`, thinking on) and flags a loop when a 40-token span repeats 3 or more times (`specrt::loops`). `kvgate kl` with `--corpus` uses the same transcript for KL. Scripts: `gate763_spark.sh`, `gate763_serve_spark.sh` (the server path with prefix cache and MTP), `gate763_desktop.sh`.

Loops, 1536 generated tokens at 150k, none looped (repeat ratio 0.000, worst repeat 1 in every run):

| build | path | arms | seeds |
| --- | --- | --- | --- |
| master (Spark) | engine, sampled | f16, fp8 (8 f16 layers), fp8 with 16 f16 layers | 2 each |
| master (desktop) | engine, greedy | f16, fp8 | 1 each |
| master (Spark) | server, prefix cache warmed at 120k, MTP on | fp8 | 2 |
| 583024b, the user's build (Spark) | server, prefix cache warmed at 120k, MTP on | fp8 | 3 |

KL of fp8 against f16 on the same transcript, 48 teacher-forced decode steps per prompt after the prefill:

| context | f16 layers | prompts | mean KL | max KL | top-1 agree | positions above 0.1 |
| --- | --- | --- | --- | --- | --- | --- |
| 32768 | 8 | 4 | 0.021 | 1.19 | 0.959 | 9 of 196 |
| 65536 | 8 | 3 | 0.129 | 14.2 | 0.952 | 4 of 147 |
| 131072 | 8 | 2 | 0.186 | 11.7 | 0.929 | 8 of 98 |
| 163840 | 8 | 2 | 0.036 | 2.25 | 0.980 | 5 of 98 |
| 65536 | 4 | 3 | 0.114 | 4.77 | 0.898 | 16 of 147 |
| 65536 | 12 | 3 | 0.00095 | 0.027 | 0.993 | 0 of 147 |
| 131072 | 12 | 2 | 0.0016 | 0.031 | 0.980 | 0 of 98 |
| 163840 | 12 | 2 | 0.00083 | 0.029 | 1.000 | 0 of 98 |

Conclusion: the loop was not reproduced on either build or either path. But fp8 with 8 f16 layers drifts badly from 64k (max KL 14, top-1 flips on 5 to 7 percent of positions), which is enough to tip a long sampled generation. With 12 f16 layers it stays clean to 160k. The 8k and 32k rows above used `spec_prompts.txt`, so they understated the drift on agentic text.

Default since #763: with fp8 KV and `kv_f16_layers` unset, the engine keeps half the attention layers in f16 up to 32768 tokens of configured context, and three quarters (12 of 16 on the 27B) above it (`Spec::for_kv_context`, applied by `Engine::load_with`, `Core::start` and the load dialog). An explicit `kv_f16_layers` still wins; below three quarters past 32k the dialog and the engine log warn. Cost: KV per token goes from 0.74x to 0.86x of f16, so fp8 holds about 1.16x the tokens of f16 instead of 1.35x. Verified bound with the default: 163840 tokens. int8 with a per-row scale (llama.cpp q8_0 style, about 0.51x of f16) is the planned long-context replacement.

### int8 KV gate (#796, 22 Nov)

Result: FAILED, code not merged. It lives on the unmerged branch `issue-796-int8-kv-code` (kir `kv_int8` shape flag, `KvDtype::Int8`, `kvgate --dtype int8`, goldens, the sink simulation). To revive it: `git merge issue-796-int8-kv-code`, rebuild `kvgate` and rerun the commands in this section on GPU1 through `gpu-run 1`.

`kv_dtype = "int8"` is implemented (one signed byte per value, f32 scale `amax / 127` per head and token row, quantised in `norm_rope_head`, dequantised in registers in all four attention families through the kir attention spec flag `kv_int8`). KV per token is 0.51x of f16 on the 27B. Tiny fixtures on GPU: KL against f16 is 2e-6 at 300 and 4096 tokens (`int8_kv_logits_stay_within_the_kl_bound_of_f16`).

27B NVFP4, GPU1, agentic transcript, 48 teacher-forced decode steps per prompt, every layer int8:

| context | f16 layers | prompts | mean KL | max KL | positions above 0.1 |
| --- | --- | --- | --- | --- | --- |
| 65536 | 0 | 3 | 0.044 | 2.26 | 7 of 147 |
| 163840 | 0 | 2 | 0.275 | 21.8 | 12 of 98 |
| 163840 | 2 | 2 | 0.356 | 22.9 | 15 of 98 |
| 163840 | 4 | 2 | 0.092 | 3.05 | 12 of 98 |

The gate (max KL under 0.05, no position above 0.1) fails: int8 with a per-row scale drifts like fp8 on the early attention layers. The spikes persist with 4 f16 layers (max 3.05). Batch-8 decode, 27B, 8k context: f16 305.6, fp8 305.7, int8 299.8 tok/s aggregate (int8 is 2 percent behind). The 32k batch-8 row is missing: it ran out of GPU memory on GPU1. Skipped: `kvgate loop` at 150k and the 128k KL row, since the 64k and 160k rows already fail the bound (an earlier 128k run with the default 8 steps was clean, max KL 0.0001, but that is 18 positions).

Because the gate fails, discover and the load dialog still offer f16, mixed fp8 and fp8 only; int8 is opt-in through `runtime.kv_dtype`.

What the numbers imply: int8 with a per-row scale has 127 levels against fp8's 3-bit mantissa and is still as bad as fp8 on the early layers, so the loss does not come from the number of bits. The likely cause is a few tokens or channels with extreme values (attention sinks, outlier channels in K) that any per-row scale handles badly: one large channel sets the scale and crushes the rest of the row.

Experiment, sink exemption (64k, 3 prompts, 48 steps, all layers). A simulation in `norm_rope_head` (env `KIR_FAKE_INT8_SKIP=<n>`, f16 cache, K and V rows at positions `>= n` pass through the int8 per-row quantise and dequantise, rows below `n` stay exact):

| exempt first tokens | mean KL | max KL | positions above 0.1 |
| --- | --- | --- | --- |
| 0 (all int8, simulated) | 0.047 | 2.58 | 7 of 147 |
| 64 | 0.023 | 1.75 | 6 of 147 |

The simulated 0 row matches the real int8 run (0.044, 2.26). Keeping the first 64 tokens exact halves the mean but leaves the spikes: max KL stays 1.75 and 6 of the 7 bad positions remain. Answer: no, the spikes are not mainly the attention sink. They are spread over later tokens, or come from outlier channels in K. Not run (did not fit the window): a per-channel K scale, and logging the position of the worst row per prompt. Next experiment: a per-channel K scale (a fixed per-layer, per-head channel scale from a calibration pass) through the same simulation hook, plus per-position error logging for the spike rows. Until then the #763 stopgap (12 of 16 layers f16 past 32k) stays the long-context choice.

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

## Decode attention speed (#651)

Batch 8 decode on the 27B (`kvgate27b.sh speed`, 3 paired runs, tok/s), K and V both converted at fragment load:

| context | f16 | fp8 |
|---|---|---|
| 8k | 299.5 to 299.9 | 298.9 to 299.1 |
| 32k | 223.0 to 223.2 | 220.8 to 220.9 |

fp8 is within 0.3 percent at 8k and 1 percent at 32k, so the default stays f16. At 12 x 32k (`long617.sh`, KV_GB=40) fp8 matches f16 (577.8 against 575.0 tok/s) and holds twice the KV blocks (75915 against 38550). The decode kernel loads raw e4m3 and the per-row scales with cp.async, converts K and V straight into the mma fragments with `cvt.rn.f16x2.e4m3x2` and applies the scales to the scores. `flash_mma` uses the same raw cp.async load for prefill.
