# specrt: long-context batched decode attention

Issue #617. Under real agent load (13 sequences around 30k tokens each, MTP k = 2) the 27B decoded at
about 6.5 tok/s per sequence. The decode attention kernel was the cause.

## What changed

`batch_decode_attention` (`templates/ops.cu`) used to run one block per (KV head, row, split) with
8 fixed splits and scalar fp32 math. Each MTP verify row re-read its sequence's whole KV, so k = 2
read the KV three times, and a 30k context gave each block about 3.7k keys to scan serially.

The new kernel:

- One block per (KV head, verify group, split). A verify group is the `qlen` rows of one sequence
  (1 for plain decode, k + 1 in the speculative verify and in the MTP layer over the verify rows).
  The group's GQA heads times its rows form the query tile (6 x 3 = 18 rows on the 27B, two 16-row
  warps), so each KV page is loaded once per group.
- Tensor-core `mma.m16n8k16` with fp16 inputs and fp32 accumulation. Q and P are split into hi and
  lo halves (`DM_SPLIT`), which keeps the result within about 4e-6 of an f64 reference, the
  precision of the old fp32 kernel. The causal mask among the verify rows is per row (`key <= pos`).
- K and V tiles of 16 keys staged through shared memory with `cp.async`, double buffered.
- Splits scale with context: chunks of 256 keys, doubling while `chunk * splits < len`, with at most
  64 splits per group (fewer when `max_ctx` is short). A chunk depends only on the group's length, so
  a row gives the same bits whether it runs alone or in its verify group when both lengths pick the
  same chunk.
- `batch_decode_combine` reads the active split count from the same rule and takes `pos` and `qlen`.

Knobs are codegen constants (`DECODE_CHUNK`, `DECODE_KEYS`, `DECODE_STAGES`, `DECODE_MAX_SPLITS`,
`DECODE_SPLIT_PRECISION` in `codegen.rs`), not env vars. The speed test patches them to sweep.

## Synthetic single layer

`tests/decode_attention.rs`, `batch_decode_attention_speed_on_a_27b_shaped_layer`: one 27B attention
layer (24 heads, 4 KV heads, head_dim 256, bf16 1-plane KV), attention plus combine, min of 8 runs.
Pages are laid out as in serving (each sequence's own pages, scattered over the pool). GB/s counts the
unique K and V bytes of all sequences once; the read ceiling is 1,640 GB/s. Quiet GPU (local-llm
stopped), other inference-bench workers took turns on the GPU under the same lock.

| seqs x ctx | qlen | before ms | after ms | before GB/s | after GB/s | speedup |
|---|---|---|---|---|---|---|
| 1 x 8k | 1 | 0.099 | 0.070 | 340 | 478 | 1.4x |
| 1 x 8k | 3 | 0.099 | 0.056 | 338 | 596 | 1.8x |
| 1 x 32k | 1 | 0.441 | 0.137 | 304 | 980 | 3.2x |
| 1 x 32k | 3 | 0.436 | 0.130 | 308 | 1,033 | 3.4x |
| 1 x 128k | 1 | 1.793 | 0.514 | 299 | 1,044 | 3.5x |
| 1 x 128k | 3 | 1.776 | 0.496 | 302 | 1,082 | 3.6x |
| 4 x 8k | 1 | 0.121 | 0.084 | 1,109 | 1,600 | 1.4x |
| 4 x 8k | 3 | 0.311 | 0.096 | 431 | 1,396 | 3.2x |
| 4 x 32k | 1 | 0.498 | 0.397 | 1,079 | 1,351 | 1.3x |
| 4 x 32k | 3 | 1.388 | 0.391 | 387 | 1,373 | 3.5x |
| 4 x 128k | 1 | 1.962 | 1.467 | 1,094 | 1,464 | 1.3x |
| 4 x 128k | 3 | 5.634 | 1.467 | 381 | 1,464 | 3.8x |
| 12 x 8k | 1 | 0.408 | 0.289 | 987 | 1,396 | 1.4x |
| 12 x 8k | 3 | 0.853 | 0.309 | 472 | 1,304 | 2.8x |
| 12 x 32k | 1 | 1.594 | 1.079 | 1,010 | 1,493 | 1.5x |
| 12 x 32k | 3 | 3.393 | 1.133 | 475 | 1,421 | 3.0x |
| 12 x 128k | 1 | 6.362 | 4.193 | 1,013 | 1,536 | 1.5x |
| 12 x 128k | 3 | 13.538 | 4.281 | 476 | 1,505 | 3.2x |

At 12 x 32k with k = 2 the kernel reads 87% of the 1,640 GB/s ceiling (target 70%). At 64 layers
of which 16 are attention, that is 16 x (3.39 - 1.13) = 36 ms saved per verify round, from 54 ms of
attention to 18 ms.

Sweeps on the same layer: 16-key tiles with 2 stages win. 3 stages lose 20-40% at 4 and 12
sequences (occupancy), 32-key tiles lose 40% (register pressure at head_dim 256), a 128-key base
chunk helps only 1 x 8k (0.044 ms), and 512 hurts short contexts.

## 27B end to end

`scripts/long617.sh` runs `bench` with `SPECRT_BENCH_SERVE=1` (the scheduler, no HTTP) on
`qwen3.8-27b-nvfp4.toml` (MTP k = 2, 1-plane bf16 KV), N sequences of `ctx - new` prompt tokens
each, 256 new tokens for one sequence and 512 for more. The bench admits every prompt before
decoding starts (`max_batched_tokens` 16,384, a 10 s ITL target), so decode runs as one batch.
Base is master e7cc0bd, new is this branch; both carry the same `bench.rs`, which counts accepted
drafts as tokens (the old one did not). One run each, interleaved per case. Quiet GPU: local-llm
stopped by the supervisor; other workers' GPU jobs were serialised with these on
`/tmp/gpu27.lock`.

`decode` is all generated tokens over all decode time; `step` is the mean full-batch step (all N
sequences decoding, nothing prefilling), one speculative round of k + 1 verify rows per sequence.

| seqs x ctx | base tok/s | new tok/s | gain | base step ms | new step ms |
|---|---|---|---|---|---|
| 1 x 8k | 106.9 | 110.6 | 1.03x | 27.4 | 26.5 |
| 4 x 8k | 311.9 | 355.1 | 1.14x | 37.2 | 32.7 |
| 12 x 8k | 584.9 | 715.4 | 1.22x | 58.4 | 47.4 |
| 1 x 32k | 83.6 | 97.9 | 1.17x | 35.5 | 30.3 |
| 4 x 32k | 201.3 | 292.9 | 1.46x | 59.3 | 40.7 |
| 12 x 32k | 306.1 | 502.8 | 1.64x | 112.7 | 66.7 |
| 1 x 128k | 49.1 | 77.0 | 1.57x | 61.0 | 38.9 |
| 4 x 128k | 79.3 | 171.3 | 2.16x | 143.0 | 66.2 |

The 12-sequence rows use a fixed 40 GB KV pool (`SPECRT_BENCH_KV_GB`, a bench-only knob). With
the pool sized from free memory the recurrent state of the later slots cannot be allocated, so
only some of the 12 decode together (#647); the first 12 x 8k and 12 x 32k runs (595 vs 478 and 259 vs
385 tok/s) were taken that way and are not comparable.

### nsys, 12 x 32k

`scripts/nsys617.sh` profiles from the first full-batch decode step to the end (128 new tokens,
42 speculative rounds, 16 attention layers plus the MTP layer per round):

| | base | new |
|---|---|---|
| kernel time | 4,052 ms | 2,419 ms |
| `batch_decode_attention` share | 60.2% | 32.5% |
| `batch_decode_attention` median | 3,631 us | 1,157 us |
| `batch_decode_combine` median | 6 us | 35 us |
| attention GB/s (unique KV, 12 x 32.7k tokens, 1.61 GB per layer) | 443 | 1,389 |

The kernel reads 85% of the 1,640 GB/s ceiling inside the full model, against 27% before.
After it, the step is the MLP GEMMs (`gate_up`, `down`, 18%), the MTP head (5%), `lin_in` and
`lin_ba` (8%) and `gdn_commit` (3.5%).

### Targets

- Attention at 70% of the read ceiling: met, 85% in the 27B at 12 x 32k (87% on the synthetic
  layer).
- Single-stream 128k decode at 60 tok/s: met, 77 tok/s (from 49).
- 3x aggregate decode at 12 x 32k: not met, 1.64x (306 to 503 tok/s). Attention went from 60% of
  the kernel time to 33% (18.7 ms of a 57.6 ms round). The other 39 ms are the MLP and delta-net
  GEMMs at 36 rows, the MTP head and `gdn_commit`; with attention at zero cost the round would
  still be about 2.4x faster than base, so 3x needs those to shrink too (#646).

### Live-load reproduction

The issue's live numbers (13 sequences, about 430k live tokens, 6.5 tok/s per sequence) came from
the user's app on master. The 12 x 32k row is the same shape (387k live tokens): 25.5 tok/s per
sequence before and 41.9 after in this bench. The live app's lower number includes prefill
interleaved with decode, sequences above 32k and HTTP; that run needs the app and was not
repeated here.

## Tests

`tests/decode_attention.rs`,
`decode_attention_matches_the_reference_across_splits_mixed_lengths_and_verify_groups`: the kernel
against an f64 reference on a 27B-shaped layer and on `tiny-qwen3.5`, with lengths 1 to 16,397
(one chunk, many chunks, chunk boundaries at 255/256/257 and 16,384/16,385), mixed lengths in
one batch, verify groups of 2, 3 and 4 rows, scattered pages. Max error 4e-6. A row of a verify
group is bit-identical to the same row run alone when both pick the same chunk.

`SPECRT_REF_DIR=~/.cache/specrt-ref-375`, each suite with `--include-ignored --skip real_27b
--test-threads=1` (`scripts/suites617.sh`), quiet-GPU window:

```
decode_attention: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 24.47s
mtp: test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 207.77s
serving: test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 103.82s
linear: test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 74.02s
truncation: test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 64.22s
generate: test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 21.00s
kv_dtype: test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 19.08s
fused: test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.61s
head_quant: test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 173.14s
prefix: test result: ok. 25 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.50s
logits: test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 19.59s
real27b: test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 250.40s
lib: test result: ok. 144 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.32s
codegen: test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.11s
spec: test result: ok. 35 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

`mtp` includes greedy identity against plain decode at batches 1, 3, 4, 5 and 8 (k = 1 to 3, every
draft head, fp8 KV) and the chi-square rejection-sampling tests. `real27b` includes the vLLM
greedy and logprob agreement and the MTP acceptance tests.

### After the rebase onto 6a536ce

The reference test also covers the 122B's GQA 32/2 shape (32 q heads over 2 KV heads, group 16,
verify groups up to 4 rows: 64 query rows, 4 warps). Shared memory holds only the K/V tiles
(32 KB at head_dim 256), so it does not grow with the group; max error 3.9e-6.

```
decode_attention: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 52.88s
mtp: test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 200.85s
serving: test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 106.31s
linear: test result: ok. 29 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 83.79s
truncation: test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 61.44s
tiny27b: test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 102.40s
real122b: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 33.63s
```

`real122b` greedy: " Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of
Portugal is Lisbon. The", the continuation in `docs/specrt-moe.md`. 12 x 32k against rebased
master: 304.8 to 504.9 tok/s (full-batch step 113.2 to 66.4 ms).
