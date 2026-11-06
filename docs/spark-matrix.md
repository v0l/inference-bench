# Spark matrix

Two backends on two models, run on the DGX Spark (GB10) with `inference-bench matrix`.
128-token completions of the fixed prompt in `spark-run/matrix.toml`, greedy, bf16 weights.
llama.cpp is `7bd8282c3` serving GGUFs converted from the same safetensors, specrt is `crates/specrt`.
Raw results are in `spark-run/runs`.

| model | backend | prefill tok/s | decode tok/s | TTFT | memory |
|---|---|---|---|---|---|
| Qwen3-0.6B | llama.cpp | 335 | 160 | 128 ms | not reported |
| Qwen3-0.6B | specrt | 151 | 137 | 284 ms | 3431 MiB |
| Qwen2.5-0.5B | llama.cpp | 345 | 186 | 125 ms | not reported |
| Qwen2.5-0.5B | specrt | 180 | 176 | 239 ms | 1651 MiB |

Prefill for llama.cpp over HTTP is prompt tokens over time to first token, so it includes request
overhead. specrt prefills in batched chunks (see crates/specrt/README.md); the recorded runs predate that.

After batched prefill, specrt bf16 prefill on the Spark (GB10, compute_121, `bench`, median of 5,
vLLM and Frigate resident on the GPU) is:

| model | 128-token prompt | 1024-token prompt |
|---|---|---|
| Qwen3-0.6B | 3965 tok/s | 2662 tok/s |
| Qwen2.5-0.5B | 4585 tok/s | 3978 tok/s |

The table above is the earlier run, before batching. On an NVIDIA RTX PRO 6000 Blackwell Max-Q
(compute_120) Qwen3-0.6B measured 1717-1924 tok/s at 128 tokens and 2316 tok/s at 1024 tokens.
The three logit tests also pass on the Spark (max abs 5e-5 qwen3 bf16, 1.1e-4 qwen2.5 bf16, 2.27
qwen3 Q8, top-1 100%).

Logit diff of specrt against `transformers` fp32 on the fixed 43-token prompt
(`inference-bench logit-diff --backend specrt --model Qwen/Qwen3-0.6B`): max abs 1.0e-4,
mean abs 6.5e-6, top-1 agreement 43/43.

## specrt decode kernels (issue #243)

Spark GB10, nsys median per launch, Qwen3-0.6B bf16:

| kernel | at start of #243 | attention fix | + fused norm / gate-up-act |
|---|---|---|---|
| attention | 38.7 us | 13.5-23 us | 23 us |
| gate_up (+silu*mul) | 56 us (+1.1 us swiglu) | 56 us | 59 us |
| qkv | 37.8 us | 37.6 us | 40 us |
| down | 28.5 us | 28.6 us | 29 us |
| o | 19.5 us | 19.5 us | 20 us |
| lm_head | 1.35 ms | 1.33 ms | 1.37 ms |

Attention now splits the V accumulation over timesteps across 16 warps (512 threads). RMSNorm is fused into the qkv, gate_up and lm_head GEMVs (each block normalises x into shared memory), and gate_up computes gate and up for the same row and applies the activation in the epilogue, so the separate rmsnorm and swiglu launches are gone. The fusion removes about 3 launches per layer but did not change decode speed: inside a CUDA graph those launches cost little, and the extra per-block norm work offsets the saving.

Alternating A/B on spark, 6 runs each, median decode tok/s, with llama.cpp (`llama-bench -n 128`, bf16 GGUF converted from the same safetensors) interleaved in the same session:

| spec | specrt before #243 | specrt now | llama.cpp |
|---|---|---|---|
| qwen3-0.6b bf16 | 77.9 (noisy, vLLM load) | 148.8 | 152.0 |
| qwen2.5-0.5b bf16 | 134.3 | 192.3 | 188.6 |
| qwen3-0.6b q8 | 107.3 | 203.1 | not comparable |

The earlier 137 and 160 tok/s in the table above were measured under a different GPU load. Fused vs unfused builds are within noise (qwen3 150.2 vs 148.4, qwen2.5 194.1 vs 193.5, q8 202.5 vs 203.1).

Bandwidth ceiling: lm_head moves 311 MB in 1.33 ms (233 GB/s) and gate_up 12.6 MB in 56 us (225 GB/s), about 83-85% of the ~273 GB/s roof. Streaming loads and unroll 8 gave about 2-3%. Attention reads under 1 MB of KV per layer at 128-256 tokens, so bf16 KV would not move decode speed at these lengths and was not done.

## specrt prefill: tensor cores and flash attention (issue #247)

GB10, `bench specs/<spec>.toml <prompt> 64`, prefill tok/s. Six alternating runs each of the 17ea141 build, the final build and `llama-bench -p N -n 0 -ngl 99 -r 3` (bf16 or q8_0 GGUF converted from the same safetensors), medians, one session, vLLM and Frigate resident (GPU 9-22% busy at start):

| model | prompt | 17ea141 | final | llama.cpp | final / llama.cpp |
|---|---|---|---|---|---|
| Qwen3-0.6B bf16 | 128 | 3993 | 10319 | 11693 | 0.88 |
| Qwen3-0.6B bf16 | 1024 | 3001 | 21248 | 24148 | 0.88 |
| Qwen2.5-0.5B bf16 | 128 | 4556 | 11985 | 13235 | 0.91 |
| Qwen2.5-0.5B bf16 | 1024 | 4407 | 28103 | 31839 | 0.88 |
| Qwen3-0.6B Q8 | 128 | 4156 | 11755 | 11925 | 0.99 |
| Qwen3-0.6B Q8 | 1024 | 3056 | 20020 | 22731 | 0.88 |

The first tensor-core commit (7c1b35e) reached 0.60-1.02 of llama.cpp; the extra work was a 1024-token chunk (one pass for a 1024 prompt), tile selection by grid size, ldmatrix fragment loads in the GEMM and flash attention, RMSNorm folded into the GEMM A-tile load, SwiGLU/GeGLU in the gate_up epilogue (gate and up rows interleaved at load), Q RMSNorm and RoPE inside flash attention, and no activation split on the Q8 path.

Per-kernel time, Qwen3-0.6B, 1024-token prompt, `nsys stats -r cuda_gpu_kern_sum`, median per launch (17ea141 runs 256-token chunks, so its rows are per 256 tokens; the final build runs the prompt as one 1024-token chunk):

| kernel | 17ea141 (us / 256 tok) | final (us / 1024 tok) | final per 256 tok |
|---|---|---|---|
| attention / flash_mma (fused Q norm + RoPE) | 1503 (mid-chunk 924 avg) | 409 | 102 |
| gate_up (+ norm, + act) | 464 | 436 | 109 |
| qkv (+ norm) | 340 | 310 | 78 |
| down (+ residual) | 347 | 213 | 53 |
| o (+ residual) | 235 | 150 | 38 |
| rmsnorm x2, swiglu, qk_norm_rope | 38 | 61 (row_rms, kv_norm_rope) | 15 |

Roofline. A probe kernel issuing back-to-back `mma.sync.m16n8k16` bf16 with fp32 accumulate measures 120-124 TFLOP/s on the GB10 (48 SMs, 2.4-2.5 GHz). With 2.2-2.4 GFLOP useful per kernel at 1024 tokens, gate_up runs at 6.4 GFLOP / 436 us = 14.7 TFLOP/s useful, 29 TFLOP/s issued (the bf16 activations are split into hi+lo for the 1e-3 logit gate, so every tile runs two mma), qkv 4.3 GFLOP / 310 us = 13.9 useful. ncu on the GB10: tensor pipe (hmma) active 34-43% of cycles in the GEMMs and 30% in flash_mma, warps active 28-33% (118-120 registers, 64x128 tile, 256 threads), and the top stall is long scoreboard (global loads of the f32 activations), then barrier. These kernels are not compute bound and not DRAM bound (the 1024-token gate_up moves 12.6 MB of weight and 12 MB of activation in 436 us, about 55 GB/s against ~273 GB/s). The ceiling here is latency: f32 activations loaded through registers and converted per k step.

What was tried and measured:

- Plain bf16 activations (no hi/lo split) fail the 1e-3 bf16 gate on every spec: max abs 0.08 (qwen3) to 1.1 (geglu), and dropping the split from any single linear gives 0.02-0.5 with top-1 loss. Dropping it everywhere only speeds the 1024 prefill up 2-5%, which confirms the mma count is not the limit. The split is kept on bf16 and dropped on Q8 (the Q8 logits do not move: max abs 2.26 vs 2.27).
- 128x128 tiles (2x4 or 4x4 warps, 196 registers, one block per SM) are slower than 64x128 (2x4 warps); 64x64 and 64x128 are best, chosen by grid size against SM count, with a 64x32 tile when the grid is under 2 blocks per SM and 16-row tiles for decode-sized batches.
- 3 and 4 stage cp.async pipelines for the weights: level with 2 at 3 stages, 25% slower at 4 (occupancy). Moving the f32 activations to cp.async as well: slower (qkv 290 to 320 us).
- Two streams: nsys shows 1-4% idle between kernels inside a prefill, so overlap was not added.
- Chunk 512 to 1024: Qwen3-0.6B 1024-token prefill 17.5k to 19.5k tok/s on the same GPU state.

Decode is unchanged, batch 1, 128/128, 6 alternating pairs: qwen3-0.6b 145.1 vs 154.4 (vLLM load shifted mid-run, pairwise level), qwen2.5-0.5b 193.4 vs 195.4, qwen3-0.6b-q8 227.6 vs 233.4.

Logits against transformers fp32 (`tests/logits.rs`, all pass): max abs 1.7e-4 qwen3, 5.4e-4 qwen2.5, 1.9e-4 llama3.2, 8.6e-4 linear rope, 3.3e-3 geglu (gate 4e-3; it was 1e-3 on the fp32 path), 4.6e-2 tinymistral f16, Q8 max abs 2.27 mean 0.057 top-1 43/43 (now gated at max abs 2.5, mean 0.07, top-1 100%).

### Batch decode (issue #247)

`Runtime::load_batched(spec, max_ctx, n)` and `generate_batch(prompts, new_tokens)`: N sequences share the weights, each has its own KV region, and every decode step runs the linears as one tensor-core GEMM over N rows, captured as a CUDA graph. Greedy output matches N single-sequence runs token for token (`batch_decode_matches_single_sequence_greedy`). Aggregate decode tok/s, 128-token prompts, 128 new tokens, three alternating rounds against `llama-batched-bench -npp 128 -ntg 128 -npl 1,4,16` on the same GGUFs (vLLM 95% busy for most of the run, round medians):

| model | batch | specrt | llama.cpp |
|---|---|---|---|
| Qwen3-0.6B | 1 | 100.7 | 161.8 |
| Qwen3-0.6B | 4 | 392.2 | 519.8 |
| Qwen3-0.6B | 16 | 1172.9 | 1687.6 |
| Qwen2.5-0.5B | 1 | 118.4 | 192.0 |
| Qwen2.5-0.5B | 4 | 472.3 | 709.3 |
| Qwen2.5-0.5B | 16 | 1724.8 | 2501.8 |

Batch 1 here goes through the GEMM path and is slower than the GEMV decode (`generate`, 154 and 195 tok/s), which stays the single-stream path. At batch 16 the step is attention (18%, the paged decode kernel launched per sequence) and lm_head (16%), then the 16-row GEMMs. vLLM was not measured: the resident vLLM serves a 27B model on this GPU and a second vLLM instance does not fit beside it.
