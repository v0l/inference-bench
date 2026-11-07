# Spark matrix

Historical record: llama.cpp and vLLM were measured with their own servers at the time, through the old `--backend` and `--worker` flags; today they are reached with `inference-bench bench --target openai:URL` and the scripts below need that change to rerun.

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

Batch 1 here goes through the GEMM path and is slower than the GEMV decode (`generate`, 154 and 195 tok/s), which stays the single-stream path. At batch 16 the step is attention (18%, the paged decode kernel launched per sequence) and lm_head (16%), then the 16-row GEMMs. vLLM was measured later, see the next section.

## vLLM, llama.cpp and specrt (issue #265)

Spark GB10, 2026-11-06, one session, bf16 weights for all three. vLLM 0.32.0 ran as a second instance on port 18000 beside the resident 27B (`--dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.08 --max-num-seqs 16 --no-enable-prefix-caching`, CUDA graphs on). llama.cpp `7bd8282c3` ran as `llama-server -np 16 -c 24576 --no-cache-prompt` on port 18080 with the GGUFs converted from the same safetensors, plus `llama-bench` and `llama-batched-bench`. specrt ran through its worker and through `bench`. The 27B was not stopped. Also on the GPU for the whole run: the 27B vLLM (49.9 GB), ComfyUI (12.9 GB, idle), Frigate (detector and three ffmpeg decoders, about 1.5 GB). CUDA free memory was 18-22 GiB at the start of each round after `drop_caches`; without the cache drop the page cache ate it and CUDA context creation failed.

Six rounds per model, the engine order rotated each round, medians. The 27B had a request running in 121 of the 994 sampled seconds, all inside llama.cpp and specrt rounds and never in a vLLM round; the worst were Qwen2.5 llama.cpp round 4 (87% of it, `llama-bench` decode fell to 38 tok/s) and specrt round 2 (50%). Medians of six keep those out. Every completion is 128 tokens (`ignore_eos`), prompts are the synthetic bench prompt (`--prompt-len`). Driver: `docs/spark-run/vllm-run.sh`, aggregation `docs/spark-run/summarise.py`, raw files and `summary.json` in `docs/spark-run/vllm-20261106/`.

Over HTTP (`inference-bench run`, streaming; prefill is prompt tokens over TTFT, decode is the remaining 127 tokens over the time after the first):

| model | prompt | engine | prefill tok/s | decode tok/s | TTFT |
|---|---|---|---|---|---|
| Qwen3-0.6B | 128 | vLLM | 7423 | 127.1 | 18.8 ms |
| Qwen3-0.6B | 128 | llama.cpp | 8334 | 163.9 | 15.5 ms |
| Qwen3-0.6B | 128 | specrt | 10485 | 156.1 | 13.4 ms |
| Qwen3-0.6B | 1024 | vLLM | 32621 | 120.7 | 31.4 ms |
| Qwen3-0.6B | 1024 | llama.cpp | 19913 | 150.6 | 51.5 ms |
| Qwen3-0.6B | 1024 | specrt | 15587 | 101.3 | 66.5 ms |
| Qwen2.5-0.5B | 128 | vLLM | 5787 | 149.6 | 22.3 ms |
| Qwen2.5-0.5B | 128 | llama.cpp | 10500 | 195.1 | 12.3 ms |
| Qwen2.5-0.5B | 128 | specrt | 12759 | 191.2 | 11.6 ms |
| Qwen2.5-0.5B | 1024 | vLLM | 36159 | 139.4 | 28.4 ms |
| Qwen2.5-0.5B | 1024 | llama.cpp | 22903 | 187.2 | 45.0 ms |
| Qwen2.5-0.5B | 1024 | specrt | 20491 | 146.6 | 51.0 ms |

In-process, no HTTP (`llama-bench -p 128,1024 -n 128 -r 1`, `bench specs/<spec>.toml <prompt> 128`):

| model | metric | llama.cpp | specrt | specrt / llama.cpp |
|---|---|---|---|---|
| Qwen3-0.6B | prefill 128 | 8927 | 10801 | 1.21 |
| Qwen3-0.6B | prefill 1024 | 24790 | 21104 | 0.85 |
| Qwen3-0.6B | decode 128 | 151.6 | 156.4 | 1.03 |
| Qwen2.5-0.5B | prefill 128 | 6286 | 11165 | 1.78 |
| Qwen2.5-0.5B | prefill 1024 | 29213 | 28694 | 0.98 |
| Qwen2.5-0.5B | decode 128 | 203.5 | 191.8 | 0.94 |

llama-bench prefill at 128 tokens was noisy this session (Qwen2.5 1932-6615 over the six rounds), so the 128-token ratios above flatter specrt; the HTTP table is the steadier comparison there.

Batch decode, 128-token prompts, 128 new tokens each, aggregate decode tok/s (sum over sequences of 127 tokens over each one's decode time). vLLM and llama-server get N concurrent streaming requests (`inference-bench batch --sizes 1,4,16`); the native columns are `llama-batched-bench -npp 128 -ntg 128 -npl 1,4,16` (`speed_tg`) and specrt `bench <spec> 128 128 <N>`:

| model | batch | vLLM (HTTP) | llama-server (HTTP) | llama-batched-bench | specrt `bench` | specrt / vLLM | specrt / llama.cpp native |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 1 | 125.0 | 163.3 | 149.8 | 101.5 | 0.81 | 0.68 |
| Qwen3-0.6B | 4 | 551.8 | 519.0 | 457.1 | 372.9 | 0.68 | 0.82 |
| Qwen3-0.6B | 16 | 1883.1 | 1255.2 | 1682.1 | 1220.3 | 0.65 | 0.73 |
| Qwen2.5-0.5B | 1 | 159.8 | 192.1 | 199.4 | 118.4 | 0.74 | 0.59 |
| Qwen2.5-0.5B | 4 | 736.9 | 624.5 | 739.1 | 475.6 | 0.65 | 0.64 |
| Qwen2.5-0.5B | 16 | 2852.9 | 1771.6 | 2722.1 | 1744.1 | 0.61 | 0.64 |

Median TTFT under load at batch 16: vLLM 73.5 and 52.5 ms, llama-server 106.8 and 85.5 ms.

Where specrt stands:

- Single-stream decode on short context is level with llama.cpp (0.94-1.03 native, 0.95-0.98 over HTTP) and 1.2-1.3x vLLM.
- Prefill is ahead at 128 tokens and 0.85-0.98 of llama.cpp at 1024 tokens. vLLM is the fastest prefill at 1024 tokens by a wide margin: 32.6k and 36.2k tok/s over HTTP, 1.8-2.1x specrt over HTTP and 1.3-1.5x specrt's in-process number.
- Decode after a 1024-token prompt drops to 101 and 147 tok/s against llama.cpp's 151 and 187 (0.67 and 0.78). The decode attention kernel's cost grows with context much faster than llama.cpp's or vLLM's (vLLM loses 5-7% from 128 to 1024, specrt 23-35%).
- Batch decode is 0.59-0.82 of llama-batched-bench and 0.61-0.81 of vLLM, and the gap widens with batch size against vLLM (0.65 and 0.61 at 16). Batch 1 through the GEMM path is 101 and 118 tok/s against 156 and 192 on the GEMV path.

Follow-ups: batched paged decode attention (#272), decode at long context (#273), 1024-token prefill against vLLM and worker overhead (#274).

Harness changes from this run: bench completions send `ignore_eos` so every engine decodes exactly `max_tokens`, `inference-bench run --prompt-len N` uses the synthetic prompt, `inference-bench batch` runs N concurrent requests per size against any backend, and the worker backends get the raw prompt on `generate` (they applied the chat template, which OpenAI completions do not). vLLM needs `--no-enable-prefix-caching` for repeated-prompt benches, or TTFT drops to the cache hit. The Bench tab already runs vLLM as an endpoint (Settings, backend endpoints: URL, API key; the model id is the HF repo).

## Decode at long context (issue #273)

Split-K decode attention: `attention_split` runs 64 chunks per head (16 timesteps minimum) and `attention_reduce` merges them. The V pass reads float4 per lane and the four warps' partial sums are folded in shared memory.

Spark GB10, 2026-11-07, `docs/spark-run/decode-ctx-run.sh`: six rounds per model, specrt and llama-server (`-np 1`, bf16 GGUF) alternating order each round, `drop_caches` before every engine run, `inference-bench run --prompt-len N --max-tokens 128`. Raw files in `docs/spark-run/decode-ctx-20261107/`.

The GPU showed 95-96% utilisation from other tenants (the 27B vLLM, ComfyUI, Frigate) in every sample, and the load was heavy enough that both engines ran in two clusters, a slow one at 20-90 tok/s and a fast one. Medians of six land in the slow cluster for both engines (specrt 65 and 86 tok/s at 1024 for Qwen3 and Qwen2.5, llama.cpp 34 and 40) so they say nothing about the kernels. Per-round decode tok/s, (llama.cpp, specrt):

| model | prompt | rounds 1-6 |
|---|---|---|
| Qwen3-0.6B | 128 | (21,74) (36,71) (36,71) (139,65) (138,143) (36,76) |
| Qwen3-0.6B | 1024 | (19,65) (33,40) (32,58) (122,91) (125,65) (36,144) |
| Qwen2.5-0.5B | 128 | (38,85) (39,78) (39,198) (35,91) (195,91) (38,90) |
| Qwen2.5-0.5B | 1024 | (42,84) (39,82) (39,88) (39,190) (82,85) (41,86) |

Fast-cluster ceiling at 1024 tokens (best round of six): Qwen3-0.6B specrt 144 against llama.cpp 125 in this session (150.6 in #265); Qwen2.5-0.5B specrt 190 against llama.cpp 82 (no clean llama.cpp round, 187.2 in #265). Against the #265 llama.cpp figures that is 0.96 and 1.01, inside 10% on both models, but only the best rounds are clean, so this is a ceiling and not a median. A median inside 10% needs an idle Spark.

Idle local RTX PRO 6000 (Blackwell, compute_120), `bench specs/<spec>.toml <prompt> 128`, median of 5 per run, three runs:

| model | prompt | before #273 | split-K | + float4 V |
|---|---|---|---|---|
| Qwen3-0.6B | 128 | 442-445 | 445 | 451-453 |
| Qwen3-0.6B | 1024 | 93-112 | 353-355 | 375-383 |
| Qwen2.5-0.5B | 128 | 503 | 502-504 | 500-513 |
| Qwen2.5-0.5B | 1024 | 212 | 433-437 | 464-470 (fast runs) |

`nsys` at 1024 context, Qwen3-0.6B, local GPU (CUDA graph node tracing, per layer instance 17976 launches over prefill-free decode):

| kernel | split-K avg / med ns | + float4 V avg / med ns |
|---|---|---|
| attention_split | 24511 / 13536 | 16777 / 9280 |
| attention_reduce | 7700 / 4384 | 6916 / 4384 |

Tried and dropped, all measured locally at 1024 context:

- Chunk size: 8 timesteps with 64 splits was level with 16; 32 gave 310 and 64 gave 253 tok/s on Qwen3. 128 splits of 8 gave no gain (365).
- f16 KV cache (rounding on write): max abs logit error 0.012 on Qwen3 and 0.020 on Qwen2.5 against the 1e-3 bound, so rejected. bf16 was 0.064 and 0.13 earlier.
- Fusing `attention_reduce` into the o_proj input load: attention_reduce is 6.7% of decode and its launch is hidden by the graph, so there is little to fuse; not done.

Batch 1 here goes through the GEMM path and is slower than the GEMV decode (`generate`, 154 and 195 tok/s), which stays the single-stream path. At batch 16 the step is attention (18%, the paged decode kernel launched per sequence) and lm_head (16%), then the 16-row GEMMs. vLLM was not measured: the resident vLLM serves a 27B model on this GPU and a second vLLM instance does not fit beside it.

## Batched paged decode attention (issue #272)

The batch decode step used the single-sequence `attention` kernel, one launch per sequence. `batch_decode_attention` now covers every sequence in one launch (grid kv head x sequence x 8 timestep splits, 8 warps per block, 8 keys in flight per warp, online softmax per warp) and `batch_decode_combine` merges the splits. The 16-row GEMM tile (`gemm_tc`, BM 16, bf16) loads 128 k values per stage instead of 32 when K divides by 128, which cuts the barrier count in lm_head and the decode GEMMs. Both are gated by `batch_decode_matches_single_sequence_greedy` and the qwen3, qwen2.5 and q8 logit tests.

Profile at batch 16 (nsys, graph nodes): attention was 24% of the step at 117 us per launch before the kernel change; lm_head 16% at 1.89 ms (bandwidth bound, 311 MB). The GEMM BK change gave 3-5% in paired runs; stage count 3 and a smaller key unroll were not better.

Same session, alternating order rotated per round, `drop_caches` before every run, 6 rounds, medians of aggregate decode tok/s, 128-token prompts and 128 new tokens. The 27B vLLM was serving at 94-96% GPU utilisation for the whole session (it was not idle in the #265 run), so every absolute number is roughly 2.5x below the #265 table and only the ratios within this table are meaningful. vLLM is the HTTP number (`inference-bench batch`), llama.cpp `llama-batched-bench` `speed_tg`, specrt `bench <spec> 128 128 N`. Raw files, driver and summary script in `docs/spark-run/ib272-20261107/`.

| model | batch | vLLM | llama-batched-bench | specrt before | specrt now | now / llama.cpp | now / vLLM |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 4 | 210.0 | 131.3 | 182.6 | 225.8 | 1.72 | 1.08 |
| Qwen3-0.6B | 16 | 739.8 | 489.0 | 567.0 | 669.2 | 1.37 | 0.90 |
| Qwen2.5-0.5B | 4 | 265.5 | 150.6 | 225.8 | 293.8 | 1.95 | 1.11 |
| Qwen2.5-0.5B | 16 | 1044.2 | 651.7 | 816.3 | 1066.2 | 1.64 | 1.02 |

specrt gained 18-31% over the per-sequence kernel and is ahead of llama.cpp at both batch sizes, at the vLLM level except Qwen3-0.6B at batch 16 (0.90). The llama.cpp numbers fall further than specrt's under the contended GPU, so its ratio here flatters specrt against the idle-GPU #265 figures (llama-batched-bench 1682 and 2722 at batch 16, which specrt's idle-GPU gain from 1220 and 1744 has not been re-measured against).

## specrt worker overhead at 1024 tokens (issue #274)

The HTTP number sat at 0.71-0.74 of `bench` because the HTTP prompt is the synthetic 1024 tokens plus the chat template, which lands just over 1024 tokens and split into a 1024-token chunk plus a tiny second chunk. Prefill cost per chunk is a full pass over the weights, so 1040 tokens ran at 19.1k tok/s against 27.3k for 1024 (RTX PRO 6000, Qwen3-0.6B, `bench`). `PREFILL_CHUNK` is now 2048 (still bounded by the score budget and `max_ctx`), so a prompt of up to 2048 tokens is one pass.

Worker, through the JSON protocol with a 1044-token prompt (Qwen3-0.6B, RTX PRO 6000, medians of 7): worker prefill 59-62 ms before, 38 ms after; first streamed text 62-65 ms before, 41.5 ms after. Qwen2.5-0.5B after: 23-26 ms prefill, 27-30 ms first text. Tokenising and the first text decode add about 3.5 ms (under 10% of the prefill). The 1044 token prompt now runs at about 27k tok/s, matching `bench` at 1024.

The worker measurement above went through the old JSON worker (`inference-bench-specrt`), which issue #292 removes; the `PREFILL_CHUNK` change is in `runtime.rs` and does not depend on it. After the change the ignored GPU tests pass with `SPECRT_REF_DIR=/tmp/ref273` (logits 8/8, generate 4/4, jit 2/2, chat 3/3).

### Per-kernel prefill, vLLM against specrt, 1024 tokens (Spark GB10, Qwen3-0.6B bf16)

vLLM 0.32.0 under `nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi`, `--enforce-eager`, `--no-enable-prefix-caching`, one warm request set then one 1024-token completion (max_tokens 1) inside `/start_profile`..`/stop_profile`, `nsys stats --report cuda_gpu_kern_sum`. The request took 43 ms end to end. specrt is `bench specs/qwen3-0.6b.toml 1024 1` under nsys, six prefills, totals divided by six. GPU time only; the GPU was shared with Frigate and ffmpeg.

| vLLM kernel | calls | total ms | share |
|---|---|---|---|
| cutlass_80 bf16 s16816 128x256 GEMM | 28 | 4.58 | 20.8% |
| cutlass_80 bf16 s16816 256x128 GEMM | 28 | 3.31 | 15.0% |
| flash_fwd_splitkv (FlashAttention 2) | 28 | 2.74 | 12.4% |
| nvjet_sm121 128x192x64 GEMM | 28 | 2.39 | 10.9% |
| gemvx (lm_head, one row) | 1 | 2.19 | 10.0% |
| fused_add_rms_norm | 56 | 1.70 | 7.7% |
| nvjet_sm121 32x96x64 GEMM | 28 | 1.50 | 6.8% |
| act_and_mul (silu) | 28 | 1.10 | 5.0% |
| rms_norm (qk norm) | 56 | 1.04 | 4.7% |
| reshape_and_cache_flash | 28 | 0.75 | 3.4% |
| rotary_embedding | 28 | 0.57 | 2.6% |
| total | | 22.0 | |

| specrt kernel | calls | total ms | share |
|---|---|---|---|
| gate_up_mma_big (norm + gate/up + silu*mul) | 28 | 13.25 | 26.9% |
| flash_mma (Q norm, RoPE, attention) | 28 | 12.20 | 24.7% |
| qkv_mma_big (norm + qkv) | 28 | 9.55 | 19.4% |
| down_mma_big | 28 | 6.43 | 13.0% |
| o_mma_big | 28 | 4.50 | 9.1% |
| lm_head_mma_tiny | 1 | 1.73 | 3.5% |
| kv_norm_rope | 28 | 1.39 | 2.8% |
| row_rms + embed | 57 + 1 | 0.29 | 0.6% |
| total | | 49.3 | |

The gap is the GEMMs and attention. vLLM's four GEMM families (cuBLAS cutlass and nvjet on bf16 activations) take 11.8 ms for the whole prefill against 33.7 ms for specrt's four fused mma GEMMs (qkv, o, gate_up, down; 2.9x), and its FlashAttention takes 2.7 ms against 12.2 ms for `flash_mma` (4.5x). The small kernels are not the problem: vLLM spends 5.2 ms on norm, RoPE, activation and cache write, specrt 1.7 ms because it fuses norm and activation into the GEMMs, and lm_head is 1.7 ms against 2.2 ms. So the remaining work is the GEMM main loop (f32 activations converted per tile, hi/lo split doubling mma count) and the attention kernel, not fusion. Note the specrt total under nsys is 49 ms (about 21k tok/s) on the Spark, against 22 ms of vLLM kernel time; vLLM's 31 ms TTFT adds about 9 ms of scheduling and sampling around its kernels.

### Closing the prefill kernel gap (issue #296)

Stage 1, GEMM main loop. The activations that feed a GEMM are now written once as two bf16 planes (hi and lo, the split that keeps the 1e-3 logit gate) by the producer instead of being loaded as f32 and split inside every GEMM tile: `rms_split` (RMSNorm, replaces `row_rms` plus the norm in the A-tile load), the gate_up epilogue (activation), `flash_mma` and `batch_decode_combine`. The GEMM loads A and W with `cp.async` into one dynamic shared-memory stage, the 16-byte chunks XOR-swizzled by row (no padding), and keeps a fresh fp32 partial per k step so the error stays at the old level. The mma wrappers are generated in `codegen.rs`, which also sets the shared-memory size per kernel.

Per-kernel, Qwen3-0.6B bf16, 1024-token prefill, Spark GB10, `nsys` totals over six prefills divided by six, base (e8db06c) and stage 1 run back to back:

| kernel | base ms | stage 1 ms |
|---|---|---|
| gate_up_mma_big | 12.59 | 10.17 |
| flash_mma | 11.65 | 11.62 |
| qkv_mma_big | 9.12 | 6.92 |
| down_mma_big | 6.24 | 5.46 |
| o_mma_big | 4.53 | 4.07 |
| lm_head_mma_tiny | 1.76 | 1.42 |
| kv_norm_rope | 1.40 | 1.40 |
| row_rms / rms_split + embed + argmax | 0.29 | 0.91 |
| total | 47.6 | 42.0 |

The four GEMMs go from 32.5 to 26.6 ms. ncu on the GB10: tensor pipe (hmma subpipe) active 37 to 54% in gate_up, 32 to 47% in qkv; the swizzle is what moved it (the unswizzled cp.async version was slower than base, with barrier and long scoreboard stalls from bank conflicts on the padded rows). What did not help on the GB10: 3 or 4 stages (level or slower), BK 64, 128x128 / 128x256 / 128x64 tiles (level or slower), M-fastest block order. Without the hi/lo split the GEMMs would be another 25% faster but logits miss the gate by 30x to 1000x (qwen3 0.09, geglu 0.99), and rounding activations to f16 instead of the split fails too (qwen3 0.011, geglu 0.10).

Logits on both GPUs (RTX PRO 6000 sm_120 and GB10 sm_121, `SPECRT_REF_DIR=/tmp/ref273`, all 8 pass): max abs 2.0e-4 qwen3, 6.0e-4 qwen2.5, 1.9e-4 llama3.2, 6.5e-4 linear rope, 3.4e-3 geglu, 4.6e-2 tinymistral, Q8 2.26 mean 0.058. Decode on the Spark, 6 alternating runs, medians: Qwen3-0.6B 128/128 batch 1 130.5 base, 134.1 stage 1 (the runs swing 114-167 with Frigate active); batch 16 aggregate 1459 base, 1637 stage 1.

Stage 2, attention. The KV cache now stores each row as an f16 hi plane and an f16 lo plane (same bytes as the f32 row it replaces), written once by `kv_norm_rope` / `qk_norm_rope`, so `flash_mma` loads K and V with swizzled `cp.async` (double buffered) instead of converting f32 per tile. Q and P are split into f16 hi/lo in registers, and each product runs three mma (hi x hi, hi x lo, lo x hi; lo x lo dropped). f16 hi/lo carries 22 mantissa bits against 16 for bf16 hi/lo, so the three-term f16 product is more precise than the old four-term bf16 one: geglu max abs 3.4e-3 to 1.3e-3. 128 query rows per block (8 warps) instead of 64, softmax in base 2 with `ex2.approx`, the Q RoPE once per pair with `sincosf`, and warps whose rows are all before a key tile skip it. The decode attention kernels read the same planes.

What was measured on the way (Spark, min of 4 alternating nsys runs): bf16 planes with all four terms 12.3 ms; f16 four terms 11.0; f16 three terms 11.0 at 4 warps, 8.9 at 8 warps; plus `ex2` and the RoPE rewrite 7.4. Dropping more terms is faster (one term 5.3 ms) but fails the gate (bf16 one term: qwen3 0.12, geglu 3.5; f16 two terms: qwen3 0.008-0.012, geglu 0.1-0.5). 64-key tiles do not fit the shared memory at 128 rows, 16-key tiles are slower (8.0), a third cp.async stage is level.

| kernel | base ms | stage 1 ms | stage 2 ms |
|---|---|---|---|
| flash_mma | 11.71 | 11.70 | 7.39 |
| gate_up_mma_big | 12.94 | 10.08 | 10.36 |
| qkv_mma_big | 9.23 | 7.20 | 7.19 |
| down_mma_big | 6.30 | 5.41 | 5.42 |
| o_mma_big | 4.49 | 4.11 | 4.02 |
| total | 48.2 | 42.4 | 38.4 |

(min of 4 alternating runs, Qwen3-0.6B 1024 tokens, Spark.) Logits after stage 2, both GPUs, all 8 pass: qwen3 1.9e-4, qwen2.5 6.1e-4, llama3.2 1.2e-4, linear rope 6.1e-4, geglu 1.3e-3, tinymistral 4.6e-2, Q8 2.27 mean 0.057.

Stage 3, GEMM accumulation: the split GEMMs sum hi and lo products into a per-k-tile partial and add it to the accumulator once per tile (instead of once per mma pair), qkv 7.1 to 6.5 ms and o 4.0 to 3.7 ms. Logits: qwen3 2.5e-4, qwen2.5 5.6e-4, llama3.2 1.1e-4, linear rope 5.6e-4, geglu 1.6e-3, all 8 pass on both GPUs.

Where the remaining GEMM time goes: dropping the hi/lo split from all linears takes the four GEMMs from 26.6 to 19.8 ms, but every single linear fails the gate on its own (max abs with only that linear unsplit, qwen3 / geglu: lm_head 0.037 / 0.070, o 0.030 / 0.21, down 0.073 / 0.31, qkv 0.063 / 0.48, gate_up 0.061 / 0.37). So against vLLM's 11.8 ms the GEMMs carry 2x the tensor work for the 1e-3 gate, and the unsplit kernel is still 1.7x off cuBLAS/CUTLASS. ncu on gate_up after stage 1: tensor pipe 52% active, 75 registers, 3 blocks per SM, stalls spread over math pipe throttle, long scoreboard and barrier.

Final, Spark GB10, all with the GPU shared with Frigate, ffmpeg and an idle ComfyUI (`nvidia-smi --query-compute-apps`: ggml-rpc-server, frigate.detector:onnx, three ffmpeg, ComfyUI python 12.9 GB). Per-kernel, Qwen3-0.6B bf16 1024-token prefill, min of 4 alternating nsys runs:

| kernel | base ms | final ms | vLLM ms |
|---|---|---|---|
| gate_up | 12.90 | 10.27 | |
| qkv | 9.19 | 6.49 | |
| down | 6.26 | 5.37 | |
| o | 4.43 | 3.68 | |
| four GEMMs | 32.8 | 25.8 | 11.8 |
| attention | 11.47 | 7.51 | 2.74 |
| total | 48.1 | 37.6 | 22.0 |

Prefill tok/s from `bench`, medians of 6 runs alternating base (e8db06c) and final, page cache dropped before each run:

| spec | prompt | base tok/s | final tok/s | ratio |
|---|---|---|---|---|
| qwen3-0.6b | 128 | 10707 | 11940 | 1.12 |
| qwen3-0.6b | 512 | 20447 | 24527 | 1.20 |
| qwen3-0.6b | 1024 | 20794 | 26803 | 1.29 |
| qwen2.5-0.5b | 128 | 11784 | 13633 | 1.16 |
| qwen2.5-0.5b | 512 | 25662 | 30976 | 1.21 |
| qwen2.5-0.5b | 1024 | 28142 | 34639 | 1.23 |
| qwen3-0.6b-q8 | 128 | 11542 | 12898 | 1.12 |
| qwen3-0.6b-q8 | 512 | 20425 | 25514 | 1.25 |
| qwen3-0.6b-q8 | 1024 | 20118 | 26532 | 1.32 |

Decode, Qwen3-0.6B 128-token prompt, 128 new, 6 alternating runs, medians: batch 1 167.2 base, 166.8 final; batch 4 aggregate 482 base, 565 final; batch 16 aggregate 1456 base, 1637 final. At a 2048-token context batch 1 decode is 124.0 base, 124.3 final. On the RTX PRO 6000 (shared with a vLLM process) the 1024 prefill goes 22.9k to 31.1k tok/s and decode is level (417 vs 416).

The 1024-token kernel time is 37.6 ms against the 26 ms target: not reached. The gap is the hi/lo split in the GEMMs (about 7 ms, needed for the gate) plus the remaining GEMM and attention kernel efficiency.

#### Issue #305 follow-up

Register double buffering of the GEMM fragments (ldmatrix for k step n+1 issued before the mma of step n, two fragment sets) was tried on the RTX PRO 6000 (nsys, same session, against the unmodified kernel): gate_up 6.9 vs 6.1 ms, qkv 4.55 vs 4.03, down 6.0 vs 6.4, o 2.83 vs 3.17 per six calls. Level or slower overall, so it is not kept; the compiler already hoists the fragment loads across the unrolled k steps. The remaining 11 ms to the 26 ms target needs either a looser logit gate (bf16 activations in the linears, about 7 ms; one-term attention, about 2 ms) or a different GEMM structure (persistent scheduler, larger warp tiles), neither of which fits the 1e-3 fp32 gate as it stands.

Decision: the gate stays at 1e-3 against the fp32 reference, so the hi/lo split stays and prefill is good enough at 37.6 ms. The persistent tile scheduler, larger warp tiles and one-term attention are parked: they need that gate changed or a bigger GEMM restructure.
