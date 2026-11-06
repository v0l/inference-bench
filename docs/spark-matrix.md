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
