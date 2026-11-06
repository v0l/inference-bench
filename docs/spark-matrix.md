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
