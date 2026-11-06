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
overhead. specrt prefills one token at a time through its decode path.

Logit diff of specrt against `transformers` fp32 on the fixed 43-token prompt
(`inference-bench logit-diff --backend specrt --model Qwen/Qwen3-0.6B`): max abs 1.0e-4,
mean abs 6.5e-6, top-1 agreement 43/43.
