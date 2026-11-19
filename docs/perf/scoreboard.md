# Perf scoreboard

Ratios are specrt / best competitor, above 1.00 means we are faster. Rows are replaced in place; the command is beside each row.
























| family | model | ours decode b1 | ours prefill | ours decode b8 | best competitor | their decode b1 | their prefill | their decode b8 | decode ratio | prefill ratio | b8 ratio | date | commit | command |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3 | `gemma-3-1b-it` | 92.2 | 15247.9 | 647.6 | llama.cpp/vllm | 96.6 | 21372.5 | 719.3 | 0.95 | 0.71 | 0.90 | 2026-11-19 | 811f24f | `crates/specrt/scripts/perfpass.sh gemma3 --model /home/kieran/ib-gates/813/models/gemma-3-1b-it --host spark` |
| gguf-q4 | `unsloth/Qwen3-0.6B-GGUF` | 226.4 | 18590.0 | 651.6 | llama.cpp | 320.4 | 20086.0 | 1400.3 | 0.71 | 0.93 | 0.47 | 2026-11-19 | 0338df9 | `crates/specrt/scripts/perfpass.sh gguf-q4 --model /home/kieran/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_K_M.gguf --host spark` |
| granite | `granite-3.3-2b-base` | 37.6 | 5405.1 | 301.4 | llama.cpp/vllm | 39.5 | 10125.2 | 290.6 | 0.95 | 0.53 | 1.04 | 2026-11-19 | 811f24f | `crates/specrt/scripts/perfpass.sh granite --model /home/kieran/ib-gates/813/models/granite-3.3-2b-base --host spark` |
| llama | `unsloth/Llama-3.2-1B` | 75.7 | 14327.0 | 621.4 | vllm | 63.7 | 25374.9 | 604.2 | 1.19 | 0.56 | 1.03 | 2026-11-19 | 811f24f | `crates/specrt/scripts/perfpass.sh llama --model /home/kieran/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da --host spark` |
| mistral | `Locutusque/TinyMistral-248M` | 421.9 | 63961.4 | 3015.6 | vllm | 307.5 | 71475.5 | 2955.5 | 1.37 | 0.89 | 1.02 | 2026-11-19 | 82ce823 | `crates/specrt/scripts/perfpass.sh mistral --model /home/kieran/.cache/huggingface/hub/models--Locutusque--TinyMistral-248M/snapshots/e84fb52945d42de4f2361b6047decdc65426dc2b --host spark` |
| olmo2 | `OLMo-2-0425-1B` | 70.0 | 9803.5 | 566.5 | llama.cpp | 42.2 | 6858.1 | 409.2 | 1.66 | 1.43 | 1.38 | 2026-11-19 | 82ce823 | `crates/specrt/scripts/perfpass.sh olmo2 --model /home/kieran/ib-gates/803/models/OLMo-2-0425-1B --host spark` |
| phi3 | `Phi-3-mini-4k-instruct` | 24.6 | 3807.3 | 199.3 | llama.cpp | 27.5 | 5683.3 | 183.2 | 0.89 | 0.67 | 1.09 | 2026-11-19 | 811f24f | `crates/specrt/scripts/perfpass.sh phi3 --model /home/kieran/ib-gates/813/models/Phi-3-mini-4k-instruct --host spark` |
| qwen2 | `060db6499f32faf8b98477b0a26969ef7d8b9987` | 191.3 | 31007.0 | 1350.0 | llama.cpp/vllm | 195.5 | 43384.1 | 1338.6 | 0.98 | 0.71 | 1.01 | 2026-11-19 | 5c53f27 | `crates/specrt/scripts/perfpass.sh qwen2 --model /home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 --host spark` |
| qwen3 | `qwen3-0.6b.toml` | 140.0 | 21997.5 | 1081.9 | llama.cpp | 158.9 | 23136.2 | 994.4 | 0.88 | 0.95 | 1.09 | 2026-11-19 | 5c53f27 | `crates/specrt/scripts/perfpass.sh qwen3 --model /home/kieran/ib-gates/803/src/crates/specrt/specs/qwen3-0.6b.toml --host spark` |
| qwen3_5 | `Qwen3.5-9B` | 12.8 | 1404.4 | 176.1 | llama.cpp | 13.0 | 2580.1 | 72.0 | 0.99 | 0.54 | 2.45 | 2026-11-19 | 0338df9 | `crates/specrt/scripts/perfpass.sh qwen3_5 --model /home/kieran/models/Qwen3.5-9B --host spark` |
| smollm3 | `SmolLM3-3B` | 31.8 | 4825.7 | 250.9 | llama.cpp/vllm | 33.9 | 9116.6 | 250.8 | 0.94 | 0.53 | 1.00 | 2026-11-19 | 811f24f | `crates/specrt/scripts/perfpass.sh smollm3 --model /home/kieran/ib-gates/813/models/SmolLM3-3B --host spark` |
