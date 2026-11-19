# Perf scoreboard

Ratios are specrt / best competitor, above 1.00 means we are faster. Rows are replaced in place; the command is beside each row.














| family | model | ours decode b1 | ours prefill | ours decode b8 | best competitor | their decode b1 | their prefill | their decode b8 | decode ratio | prefill ratio | b8 ratio | date | commit | command |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3 | `gemma-3-1b-it` | 95.7 | 11336.0 | 646.9 | llama.cpp/vllm | 93.8 | 25413.1 | 680.9 | 1.02 | 0.45 | 0.95 | 2026-11-19 | 82ce823 | `crates/specrt/scripts/perfpass.sh gemma3 --model /home/kieran/ib-gates/803/models/gemma-3-1b-it --host spark` |
| gguf-q4 | `unsloth/Qwen3-0.6B-GGUF` | 226.4 | 18590.0 | 651.6 | llama.cpp | 320.4 | 20086.0 | 1400.3 | 0.71 | 0.93 | 0.47 | 2026-11-19 | 0338df9 | `crates/specrt/scripts/perfpass.sh gguf-q4 --model /home/kieran/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_K_M.gguf --host spark` |
| granite | `granite-3.3-2b-base` | 17.4 | 2011.1 | 298.4 | llama.cpp/vllm | 39.9 | 11437.3 | 295.5 | 0.44 | 0.18 | 1.01 | 2026-11-19 | 0338df9 | `crates/specrt/scripts/perfpass.sh granite --model /home/kieran/ib-gates/803/models/granite-3.3-2b-base --host spark` |
| llama | `9535bd9b1d1dea6acafbdc4813b728796aeb28da` | 80.2 | 11183.1 | 634.4 | llama.cpp/vllm | 82.1 | 23607.8 | 615.3 | 0.98 | 0.47 | 1.03 | 2026-11-19 | 82ce823 | `crates/specrt/scripts/perfpass.sh llama --model /home/kieran/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da --host spark` |
| mistral | `Locutusque/TinyMistral-248M` | 421.9 | 63961.4 | 3015.6 | vllm | 307.5 | 71475.5 | 2955.5 | 1.37 | 0.89 | 1.02 | 2026-11-19 | 82ce823 | `crates/specrt/scripts/perfpass.sh mistral --model /home/kieran/.cache/huggingface/hub/models--Locutusque--TinyMistral-248M/snapshots/e84fb52945d42de4f2361b6047decdc65426dc2b --host spark` |
| olmo2 | `OLMo-2-0425-1B` | 70.0 | 9803.5 | 566.5 | llama.cpp | 42.2 | 6858.1 | 409.2 | 1.66 | 1.43 | 1.38 | 2026-11-19 | 82ce823 | `crates/specrt/scripts/perfpass.sh olmo2 --model /home/kieran/ib-gates/803/models/OLMo-2-0425-1B --host spark` |
| phi3 | `Phi-3-mini-4k-instruct` | 23.3 | 2937.3 | 198.5 | llama.cpp | 27.7 | 5786.9 | 185.9 | 0.84 | 0.51 | 1.07 | 2026-11-19 | 82ce823 | `crates/specrt/scripts/perfpass.sh phi3 --model /home/kieran/ib-gates/803/models/Phi-3-mini-4k-instruct --host spark` |
| qwen2 | `060db6499f32faf8b98477b0a26969ef7d8b9987` | 191.3 | 31007.0 | 1350.0 | llama.cpp/vllm | 195.5 | 43384.1 | 1338.6 | 0.98 | 0.71 | 1.01 | 2026-11-19 | 5c53f27 | `crates/specrt/scripts/perfpass.sh qwen2 --model /home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 --host spark` |
| qwen3 | `qwen3-0.6b.toml` | 140.0 | 21997.5 | 1081.9 | llama.cpp | 158.9 | 23136.2 | 994.4 | 0.88 | 0.95 | 1.09 | 2026-11-19 | 5c53f27 | `crates/specrt/scripts/perfpass.sh qwen3 --model /home/kieran/ib-gates/803/src/crates/specrt/specs/qwen3-0.6b.toml --host spark` |
| qwen3_5 | `Qwen3.5-9B` | 12.8 | 1404.4 | 176.1 | llama.cpp | 13.0 | 2580.1 | 72.0 | 0.99 | 0.54 | 2.45 | 2026-11-19 | 0338df9 | `crates/specrt/scripts/perfpass.sh qwen3_5 --model /home/kieran/models/Qwen3.5-9B --host spark` |
| smollm3 | `SmolLM3-3B` | 31.9 | 4068.0 | 249.0 | llama.cpp/vllm | 33.9 | 9869.3 | 249.8 | 0.94 | 0.41 | 1.00 | 2026-11-19 | 82ce823 | `crates/specrt/scripts/perfpass.sh smollm3 --model /home/kieran/ib-gates/803/models/SmolLM3-3B --host spark` |
