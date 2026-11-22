# Perf scoreboard

Ratios are specrt / best competitor, above 1.00 means we are faster. Rows are replaced in place; the command is beside each row.

| family | model | ours decode b1 | ours prefill | ours decode b8 | best competitor | their decode b1 | their prefill | their decode b8 | decode ratio | prefill ratio | b8 ratio | date | commit | command |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3 | `gemma-3-1b-it` | 100.4 | 18408.3 | 670.5 | llama.cpp/vllm | 101.5 | 26430.8 | 743.6 | 0.99 | 0.70 | 0.90 | 2026-11-19 | 9f648cd | `crates/specrt/scripts/perfpass.sh gemma3 --model /home/kieran/ib-gates/821/models/gemma-3-1b-it --host spark` |
| gguf-q4 | `unsloth/Qwen3-0.6B-GGUF` | 233.2 | 22223.3 | 1544.3 | llama.cpp | 331.6 | 21276.5 | 1462.5 | 0.70 | 1.04 | 1.06 | 2026-11-19 | 3f76803 | `crates/specrt/scripts/perfpass.sh gguf-q4 --model /home/kieran/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_K_M.gguf --host spark` |
| gguf-q8 | `Qwen3-0.6B-Q8_0.gguf` | 204.2 | 23534.0 | 882.4 | llama.cpp | 236.5 | 20147.6 | 1252.9 | 0.86 | 1.17 | 0.70 | 2026-11-19 | d29bb13 | `crates/specrt/scripts/perfpass.sh gguf-q8 --model /home/kieran/ib-gates/803/models/Qwen3-0.6B-Q8_0.gguf --host spark` |
| granite | `granite-3.3-2b-base` | 39.1 | 6684.0 | 312.2 | llama.cpp/vllm | 41.0 | 9461.6 | 299.7 | 0.95 | 0.71 | 1.04 | 2026-11-19 | 9f648cd | `crates/specrt/scripts/perfpass.sh granite --model /home/kieran/ib-gates/821/models/granite-3.3-2b-base --host spark` |
| llama | `unsloth/Llama-3.2-1B` | 82.1 | 16615.7 | 649.7 | llama.cpp/vllm | 84.2 | 24239.2 | 609.6 | 0.97 | 0.69 | 1.07 | 2026-11-19 | 9f648cd | `crates/specrt/scripts/perfpass.sh llama --model /home/kieran/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da --host spark` |
| mistral | `Locutusque/TinyMistral-248M` | 421.9 | 63961.4 | 3015.6 | vllm | 307.5 | 71475.5 | 2955.5 | 1.37 | 0.89 | 1.02 | 2026-11-19 | 82ce823 | `crates/specrt/scripts/perfpass.sh mistral --model /home/kieran/.cache/huggingface/hub/models--Locutusque--TinyMistral-248M/snapshots/e84fb52945d42de4f2361b6047decdc65426dc2b --host spark` |
| olmo2 | `OLMo-2-0425-1B` | 70.0 | 9803.5 | 566.5 | llama.cpp | 42.2 | 6858.1 | 409.2 | 1.66 | 1.43 | 1.38 | 2026-11-19 | 82ce823 | `crates/specrt/scripts/perfpass.sh olmo2 --model /home/kieran/ib-gates/803/models/OLMo-2-0425-1B --host spark` |
| phi3 | `Phi-3-mini-4k-instruct` | 25.1 | 4610.8 | 203.0 | llama.cpp | 28.7 | 5763.5 | 191.4 | 0.87 | 0.80 | 1.06 | 2026-11-19 | 9f648cd | `crates/specrt/scripts/perfpass.sh phi3 --model /home/kieran/ib-gates/821/models/Phi-3-mini-4k-instruct --host spark` |
| qwen2 | `060db6499f32faf8b98477b0a26969ef7d8b9987` | 191.3 | 31007.0 | 1350.0 | llama.cpp/vllm | 195.5 | 43384.1 | 1338.6 | 0.98 | 0.71 | 1.01 | 2026-11-19 | 5c53f27 | `crates/specrt/scripts/perfpass.sh qwen2 --model /home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 --host spark` |
| qwen3 | `qwen3-0.6b.toml` | 140.0 | 21997.5 | 1081.9 | llama.cpp | 158.9 | 23136.2 | 994.4 | 0.88 | 0.95 | 1.09 | 2026-11-19 | 5c53f27 | `crates/specrt/scripts/perfpass.sh qwen3 --model /home/kieran/ib-gates/803/src/crates/specrt/specs/qwen3-0.6b.toml --host spark` |
| qwen3_5 | `Qwen3.5-9B` | 12.5 | 2180.0 | 180.4 | llama.cpp | 12.9 | 2457.2 | 69.6 | 0.97 | 0.89 | 2.59 | 2026-11-19 | d9d799c | `crates/specrt/scripts/perfpass.sh qwen3_5 --model /home/kieran/models/Qwen3.5-9B --host spark` |
| qwen3_moe | `Qwen3-30B-A3B-FP8` | 55.3 | 3306.8 | 185.9 | vllm | 40.3 | 3959.2 | 133.3 | 1.30 | 1.37 | 0.84 | 1.39 | 2026-11-20 | issue885 |
| qwen3_next | `Qwen3-Next-80B-A3B-Instruct-NVFP4` | 38.3 | 2045.0 | 141.9 | vLLM and SGLang: not run, the 80B doesn't fit beside the resident vLLM on the Spark | n/a | n/a | n/a | n/a | n/a | n/a | 2026-11-19 | 895 | `crates/specrt/scripts/perfpass.sh qwen3_next --model /home/kieran/ib-gates/817/models/Qwen3-Next-80B-A3B-Instruct-NVFP4 --host spark` |
| smollm3 | `SmolLM3-3B` | 33.4 | 5808.3 | 264.3 | llama.cpp/vllm | 35.2 | 9055.0 | 259.4 | 0.95 | 0.64 | 1.02 | 2026-11-19 | 9f648cd | `crates/specrt/scripts/perfpass.sh smollm3 --model /home/kieran/ib-gates/821/models/SmolLM3-3B --host spark` |

## SGLang

Indicative only: SGLang ran with the radix cache off at batch 8, median of 3, while another job shared the Spark GPU. Ratios are specrt / SGLang against the row above. `triton` marks runs where the default flashinfer backend failed.

| family | sglang prefill | sglang decode b1 | sglang decode b8 | prefill ratio | decode ratio | b8 ratio | backend | date |
|---|---|---|---|---|---|---|---|---|
| gemma3 | 19840.3 | 66.5 | 552.0 | 0.93 | 1.51 | 1.21 | flashinfer | 2026-11-19 |
| granite | 8826.1 | 28.7 | 216.8 | 0.76 | 1.36 | 1.44 | flashinfer | 2026-11-19 |
| llama | 24568.4 | 63.0 | 602.7 | 0.68 | 1.30 | 1.08 | flashinfer | 2026-11-19 |
| mistral | 73967.2 | 308.5 | 2627.5 | 0.86 | 1.37 | 1.15 | triton | 2026-11-19 |
| olmo2 | 19164.4 | 57.5 | 534.5 | 0.51 | 1.22 | 1.06 | triton | 2026-11-19 |
| phi3 | 8075.3 | 22.1 | 181.0 | 0.57 | 1.14 | 1.12 | triton | 2026-11-19 |
| qwen2 | 39370.1 | 127.5 | 1139.1 | 0.79 | 1.50 | 1.19 | flashinfer | 2026-11-19 |
| qwen3 | 31672.8 | 92.6 | 778.9 | 0.69 | 1.51 | 1.39 | flashinfer | 2026-11-19 |
| qwen3_5 | 4287.8 | 11.1 | 71.1 | 0.51 | 1.13 | 2.54 | flashinfer | 2026-11-19 |
| qwen3_moe | 5114.2 | 53.0 | 221.8 | 0.06 | 0.99 | 0.72 | flashinfer | 2026-11-19 |
| smollm3 | 6927.9 | 21.1 | 195.4 | 0.84 | 1.59 | 1.35 | flashinfer | 2026-11-19 |

## Paired A/B claims

The single list of current keep/drop claims. Each row has its command, date, card state and verdict. Retract a claim in place (verdict `retracted`, with the reason). Speeds in older docs (`docs/specrt-mtp.md` and the other `docs/specrt-*.md`) are single runs or medians of 6 on a shared card and are indicative until a row here replaces them. `perfpass_report.py` leaves this section untouched.

Use `bench --ab` for every keep/drop decision:

```bash
bench --ab [--rounds N] [--b-graphs on|off] <A spec> [B spec] [prompt_len] [new_tokens]
```

- Both arms load in one process and alternate A/B/A/B. 20 rounds per pass by default (`--rounds`), two passes. Run counts are flags (`--runs`, `--rounds`); `SPECRT_BENCH_RUNS` is gone.
- Each pass is preceded by an A/A control (A against itself). The statistic is the median of per-round ratios B/A with its IQR.
- The bench refuses instead of printing a verdict, and exits non-zero, when a control median is outside 1 +/- 0.02, any control or A/B spread (IQR/median) is above 0.10, the two A/B passes differ by more than 3%, or a pass has fewer than 20 rounds. A noisy card gives `REFUSE (reason)`; rerun on a quieter card.
- Otherwise the verdict is `PASS: B faster`, `PASS: B slower` or `PASS: no difference outside the A/A band`.
- The bandwidth probe (`crates/specrt/scripts/bandwidth.cu`) runs in the same process. The run prints weight bytes per token and decode GB/s as a fraction of the read wall. Small models can exceed 100% because L2 serves part of the reads.
- A second spec path makes B a different spec; `--b-graphs off` makes B the same spec with `[runtime] graphs = false`.
- Prefill ratios are printed too but are noisy at this size; judge on decode unless the change targets prefill.

| claim | value | command | date | card state | verdict |
|---|---|---|---|---|---|
| A/A control, Qwen3-0.6B bf16 decode | ratio 0.9996 and 1.0007 (spread 1.1% and 4.7%) | `gpu-run any -- bench --ab crates/specrt/specs/qwen3-0.6b.toml` (one spec path: B is the same spec) | 2026-11-21 | Spark GB10, other jobs resident, 1% util | pass, within 0.98 to 1.02 |
| CUDA graph decode vs no graphs, Qwen3-0.6B bf16 | B (graphs off) 0.93x of A, spread 0.6% to 2.2% | `gpu-run any -- bench --ab --b-graphs off crates/specrt/specs/qwen3-0.6b.toml` | 2026-11-21 | Spark GB10, shared with resident vLLM | PASS: B slower (detected) |
| Qwen3-0.6B bf16 decode b1 | 175.2 tok/s, 1.50 GB per token, 263 GB/s, 112% of probe read wall 236 GB/s | same command, arm A | 2026-11-21 | Spark GB10, shared with resident vLLM | indicative, wall includes L2 hits |
