# Perf scoreboard

Ratios are specrt / best competitor, above 1.00 means we are faster. Rows are replaced in place; the command is beside each row.




| family | model | ours decode b1 | ours prefill | ours decode b8 | best competitor | their decode b1 | their prefill | their decode b8 | decode ratio | prefill ratio | b8 ratio | date | commit | command |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3 | `qwen3-0.6b.toml` | 140.0 | 21997.5 | 1081.9 | llama.cpp | 158.9 | 23136.2 | 994.4 | 0.88 | 0.95 | 1.09 | 2026-11-19 | 5c53f27 | `crates/specrt/scripts/perfpass.sh qwen3 --model /home/kieran/ib-gates/803/src/crates/specrt/specs/qwen3-0.6b.toml --host spark` |
