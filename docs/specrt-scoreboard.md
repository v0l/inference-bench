# specrt scoreboard

One table of current perf claims. Every row has the command that produced it, the date, the card state and a verdict. A claim that is retracted is edited in place (verdict `retracted`, with the reason), never deleted. Older numbers in other docs are indicative history; this table wins.

## Rules for a keep/drop decision

Use `bench --ab`, which loads both arms in one process and alternates them:

```bash
bench --ab [--rounds N] [--b-graphs on|off] <A spec> [B spec] [prompt_len] [new_tokens]
```

- Interleaved A/B rounds, 20 by default (`--rounds`), two passes. The run counts are flags, not env vars (`SPECRT_BENCH_RUNS` is gone, use `--runs`).
- Each pass first runs an A/A control (the same runtime against itself). The control median must be within 1 +/- 0.02 and its IQR/median at most 0.10, otherwise the run is refused.
- The statistic is the median of per-round ratios B/A with its IQR. A pass with IQR/median above 0.10 is refused. The two passes must agree within 3%.
- Verdict: `PASS: B faster`, `PASS: B slower`, `PASS: no difference outside the A/A band`, or `REFUSE (reason)`, which exits non-zero. Rerun refused runs on a quiet card.
- Bandwidth wall: the probe in `crates/specrt/scripts/bandwidth.cu` runs in the same process. The run prints weight bytes per token and decode GB/s as a fraction of the read wall. The fraction can exceed 100% for small models because the L2 cache serves part of the reads.
- A second spec path makes B a different spec; `--b-graphs off` makes B the same spec with `[runtime] graphs = false`.
- Prefill ratios are printed too, but prefill is noisy at this size; judge on decode unless the change targets prefill.

#401 and #404 use this for their keep/drop decision.

## Current claims

| claim | value | command | date | card state | verdict |
|---|---|---|---|---|---|
| A/A control, Qwen3-0.6B bf16 decode | ratio 0.9996, 1.0007 (IQR 1.1-4.7%) | `bench --ab crates/specrt/specs/qwen3-0.6b.toml` (second spec path or `--b-graphs`; same-arm control is built in) | 2026-11-21 | Spark GB10, other jobs resident (vLLM idle-ish, 1% util) | pass, within 0.98-1.02 |
| CUDA graph decode vs no graphs, Qwen3-0.6B bf16 | B (graphs off) 0.93x of A, IQR 0.6-2.2% | `gpu-run any -- bench --ab --b-graphs off crates/specrt/specs/qwen3-0.6b.toml` | 2026-11-21 | Spark GB10, shared with resident vLLM | PASS: B slower (detected) |
| Qwen3-0.6B bf16 decode b1 | 175.2 tok/s, 1.50 GB per token, 263 GB/s, 112% of probe read wall 236 GB/s | same command, arm A | 2026-11-21 | Spark GB10, shared with resident vLLM | indicative, wall includes L2 hits |

## Older claims

Tables in `docs/perf/scoreboard.md` (per-family perfpass), `docs/specrt-mtp.md` and the other `docs/specrt-*.md` files are single runs or medians of 6 on a shared card. Treat them as indicative until a row here replaces them.
