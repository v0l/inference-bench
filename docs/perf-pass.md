# Perf pass: the standard per-architecture loop

One pass per model family. Every later pass follows these steps, in order, and ends with a row in [perf/scoreboard.md](perf/scoreboard.md) and a report in `docs/perf/<family>/<date>.md`.

## 1. Pick the model

The smallest real checkpoint of the family, plus one mid size if it fits beside what else is on the card. `perfpass.sh` has a default per family (`pick_model`): Qwen2.5-0.5B, Qwen3-0.6B, Llama-3.2-1B, TinyMistral-248M, gemma-3-1b-it, Phi-3-mini, SmolLM3-3B, OLMo-2-1B, granite-3.3-2b, Qwen3.5-9B, GGUF Q4_K_M of Qwen3-0.6B. Override with `--model <checkpoint dir | spec.toml | .gguf>`.

## 2. Measure the hot paths

`crates/specrt/scripts/perfpass.sh <family> --host spark` runs, under `flock /tmp/gpu27.lock` and `systemd-run -p MemoryMax=60G`:

- `bench` single stream at the prompt lengths in `--lens` (default 1024 and 8192), 128 new tokens: prefill tok/s and decode tok/s, median of `--rounds` (default 3).
- `bench --serve` at batch 8: aggregate decode tok/s and the wall step time.
- `nsys` per-kernel tables (`cuda_gpu_kern_sum`) for decode at batch 1 and batch 8, and for prefill at the first and last length. Each table lists the kernel's share of the step and its median time.
- Decode roofline: weight bytes times decode tok/s against the device bandwidth (`ROOF_GBS`, 273 GB/s on the GB10). For a kernel, bytes it must read over its median time.
- Gap: batch step wall time minus the nsys kernel sum per step. A gap above about 10% of the step is the first thing to fix, before any kernel.

## 3. Iterate

One change at a time, in kir prims (never a per-architecture kernel). Judge with a paired A/B:

```bash
crates/specrt/scripts/perfpass.sh --ab <bench-base> <bench-branch> <model> --rounds 5 --lens 1024,8192
```

It alternates A, B, A, B for the given number of rounds and prints the median B/A ratio per length with the IQR. It refuses a result whose IQR is above 10% of the median (re-run on a quieter card). Keep a change only on a 5% gain or better with the correctness gates holding (logits KL and greedy against the reference, the tests in `docs/testing.md` for that family). Stop when the top kernels are about 80% of their roofline, or each remaining one has a written reason in the family report. Build each side in its own target dir (`CARGO_TARGET_DIR=~/ib-gates-target/803-<side>`), and check the JIT source in the PTX before trusting numbers.

## 4. Compare

Same model, same hardware, same lengths, same concurrency:

- llama.cpp: the checkpoint is converted to a bf16 GGUF (`convert_hf_to_gguf.py`, in a throwaway venv under `~/ib-gates/803/venv`), then `llama-bench -p <lens> -n 0`, `llama-bench -n 128` and `llama-batched-bench -npl 8`, flash attention on.
- vLLM: `vllm_perf.py` from `~/.venv-vllm`, read only, `gpu_memory_utilization 0.12`, prefix caching off, caches redirected to `~/ib-gates/803` so nothing under `~/.cache/vllm` or `~/.cache/flashinfer` is written. Prefill is the time to one token, decode is the slope between 1 and 128 new tokens.
- SGLang: `sglang_perf.py` from `~/.venv-sglang` (sglang 0.5.23, installed on the Spark), radix cache off, same lengths. `perfpass.sh` runs it when the venv exists; otherwise the column says `not run`.
- A column that cannot run says `not run` with the reason in the family report.

Prefill and decode of competitors are timed differently (HTTP-free in all three here, but vLLM's scheduler adds a few ms per request), so compare ratios within a few percent only.

## 5. Record

`perfpass.sh` writes the report and merges one row into `docs/perf/scoreboard.md` (replacing the family's old row): our tok/s, the best competitor's, the ratio (above 1.00 means we are faster), date, commit and command. Add what you changed and why to the family report by hand, and commit.

## Hardware rules

- Spark only unless the supervisor opens a window. The desktop cards serve the user's models: `perfpass.sh --host desktop` refuses.
- `~/ib-gates/<issue>` is the work dir, `~/ib-gates-target/<issue>` the target dir. MemAvailable must be 40 GB or more. Never touch Frigate, ComfyUI, nmirror, ggml-rpc-server or inference-bench-host.service. Delete the target dir, models and caches when the pass is over.
- Spark clocks run under load at about 2.2 GHz and the card is shared with a resident vLLM and Frigate, so repeat a number before believing it.

## Families and the model each uses

| family | default model | note |
|---|---|---|
| qwen2 | Qwen2.5-0.5B | |
| qwen3 | Qwen3-0.6B | |
| qwen3_moe | Qwen3-30B-A3B | 61 GB bf16, does not fit the 60 GB gate on the Spark, desktop only |
| qwen3_5 | Qwen3.5-9B (hybrid delta-net) | |
| qwen3_next | none on the Spark | 80B, desktop only |
| llama | Llama-3.2-1B | |
| mistral | TinyMistral-248M | |
| gemma3 | gemma-3-1b-it | |
| phi3 | Phi-3-mini-4k | |
| smollm3 | SmolLM3-3B | |
| olmo2 | OLMo-2-1B | |
| granite | granite-3.3-2b-base | |
| gguf-q4 | Qwen3-0.6B Q4_K_M | llama.cpp compares on the same file |
| gguf-q8 | Qwen3-0.6B Q8_0 | quantise with `llama-quantize` from the bf16 GGUF |

## Desktop-only measurements

For the supervisor's next GPU window (27B and 122B on sm_120). Run under `flock /tmp/gpu27.lock`, with the stop and restart of `local-llm.service` inside the same locked script as described in the project notes:

```bash
crates/specrt/scripts/bench27b-nvfp4.sh
crates/specrt/scripts/bench4b.sh
crates/specrt/scripts/vllm27b-nsys.sh
```

and the nsys tables for the 27B with `crates/specrt/scripts/nsys617.sh` (`SEQS=1` and `SEQS=8`). Put the numbers in `docs/perf/qwen3_5/<date>.md` with `desktop` in the title and add the row by hand with `perfpass_report.py merge`.

## Gaps from the first pass (#817)

- qwen3_moe: measured on `Qwen/Qwen3-30B-A3B-FP8` (31 GB, fits the Spark gate), not the 61 GB bf16 checkpoint. specrt decode is ahead of vLLM (52.6 vs 40.3 tok/s b1, 158.7 vs 133.3 b8) but prefill is 306 tok/s against vLLM's 3959 (ratio 0.08). SGLang did not fit beside the resident Spark load at 0.45 memory fraction (needs above 0.533): not run.
- qwen3_next: not run. The only checkpoints on the Spark are 89 GB (Qwen3-Coder-Next FP8) and 99 GB (Qwen3.8-Flash-Next NVFP4), over the 60 GB gate. Needs a desktop window opened by the supervisor.
- gguf-q8: `Qwen3-0.6B-Q8_0.gguf` made with `convert_hf_to_gguf.py --outtype q8_0`; measured.
- SGLang 0.5.23 installs and runs on the Spark (aarch64, sm_121) in `~/.venv-sglang`. Qwen3-0.6B: prefill 37118 (1024) and 27087 (8192) tok/s, decode 109.4 b1 and 1012.0 b8, against specrt qwen3 21998 / 140.0 / 1081.9. `perfpass.sh` now runs `sglang_perf.py` for every safetensors family.
- qwen3_next: `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` (48 GB) fits the Spark gate but specrt cannot load it yet (fused `in_proj_qkvz`/`in_proj_ba` tensor layout), see `docs/perf/sglang-2026-11-19.md`.
- SGLang column for every family: `docs/perf/sglang-2026-11-19.md`; qwen3_moe at memory fraction 0.6 reaches 5114 / 6690 prefill tok/s and 53.0 / 221.8 decode, so specrt's decode b1 is level with it and b8 is 0.72x.
