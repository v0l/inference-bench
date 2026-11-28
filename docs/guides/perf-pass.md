# Perf pass: the per architecture speed loop

One pass per model family. A pass ends with a row in [../generated/perf/scoreboard.md](../generated/perf/scoreboard.md); the per family report with the nsys tables goes to `out/perf/<family>/<date>.md` (not committed) and its findings go into the issue and the commit message. Speed work in the engine is internal tooling: comparisons are never a user facing feature.

## 1. Pick the model

The smallest real checkpoint of the family, plus one mid size if it fits. `perfpass.sh` has a default per family (`pick_model`); override with `--model <checkpoint dir | spec.toml | .gguf>`.

| Family | Default model | Note |
|---|---|---|
| qwen2 | Qwen2.5-0.5B | |
| qwen3 | Qwen3-0.6B | |
| qwen3_moe | Qwen3-30B-A3B | 61 GB bf16; the FP8 copy fits the Spark gate |
| qwen3_5 | Qwen3.5-9B | hybrid delta-net |
| qwen3_next | none | 80B, desktop only |
| llama | Llama-3.2-1B | |
| mistral | TinyMistral-248M | |
| gemma3 | gemma-3-1b-it | |
| phi3 | Phi-3-mini-4k | |
| smollm3 | SmolLM3-3B | |
| olmo2 | OLMo-2-1B | |
| granite | granite-3.3-2b-base | |
| gguf-q4 | Qwen3-0.6B Q4_K_M | llama.cpp on the same file |
| gguf-q8 | Qwen3-0.6B Q8_0 | `llama-quantize` from the bf16 GGUF |

## 2. Measure the hot paths

`crates/specrt/scripts/perfpass.sh <family> [--model M] [--rounds N] [--lens 1024,8192]` runs under `systemd-run -p MemoryMax=60G`:

- `bench` single stream at each prompt length, 128 new tokens: prefill and decode tok/s, median of the rounds.
- `bench --serve` at batch 8: aggregate decode tok/s and the wall step time.
- `nsys` kernel tables (`cuda_gpu_kern_sum`) for decode at batch 1 and 8 and for prefill at the first and last length.
- Decode roofline: weight bytes times decode tok/s against the device bandwidth (`ROOF_GBS`).
- Gap: batch step wall time minus the kernel sum per step. A gap above about 10 percent of the step is the first thing to fix.

## 3. Iterate

One change at a time, in kir (never a per architecture kernel). Judge with a paired A/B:

```bash
crates/specrt/scripts/perfpass.sh --ab <bench-base> <bench-branch> <model> --rounds 5 --lens 1024,8192
bench --ab [--rounds N] [--a-set key=value] [--b-set key=value] <spec> [prompt_len] [new_tokens]
```

`perfpass.sh --ab` alternates two binaries and prints the median B/A per length with the IQR, refusing above 10 percent IQR (`MODEL_B` gives side B its own spec). `bench --ab` loads both arms in one process, runs an A/A control first, and refuses when the control is outside 1 plus or minus 0.02, the spread is above 0.10 or the two passes differ by more than 3 percent; `--a-set` and `--b-set` take any `speculative.*`, `runtime.*` or `spec.*` key. Keep a change on a 5 percent gain with the correctness gates holding (logits KL and greedy against the reference, the family's tests in [testing.md](testing.md)). Build each side in its own target dir and check the JIT source in the PTX before trusting numbers.

## 4. Compare

Same model, hardware, lengths and concurrency:

- llama.cpp: the checkpoint converted to a bf16 GGUF (`convert_hf_to_gguf.py` in a throwaway venv), `llama-bench` and `llama-batched-bench -npl 8`, flash attention on.
- vLLM: `crates/specrt/scripts/vllm_perf.py` from `~/.venv-vllm`, read only, prefix caching off, caches redirected to the work dir so nothing under `~/.cache/vllm` or `~/.cache/flashinfer` is written.
- SGLang: `crates/specrt/scripts/sglang_perf.py` from `~/.venv-sglang`, radix cache off.
- A column that cannot run says `not run` with the reason.

## 5. Record

`perfpass.sh` merges one row into the scoreboard (`perfpass_report.py merge`, replacing the family's row): our tok/s, the best competitor's, the ratios, date, commit and command. Keep and drop claims from `bench --ab` go in the scoreboard's paired A/B table, retracted in place when they stop holding.

## Hardware

Workers run on desktop GPU1 under `~/.agents/bin/gpu-run 1 --` (see [testing.md](testing.md)); `perfpass.sh` passes `CUDA_VISIBLE_DEVICES` through. `--host spark` syncs the tree to the Spark and runs there; workers do not use it without the user. `--host desktop` refuses (that path is the supervisor's window). A Spark run needs 40 GB MemAvailable and leaves the resident services alone; its clocks run about 2.2 GHz under load, so repeat a number before believing it.

## Desktop measurements of the 27B

```bash
crates/specrt/scripts/bench27b-nvfp4.sh
crates/specrt/scripts/bench4b.sh
crates/specrt/scripts/vllm27b-nsys.sh
crates/specrt/scripts/nsys617.sh           # SEQS=1 and SEQS=8
crates/specrt/scripts/specrt_baseline.sh   # the vLLM baseline harness, see ../generated/qwen27-baseline.md
```

Per round kernel breakdown at N sequences: profile with graph nodes traced, export to sqlite, cut per MTP round at `spec_accept` (or per plain step at `sample_rows`):

```bash
~/.agents/bin/gpu-run 1 -- nsys profile -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=none -o /tmp/r16 --force-overwrite true \
    bench --speculative --profile --kv-gb 20 crates/specrt/tests/fixtures/qwen3.8-27b-nvfp4.toml 512 256 16
nsys export -t sqlite -o /tmp/r16.sqlite /tmp/r16.nsys-rep
python3 crates/specrt/scripts/nsys_rounds.py /tmp/r16.sqlite 28 16
python3 crates/specrt/scripts/nsys_rounds.py /tmp/r16.sqlite 28 16 --plain
```
