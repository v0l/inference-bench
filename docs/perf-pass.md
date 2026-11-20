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
- `~/ib-gates/<issue>` is the work dir, `~/ib-gates-target/<issue>` the target dir. MemAvailable must be 40 GB or more. Never touch Frigate, ComfyUI, the media mirror service, ggml-rpc-server or inference-bench-host.service. Delete the target dir, models and caches when the pass is over.
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
- qwen3_next: `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` (48 GB) loads through `[linear] fused_interleaved = true` in `family-qwen3_next.toml` (fused `in_proj_qkvz`/`in_proj_ba` split by key-head group at load). Spark GB10: prefill 2045 (1024) / 2203 (8192) tok/s, decode b1 38.3, b8 141.9 aggregate. Spec: `quant = bf16` (the checkpoint leaves q/k/v, in_proj_qkvz, in_proj_ba, router and lm_head unquantised, so those run as bf16 `*_mma_*` kernels), `[mlp] quant = nvfp4_g16`, `[mlp] act_quant = fp4` (read from the modelopt `quant_algo = "NVFP4"`, which before this change was dropped for MoE and sent prefill down the slow grouped gemv at 273 tok/s). Experts run native NVFP4: `moe_up_fp4`/`moe_down_fp4` at decode b1, `moe_mma_up_s`/`moe_mma_down_s` at b8 and prefill. `o_proj` and linear `out_proj` are NVFP4 in the checkpoint but dequantised to bf16 at load (top-level `quant = bf16`). vLLM and SGLang: not run, the 80B does not fit beside the resident vLLM on the Spark. Accuracy: full model against vLLM (marlin backends, 8 prompts x 32 tokens): mean KL 0.0101, max 0.119, teacher-forced top-1 244/256 (`tests/real_next.rs`, reference `scripts/vllm_logprobs_offline.py`); first three layers against transformers bf16: worst KL 1.3e-4, top-1 8/8 (`scripts/next_slice.py`). A four-layer slice reaches KL 0.03 to 0.16 on 300 token prompts, from the first full-attention layer; not isolated, the 1 to 3 layer slices and the full-model vLLM comparison are clean.
- gguf-q8: `Qwen3-0.6B-Q8_0.gguf` made with `convert_hf_to_gguf.py --outtype q8_0`; measured.
- SGLang 0.5.23 installs and runs on the Spark (aarch64, sm_121) in `~/.venv-sglang`. Qwen3-0.6B: prefill 37118 (1024) and 27087 (8192) tok/s, decode 109.4 b1 and 1012.0 b8, against specrt qwen3 21998 / 140.0 / 1081.9. `perfpass.sh` now runs `sglang_perf.py` for every safetensors family.
- qwen3_next: `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` (48 GB) loads through `[linear] fused_interleaved = true` in `family-qwen3_next.toml` (fused `in_proj_qkvz`/`in_proj_ba` split by key-head group at load). Spark GB10, quant=Bf16 path: prefill 272.6 (1024) / 292.5 (8192) tok/s, decode b1 35.4 / 37.9 tok/s, b8 162 tok/s aggregate; output checked ("The capital of France is" gives Paris). The vLLM and llama.cpp columns were not produced: the vLLM run was stuck compiling FlashInfer MoE kernels and was killed, and no GGUF converts for this model.
- SGLang column for every family: `docs/perf/sglang-2026-11-19.md`; qwen3_moe at memory fraction 0.6 reaches 5114 / 6690 prefill tok/s and 53.0 / 221.8 decode, so specrt's decode b1 is level with it and b8 is 0.72x.
