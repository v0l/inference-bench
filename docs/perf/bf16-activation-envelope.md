# bf16 activation envelope (#821)

KL(fp32 transformers || engine) and top-1 agreement with the fp32 argmax, on 8 prompts of 21 to 48 tokens (the reference prompts, first position of each prompt dropped), measured on the Spark. fp32 transformers runs with TF32 off. vLLM is `~/.venv-vllm` 0.33 with its default bf16 path, llama.cpp is the bf16 GGUF (converted with `convert_hf_to_gguf.py`) on CUDA. specrt single plane is `[runtime] split_activations = false` on all four dense linears and the lm_head; the exact path (hi and lo planes) sits at 1e-10 to 4e-10 mean KL on every family.

| family | rows | vLLM KL mean / max / top-1 | llama.cpp bf16 KL mean / max / top-1 | specrt single plane KL mean / max / top-1 |
|---|---|---|---|---|
| llama | 311 | 4.27e-04 / 2.31e-03 / 299 | 2.52e-05 / 2.48e-04 / 308 | 2.46e-05 / 3.23e-04 / 311 |
| gemma3 | 318 | 1.07e-03 / 7.68e-02 / 313 | 1.32e-04 / 6.46e-03 / 317 | 4.09e-04 / 6.98e-02 / 315 |
| granite | 318 | 4.20e-04 / 3.38e-03 / 317 | 2.25e-05 / 4.62e-04 / 318 | 2.66e-05 / 1.54e-03 / 318 |
| smollm3 | 308 | 5.45e-04 / 3.96e-03 / 306 | 4.69e-05 / 1.77e-03 / 308 | 4.73e-05 / 1.80e-03 / 308 |
| phi3 | 320 | 1.16e-03 / 6.50e-03 / 313 | 3.23e-05 / 1.31e-03 / 319 | 2.93e-05 / 5.59e-04 / 319 |

Single plane is at or better than vLLM's own distance from fp32 on every family, by 2.6x (gemma3) to 40x (phi3) on mean KL, and its top-1 is never worse than vLLM's. The max KL is also below vLLM's on every family. The table is the first single plane run (before the direct accumulation kernel change); the GPU test reruns the shipped kernel (llama 2.2e-5, gemma3 3.0e-4, granite 2.4e-5, smollm3 4.3e-5, phi3 2.6e-5).

Rerun: `crates/specrt/scripts/envelope.sh [model ...]` (Spark, under `flock /tmp/gpu27.lock`; needs the checkpoints in `~/ib-gates/821/models` and the bf16 GGUFs in `~/ib-gates/821/gguf`). The recorded numbers for the tests are in `crates/specrt/tests/golden/envelope.json`; the GPU test is `tests/envelope.rs` with `SPECRT_ENVELOPE_DIR` and `SPECRT_ENVELOPE_MODELS`.

## Default

Single plane activations are the default for bf16 prefill in the llama (and mistral), gemma3, granite, smollm3 and phi3 family specs (`[runtime] split_activations = false` in `specs/family-<name>.toml`). The model spec or the server `[runtime]` table can set `split_activations = true` for the exact path. Families not measured here (qwen2, qwen3, olmo2, qwen3.5) keep the split.

## Retune for single plane (Spark, llama-3.2-1b, 1,024 tokens, paired A/B)

Kept: the single plane bf16 path accumulates straight into the fragment (one mma per fragment, no temporary and add). 1.11x prefill, IQR 0.08, 6 rounds; decode flat.

Tiles, 4 runs each: 256x128 (4x2) 15.3-15.6k tok/s, 128x256 11.8k, 128x128 (2x4) 9.5-11k, 128x128 (4x2) 11.1k, 256x256 8k, 64x256 7k. The default tile stays. bk 64 with 2 stages gave 16.2-16.6k against 15.6-15.9k for bk 32 (about +3%, under the 5% bar, not kept); 3 and 4 stages at bk 32 were level (16.1k, 15.3-16.1k); bk 64 with 3 stages and bk 128 do not fit the 99 KB shared memory limit.

## Scoreboard after (prefill ratio at 1,024 tokens against the best competitor, before in brackets)

| family | specrt prefill 1024 | best competitor | ratio | decode b1 ratio | b8 ratio |
|---|---|---|---|---|---|
| llama | 16615.7 (14327.0) | 24239.2 vllm | 0.69 (0.56) | 0.97 | 1.07 |
| gemma3 | 18408.3 (15247.9) | 26430.8 vllm | 0.70 (0.71) | 0.99 | 0.90 |
| granite | 6684.0 (5405.1) | 9461.6 vllm | 0.71 (0.53) | 0.95 | 1.04 |
| smollm3 | 5808.3 (4825.7) | 9055.0 vllm | 0.64 (0.53) | 0.94 | 1.04 |
| phi3 | 4610.8 (3807.3) | 5763.5 llama.cpp | 0.80 (0.67) | 0.87 | 1.06 |

The 0.8x target at 1,024 tokens is reached on phi3 only. phi3's vLLM column did not run on the last pass, so its ratio is against llama.cpp (its earlier vLLM figure was lower than llama.cpp). The rest of the gap is the main loop (the kernel is at about 85% of cuBLAS's 79 TFLOP/s, see docs/perf/llama/2026-11-19.md) and the non-GEMM kernels (flash_mma 6-14%).

## qwen3_5 (#815)

Qwen3.5-9B on the Spark, prefill paired A/B (split vs single plane, 4 rounds): 1024 tokens 1850 to 2040-2190 tok/s (about 1.11x), 4096 tokens 1720 to 2020-2150 (about 1.20x); decode level. Against fp32 transformers (CPU, TF32 off, 312 rows of 8 prompts, same prompts as the other families):

| path | mean KL | max KL | top-1 of 312 |
|---|---|---|---|
| vLLM | 2.78e-04 | 1.54e-03 | 306 |
| specrt split (old default) | 1.1e-10 | 9.9e-10 | 312 |
| specrt single plane (new default) | 1.07e-05 | 2.09e-04 | 311 |

Single plane is 26x closer to fp32 than vLLM on mean KL, 7x on max, with more top-1 matches. Rerun: `envelope <checkpoint> prompts.json out split plain` then `envelope.py compare`. `family-qwen3_5.toml` now sets `split_activations = false`; `[runtime] split_activations = true` restores the exact path. The remaining gap to llama.cpp is the bf16 GEMM main loop (gate_up_mma_big 45% of prefill) and gdn_chunk_prep/scan (5%).
