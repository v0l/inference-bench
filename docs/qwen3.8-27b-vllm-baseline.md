# Qwen3.8 27B: what vLLM does today, and what our server would have to do

Source: the `local-llm.service` journal on the RTX PRO 6000 box (26 Oct to 7 Nov 2026, 562,621 lines, 78,349 ten-second throughput samples). The numbers are from real traffic, not a benchmark: they mix prompt lengths, reasoning output, tool calls and several clients at once. They are the traffic our server would have to carry.

## The deployment being measured

| | |
|---|---|
| Model | `orcarouter/Qwen3.8-27B-Uncensored-NVFP4` (compressed-tensors, mixed precision) |
| Engine | vLLM 0.27.1, one GPU, `FLASHINFER` attention, `FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM |
| GPU | RTX PRO 6000 Blackwell Max-Q, 96 GB, compute capability 12.0 |
| Launch | `--max-model-len 262144 --max-num-seqs 32 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.75 --kv-cache-dtype fp8 --enable-prefix-caching` |
| Speculative decoding | MTP, `num_speculative_tokens 2` (the model's own multi-token-prediction layer) |
| API surface | OpenAI chat completions, auto tool choice with the `qwen3_coder` parser, `qwen3` reasoning parser, `reasoning_effort` default `medium` |
| Startup | weights 10.3 s, `torch.compile` 68.7 s, whole engine init 153 s (compile cache warm) |
| Memory | weights 23.3 GiB, KV cache 41.8 GiB = 1,199,010 tokens, 4.57x concurrency at the full 262,144 context |

## Model shape (from `config.json`)

64 layers, hidden 5120, FFN 17408, vocab 248,320, 262,144 positions. **48 linear-attention layers and 16 full-attention layers** (one full layer every fourth). Full attention: 24 query heads, 4 KV heads, head dim 256. Linear attention: 48 value heads, 16 key heads, key and value head dim 128, conv kernel 4 (a gated delta-net style recurrent layer). One MTP layer. Quantisation: FP8 weights and activations for the attention projections, **NVFP4 (4-bit float, group size 16) for every MLP**. The vision tower is ignored (text-only).

## Measured performance

Decode, aggregate generation tokens per second in 10 s windows with N requests running:

| running requests | windows | median aggregate | p90 aggregate | per request (median) |
|---|---|---|---|---|
| 1 | 5,332 | 115.7 | 276.0 | 115.7 |
| 2 | 6,007 | 234.6 | 368.4 | 117.3 |
| 3 | 8,397 | 313.7 | 431.8 | 104.6 |
| 4 | 9,150 | 361.8 | 497.3 | 90.5 |
| 6 | 7,192 | 449.0 | 628.0 | 74.8 |
| 8 | 6,233 | 512.2 | 710.7 | 64.0 |
| 12 | 1,343 | 601.0 | 856.3 | 50.1 |
| 16 | 299 | 796.2 | 1081.8 | 49.8 |
| 20 | 97 | 924.2 | 1208.2 | 46.2 |

- Single stream with no prefill in the window: median **162.5 tok/s**, p10 104.4, p90 288.8, best window 639.9. The spread is the speculative decoder: the same stream runs at about 100 tok/s on hard text and well over 250 tok/s on predictable text.
- Peak aggregate in any window: 1,498 tok/s (31 Oct, 27 requests running).
- Prefill with at most 2 requests running: median **1,782 tok/s**, p90 2,065. Bursts reach 8,761 tok/s at low concurrency and 32,677 tok/s (4 Nov, 10 running) when long prompts arrive together; the 16,384 batched-token budget is what bounds a burst.
- Speculative decoding (MTP, 2 draft tokens): mean acceptance length median **2.79** tokens per step (p10 2.69, p90 2.87), draft acceptance 89%, per-position acceptance about 0.93 then 0.84. So decoding gets about 2.8 tokens per forward pass for one extra small layer.
- Prefix cache hit rate over a day: 3% to 82%, typically 16% to 29%; 55% to 82% on 3 and 4 Nov when long shared agent contexts dominated.

Load profile over the whole log: 288.3 M tokens generated and 997.9 M prompt tokens processed. The engine had generation in 96.5% of the windows (it never idles: agents run all day). Peak KV usage 93.7% (4 Nov, 16 running); typical peaks 35 to 53%. A queue formed in 1,311 windows (1.7%). Process crashes or preemptions: none logged; the ERROR lines are the video processor docstring warning and OpenTelemetry tracebacks, not request failures.

Per-day median generation rates while busy: 97 (26 Oct, first day) up to 557 (1 Nov) and 303 to 355 in the last week. The day-to-day spread is concurrency, not engine speed.

## Can our server run this model

Not today. specrt is a spec-driven runtime for **dense** decoder models: `Quant` is `Bf16` or `Q8`, `Block` is `Dense`, `AttentionKind` is `Gqa`, norm is RMSNorm. Qwen3.8 27B needs five things it does not have:

1. **Gated delta-net linear attention** (48 of 64 layers): a recurrent state per sequence, a causal conv1d (kernel 4), gated delta-rule kernels for prefill (chunked) and decode, and a per-sequence state cache next to the KV cache. This is the largest piece and it is new kernel work.
2. **NVFP4 weights and activations** for the MLPs: FP4 e2m1 with group-16 scales, dynamic activation quantisation, and tensor-core FP4 GEMM on sm_120 (the card here is 12.0; the GB10 Spark is 12.1). specrt has no FP4 path.
3. **FP8 per-channel weights with per-token dynamic activation scales** for the attention and linear-attention projections (the checkpoint mixes both).
4. **MTP speculative decoding**: draft with the model's own MTP layer, verify 3 tokens per step, accept or reject against the sampler. This is a large part of the 160 tok/s single-stream number, so leaving it out cuts the single-stream rate by roughly 2.5x on the same weights.
5. **fp8 KV cache** (specrt keeps KV as f32 or f16) and **prefix caching** across requests, plus the server pieces vLLM gives for free here: the `qwen3_coder` tool call parser, the `qwen3` reasoning parser, `reasoning_effort` in the chat template, a 262,144-token context and a 32-request batch.

## What a fair target looks like

Match these on the same GPU and the same traffic mix before calling it a replacement:

| measure | vLLM today | target |
|---|---|---|
| single-stream decode (median, with MTP) | 162 tok/s | at least 130 (80%) |
| single-stream decode without MTP | not measured here | report it, so MTP's share is clear |
| aggregate decode at 8 running | 512 tok/s | at least 410 |
| aggregate decode at 16 running | 796 tok/s | at least 640 |
| prefill, one request | about 1,800 tok/s in these logs (mixed, short prompts) | measure on a fixed 4k and 16k prompt, within 20% of vLLM |
| context | 262,144, 4.57x concurrent | 131,072 first, 262,144 after |
| tool calls and reasoning | `qwen3_coder`, `qwen3` | byte-identical to vLLM on a 200-request replay |
| logits vs a reference | n/a | max abs under 1e-2 on NVFP4 (it cannot be 1e-3), top-1 agreement 99% or more |

The replay is the key test: record real requests (from the router if it logs bodies, otherwise from a capture proxy in front of vLLM), send the same 200 to both servers, compare outputs, tool-call JSON and latency.

## Suggested route, smallest steps first

1. Fixed benchmark of vLLM itself on this box (nothing else on the GPU, the 27B alone): single stream at 128, 1k, 4k, 16k prompts, then 1, 4, 8, 16 concurrent, with and without MTP. The log numbers above are a mix; this gives the clean baseline to beat. Stop other GPU users first.
2. Dense-only first: run a smaller dense Qwen3 (the 0.6B spec exists) through the same server and the same router, so the OpenAI surface, tool parsing and reasoning split are proven before the architecture work starts.
3. Add the 27B's pieces in order of how much each is worth: FP8 projections and fp8 KV, then NVFP4 GEMM, then the delta-net layers (prefill, then decode), then prefix caching, then MTP. Each step gated by logits against the reference and the replay.
4. Only then compare end to end against the numbers above.

Estimated effort (my guess, not measured): the delta-net layers and NVFP4 are each a multi-week kernel project, and MTP needs the delta-net layers working first because the draft shares their state. The model's architecture is the cost, not the serving code.

## Method notes

- Throughput lines are the engine's own 10 s averages (`loggers.py`), parsed from the journal. "Running" is the number of requests in the batch at the time of the line.
- Windows with `Running: 1` and zero prompt throughput are the single-stream sample; windows with prefill in them are excluded from it.
- The journal also holds earlier deployments (Solstice-AI and FP8 variants of the same model, and runs from 26 Oct). The tables use all windows; the launch and startup facts are from the 7 Nov 11:41 start, which is the current service.
