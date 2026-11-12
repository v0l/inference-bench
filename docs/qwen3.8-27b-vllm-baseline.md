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

## Clean baseline (10 Nov 2026, #307)

vLLM 0.27.1 started by hand with the service's exact flags (`docs/qwen27-run/launch.sh`, port 8011) while `local-llm.service` was stopped. Only desktop apps and agentbench-server (4.4 GB) were on the GPU (RTX PRO 6000 Blackwell Max-Q, driver 595.112.07). No request other than the harness reached the benchmark server (checked in the vLLM logs). Harness: `crates/specrt/scripts/vllm_baseline.py`, streaming chat, `ignore_eos`, thinking off, unique prompt per request so the prefix cache only hits where a suite asks for it. Raw JSON and logs are in `docs/qwen27-run/`. MTP on and MTP off are two sequential launches, not alternating runs. The 200 request replay set was skipped: no recorded request bodies exist on this machine. A first MTP attempt on port 8001 was discarded because routed YALR traffic reached it (`contaminated-8001/`).

KV capacity from the startup log: **400,663 tokens with MTP, 439,919 without** (14 GB fp8 cache), which is 1.53x and 1.68x of one 262,144 token request.

### Single stream, 128 output tokens

| prompt tokens | runs | TTFT s (MTP / off) | prefill tok/s (MTP / off) | decode tok/s (MTP / off) |
|---|---|---|---|---|
| 128 | 6 | 0.055 / 0.046 | 2,195 / 2,633 | 112.0 / 59.9 |
| 1,024 | 6 | 0.128 / 0.113 | 7,912 / 8,986 | 105.6 / 59.4 |
| 4,096 | 6 | 0.466 / 0.456 | 8,781 / 8,968 | 118.2 / 58.7 |
| 16,384 | 6 | 2.04 / 2.05 | 8,016 / 8,001 | 111.8 / 57.2 |

### Concurrency, 128 token prompts, aggregate decode tok/s

| streams | MTP | off |
|---|---|---|
| 1 | 112.5 | 59.3 |
| 4 | 431.5 | 215.2 |
| 8 | 818.6 | 434.5 |
| 16 | 1,368 | 806.1 |
| 32 | 2,319 | 1,386 |

## Long context (up to 260k)

Single stream, 128 output tokens, 6 runs up to 65k and 3 runs from 131k (medians). The specrt column is the 27B NVFP4 spec (`qwen3.8-27b.toml`) through `prefixbench long` (scheduler, no HTTP, bf16 KV, 24 GB KV pool, prefix cache on), a single cold run per point.

| prompt tokens | vLLM MTP TTFT s | vLLM off TTFT s | vLLM MTP prefill tok/s | vLLM off prefill tok/s | vLLM MTP decode | vLLM off decode | specrt TTFT s | specrt prefill tok/s | specrt decode |
|---|---|---|---|---|---|---|---|---|---|
| 32,768 | 4.78 | 4.67 | 6,849 | 7,023 | 106.0 | 56.0 | 28.6 | 1,145 | 78.3 |
| 65,536 | 12.29 | 11.75 | 5,332 | 5,578 | 105.1 | 54.2 | 68.4 | 959 | 63.8 |
| 131,072 | 34.82 | 33.13 | 3,764 | 3,956 | 98.7 | 50.5 | 172.8 | 758 | 47.1 |
| 196,608 | 67.56 | 64.39 | 2,910 | 3,053 | 83.4 | 47.3 | 314.6 | 625 | 37.3 |
| 260,000 | 109.1 | 103.9 | 2,383 | 2,501 | 85.1 | 44.3 | 488.3 | 532 | 31.0 |

specrt ran every length to 260k without an out of memory error at KV 24 GB. Its long prefill is 4x to 5x slower than vLLM and falls with length (1,374 tok/s at 4k to 532 at 260k), so prefill attention at long context is the gap to close.

#505, specrt prefill on `qwen3.8-27b.toml`, `prefixbench long`, bf16 KV, cache on, cold. The GPU was never idle: the served vLLM (local-llm, port 8001) had live requests the whole time, so every wall-clock number below is under load and is only comparable inside one interleaved A/B (master, branch, master, branch, 3 rounds each, medians, all three values shown). The 1,101 tok/s at 32k in the table above was taken on a quieter GPU and cannot be compared with these.

| prompt | master tok/s (3 runs) | branch tok/s (3 runs) | median gain |
|---|---|---|---|
| 4,096 | 689 (689, 1259, 664) | 1,474 (817, 1474, 1542) | too noisy |
| 32,768 | 579 (573, 587, 579) | 916 (916, 739, 930) | 1.6x |
| 65,536 | 549 (501, 549, 828) | 746 (738, 746, 1015) | 1.4x |

So the earlier 596 vs 1,101 was contention, not a regression: master measured 579 in the same window.

Profile (nsys, 8k to 16k prompt): prefill is GEMM bound. gate_up 37%, down 16%, lin_in 11%, flash_mma 8% at 16k and 28% at 64k. Per-kernel medians (kernel time does not depend on the load much, but they overlap vLLM too):

| kernel | master 64x128 tile | 128x128 tile (branch) |
|---|---|---|
| gate_up_mma_big (FP8) | 23.5 ms | 19.3 ms |
| down_mma_big | 9.7 ms | 9.6 ms |
| lin_in_mma_big | 9.3 ms | 8.6 ms |
| gate_up_f4w_big (NVFP4 spec) | 18.7 ms | 17.2 ms |

At 2048 tokens gate_up is 2 x 2048 x 34816 x 5120 = 0.73 TFLOP, so 19.3 ms is 38 TFLOP/s. cuBLAS on the same shape in this venv: bf16 about 240 to 250 TFLOP/s, FP8 `_scaled_mm` about 500 TFLOP/s (down and lin_in shapes measured; gate_up in the same range). The kernels here run the `mma.sync` bf16 path with weights dequantised in the main loop and the activation split in hi and lo planes (two MMAs per tile), so they sit at roughly 15% of cuBLAS bf16 and 8% of FP8. Closing that needs a different design (tcgen05 or native FP8/FP4 MMA with block scales on the weight side, no hi/lo split), not tuning. The `[runtime] big_tile` sweep (nvfp4 and fp8 specs, tiles 64x128, 128x128 with 2x4 and 4x2 warps, 64x256, 128x256) found 128x128 with 2x4 warps best; 64x256 is slightly worse, 128x128 with 4x2 warps is worse. 128x256 and 3-stage FP8 do not fit in shared memory.

Changes: flash_mma skips the unused lo KV plane with 1-plane KV (flash_mma 8.6 ms to 7.7 ms median at 16k, 32 keys per tile for head_dim 256); the default big tile is 128x128 (was 64x128); NVFP4 weight GEMM stage count now fits the shared memory limit so larger tiles load. The `qwen3.8-27b-nvfp4-act.toml` spec (native FP4 MLP, FP8 projections) is about 2x faster at prefill but changes numerics; the serving default stays `qwen3.8-27b.toml`.

Still about 5x below vLLM at 32k (6,849 tok/s, quiet GPU) on specrt-under-load numbers; on a quiet GPU the branch would be higher.

#517, prefill GEMMs on native low-precision MMA. All numbers were taken with the served vLLM (port 8001) up and under load.

The native kernels were already there: `gemm_fp8` (`mma.sync.m16n8k32.e4m3`, FP8 activations quantised per 128 columns or per token by `quant_rows_fp8`, no hi/lo split) and `gemm_f4a` (`mma.sync.kind::mxf4nvf4.block_scale` with the NVFP4 block scales in the MMA). Both load with a `cp.async` pipeline. The spec picks them (`act_quant = "fp8"`, `[mlp] act_quant = "fp4"`), not the architecture. At 16k tokens they lost a lot to L2 misses. With row-major block order, a 128-row M tile streams the whole weight matrix, so `gate_up` read 178 MB 128 times. The change orders blocks in groups of 32 M tiles per column sweep (`grouped_tile`). This only applies when the GEMM has at least 256 column tiles (`gate_up`). For the narrower `down` and `lin_in` it measured flat or slightly worse, so they keep the old order.

Kernel time per call, `tests/gemm_speed.rs` (CUDA events, median of 10), against cuBLAS FP8 `torch._scaled_mm` with per-row and per-column scales (`scripts/cublas_gemm.py`). The cuBLAS numbers vary 447 to 526 TFLOP/s between runs because of the load.

| GEMM, 16,384 tokens | before | after | cuBLAS FP8 | share of cuBLAS |
|---|---|---|---|---|
| `gate_up` FP8 block-128 (`qwen3.8-27b-act`) | 22.6 ms, 259 TFLOP/s | 17.1 ms, 342 | 11.9 ms, 489 | 70% |
| `down` FP8 block-128 | 8.1 ms, 360 | 8.0 ms, 363 | 5.9 ms, 493 | 74% |
| `lin_in` FP8 block-128 | 7.2 ms, 380 | 7.2 ms, 384 | 5.6 ms, 487 | 79% |
| `gate_up` FP8 per-token (`qwen3.8-27b-nvfp4-act`) | 19.3 ms, 303 | 12.3 ms, 473 | | 97% |
| `down` FP8 per-token | 6.1 ms, 477 | 6.0 ms, 484 | | 98% |
| `lin_in` FP8 per-token | 5.7 ms, 483 | 5.6 ms, 487 | | 100% |
| `gate_up` NVFP4 (`gemm_f4a`) | 10.3 ms, 569 | 9.6 ms, 610 | | |
| `down` NVFP4 | 4.8 ms, 603 | 4.8 ms, 604 | | |

At 2,048 tokens block-128 `gate_up` goes from 327 to 357 TFLOP/s and per-token `gate_up` from 390 to 459. cuBLAS FP8 reaches 493 to 526 there. Against the bf16 hi/lo path in the table above (`gate_up_mma_big` 38 TFLOP/s), the native block-128 kernel is 9x faster.

nsys, `prefixbench long` at 16,384 tokens on `qwen3.8-27b-act.toml`, branch build: 3,763 tok/s. The top kernels are `gate_up_f8_huge` 19.9% (2.36 ms median per 2,048-token chunk), `flash_mma` 10.3%, `down_f8_huge` 10.3% (1.30 ms) and `lin_in_f8_huge` 6.4% (1.02 ms). GEMMs are no longer most of prefill: `gate_up`, `down` and `lin_in` together are 37% of kernel time, against about 64% before. `quant_rows_fp8` costs 2.8%.

End to end, `prefixbench long`, cold, interleaved base/branch, 3 rounds, all three values shown (`scripts/ab517.sh`):

| prompt | master, `qwen3.8-27b.toml` (bf16 hi/lo) | branch, `qwen3.8-27b-act.toml` | master, `qwen3.8-27b-act.toml` | branch gain on the same act spec |
|---|---|---|---|---|
| 4,096 | 1,611 (1,653, 1,611, 857) | 4,129 (4,168, 4,129, 4,129) | 3,791 (3,837, 3,791, 3,790) | 1.09x |
| 32,768 | 1,453 (1,491, 1,453, 1,421) | 3,338 (3,364, 3,338, 3,337) | 3,090 (3,289, 3,090, 3,089) | 1.08x |
| 65,536 | 1,272 (1,319, 1,272, 1,244) | 2,708 (2,723, 2,708, 2,305) | 2,534 (2,586, 2,534, 2,530) | 1.07x |

So switching a 27B FP8 deployment to `act_quant = "fp8"` gives 2.1x to 2.6x prefill, and this branch adds 7 to 9% on top. The second and third columns come from one interleaved run. The fourth column and the gains come from a second interleaved run, which repeated the branch arm and got the same values within 1%.

Numerics (`tests/real27b.rs`, `qwen3.8-27b-act.toml`, this branch): logits KL against the weight-only FP8 reference 1.08e-3 (gate 1.5x the act reference's 0.86e-3), top-1 0.983 (gate 0.990 - 0.02), error against the activation-quantised reference 0.087 of the largest logit, inside the 0.33 envelope. Greedy runs 32 of 32 tokens on 3 prompts. The other 2 diverge at near ties in the reference (margins 0.10 and 0.17), as before #517. The serving default `qwen3.8-27b.toml` stays weight-only. Its gate (greedy identical, logits 2e-4) and the 1e-3 per-layer trace hold only without activation quantisation, so native FP8 stays opt-in through the spec. `qwen3.8-27b-nvfp4-act.toml` against vLLM: KL mean 0.0126, teacher-forced top-1 1,220 of 1,280.

Decode of 512 tokens after a held prefix (median tok/s):

| held prefix | MTP | off |
|---|---|---|
| 131,064 | 96.8 | 50.2 |
| 259,992 | 89.3 | 43.8 |

Prefix cache at long context: the same prompt resent with a new 1k suffix.

| prompt | vLLM MTP cold / warm TTFT s | vLLM off cold / warm TTFT s | specrt cold / warm TTFT s |
|---|---|---|---|
| 8,192 | 0.98 / 0.34 | 0.94 / 0.14 | 2.98 (4k) / 0.81 (4k) |
| 131,072 | 34.85 / 1.86 | 33.22 / 0.83 | 172.8 / 2.12 |
| 260,000 | n/a | n/a | 488.3 / 3.44 |

vLLM warm hits cached 128,000 of 131,064 tokens with MTP and 130,144 without; the shorter match with MTP is a block size effect, not measured further.

4 concurrent requests at 65,536 tokens (all four fit in the KV pool): aggregate decode 108.8 tok/s with MTP and 63.3 without, median TTFT 36.2 s and 33.5 s. specrt was not run with 4 concurrent 64k requests.

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

#551, prefill serving default and flash_mma. The server specs are now the activation-quantised ones: `qwen3.8-27b.toml` (FP8 projections, `act_quant = "fp8"`) and `qwen3.8-27b-nvfp4.toml` (FP8 projections, FP4 MLP). The old weight-only specs are `qwen3.8-27b-exact.toml` and `qwen3.8-27b-nvfp4-exact.toml`, used by the exact-reference tests (`SPECRT_27B_SPEC` defaults to them). Gates on this branch: `real27b` weight-only reference (exact spec) greedy 5 of 5 prompts for 32 tokens, logits error 2e-4; act spec KL 1.08e-3 against the 1.5x gate of 0.86e-3, top-1 0.983, greedy 3 of 5 (the two misses are near ties, margins 0.10 and 0.17, as before); NVFP4 act spec against vLLM KL mean 0.0126, teacher-forced top-1 1,220 of 1,280. Non-27B suite: 0 failures (the jit tests need free disk, the MTP reference file comes from `reference.py tiny-mtp`).

flash_mma with 1-plane KV: 3 pipeline stages of 32 keys (was 2 stages, since the lo plane's smem is unused), and the Q staging buffer is bf16-free f32 half tiles (8 rows at a time), so smem is 48 KB. Kernel time, `tests/flash_speed.rs` (2,048 query tokens, head_dim 256, 24:4 heads, CUDA events, min of 10, GPU shared with vLLM):

| keys before the chunk | before | after |
|---|---|---|
| 16,384 | 7.4 ms, 119 TFLOP/s | 5.5 ms, 160 TFLOP/s |
| 30,720 | 17.4 ms, 92 TFLOP/s | 10.3 ms, 156 TFLOP/s |

Sweeps of keys per tile (16, 32, 64) and stages (2, 3, 4) at 16k found 32 keys with 3 stages best; 4 stages and 64 keys do not fit or are slower.

End to end, `prefixbench long`, cold, interleaved, 3 rounds (`scripts/ab517.sh`), master `-exact` vs this branch's default. The served vLLM was at 90-100% GPU util the whole time, so the values swing between rounds (the first two rounds of the branch arm ran against heavier vLLM traffic):

| prompt | master weight-only, tok/s (3 runs) | branch default, tok/s (3 runs) |
|---|---|---|
| 4,096 | 786 (1,470, 824, 786) | 3,802 (2,165, 3,802, 3,952) |
| 32,768 | 768 (772, 746, 768) | 1,849 (1,849, 1,797, 3,433) |
| 65,536 | 700 (700, 693, 710) | 1,612 (1,567, 1,612, 2,511) |

Best branch values (3,952 / 3,433 / 2,511) are 45% / 50% / 47% of the vLLM quiet-GPU column (8,781 / 6,849 / 5,332). The 70% / 60% targets are not met: GEMM is still about 50% of prefill kernel time (`gate_up_f8_huge` 20%, `down` 10%, `lin_in` 7%, #441 owns that), then `flash_mma` 8% at 16k, `gdn_chunk_scan` 6.7%, `gdn_chunk_prep` 4%, `quant_rows_fp8` 3%. nsys of 32k before the flash change (branch default): `gate_up_f8_huge` 20.6%, `flash_mma` 20.1% (15.1 ms mean per call), `down_f8_huge` 10.9%, `gdn_chunk_scan` 6.8%, `lin_in_f8_huge` 6.7%, `gdn_chunk_prep` 4.2%, `quant_rows_fp8` 3.0%. `scripts/flashprof.sh` reproduces the table. 128k and 260k, chunk size 4k/8k and two-stream overlap were not measured.

#551, quiet-GPU window (local-llm stopped, GPU 2.5 GB used, 13% util at the start, `flock /tmp/gpu27.lock` held, service restarted and answering a chat request afterwards). `prefixbench long`, cold, 1-plane bf16 KV, 3 interleaved rounds, all values. master is the weight-only spec before this issue; branch is the serving default (`qwen3.8-27b.toml`, 2,048-token chunk) with the 3-stage 128-row `flash_mma`. These are quiet-GPU numbers.

| prompt | master tok/s | branch tok/s | branch, 4,096 chunk | vLLM quiet | branch share of vLLM |
|---|---|---|---|---|---|
| 4,096 | 1,541, 1,527, 1,528 | 4,018, 3,954, 3,952 | | 8,781 | 45% |
| 32,768 | 1,379, 1,372, 1,374 | 3,464, 3,434, 3,439 | 3,732 | 6,849 | 50% (54% at 4k chunk) |
| 131,072 | 1,070 (one run) | 2,370, 2,372, 2,365 | 2,596 (8k chunk 2,615) | 3,764 | 63% (69% at 4k chunk) |
| 260,000 | n/a | 1,694, 1,712, 1,731 | | 2,383 | 72% |

The server specs set `[runtime] prefill_chunk = 4096` (new spec option, default 2,048): 8% to 10% faster at 32k and 128k; 8k chunks add nothing over 4k. Targets: 128k (69%) and 260k (72%, 2,048 chunk) meet the 60% bar; 4k (45%) and 32k (54%) miss the 70% bar. At those lengths the remaining gap is the GEMMs (`gate_up_f8_huge` 20%, `down` 10%, `lin_in` 7% of kernel time, #441) plus `gdn_chunk_scan` 6.7% and `gdn_chunk_prep` 4%. Not done: two-stream overlap of the delta-net scan with attention, split-KV attention, K/V sharing across the 6 query heads of a group, and fixing the gdn kernels; the flash kernel reaches 140 to 170 TFLOP/s at head_dim 256 (FA2-class on this GPU is not measured here).

#551 final quiet-GPU A/B, run through `crates/specrt/scripts/quiet_window.sh` (takes `/tmp/gpu27.lock`, stops local-llm, restarts it in an EXIT/INT/TERM trap, polls :8001 and sends one chat request; tested with a command exiting 7 and with a SIGTERM, the service came back both times). GPU 2.6 GB used, 2% util at the start. Master is the weight-only 27B spec before this issue; branch is the serving default with `prefill_chunk = 4096`. Two interleaved rounds, all values, cold `prefixbench long`:

| prompt | master tok/s | branch tok/s | vLLM quiet | branch / vLLM (best) |
|---|---|---|---|---|
| 4,096 | 1,639, 1,582 | 4,102, 4,220 | 8,781 | 48% |
| 32,768 | 1,455, 1,427 | 3,639, 3,719 | 6,849 | 54% |
| 131,072 | 1,095 (one run) | 2,601, 2,610 | 3,764 | 69% |
| 260,000 | n/a | 1,888, 1,879 | 2,383 | 79% |

128k and 260k meet the 60% bar; 4k and 32k are below 70%. Gates in the same window, `real27b` with `--include-ignored --test-threads=1`: act default (`SPECRT_27B_SPEC=qwen3.8-27b`, NVFP4 act spec for the vLLM test) `test result: ok. 10 passed; 0 failed; finished in 283.48s`; exact specs (`qwen3.8-27b-exact`, `qwen3.8-27b-nvfp4-exact`) `test result: ok. 10 passed; 0 failed; finished in 425.25s`. The MTP tests are among the ten.

Attention experiments that did not clear the 10% bar: skipping the per-element causal mask on tiles fully below the diagonal left `flash_mma` at 5.0 ms (16k keys) and 9.5 ms (30k keys), unchanged, so it was reverted. 128 query rows per CTA already shares each K/V tile across 8 warps; sharing it across the 6 query heads of a group, split-KV and the two-stream scan/attention overlap are not implemented. At 32k `flash_mma` is 12.7% of kernel time (170 calls), `gate_up_f8_huge` 22.7%, `down` 11.2%, `gdn_chunk_scan` 7.7%, `lin_in` 7.7%, `gdn_chunk_prep` 4.4%, `quant_rows_fp8` 3.3%.

#551 attention follow-up, measured instead of built. (1) GQA K/V sharing: with the K/V `cp.async` loads compiled out of `flash_mma` (`SPECRT_FLASH_NOLOAD=1` in `tests/flash_speed.rs`, patch asserted to apply) the kernel runs at the same speed, 3.52 ms vs 3.51 ms at 15k keys, 6.75 vs 6.75 ms at 30k, 27.5 vs 27.5 ms at 129k keys (2,048 query tokens). K/V traffic is hidden behind the MMAs, so a CTA per KV group cannot cut kernel time by 10%; the limit is the mma/softmax work (about 190 TFLOP/s on a quiet GPU, flat from 15k to 258k keys: 4.37, 8.43, 34.7, 69.3 ms at 15k, 30k, 129k, 258k keys). Not built. (2) Split-KV: the grid is 16 query tiles x 24 heads = 384 CTAs for a 2,048 chunk and 768 for a 4,096 chunk, enough to fill the GPU, and TFLOP/s is already flat in key count, so the partial-softmax plus reduce would only help chunks under about 1,024 tokens (143 to 152 TFLOP/s at 512 and 1,024 query tokens). Not built. (3) Two-stream overlap: in a single-sequence prefill every layer reads the previous layer's output, so a delta-net scan and a full-attention layer are never independent and there is nothing to overlap; only the mtp or multi-sequence batches could, which is the concurrency phase. Not built. Final `flash_mma`: 4.4 ms (15k keys), 8.4 ms (30k), 34.7 ms (129k), 69.3 ms (258k) for 2,048 query tokens.

Gates for the final tree in one quiet window (`quiet_window.sh`), non-27B suite: every `test result:` line `ok`, 0 failed (lib `94 passed`, codegen 13, spec 35, mtp 15, serving, fp4 and fp8 batches, jit 4, vision_lm 3 and the rest). real27b act default (`SPECRT_27B_SPEC=qwen3.8-27b`, `SPECRT_27B_NVFP4_SPEC=qwen3.8-27b-nvfp4`): `test result: ok. 10 passed; 0 failed; 0 ignored; finished in 242.95s`. real27b exact specs: `test result: ok. 10 passed; 0 failed; 0 ignored; finished in 381.73s`. The remaining 4k and 32k gap to 70% of vLLM is the GEMMs (`gate_up`, `down`, `lin_in` are 41% of kernel time at 32k) and belongs to #441; the non-GEMM part owned here (`flash_mma` 12.7%, gdn 12%, quant 3.4%) is already at its measured limit, so the attention side reaches 54% at 32k on its own (3,719 of 6,849 tok/s) and 48% at 4k.

## #555: where the 4k and 32k prefill gap is now

Quiet GPU (local-llm stopped through `quiet_window.sh`, 6.7 GB used and 0% util at the start). Master at af5e214 with #441's fp4 GEMM, `prefixbench long`, cold, `kv_gb` 12, 3 rounds, all values (`scripts/prefill555.sh`). The nvfp4 spec is the like-for-like one: vLLM serves the same NVFP4 checkpoint.

| prompt | `qwen3.8-27b-nvfp4` tok/s | `qwen3.8-27b` (FP8) tok/s | vLLM quiet (MTP on) | nvfp4 / vLLM (best) |
|---|---|---|---|---|
| 4,096 | 5,884, 5,719, 5,775 | 4,373, 4,355, 4,358 | 8,781 | 67% |
| 32,768 | 5,065, 4,868, 4,880 | 3,828, 3,813, 3,812 | 6,849 | 74% |
| 131,072 | 3,230, 3,158, 3,157 | 2,670, 2,664, 2,661 | 3,764 | 86% |

32k and 128k pass the targets. 4k is 3 points under 70%. The user judged these numbers good enough and closed the phase here.

Timeline (nsys `--trace=cuda` of the nvfp4 spec, `scripts/prefill_timeline.py`). The GPU is busy for 99.9% of the prefill window: 723.2 of 723.6 ms at 4k, 6,844 of 6,848 ms at 32k, no gap over 20 us at 4k and 0.2 ms in total at 32k. There is nothing to win from launch overhead, host syncs or CUDA graphs for prefill chunks. The gap is all kernel time.

Per-kernel, 4,096 tokens, against the #413 vLLM capture (MTP off, 4,096-token request, 518 ms of kernels including one decode step):

| class | specrt kernels, ms (share) | vLLM, ms | ratio |
|---|---|---|---|
| linears, NVFP4 MLP | `gate_up_f4a` 148.0 (20.5%), `down_f4a` 81.0 (11.2%), `quant_rows_fp4` 13.4 | 412 for all linears and the head | |
| linears, FP8 | `lin_in_f8` 82.4 (11.4%), `lin_out_f8` 34.8, `gate_up_f8` 30.2 and `down_f8` 15.6 (the 8 layers after `quant_until = 56`), `qkv_f8` 24.4, `o_f8` 11.5, `lin_ba_mma` 8.0, `quant_rows_fp8_token` 8.6 | | |
| linears total | about 458 | 412 | 1.11x |
| MTP layer on the prompt | 38.9 (`mtp_*_mma`) | 0 (MTP off in the capture) | |
| delta-net | `gdn_chunk_scan` 103.8 (14.4%), `gdn_chunk_prep` 58.1 (8.0%), `gdn_conv_prefill` 12.4, `gdn_gated_norm` 7.6, `gdn_qk_prep` 3.0: 185 | 42 | 4.4x |
| attention | `flash_mma` 28.9 (4.0%) | 29 | 1.0x |
| norms | `rms_split` 8.9 | in Inductor fusions | |
| total | 723 | 518 | 1.40x |

At 32,768 tokens the order changes: `flash_mma` 18.7%, `gate_up_f4a` 17.4%, `gdn_chunk_scan` 12.2%, `lin_in_f8` 9.6%, `down_f4a` 9.5%, `gdn_chunk_prep` 6.8%, `lin_out_f8` 4.1%. Every other class is under 3.5%.

Where the in-model fp4 GEMM loses its 20%. `tests/gemm_speed.rs` `prefill_gemm_fp4_in_model_probe` times `gate_up_f4a_big` and `down_f4a_big` with three kinds of data: the old masked bytes with constant scales, uniform random bytes and scales, and Gaussian values put through `nvfp4::quantize`. For each it measures four L2 states: warm and idle between calls, cold after a 512 MB flush, a back-to-back chain on one weight, and a chain rotating over 4 weight copies. Quiet GPU, median, TFLOP/s:

| | masked, warm | Gaussian, warm | Gaussian, cold L2 | Gaussian, rotating chain |
|---|---|---|---|---|
| `gate_up` 2,048 | 727 | 667 | 635 | 582 |
| `gate_up` 4,096 | 730 | 677 | 656 | 592 |
| `down` 2,048 | 776 | 732 | 658 | 609 |
| `down` 4,096 | 680 | 660 | 648 | 586 |

The #441 "matches vLLM in isolation" figure (738 TFLOP/s) was on the masked data, which has low bit entropy and a constant scale byte. Realistic data costs 8%, a cold L2 another 3 to 10%, and back-to-back calls with fresh weights another 7 to 8%. That is the 20%: in the model, `gate_up` at 4,096 tokens is 2.64 ms (553 TFLOP/s). FlashInfer's CUTLASS NVFP4 kernel (`flashinfer_scaled_fp4_mm`, the one vLLM calls), timed once on the same shapes with real quantised Gaussian data, ran at about 1,000 TFLOP/s cold (1,013 at 2,048 tokens, 915 at 4,096) and 844 to 851 in a rotating chain. So isolated, the fp4 GEMM is 1.4x to 1.6x behind CUTLASS on realistic data. That standalone script was removed: calling FlashInfer outside the service rebuilt vLLM's fp4 JIT module with other build flags, and the service then recompiled it with 20 jobs at its next start. Compare against vLLM only through existing nsys captures.

What is left, by size at 4k:

1. Delta-net prefill, 185 ms against 42 (4.4x, about 143 ms, 20% of prefill). `gdn_chunk_prep` does a scalar forward substitution over the 64 rows of each chunk. Every product in both kernels goes through the three-term hi/lo bf16 split (`mma_split_tile`). FLA uses a block-merged 16x16 inverse and single-term products. This is #418's prefill half.
2. The NVFP4 GEMMs, `gate_up` and `down` about 229 ms. On realistic data they reach 1.4x to 1.6x CUTLASS's time, so closing that is about 70 ms (#441's mainloop: issuing the loads and a `__syncthreads` per k tile from the mma warps, with one block per SM).
3. The FP8 projections, 177 ms. vLLM runs CUTLASS FP8 for these at 410 to 454 TFLOP/s (#413 table above).
4. The MTP layer on the prompt, 39 ms (5%). vLLM's capture had MTP off, so this part is not a gap against the MTP-on vLLM numbers in the table.
