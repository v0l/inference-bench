# specrt on the real Qwen3.8 27B (FP8)

Issue #320. The checkpoint is `Qwen/Qwen3.8-27B-FP8` from the HF cache (68 shards: `layers-N`,
`outside`, `mtp`). The spec is `crates/specrt/tests/fixtures/qwen3.8-27b.toml` with
`quant = "fp8_block128"`. `Spec::from_hf_config` derives that quant from the config's
`quantization_config` (`fp8`, `e4m3`, `weight_block_size [128, 128]`) and refuses any other scheme.

## What is in place

- **Sharded loader** (`weights.rs`, `Checkpoint`). It reads `model.safetensors.index.json` (or
  one `model.safetensors`), mmaps every shard and reads only the headers up front. The runtime
  then pulls one layer at a time, uploads it, synchronises and calls `madvise(DONTNEED)` on the
  maps, so host memory holds one fused layer at most. bf16 tensors are borrowed straight from
  the map with no copy. `Runtime` reports `load_secs`, `weight_secs`, `weight_bytes`,
  `device_bytes` and `peak_host_rss_bytes` (VmHWM), and `bench` prints them.
- **FP8 weights.** Every projection is e4m3 with one bf16 `weight_scale_inv` per 128 x 128 block:
  `q/k/v/o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate/up/down_proj`. These stay bf16:
  `in_proj_a`, `in_proj_b` (48 rows each), the conv, norms, `embed_tokens` and `lm_head`. The
  96 gate rows are now their own bf16 linear (`lin_ba`) for every quant, so the FP8 `in_proj` is
  a clean 16384-row matrix. Scales stay as the checkpoint's 128 x 128 grid (f32 on the device).
  Each FP8 matrix's scale buffer starts with one u32 per fused row, the offset of that row's grid
  row, so head splitting, gate/up interleaving and qkv fusion only remap rows (#334). On the 27B
  that is 15.6 MB of map plus 5.9 MB of grid, 21.5 MB in all, down from 0.76 GB of per-row
  expanded scales.
- **Decode GEMV** (`row_dot<WF_FP8>`, `linear_fp8`). Each lane streams 16 bytes of weight
  (`ld.global.nc.L1::no_allocate`), decodes e4m3 in registers with two integer ops and one
  multiply, and applies the block scale once per 128 columns.
- **Prefill GEMM, two paths.**
  - Default: `gemm_tc` with FP8 weight tiles in shared memory, converted to bf16 per fragment and
    run through `mma.m16n8k16.bf16` against the hi/lo split activations. The block scale is
    applied per 64-wide k tile. Activations stay effectively fp32, so this is exact to the
    dequantised weights.
  - `SPECRT_FP8_ACT=1`: `gemm_fp8`, a native `mma.sync.m16n8k32.e4m3.e4m3.f32` kernel. A
    `quant_rows_fp8` pass quantises each activation row per 128-column block (dynamic, the
    checkpoint's own `activation_scheme`), and the kernel applies activation scale x weight
    scale per 128-wide k block. NVRTC compiles it for `compute_120`.

## Correctness

| check | fixture | result |
|---|---|---|
| tiny FP8, 4 shards, block scales (`tiny-qwen3.5-fp8`, built by `reference.py tiny-fp8` from the head_dim 256 fixture) | decode 32 tokens, per layer | max abs 3e-6 vs transformers on the dequantised weights |
| same | chunked prefill 1024 tokens | layers 7e-5, logits 5e-5, state 1.3e-5 |
| same | logits + 64 greedy tokens | 4e-5, greedy identical |
| same | batch 4 decode vs single | identical |
| real 27B, first 4 layers (`reference.py real-slice`, real FP8 shards, vocab cut to 32768 rows that hold the prompt) | 64-token prefill, per layer | max error / max abs 3e-6, logits 5e-6, top-1 64/64 |
| same | 64 recurrent decode steps | logits 6e-7 relative |
| native FP8 mma (`SPECRT_FP8_ACT=1`), tiny FP8 | prefill logits vs weight-only reference | 0.30, against a 0.29 gap between the weight-only and activation-quantised references, so it stays inside the envelope that FP8 activations cost |

A mutation check: doubling the e4m3 exponent bias in the kernel fails all three FP8 reference
tests.

The whole-model results are under "Real-model run" below.

The full-model oracle is `reference.py real-full`. It runs the whole 27B on the CPU one layer at a
time from the FP8 shards, dequantised to f32, for all five prompts at once. It takes 108 s on 24
threads. Peak anonymous RSS is 3.7 GB; VmHWM shows 33 GB, but that is page cache from the shard
mmaps, which the kernel can drop. It writes the full-sequence logits, a per-layer trace of prompt
0 and the reference argmax. Run with `act` it also quantises every projection input per 128
block, the way vLLM's dynamic FP8 does. The gap between the two runs is the tolerance envelope.

Tolerance rule (`tests/real27b.rs`, `real_27b_greedy_and_logits_match_the_fp8_reference`):
specrt's logits error against the weight-only FP8 reference must be no larger than the error
dynamic FP8 activations themselves introduce, and greedy tokens must be identical for 32 tokens
on all 5 prompts. "bf16 transformers" here means the FP8 checkpoint dequantised to f32. The bf16
checkpoint is not in the cache and is 54 GB.

## Memory budget (RTX PRO 6000, 96 GB)

From the checkpoint headers and the allocation code in `runtime.rs`:

| item | size | notes |
|---|---|---|
| FP8 projection weights | 24.33 GB | 64 layers |
| FP8 block scales: grid plus row map | 0.02 GB | 5.9 MB grid, 15.6 MB map; was 0.76 GB per-row before #334 |
| `embed_tokens` + `lm_head` bf16 | 5.09 GB | 248,320 x 5120 each |
| norms, conv, `in_proj_a/b`, A, dt | 0.05 GB | |
| **weights on the device** | **29.5 GB** | MTP (0.48 GB) and vision (0.92 GB) are not loaded |
| linear-attention state per sequence | 151 MB + 5.9 MB conv | 48 layers x 48 heads x 128 x 128 x f32; independent of context |
| KV per token, specrt today (f16 hi/lo pair) | 128 KiB | 16 layers x 4 heads x 256 x (K, V) x 2 planes x 2 bytes; the serving layout (#335, `docs/specrt-scheduler.md`) stores one f16 plane, 64 KiB |
| KV per token in bf16 / fp8 | 64 KiB / 32 KiB | what vLLM's `--kv-cache-dtype fp8` stores |
| decode scratch | 3 MB | |
| prefill scratch | 0.24 / 0.06 / 0.02 GB at 8k / 32k / 128k | the chunk shrinks as context grows (`SCORE_BUDGET_FLOATS`) |
| attention score buffer | 0.27 GB | above 4096 context |
| CUDA graph, context, JIT module | about 0.5 GB | measured as `device_bytes` minus the rest on the slice |

| context | KV specrt (hi/lo) | KV if bf16 | KV if fp8 | total specrt, one sequence |
|---|---|---|---|---|
| 8,192 | 1.07 GB | 0.54 GB | 0.27 GB | about 31.8 GB |
| 32,768 | 4.29 GB | 2.15 GB | 1.07 GB | about 34.8 GB |
| 131,072 | 17.18 GB | 8.59 GB | 4.29 GB | about 47.7 GB |

Beside the live vLLM (58.2 GB used after its KV was trimmed to 14 GB) 38.9 GB are free. That is
enough for the 29.5 GB of weights plus 8k of context, and that is how the real-model run below
was done, with max context capped at 8,192. Longer contexts beside vLLM would need fp8 KV.

### Scale grid and row map (#334)

The kernels read a row's scales as `scales[map[row] + k / 128]`. `gemm_tc` and `gemm_fp8` load
the map entry once per column before the k loop; `row_dot` loads it once per row. Kernel
medians from `nsys` on the full 27B, 1,024-token prompt, old and new binaries alternated, min of
3 runs each (vLLM serving beside it):

| kernel | per-row scales | grid + map | ratio |
|---|---|---|---|
| `gate_up_act` decode GEMV | 137 us | 138 us | 1.009 |
| `lin_in_norm` decode GEMV | 76 us | 77 us | 1.009 |
| `qkv_proj_norm` decode GEMV | 69.8 us | 69.8 us | 0.999 |
| `down_proj` decode GEMV | 66.5 us | 62.5 us | 0.94 |
| `o_proj` decode GEMV | 25.2 us | 23.7 us | 0.94 |
| `lin_out_proj` decode GEMV | 25.5 us | 24.5 us | 0.96 |
| `gate_up` prefill GEMM | 10.9 ms | 10.8 ms | 0.99 |
| `down` prefill GEMM | 5.33 ms | 5.19 ms | 0.97 |
| `lin_in` prefill GEMM | 4.70 ms | 4.62 ms | 0.98 |

The real 27B check still passes on the default path: logits 1.98e-4 of the largest logit and
greedy identical for 32 tokens on all 5 prompts. The greedy tokens are the same as before the
change. The tiny FP8 tests give the same errors as before on both paths (default logits 4.3e-5,
`SPECRT_FP8_ACT=1` 0.299).

## Prefill GEMM: dequant-to-bf16 against native FP8 mma

Measured on the real 4-layer slice (`bench out/qwen3.8-27b-slice`, vLLM running beside it at
97% GPU use, so absolute numbers are low and noisy). These are medians of 6, two rounds each:

| prompt | dequant to bf16 (default) | native e4m3 mma (`SPECRT_FP8_ACT=1`) |
|---|---|---|
| 128 | 8,700 to 11,000 tok/s | 11,100 to 17,300 tok/s |
| 1,024 | 10,400 to 10,500 | 17,400 to 20,400 |
| 4,096 | 10,000 to 10,300 | 20,600 to 21,000 |

On this slice the GEMMs are 87% of prefill time (`nsys`: `gate_up` 42%, `down` 21%, `lin_in`
13%, `lin_out` 6%). The native path roughly doubles prefill at 1k and 4k tokens. It is opt-in
because it adds the activation-quantisation error, which is as large as the whole FP8
envelope. The weight-only default is exact to the dequantised checkpoint. Running the default
path without the hi/lo activation split (`SPECRT_UNSPLIT=...`) reaches about 13,600 tok/s but
fails the 1e-3 reference bound.

Decode is the same on both paths (single-token GEMVs): about 360 tok/s for 4 layers, which
works out to about 22 tok/s for 64 layers if the whole model scales the same. Measured bandwidth
of the decode GEMVs here is about 700 GB/s, against 825 GB/s for a plain read on this card
under the same load (the card sits at its 300 W limit with vLLM running).

## Real-model run

Run on 8 Nov 2026 next to the live vLLM (`local-llm.service` serving agents, about 100% busy,
KV trimmed to 14 GB so 38.9 GB of the card was free). Max context was capped at 8,192 for
every specrt run. The service was not touched; its journal logged no errors during the runs,
and free memory went back to 38,953 MiB after each process exited.

### Load

| | |
|---|---|
| weights on the device | 30.24 GB, uploaded in 5.6 s from the page cache |
| device memory, 8k context | 31.7 GB (`bench`), 32.0 GB in the greedy test |
| whole load (JIT cache warm) | 5.8 s; 11.6 s on the first run |
| peak host RSS | 5.3 to 5.5 GB |

The peak host RSS is VmHWM, so it includes the mapped shard pages of the layer being uploaded
and the 2.5 GB `lm_head`, which the fused bf16 head copies once. Nothing close to the 25 GB
limit.

### Correctness, whole model

`tests/real27b.rs`, 5 prompts of 64 tokens (prose, Python, science, a letter, a word problem),
32 greedy tokens each, then the logits of all 96 positions against the CPU oracle:

| path | logits max error / max abs logit vs its own reference | vs weight-only reference | greedy, 32 tokens x 5 |
|---|---|---|---|
| default (dequant to bf16 tiles, fp32 activations) | 2.0e-4 | 2.0e-4 | identical on all 5 |
| `SPECRT_FP8_ACT=1` (native e4m3 mma, dynamic per-token activations) | 0.087 against the activation-quantised reference | 0.32 | 3 of 5 identical; the other two flip at a near tie (0.10 and 0.17 logit margin in the reference) |

The envelope is the gap between the activation-quantised and weight-only references: 0.33 of
the largest logit. Both paths are inside it. The default path is 1600 times inside it and is the
one that meets "greedy identical for 32 tokens on 5 prompts". The native path's own error
against a reference that quantises activations the same way is 0.087. Its greedy test passes
only where the reference itself is within 0.25 logits of a tie (`NEAR_TIE`). The 0.087 is not
explained further here: on the tiny fixture the same kernel lands inside the envelope too, and a
per-layer trace of prompt 0 (`real27b-ref-act.layers.bin`) is there to dig into it.

Sample continuations (default path): "...the cost of a book fell" continues " the cost of paper
fell, the number of books printed in Europe rose from a few million..."; the Python prompt
continues with the correct `out.append(b[j])` and `out.extend(a[i:])`; the word problem
continues "...The first train has a head start of 1.5".

### Speed: shared GPU, vLLM serving, indicative only

`scripts/bench27b.sh`: 3 rounds alternating the two paths, 2 runs per round, so 6 runs per
mode, single stream, 128 new tokens. All of these tok/s figures are from a shared GPU with vLLM
serving and are indicative only. The clean comparison is #307's window.

| prompt | prefill, default | prefill, native FP8 | decode, default | decode, native FP8 |
|---|---|---|---|---|
| 128 | 553 tok/s (489 to 584) | 1,062 (1,014 to 1,112) | 21.6 (20.6 to 24.4) | 23.8 (22.5 to 24.7) |
| 1,024 | 611 (542 to 652) | 1,308 (1,105 to 1,333) | 23.4 (20.4 to 24.2) | 22.1 (20.3 to 23.7) |
| 4,096 | 558 (527 to 615) | 1,081 (1,069 to 1,269) | 22.3 (19.8 to 22.9) | 21.4 (19.8 to 23.1) |

Kernel times per call from `nsys` (1,024-token prompt, 64 new tokens). These per-call times are
more robust than wall clock under sharing:

| kernel | default | native FP8 | rate |
|---|---|---|---|
| `gate_up` prefill GEMM, 1024 x 34816 x 5120 | 11.0 ms | 5.0 ms | 33 vs 72 TFLOP/s |
| `down` prefill GEMM, 1024 x 5120 x 17408 | 5.2 ms | 2.0 ms | 35 vs 90 TFLOP/s |
| `lin_in` prefill GEMM, 1024 x 16384 x 5120 | 4.3 ms | 1.5 ms | 40 vs 111 TFLOP/s |
| `gate_up_act` decode GEMV (178 MB FP8) | 265 us | same | 672 GB/s |
| `down_proj` decode GEMV (89 MB) | 126 us | same | 709 GB/s |
| `lin_in_norm` decode GEMV (84 MB) | 145 us | same | 581 GB/s |
| `lm_head_norm` decode GEMV (2.5 GB bf16) | 3.4 ms | same | 744 GB/s |

Decode reads 26.9 GB per token. At the 825 GB/s this card reaches for a plain read under the
same load, the ceiling is 30.6 tok/s. specrt runs at 21 to 24 tok/s, about 75% of it. The bf16
`lm_head` is 2.5 GB of the 26.9 GB, about 10% of decode time.

### Against vLLM

vLLM serves the NVFP4 build of this model (MLPs in 4-bit, so about 17 GB of weights read per
token, against specrt's 27 GB) with MTP speculative decoding at 2.8 tokens per step. Its
single-stream median is 162.5 tok/s (`docs/qwen3.8-27b-vllm-baseline.md`). specrt's 22 tok/s
under sharing is about 7x lower. The gap has three parts, roughly multiplicative:

- MTP: about 2.8x, the measured acceptance length. specrt has no MTP yet.
- Bytes per token: NVFP4 MLPs read about 1.6x fewer bytes than FP8. The NVFP4 issue covers it.
- Sharing: these numbers ran beside vLLM at 100% load and the 300 W cap, so the clean number
  will be higher. #307 measures it.

Without MTP the same vLLM deployment would be about 162.5 / 2.8 = 58 tok/s; with NVFP4's
byte count, specrt's 75%-of-bandwidth decode would land at about 35 to 40 tok/s on a free GPU.
This is an estimate, not a measurement.

Prefill: vLLM's logged median is 1,782 tok/s on mixed short prompts. specrt's native FP8 path
reaches 1,062 to 1,308 tok/s here under sharing, and the default path about half of that.

### Commands

```
cd crates/specrt
python scripts/reference.py real-prompts Qwen/Qwen3.8-27B-FP8 out
cargo test --release --test real27b real_27b_generates_greedy_and_dumps_logits -- --ignored --nocapture
SPECRT_FP8_ACT=1 cargo test --release --test real27b real_27b_generates_greedy_and_dumps_logits -- --ignored --nocapture
python scripts/reference.py real-full Qwen/Qwen3.8-27B-FP8 out out/real27b-specrt-weight.json
python scripts/reference.py real-full Qwen/Qwen3.8-27B-FP8 out out/real27b-specrt-weight.json act
cargo test --release --test real27b real_27b_greedy_and_logits_match -- --ignored --nocapture
./scripts/bench27b.sh
```

The GPU steps need about 32 GB free (the bench script stops below 36 GB, and if vLLM logs an
error). The CPU references take about 95 s each on 24 threads, with 3.7 GB of anonymous RSS.
