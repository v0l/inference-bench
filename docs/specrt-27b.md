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
  a clean 16384-row matrix. Scales are expanded at load to one f32 per row per 128 columns, so
  head splitting, gate/up interleaving and qkv fusion stay plain row operations. That costs
  0.76 GB on the 27B (#334 tracks shrinking it).
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
| per-row f32 block scales | 0.76 GB | the checkpoint grid is 6 MB; #334 |
| `embed_tokens` + `lm_head` bf16 | 5.09 GB | 248,320 x 5120 each |
| norms, conv, `in_proj_a/b`, A, dt | 0.05 GB | |
| **weights on the device** | **30.2 GB** | MTP (0.48 GB) and vision (0.92 GB) are not loaded |
| linear-attention state per sequence | 151 MB + 5.9 MB conv | 48 layers x 48 heads x 128 x 128 x f32; independent of context |
| KV per token, specrt today (f16 hi/lo pair) | 128 KiB | 16 layers x 4 heads x 256 x (K, V) x 2 planes x 2 bytes |
| KV per token in bf16 / fp8 | 64 KiB / 32 KiB | what vLLM's `--kv-cache-dtype fp8` stores |
| decode scratch | 3 MB | |
| prefill scratch | 0.24 / 0.06 / 0.02 GB at 8k / 32k / 128k | the chunk shrinks as context grows (`SCORE_BUDGET_FLOATS`) |
| attention score buffer | 0.27 GB | above 4096 context |
| CUDA graph, context, JIT module | about 0.5 GB | measured as `device_bytes` minus the rest on the slice |

| context | KV specrt (hi/lo) | KV if bf16 | KV if fp8 | total specrt, one sequence |
|---|---|---|---|---|
| 8,192 | 1.07 GB | 0.54 GB | 0.27 GB | about 32.5 GB |
| 32,768 | 4.29 GB | 2.15 GB | 1.07 GB | about 35.5 GB |
| 131,072 | 17.18 GB | 8.59 GB | 4.29 GB | about 48.4 GB |

Beside the live vLLM (which holds about 58 GB of the card today, `nvidia-smi`), about 38 GB are
free: enough for the 30 GB of weights at 8k context, but not with headroom. The full-model runs
below therefore need `local-llm.service` stopped. Everything above was checked on the 4-layer
real slice (3.0 GB on the device) while vLLM was running.

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

Needs `local-llm.service` stopped (see the memory budget). Commands, in order:

```
cd crates/specrt
python scripts/reference.py real-prompts Qwen/Qwen3.8-27B-FP8 out
cargo test --release --test real27b real_27b_generates_greedy_and_dumps_logits -- --ignored --nocapture
python scripts/reference.py real-full Qwen/Qwen3.8-27B-FP8 out out/real27b-specrt.json
python scripts/reference.py real-full Qwen/Qwen3.8-27B-FP8 out out/real27b-specrt.json act
cargo test --release --test real27b real_27b_greedy_and_logits_match -- --ignored --nocapture
bench tests/fixtures/qwen3.8-27b.toml 128,1024,4096 128
SPECRT_FP8_ACT=1 bench tests/fixtures/qwen3.8-27b.toml 128,1024,4096 128
```

Only the first GPU test and the two `bench` runs need the GPU. The CPU reference can run with vLLM
up. Results: pending the GPU window.
