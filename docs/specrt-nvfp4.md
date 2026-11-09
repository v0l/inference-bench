# specrt on the NVFP4 27B (the checkpoint vLLM serves)

Issue #321. The checkpoint is `orcarouter/Qwen3.8-27B-Uncensored-NVFP4`, snapshot
`d350e02d82203d5bf07a1e203c997581f235f407` in the HF cache, 23.0 GiB in 5 shards plus
`model-extra-00001-of-00001.safetensors` (MTP). `config.json` says `quant_method =
"compressed-tensors"`, `format = "mixed-precision"`, compressed-tensors 0.18.0, transformers 5.15.

## Tensor layout, read from the safetensors headers

`N` is the layer index, `R` = `model.language_model`. Shapes are `[rows, cols]` as stored.

### Quantisation groups (`quantization_config.config_groups`)

| group | format | weights | activations | targets |
|---|---|---|---|---|
| `group_0` | `float-quantized` | e4m3, 8 bit, `strategy = "channel"`, symmetric, static | e4m3, `strategy = "token"`, `dynamic = true` | `self_attn.{q,k,v,o}_proj` (16 layers), `linear_attn.{in_proj_qkv, in_proj_z, out_proj}` (48 layers), and `mlp.{gate,up,down}_proj` of layers 56 to 63 |
| `group_1` | `nvfp4-pack-quantized` | e2m1, 4 bit, `strategy = "tensor_group"`, `group_size = 16`, scales e4m3 | e2m1, `group_size = 16`, `dynamic = "local"`, scales e4m3, static global scale | `mlp.{gate,up,down}_proj` of layers 0 to 55 |

Not quantised (`ignore`): the vision tower, `lm_head`, `in_proj_a`, `in_proj_b`, the gated norm
and conv of every linear-attention layer, and everything under `mtp.`. `kv_cache_scheme` is
static per-tensor fp8 with `k_scale`/`v_scale` stored per attention layer.

The last 8 MLPs being FP8 instead of NVFP4 is not in the issue text; it is in the targets list
and the headers agree (layers 56 to 63 have `mlp.*.weight` e4m3, the rest `weight_packed`).

### NVFP4 MLP, layers 0 to 55

| tensor | dtype | shape |
|---|---|---|
| `R.layers.N.mlp.gate_proj.weight_packed` | U8 | [17408, 2560] |
| `R.layers.N.mlp.gate_proj.weight_scale` | F8_E4M3 | [17408, 320] |
| `R.layers.N.mlp.gate_proj.weight_global_scale` | F32 | [1] |
| `R.layers.N.mlp.gate_proj.input_global_scale` | F32 | [1] |
| `R.layers.N.mlp.up_proj.*` | same as gate | same as gate |
| `R.layers.N.mlp.down_proj.weight_packed` | U8 | [5120, 8704] |
| `R.layers.N.mlp.down_proj.weight_scale` | F8_E4M3 | [5120, 1088] |
| `R.layers.N.mlp.down_proj.weight_global_scale` | F32 | [1] |
| `R.layers.N.mlp.down_proj.input_global_scale` | F32 | [1] |

Byte `b` of row `r` of `weight_packed` holds columns `2b` (low nibble) and `2b + 1` (high
nibble). A nibble is `s eem`: sign bit 3, magnitude index `0..7` mapping to
`0, 0.5, 1, 1.5, 2, 3, 4, 6` (`compressed_tensors/compressors/nvfp4/helpers.py`).
`weight_scale[r, g]` covers columns `16g .. 16g + 15`. The dequantised weight is

```
w[r, c] = e2m1(nibble(r, c)) * e4m3(weight_scale[r, c / 16]) / weight_global_scale
```

(compressed-tensors `dequantize`: `scale = scale / global_scale; w = x_q * scale`.) The
global scales are stored as the divisor, so they are large: `weight_global_scale` runs from 2272
to 19968, `input_global_scale` from 7.6 to 940. Gate and up share both global scales in every
layer, so the fused gate/up matrix has one. The group scales sit between 4 and 448 (none is zero
or negative). The product of an e2m1 value and an e4m3 scale has at most 5 significant bits, so
`e2m1 * e4m3` is exact in bf16; only the global scale needs f32.

Activations (vLLM `scaled_fp4_quant`, `ref_nvfp4_quant`): per row and 16-column group,
`sx = e4m3(input_global_scale * amax / 6)` (clamped to 448), `x_q = e2m1_rne(clamp(x *
input_global_scale / sx, -6, 6))`. The GEMM is `sum(x_q * sx * w_q * sw) * alpha` with
`alpha = 1 / (input_global_scale * weight_global_scale)`.

### FP8 per-channel projections

| tensor | dtype | shape |
|---|---|---|
| `R.layers.N.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] (q and the output gate, interleaved per head) |
| `R.layers.N.self_attn.{k,v}_proj.weight` | F8_E4M3 | [1024, 5120] |
| `R.layers.N.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] |
| `R.layers.N.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] |
| `R.layers.N.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] |
| `R.layers.N.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] |
| `R.layers.N.mlp.{gate,up}_proj.weight`, N = 56..63 | F8_E4M3 | [17408, 5120] |
| `R.layers.N.mlp.down_proj.weight`, N = 56..63 | F8_E4M3 | [5120, 17408] |
| every one of these `.weight_scale` | BF16 | [rows, 1] |

`w[r, c] = e4m3(weight[r, c]) * weight_scale[r]`. There is no `weight_scale_inv` and no
128 x 128 grid, unlike `Qwen/Qwen3.8-27B-FP8`.

### Unquantised

| tensor | dtype | shape |
|---|---|---|
| `R.embed_tokens.weight`, `lm_head.weight` | BF16 | [248320, 5120] |
| `R.layers.N.linear_attn.in_proj_{a,b}.weight` | BF16 | [48, 5120] |
| `R.layers.N.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] |
| `R.layers.N.linear_attn.{A_log, dt_bias}` | BF16 | [48] |
| `R.layers.N.linear_attn.norm.weight` | BF16 | [128] |
| `R.layers.N.self_attn.{q,k}_norm.weight` | BF16 | [256] |
| `R.layers.N.self_attn.{k,v}_scale` | BF16 | [1] (fp8 KV cache scales, about 0.02 to 0.3) |
| `R.layers.N.{input,post_attention}_layernorm.weight`, `R.norm.weight` | BF16 | [5120] |

Shards: `model-00001` holds layers 0 to 5, `embed_tokens` and the vision tower; `00002` to
`00004` hold layers 5 to 60; `00005` holds 60 to 63, `lm_head` and the final norm; the extra
shard holds `mtp.*` (bf16).

### Bytes

| part | GB |
|---|---|
| FP8 attention and linear-attention projections | 7.21 |
| FP8 MLPs, layers 56 to 63 | 2.14 |
| NVFP4 MLP packed weights | 7.49 |
| NVFP4 group scales | 0.94 |
| `lm_head` (bf16) | 2.54 |
| `embed_tokens` (bf16) | 2.54 |
| MTP (bf16) | 0.85 |
| vision | 0.92 |
| norms, conv, `in_proj_a/b`, scales | 0.06 |

Decode reads every projection plus `lm_head` once per token: 20.3 GB, against 26.9 GB for the
FP8 checkpoint. The NVFP4 MLPs are 8.4 GB of it; at FP8 they would be 15.0 GB.

## Spec and loader

`tests/fixtures/qwen3.8-27b-nvfp4.toml` (`min_engine = 4`): `quant = "fp8_channel"` for the
projections, `[mlp] quant = "nvfp4_g16"` and `quant_until = 56`, so `Spec::mlp_quant(layer)` is
NVFP4 below 56 and FP8 from there. `Spec::from_hf_config` derives exactly this from the
compressed-tensors `config_groups` (it reads each group's format, type, bits, strategy and group
size and maps every target to a layer and module), and refuses anything else.

`src/nvfp4.rs` holds the format: e2m1 decode and round to nearest, `Fp4` (borrowed packed bytes
and e4m3 group scales, the weight and input global scales), gate/up interleave and the CPU
dequant. FP8 per-channel weights go through the existing FP8 path with the per-row scale
repeated per 128 columns. On the device the scale buffer of an `Fp4` matrix is a 16-byte header
(`1 / weight_global_scale`, `input_global_scale` as f32) followed by the raw e4m3 group scales.

Dequant oracle: `reference.py nvfp4-dequant out` dequantises 14 tensors' rows (NVFP4 gate, up and
down of layers 0, 30 and 55; FP8 MLPs of 56 and 63; FP8 `in_proj_qkv`, `in_proj_z`, `o_proj`)
with compressed-tensors' own `unpack_fp4_from_uint8` and `dequantize`, in f32.
`tests/nvfp4.rs` loads the same layers through `Checkpoint::layer` (so through the gate/up
interleave and the qkv/z fusion) and matches all 304,128 values bit for bit. Swapping the
nibble order fails it (4,804 of the first 5,120 values differ).

## Kernels

`templates/fp4.cu`, rendered only when `[mlp] quant = "nvfp4_g16"`:

- **Decode GEMV** (`gate_up_act_f4`, `down_proj_f4`). One warp per output row; each lane
  streams one 16-column group (8 bytes, `ld.global.nc.L1::no_allocate`) and its e4m3 scale per
  step, decodes e2m1 with two integer ops and a multiply, and applies `1 / weight_global_scale`
  once per row. The normalised input sits in shared memory transposed by group
  (`xt[c % 16][c / 16]`), so the 32 lanes of a warp read 32 consecutive floats for each nibble
  position and the shared loads are conflict free. The gate/up kernel writes the SwiGLU output in
  the same transposed order, which is what `down_proj_f4` reads.
- **Prefill GEMM, two paths.**
  - Default: `gemm_f4w`, the bf16 tensor-core GEMM with NVFP4 weight tiles. A 64-wide k tile of
    a weight row is 32 packed bytes plus its 4 group scales (one 48-byte smem row). Each
    fragment is decoded straight to bf16 (`e2m1 * e4m3` is exact in bf16) and run through
    `mma.m16n8k16.bf16` against the hi/lo split activations, so the result is exact to the
    dequantised weights. The global scale is applied in the epilogue.
  - `[mlp] act_quant = "fp4"`: `gemm_f4a`, native
    `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64...e2m1.e2m1.f32.ue4m3`.
    `quant_rows_fp4` quantises each activation row per 16-column group the way vLLM's
    `scaled_fp4_quant` does (`sx = e4m3(input_global_scale * amax / 6)`, `cvt.rn.satfinite.e2m1x2`),
    and the GEMM multiplies by `alpha = 1 / (weight_global_scale * input_global_scale)` at the
    end. Scales sit in smem next to their row (a 128-wide k tile is 64 packed bytes plus 8 scale
    bytes per row); each thread hands the mma its row's four scale bytes as one 32-bit operand.
    This is the same per-row, per-16 scale layout CUTLASS uses before vLLM's
    `swizzle_blockscale` reorders it into 128 x 4 tiles for TMA; with plain `ld.shared` there is
    no need for the swizzle.
- The block-scaled mma is refused by ptxas for `sm_120` and accepted for `sm_120a` and
  `sm_120f`. NVRTC 13.2 compiles it for `compute_120a`, so `Jit` switches a module to
  `compute_120a` when it has the native kernels (`Module::arch_features`) and otherwise keeps
  `compute_120`. A one-warp probe of the fragment layout (random e2m1 and e4m3 inputs against a
  CPU dot product) matched exactly.

## Correctness

| check | fixture | result |
|---|---|---|
| tiny NVFP4 (`tiny-qwen3.5-nvfp4`, `reference.py tiny-nvfp4`: layers 0 and 1 NVFP4 MLPs, layer 2 FP8 channel, everything else FP8 channel or bf16, the real checkpoint's mix in small) | decode 32 tokens, per layer | max abs 4e-6 against transformers on the dequantised weights |
| same | chunked prefill 1,024 tokens | layers 7e-5, logits 5e-5, state 1.2e-5, every tile shape (`[runtime] tile = "big"`, `"small"`, `"narrow"`) |
| same | logits and 64 greedy tokens | 3.8e-5, greedy identical |
| same | batch 4 decode against single | identical |
| same, native fp4 (`[mlp] act_quant = "fp4"`) | 1,024-token prefill against a reference that quantises MLP inputs like vLLM | layer 0 rms 0.0028 against the 0.097 the activation quantisation itself costs; logits rms 0.019 against 0.127 |
| real 27B, first 4 layers (`reference.py real-slice`) | 64-token prefill and 64 decode steps | per-layer max error / max abs at most 2.5e-5, logits 1e-5, decode 8e-7, top-1 64/64 |
| real 27B, all 64 layers (`real_27b_per_layer_trace_matches_the_cpu_oracle`) | 64-token prefill, per layer | at most 6.7e-5 relative on every layer |

Mutation checks: flipping the e2m1 sign bit in the GPU decode fails the tiny decode, prefill
and logits tests; scaling `alpha` by 1.1 fails the native test (layer 0 rms 0.22).

### Whole model against the CPU oracle

`reference.py real-full orcarouter/Qwen3.8-27B-Uncensored-NVFP4` now reads compressed-tensors
checkpoints: it dequantises with the same formulas as compressed-tensors (bit for bit, above)
and, with `act`, quantises every MLP input to NVFP4 with the checkpoint's `input_global_scale`
and every FP8 projection input per token, which is what vLLM does on this checkpoint. Same 5
prompts of 64 tokens and 32 greedy tokens as the FP8 run (`tests/real27b.rs`,
`SPECRT_27B_SPEC=qwen3.8-27b-nvfp4`). 131 s per reference on 16 threads.

| path | logits max error / max abs vs its own reference | KL(weight-only reference \|\| ours) | top-1 vs weight-only | greedy, 32 x 5 |
|---|---|---|---|---|
| default (NVFP4 weights dequantised in the GEMM, f32 activations) | 1.8e-3 | 1.2e-9 | 100% | identical on all 5 |
| `act_quant = "fp8"`, `[mlp] act_quant = "fp4"` (native fp4 MLPs, native FP8 projections, dynamic activations) | 0.44 against the act reference | 0.0093 | 97.1% | 4 of 5; the fifth flips at step 6 |
| the act reference itself, for scale | 0.22 from weight-only | 0.0105 | 97.3% | |

The envelope here is what quantising the activations costs, measured on the real checkpoint:
KL 0.0105 and 97.3% top-1 between the weight-only and activation-quantised oracles. The native
path sits inside it (KL 0.0093, top-1 97.1%) and the test requires KL within 1.5x and top-1
within 2 points. Its max error against the act oracle (0.44) is larger than the gap between the
oracles because the two quantise the same values from slightly different inputs (bf16 hi/lo
planes against f32), and a group whose amax lands on an e4m3 or e2m1 rounding boundary flips a
whole step; that grows through 64 layers, so the max is a poor gate and KL and top-1 are used.
One known difference: `act_quant = "fp8"` quantises FP8 projection inputs per 128 columns (the
scheme of the block-128 checkpoint), while this checkpoint and the oracle use one scale per
token. Per 128 is the finer of the two, so it is not a source of extra error.

### Against vLLM

`scripts/vllm_logprobs.py out` sends 20 prompts (the 5 above plus 15 short ones: code, SQL,
maths, prose, translation) to the live vLLM (`local-llm.service`, this checkpoint, MTP on) and
records 64 greedy tokens and the top-20 logprobs at each of them. `real_27b_nvfp4_agrees_with_vllm_greedy_and_logprobs`
runs specrt on the same prompts, then teacher-forces vLLM's tokens and computes KL(vLLM ||
specrt) on vLLM's top-20 plus one bucket for the rest.

| | specrt default | specrt native fp4 + fp8 | vLLM against itself, two runs |
|---|---|---|---|
| teacher-forced top-1 = vLLM's token | 1,236 / 1,280 (96.6%) | 1,220 / 1,280 (95.3%) | |
| KL mean / median / p99 | 0.0097 / 0.0025 / 0.087 | 0.0137 / 0.0034 / 0.107 | |
| greedy tokens equal to vLLM's, 20 x 64 | 494 / 1,280 | 582 / 1,280 | 913 / 1,280 |
| greedy agreement, first 5 prompts | 138 / 320 tokens before the first divergence | 138 / 320 | 110 / 320 |
| vLLM's own logprob margin at the first divergence | 0 to 0.63, median 0.125 | 0 to 0.375 | |

The issue's "90% of tokens for 64 tokens on 5 prompts" cannot hold against vLLM here: vLLM
does not agree with itself at that level (110 of 320 on a second run, 913 of 1,280 overall),
because it serves this model with MTP speculation and batch-dependent kernels, so greedy
decoding is not reproducible. Every specrt divergence is at a near tie in vLLM's own
distribution (margin at most 0.63 nats, median 0.125). The gate in the test is therefore
teacher-forced: top-1 at least 90% (96.6%) and mean KL at most 0.05 (0.0097). Against the CPU
oracle, where the reference is deterministic, greedy is identical on all 5 prompts.

The first attempt read `prompt_logprobs` from vLLM. On this deployment those are wrong (the
prompt tokens sit at ranks in the tens of thousands with logprob -11 to -15, and the top-20
mass is 1 to 5%): MTP or prefix caching corrupts prompt-side logprobs. The script reads the
generation-side `logprobs` instead, whose top-20 hold 66 to 99% of the mass.

## Memory

| item | size |
|---|---|
| weights on the device | 23.2 GB: 22.9 GB as stored in the checkpoint (MTP and vision not loaded) plus 0.3 GB of FP8 channel scales |
| device memory, 8k context, one sequence | 24.3 to 25.4 GB |
| load | 5.0 to 5.5 s to upload, 14 s with a cold JIT |
| peak host RSS | 5.3 to 5.5 GB |

The FP8 per-channel scales use #334's row map with each row's single scale repeated 40 times
in its own grid row (0.3 GB), so the existing FP8 kernels read them unchanged.

## Speed

All on the shared GPU: vLLM serving at 92 to 100% utilisation and the card at its 300 W cap
the whole time, max context 8,192, 128 new tokens, medians of 3, two rounds
(`scripts/bench27b-nvfp4.sh`). Indicative, as the FP8 numbers in `specrt-27b.md` were.

Memory bandwidth on this card (`scripts/bandwidth.cu`, 2 GiB, best of 10): a streaming read
reaches 1,640 to 1,660 GB/s even beside vLLM, a copy 800 to 1,360 GB/s (read plus write).

### Single stream

| prompt | prefill, default | prefill, native fp4 + fp8 | decode, default | decode, native |
|---|---|---|---|---|
| 128 | 493 to 588 tok/s | 1,242 to 1,420 | 24.3 to 25.6 tok/s | 23.9 to 28.9 |
| 1,024 | 663 to 757 | 1,875 to 1,926 | 26.6 to 27.5 | 27.7 to 28.5 |
| 4,096 | 728 to 783 | 1,643 to 1,830 | 25.1 | 23.3 to 25.4 |

Against the FP8 checkpoint under the same sharing (`specrt-27b.md`): decode goes from 21 to 24
tok/s to 24 to 29, prefill on the native path from 1,062 to 1,308 tok/s to 1,240 to 1,930.
vLLM's logged single-request prefill median is 1,782 tok/s; specrt's native path is at or above
that at 1k tokens.

### Prefill GEMM, per call (`nsys`, 1,024-token prompt)

| kernel | default (dequant to bf16) | native | rate |
|---|---|---|---|
| NVFP4 `gate_up`, 1024 x 34816 x 5120 | 8.4 ms | 1.16 ms | 44 vs 315 TFLOP/s |
| NVFP4 `down`, 1024 x 5120 x 17408 | 4.7 ms | 0.61 ms | 39 vs 299 TFLOP/s |
| FP8 `gate_up` (layers 56 to 63) | 10.1 ms | 4.9 ms | 36 vs 74 |
| FP8 `lin_in` | 4.6 ms | 0.95 ms | |

The native fp4 mma is 7x the dequant path. With it the MLPs are 32% of prefill time and the FP8
projections (`lin_in`, `lin_out`, `o`, `qkv`) another 37%.

### Decode GEMV bandwidth (`nsys` with graph node tracing, 128-token prompt)

| kernel | bytes | median | GB/s |
|---|---|---|---|
| `gate_up_act_f4` (NVFP4) | 100 MB | 91 us | 1,104 |
| `down_proj_f4` (NVFP4) | 50 MB | 43 us | 1,161 |
| `gate_up_act` (FP8, layers 56 to 63) | 178 MB | 135 us | 1,325 |
| `down_proj` (FP8) | 89 MB | 65 us | 1,373 |
| `lin_in_norm` (FP8) | 84 MB | 78 us | 1,081 |
| `qkv_proj_norm` (FP8) | 73 MB | 71 us | 1,029 |
| `lin_out_proj`, `o_proj` (FP8) | 31 MB | 26 us | 1,220 |
| `lm_head_norm` (bf16) | 2.54 GB | 2.7 ms | 929 |

Decode reads 21.3 GB per token. The GEMVs take 19.7 ms of it, 1,081 GB/s on average: 66% of
the 1,640 GB/s read ceiling measured beside vLLM, against the issue's target of 80%. GEMVs are
88% of decode kernel time. The NVFP4 GEMVs (about 67% of the ceiling) lag the FP8 ones (80% to
84%) and the bf16 `lm_head` is the slowest per byte (57%). A 16-byte-per-lane variant of the
fp4 GEMV (two groups per load) measured worse (107 us for gate/up) and was not kept.

At 1,081 GB/s, 21.3 GB per token is a 19.7 ms floor, about 50 tok/s; specrt measures 24 to 29.
The rest is per-token overhead outside the GEMVs (48 `gdn_step` launches, small norms, the
attention layers) and the sharing: each kernel here ran beside vLLM's own.

### Decode GEMV tuning (#362)

The GPU was never free (vLLM at 94% utilisation, 300 W cap), so every number is beside vLLM;
medians swing with its load (the bf16 `lm_head` median moved from 2.7 ms to 4.3 ms between two
runs of the same kernel) while minimums stay put. Standalone microbenchmarks
(`scripts/fp4_gemv_bench.cu`, 16 layers of buffers so nothing is L2 resident, streaming read
ceiling 1,652 GB/s from `scripts/bandwidth.cu`):

| variant | gate/up GB/s | down GB/s |
|---|---|---|
| before (1 row per warp, 256 threads, unroll 4) | 1,292 | 1,461 |
| 2 rows per warp (gate and up), 2 groups per lane in flight, 512 threads | 1,366 | 1,461 |
| 1,024 threads | 1,380 | |
| 4 rows per warp | 1,252 | 1,415 |
| several row pairs per block, prefetching the next tile | no gain | |

`down_proj_f4` was already at 1,461 GB/s (88% of the ceiling) with the full-grid launch, so split-k
is not needed. Kept: `gate_up_act_f4` loads gate and up rows together with two groups per lane
in flight, 512-thread blocks for both `_f4` kernels (`FP4_GEMV_THREADS`).

`lm_head_norm` standalone reaches 1,640 to 1,646 GB/s for any unroll or block size
(5,120 x 248,320 bf16, 1.55 ms), so its 929 GB/s in the table below is vLLM contention, not the
kernel. An FP8 or NVFP4 head changes logits and needs its own accuracy gate; not done.

In the graph, under the same sharing (`nsys --cuda-graph-trace=node`, 128-token prompt):

| kernel | before median | after median | after min |
|---|---|---|---|
| `gate_up_act_f4` | 1,107 GB/s | 1,278 | 1,470 |
| `down_proj_f4` | 1,161 | 1,172 | 1,476 |
| other GEMVs | unchanged | | |

`gate_up_act_f4` median 90.6 us to 78.5 us. Per-layer minimums of the NVFP4 GEMVs are 89% of the
ceiling; medians stay near 78% because of the sharing.

### Batched decode (native prefill, 128-token prompts, `bench ... 128 128 N`)

Before #361 (MLPs through the prefill GEMM's tiny tile):

| batch | aggregate | per sequence | vLLM at that many running (with MTP) |
|---|---|---|---|
| 4 | 52.3 tok/s | 13.1 | 362 |
| 8 | 98.7 to 104.3 | 12.3 to 13.0 | 512 |
| 16 | 172.6 | 10.8 | 796 |

With the batched fp4 GEMV (#361), same command (the `qwen3.8-27b-nvfp4-act.toml` spec), vLLM at
95 to 100% beside it:

| batch | aggregate | per sequence |
|---|---|---|
| 4 | 88.3 tok/s | 22.1 |
| 8 | 194.7 | 24.3 |
| 16 | 325.8 | 20.4 |

That is 1.7x, 1.9x and 1.9x the old numbers at batch 4, 8 and 16; per sequence, batch 8 now
runs at the single-stream rate (24 to 29 tok/s). vLLM, with MTP, is still 2.4 to 4x ahead.

#### NVFP4 MLP kernels per call (`nsys --cuda-graph-trace=node`, 128-token prompt, 64 new)

`gate_up` reads 100.3 MB (packed weights plus group scales), `down` 50.1 MB. Medians of
3,920 calls; the batched rows include the split-K reduce launch (2 to 5 us).

| batch | `gate_up` tile | `gate_up` batched | `down` tile | `down` batched |
|---|---|---|---|---|
| 4 | 251 us, 400 GB/s | 105 us, 960 GB/s | 239 us, 210 GB/s | 53 us, 955 GB/s |
| 8 | 257 us, 391 GB/s | 106 us, 945 GB/s | 243 us, 206 GB/s | 54 us, 922 GB/s |
| 16 | 261 us, 384 GB/s | 124 us, 807 GB/s | 248 us, 202 GB/s | 81 us, 622 GB/s |
| 1 (single-row GEMV after #362, for scale) | `gate_up_act_f4` 78.5 us | | `down_proj_f4` 43 us | |

At batch 8 the MLP pair takes 160 us against 134 us for the single-row GEMVs before #362
(1.19x) and 121.5 us after it (1.32x), so it sits at the issue's 1.3x target, just above it
against the tuned GEMV. At batch 16 it is 205 us (1.7x): the 16-token variant runs twice the
mmas per weight byte and the dequant is no longer hidden.

`templates/fp4_batch.cu` (`gemv_f4b`): each warp owns 32 weight rows and puts them on the
M side of `mma.m16n8k16.bf16`, so the tokens are the N side, 8 per tile (`gate_up_f4b8` for 1
to 8 rows, `gate_up_f4b16` for 9 to 16). A lane streams 16 bytes of one row per 128-column
chunk (`ld.global.nc.L1::no_allocate`, prefetched one chunk ahead) and decodes them straight
to mma A fragments, so every weight byte is read once whatever the batch. The activations sit
in shared memory as hi/lo bf16 planes in a 3-stage `cp.async` ring, laid out so each B
fragment is one conflict-free 8-byte load. K is split across `blockIdx.y` to fill the card
(`f4b_splits`: the fewest splits that give at least one full wave of 4 blocks per SM at 85%
occupancy of the last wave, 5 for `gate_up` and 32 for `down` on 188 SMs), and
`*_f4b_reduce` sums the partial rows and applies the epilogue (SwiGLU into the hi/lo planes,
or the residual add). Everything is derived from the spec dims and the SM count; there is no
switch. It serves the default and the `[mlp] act_quant = "fp4"` paths alike; more than 16 rows
(prefill chunks) keep the GEMMs.

On a random 5120 x 17408 and 34816 x 5120 matrix (`tests/fp4_batch.rs`) both paths are
checked against an f64 CPU product: the batched kernel's max error is 7.7e-5 to 8.6e-5 on
`down` (max abs 60 to 72) and 4.4e-3 to 5.4e-3 on `gate_up` (max abs 600 to 880), against
2.2e-3 to 2.9e-3 and 1.4e-2 to 1.8e-2 for the tile path, whose bf16 epilogue planes round
once more. Both must stay within 1e-4 of the largest output.

The FP8 projections in batched decode (`lin_in_mma_tiny` 69 to 81 us, `lin_out_mma_tiny` 43
to 49 us, `qkv_mma_tiny` 64 us) still go through the tile GEMM; they are the next largest
share of the step.

## Commands

```
cd crates/specrt
PY=~/.venv-vllm/bin/python
$PY scripts/reference.py nvfp4-dequant out
cargo test --release --test nvfp4 -- --ignored
$PY scripts/reference.py tiny-nvfp4 tests/fixtures/tiny-qwen3.5-nvfp4 out
cargo test --release --test linear nvfp4 -- --include-ignored
cargo test --release --test fp4_act -- --ignored
cargo test --release --test fp4_batch -- --ignored --nocapture
$PY scripts/reference.py real-slice orcarouter/Qwen3.8-27B-Uncensored-NVFP4 out/qwen3.8-27b-nvfp4-slice out 4 32768
cargo test --release --test real27b real_27b_nvfp4_first_layers -- --ignored
# whole model, GPU steps under flock /tmp/gpu27.lock with 36 GB free
export SPECRT_27B_SPEC=qwen3.8-27b-nvfp4
$PY scripts/reference.py real-prompts orcarouter/Qwen3.8-27B-Uncensored-NVFP4 out
cargo test --release --test real27b real_27b_generates_greedy_and_dumps_logits -- --ignored
SPECRT_27B_SPEC=qwen3.8-27b-nvfp4-act cargo test --release --test real27b real_27b_generates_greedy_and_dumps_logits -- --ignored
$PY scripts/reference.py real-full orcarouter/Qwen3.8-27B-Uncensored-NVFP4 out out/real27b-specrt-weight.json
$PY scripts/reference.py real-full orcarouter/Qwen3.8-27B-Uncensored-NVFP4 out out/real27b-specrt-weight.json act
cargo test --release --test real27b real_27b_greedy_and_logits_match -- --ignored
cargo test --release --test real27b real_27b_per_layer_trace -- --ignored
$PY scripts/vllm_logprobs.py out
cargo test --release --test real27b real_27b_nvfp4_agrees_with_vllm -- --ignored
./scripts/bench27b-nvfp4.sh
```
