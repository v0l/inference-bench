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
kernel. An FP8 or NVFP4 head changes logits and needs its own accuracy gate; see the next section.

In the graph, under the same sharing (`nsys --cuda-graph-trace=node`, 128-token prompt):

| kernel | before median | after median | after min |
|---|---|---|---|
| `gate_up_act_f4` | 1,107 GB/s | 1,278 | 1,470 |
| `down_proj_f4` | 1,161 | 1,172 | 1,476 |
| other GEMVs | unchanged | | |

`gate_up_act_f4` median 90.6 us to 78.5 us. Per-layer minimums of the NVFP4 GEMVs are 89% of the
ceiling; medians stay near 78% because of the sharing.

### Quantised lm_head (#363)

`head_quant` in the spec picks the output head format: `bf16` (the default, what the checkpoint
ships and what vLLM uses), `fp8_channel` or `nvfp4_g16`. The loader quantises the bf16 head at load
time, so the checkpoint is unchanged: `fp8_channel` takes one scale per row (`absmax / 448`,
round to nearest even e4m3) and reuses the FP8 GEMV and `gemm_tc` kernels; `nvfp4_g16` takes a
global scale of `448 * 6 / absmax` over the whole head and e4m3 scales per 16 columns, exactly
the checkpoint's own NVFP4 scheme, and runs `lm_head_norm_f4` (the `fp4_rows_dot` GEMV, two rows
per warp) and `lm_head_f4w*` (`gemm_f4w`) from `templates/fp4_head.cu`. A tied head quantises only
its output copy; the embedding lookup stays bf16. The head never goes through the native FP8 or
FP4 activation paths. `nvfp4_g16` needs the tensor-core path (it is refused with `[runtime]
gemm_fp32 = true`).

Tiny fixtures (`tests/head_quant.rs`): decode GEMV and prefill GEMM logits match a CPU f64 matmul of
the dequantised head to at most 3.1e-6 relative on all four fixtures and both formats (bound
1e-4); every layer stays bit identical; the default and an explicit `bf16` are bit identical to
master (logits and greedy, compared byte for byte against `9b2e9d2` on qwen3-0.6b and three tiny
fixtures); batch decode, the stepped decode and the 64-sequence scheduler with preemption
(`tests/serving.rs`) equal single-sequence greedy with either head.

| 20 prompts, 1,500 positions | fp8_channel top-1 / KL(bf16 \|\| quantised) | nvfp4_g16 top-1 / KL |
|---|---|---|
| tiny-qwen3.5 (random weights, flat logits) | 94.9% / 3.5e-4 | 83.9% / 4.7e-3 |
| tiny hd256, tiny fp8, tiny nvfp4 | 94.8 to 95.9% / 3.5e-4 | 82.2 to 83.5% / 4.6e-3 |
| Qwen3-0.6B (tied head) | 95.5% / 2.4e-3 | 85.2% / 2.8e-2 |

Real NVFP4 27B, beside vLLM, one binary for all three heads (`scripts/head27b.sh`, `nsys
--cuda-graph-trace=node`, 128-token prompt, 64 decode steps, max context 192):

| head | bytes | `lm_head` GEMV median / min | GB/s at median / min | decode kernel time per token (sum of medians) |
|---|---|---|---|---|
| bf16 | 2.54 GB | 1,821 / 1,709 us | 1,396 / 1,488 | 21.80 ms |
| fp8_channel | 1.31 GB | 1,313 / 1,012 us | 999 / 1,297 | 21.46 ms |
| nvfp4_g16 | 0.72 GB | 656 / 529 us | 1,090 / 1,352 | 20.97 ms |

The NVFP4 head saves 1.17 ms of the 1.82 ms GEMV, 3.8% of decode kernel time; FP8 saves 0.5 ms
(2.3%). End-to-end decode tok/s swung from 28 to 55 between runs of the same binary as vLLM's load
moved, so it is not a usable before/after; the kernel times are.

Accuracy, whole model against the CPU oracle (`real_27b_greedy_and_logits_match_the_fp8_reference`,
weight-only reference, 5 prompts x 96 positions) and teacher-forced against vLLM
(`real_27b_nvfp4_agrees_with_vllm_greedy_and_logprobs`, 20 x 64):

| head | KL(oracle \|\| ours) | top-1 vs oracle | greedy vs oracle, 32 x 5 | vLLM teacher-forced top-1 | KL(vLLM \|\| ours) mean / median / p99 |
|---|---|---|---|---|---|
| bf16 | 1.2e-9 | 100% | identical on all 5 | 1,241 / 1,280 (97.0%) | 0.0091 / 0.0025 / 0.087 |
| fp8_channel | 6.5e-4 | 97.3% | 1 of 5; flips at steps 9 to 19, margins 0.007 to 0.09 | 1,237 (96.6%) | 0.0095 / 0.0028 / 0.082 |
| nvfp4_g16 | 6.2e-3 | 95.8% | 2 of 5; flips at steps 0 to 29, margins 0.06 to 0.39 | 1,226 (95.8%) | 0.0136 / 0.0044 / 0.107 |
| envelope: the act reference against weight-only | 0.0105 | 97.3% | | gate: no worse than 96.6% / 0.0097 | |

Both heads pass the test's envelope (KL within 1.5x of 0.0105, top-1 within 2 points of 97.3%).
Against the issue's vLLM gate (no worse than 96.6% top-1 and 0.0097 KL), `fp8_channel` passes
(96.6%, 0.0095) and `nvfp4_g16` fails (95.8%, 0.0136). The FP8 head buys 2.3% of decode kernel
time and costs 0.4 points of vLLM top-1 at the edge of the gate. That is not worth a change to the
default, which stays `bf16`; `fp8_channel` is there for anyone who wants the speed and accepts the
drift. `nvfp4_g16` is outside the gate.

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
to 49 us, `qkv_mma_tiny` 64 us) went through the tile GEMM until #375.

#### FP8 projections on the batched GEMV (#375)

`templates/fp8_batch.cu` (`gemv_f8b`) is `gemv_f4b` for e4m3 per-channel weights: the same
warp layout, activation ring and split-K, with a lane streaming 32 bytes of one row per
128-column chunk and the row's scale for that chunk applied to the chunk's partial sums, so
the block-128 checkpoint runs through it too. It covers `qkv`, `o`, `lin_in`, `lin_out` and the
FP8 MLPs of layers 56 to 63 (`*_f8b8`, `*_f8b16`, `*_f8b_reduce`); splits come from
`f8b_splits` (10 for `qkv` and `lin_in`, 12 for `o` and `lin_out`, 5 for `gate_up`, 32 for
`down` on 188 SMs). `lin_ba` and `lm_head` are bf16 and keep the tile GEMM.

Per call, `nsys --cuda-graph-trace=node`, 128-token prompt, 64 new, `qwen3.8-27b-nvfp4-act.toml`,
vLLM at 93 to 100% beside it. Medians of 3,360 calls for the 48 linear-attention layers, 1,120
for the 16 attention layers and 560 for the 8 FP8 MLPs; the batched figure adds the reduce
median (2 to 5 us).

| kernel | batch 4 tile | batch 4 batched | batch 8 tile | batch 8 batched | batch 16 tile | batch 16 batched |
|---|---|---|---|---|---|---|
| `lin_in` 16384 x 5120 | 79.1 us | 74.6 us | 86.4 | 76.1 | 99.1 | 102.4 |
| `lin_out` 5120 x 6144 | 46.9 | 32.5 | 50.8 | 34.6 | 54.2 | 45.8 |
| `qkv` 14336 x 5120 | 72.8 | 68.1 | 78.9 | 69.8 | 90.8 | 94.0 |
| `o` 5120 x 6144 | 47.9 | 33.4 | 50.6 | 32.2 | 53.1 | 37.4 |
| `gate_up` 34816 x 5120 | 147.0 | 146.8 | 157.2 | 148.7 | 184.0 | 186.4 |
| `down` 5120 x 17408 | 135.5 | 81.1 | 143.4 | 82.7 | 149.2 | 116.0 |

Summed over a decode step that is 11.1 ms of FP8 projections on the tile path against 8.8 ms
batched at batch 8 (minus 20%), 10.2 ms against 8.6 ms at batch 4 (minus 16%) and 12.3 ms
against 11.6 ms at batch 16 (minus 6%). The narrow-K shapes
(`o`, `lin_out`, `down`) gain most, from the split-K filling the card; `lin_in`, `qkv` and
`gate_up` were already near the memory roofline on the tile path (1.0 to 1.2 TB/s) and stay
there. At batch 16 the 16-token variant runs twice the mmas per weight byte, the dequant is no
longer hidden, and `lin_in`, `qkv` and `gate_up` come out 1 to 4% behind the tile path while
the other three stay ahead.

Whole-step tok/s (`bench ... 128 128 N`, median of 3, same binary with the batched FP8 path
switched off for the tile column, run interleaved, vLLM at 93 to 100%):

| batch | tile path | batched FP8 |
|---|---|---|
| 4 | 89.1 tok/s | 90.5 |
| 8 | 140.6 | 177.7 |
| 16 | 285.5 | 254.1 |

These swing by 20 to 30% between runs with vLLM busy beside it (the #361 numbers above were
taken at 95 to 100% vLLM load too, and a `scripts/bench27b-nvfp4.sh` run during this work gave
75.1, 137.0 and 220.6 on the batched path), so the per-kernel table is the measure.

On random matrices of every projection shape (`tests/fp8_batch.rs`, batch 1, 2, 3, 4, 8, 13,
16) the batched kernel's max error against an f64 CPU product is 1.1e-5 to 1.5e-5 on `qkv`,
`o`, `lin_in` and `lin_out` (max abs 30 to 40), 2.4e-5 to 2.6e-5 on `down` (max abs 57 to 66)
and 1.1e-3 to 3.2e-3 on `gate_up` (max abs 400 to 560), never worse than the tile path.

#### 17 to 32 rows on the batched GEMV (#414)

`gemv_f4b` and `gemv_f8b` take the warp count, ring depth and rows per warp as template
parameters (`BatchShape` in `codegen.rs`). The 32-token variants (`*_f4b32`, `*_f8b32`) run 8
warps of one 16-row mma tile each, so a block still covers 128 weight rows and each warp keeps
four 8-wide N tiles of accumulators: with two row tiles per warp the FP8 kernel spilled 464 bytes
at 128 registers, with one it does not spill. The ring is 3 stages of 32 x 128 hi/lo columns (48
KB), so 2 blocks fit an SM; `BatchShape::blocks_per_sm` derives that from the shared memory and
the splits follow from it (`gate_up` 4, `down` 16, `qkv` 6, `lin_in` 5, `o` and `lin_out` 12 on
188 SMs). Every weight byte is still read once for 17 to 32 rows. Dispatch: `kernel_for` takes
the batched GEMV for every FP8/NVFP4 projection up to 32 rows on `tile = "auto"`, before the
activation-quantised GEMMs; above 32 the GEMMs keep it.

Errors at 17, 24 and 32 rows (`tests/fp4_batch.rs`, `tests/fp8_batch.rs`) are the 16-row ones:
8.6e-5 on fp4 `down`, 5.4e-3 on fp4 `gate_up`, 1.2e-5 to 2.8e-5 on the FP8 shapes, 3.5e-3 on FP8
`gate_up`.

Per call on the real 27B (`nsys --cuda-graph-trace=node`, 128-token prompt, 64 new,
`qwen3.8-27b-nvfp4.toml`, base `f5b1ef0` against #414, vLLM 93 to 100% busy):

| kernel | batch 24 base | batch 24 #414 | batch 32 base | batch 32 #414 |
|---|---|---|---|---|
| NVFP4 `gate_up` (`f4w` 34816 x 5120) | 341.0 us | 160.5 | 343.7 | 168.6 |
| NVFP4 `down` (`f4w_narrow` 5120 x 17408) | 547.5 | 101.4 | 590.0 | 110.9 |
| FP8 `lin_in` 16384 x 5120 | 175.5 | 115.0 | 178.4 | 121.3 |
| FP8 `qkv` 14336 x 5120 | 172.6 | 100.2 | 175.2 | 106.4 |
| FP8 `lin_out` 5120 x 6144 | 95.1 | 47.1 | 104.0 | 48.3 |
| FP8 `gate_up` 34816 x 5120 | 325.2 | 201.1 | 331.0 | 209.8 |

Total kernel time under nsys falls from 23.6 to 17.3 s at batch 24 and 25.6 to 22.6 s at batch
32, and the whole-step decode rate in the same runs goes from 153.3 to
276.0 tok/s at batch 24 and 221.3 to 349.3 at batch 32 (plus 58%). On the `[mlp] act_quant =
"fp4"` spec plain batch 32 is unchanged (450.2 against 450.8 under nsys): its native fp4 and fp8
GEMMs already beat the bf16 dequant GEMV there, so the dispatch keeps them.
Real-27B tests on master f03f786 (with #375 and #379), references regenerated for this run
into one dir (`real-prompts`, both `real-slice`s, `real-full` weight-only and `act`, and
`vllm_logprobs.py` against the live vLLM), `cargo test --release --test real27b -- --ignored
--test-threads=1`: 6 passed, 0 failed.

| test | result |
|---|---|
| `real_27b_fp8_first_layers_match_transformers` | per-layer 6.9e-7 to 3.1e-6, logits 4.7e-6, top-1 64/64, decode 6.3e-7 |
| `real_27b_nvfp4_first_layers_match_the_dequantised_checkpoint` | per-layer 1.3e-5 to 2.5e-5, logits 9.9e-6, top-1 64/64, decode 8.4e-7 |
| `real_27b_per_layer_trace_matches_the_cpu_oracle` | all 64 layers 7.6e-7 to 1.4e-5, none above 1e-3 |
| `real_27b_generates_greedy_and_dumps_logits` | weight and act dumps written |
| `real_27b_greedy_and_logits_match_the_fp8_reference` | weight: logits 2.0e-4 of max, greedy 32/32 on all 5; act: KL 0.00108 against 0.00086 for the act reference, top-1 98.3% against 99.0% |
| `real_27b_nvfp4_agrees_with_vllm_greedy_and_logprobs` | teacher-forced top-1 1,217 / 1,280 (95.1%), KL mean / median / p99 0.0096 / 0.0029 / 0.080, greedy 504 / 1,280 |

These run single sequences and prefill chunks over 16 tokens, so they check that #375 left the
other paths alone; the batched kernel itself is covered by the batch greedy tests and
`tests/fp8_batch.rs`.

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
cargo test --release --test fp8_batch -- --ignored --nocapture
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
# quantised head: run head27b.sh under flock /tmp/gpu27.lock; the other two take the lock per GPU step; all need 36 GB free
flock /tmp/gpu27.lock env BENCH=<bench> OUT=out/nsys363 ./scripts/head27b.sh
SPECRT_REF_DIR=out TESTBIN=<real27b test binary> ./scripts/head27b-gates.sh
SPECRT_REF_DIR=out TESTBIN=<real27b test binary> ./scripts/head27b-vllm.sh
cargo test --release --test head_quant -- --include-ignored
```

## Prefill tile for the native fp4 GEMM (#432)

`gemm_f4a` (`[mlp] act_quant = "fp4"`) now defaults its `_big` tile to 128x256 with 4x4 warps,
settable with `[runtime] fp4_tile = [bm, bn, wm, wn]`. Per-call nsys on the 27B, gate_up:

| tile | 1,024 tokens | 4,096 tokens | prefill tok/s at 1,024 / 4,096 |
|---|---|---|---|
| 64x128, 2x4 (old) | 1.14 ms | 2.31 ms | 3,996 / 3,832 |
| 128x128, 4x2 | 0.80 ms | 1.60 ms | 4,164 / 4,003 |
| 128x256, 4x4 (default) | 0.72 ms | 1.48 ms | 4,223 / 4,051 |

gate_up goes from about 160 to 250 TFLOP/s at 1,024 tokens, a 37% cut in GEMM time. Below 512
tokens the larger tile loses (256 tokens: 2,898 against 2,756 tok/s is the new default, still ahead;
see the bench run). ncu needs GPU counter permission (ERR_NVGPUCTRPERM), so stall reasons and bank
conflicts were not collected; the deeper pipeline, TMA and warp specialisation ports are not done.
