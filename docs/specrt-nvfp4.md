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
