# Linear attention, attention and long context

Hybrid models (Qwen3.5 and 3.8, Qwen3-Next, Kimi Linear, the Mamba hybrids) mix full attention layers with recurrent ones. This doc covers the gated delta-net layer, the attention kernels and long context. Described mixers (Mamba2, KDA) are spec only, see [kir.md](kir.md).

## Gated delta-net

Spec, for example `crates/specrt/tests/fixtures/qwen3.8-27b.toml`:

```toml
[layers]
count = 64
pattern = ["linear", "linear", "linear", "full"]

[attention]
kind = "gqa"
heads = 24
kv_heads = 4
head_dim = 256
qk_norm = true
output_gate = true
rotary_dim = 64

[linear_attention]
kind = "gated_delta_net"
key_heads = 16
value_heads = 48
key_head_dim = 128
value_head_dim = 128
conv_kernel = 4
numerics = "exact"
```

`norm.unit_offset` stores RMSNorm weights as `w - 1`; `attention.output_gate` doubles `q_proj` to `[query, gate]` per head; `rotary_dim` rotates only part of each head. `validate` refuses value heads that are not a multiple of key heads, head dims that differ or are not a multiple of 32 up to 128, a conv kernel outside 2 to 8, and a layer count that is not a multiple of the pattern.

Per token per linear layer: one `in_proj` GEMM over `[qkv | z | b | a]`; causal depthwise conv over qkv then SiLU; L2 normalised q and k (q scaled by `1/sqrt(dk)`); `g = -exp(A_log) * softplus(a + dt_bias)`, `beta = sigmoid(b)`; the delta rule on an f32 state per value head (`S = exp(g) S; S += k (beta (v - S^T k))^T; o = S^T q`); gated RMSNorm `norm_w * rmsnorm(o) * silu(z)`; `out_proj` into the residual. Kernels are in `crates/kir/src/cuda/linear/`.

- Decode: `gdn_conv_step`, `gdn_step` (the recurrent form, one block per value head, state slice in registers).
- Prefill: chunk 64 in the WY form, the same algorithm as flash-linear-attention `chunk_gated_delta_rule`. `gdn_chunk_prep` (all chunks in parallel: cumulative g, the strictly lower `A`, block merged inverse `(I - A)^-1`, `u`, `w`, masked intra chunk qk) and `gdn_chunk_scan` (sequential over chunks, 32 state columns per block). Products run on tensor cores.
- Verify and commit for MTP: [mtp.md](mtp.md).
- `[runtime] chunked_scan` (on) reads long prompts through described recurrent mixers with tensor core chunk kernels.

The conv window and state live in the state pool beside the KV cache, so the KV pool holds only the full attention layers (16 of 64 on the 27B). The state is 151 MB per sequence on the 27B and is reset at the start of each sequence.

`[linear_attention] numerics`:

| Value | Chunk products | Allowed |
|---|---|---|
| `exact` (default) | hi/lo bf16 split, three MMAs per tile | always; checkpoint configs and the exact specs are always exact |
| `act` | single bf16 operands, f32 accumulate | only with `act_quant` on |

`act` cuts the prefill delta-net kernels on the 27B from 112.8 to about 79 ms at 4,096 tokens but misses the vLLM gate on the NVFP4 act spec (top-1 1,207 against 1,217 of 1,280), so no checked in spec uses it (#1517).

The chunked state equals the recurrent state to 1.5e-5 and prefill then decode gives the logits of one long prefill to 3e-5; it is not bit identical because the two forms sum in a different order.

## Attention kernels

`kir::attention` families ([kir.md](kir.md)): `RopeKv` (qk norm, rope, KV write in bf16 or fp8), `SplitDecode` (single stream decode), `FlashLegacy` (scalar prefill), `FlashMma` (tensor core prefill, online softmax, Q norm and rope in the kernel, 64 query rows and 16 key tiles above head dim 128 so it fits 99 KB) and `GroupedDecode` (batched decode and verify).

`batch_decode_attention` runs one block per (KV head, verify group, split): the group's GQA heads times its rows form one query tile, so each KV page is read once per sequence. `mma.m16n8k16` in fp16 with hi/lo Q and P keeps the result within 4e-6 of an f64 reference. K and V tiles of `DECODE_KEYS` (16) keys go through shared memory in `DECODE_STAGES` (2) cp.async stages; splits are `decode_splits(max_ctx)` chunks of `DECODE_CHUNK` (256) keys, doubling with length, at most `DECODE_MAX_SPLITS` (64). A row gives the same bits alone or in its verify group when both pick the same chunk. Constants are in `crates/kir/src/attention.rs`; 3 stages and 32 key tiles lost to occupancy and registers at head dim 256 (#617).

Sliding windows, soft caps, query scales and per layer rope bases come from the spec. Variant attentions (sinks, chunked masks, MLA, DSA, compressed sparse attention) are described kernels.

## YaRN context extension

`spec::yarn_inv_freq` and `Spec::rope_constants` follow transformers' `_compute_yarn_parameters` (the ramp between the `beta_fast` and `beta_slow` correction dims, the attention factor, only rotated dims, composes with interleaved mrope). The constants are baked into the kernels. A checkpoint that declares YaRN gets it from its config; others can opt in:

```toml
[runtime]
rope_extension = { type = "yarn", factor = 4 }
```

It multiplies `max_position_embeddings`, which the serving `max_model_len` check and the load dialog follow. Off by default because it changes short context numerics; refused when the model already declares rope scaling; part of the prefix cache namespace. Dynamic YaRN is not supported.

KV on the 27B per token: 64 KiB in f16, about 33 KiB in fp8; 64 and 33 GiB at 1,048,576 tokens.

## Measured state

27B NVFP4, desktop GPU1:

| Measure | Value | Source |
|---|---|---|
| prefill delta-net kernels at 4,096 tokens, exact | 112.8 ms over 48 layers | #1517, 27 Nov 2026 |
| prefill tok/s at 4k / 32k / 128k | 6,139 / 4,932 / 2,899 | #1517 |
| `batch_decode_attention` at 12 x 32k inside the model | 85 percent of the 1,640 GB/s read ceiling | #617 |
| decode at 1 x 128k, MTP k = 2 | 77 tok/s | #617 |
| aggregate decode at 12 x 32k | 559 tok/s | #646 |

At 12 x 32k the round is bound by weights, bf16 KV and the recurrent state: about 31 ms at 1,640 GB/s against 56 ms measured; fp8 KV halves the KV term.

## Tests

`crates/specrt/tests/linear.rs` (tiny Qwen3.5 fixtures against transformers: decode, chunked prefill, state hand off, 1e-3 activations and 1e-4 states), `tiny27b.rs` (chunked against recurrent over 16k), `decode_attention.rs` (the kernel against f64 across splits, lengths 1 to 16,397, verify groups, the 122B GQA 32/2 shape), `gdn_speed.rs` and `flash_speed.rs` (timing), `yarn.rs` (tables against `tests/fixtures/yarn_reference.json`, 1M positions on GPU), `real27b.rs` needle tests at 500k and 1M. References: `crates/specrt/scripts/reference.py tiny <fixture> <dir> [256]`, see [../guides/testing.md](../guides/testing.md).
