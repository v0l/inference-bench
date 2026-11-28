# GLM-5.3 Flash

`Piecrust/GLM-5.3-Flash-FP4-Abliterated` (`model_type glm5_next`), served by `specs/family-glm5_next.toml` as text only, with no architecture name in Rust. Config copy without weights: `crates/specrt/tests/fixtures/glm5.3-flash.config.json`.

## The model

45 decoder layers of hidden 4096. Layers 3, 7, ..., 43 (11 layers) are DeepSeek sparse attention: MLA with no rope (latent 512, q/k nope dim 256, v 256) and a top-k indexer over keys pooled 4 at a time. The other 34 are Kimi delta attention (KDA): 64 heads of 128, separate q, k, v projections and convs, a per key channel forget gate. Layers 0 to 2 have a dense SwiGLU MLP (12288); layers 3 to 44 a 288 expert MoE (top 8, sigmoid with `e_score_correction_bias`, `routed_scaling_factor` 2.5, one 2048 wide shared expert, no shared gate). Every block sits inside manifold constrained hyper-connections (mHC): 4 residual streams mixed by a Sinkhorn normalised 4 x 4 matrix per token (20 iterations). Clamped SwiGLU (limit 10). Layer 45 is one MTP layer; a 24 block vision tower is attached. Vocabulary 154,880, context 1,048,576. Reference: transformers 5.23 `modeling_glm5_next.py`.

## How the family maps it

| Piece | Family data and engine pieces |
|---|---|
| mHC | `[hyper]` (`hc_mult`, `sites = hc_attn, hc_ffn`); plan ops `HyperExpand`, `HyperPre`, `HyperPost`, `HyperMean`; kir rows `hyper_pre`, `hyper_post`, `hyper_expand`, `hyper_mean`; decoder graph blocks `hyper_*` |
| no rope MLA | `[rope] nope_all = true`; `dsa_latent_attention` keys on the latent only; `mla_q_lora_norm`, `mla_kv_lora_norm` |
| pooled indexer | `index_key` state of key and compress gate per token (`pool_index_write`), `pool_topk` (softmax pooled scores, `topk / 4` pools expanded to tokens, plus the open tail) |
| KDA | `[mixer]` with the fused input linear, three `seq_conv` nodes, `head_matvec` for the low rank gates, `kda_recurrence`, a sigmoid gated `group_norm`. Prefill runs the recurrent form: the chunked lowering refuses a state read twice |
| MoE and MLP | `[moe]` sigmoid router with score bias, `dense_layers` from `first_k_dense_replace`, `[mlp.gated] expr` with the clamp |
| Not served | the MTP layer (a `[mixer]` with `[mtp]` is refused, so `mtp = false`) and the vision tower (`[vision] ignore = true`; an image request is refused by name) |

## Quantisation: ModelOpt MIXED_PRECISION

The family sets `[quantization] per_tensor = true`; `[[quantization.tensor_rules]]` in `family-base.toml` map each `quantized_layers` entry to a `TensorQuant`, kept on the spec as `quant_map` (`crates/specrt/src/quant_map.rs`). `Checkpoint::check_quant_layout` refuses an unknown algo, encoding, dtype or shape with the tensor named.

| Tensors | Stored | Served |
|---|---|---|
| routed experts, layers 3 to 44 (162.5 GB) | NVFP4 with CSF compressed scales (`csf_32x2_v1`) | NVFP4 expert stack as stored, weight only; scales decoded by `csf::CsfPlane::decode` (byte equal to vLLM's `logical_csf_components`) |
| shared experts | FP8_PB_WO (e4m3, f32 per 128 x 128, a 4-D `weight_scale`) | requantised to NVFP4 into the stack's shared slot |
| KDA q, k, v, o; dense MLP; lm_head | MXFP8 (e4m3, e8m0 per 32) | decoded to bf16 at load (`dequant::MXFP8`, the kir `BlockDequant` block) |
| DSA q_a, q_b, kv_a, o | FP8_PB_WO | decoded to bf16 |
| MTP experts, vision MLP | W4A16 NVFP4 | (not served) |
| embedding, KDA gates, indexers, routers, mHC, norms | bf16 and f32 | as stored |

Resident text model: about 18 GB besides the routed experts (189.5 GB in all as the describe path counts it, 173.4 GB in the load dialog without the MTP layer). KV per token over the 11 DSA layers: 11.0 KB bf16 latent; the KDA state is 136 MiB per sequence.

## Measured state

Real checkpoint on both desktop cards, the user's app stopped, a two stage pipeline with every weight resident (gpu0 layers 0..23, 88.6 GB; gpu1 23..45, 95.4 GB), engine commit a2f1dc52 (#1335):

| Prompt tokens | Time to first token | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| 27 | 0.35 s | 76 | 4.41 |
| 463 | 2.66 s | 174 | 4.35 |
| 3,101 | 20.0 s | 155 | 3.96 |

Decode is about 20x off the planned 11.3 ms a step, the same two stage pipeline gap as Qwen4Exp (#1389). Greedy text is coherent on the four probe prompts. Not compared against vLLM yet: with vLLM up, capture the same prompts with `logprobs: 20` and compare top-1 and the first divergence.

## Tests

`crates/specrt/tests/glm5_next.rs` (tiny `tests/fixtures/tiny-glm5-next` from `scripts/tiny_glm5_next.py`; a host f32 model in `tests/glm5_host`; on GPU1 streams after layers 0, 1, 3 within 7.1e-5, prefill logits held to `BF16_SPLIT_TOLERANCE` 5e-4, decode within 3.8e-4, red with a wrong Sinkhorn count or a dropped tail pool), `glm5_mixed.rs` (MIXED_PRECISION copy against a bf16 twin, identical logits), `quant_map.rs`, `csf.rs`, `mxfp8.rs`, `fp8_pb_wo.rs`, `w4a16_nvfp4.rs`, `spec.rs` `glm5_3_flash_serves_text_with_its_glm_keys`, `pipeline.rs` (MoE with dense first layers and hyper streams split into stages is bit identical). GPU checks of the loader: `crates/specrt/scripts/gate1190_gpu.sh`.
