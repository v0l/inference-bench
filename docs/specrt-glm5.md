# GLM-5.3 Flash (`glm5_next`) in specrt: gap list (#1179 stage 1)

Read-only analysis of what specrt needs to load and serve `Piecrust/GLM-5.3-Flash-FP4-Abliterated`, the checkpoint the user already serves with vLLM. No engine code changed for this document.

## Sources

| short name | path |
| --- | --- |
| checkpoint | `~/.cache/huggingface/hub/models--Piecrust--GLM-5.3-Flash-FP4-Abliterated/snapshots/5c15d09*/` (40 shards, `model-inputscales.safetensors`, `tensors/native-QB.safetensors`, `model.safetensors.index.json`, 295,041 tensors, 177.4 GB) |
| config | `config.json` in that snapshot (`transformers_version` 5.16.0). `hf_quant_config.json` is byte for byte the `quantization_config` block of `config.json`. A copy without weights is `crates/specrt/tests/fixtures/glm5.3-flash.config.json` |
| HF | transformers 5.23.0 in `~/.venv-vllm`: `transformers/models/glm5_next/modeling_glm5_next.py` and `configuration_glm5_next.py`. 5.16.0 is not installed; 5.23.0 has the model, so it is the reference for the tiny fixture (stage 2). It has no MTP layer (`_keys_to_ignore_on_load_unexpected = layers.45`) |
| vLLM | `~/.venv-vllm` vLLM has no glm5_next. The served image `ghcr.io/local-inference-lab/vllm:release-10152026` (`~/glm-5.3-flash-uncensored.sh`) has it under `/opt/venv/lib/python3.12/site-packages/vllm/models/glm5next/` (`common/{model,attention,kda,mtp}.py`, `nvidia/{pooled_indexer,attention,kda,mtp}.py`) and the loaders `model_executor/layers/quantization/modelopt.py`, `model_executor/model_loader/csf_utils.py`, `flashinfer/experimental/b12x/loader/_csf_blocks.py`. Read with `docker run --rm --entrypoint bash <image> -c ...`, nothing started on a GPU |
| README | the checkpoint's `README.md`: serving numbers on 2 x RTX PRO 6000 |

Line numbers below are in HF `modeling_glm5_next.py` unless named.

## The model in one paragraph

45 decoder layers of hidden 4096. Layers 3, 7, ..., 43 (11 layers) are DeepSeek sparse attention: MLA with no rope at all and a top-k indexer over keys pooled 4 at a time. The other 34 are Kimi delta attention (KDA). Layers 0 to 2 have a dense SwiGLU MLP (12288), layers 3 to 44 a 288 expert MoE (top 8, sigmoid, one 2048 wide shared expert). Every block sits inside manifold-constrained hyper-connections (mHC): the residual is 4 streams of 4096, mixed by a Sinkhorn-normalised 4 x 4 matrix per token. Layer 45 is one MTP layer (DSA plus MoE, no mHC). A 24 block vision tower is attached. Vocabulary 154,880, context 1,048,576.

## What specrt says today

`Spec::describe_hf_config` on the real config (golden `crates/specrt/tests/golden/glm5.3-flash.describe.json`) and `from_hf_config_serving` (`model_type glm5_next is not supported`). The describe output is wrong in five places, so the family spec cannot be a rename of an existing one:

| field | describe says | real model | source |
| --- | --- | --- | --- |
| attention | `gqa`, 64 kv heads, head_dim 64 | MLA, latent 512, q/k nope dim 256, v 256, no rope | 1101-1277, config 114 and 223-225 |
| linear_attention | `gated_delta_net` | KDA: per key channel gate, separate q/k/v convs | 339-371, 620-770 |
| norm | `unit_offset: true` | plain `weight * x` (no offset) | 65-82 |
| moe.shared_gate | true | no shared gate, the shared expert output is added | 185-206 |
| residual | one stream | mHC, 4 streams | 218-336, 1279-1351 |

## Architecture gap list

Legend: **have** = exists and fits as is, **spec** = expressible with family TOML over existing kir families, **desc** = needs a new kir description or a new stage in an existing family, **load** = loader or quant work, **plan** = plan/runtime/state work.

| # | piece | real model | specrt today | gap | HF | vLLM |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | mHC hyper-connections | `hc_mult` 4, `hc_sinkhorn_iters` 20, `hc_eps` 1e-6. Per block: unweighted RMSNorm of the 16384 flattened streams, `fn` (24 x 16384, bf16) to pre(4), post(4), comb(16); `pre = sigmoid(w*s+b)+eps`, `post = 2 sigmoid(...)`, comb softmax then 20 Sinkhorn row/column passes; block input `sum(pre * streams)`, output `post * y + comb^T @ streams`. Final head: mean of the 4 streams, then norm. Embedding is copied to 4 streams | nothing (`hc_*` appears only in config fixtures; V4 attention family left mHC out) | **desc + plan**: residual buffer 4 x hidden; a compose kernel for the 24 weights (small GEMV plus Sinkhorn loop in registers) and a collapse and an expand kernel; graph block wrapping attention and MLP. Largest missing piece, not in the original brief | 209-336, 1290-1351, 1449, 1510-1519 | `common/model.py` 375-430 (`hc_contract`, `hc_expand`, `MHCPreOp`, `MHCPostOp`, `MHCFusedPostPreOp`) |
| 2 | KDA mixer | 64 heads x 128, separate `q_proj`/`k_proj`/`v_proj` (8192 each), three causal convs (`q_conv1d` etc., 8192 x 4, silu), L2 norm of q and k, beta `sigmoid(b_proj x)`, forget gate `-5 * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))` per channel, output `RMSNormGated(128)` with sigmoid gate `g_b(g_a(x))`, `o_proj` | `specs/kda_recurrence.toml` (#808) is the exact recurrence with this gate (`tests/mamba.rs` against a host delta rule) but no family uses it; the `[mixer]` path exists (bamba, granitehybrid) | **spec + desc**: a `[mixer]` with three projections and three conv weights (`mamba2_conv` binds one weight tensor); low rank gate projections f_a/f_b and g_a/g_b as two chained projections; a gated norm with a sigmoid gate (`mamba2_gated_norm` is silu: an expression change); a chunked prefill form, since `kir/src/recur/chunk.rs` refuses key stages and KDA has per channel decay | 339-395, 463-770 | `common/kda.py`, `nvidia/kda.py` |
| 3 | DSA attention, NoPE MLA | q_lora 1536 (+ RMSNorm), q_b to 64 x 256, kv_a to 512 (+ RMSNorm), kv_b 512 to 64 x (256 + 256), o_proj 16384 to 4096, scale `256^-0.5`. `qk_rope_head_dim = 0`, `position_embeddings=None` | `family-glm_moe_dsa.toml` (tiny-glm-dsa) with absorbed q, latent write, DSA attention, `mla_expand_v` | **spec**: rope dim 0 means the `dsa_rope_q`/`dsa_rope_k` nodes must go (they have `rot = qk_rope_head_dim`), and `runnable` refuses `rot == 0` and a missing theta unless the family uses `nope_all` (exists, nemotron_h). Check `dsa_attention` and `dsa_absorb_q` at head 256, latent 512 | 1101-1277, 1497-1500 | `common/attention.py`, `nvidia/attention.py` |
| 4 | Rope theta | none needed for text: attention and indexer have no rotary part (`indexer_rope_interleave` is in the config but nothing rotates at dim 0). The test's assumed 10 000 is a placeholder that is never used | `spec.rope.theta` None, test asserts the 10 000 fallback | **spec**: `[rope] nope_all = true` gives the placeholder and no rope kernels. Vision rope is separate (axial, vision config) | 1499, config 114, 223-225 | |
| 5 | DSA indexer with key pools | 32 heads x 128, `wq_b` on the q latent, `wk` + LayerNorm (with bias) on x, `weights_proj` scaled `32^-0.5`, relu scores. Keys pooled 4 at a time: softmax over `compress_gate x + compress_ape` weights each key of the pool; top `2048/4 = 512` pools, expanded back to tokens, plus the incomplete tail (up to 3 tokens) always selected | `dsa_index_q`, `dsa_index_k_norm`, `dsa_index_write`, `dsa_index_topk` work per token with a bf16 per token key state | **desc + state**: a pooled key state (one 128 key per 4 tokens, plus the gate logits or the pooled key written when a pool completes), the pool softmax, pool to token expansion with the tail, top-k of 512 over n/4 pools (the tiny fixture used k = 8; check the top-k kernel at k = 512 and pools in the 100k range) | 773-1061 | `nvidia/pooled_indexer.py` (pool 4, fp8 query through `fwht128_quant_fp8`), `common/attention.py` 78-160 (`Glm5NextIndexerCache`, `Glm5NextTailCache`) |
| 6 | Indexer sharing | all 45 `indexer_types` are `full`: no layer shares in this checkpoint. Only MTP shares: `index_share_for_mtp_iteration`, draft steps 1+ reuse step 0's top-k | `glm_moe_dsa` has `full`/`shared` kinds and a shared index buffer | **have** for layers (all full). **plan** for MTP: keep the draft step 0 top-k for steps 1 and 2 | 1131-1134 | `common/mtp.py` 146 (`set_skip_topk`), 51-66 |
| 7 | MoE routing | sigmoid in fp32, `e_score_correction_bias` for choice only, `n_group 1`, `topk_group 1`, top 8 of 288, `norm_topk_prob`, `routed_scaling_factor 2.5` | glm4_moe: `router = "sigmoid"`, `score_bias = true`, group-limited routing from #809 | **spec**: groups = 1 is the plain case; set `shared_gate = false`. Check the route kernels at 288 experts (not a power of two) | 144-182 | `common/model.py` MoE |
| 8 | Shared expert | one, 2048 wide (`moe_intermediate * n_shared_experts`), same as a routed expert, added with no gate | supported when as wide as the routed experts | **have** | 185-206 | |
| 9 | Clamped SwiGLU | `swiglu_limit 10`: gate `min(g, 10)`, up `clamp(u, -10, 10)`, then `silu(g) * u`, in dense MLP, experts and shared expert | `[mlp.gated] expr` with `[mlp.gated.params]` (gpt-oss, `family-clamped.toml`) | **spec**: `expr = "silu(min(g, limit)) * max(min(u, limit), -limit)"`, `limit = { config = "swiglu_limit" }`. Check the MoE kernels take the expression on fp4 experts | 85-142 | |
| 10 | Dense first layers | `mlp_layer_types` dense for 0 to 2 (`first_k_dense_replace 3`), intermediate 12288 | `dense_layers` from `first_k_dense_replace` (golden shows [0, 1, 2]) | **have** | 1288-1290 | |
| 11 | Norms | RMSNorm `weight * x`, eps 1e-5. Final `norm.weight` is stored F32, two `post_attention_layernorm` weights are F32, the rest bf16 | `norm()` reads f32 or bf16 | **spec**: `unit_offset = false` (base default is true) | 65-82 | |
| 12 | MTP | layer 45: `enorm`, `hnorm`, `eh_proj` (8192 to 4096, bf16), one DSA layer with its own indexer, a MoE layer (W4A16 NVFP4 routed, MXFP8 shared), `shared_head.norm`, no own head (uses `lm_head`), no mHC. vLLM runs it 3 times (`num_speculative_tokens 3`) | `[mtp.layout]` for a plain attention MTP (glm4_moe, #976) | **plan**: MTP over an MLA/DSA layer with MoE; `[mixer] cannot be combined with [mtp] yet` (spec.rs) must go for a hybrid; the MTP layer's KV (a 12th latent cache) and index state; reuse of top-k across draft steps | none (HF ignores layer 45) | `common/mtp.py` 40-110 |
| 13 | Vision tower | 24 blocks, hidden 1024, patch 14, temporal 2, merge 2 to 4096; MXFP8 attention, W4A16 NVFP4 MLP and merger | Llama 4 style `[vision] ignore = true` | **spec**: text only first, images refused by name | 1523-1890 | `common/multimodal.py` |
| 14 | Chat and tools | `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`, `reasoning_effort` and `clear_thinking` template variables | `xml_kv` tool parser (#976) | **have** for the parser; check reasoning (vLLM uses `glm45`) | `chat_template.jinja` | `--tool-call-parser glm47 --reasoning-parser glm45` |

## Quantisation: ModelOpt MIXED_PRECISION

`quantized_layers` has 494 modules in four formats. Counts are modules in `quantized_layers`, weight matrices in the safetensors index, and bytes of every tensor that belongs to the format (weights, scales, CSF parts, input scales).

| format | modules | weight matrices | bytes | what it covers |
| --- | --- | --- | --- | --- |
| NVFP4 (group 16, `weight_scale_encoding: csf_32x2_v1`) | 42 | 36,288 | 162.52 GB | routed experts of layers 3 to 44 (42 x 288 x gate/up/down) |
| W4A16_NVFP4 (group 16, plain scales) | 77 | 940 | 4.33 GB | MTP routed experts (layer 45: 288 x 3 = 864), vision MLP (72) and merger (4) |
| MXFP8 (e4m3, e8m0 scale per 1 x 32) | 205 | 205 | 6.09 GB | `lm_head`; dense MLP of layers 0 to 2 (9); KDA `q_proj`, `k_proj`, `v_proj` (3 x 34) and `o_proj` (34 + the MTP layer's); MTP `q_a`, `q_b`, `kv_a`, `kv_b`, indexer `wq_b`, `wk`, `weights_proj`, shared expert (10); vision `attn.qkv` and `attn.proj` (48) |
| FP8_PB_WO (e4m3, f32 scale per 128 x 128, weight only) | 170 | 170 | 2.17 GB | shared experts of layers 3 to 44 (126); DSA `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `o_proj` (44) |
| not quantised | | | 2.32 GB bf16 + f32 | embedding (1.27 GB); KDA `f_a`, `f_b`, `g_a`, `g_b`, `b_proj` (170); DSA `kv_b_proj` (11) and the indexers of layers 3 to 43 (`tensors/native-QB.safetensors`, 214 tensors); routers; `eh_proj`; mHC `hc_*_fn`; convs; norms; A_log, dt_bias |

Tensor layout per format, from the shard headers:

- MXFP8: `weight` F8_E4M3 [rows, cols], `weight_scale` U8 [rows, cols / 32] (e8m0). The README says scale = 2^ceil(log2(amax / 448)).
- FP8_PB_WO: `weight` F8_E4M3, `weight_scale` F32 [rows/128, 1, cols/128, 1]. vLLM squeezes it to [rows/128, cols/128] (`modelopt.py` 2246-2296).
- NVFP4 (layers 3 to 44): `weight` U8 [rows, cols / 2], `weight_scale_2` F32 scalar, `input_scale` F32 scalar (in `model-inputscales.safetensors`), and no `weight_scale` tensor: the e4m3 block scales are stored losslessly compressed as `weight_scale.csf32_{bases,codes,page_bases,offsets,exceptions}` (per row a base byte, a 4 bit offset per scale in 32 x 2 blocks ordered by 128 x 4 tiles, exceptions as 16 bit records in pages of 32 tile groups). 9.6 GiB compressed, 17.7 GiB as plain e4m3. Layout: `_csf_blocks.py` (`CsfBlockPlane`, `logical_csf_components`), reader `csf_utils.py`. The README says upstream vLLM cannot read it.
- W4A16_NVFP4: `weight` U8, `weight_scale` F8_E4M3 [rows, cols / 16], `weight_scale_2` F32, no `input_scale`.
- Expert tensors are per expert (`mlp.experts.N.gate_proj.weight`), not stacked.

Gaps (all **load**, none needs a per-architecture name):

| # | gap | specrt today |
| --- | --- | --- |
| Q1 | a per module quant map from `quantized_layers` (format per tensor, not per kind) | `quant_groups` already groups `quantized_layers` by algo (the Darkstar modelopt test), but no `group_rules` entry matches MXFP8 or FP8_PB_WO, so `compressed_group` bails. Past that, `parse_compressed` (spec.rs) requires every attention projection in one format and that format `fp8_channel`, and MLP formats as one prefix range; `target_slot` refuses `model.visual.*` (no `layers.` marker), and `projections` in `family-base.toml` lists none of the MLA, KDA, indexer or shared expert modules |
| Q2 | MXFP8 as a weight format | `dequant.rs` already has `Elem::E4m3` and `ScaleKind::E8m0`, so `BlockFormat { elem = "e4m3", scale = "e8m0", group = 32 }` decodes at load as data. No native GEMV/GEMM: needs a kir dequant block (the MXFP4 e8m0 path of #809 is the template), or the sm_120 `mma ... kind::mxf8f6f4.block_scale` |
| Q3 | FP8_PB_WO | `Quant::Fp8Block128` is the same math, but `Checkpoint::raw` (weights.rs 444-457) looks for `weight_scale_inv` [rows/128, cols/128] or a per channel `weight_scale` [rows, 1]; the 4-D f32 `weight_scale` must be read as the block grid. Weight only: the act-quant rules must not turn on fp8 activations for these |
| Q4 | NVFP4 with CSF scales | the NVFP4 loader reads `{module}.weight_scale` F8_E4M3 [rows, cols / 16] (weights.rs 1369). Needs a CSF decoder (host side at load is enough; the format is lossless) |
| Q5 | W4A16_NVFP4 (fp4 weights, bf16 activations) | the `nvfp4_g16` group rule already matches it and the act rule turns fp4 activations on only for plain `NVFP4`, so the existing weight only NVFP4 GEMV is the path. Note: the user's vLLM forces A16 for the NVFP4 experts too (`VLLM_B12X_MOE_FP4_FORCE_A16=1`), so W4A16 everywhere is the like for like default |
| Q6 | bf16 tensors from a second file | `tensors/native-QB.safetensors` is in the index `weight_map`, and `shard_files` reads every file the map names: **have** |

## Memory: resident set and KV per token

Bytes as stored (the formats above, kept native on the GPU):

| part | bytes |
| --- | --- |
| routed experts, layers 3 to 44 | 162.52 GB (3.87 GB per layer, 13.4 MB per expert) |
| KDA attention, 34 layers | 4.95 GB (146 MB per layer) |
| MTP layer | 4.30 GB (4.08 GB of it routed experts) |
| DSA attention, 11 layers | 1.48 GB, plus 0.16 GB indexers |
| embedding (bf16) | 1.27 GB (vLLM keeps it on the host: `VLLM_GLM53_EMBED_HOST=1`) |
| shared experts | 1.06 GB |
| lm_head (MXFP8) | 0.65 GB |
| dense MLP, layers 0 to 2 | 0.47 GB |
| vision tower | 0.39 GB |
| routers, mHC, norms | 0.17 GB |
| total | 177.4 GB |

Everything except the routed experts (and vision): 10.2 GB, 10.4 GB with the MTP layer's non-routed part. With the MTP experts: 14.5 GB. Routed experts are 92% of the checkpoint.

Weights read per decode token (no MTP): 8 experts x 42 layers x 13.4 MB = 4.5 GB, plus about 8.9 GB of the rest (all but the embedding): about 13.4 GB. At the 1,640 GB/s this project measured on the card, that is a bandwidth bound of about 8 ms, about 120 tok/s for one stream fully resident (vLLM reports 174 with MTP 3 on TP2).

State per sequence and KV per token (11 DSA layers; vLLM adds a 12th for the MTP layer):

| item | per token | 1M tokens, one sequence |
| --- | --- | --- |
| MLA latent, bf16 (512 values per layer, no rope part) | 11 x 1,024 B = 11.0 KB | 11.0 GiB |
| MLA latent, fp8 (512 e4m3 + 4 f32 scales, estimate) | 11 x 528 B = 5.7 KB | 5.4 GiB |
| indexer keys, unpooled bf16 key + gate (as HF caches them) | 11 x 512 B = 5.5 KB | 5.4 GiB |
| indexer keys pooled 4 to 1, fp8 128 + scale (as vLLM stores them, estimate) | 11 x 33 B = 0.4 KB | 0.35 GiB |
| KDA recurrent state, f32 64 x 128 x 128 per layer | per sequence: 34 x 4 MiB = 136 MiB | 136 MiB |
| KDA conv state, f32 3 x 8192 x 3 | per sequence: 34 x 295 KB = 10 MB | 10 MB |

Note the latent is 512 values per layer, not 576: this model has no 64 wide rope part.

Check against the measured vLLM number: the README reports 1,687,708 KV tokens in 7 GB per card x 2 (TP2, DCP2, fp8 KV, `max-model-len` 1M). That is 8.3 KB per token over 12 layers, about 690 B per layer per token, which is the fp8 latent plus the pooled index plus page overhead. The table above is consistent with it.

The 1M context claim holds for memory: one 1M sequence needs 6 to 16 GiB of KV depending on dtype, small next to the weights. What 1M costs is time:

- decode: sparse attention reads 2048 + 3 latent rows per layer, independent of length; the indexer reads n/4 pooled keys per layer: at 1M, 11 x 262,144 x 33 B = 95 MB and about 24 GFLOP per token. Cheap next to the 13 GB of weights.
- prefill: the indexer is quadratic: about 11 x 1,024 x n^2 FLOP, 12.4 PFLOP at n = 1M, on top of about 2 x 16.5 B active parameters per token (33 PFLOP for 1M tokens). The README measured a 240K prompt at 91 s on 2 cards; it limits single requests to 262,144 tokens in front of vLLM and never tested past that with this build.

## How it can run here

| option | what fits | expected |
| --- | --- | --- |
| A. both cards, TP2 (what vLLM does) | 177.4 / 2 = 88.7 GB of weights per card plus KV, as the README does with 7 GB KV per card | about vLLM's decode once the kernels exist. Needs both GPUs free: the user's app holds about 94 GB on each. Needs TP over MLA, KDA, mHC and MoE (`tp::supports` refuses MoE TP except #798's paths) |
| B. GPU1 alone with the expert cache (#1165, #1169, #1173) and the NVMe tier (#1022) | resident 10.4 GB (14.5 with the MTP experts) plus KV plus expert slots; host RAM is 62 GB with about 9 GB available today, so the experts stream from the safetensors files on `nvme0n1p2` (9.7 GB/s measured with `dd iflag=direct` on a shard) | each miss reads 13.4 MB. At hit ratio h, a decode token reads 336 x (1 - h) experts: h = 0.5 gives 2.26 GB per token, about 0.23 s, about 4 tok/s; h = 0.9 gives 0.45 GB, about 20 tok/s. The 122B on the same cache measured hit ratio 0.47 to 0.51 and 2.0 tok/s decode (`docs/specrt-offload.md`). Prefill reads nearly every expert of a layer per chunk: 162.5 GB per pass at 9.7 GB/s is about 17 s per chunk, so roughly 120 tok/s with 2,048 token chunks |
| C. the user stops the app for a window | as A | as A |

B is the only one that runs without the user, and only at the speeds above.

## Order of work (stages 2 to 5)

1. Tiny fixture `tiny-glm5-next` from HF 5.23.0 (4 layers: KDA, KDA, KDA, DSA; 8 experts top 2; layer 0 dense; mHC on; swiglu limit; pool 4, small index top-k). MTP reference from vLLM's `common/mtp.py` formula, recomputed in torch on the CPU, since HF has no MTP.
2. Engine pieces in this order, each with a test: mHC (#1 above), the KDA mixer (#2, reusing `kda_recurrence`), NoPE MLA through the existing DSA nodes (#3, #4), pooled indexer state and top-k at k = 512 (#5), clamped SwiGLU on the MoE (#9). Then MTP over DSA plus MoE (#12).
3. Quant loader: per module map (Q1), FP8_PB_WO scale (Q3), MXFP8 block format (Q2), CSF decoder (Q4). These can be proven on CPU against the real shards without a GPU (dequantise one matrix and compare with vLLM's dequant or the README's re-encode rule).
4. `family-glm5_next.toml` last, flipping `glm5_3_flash_describes_with_its_glm_keys_and_refuses_to_serve`.
5. Real checkpoint run, parked: option A or C needs both GPUs, option B needs the expert cache and NVMe tier on GPU1. Reference: the user's running vLLM (same checkpoint, same machine), logits and top-1 against its logprobs.

## Other MoE configs in the fixtures

`from_hf_config_serving` over every `crates/specrt/tests/fixtures/*.config.json` (from the `*.serving.json` goldens, which `tests/golden.rs` checks): every MoE config is accepted except `glm5_next`.

| config | model_type | experts | serving |
| --- | --- | --- | --- |
| deepseek-v2-lite | deepseek_v2 | 64 | accepted |
| glm4.5-air | glm4_moe | 128 | accepted |
| glm5.3-flash | glm5_next | 288 | refused: `model_type glm5_next is not supported` |
| gpt-oss-20b | gpt_oss | 32 | accepted |
| granite-4.0-h-tiny | granitemoehybrid | 64 | accepted |
| llama4-scout | llama4 | 16 | accepted |
| nemotron-3-nano-30b-a3b | nemotron_h | 128 | accepted |
| qwen3-next-80b-nvfp4 | qwen3_next | 512 | accepted |
| qwen3.5-122b-a10b-nvfp4 | qwen3_5_moe | 256 | accepted |
| qwen3.6-35b-a3b-fp8 | qwen3_5_moe | 256 | accepted |

The only other refusal among the 27 configs is `gemma4-31b` (`model_type gemma4 is not supported`), which is dense (`enable_moe_block` false). DeepSeek V3/R1, Kimi K2, Mixtral and Qwen2-MoE have no config in the fixtures, so they are not covered here; #1183 surveys them.
