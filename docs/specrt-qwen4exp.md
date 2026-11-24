# Qwen3.8-Flash-Next (qwen4_exp) gap list (#1196)

Source of truth: `transformers` 5.23.0 in `~/.venv-vllm`, `models/qwen4_exp/modeling_qwen4_exp.py` (it has the model, no remote code needed). Real config: `Qwen/Qwen3.8-Flash-Next` text config `qwen4_exp_text`, 48 layers, hidden 2560, 3 delta-net then 1 full attention (12 full), 24 heads, 2 KV heads, head_dim 256, partial rotary 0.25, 512 experts top 10 (norm_topk_prob, softmax router), routed and shared width 640, one MTP layer (`hybrid`, full attention), vocab 248320.

The config adds four things to `family-qwen3_5.toml`. Two of them are not the mechanisms #1191 and #875 build, so the issue's "same mHC" and "same indexer" do not hold.

## Pieces

| # | Piece | Qwen4Exp does | Exists in specrt | Status |
|---|---|---|---|---|
| 1 | Hyper-connections | `hc_count` 4 streams of hidden (`repeat` of the embedding). Per block, `attn_hyper_connection` and `mlp_hyper_connection`: grouped RMSNorm of the 4 x 2560 vector (`hc_norm`, group 2560), `input_mix_weight_down` (320 x 10240) / hc_count, silu, `input_mix_weight_up` (10240 x 320), sigmoid, then block input = mean over streams of `mix * normed`. After the block: `out = stream + y * 2 sigmoid(block_inject(normed) / hc_count)` per stream. A final `hyper_connection_mixer` (no inject) collapses the streams before `norm` and `lm_head`. | `Spec.hyper` is read (`[hyper]` in `family-generic.toml`, glm5 style: Sinkhorn comb matrix, `hc_mult`). No engine node. #1191 has only a host reference | **new formulation**: low rank sigmoid gate, no Sinkhorn, no comb matrix. Shares the 4 stream residual buffer and the collapse and expand kernels with #1191, not the compose kernel |
| 2 | QSA pooled indexer, full attention layers only | `index_qk_proj` (hidden to 5 x 128: 4 q heads and 1 k head), RMS norm on q and k, rope on q at its position. Keys are the raw per token k, mean pooled over complete groups of `compress_ratio` 4 visible tokens, normed, roped at the group start. Score = sum over heads of relu(q.k) / sqrt(128). Top `budget/ratio` = 512 groups, expanded to 2048 tokens, plus the incomplete tail tokens. Result is a boolean mask over the normal causal attention (full KV is still stored) | `[attention.indexer]` reads heads, head_dim, topk, kpool. `dsa_index_*` stages exist for glm5 (per head weights, top-k of tokens, key pool with learned ape) | **built** (#1278), see below. Not wired into a family yet |
| 3 | n-gram hashed embedding | Per PLE layer, 16 heads (2 n-gram sizes x `heads_per_ngram` 8). Token history is shifted right by 1 and 2 with EOS reset at each EOS. Per n-gram size, `xor` over `shift_k * multiplier_k` (splitmix64 seeded by `seed` and the PLE index, int64 wraparound), then modulo a distinct prime per head (primes after `ngram_vocab_size_base`), plus the head offset. Each head looks up `ngram_embedding` (rows padded to 128, width `ple_embed_dim / heads`), heads are concatenated | none | **new**: a hash plus gather stage over the token ids (needs int64 multiply and xor in kir, `[[state]]` of the last 2 token ids per slot for decode). The table is huge in the real model (about 45 B parameters at layer 2) so it needs the loader offload path |
| 4 | PLE layer (on `ple_layer_ids`, 1 indexed, linear layers only; the real model uses layer 2) | n-gram embedding, `key_proj` to 4 x hidden, `value_proj` to hidden, grouped norm on keys and on the 4 stream state, per stream gate `sigmoid(sqrt(abs(g)) * sign(g))` with g = key.query / sqrt(hidden), gated value norm, a depthwise conv of kernel 4 with dilation `ngram_size` 3 (silu), output added to every stream before the block | none (the delta-net conv is undilated, one stream) | **new**: dilated depthwise conv with a 9 token state, per layer, over the 4 stream width |
| 5 | Gated attention output | `q_proj` is 2 x heads x head_dim, sigmoid gate on the attention output | `attn_output_gate` read, qwen3_5 path | exists |
| 6 | Gated delta-net | same math as qwen3_5 (`linear_*` keys, 16 key and 48 value heads in the real model) | qwen3_5 | exists |
| 7 | MoE | softmax router, top k, optional renorm, shared expert with sigmoid gate, `silu` | qwen3_5_moe | exists |
| 8 | MTP | `mtp` object with `hybrid` true and `layer_types ["full_attention"]`, one layer, weights excluded from the NVFP4 quantisation | `[mtp]` on qwen3_5. The MTP layer also carries an indexer and hyper-connection weights are not listed for it | needs a check against a real shard (the HF model ignores `mtp.*`) |
| 9 | Vision tower | `Qwen4ExpVisionModel` | qwen3_5 vision | out of scope for this issue (text first) |

Layer I/O names for the family spec: attention `self_attn.{q,k,v,o}_proj`, `q_norm`, `k_norm`, `self_attn.indexer.{index_qk_proj,q_layernorm,k_layernorm}`, delta-net `linear_attn.*` as qwen3_5, `mlp.gate`, `mlp.experts.N.{gate,up,down}_proj`, `mlp.shared_expert*`, `{attn,mlp}_hyper_connection.{hc_norm,input_mix_weight_down,input_mix_weight_up,block_inject_weight}`, `ple.{ple_embedding.ngram_embedding,key_proj,value_proj,norm_key,norm_query,norm_conv,conv1d}`, model level `hyper_connection_mixer.*`. The NVFP4 copy excludes attention, delta-net, router, shared expert, hyper-connections, `ple` and embeddings, so only routed experts are 4 bit.

## QSA indexer (#1278)

Selection rules, read from `Qwen4ExpTextQSAIndexer`: for a query at position p the visible tokens are 0..=p, the complete groups are the first `(p + 1) / ratio`, a group's key is the mean of its raw keys, then `q_layernorm`/`k_layernorm` style RMS norm with weight `1 + w`, then rope at the group's first position (the first `rotary_dim` lanes, halves style, same theta as the attention). The top `min(budget / ratio, groups)` groups are kept, and every token after the last complete group (the incomplete tail, up to `ratio - 1` tokens plus p itself) is always kept. Ties do not occur in the fixture (smallest score gap 0.08).

What was reused and what is new:

- `index_qk_proj` is an appended `[[attention.projection]]` (`append = true`): its rows follow q, k, v and the output gate in the existing qkv GEMM, so there is no extra launch and the loader concatenates it (`Spec::appended_offset`, symbol `qkv_append`).
- `qsa_index_raw_write` (family `state_write`, reused) stores the raw per token index key in the per token state `index_raw`.
- `qsa_index_pool` (new kir row family `group_pool`) runs at the last token of each complete group: mean, norm, rope at the group start, written to the state `index_pool` at that token.
- `qsa_index_select` (new kir row family `group_topk`) norms and ropes the index q at its position, scores the visible groups, keeps the top groups and writes the selected token list (`budget + ratio - 1` slots, `-1` padded) into the int scratch `index_sel`.
- `qsa_attention` (family `scan_attention`, reused with the existing `select_from` stage) is the GQA attention with the output gate, limited to the selected tokens.
- `index_topk` (glm5 DSA) and `block_pool` (DeepSeek V4) are not reused: per token top-k with per head weights, and a softmax gated pool with tail rope pairs.

The nodes, params and states are in `tests/fixtures/tiny-qwen4-exp/qsa-indexer.toml`, written in the syntax a family file uses, so #1280 copies them into `family-qwen4_exp.toml` with `[attention.params]` read from the config (`index_groups = indexer_budget / indexer_compress_ratio`, `index_out = indexer_budget + indexer_compress_ratio - 1`, `index_scale = rsqrt(indexer_head_dim)`) and `layers = ["full"]` on each kernel node. The select kernel keeps one score per group in shared memory, so `max_context / ratio` groups plus the 4 x 128 query must fit 12288 floats: 47 K tokens on the real model. Longer contexts need the scores in global scratch (#1280 or a follow-up).

Tests (`tests/qsa_indexer.rs`): a CPU reference of the selection reproduces the transformers token sets for all 34 rows of the fixture (30 prompt rows, 4 decode rows; decode step 2 completes group 7 and the chosen groups change); a hand built case where the best groups are not the most recent; the spec, loader and kernel module checks; an NVRTC compile; and the GPU test, which runs `qk_norm_rope` then the described nodes on the layer 3 input and compares the selected tokens and group ids exactly and the gated attention output (pre `o_proj`) within 1e-3. The fixture reference comes from `scripts/tiny_qwen4_exp.py indexer`.

## What is done

- `crates/specrt/scripts/tiny_qwen4_exp.py` builds `tests/fixtures/tiny-qwen4-exp` (4 layers: 3 delta-net, 1 indexed attention, 8 experts top 2, hidden 128, PLE on layer 2, indexer budget 16 so selection is active at 30 prompt tokens) and the transformers CPU reference (prefill logits, 4 greedy decode steps recomputed against the full forward at 3e-5, hidden streams of layers 0, 1 and 3). Experts are stored per expert as the checkpoints do. Run it with `~/.venv-vllm/bin/python`.
- `family-generic.toml` aliases `hc_count` and `indexer_*` so `describe_hf_config` reports the 4 streams and the pooled indexer (4 heads, 64 dim, top 16, pool 4).
- `tests/qwen4_exp.rs` pins the description, the tensor names and shapes, the reference, and that the engine still refuses the model.

## Not done, and why `family-qwen4_exp.toml` is not added

A family file makes `from_hf_config_serving` accept the model, so the coverage script would call the real config TOO-BIG while the engine cannot run it. The family file goes in last, after these engine pieces each have a kir description and a test against `tiny-qwen4-exp`:

1. 4 stream residual with the low rank gate, collapse and inject kernels (with #1191's stream buffer when it lands).
2. QSA block pool, score and block top-k stage over a GQA layer, producing a mask or a selected index list for `flash_attention`.
3. n-gram hash and gather stage plus token history state.
4. PLE gate and dilated conv stage.
5. `family-qwen4_exp.toml`, `unsupported` entries removed, `model_coverage.sh` rerun (expected: Qwen/Qwen3.8-Flash-Next TOO-BIG, NVFP4 copy TOO-BIG with 2 cards), GPU1 logits check through `gpu-run 1 --`.
