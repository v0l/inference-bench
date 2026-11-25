# Qwen3.8-Flash-Next (qwen4_exp) gap list (#1196)

Source of truth: `transformers` 5.23.0 in `~/.venv-vllm`, `models/qwen4_exp/modeling_qwen4_exp.py` (it has the model, no remote code needed). Real config: `Qwen/Qwen3.8-Flash-Next` text config `qwen4_exp_text`, 48 layers, hidden 2560, 3 delta-net then 1 full attention (12 full), 24 heads, 2 KV heads, head_dim 256, partial rotary 0.25, 512 experts top 10 (norm_topk_prob, softmax router), routed and shared width 640, one MTP layer (`hybrid`, full attention), vocab 248320.

The config adds four things to `family-qwen3_5.toml`. Two of them are not the mechanisms #1191 and #875 build, so the issue's "same mHC" and "same indexer" do not hold.

## Pieces

| # | Piece | Qwen4Exp does | Exists in specrt | Status |
|---|---|---|---|---|
| 1 | Hyper-connections | `hc_count` 4 streams of hidden (`repeat` of the embedding). Per block, `attn_hyper_connection` and `mlp_hyper_connection`: grouped RMSNorm of the 4 x 2560 vector (`hc_norm`, group 2560), `input_mix_weight_down` (320 x 10240) / hc_count, silu, `input_mix_weight_up` (10240 x 320), sigmoid, then block input = mean over streams of `mix * normed`. After the block: `out = stream + y * 2 sigmoid(block_inject(normed) / hc_count)` per stream. A final `hyper_connection_mixer` (no inject) collapses the streams before `norm` and `lm_head`. | `Spec.hyper` is read (`[hyper]` in `family-generic.toml`, glm5 style: Sinkhorn comb matrix, `hc_mult`). No engine node. #1191 has only a host reference | **new formulation**: low rank sigmoid gate, no Sinkhorn, no comb matrix. Shares the 4 stream residual buffer and the collapse and expand kernels with #1191, not the compose kernel |
| 2 | QSA pooled indexer, full attention layers only | `index_qk_proj` (hidden to 5 x 128: 4 q heads and 1 k head), RMS norm on q and k, rope on q at its position. Keys are the raw per token k, mean pooled over complete groups of `compress_ratio` 4 visible tokens, normed, roped at the group start. Score = sum over heads of relu(q.k) / sqrt(128). Top `budget/ratio` = 512 groups, expanded to 2048 tokens, plus the incomplete tail tokens. Result is a boolean mask over the normal causal attention (full KV is still stored) | `[attention.indexer]` reads heads, head_dim, topk, kpool. `dsa_index_*` stages exist for glm5 (per head weights, top-k of tokens, key pool with learned ape) | **built** (#1278), see below. Not wired into a family yet |
| 3 | n-gram hashed embedding | Per PLE layer, 16 heads (2 n-gram sizes x `heads_per_ngram` 8). Token history is shifted right by 1 and 2 with EOS reset at each EOS. Per n-gram size, `xor` over `shift_k * multiplier_k` (splitmix64 seeded by `seed` and the PLE index, int64 wraparound), then modulo a distinct prime per head (primes after `ngram_vocab_size_base`), plus the head offset. Each head looks up `ngram_embedding` (rows padded to 128, width `ple_embed_dim / heads`), heads are concatenated | `specs/ple_ngram_hash.toml` (kir `token_hash` family over `kir::int` 64 bit integer expressions), `specs/ple_embed_gather.toml` (`group_map` gather) | **kir done (#1279)**: bit exact against numpy and transformers on the tiny and real configs, EOS resets, int64 wraparound, whole, chunked and one token at a time. Not yet planned into a layer (#1280). The real table (about 45 B parameters) still needs the loader offload path |
| 4 | PLE layer (on `ple_layer_ids`, 1 indexed, linear layers only; the real model uses layer 2) | n-gram embedding, `key_proj` to 4 x hidden, `value_proj` to hidden, grouped norm on keys and on the 4 stream state, per stream gate `sigmoid(sqrt(abs(g)) * sign(g))` with g = key.query / sqrt(hidden), gated value norm, a depthwise conv of kernel 4 with dilation `ngram_size` 3 (silu), output added to every stream before the block | `specs/ple_key_proj.toml`, `ple_value_proj.toml` (`head_matvec`), `ple_gate.toml` (`group_map`), `ple_conv.toml` (`seq_conv` with `dilation`), `ple_add.toml` | **kir done (#1279)**, see below |
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

Tests (`tests/qsa_indexer.rs`): a CPU reference of the selection reproduces the transformers token sets for all 34 rows of the fixture (30 prompt rows, 4 decode rows; decode step 2 completes group 7 and the chosen groups change); a hand built case where the best groups are not the most recent; a CPU attention over the reference token sets that reproduces the gated attention output within 1e-3 (which also pins the rope and norm conventions); the spec, loader and kernel module checks; an NVRTC compile; and the GPU test, which runs `qk_norm_rope` then the described nodes on the layer 3 input and compares the selected tokens and group ids exactly on all rows and the gated attention output (pre `o_proj`) within 1e-3 from the first sparse row (19) on. The fixture reference comes from `scripts/tiny_qwen4_exp.py indexer`; the attention output is stored as int16 with one scale (`indexer.attn.i16`, quantisation error under 3.1e-5) to keep the fixture folder under its 5 MiB budget.

## What is done

- `crates/specrt/scripts/tiny_qwen4_exp.py` builds `tests/fixtures/tiny-qwen4-exp` (4 layers: 3 delta-net, 1 indexed attention, 8 experts top 2, hidden 128, PLE on layer 2, indexer budget 16 so selection is active at 30 prompt tokens) and the transformers CPU reference (prefill logits, 4 greedy decode steps recomputed against the full forward at 3e-5, hidden streams of layers 0, 1 and 3). Experts are stored per expert as the checkpoints do. Run it with `~/.venv-vllm/bin/python`.
- `family-generic.toml` aliases `hc_count` and `indexer_*` so `describe_hf_config` reports the 4 streams and the pooled indexer (4 heads, 64 dim, top 16, pool 4).
- `tests/qwen4_exp.rs` pins the description, the tensor names and shapes, the reference, and that the engine still refuses the model.

## Four stream residual (#1277)

The gated residual is a `[gated_residual]` table: no Rust names it, and the kernels are kir descriptions in `crates/specrt/specs/hc_*.toml`. It is separate from glm5's `[hyper]` (#1191: Sinkhorn, comb matrix, `hyper_*` rows); a spec takes one of the two.

- Spec. `[gated_residual]` has `streams` (config key), `module` (the final mixer), `sites` (one or two per block modules), `params`, `scratch` and `[[gated_residual.node]]`. Each node is `expand` (embedding to every stream), `pre` (before a block; a node with `when = "site"` runs only inside blocks, so the final mixer skips the inject weights) or `post` (after a block). Nodes bind `x`, `streams`, a scratch buffer or a weight under the site module. `Spec::validate_hyper` refuses `[hyper]` beside it, pre or sandwich norms, MTP and mixers, and checks every grid symbol, buffer and weight length.
- Kernels. `hc_norm` (`group_norm`, `(1 + w) * n` per hidden wide group), `hc_down`, `hc_up`, `hc_inject` (`head_matvec`), `hc_act` (`compose`, `silu(v / streams)`), `hc_mix` (`group_sum`: `sum over streams of sigmoid(gate) * normed / streams`), `hc_inject_add` and `hc_expand` (`group_update`: `stream + y * 2 sigmoid(c / streams)` and the copy). `group_sum` and `group_update` are generic stream families in `kir::recur`, tested in `kir/tests/recur.rs`.
- Plan. `decoder.toml` has `gated_linear_moe` and `gated_attention_moe` blocks: `gated_pre 0`, the qwen3_5 block body reading `x` (no input norm, `split` straight to planes), `clear x`, the output linear into `x`, `gated_post 0`, then the same around the MoE (the route skips its own norm, `MoeShape::input_norm`). The head adds `gated_expand`; the tail runs `gated_collapse` then `split` instead of the final norm. Each graph node lowers to one `Op::Gated { node, site }` per described node.
- Stream buffer. `GatedBuffers` in `runtime.rs` owns the 4 x hidden per row stream buffer and the scratch.
- Test. `the_four_stream_residual_matches_the_reference_streams_of_layers_0_1_and_3` (ignored, GPU) loads the fixture with `tests/fixtures/tiny-qwen4-exp/hyper-harness.toml`, a test only family that extends `qwen3_5` and adds the `[gated_residual]` table, through `Spec::from_hf_config_serving_in`. `Runtime::trace_streams` runs one prefill chunk, copies the streams after each layer and adds stubbed deltas before chosen layers: the PLE output of layer 1 is the first 30 rows of `tiny-qwen4-exp.ple.out.bin` (#1279's capture). Layers 0 and 1 compare every row; layer 3 compares the rows before the indexer selects (positions whose complete groups fit the 16 token budget). The bar is the qwen3.5 tiny fixtures' 1e-3, and the test also checks that leaving out the PLE delta fails at layer 1.
- Delta-net output gate. Qwen4Exp gates the gated RMSNorm of the delta-net output with `output_gate_type` (fallback `hidden_act`); the tiny fixture uses `sigmoid`, qwen3_5 uses `silu`. `[linear] gate_key = "output_gate_type"` in a family sets `LinearAttention::gate_sigmoid`, and `kir::linear` emits `la_sigmoid` instead of `silu` in `gdn_step`, `gdn_verify` and `gdn_gated_norm` (`kir/tests/linear.rs`); without the key the text is unchanged (`linear_unchanged`). The GPU check found this: before it, layers 0, 1 and 3 were off by about 4.
- Reference size. The hidden streams are stored as int16 with a per layer scale (`tiny-qwen4-exp.hidden.streams.i16`, scales in `hidden.json`, rounding at most 2.4e-4, a quarter of the bar); the PLE stub reuses #1279's `ple.out.bin`. The fixture folder is 5087428 bytes.
- GPU1 (shared with the user's app, `gpu-run 1 --min-free 2000`): layer 0 max abs 9.5e-5, layer 1 2.6e-4, layer 3 2.4e-4 (19 rows), including the int16 rounding.

## n-gram hash and PLE (#1279)

Order, read from `Qwen4ExpTextPLELayer.forward`: `out = gated + silu(conv(norm_conv(gated)))` where `gated[s] = sigmoid(sqrt(abs(g_s)) sign(g_s)) * value` and `g_s = norm_key(key)[s] . norm_query(x)[s] / sqrt(hidden)`; the residual is `x + out` per stream. Norms are grouped per stream with a `1 + w` weight. The conv state holds the last `(taps - 1) * ngram_size` = 9 normed inputs per channel; the n-gram history holds the last `ngram_size - 1` tokens, starts as EOS, and is stored xor EOS so a zeroed slot means "all EOS" (prefill chunks and slot reuse need only the usual zeroing).

kir pieces added:

- `kir::int`: 64 bit integer expressions (`+ - * / % ^ & | << >>` wrapping, `rem` as floor modulo of signed values like `torch.remainder`, `ge`, `select`, `min`, `max`) plus build time `splitmix64`, `prime_after`, `prime_sum`, which fold to constants.
- `token_hash` sequence family: `[[int]]` entries `per = "const" | "head" | "row"` over the shape symbols, the history `t0..tK` (EOS reset already applied) and `head`; `index` gives the row id per head. One block per sequence, history in a per sequence `[[state]]`. `TokenHash::eval` is the host model of the kernel.
- `group_map` sequence family: per group (stream) loads and gathers, element values, block sums, scalars and stores. Used for the gather, the gate and the add.
- `seq_conv` takes `dilation` (default 1 emits the old text byte for byte).

Spec side for #1280: the params the descriptions read are in `tests/qwen4_exp_ple.rs` `PARAMS` (`streams`, `ngram_heads`, `ngram_base`, `ngram_seed`, `ngram_eos`, `ple_embed_dim`, `ple_head_dim`, `ple_conv_taps`, `ple_conv_state`, `ple_gate_scale`, plus `ple_index` and `ple_layers` per layer) and the two states in `STATES` (`ngram_history`, `ple_conv`, owned by the PLE layer). The family file wires them as `[[mixer.node]]` style nodes before the attn hyper-connection of each `ple_layer_ids` layer.

Tests: `crates/kir/tests/ple.rs` (CPU: integer semantics, the hash description against `reference/tiny-qwen4-exp.ngram.json`, five cases checked against transformers when written, red when a multiplier changes), `crates/specrt/tests/qwen4_exp_ple.rs` (CPU: params from the tiny and real configs, every description builds, the state table sizes, the host reference against the transformers PLE capture at 2e-6; GPU: prefill and decode steps 1 to 4, chunked prefill plus slot reuse, two sequences side by side). Regenerate the references with `~/.venv-vllm/bin/python crates/specrt/scripts/tiny_qwen4_exp.py ple`.

## Not done before #1280, and why `family-qwen4_exp.toml` waited

## The family (#1280)

`crates/specrt/specs/family-qwen4_exp.toml` matches `qwen4_exp` and `qwen4_exp_text` and extends `qwen3_5` (delta-net, MoE with a sigmoid gated shared expert, gated attention output). It adds, as data only:

- `[gated_residual]` from #1277 (copied from the old test harness, now removed).
- The QSA indexer from #1278: `[attention.params]` read from the config (`index_groups = indexer_budget / indexer_compress_ratio`, `index_out = indexer_budget + indexer_compress_ratio - 1`, `index_scale = rsqrt(indexer_head_dim)`), the appended `index_qk_proj` projection, the `index_raw` and `index_pool` per token states and the four kernel nodes. No `layers = ["full"]` is needed: attention nodes only run on attention layers.
- The PLE from #1279 as a `[prelude]`: a new spec table of described kernel nodes that run at the start of each layer named by `ple_layer_ids` (1 based), before the attn gated residual. Its seven nodes are the `specs/ple_*` descriptions in the order of `Qwen4ExpTextPLELayer.forward`: hash, gather, key and value projections, gate, dilated conv, add into the streams. `ngram_history` and `ple_conv` are `layers = "prelude"` per sequence states, zeroed on reset and carried in state snapshots (the slot pending section), swap and sessions. The table row count is an `{ int = ... }` param: `ceil(prime_sum(ngram_vocab_size_base - 1, 1, ngram_heads) / 128) * 128`, 4096 on the tiny config and 320001536 on the real one (128 shards of 2500012).
- The vision tower is ignored (text only, #1237). The MTP layer is refused by name with `[[refuse]] part = "mtp"` (#1350): the spec drops `[mtp]` and records the message, so text serves and `[speculative] enabled = true` fails with it.

The gated residual has no fused prefill plus decode step and no single row plan, so the scheduler runs prefill and decode as separate steps.

GPU1 (`gpu-run 1 --min-free 2000`, the user's app on the card), `tests/qwen4_exp.rs`, bar 1e-3 as the qwen3.5 tiny fixtures:

| Check | Max abs |
|---|---|
| layer 0 streams, 30 rows | 9.5e-5 |
| layer 1 streams, 30 rows, PLE computed in the engine | 2.6e-4 |
| layer 3 streams, all 30 rows (the indexer selects from row 19) | 2.5e-4 |
| prefill logits, 30 rows, every row's argmax equal | 2.5e-4 |
| decode steps 1 to 4, greedy equal | 1.4e-4, 1.4e-4, 2.0e-4, 1.7e-4 |
| in engine PLE against the injected #1279 capture, layer 1 | 2.4e-5 |

Two served slots (prefill in chunks of 7, then 4 decode steps side by side) give the transformers greedy for slot 0 and the same tokens for slot 1 as its prompt served alone, also with the slots swapped after reuse. A state snapshot of slot 0 restored into slot 1 decodes the same greedy tokens; with the prelude restore disabled it does not (checked red once).

The expert GEMMs were refused at an expert width of 640 by an old `% 256` rule; the grouped NVFP4 and FP8 MMA kernels only need `N % 64` and `K % 128`, and both match the CPU at hidden 512, intermediate 640 on GPU1 (`moe_mma.rs`, `moe_mma_fp8.rs`), so the rule is now `% 128` for quantised experts.

Coverage (`model_coverage.sh`): Qwen/Qwen3.8-Flash-Next (360 GB bf16) 2 cards plus 21 GB host plus 166 GB NVMe, the planner models cards and host only; Qwen/Qwen3.8-Flash-Next-FP8 (185.5 GB) 2 cards plus 13 GB host, planner accepts; RadixArk/Qwen3.8-Flash-Next-NVFP4 (135.2 GB) 2 cards, planner accepts. All three SUPPORTED-WITH-OFFLOAD. The GGUF copy waits on #1204 (the GGUF spec is built in Rust and has none of these tables); the MXFP4 GPTQ copy on its quant format.

## Real checkpoint tensors (#1349)

Read from the three repos' shard headers (no download); names, dtypes, shapes, the fp8 scale and the three I64 buffers are in `tests/fixtures/qwen4-exp-real/headers.json`.

- The n-gram table is 128 `ple_embedding.ngram_embedding.shard_N.weight` tensors of [2500012, 160]: bf16 in the base repo, F8_E4M3 in the FP8 and NVFP4 repos with one bf16 `ngram_embedding.weight_scale` (1.99e-4). `Checkpoint` joins any complete `X.shard_0..N.leaf` run into `X.leaf` along rows (`has`, `shape_of`, `dtype_of`, `parts`), so the family still binds `ple_embedding.ngram_embedding.weight`.
- `[prelude.stored]` names prelude weights that stay in their stored dtype: `ple_table = "ple_embedding.ngram_embedding.weight"`. At load `complete_prelude` reads the dtype from the checkpoint into `[prelude.words]` (`ple_table_dtype` = f32, bf16 or fp8) and the fp8 scale into `ple_table_scale`. The gather's `table` param carries `stored = "ple_table"`, so kir types it `bf16_t` or `unsigned char` and decodes each element (bf16 by shift, e4m3 times the scale) in the `group_map` load. The table is copied part by part from the mmap into one device buffer (fp8 51 GB, bf16 102 GB) and kept out of the staged prefetch groups. The bf16 copy needs the host tier (#1355).
- The bf16 repo fuses experts as `mlp.experts.gate_up_proj` [512, 1280, 2560] (gate rows first) and `mlp.experts.down_proj` [512, 2560, 640] with no `.weight` suffix; `[moe] fused_experts = true` loads both spellings, with the per matrix `shared_expert` appended. The FP8 and NVFP4 repos use per expert names.
- `layer_multipliers`, `ngram_heads_offsets` and `ngram_heads_vocab_sizes` are not loaded: the folded `ple_ngram_hash` constants equal them (checked on CPU against the real values).
- The QSA select keeps group scores in shared memory, so `max_context` stops at 47104 on the real model (#1356).

Tests (`tests/qwen4_exp_checkpoint.rs`): CPU, the real header names join into the table the family reads with the folded `ngram_rows`; the hash constants equal the real buffers; an fp8 sharded copy of the tiny table loads through the name map in its stored bytes; bad `[prelude.stored]` or dtype words are refused; fused bf16 experts load identical stacks to the per expert copy. GPU1: the fp8 gather equals the f32 gather of the decoded table, and the bf16 gather reads the table bit for bit; the engine on bf16 shards and on fused experts gives logits identical to the original fixture, and on fp8 shards identical to the decoded f32 table.

The real load itself needs both cards and a window (#1353).

## History: why the family came last

A family file makes `from_hf_config_serving` accept the model, so the coverage script would call the real config TOO-BIG while the engine cannot run it. The family file goes in last, after these engine pieces each have a kir description and a test against `tiny-qwen4-exp`:

1. 4 stream residual with the low rank gate, collapse and inject kernels: done in #1277 (above).
2. QSA block pool, score and block top-k stage over a GQA layer: done in #1278.
3. n-gram hash and gather stage plus token history state: done in #1279.
4. PLE gate and dilated conv stage: done in #1279.
5. `family-qwen4_exp.toml`: done in #1280 (above).
