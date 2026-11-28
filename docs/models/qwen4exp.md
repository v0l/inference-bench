# Qwen3.8-Flash-Next (qwen4_exp)

`Qwen/Qwen3.8-Flash-Next` and its FP8 and NVFP4 copies, served by `specs/family-qwen4_exp.toml` (extends `family-qwen3_5`) as text only, with no architecture name in Rust. Reference: transformers 5.23 `models/qwen4_exp/modeling_qwen4_exp.py`; the MTP layer from vLLM `vllm/models/qwen4_exp/nvidia/mtp.py` (transformers ignores it).

## The model

Text config `qwen4_exp_text`: 48 layers, hidden 2560, 3 delta-net then 1 full attention (12 full), 24 heads, 2 KV heads, head_dim 256, partial rotary 0.25, 512 experts top 10 (softmax router, `norm_topk_prob`, routed and shared width 640, sigmoid gated shared expert), one MTP layer, vocab 248,320. Four additions over Qwen3.5:

| Piece | What it does | In the family |
|---|---|---|
| Gated residual | 4 streams of hidden; per block a grouped RMS norm of the 4 x 2560 vector, a low rank sigmoid gate (`input_mix_weight_down`, `input_mix_weight_up`), the block input as the mean of `mix * normed`, and after the block `stream + y * 2 sigmoid(block_inject(normed) / 4)`; a final mixer collapses the streams | `[gated_residual]` with `expand`, `pre` and `post` nodes over `specs/hc_*.toml` (`group_norm`, `head_matvec`, `group_sum`, `group_update`); decoder blocks `gated_linear_moe`, `gated_attention_moe`; buffers in `GatedBuffers`. Not the glm5 mHC (`[hyper]`), and a spec takes one of the two |
| QSA indexer, full attention layers | `index_qk_proj` (4 q heads and 1 k head of 128), keys mean pooled over complete groups of 4 tokens, normed and roped at the group start; relu scores; top `budget / 4` = 512 groups expanded to tokens plus the incomplete tail, as a mask over normal causal attention | `index_qk_proj` appended to the qkv GEMM; `qsa_index_raw_write` (`state_write`), `qsa_index_pool` (`group_pool`), `qsa_index_select` (`group_topk`), `qsa_attention` (`scan_attention` with `select_from`); states `index_raw`, `index_pool`. Group scores live in shared memory up to 47,104 tokens, past that in `[attention.scratch] index_scores` |
| n-gram hashed embedding (PLE) | per PLE layer 16 heads over 2 n-gram sizes, the token history shifted by 1 and 2 with EOS resets, splitmix64 seeded hashes modulo per head primes, a gather from `ngram_embedding` | `[prelude]` on `ple_layer_ids` (layer 2): `ple_ngram_hash` (`token_hash` over `kir::int` 64 bit expressions), `ple_embed_gather` (`group_map`); table rows from an `{ int = ... }` param (320,001,536 on the real model); history state `ngram_history` |
| PLE gate and conv | key and value projections, a per stream gate `sigmoid(sqrt(abs(g)) sign(g))`, a depthwise conv of 4 taps with dilation 3, added to every stream | `ple_key_proj`, `ple_value_proj` (`head_matvec`), `ple_gate` (`group_map`), `ple_conv` (`seq_conv` with `dilation`), `ple_add`; state `ple_conv` |

The delta-net output gate is `sigmoid` (`[linear] gate_key = "output_gate_type"`). The vision tower is ignored.

## MTP

```
e       = fc_embedding(norm_embed(embed(token)))
streams = fc_hidden(norm_hidden(previous streams))   norm over all 4 x hidden
x       = streams + e on every stream
out     = one Qwen4Exp decoder layer (gated residual sites, indexed attention, MoE), no PLE
logits  = lm_head(hyper_connection_mixer(out))
```

`[mtp_layout]` names `fc`, `fc_hidden`, `norm_embed`, `norm_hidden` and `mixer`; `decoder.toml` has gated copies of `[[mtp]]` and `[[mtp_head]]`. Verify rows run the PLE prelude, whose n-gram history and conv window have no per row log, so a round saves those states before verify, restores them after acceptance and replays the hash and conv one position at a time on the accepted tokens.

## Real checkpoint tensors

- The n-gram table is 128 `ple_embedding.ngram_embedding.shard_N.weight` tensors of [2500012, 160], bf16 in the base repo and F8_E4M3 with one bf16 scale in the FP8 and NVFP4 repos. `Checkpoint` joins a complete shard run into `X.leaf`. `[prelude.stored] ple_table` keeps it in its stored dtype (fp8 51 GB, bf16 102 GB); the gather decodes each element in the load. The bf16 copy needs the host tier (#1355).
- The bf16 repo fuses experts (`mlp.experts.gate_up_proj` [512, 1280, 2560], no `.weight` suffix); `[moe] fused_experts = true` reads both spellings.
- `layer_multipliers`, `ngram_heads_offsets` and `ngram_heads_vocab_sizes` are not loaded: the folded hash constants equal them.
- Stages cannot start in front of a prelude layer: `StageRange::check` and the planner refuse it.
- The gated residual has no fused prefill plus decode step and no single row plan.

## Measured state

NVFP4 copy (`RadixArk/Qwen3.8-Flash-Next-NVFP4`) across both desktop cards, a two stage pipeline (gpu0 layers 0..2 with the 51 GB table, gpu1 2..48), engine commit 18a8755e, quiet GPUs (#1353):

| Prompt tokens | Time to first token | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| 20 | 0.29 s | 70 | 3.7 |
| 200 | 0.60 s | 330 | 3.7 |
| 2,000 | 4.24 s | 470 | 3.5 |

Decode is 270 ms a step against the planned 36.5 ms: the pipelined path runs one stream through both stages a row at a time (#1389). Greedy text is coherent ("The capital of France is Paris. Paris is the capital of France. ..."). The FP8 copy gave the same speeds with the split pinned at layer 14. Not compared against vLLM yet.

## Tests

`crates/specrt/tests/qwen4_exp.rs` (tiny `tests/fixtures/tiny-qwen4-exp` from `scripts/tiny_qwen4_exp.py`: streams after layers 0, 1, 3 within 2.6e-4, prefill logits 2.5e-4, 4 decode steps, two served slots, state snapshots), `qsa_indexer.rs` (selection against transformers on all 34 rows), `qwen4_exp_ple.rs` and `crates/kir/tests/ple.rs` (hash bit exact against numpy and transformers), `qwen4_exp_mtp.rs` (draft logits 1.4e-4, speculative greedy equal to plain for k = 1 to 3), `qwen4_exp_checkpoint.rs` (real shard headers in `tests/fixtures/qwen4-exp-real/headers.json`, fp8 and bf16 table gathers bit exact), `moe_mma.rs` and `moe_mma_fp8.rs` (expert width 640).
