# specrt: MTP speculative decoding with delta-net rollback

Issue #322. Qwen3.5/3.8 checkpoints carry one multi-token-prediction (MTP) layer that drafts the
next tokens from the target's last hidden state. vLLM serves the 27B with
`{"method":"mtp","num_speculative_tokens":2}` and gets a mean acceptance length of 2.79 tokens per
step. This document is the algorithm as vLLM runs it, the tensors as they are on disk, and how
specrt does the same with a hybrid model whose recurrent state cannot simply be truncated.

## Configuration rule

Everything is set from config, never from environment variables: the `[mtp]` block of the model
spec (the layer, its weight format, the draft length, whether to speculate and the batch size
above which speculation switches off) and the `[speculative]` table of the server config, which
overrides the spec's tunables. Tests set config, not env vars. No `SPECRT_*` variable exists for
anything in this document.

```toml
[mtp]
layers = 1
quant = "bf16"        # weight format of the MTP decoder layer; fc and norms are always bf16
draft_tokens = 2      # k, 1 to 3
speculate = true
max_batch = 8         # optional: plain decode when more sequences decode at once
```

```toml
# server config
[speculative]
draft_tokens = 3
enabled = true
max_batch = 4
```

`Spec::from_hf_config` adds `[mtp]` when `mtp_num_hidden_layers` is 1 and the checkpoint has the
`mtp.*` tensors' format in its quantisation config: compressed-tensors checkpoints ignore `mtp.`
(bf16), the block-FP8 checkpoints quantise the MTP projections to FP8 like the rest.

## Tensors on disk

`orcarouter/Qwen3.8-27B-Uncensored-NVFP4`, `model-extra-00001-of-00001.safetensors`, all bf16:

| tensor | shape |
|---|---|
| `mtp.fc.weight` | [5120, 10240] |
| `mtp.pre_fc_norm_embedding.weight` | [5120] |
| `mtp.pre_fc_norm_hidden.weight` | [5120] |
| `mtp.layers.0.input_layernorm.weight` | [5120] |
| `mtp.layers.0.self_attn.q_proj.weight` | [12288, 5120] (query and output gate per head, as in the target) |
| `mtp.layers.0.self_attn.{k,v}_proj.weight` | [1024, 5120] |
| `mtp.layers.0.self_attn.o_proj.weight` | [5120, 6144] |
| `mtp.layers.0.self_attn.{q,k}_norm.weight` | [256] |
| `mtp.layers.0.post_attention_layernorm.weight` | [5120] |
| `mtp.layers.0.mlp.{gate,up}_proj.weight` | [17408, 5120] |
| `mtp.layers.0.mlp.down_proj.weight` | [5120, 17408] |
| `mtp.norm.weight` | [5120] |

`Qwen/Qwen3.8-27B-FP8` (`mtp.safetensors`) and `orcarouter/Qwen3.8-27B-Uncensored-FP8`
(`model-00007-of-00007.safetensors`) have the same names; there the `self_attn` and `mlp`
projections are `F8_E4M3` with `weight_scale_inv` per 128 x 128 block, and `fc` and the norms are
bf16. All norms are stored `w - 1` like the target's (`(1 + w)` RMSNorm). There is no MTP
embedding or head: the layer shares `embed_tokens` and `lm_head` with the target
(`mtp_use_dedicated_embeddings = false`). 0.85 GB on the device.

## vLLM's algorithm

From `model_executor/models/qwen3_5_mtp.py`, `v1/spec_decode/llm_base_proposer.py` and
`v1/sample/rejection_sampler.py` (vLLM in `~/.venv-vllm`).

**The MTP layer** (`Qwen3_5MultiTokenPredictor.forward`). For a row with token `t` and hidden `h`:

```
x = fc([rmsnorm_e(embed(t)), rmsnorm_h(h)])      # concat embedding first, then hidden
x = full-attention decoder layer(x, position)    # own KV cache, same shape as a target full layer
g = rmsnorm_mtp(x)
draft logits = lm_head(g); next hidden = g
```

`h` is the target's final hidden state after the model's final norm (the tensor its `lm_head`
reads).

**Draft chain** (`propose`). The first pass is over every token the target just computed, shifted
by one: row `i` gets the target hidden at position `q_i` and the token at `q_i + 1`, at position
`q_i`; the last row of a request gets the token the target just sampled. That pass writes the MTP
KV for all of them and its last row's logits give draft 1. Drafts 2..k are single rows: token
`d_j`, hidden `g_j` (the MTP's own output), position plus one. Drafts are argmax
(`draft_sample_method = "greedy"`, the default), so there are no draft probabilities.

**Verification.** The target runs on `[t, d_1 .. d_k]` at positions `p .. p + k` in one forward,
giving `k + 1` logit rows.

**Acceptance.** Greedy requests (`rejection_greedy_sample_kernel`): accept `d_j` while it equals
the target argmax of row `j - 1`; the first mismatch emits the target argmax and stops; if all `k`
match, the argmax of row `k` is the bonus token. Sampled requests (`rejection_random_sample_kernel`
with `NO_DRAFT_PROBS`): with `p_j` the temperature softmax of row `j`, accept `d_{j+1}` when
`p_j(d_{j+1}) >= u` (`u` uniform); on rejection emit a recovered token sampled from `p_j` with
`d_{j+1}` removed (`sample_recovered_tokens_kernel`, an exponential race); after `k` accepts sample
the bonus from `p_k`. With a one-hot draft this emits `x` with probability `p(x)` exactly, so the
output follows the target distribution.

**State commit.** The `a + 1` emitted tokens are appended; the target KV is valid through
position `p + a` and later positions are simply overwritten by the next step. The next step's MTP
first pass is over exactly those `a + 1` positions with the target hiddens.

**Metrics.** vLLM's `SpecDecoding metrics` line logs drafted and accepted draft tokens; mean
acceptance length = `1 + accepted / (drafted / k)`.

## specrt

### State

Per slot, besides KV and recurrent state: one *pending* target hidden (the residual before the
final norm) for the newest computed position, and the `k` drafts made from it. `mtp_fc_in` applies
the target's final norm (or, for chained rows, the MTP's own `mtp.norm`) and then the pre-fc norm,
so the pending buffer holds raw residuals for both.

- Prefill chunk `[s, s + n)` (`mtp_after_prefill`): after the target, the MTP runs as a prefill over
  positions `s - 1 .. s + n - 2` (the pending row from the previous chunk, then the chunk) with
  tokens `s .. s + n - 1`, writing its KV with `flash_mma`. The hidden at `s + n - 1` becomes
  pending; the drafts are dropped.
- Speculative step (`Runtime::speculate`): if a row has no drafts yet, one MTP row (pending hidden,
  newest token) and the chain make them. The target verifies `k + 1` rows per sequence
  (`verify_rows`), `spec_accept` decides on the device, the commit kernels advance the recurrent
  state by the accepted rows, then the MTP runs over the `a + 1` accepted rows (writing its KV) and
  chains `k` new drafts from the last one.
- Plain decode on an MTP spec (`mtp_after_decode`, above `max_batch`): after the decode the MTP
  writes its KV for the row before and the decode row's hidden becomes pending, so a sequence can
  move between plain and speculative steps.
- `set_drafts` replaces a slot's drafts; the tests use it to force rejections.

### Delta-net rollback: deferred commit

A delta-net state is 3.1 MB per layer per sequence (151 MB per sequence), so a snapshot ring of
`k + 1` states would cost 450 MB per sequence. specrt does not write the state during
verification at all:

- `gdn_conv_verify` builds each row's conv window from the stored window plus the earlier rows of
  the same sequence and logs the row's conv input.
- `gdn_verify` runs the recurrent step of `gdn_step` over the `k + 1` rows of a sequence in order,
  with the state in registers, and logs per row and value head the normalised key, the delta
  `beta (v - S^T k)` and the decay. It writes the outputs but never the state.
- After acceptance, `gdn_commit` reloads the state and applies the logged rank-1 updates of the
  first `a + 1` rows (`S = exp(g) S; S += k delta^T`, the same float operations in the same order
  as `gdn_step`), and `gdn_conv_commit` shifts the accepted inputs into the window.

The extra cost is one more read of the state per step (151 MB, about 0.1 ms on this card) and a
log of 1.9 MB per sequence and row on the 27B. Rejected rows cost nothing to undo. KV needs no
rollback: rejected positions are overwritten before they are read.

Because the verify row computes exactly what `gdn_step` computes, greedy speculative output equals
greedy plain output token for token as long as the GEMMs pick the same tile for the verify batch as
for plain decode (`tc` uses the `_tiny` tile, and the batched fp4/fp8 GEMVs, up to 16 rows, so up
to 5 sequences at k = 2). Past 16 verify rows the GEMMs switch tile and K step, which sums in a
different order: the result is still correct to the float tolerance but a near-tie can flip a
greedy token. The tests cover 1 to 5 sequences (4 to 15 rows).
The delta-rule multiplies and adds are written as `__fmul_rn` and `__fmaf_rn` in `gdn_step`,
`gdn_verify` and `gdn_commit` so the compiler cannot contract them differently in the three kernels.

### KV

The MTP layer's attention is one more layer in the block-major KV pool (`Spec::kv_layers()` =
full layers + MTP layers), so a block still holds every layer and the block table is shared.
Blocks are allocated for `tokens + k` so verification and the draft chain write inside the
sequence's table. A request must fit `prompt + max_tokens + k` in `max_model_len`.

### Batching

Every speculating sequence drafts `k` and verifies `k + 1` rows, so the verify batch is dense
(`rows = sequences x (k + 1)`); sequences accept different counts and only the host bookkeeping
differs. The scheduler speculates when the decoding sequences fit `[mtp] max_batch` (default: every
slot) and decodes plainly otherwise; prefill chunks then run in the same step through
`prefill_slot`, unfused, because the fused step has no MTP catch-up. A row's `max_accept` caps the
accepted drafts at what is left of `max_tokens`, so a sequence never emits past its limit; a stop
token in the middle of an accepted run ends the sequence there and the rest is dropped.

## Tests

The oracle for the MTP layer is a small torch module written for this (`TinyMtp` in
`scripts/reference.py`): transformers 5.15 has no Qwen3.5 MTP, it drops `mtp.*` on load. It is built from transformers'
`Qwen3_5DecoderLayer` (layer type `full_attention`), `Qwen3_5RMSNorm` and the target's own
`embed_tokens`, `rotary_emb` and `lm_head`, wired as vLLM's `Qwen3_5MultiTokenPredictor.forward`.
The fixture `tests/fixtures/tiny-qwen3.5-mtp` is `tiny-qwen3.5-nvfp4` (NVFP4 MLPs, FP8 channel
projections) plus `mtp.safetensors` with random bf16 weights (seed 7) and `mtp_num_hidden_layers =
1` in its config. `reference.py tiny-mtp tests/fixtures/tiny-qwen3.5-mtp <out>` writes it the first
time and dumps `tinymtp-prompt.*`: the target hiddens and logits for a 96-token prompt, the MTP's
first-pass logits over all 96 positions, and the logits of the chained second draft.

| test (`tests/mtp.rs` unless noted) | what it proves |
|---|---|
| `tiny_mtp_draft_logits_match_the_torch_reference` | first pass over the prompt plus one chained step: logits max abs 2.8e-5 (scale 3.7), both drafts equal the torch argmax |
| `drafts_after_speculative_steps_equal_drafts_from_a_fresh_prefill_of_the_same_prefix` | after speculative steps with forced rejections the MTP KV and pending hidden are what a fresh prefill of the accepted prefix gives (k = 3) |
| `speculative_greedy_is_identical_to_plain_greedy_for_128_tokens_over_20_prompts` | for k = 1, 2, 3: MTP drafts, oracle drafts (everything accepted), a wrong draft at each position 0..k, a rotating wrong position, random drafts; 20 prompts of 5 to 124 tokens, 128 tokens each, byte-identical to plain greedy |
| `batched_speculation_with_different_accept_counts_matches_plain_greedy` | 4 and 5 sequences per verify batch, rotating rejection position per sequence so rows accept 0..k in the same step |
| `rejection_sampling_follows_the_target_distribution_at_both_draft_positions` | temperature 0.3, 4,000 seeds per case: emitted token at draft position 0 against the target softmax, chi-square 30.2 on 26 dof (z 0.58), accept rate 0.352 for p(draft) 0.346; position 1 after accepting position 0, chi-square 27.1 on 24 dof, accept 0.161 for 0.166; a low-probability draft, chi-square 19.9 on 26 dof |
| `scheduled_speculation_matches_single_sequence_greedy_and_counts_drafts` | 12 requests through the `Scheduler` on 6 slots with chunked prefill beside speculation, with and without `max_batch = 3`; output equals plain greedy, every block is returned, drafted counts match the runtime's |
| `sched.rs` `speculative_steps_with_different_accept_counts_match_single_sequence_output_under_preemption` | fake executor that checks every KV write goes through the block table and the recurrent state position: 64 sequences, k = 1..3, every accept count 0..k seen, stop tokens inside accepted runs, preemption, `max_batch` switch to plain decode |
| `real27b.rs` `real_27b_mtp_weights_load_from_both_checkpoints_with_fp8_dequantised_to_bf16` | the MTP layer loads from the NVFP4 and the FP8 27B checkpoints; the FP8 one is dequantised bit for bit |

Mutation checks: doubling the accept probability in `spec_accept` fails the sampling test (z 314,
accept 0.706 for 0.346); committing one row too many in `gdn_commit` fails the fresh-prefill test.

The random-weight MTP of the tiny fixture predicts nothing, so its own drafts are almost never
accepted (mean acceptance length 1.000); the oracle and forced-wrong drafts are what exercise
acceptance and rollback on it.

## Commands

```
cd crates/specrt
PY=~/.venv-vllm/bin/python
export SPECRT_REF_DIR=~/.cache/specrt-ref-322
$PY scripts/reference.py tiny-mtp tests/fixtures/tiny-qwen3.5-mtp $SPECRT_REF_DIR
cargo test --release --test mtp -- --include-ignored --test-threads=1
cargo test --release --lib sched
cargo test --release --test real27b real_27b_mtp_weights -- --ignored
flock /tmp/gpu27.lock ./scripts/spec27b.sh          # BATCHES, NEW
```
