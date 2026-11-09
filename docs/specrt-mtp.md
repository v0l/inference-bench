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

Per slot, besides KV and recurrent state: a *pending* list of up to `k + 1` normed target hiddens
at positions `q .. q + m - 1`, whose MTP rows have not run yet because their next token was not
known. The newest token always sits at position `q + m`.

- Prefill chunk `[s, s + n)`: after the target, the MTP runs as a prefill over positions
  `s - 1 .. s + n - 2` (the pending row from the previous chunk, then the chunk) with tokens
  `s .. s + n - 1`, writing its KV with `flash_mma`. The hidden at `s + n - 1` becomes pending.
- Speculative step: MTP pass 1 over the pending rows (ragged across sequences, one batched launch
  per kernel) gives `d_1` from each sequence's last row, then `k - 1` single-row MTP steps; the
  target verifies `k + 1` rows per sequence; the accept kernel decides on the device; the commit
  kernels advance the recurrent state by the accepted rows; the hiddens of the accepted rows become
  pending.
- Plain decode on an MTP spec (above `max_batch`): after the decode, the MTP pass 1 runs over the
  pending rows without logits (KV only) and the decode row's hidden becomes pending, so a
  sequence can move between plain and speculative steps.

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

Because the verify row computes exactly what `gdn_step` computes and the GEMMs give each row the
same result whatever the other rows hold, greedy speculative output equals greedy plain output
token for token.

### KV

The MTP layer's attention is one more layer in the block-major KV pool (`Spec::kv_layers()` =
full layers + MTP layers), so a block still holds every layer and the block table is shared.
Blocks are allocated for `tokens + k` so verification and the draft chain write inside the
sequence's table. A request must fit `prompt + max_tokens + k` in `max_model_len`.

### Batching

Every speculating sequence drafts `k` and verifies `k + 1` rows, so the verify batch is dense
(`rows = sequences x (k + 1)`); sequences accept different counts and only the host bookkeeping
differs. Prefill chunks run beside speculative steps unfused (`prefill_slot` then the MTP
catch-up).
