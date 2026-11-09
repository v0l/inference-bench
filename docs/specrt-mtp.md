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
graph = false         # optional: capture the whole speculative round in a CUDA graph (needs [runtime] graphs)
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

## Real 27B NVFP4

`scripts/spec27b.sh` under `flock /tmp/gpu27.lock`, 9 Nov 2026, beside the live vLLM
(`local-llm.service`, 51.7 GB, 97 to 100 percent busy): 43.3 GB free before each run and back to
43.3 GB after. Context 8,192, 1 GB of KV, the 20 prompts of `scripts/vllm_logprobs.py` (the 5
real-prompts plus 15 short ones: code, SQL, maths, prose, translation) in
`scripts/spec_prompts.txt`, 128 greedy tokens each, `bench --speculative`. Decode tok/s counts
only steps with no prefill in them. One run each, on a shared card, so read the speeds as
indicative.

| k | batch | mean acceptance length | draft acceptance | decode tok/s, speculative | plain | speedup |
|---|---|---|---|---|---|---|
| 1 | 1 | 1.830 | 83.0% | 27.2 | 23.1 | 1.18x |
| 2 | 1 | 2.428 | 71.4% | 31.4 | 23.1 | 1.36x |
| 3 | 1 | 2.807 | 60.2% | 31.6 | 23.1 | 1.37x |
| 2 | 1, native fp4 + fp8 activations (`qwen3.8-27b-nvfp4-act`) | 2.417 | 70.8% | 30.1 | 23.8 | 1.26x |
| 2 | 4 | 2.428 | 71.4% | 110.1 aggregate | 94.3 | 1.17x |
| 2 | 8 | 2.428 | 71.4% | 115.9 aggregate | 143.7 | 0.81x |

Greedy output with and without speculation agrees on the 27B too (the bench prints the first 24
tokens of prompt 0 for both). The acceptance numbers are the same at every batch size, as they
must be for greedy.

vLLM on the same checkpoint, from its own `/metrics` over its whole uptime of real traffic:
12,123,426 accepted of 13,617,794 drafted in 6,808,897 steps, so mean acceptance length 2.78, draft
acceptance 89.0%, per position 0.933 and 0.847. On the same 20 prompts and 128 greedy tokens
(counter deltas around two runs sent to `local-llm.service`): mean acceptance length 2.66 and 2.65,
draft acceptance 82.8%, position 0 0.917 and 0.918, position 1 0.816 and 0.811. specrt on the
same prompts: 2.43, 71.4%.

Where the gap is not: `real_27b_mtp_dumps_hiddens_and_draft_logits_for_the_cpu_oracle` runs specrt
on 5 of vLLM's own prompt plus greedy sequences and dumps the target hiddens and the chained draft
logits; `reference.py real-mtp` recomputes the MTP in f32 torch from the checkpoint tensors (the
`TinyMtp` module above, independent of specrt's loader). Both drafts are identical on all 5,
logits relative error 2e-6 to 3e-6, KL 1e-7. Teacher-forced on vLLM's greedy tokens
(`real_27b_mtp_acceptance_on_vllms_own_greedy_tokens`), specrt's drafts reach mean length 2.34,
position 0 0.760 and position 1 0.576, and the CPU oracle predicts the token after next on the
generated part at 0.773. So specrt computes the MTP as `qwen3_5_mtp.py` describes it to float
accuracy. The 16-point gap to the served vLLM was a measurement artifact: `local-llm.service` is
97 to 100 percent busy with real traffic (acceptance 0.93 and 0.85 over its uptime), so counter deltas
around two runs include other requests' drafts. An isolated vLLM 0.27.1 instance
(`scripts/vllm_mtp_dump.py`, offline `LLM`, same checkpoint, fp8 KV, k = 2, one request at a time,
the 20 prompts as token ids, 128 greedy tokens) accepts 1,494 of 2,138 drafts over 1,069 steps with
CUDA graphs and torch.compile (1,496 of 2,128 over 1,064 steps in eager mode): mean acceptance
length 2.40, position 0 0.805 (861 of 1,069), position 1 0.592. specrt `bench --speculative` on the
same prompts: 2.43. Its greedy tokens equal vLLM's on 1 of 20 prompts only because of the
target's different kernels, so the drafts are compared on vLLM's own tokens.

`bench --speculative` now prints and emits as JSON the per-position accepted counts and the
rows proposed (`per_position_accepted`, `per_position_proposed`, `per_position_rate`); the
`per_position_acceptance_counts_follow_the_forced_wrong_draft_position` test in `tests/mtp.rs`
checks them on the tiny MTP fixture. On the 20 prompts, 128 greedy tokens, k = 2, batch 1:

| | steps | mean length | position 0 | position 1 |
|---|---|---|---|---|
| isolated vLLM 0.27.1, graphs | 1,069 | 2.40 | 0.805 | 0.592 |
| specrt `bench --speculative` | 1,046 | 2.428 | 0.827 (865) | 0.601 (629) |
| specrt teacher-forced on isolated vLLM's tokens | 1,080 | 2.346 | 0.772 | 0.574 |

On its own trajectories specrt's position 0 is 2.2 points above vLLM's, position 1 0.9 above. The
two systems do not generate the same text (greedy near-ties), and about 1,050 steps give a
standard error of 1.2 points per rate, so the two are equal within measurement noise.

The forward hook in `scripts/vllm_mtp_dump.py` records the drafter's `input_ids`, `positions`,
target `hidden_states` and argmax at every call; the pattern matches the algorithm above.
`scripts/mtp_variants.py` replays vLLM's own recorded target hiddens through the f32 torch oracle
(prompt-wide, the same positions vLLM drafted at) and scores the next-next token against vLLM's
tokens, 1,056 steps:

| MTP precision on vLLM's hiddens | position 0 |
|---|---|
| vLLM's own drafts | 0.7955 |
| f32 | 0.7926 |
| bf16 weights and activations | 0.7917 |
| f32, fp8 (e4m3) rounded K and V in the MTP attention | 0.7926 |
| bf16 plus fp8 K and V | 0.7917 |

The checkpoint's `mtp.*` tensors are unquantised bf16 (`ignore: re:mtp\..*`), so there is no
quantised MTP linear to mimic. MTP precision moves position 0 by less than 0.1 point, and the
oracle fed vLLM's hiddens reproduces vLLM's position 0 to 0.3 points. So a precision option for
the MTP would measure no different and was not added. The remaining difference between specrt
teacher-forced (0.772) and the oracle on vLLM's hiddens (0.793) is about 2 points and sits in the
target hidden state (specrt's FP8 and NVFP4 kernels against vLLM's), not in the MTP path; it does
not show in `bench` (0.827), where specrt's drafter sees specrt's own hiddens on its own text.
The earlier 0.917 and 0.816 for the served vLLM do not reproduce in isolation.

Crossover: speculation still wins at batch 4 (12 verify rows). Until #414 it lost at batch 8 (24
rows): past 16 rows the verify GEMMs left the batched fp4/fp8 GEMV path for the `_tiny` mma tile,
which reads the weights at about 400 GB/s (see `specrt-nvfp4.md`). With the 32-token GEMVs
(`*_f4b32`, `*_f8b32`) a 24-row verify reads every weight once, and batch 8 speculation is level
with plain decode. `scripts/spec27b.sh`-style runs, the same prompts, base (`f5b1ef0`) and #414
binaries interleaved, three rounds, vLLM 93 to 100 percent busy beside it:

| batch | binary | speculative tok/s (3 runs) | plain tok/s (3 runs) | median ratio |
|---|---|---|---|---|
| 4 | base | 297.3, 98.0, 96.3 | 172.9, 88.4, 84.2 | 1.11x |
| 4 | #414 | 276.7, 92.8, 93.0 | 172.8, 86.8, 88.0 | 1.06x |
| 8 | base | 262.8, 106.0, 90.6 | 278.0, 130.0, 141.0 | 0.75x |
| 8 | #414 | 387.7, 147.3, 137.1 | 140.3, 141.3, 137.8 | 1.05x |

The runs swing 3x with vLLM's load, so compare within a row. Batch 4 is unchanged (12 rows were
already on the 16-token GEMV). `[mtp] max_batch = 8` is the setting for this card now; above 32
verify rows (batch 11 at k = 2) the GEMMs take over again.

The 31.4 against 23.1 tok/s of this table is stale: it was taken on a card where vLLM's load
slowed both columns, before the batched GEMV work (#399, 32-row GEMV) and the deferred commit
settled. The paired runs in "Device-side round (#401)" below are current: 85 tok/s speculative
against 40 plain at batch 1 (2.1x for 2.43 tokens per round), and a round of 28.8 ms against a plain
step of 24.4 ms (1.18x the plain step, not 1.8x).

## Commands

```
cd crates/specrt
PY=~/.venv-vllm/bin/python
export SPECRT_REF_DIR=~/.cache/specrt-ref-322
$PY scripts/reference.py tiny-mtp tests/fixtures/tiny-qwen3.5-mtp $SPECRT_REF_DIR
cargo test --release --test mtp -- --include-ignored --test-threads=1
cargo test --release --lib sched
cargo test --release --test real27b real_27b_mtp_weights -- --ignored
flock /tmp/gpu27.lock ./scripts/spec27b.sh          # BATCHES, NEW, SPEC
# vllm-spec-greedy.json: the 20 prompts' greedy tokens from local-llm.service (logprobs 1)
flock /tmp/gpu27.lock cargo test --release --test real27b real_27b_mtp_ -- --ignored --nocapture
$PY scripts/reference.py real-mtp orcarouter/Qwen3.8-27B-Uncensored-NVFP4 $SPECRT_REF_DIR
```

## Device-side round (#401)

The next round's row indices, positions, slots and copy indices are built on the device from
`spec_accept`'s `accepted` output (`mtp_round_in`, `mtp_chain_rows`, `mtp_pick_rows`). Drafts live
in a device buffer per slot. A round uploads one packed buffer (`rin`: tokens, positions, slots,
accept caps, sampling params), runs verify, accept, commit, the MTP pass over the emitted rows and
the k chained drafts, and reads back accept counts and emitted tokens once. The stream is
synchronised once per round (test `a_speculative_round_synchronises_the_stream_at_most_once`).
`[mtp] graph = true` captures that round per padded batch size (1, 2, 4, 8, ...) as `decode` does.

### Paired measurement

`scripts/spec27b.sh`, batch 1 and 4, k = 2, acceptance 2.43, 128 greedy tokens over 20 prompts,
`flock /tmp/gpu27.lock`, at least 37 GB free before each run. Arms alternated base, eager, graph
for 5 rounds (the sixth was stopped). base is origin/master at 6031fd3 built in a temp worktree;
eager and graph are this change with `[mtp] graph` off and on. Speculative decode tok/s, median
(min to max). The served vLLM was 100 percent busy throughout: `nvidia-smi` utilisation sampled
every 2 s during each run averaged 68 to 100 percent per run (about 85 percent over the 15 runs) and
dipped to 1 percent only in the gaps between benchmark phases, so every run shared the card.

| arm | batch 1 | batch 4 | plain batch 1 | plain batch 4 |
|---|---|---|---|---|
| base (before) | 85.5 (70.8 to 87.1) | 246.7 (196.5 to 287.0) | 39.9 (34.4 to 43.9) | 161.0 (101.6 to 171.3) |
| device round, eager | 82.1 (42.8 to 86.5) | 283.3 (152.8 to 283.9) | 41.3 (22.9 to 44.2) | 168.2 (87.4 to 169.7) |
| device round, graph | 82.9 (44.6 to 88.7) | 227.7 (153.0 to 281.7) | 38.4 (23.5 to 40.6) | 167.0 (82.4 to 169.7) |

The spread comes from vLLM taking time slices: the slowest runs of every arm (the 42.8, 44.6 and
the 150s) are the ones where plain decode dropped to the 20s and 80s too. Medians of the three
arms are within that spread at batch 1 (85.5, 82.1, 82.9) and the eager arm leads at batch 4.
Graph gains nothing at batch 1 (-3 percent against base) and nothing reliable at batch 4. Neither
reaches the 15 percent bar, so the graph stays off and the eager device round stays on (the round
is not slower: nsys wall 29.22 ms against 29.38 ms, and the idle gaps fell from 0.74 to 0.10 ms).

### Where a round goes now

`nsys profile --cuda-graph-trace=node`, `bench --speculative` on the same spec, rounds delimited
by `spec_accept` (204 rounds at batch 1, 63 at batch 4), plain steps by `embed` with spec off (471
and 126 steps). Median per round, beside the same busy vLLM:

| | plain step b1 | spec round b1 | plain step b4 | spec round b4 |
|---|---|---|---|---|
| wall | 24.4 ms | 28.8 ms | 25.0 ms | 34.3 ms |
| kernel sum | 23.3 ms | 27.9 ms | 23.9 ms | 32.6 ms |
| idle gaps of 20 us and over | 1.03 ms | 0.60 ms | 0.84 ms | 0.80 ms |
| kernels | 884 | 918 | 884 | 918 |

Spec round at batch 1, by kernel group: weight GEMVs of the 3-row verify (`gate_up_f4b8`
6.1 ms, `lin_in_f8b8` 3.4 ms, `down_f4b8` 3.1 ms, `lin_ba` 1.6 ms, `lin_out` 1.3 ms, qkv, o and the
fp8 MLP: 17.8 ms in all), `lm_head` three times (verify rows plus two draft heads: 5.0 ms, 1.75 ms
each, the plain step pays it once), attention 1.8 ms, MTP layer GEMVs 1.3 ms, delta-net verify and
commit 0.9 ms, norm and embed 0.7 ms, `spec_accept` 0.28 ms, `mtp_draft` 0.04 ms. At batch 4 the
verify GEMVs grow to 20.7 ms (`b16` kernels) and delta-net verify and commit to 2.0 ms. The
round is 1.18x the plain step at batch 1 for 2.43 tokens, so the cost is the weights, not the
host: gaps are 2 percent of the round (0.6 ms), which is all a graph could remove, and the idle
that remains in a 24 ms step is launch gaps of the same size.

Against #413's 84 ms round (52 ms for vLLM): the kernel sum is the same (28.4 ms here, 29.7 ms
there), the wall is not. #413's 34.4 ms of idle gaps was not host synchronisation: the pre-change
build run alone here has 0.74 ms of gaps in a 29.4 ms round, with every upload still
synchronising. It is time slicing against the busy vLLM (the same run's mean kernel sum was
51.6 ms against a 29.7 ms median, and plain decode in that run was 22.5 tok/s against 40 now).
The 84 ms is therefore most likely contention, not the old build (an inference: #413 has no isolated specrt round to check it against).
An isolated vLLM round of 52 ms against specrt's 29 ms is not a like-for-like pair.

What would still cut the round: the two extra `lm_head` passes (3.3 ms, #415 quantises the draft
head) and the verify GEMV bandwidth (17.8 ms, #361, #362); the host side is done.
