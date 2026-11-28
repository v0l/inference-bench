# MTP speculative decoding

Checkpoints with a multi token prediction layer (Qwen3.5 and 3.8, the Qwen4Exp and GLM families through `[mtp.layout]`) draft the next tokens from the target's last hidden state; the target verifies `k + 1` rows in one forward. Code: `crates/specrt/src/mtp.rs`, `crates/specrt/src/runtime/mtp.rs`, `crates/specrt/src/drafts.rs`, the `[[mtp]]` and `[[mtp_head]]` graph nodes and `Phase::Verify` ([kir.md](kir.md)).

## Options

Model spec `[mtp]`:

| Key | Default | Meaning |
|---|---|---|
| `layers` | from config (`mtp_num_hidden_layers`) | MTP layers |
| `quant` | checkpoint | weight format of the MTP decoder layer; `fc` and norms are bf16 |
| `draft_tokens` | 2 | k, 1 to 3 |
| `speculate` | true | |
| `max_batch` | every slot | above this many decoding sequences, decode plainly |
| `graph` | false | capture the whole round per padded batch size |
| `defer_commit` | true | fold the delta-net commit into the next round's verify |
| `head_quant` | none | the draft head's format, see `speculative.head_quant` |

Server or load options `speculative.*` override the spec: `enabled` (true), `draft_tokens` (a number or `auto`), `max_batch`, `head_quant` (`nvfp4_g16`, `bf16` under `gemm_fp32`; `bf16` shares the target head). `Spec::from_hf_config` adds `[mtp]` when the config has one MTP layer and the checkpoint has its tensors.

`draft_tokens = "auto"` compiles k = 1 to 3 and picks k per step from the measured step time per (rows, k) and acceptance per position (Laplace's rule), maximising expected tokens per second; k = 0 is retested on a doubling schedule. Output is independent of k. On the 27B it is within 0.5 percent of fixed k = 3, so the default stays fixed (#403).

## The round

The MTP layer (vLLM's `Qwen3_5MultiTokenPredictor`): `x = fc([rmsnorm_e(embed(t)), rmsnorm_h(h)])`, one full attention decoder layer with its own KV, `g = rmsnorm_mtp(x)`, draft logits from `lm_head(g)`. `h` is the target hidden after the final norm. The MTP's KV is one more layer in the block major pool (`Spec::kv_layers()`), and blocks are allocated for `tokens + k`; a request must fit `prompt + max_tokens + k`.

Per slot the runtime keeps one pending target hidden and the k drafts made from it.

1. Prefill chunk: after the target, the MTP runs over the chunk shifted by one and writes its KV; the last hidden becomes pending.
2. Round (`Runtime::speculate`): the target verifies `k + 1` rows per sequence; `spec_accept` decides on the device; the MTP runs over the accepted rows and chains k new drafts. Row indices, positions and copies for the next round are built on the device (`mtp_round_in`, `mtp_chain_rows`, `mtp_pick_rows`), one packed upload and one read back per round, one stream sync.
3. Plain decode above `max_batch` keeps the MTP KV and pending hidden current (`mtp_after_decode`), so a sequence can switch between plain and speculative steps.

Acceptance: greedy accepts while the draft equals the target argmax and emits the target token at the first mismatch, or a bonus token when all k match. Sampled rows accept `d` when `p(d) >= u` and on rejection sample from `p` with `d` removed, so output follows the target distribution. `max_accept` caps accepts at what is left of `max_tokens`; a stop token inside an accepted run ends the sequence there.

## Delta-net rollback

The 27B's delta-net state is 151 MB per sequence, so there is no snapshot ring. `gdn_conv_verify` and `gdn_verify` run the `k + 1` rows with the state in registers and log per row the normalised key, the delta and the decay. After acceptance `gdn_defer` records the accepted count and the conv window is shifted; the state is not written. The next round's `gdn_verify` first replays the pending rank one updates (the same float operations in the same order as `gdn_step`, written with `__fmul_rn` and `__fmaf_rn`), stores once, then verifies. Anything else that reads or replaces a slot's state settles it first (`gdn_commit`). The state crosses the bus twice per round instead of three times; `defer_commit = false` commits every round.

Greedy speculative output equals plain greedy as long as the verify GEMMs pick the same kernels as plain decode. Above 8 rows act quant is allowed and past 16 rows the tile and K step change, so a near tie can flip a token (correct to float tolerance).

## Measured state

27B NVFP4, k = 2, desktop GPU1, 20 prompts, 128 greedy tokens:

| Measure | Value | Source |
|---|---|---|
| mean acceptance length, batch 1 | 2.43 (isolated vLLM 0.27.1 on the same prompts: 2.40) | #322 |
| batch 1 decode, speculative / plain | 98.8 / about 42 tok/s | #1515, #597 |
| batch 16, `bench --ab` default against `defer_commit = false` | 787.0 / 759.5 tok/s | #1515, 27 Nov 2026 |
| full batch round at 16 sequences | 37.6 ms, 1,034 tok/s | #1515 |

The MTP computes what `qwen3_5_mtp.py` describes to float accuracy (drafts identical to a f32 torch oracle on 5 prompts, KL 1e-7); MTP precision moves position 0 acceptance by under 0.1 point, so there is no precision option. A prompt lookup drafter gained at most +0.16 tokens per step in replay against a +0.3 bar, so it was not built (#405).

## Tests

`crates/specrt/tests/mtp.rs` against `TinyMtp` in `crates/specrt/scripts/reference.py` (`reference.py tiny-mtp tests/fixtures/tiny-qwen3.5-mtp <dir>`): draft logits, drafts after rejections equal a fresh prefill, speculative greedy identical to plain greedy for k = 1 to 3 over 20 prompts with forced wrong drafts, batched accept counts, chi square tests of rejection sampling, the scheduler with preemption, per position counts, one sync per round. `sched.rs` has the fake executor test with every accept count under preemption. Real model: `real27b.rs` `real_27b_mtp_*`, `crates/specrt/scripts/spec27b.sh`.
