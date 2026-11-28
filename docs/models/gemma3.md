# Gemma 3

Gemma 3 text runs from `specs/family-gemma3.toml` and the graph in `graphs/decoder.toml`: no Gemma code in Rust.
Both the language model and the SigLIP image path are covered here, with the commands and the current results. Gemma 4 is described in [../design/specs.md](../design/specs.md).

## Text

- Five sliding layers (window 1024 on the 4B and 27B, 512 on the 1B) then one global layer. A config with
  `layer_types` states the pattern; the older 1B, 4B and 27B configs carry `sliding_window_pattern` instead, and
  the family names that key (`[blocks] sliding_period_key`).
- Local layers rotate with `rope_local_base_freq`, global layers with `rope_theta` and the linear scaling factor
  (`[rope] local_theta_key`).
- `(1 + w)` norm weights, sandwich norms around attention and the MLP, qk-norm, `query_pre_attn_scalar`, GeGLU
  (`gelu_pytorch_tanh`), embeddings scaled by `sqrt(hidden)`, tied embeddings (`[embedding] tie`). Attention and
  final logit soft-caps are read when the config sets them (Gemma 3 sets neither; Gemma 2 sets both).
- Single-stream decode runs the same sandwich norms and final soft-cap as prefill (`tests/gemma3.rs`
  `gemma3_single_stream_decode_matches_prefill`).

## Gemma 3n text

`family-gemma3n.toml` extends `gemma4` (per-layer inputs, KV sharing, described attention with no query scale) and adds a `[gated_residual]` with tagged node groups run by the `gated_sandwich` decoder block: `predict` (router, 4x4 coefficient mix, stream 0 into x), `laurel`, `laurel_add` (attention residual plus Laurel, times `1/sqrt(2)`), `act` (stored `gate_up`, row mean and std cut by `probit(p)` on layers with sparsity, gelu times up into the act planes) and `correct`. `expand` and `collapse` nodes rescale streams 1 to 3 to stream 0's magnitude. Per-layer sparsity is `[gated_residual.layer_params]`, read from the config array. The per-layer residual adds to streams 1 to 3. All kernels are kir descriptions `specs/g3n_*.toml`; no Rust names the architecture.

Tiny fixture (`scripts/tiny_gemma3n.py`, 4 layers, 2 shared KV layers, laurel rank 8, per-layer input 16, sparsity on layer 0) on GPU1: streams after every layer within 9e-5, logits 1.2e-4, greedy and two-slot serving equal. Dropping sparsity gives 6.5 and dropping Laurel 7.9 on the logits.

## Image path

The tower is the generic `pooled` kind of vision config, selected by `[vision] pooled = true` in the family:

| Step | Where |
|---|---|
| Resize each tile to 896 x 896, bilinear, no aspect change | `vision::preprocess_tiles`, `resize_bilinear` |
| Pan and scan: whole image plus up to 4 crops of wide or tall images | `vision::pan_and_scan_boxes`, option `runtime.vision.pan_and_scan` |
| 14 x 14 patch embed, learned position table, 27 blocks of layernorm, attention (no rope), MLP | `vision_encoder.rs`, graph nodes `vnorm vqkv vattention vproj vmlp_up vmlp_down` |
| Post layernorm, 4 x 4 average pool to 256 tokens, soft embedding norm `(1 + w)`, projector | graph node `vpool`, kernel `vis_pool` |
| Splice at the image tokens | `splice` head node |
| Image tokens attend to each other in both directions inside their block | kir attention constant `BIDI`, per-slot span table; the scheduler never ends a chunk inside an image block |

Pan and scan is off by default. With it on, each crop adds 256 tokens and the prompt gets the processor's
"Here is the original image ... and here are some crops" text.

## Tests and commands

GPU runs go under `~/.agents/bin/gpu-run 1 --` and `systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0` (see [../guides/testing.md](../guides/testing.md)).

| What | Command | Reference |
|---|---|---|
| Tiny text fixture, both layer types | `cargo test --release -p inference-bench-specrt --test gemma3 -- --include-ignored` | `scripts/tiny_gemma3.py <dir>` |
| Gemma 3 1B vs transformers | `GEMMA3_1B_DIR=<unsloth/gemma-3-1b-it> ... --test gemma3 gemma3_1b` | `scripts/gemma3_real.py <model> <dir>` |
| Tiny random SigLIP fixture | `... --test gemma3_tiny_vision -- --include-ignored` | `scripts/tiny_gemma3_vision.py <dir>`, which also regenerates `tests/fixtures/tiny-gemma3-vision` |
| 4B image and pan and scan vs transformers | `GEMMA3_4B_DIR=<unsloth/gemma-3-4b-it> ... --test gemma3_vision -- --include-ignored` | `scripts/gemma3_vision.py <model> <house.png> <dir>` |
| 27B FP8 config and vs vLLM | `GEMMA3_27B_DIR=<RedHatAI/gemma-3-27b-it-FP8-dynamic> ... --test gemma3_27b -- --include-ignored` | `scripts/vllm_gemma3.sh <model> <dir>` |

`vllm_gemma3.sh` starts its own vLLM on port 18639 (not the served one), refuses to start below 42 GB free,
and stops it on exit. It never touches `local-llm.service`.

## Results

Desktop RTX PRO 6000, accuracy only; none of these are speed numbers.

| Check | Result |
|---|---|
| 1B, 700 token prompt, last 4 rows against transformers (bf16) | KL 0.0009 to 0.0013, top-1 equal, 32 of 32 greedy tokens |
| 4B image, patch tensor | identical to the processor (max abs 0) |
| 4B image embeddings against the f32 tower | relative L2 2.3e-3, cosine 0.999997 (transformers' own bf16 tower is 8.8e-2 away from f32) |
| 4B image, logits | KL 1.7e-4 and below on the last 4 rows, 24 of 24 greedy tokens |
| 4B pan and scan (5 tiles, 1336 tokens) | patch tensor identical, embeddings 1.9e-3 from f32, teacher-forced top-1 23 of 24, the one miss is a 0.16 logit near-tie in the reference |
| Tiny SigLIP fixture, square and 4-tile | embeddings 2e-4, logits KL under 2e-5 on every row that predicts a token |
| 27B FP8 (RedHatAI) against vLLM, 9 prompts including a 3,630 token one past the window | top-1 419 of 432, mean KL 0.0037, max 0.11 |

The tiny vision fixture's reference applies `final_logit_softcapping` itself, because `Gemma3ForConditionalGeneration` in transformers 5.22 does not (only `Gemma3ForCausalLM` does); the engine applies the cap the config sets, and every row is checked at 1e-4. The bidirectional image block option (`BIDI`) is compiled out for models without it, so causal prefill is unchanged.
