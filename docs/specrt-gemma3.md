# Gemma 3

Gemma 3 text runs from `specs/family-gemma3.toml` and the graph in `graphs/decoder.toml`: no Gemma code in Rust.
Both the language model and the SigLIP image path are covered here, with the commands and the last results.

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

All GPU runs go under `flock /tmp/gpu27.lock` and `systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0`.

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

Measured on the RTX PRO 6000 beside other jobs; none of these are speed numbers.

| Check | Result |
|---|---|
| 1B, 700 token prompt, last 4 rows against transformers (bf16) | KL 0.0009 to 0.0013, top-1 equal, 32 of 32 greedy tokens |
| 4B image, patch tensor | identical to the processor (max abs 0) |
| 4B image embeddings against the f32 tower | relative L2 2.3e-3, cosine 0.999997 (transformers' own bf16 tower is 8.8e-2 away from f32) |
| 4B image, logits | KL 1.7e-4 and below on the last 4 rows, 24 of 24 greedy tokens |
| 4B pan and scan (5 tiles, 1336 tokens) | patch tensor identical, embeddings 1.9e-3 from f32, teacher-forced top-1 23 of 24, the one miss is a 0.16 logit near-tie in the reference |
| Tiny SigLIP fixture, square and 4-tile | embeddings 2e-4, logits KL under 2e-5 on every row that predicts a token |
| 27B FP8 (RedHatAI) against vLLM, 9 prompts including a 3,630 token one past the window | top-1 419 of 432, mean KL 0.0037, max 0.11 |

## The image-token rows of the tiny fixture

An earlier version of the tiny test left out the rows of the image tokens, because their logits were 0.04 to 0.05
KL from the reference while their hidden states agreed. The cause was the reference, not the engine:
`Gemma3ForConditionalGeneration` in transformers 5.22 does not apply `final_logit_softcapping` (only
`Gemma3ForCausalLM` does), so its logits are uncapped. The fixture sets a cap of 30, which the text-only model
and this engine apply.

| Row group | After final norm, ours against `hidden_states[-1]` | Logits against HF, uncapped | Logits against HF with the cap applied |
|---|---|---|---|
| text rows | relative up to 6e-4 | KL 1e-6 to 2e-5 | KL below 1e-8 |
| image rows (4) | relative 7e-4 to 1.3e-3 | KL 3.7e-2 to 5.3e-2 | KL 4e-7 to 5e-6 |

The head itself is the same on every row: our kernel logits equal the head applied to our own hidden state to
5e-5. `scripts/tiny_gemma3_vision.py` now applies the cap to the reference, and the test checks every row with a
bound of 1e-4. The real 4B and 27B configs set no final soft-cap, so they never saw this.
