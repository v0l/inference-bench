# Llama 4

Llama 4 Scout and Maverick (`model_type llama4`, `llama4_text`) run from `specs/family-llama4.toml` over `family-base.toml`, with no Llama code in Rust. Text and the tiled image tower are both served; Scout and Maverick need offload over two cards and host memory ([../generated/model-coverage.md](../generated/model-coverage.md)).

## What the family maps

| Piece | Config | Family |
|---|---|---|
| layer pattern | `no_rope_layers`: three chunked attention layers with rope, then one full NoPE layer | `[blocks] types_key`, `[blocks.layer_types]`, `[rope] nope_flags` |
| chunked local attention | `attention_chunk_size` (8192 on Scout) | `sliding_window` alias; `[[attention.node]]` kernel `attn_chunked` (`specs/attn_chunked.toml`, a `scan_attention` with a chunked mask) |
| attention temperature on NoPE layers | `attn_scale`, `floor_scale`, `attn_temperature_tuning` | `[attention.params]`, bound into `attn_chunked`: `1 + attn_scale * ln(1 + floor((pos + 1) / floor_scale))` |
| rope | interleaved pairs | `[rope] interleaved` |
| qk norm | weightless, `use_qk_norm` | `qk_norm_unit` |
| MoE | `interleave_moe_layer_step` or `moe_layers`, `num_local_experts` 16 or 128, top 1 | `[moe] router = "sigmoid"`, `input_scaled` (the router weight scales the expert input), `norm_topk = false`, `shared_gate = false`, `experts_transposed` (fused `gate_up_proj` [E, H, 2I]) |
| dense layers | `intermediate_size_mlp` (16384) against the expert width `intermediate_size` (8192) | `[mlp] dense_intermediate_key` |
| tensor names | `.feed_forward.` | `[tensors]` renames to `.mlp.` |
| vision | tiled tower with `<|tile_x_separator|>`, `<|tile_y_separator|>` and a global tile | `[vision] tiled`, `[vision.grid_tokens]`, `[vision.token_keys]` |

`[runtime] split_activations = true` keeps the hi/lo activation planes for this family.

## Tests

`crates/specrt/tests/llama4.rs` (CPU: dense and expert widths, NoPE layers every fourth, rope pairs and weightless qk norm, top 1 sigmoid routing with a scaled input, the route logits shared memory at hidden 4096, chunked attention params, the tiny checkpoint as a transposed fused expert model; GPU: `tests/fixtures/tiny-llama4` against transformers logits and greedy, references from `scripts/tiny_variants.py`), `llama4_vision.rs` and `llama4_preprocess.rs` (canvas choice and tiling against the processor, the tiny tower against transformers from `scripts/tiny_llama4_vision.py`), `llama4_preprocess_reference.py` and `llama4_real_vision_reference.py` for the real tower. The config is `tests/fixtures/llama4-scout.config.json`; the greedy gap table in [../design/kir-compiler.md](../design/kir-compiler.md) lists its prefill plan.
