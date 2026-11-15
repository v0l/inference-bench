# Graph specs

A graph spec (`crates/specrt/graphs/*.toml`, `kind = "graph"`) describes the computation of a model:
a head (embedding), one block per layer kind, and a tail (final norm and logits). Family specs
(#619) still map `config.json` to a `Spec`; the graph says what runs.

## Grammar

```toml
name = "dense"
kind = "graph"

[[head]]
op = "embed"
out = "x"

[[block]]
name = "attention"
layers = ["dense", "full", "sliding"]

[[block.node]]
op = "rmsnorm"
weight = "input_layernorm.weight"
in = "x"
out = "xn_planes"
```

- A node has `op`, optional `weight`, `in`, `out` (named buffers) and `phase` (`prefill` or `decode`;
  absent means both).
- A block applies to the layer kinds in `layers`; the planner picks the block for each layer from
  the spec's block kinds (`layer_types`, full attention interval).
- Unknown ops, buffers and weights are errors at lowering time.

## Status

- `graphs/decoder.toml` is the one embedded graph. Blocks carry `when` conditions (`plain`,
  `sandwich`, `tensor_core`, `legacy`, `logit_softcap`, joined with `+`) and `mlp = "dense" | "sparse"`.
  `Plan::build` lowers it through `graph.rs`; `plan.rs` has no block-kind or op-sequence code.
  `tests/graph.rs` checks the lowered plan equals the previous hand-written plan for every fixture
  config, and greps `plan.rs`, `runtime*` and `kir` for architecture knowledge.
- Generic parameters, all read from the family TOML and the model config: `(1+w)` norm offset
  (`[norm] unit_offset`), sandwich norms (`[norm] sandwich`, nodes `norm_add` and `clear`), embedding
  scale (`[embedding] scale`), final logit soft-cap (`logit_softcap` node), attention soft-cap, query
  scalar, sliding window per layer, and a per-layer rope base (`rope_parameters` keyed by layer type).
- kir light primitives: `rmsnorm_offset`, `add_rmsnorm`, `norm_add`, `add` and `pointwise` (scale,
  soft-cap, fill). `norm_add` and `pointwise` run from graph nodes; `add_rmsnorm` and `add` are
  emitted and tested but not launched. WGSL refuses the new bodies with a clear error (tested).
- Attention kernels (`flash_mma`, `flash_attention`, `attention_split`, grouped
  `batch_decode_attention`) take `window` and a compiled soft-cap and query scale.
- Gemma 3 runs from `family-gemma3.toml` plus the sandwich block, with no Gemma-specific Rust:
  `tests/gemma3.rs` against `scripts/tiny_gemma3.py` (sliding and global layers, both soft-caps).
  Sliding is proven by `tests/sliding.rs`.

## Not done

- Residual plus norm through the fused `add_rmsnorm` in the existing dense block (the GEMM
  residual epilogue already fuses the add; a fused kernel needs a GEMM without that epilogue).
- MTP and vision are still driven by their own runtimes, not graph nodes.

## Prior art

TVM Relax and MLC-LLM define models in code, which we avoid. TensorRT-LLM's model builder and
llama.cpp's ggml graphs are per-architecture code. The data-only graph here is closest to a
declarative IR with primitives keyed by parameters, not names.
