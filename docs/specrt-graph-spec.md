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

Slice 1 (landed): grammar, parser (`graph.rs`), `graphs/dense.toml` for qwen2/qwen3/llama/mistral.
A test proves the graph lowers to exactly the ops `Plan::build` produces today, so the existing
kernels and golden logits are unchanged.

## Slices left

1. Light primitives generated and fused by kir from node parameters (norm plus quant, act plus
   quant, rope plus KV write, residual plus norm).
2. Heavy primitives selected by node parameters (window, softcap, sinks, GQA, head_dim, quant).
3. Hybrid delta-net, MTP, vision and MoE graphs.
4. Switch `Plan::build` to the graph and delete the fixed `Op` list and block-kind match.
5. Gemma 3 by TOML alone against a tiny transformers reference, then the grep test that keeps
   architecture knowledge out of `plan.rs`, `runtime*` and `kir`.

## Prior art

TVM Relax and MLC-LLM define models in code, which we avoid. TensorRT-LLM's model builder and
llama.cpp's ggml graphs are per-architecture code. The data-only graph here is closest to a
declarative IR with primitives keyed by parameters, not names.
