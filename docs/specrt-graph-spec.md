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

- `graphs/decoder.toml` is the one embedded graph. It has the head, the tail, and four blocks
  (attention, linear, attention with sparse MLP, linear with sparse MLP), each node tagged
  `when = "tensor_core"` or `"legacy"` and `phase`. `Plan::build` lowers it through `graph.rs`;
  `plan.rs` has no block-kind or op-sequence code. `tests/graph.rs` checks the lowered plan equals the
  previous hand-written plan for every fixture config in every phase and kernel tier, and greps
  `plan.rs` for architecture knowledge.
- kir gained generic light primitives: `rmsnorm_offset` (`(1+w)` norm), `add_rmsnorm` (residual add
  fused with the norm), `pointwise` (`Scale` for embedding scale, `SoftCap`). Emitted for CUDA, with
  tests in `crates/kir/tests/emit.rs`. The runtime does not launch them yet.

## Not done

- Launching the new kir primitives from graph nodes (the runtime still uses the fixed kernels).
- Sliding-window attention: the flash and decode attention kernels have no window option, and
  `Spec::validate` refuses sliding layers. The Gemma 3 proof needs that kernel option, the extra
  pre/post feed-forward norms, per-layer rope base and the embedding scale, each as a generic
  graph parameter. Waiting on the attention kernel family work (#609, #620-#623).
- MTP and vision are still driven by their own runtimes, not graph nodes.

## Prior art

TVM Relax and MLC-LLM define models in code, which we avoid. TensorRT-LLM's model builder and
llama.cpp's ggml graphs are per-architecture code. The data-only graph here is closest to a
declarative IR with primitives keyed by parameters, not names.
