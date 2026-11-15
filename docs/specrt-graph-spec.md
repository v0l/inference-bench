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

## MTP as graph nodes

The graph has two more node lists next to `head`, `block` and `tail`, both guarded by `when = "speculative+tensor_core"`:

- `[[mtp]]`: the draft layer. `mtp_in` (embed of the token, the previous hidden row, their two norms and
  the concat into planes), `linear mtp_fc`, then the same norm, `linear`, attention, norm and MLP nodes
  a full attention block has, bound to the draft layer's weights. Attention picks `kv_rope`+`flash` in
  the `prefill` phase and `decode_attention` in the `decode` phase, as in a block.
- `[[mtp_head]]`: `mtp_gather` (the rows that propose), a norm with `mtp.norm.weight`, `linear mtp_head`
  (the draft head, or the shared lm_head when the spec has none) and `mtp_draft` (argmax into the draft table).

A third phase, `verify`, runs the main model over `k+1` rows per sequence. It is the decode plan with
`gdn_verify` in place of `gdn_decode` (it logs the per-row conv and delta states that `commit` keeps up
to the accepted row). Node phases are `prefill`, `decode`, `verify`, or `rows` (decode and verify).
`Run` carries the attention query length (`1`, or `k+1` in verify and the chained draft).
`runtime/mtp.rs` keeps only the round orchestration (drafting loop, accept, commit, graph capture);
every kernel launch of the draft layer, the head and the verify forward comes from these nodes.
`tests/graph.rs` compares the lowered `mtp`, `mtp_head` and verify plans with a reference for every fixture spec.

## Vision as graph nodes

`[[vision_block]]` and `[[vision_head]]` follow the same pattern for the encoder (patch embed, blocks,
merger) and `splice` for the splice into the LM embeddings. See Status for what is lowered.

## Not done

- Residual plus norm through the fused `add_rmsnorm` in the existing dense block (the GEMM
  residual epilogue already fuses the add; a fused kernel needs a GEMM without that epilogue).
- MTP and vision are still driven by their own runtimes, not graph nodes.

## Prior art

TVM Relax and MLC-LLM define models in code, which we avoid. TensorRT-LLM's model builder and
llama.cpp's ggml graphs are per-architecture code. The data-only graph here is closest to a
declarative IR with primitives keyed by parameters, not names.
