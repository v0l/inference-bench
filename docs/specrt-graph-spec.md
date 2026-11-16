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
- MTP, vision and linear quant are graph nodes (sections below).
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

The graph has `[[mtp]]` and `[[mtp_head]]` node lists next to `head`, `block` and `tail`, guarded by `when = "speculative+tensor_core"`:

- `[[mtp]]`: the draft layer. `mtp_in` (embed of the token, the previous hidden row, their two norms and
  the concat into planes), `linear mtp_fc`, then the norm, `linear`, attention, norm and MLP nodes of a
  full attention block, bound to the draft layer's weights (`mtp_qkv`, `mtp_o`, `mtp_gate_up`, `mtp_down`).
- `[[mtp_head]]`: `mtp_gather` (the rows that propose), `mtp_norm` with the draft norm, `linear mtp_head`
  (`quant = "draft_head"`: the draft head's quant, or the lm_head when the spec has none) and `mtp_draft`.

`Phase::Verify` is the third plan. It is the decode plan with `gdn_verify` in place of `gdn_decode`
(it logs the per-row conv and delta state that `commit` keeps up to the accepted row). Node phases are
`prefill`, `decode`, `verify` or `rows` (decode and verify). `Run.qlen` carries the attention query length
(1, or k+1 in verify and in the draft pass over the verified rows). `runtime/mtp.rs` keeps the round
orchestration (accept, commit, chain, graph capture); every kernel launch of the draft layer, the head and
the verify forward is run from these plans. `tests/graph.rs` compares the lowered plans for
`tiny-qwen3.5-mtp`, `tiny-qwen3.5-27b` and `tiny-qwen3.5-moe-mtp`.

## Linear nodes and quant

A `linear` node carries `weight` (which matrix) and `quant` (where its quantisation comes from:
`attn`, `mlp` (per layer), `bf16`, `head`, `draft_head`). `Op::Linear` resolves the quant from the spec and
calls `gemm::select` directly; there is no per-op adapter.

## Vision as graph nodes

`[[vision_head]]` (`patch_embed`), `[[vision_block]]` (norms, qkv, rope, attention, proj, mlp, and a
`deepstack_merger` guarded by `when = "deepstack"`) and `[[vision_tail]]` (`vmerger`) describe the encoder.
`Graph::vision_plan` lowers them per layer, and `VisionEncoder::run` executes the plan op by op. The splice
into the LM is the `splice` head node (`when = "vision+tensor_core"`), after `embed`.

## Fused residual plus norm: dropped, with a measurement

The GEMM residual epilogue already does the add, so the only thing a fused `add_rmsnorm` can remove is
the separate norm pass's read of `x`. nsys of the 27B NVFP4 spec with MTP (`scripts/nsys617.sh`,
`SEQS=1` and `SEQS=8`, 2048 context, 128 new tokens, 3 GB KV, desktop second card, 2026-11-16):

| | kernel time | `rms_split` | `qk_norm_rope` |
|---|---|---|---|
| b1 | 1173.6 ms | 2.5% (6075 launches) | 0.2% |
| b8 | 2049.4 ms | 1.8% (8100 launches) | 0.2% |

`rms_split` is the whole input and post-attention norm including the bf16 plane split, so the share a fusion
could save is a fraction of 2.5% (b1) and 1.8% (b8). It would also need a store-epilogue variant of `o` and
`down` on every GEMM path (mma, f8a, f4a, f8b, f4b), and the extra `add_rmsnorm` pass costs the traffic it
saves. The fusion is not built; `add_rmsnorm` stays in kir as a tested primitive.

## Prior art

TVM Relax and MLC-LLM define models in code, which we avoid. TensorRT-LLM's model builder and
llama.cpp's ggml graphs are per-architecture code. The data-only graph here is closest to a
declarative IR with primitives keyed by parameters, not names.
