# Family specs

A family spec is a TOML file that says how one architecture family turns a `config.json` (or GGUF
metadata) into a `Spec`. The Rust mapper in `crates/specrt/src/family.rs` has no architecture names:
everything per family is data. Families ship in `crates/specrt/specs/family-*.toml`, are embedded in
the binary and are delivered by the specs updater like any other spec file.

## Layers

Families come from the same layers as model specs (your folder, then the updated cache, then the
embedded snapshot). The first family whose `[match]` fits wins. A per-model spec (a plain spec TOML
with `repo`) still layers over the result: the family builds the base `Spec`, the model file
overrides it.

## Strict and describe modes

`from_hf_config_serving` and `from_hf_config` are strict: the config must match a family, unknown
activations, dtypes, rope scalings and quantisations are errors, and the vision section is read.
`describe_hf_config` is lenient and is used for the detail view of any model, supported or not: it
uses the built-in `generic` family (no `[match]`), maps whatever it can, and falls back to SwiGLU,
no quantisation and no vision instead of failing. Structure (qk-norm, bias, output gate) is then
corrected from tensor names by `apply_structure`.

## Decision table

Every decision the Rust code used to make per family, and the family field that now carries it.
Shared defaults live in `family-base.toml`; the other families `extends = "base"`.

| Decision | Family field |
|---|---|
| allowed `model_type` / architectures / GGUF architecture | `[match] model_types`, `architectures`, `gguf` |
| qkv bias, qk-norm, output gate | `[attention]` (a bool, or `{ config = "key", default = bool }`) |
| dense, hybrid or decided by the config | `[blocks] kind` (hybrid also gives unit-offset norm and MoE top-k renormalisation) |
| `layer_types` strings to blocks | `[blocks.layer_types]`, `other_layer_type` |
| MTP layers | `mtp = true` |
| activation names | `[activation.map]`, `lenient_default` |
| accepted dtypes and the keys that state them | `[dtypes] keys`, `accepted` |
| config key aliases | `[fields] canonical = ["alias", ...]` |
| rope scaling type keys, names, parameter keys | `[rope] type_keys`, `unscaled`, `[rope.types]`, `[rope.keys]` |
| quantisation method, group layout, markers | `[quantization] method_key`, `grouped_methods`, `groups_key`, `all_linear_target`, `layer_marker`, `mlp_marker` |
| weight quant from a group or flat config | `[[quantization.group_rules]]`, `[[quantization.flat_rules]]` (first rule whose `when` paths all match wins; `defaults` fill absent paths) |
| act-quant from `input_activations` | `[[quantization.act_rules]]` |
| fused `qkv_proj` and `gate_up_proj` tensors | `[attention] fused_qkv`, `[mlp] fused_gate_up` |
| post-norm placement (no input or pre-MLP norm, post norms on the branch outputs) | `[norm] pre = false` with `sandwich = true` |
| qk-norm over the whole q and k vector instead of per head | `[attention] qk_norm_full = true` |
| config multipliers (embedding, attention scale, residual, logits divisor) | `[multipliers] embedding`, `attention`, `residual`, `logits_divisor` (each names a config key) |
| layers without rotary embedding (NoPE) | `[rope] nope_flags` (a config key holding 0/1 per layer) |
| GGUF metadata keys and tensor names | `[gguf.keys]` (`{arch}` is replaced), `[gguf.tensors]` |
| GGUF rope scaling and pre-tokenizers | `[gguf.rope_scaling]`, `unscaled`, `[gguf.pretokenizers]` |

What stays in Rust is the schema: the `Quant`, `ActQuant`, `Activation` and `RopeScalingKind` enums,
the arithmetic of each scaling kind, and the layout rules (one MLP quant for the first layers, the
projection quant after them). A new architecture whose config uses these pieces needs only a TOML.

## Strict and describe modes

Strict (`from_hf_config*`) needs a matching family and refuses unknown activations, dtypes, rope
types and quantisations. Describe (`describe_hf_config`) uses the `generic` family, falls back to
its `lenient_default` activation, and ignores quantisation and rope errors.

## Example

```toml
name = "acme"
kind = "family"
extends = "base"

[match]
model_types = ["acmelm"]
gguf = ["acmelm"]

[attention]
qk_norm = true

[activation.map]
swish_glu = "swiglu"

[fields]
hidden_size = ["n_embd"]
```

## Dense families (#638)

Mistral Small 3 (`mistral`, sliding window off), Phi-4 and Phi-3 (`phi3`), SmolLM3 (`smollm3`),
OLMo 2 (`olmo2`) and Granite 3 (`granite`) run from a family file each, with no per-architecture
Rust. Mistral needed no change: `family-llama.toml` already matches it and `sliding_window = null`
leaves the window off. The generic additions were:

| Field | What it does |
|---|---|
| `[attention] fused_qkv`, `[mlp] fused_gate_up` | The loader reads one fused tensor and slices it into the q, k, v and gate, up rows. Partial rotary already came from `partial_rotary_factor`. |
| `[norm] pre = false` (with `sandwich = true`) | The graph drops the input and pre-MLP norms (a `split` node copies `x` into the GEMM planes) and norms each branch before the add, which is OLMo 2's post-norm placement. |
| `[attention] qk_norm_full` | A kir `rmsnorm_segment` kernel normalises the q and k slices of each qkv row over the full width, before rope. The attention kernels skip their per-head norm. |
| `[multipliers]` | Each entry names a config key. `embedding` sets `embed_scale`, `attention` becomes the query scale, `residual` scales each branch before the add (kir `add_scaled`, node `add_scaled`), `logits_divisor` becomes a `logit_scale` node after the head. |
| `[rope.types] longrope` | Phi-4-mini's `longrope`: per-frequency short and long factors (chosen by `max_ctx` against `original_max_position_embeddings`) divide the inverse frequencies (rope kind 4, kir `RopeShape.freq_factors`) and an attention factor scales cos and sin. Not on wgpu. |
| `[rope] nope_flags` | Layers whose flag is 0 skip rope. kir attention shapes carry a per-layer `nope` table and both the rope-and-cache-write and the flash kernels test it. |

Tests: `tests/families.rs` (spec mapping, plan shape, and GPU logits, greedy and served greedy
against transformers on `tests/fixtures/tiny-{mistral,phi3,smollm3,olmo2,granite}` built by
`scripts/tiny_families.py`, plus real checkpoints against `scripts/families_real.py`). Branch
families (`uses_branch`) turn off the fused prefill-plus-decode step, which only runs the plain layer.

Real checks: Phi-4-mini (partial rotary 0.75 plus longrope), Phi-3-mini, OLMo-2-1B, Granite-3.3-2B, SmolLM3-3B, and Mistral-Small-24B-Instruct-2501 FP8 (`RedHatAI/...-FP8-dynamic`, all 40 layers) against transformers.

## Two attention shapes per layer kind (#1194, Gemma 4)

Gemma 4 sliding layers use `head_dim` and `num_key_value_heads`, full layers use `global_head_dim` and `num_global_key_value_heads`, and with `attention_k_eq_v` the full layers have no `v_proj` (keys are the values). `family-gemma4.toml` extends `gemma3` and describes this with data only:

| Field | What it does |
|---|---|
| `[[attention.projection]] slot` | Projections naming the same `[attention.params]` width share one fused row range; each layer loads the one present in its kind and zero pads to the slot width. The qkv row is the sum of the slot widths. |
| `[[attention.projection]] when` | `"param"` or `"!param"`: the projection is loaded only when that param is non zero (or zero). k_eq_v loads `k_proj` into the v slot. |
| `[[attention.o_cols_by]]` | o_proj columns per layer kind; narrower layers are zero padded to `o_cols`. |
| `[[state]] blocks` | A per token state kept only for these layer kinds, so each kind's KV holds its own width. Kernels bind `layer = "kind_layer"` (index among layers of that kind). |
| `[norm] layer_scale` | A per layer scalar tensor that multiplies the layer output. Folded at load into the post norms as `1 / prod(earlier scalars)`, so the residual runs unscaled until the final norm, which is scale invariant. |
| `[[refuse]]` | `when` is an expression over config keys; a hit refuses the strict spec with `message`. Gemma 4 still uses it for the MoE blocks (#1238); per-layer inputs, KV sharing and the double wide MLP are served (#1239). |
| `[[attention.projection]] kv_owner` | The projection exists only on layers that own a cache. With `[blocks] shared_tail_key` set, the last N layers have no k/v projections or norms and read the cache of the last earlier layer of the same kind. |
| `[blocks] shared_tail_key` | Names the config key holding the count of trailing layers that reuse an earlier layer's KV. `Spec::shares_kv`, `kv_source`, `kv_index` and `kind_index` map a layer to its source; KV state sizes and byte counts cover only the owners. Plan blocks come in two variants (`kv_owner` nodes run on owners, `kv_reader` nodes on consumers). |
| `[mlp] shared_tail_wide_key` | A bool config key: the shared layers run `gate_up` and `down` at twice `intermediate_size` (`Weight::GateUpWide`, `Weight::DownWide` and the `_wide` gemm shapes). Other layers keep byte identical kernels (`tests/linear_unchanged.rs`, `tests/golden`). |
| `[per_layer]` | Per-layer input embeddings: `dim` is the config key of the per layer width. `[[per_layer.node]]` is a `head` (token gather from `embed_tokens_per_layer`, the projection, norm and sum) or a `layer` (gate, gelu, slice of the per layer inputs, projection, norm and add to x) node naming a kir kernel. Reuses the `group_map` family (with `gather_row` and a layer `shift`) and `head_matvec`. `[per_layer.params]` holds its symbols; `[per_layer.scratch]` its buffers. |

The kernels are kir descriptions (`specs/g4_*.toml`): `head_rms` with an optional weight (q, k norms; v norm without), `span_rope` with an explicit `theta` and `angles` (proportional rope rotates `partial_rotary_factor * head_dim / 2` angles over the full head), `state_write`, and a geometry `scan_attention` writing the split attention planes. The vision and audio towers are ignored (`[vision] ignore`); image input gets a text-only refusal (#1237).

Tests: `tests/gemma4.rs` (31B and 12B mapping, the E2B and E4B configs mapping their shared tail, wide MLP and per-layer inputs, per kind KV bytes, every kernel builds, and on GPU1 tiny-gemma4 and tiny-gemma4-e logits, greedy and per-layer states against transformers from `scripts/tiny_gemma4.py <dir>` with `G4_FIXTURE` and `G4_STYLE=e`: max abs 4.6e-4 and 2.0e-4 on logits). The E fixture is 6 layers with per layer input 16, 2 shared KV layers and the double wide MLP; its KV bytes are smaller than a spec with no sharing.

## A prelude before chosen layers, partial refusals (#1280, Qwen3.8-Flash-Next)

`family-qwen4_exp.toml` extends `qwen3_5` and adds three tables, all data: `[gated_residual]` (four streams with a low rank gate, #1277), the QSA indexer as `[[attention.node]]` (#1278) and the PLE as a `[prelude]` (#1279). Details in [specrt-qwen4exp.md](specrt-qwen4exp.md).

| Field | What it does |
|---|---|
| `[prelude] module`, `layers_key`, `layers_base` | Described kernel nodes that run at the start of each layer the config list names (`ple_layer_ids`, 1 based), before the block's own nodes. Weights load from `layers.N.<module>.*`. |
| `[prelude] index`, `count` | Symbols the descriptions read: the layer's position in the list and the list length. One layer for now (kernels fold the index in). |
| `[prelude.params]`, `[prelude.scratch]`, `[prelude.int_scratch]` | Params from the config; f32 and i32 buffers of the given row width. Nodes bind `x`, `streams`, `tokens`, a scratch buffer, a weight or a state. |
| `{ int = "..." }` param | A 64 bit integer expression (`kir::int`: `prime_sum`, `splitmix64`, wrapping ops) over config keys and earlier params, for sizes like the n-gram table rows. |
| `[[state]] layers = "prelude"` | A per sequence f32 state with one slice per prelude layer; zeroed on reset, carried in snapshots, swap and session restore. |
| `[[refuse]] part = "mtp"` | A hit drops only that part (the MTP layer here) and records the message on the spec, so `[speculative] enabled = true` is refused by name while text still serves. |

The gated residual runs prefill and decode as separate steps (no fused prefill plus decode, no single row plan).
