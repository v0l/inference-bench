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
