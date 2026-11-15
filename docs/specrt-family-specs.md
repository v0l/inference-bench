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
