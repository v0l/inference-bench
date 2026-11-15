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

Where the master code decided per family (before this change, `crates/specrt/src/spec.rs`), and the
family field that now carries it.

| Decision | Master | Family field |
|---|---|---|
| allowed `model_type` list | `hf_config`, strict `ensure!` | `[match] model_types` |
| nested `text_config` takes the outer `model_type` | `hf_config` | `[match] nested_text_config` (generic) |
| qkv bias | `match hf.model_type` | `[attention] qkv_bias` |
| qk-norm | `match hf.model_type` | `[attention] qk_norm` |
| output gate | `attn_output_gate \|\| qwen3_next` | `[attention] output_gate` |
| hybrid with delta-net | `family_hybrid` list | `[blocks] kind = "hybrid"` |
| unit-offset RMSNorm | follows hybrid | follows `[blocks] kind` |
| MTP layers | hybrid and no MoE | `mtp = true` |
| MoE top-k renormalisation default | `family_hybrid` | follows `[blocks] kind` (hybrid renormalises) |
| activation map | silu, gelu variants | `[activation]` (generic) |
| accepted dtypes | bf16, f16, f32 | `[dtypes]` (generic) |
| config keys for each dimension | `HfConfig` struct | `[fields]` (generic) |
| quantisation methods | `parse_quantization` | `[quantization] methods` |
| GGUF architecture | `Gguf::spec` allowlist | `[match] gguf` |
| GGUF key names | `Gguf::spec` literals | `[gguf_fields]` |

## Schema

```toml
name = "qwen3_5"

[match]
model_types = ["qwen3_5", "qwen3_5_text"]
architectures = ["Qwen3_5ForConditionalGeneration"]
gguf = ["qwen35"]

[attention]
qkv_bias = { config = "attention_bias", default = false }
qk_norm = true
output_gate = { config = "attn_output_gate", default = false }

[blocks]
kind = "hybrid"

mtp = true


[quantization]
methods = ["compressed-tensors", "fp8"]
```

A flag is `true`, `false`, or `{ config = "<key>", default = <bool> }`. A family may set
`extends = "<other family>"`: its table is merged over the parent's before parsing, so `qwen3_moe`
only states what differs from `qwen3`.
