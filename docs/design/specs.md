# Family specs

How a checkpoint becomes a `Spec`. The kernel side of a spec (graph nodes, `[[state]]`, kernel descriptions) is in [kir.md](kir.md); spec delivery is [../guides/specs-update.md](../guides/specs-update.md); per model notes are under [../models/](../models/).

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

## Dense families

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

## Two attention shapes per layer kind (Gemma 4)

Gemma 4 sliding layers use `head_dim` and `num_key_value_heads`, full layers use `global_head_dim` and `num_global_key_value_heads`, and with `attention_k_eq_v` the full layers have no `v_proj` (keys are the values). `family-gemma4.toml` extends `gemma3` and describes this with data only:

| Field | What it does |
|---|---|
| `[[attention.projection]] slot` | Projections naming the same `[attention.params]` width share one fused row range; each layer loads the one present in its kind and zero pads to the slot width. The qkv row is the sum of the slot widths. |
| `[[attention.projection]] when` | `"param"` or `"!param"`: the projection is loaded only when that param is non zero (or zero). k_eq_v loads `k_proj` into the v slot. |
| `[[attention.o_cols_by]]` | o_proj columns per layer kind; narrower layers are zero padded to `o_cols`. |
| `[[state]] blocks` | A per token state kept only for these layer kinds, so each kind's KV holds its own width. Kernels bind `layer = "kind_layer"` (index among layers of that kind). |
| `[norm] layer_scale` | A per layer scalar tensor that multiplies the layer output. Folded at load into the post norms as `1 / prod(earlier scalars)`, so the residual runs unscaled until the final norm, which is scale invariant. |
| `[[refuse]]` | `when` is an expression over config keys; a hit refuses the strict spec with `message` (used by Gemma 3n, Kimi K3, Haru, Whisper and Phonon). Gemma 4 has none left: MoE (26B-A4B, see [moe-offload.md](moe-offload.md)), per-layer inputs, KV sharing and the double wide MLP all serve. |
| `[[attention.projection]] kv_owner` | The projection exists only on layers that own a cache. With `[blocks] shared_tail_key` set, the last N layers have no k/v projections or norms and read the cache of the last earlier layer of the same kind. |
| `[blocks] shared_tail_key` | Names the config key holding the count of trailing layers that reuse an earlier layer's KV. `Spec::shares_kv`, `kv_source`, `kv_index` and `kind_index` map a layer to its source; KV state sizes and byte counts cover only the owners. Plan blocks come in two variants (`kv_owner` nodes run on owners, `kv_reader` nodes on consumers). |
| `[mlp] shared_tail_wide_key` | A bool config key: the shared layers run `gate_up` and `down` at twice `intermediate_size` (`Weight::GateUpWide`, `Weight::DownWide` and the `_wide` gemm shapes). Other layers keep byte identical kernels (`tests/linear_unchanged.rs`, `tests/golden`). |
| `[per_layer]` | Per-layer input embeddings: `dim` is the config key of the per layer width. `[[per_layer.node]]` is a `head` (token gather from `embed_tokens_per_layer`, the projection, norm and sum) or a `layer` (gate, gelu, slice of the per layer inputs, projection, norm and add to x) node naming a kir kernel. Reuses the `group_map` family (with `gather_row` and a layer `shift`) and `head_matvec`. `[per_layer.params]` holds its symbols; `[per_layer.scratch]` its buffers. |

The kernels are kir descriptions (`specs/g4_*.toml`): `head_rms` with an optional weight (q, k norms; v norm without), `span_rope` with an explicit `theta` and `angles` (proportional rope rotates `partial_rotary_factor * head_dim / 2` angles over the full head), `state_write`, and a geometry `scan_attention` writing the split attention planes. The vision and audio towers are ignored (`[vision] ignore`); image input gets a text-only refusal (#1237).

Tests: `tests/gemma4.rs` (31B and 12B mapping, the E2B and E4B configs mapping their shared tail, wide MLP and per-layer inputs, per kind KV bytes, every kernel builds, and on GPU1 tiny-gemma4 and tiny-gemma4-e logits, greedy and per-layer states against transformers from `scripts/tiny_gemma4.py <dir>` with `G4_FIXTURE` and `G4_STYLE=e`: max abs 4.6e-4 and 2.0e-4 on logits). The E fixture is 6 layers with per layer input 16, 2 shared KV layers and the double wide MLP; its KV bytes are smaller than a spec with no sharing.

## A prelude before chosen layers, partial refusals (Qwen3.8-Flash-Next)

`family-qwen4_exp.toml` extends `qwen3_5` and adds three tables, all data: `[gated_residual]` (four streams with a low rank gate, #1277), the QSA indexer as `[[attention.node]]` (#1278) and the PLE as a `[prelude]` (#1279). Details in [../models/qwen4exp.md](../models/qwen4exp.md).

| Field | What it does |
|---|---|
| `[prelude] module`, `layers_key`, `layers_base` | Described kernel nodes that run at the start of each layer the config list names (`ple_layer_ids`, 1 based), before the block's own nodes. Weights load from `layers.N.<module>.*`. |
| `[prelude] index`, `count` | Symbols the descriptions read: the layer's position in the list and the list length. One layer for now (kernels fold the index in). |
| `[prelude.params]`, `[prelude.scratch]`, `[prelude.int_scratch]` | Params from the config; f32 and i32 buffers of the given row width. Nodes bind `x`, `streams`, `tokens`, a scratch buffer, a weight or a state. |
| `{ int = "..." }` param | A 64 bit integer expression (`kir::int`: `prime_sum`, `splitmix64`, wrapping ops) over config keys and earlier params, for sizes like the n-gram table rows. |
| `[[state]] layers = "prelude"` | A per sequence f32 state with one slice per prelude layer; zeroed on reset, carried in snapshots, swap and session restore. |
| `[[refuse]] part = "mtp"` | A hit drops only that part (the MTP layer here) and records the message on the spec, so `[speculative] enabled = true` is refused by name while text still serves. |

The gated residual runs prefill and decode as separate steps (no fused prefill plus decode, no single row plan).

## Kimi K3

`family-kimi_k3.toml` extends `kimi_linear` for `kimi_k3` and `kimi_k31`. The reference is the checkpoint's own `modeling_kimi_linear.py` (moonshotai/Kimi-K3 at c7945ea0, sha256 pinned in `scripts/tiny_kimi_k3.py`), run with `fla` stubbed by the torch fallbacks of transformers' `kimi_linear`. The stub reproduces transformers on `tiny-kimi-linear` (prefill 5.3e-5, decode 4.0e-5) before it builds `tests/fixtures/tiny-kimi-k3` (6 layers, all six pieces).

| Piece | Family data | Engine |
|---|---|---|
| SiTU | `[activation.map] situ`, `[mlp.gated]` with `act = "situ"` (applies only when `hidden_act` names it) and the two betas from the config | the gated expression reaches the dense GEMM epilogue, the shared expert and every MoE expert kernel |
| MLA q LoRA | `q_a_proj` as an attention projection, `mla_q_lora_norm_e6`, `dsa_q_b`, `dsa_absorb_q`, `mla_lora_attention` (q rope part from `qr`) | `q_a_layernorm` and `kv_a_layernorm` use KimiRMSNorm's default eps 1e-6, not `rms_norm_eps` (`mla_lora_eps`) |
| MLA output gate | `g_proj` appended to the qkv GEMM; `mla_expand_v_rows` writes f32, `mla_out_gate` (`group_map`, new `planes` stage) stores `v * sigmoid(g)` to the attention planes | none |
| Full rank KDA gate | `g_proj` as a mixer projection, `kda_gated_norm_mix` reads the gate from the mixer input; `kda_recurrence` with `gate_lower_bound` | none |
| Latent experts | `[moe.latent]` names width, down, up, norm and the config key that turns the norm on | `Moe::latent`, `MoeShape.width` (expert kernels run at `MOE_WIDTH`, the router stays at `HIDDEN`) and `assign` combine; graph linears `latent_down`, `latent_up`, `shared_gate_up`, `shared_down` and op `latent_norm`; the shared expert runs at full width as a dense GEMM pair |
| Block attention residuals | `[block_residual]` names the block size key, the two sites and the output module | op `block_res` with stages enter, mixed, done, tail; kir row family `block_res` (mix, push, add, set, out); the score vector is `norm.weight * proj.weight` folded at load; the block stack is per row in the prefill scratch (at most 16 blocks) |

The `blockres_*` graph blocks clear `x` before each branch so `x` holds the branch output; the prefix sum lives beside it, and `done` writes the layer output back to `x`, which is what the hidden trace reads. Variants the family does not describe are refused by name: MLA without the output gate, KDA without the full rank gate, and K3.1's `dsa_config` (#1361).

Tests: `tests/kimi_k3.rs`. CPU: spec, plan validation and module build for the fixture and the real 93 layer K3 config, the K3.1 refusal. GPU1: all 6 layer outputs within 3.4e-4, prefill logits within 6.3e-5 and decode within 1.8e-5 of the largest logit, greedy tokens equal, and four red checks (silu for SiTU, the softplus gate, latent experts without their norm, MLA without the output gate) that each miss by more than 0.19.

The logit checks are relative to the largest logit (bound 1e-4, about 5e-3 absolute) rather than kimi_linear's absolute 7e-4, because the K3 fixture's logits reach about 50 and an f32 host head on the engine's own last layer output already misses the reference by about the same 3e-3, so the gap is head arithmetic on large logits; the layer-wise 3.8e-4 bound on every hidden state is the real check.
