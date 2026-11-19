# kir as data: kernel descriptions and the runtime spec (#802)

Goal: a new architecture that is a variant of what we run ships as spec files only, delivered by the spec auto-updater, with no new binary. New Rust is needed only for a new hardware primitive (a new mma kind, a new memory op).

Today the graph is data (`graphs/decoder.toml`, family specs, #645) but every kernel shape is a Rust builder in `crates/kir`. The family specs can select and parameterise kernels the binary has. This doc says how to close that gap and in what order.

## 1. Gap analysis

Legend: **spec** = expressible with graph TOML plus the current kir builders; **desc** = needs a kernel description (section 2); **prim** = needs a new hardware primitive; **state** = needs a new state kind (section 3).

| Architecture | Piece | Today | Missing |
|---|---|---|---|
| DeepSeek V2/V3 | MLA projections (q_a, q_b, kv_a, kv_b) | `linear` nodes work | spec: nothing |
| | Decoupled rope (rope on a slice of the head) | `partial_rotary` exists for the rope kernel | desc: rope over a head slice with a separate rope-only K stream |
| | Absorbed-weight decode: q projected into latent space, attention over the latent, output expanded | none | desc: attention whose K and V are the same latent buffer (K = latent plus rope slice, V = latent), head_dim 576 for QK and 512 for V. Needs dim-split QK vs V in the attention description |
| | Latent KV cache | KV state is `[layers][planes][kv_heads][head_dim]` | state: a cache kind with its own row width (`kv_lora_rank + rope_dim`), one plane, no head axis |
| | Sigmoid group-limited MoE routing | `MoeShape.sigmoid` exists, no groups | desc: group top-k then top-k within kept groups, routed scaling, bias-corrected selection |
| | Dense first layers, shared experts | graph `mlp = dense/sparse` per layer, shared expert exists | spec: layer ranges for the `dense` block (`first_k_dense_replace`) |
| DeepSeek V4 | Compressed sparse attention (CSA/HCA): KV compressed by a ratio, sparse top-k over compressed blocks, sliding window, hash layers | none | desc: block-mean/learned compress as a pooled write kernel, top-k select over compressed keys, gather attention. state: compressed KV cache. spec: per-layer compress ratio, hash-routed layers (token id hashed to experts, a gather by a lookup table) |
| GLM-5.x (`glm5_next`) | KDA linear attention | delta-net family (`linear.rs`) is a fixed Rust family with fixed gating | desc: delta rule with per-channel gate (KDA) as a scan recurrence description. state: scan state exists, but its shape and snapshot rule come from the description |
| | DSA top-k indexer with a shared indexer | none | desc: an index-score kernel (small mma plus relu plus weighted sum), top-k, then gather attention. state: an index buffer written by one layer and read by later layers (sharing) |
| | MLA latents | as DeepSeek | as DeepSeek |
| gpt-oss | Attention sinks | attention kernels take window, soft-cap, scale; no sink | desc: a per-head learned logit joining the softmax denominator (one extra term in the online softmax init) |
| | Alternating sliding and full | spec (`layer_types`, `window`) | none |
| | MXFP4 experts | NVFP4 (e2m1, e4m3 scale per 16) only | desc: dequant block with e8m0 scale per 32 elements; no new mma (decode to bf16, or block-scaled mma where the target has `BlockScaledFp4`). prim only if mxf4 native mma differs from nvf4 |
| | Clamped SwiGLU with alpha | `GatedActivation` has silu and gelu | desc: epilogue expression language (clamp, alpha, limit) |
| Mamba2 SSD | Conv1d, in-proj, gated norm | conv exists in delta-net | desc: reuse |
| | Chunk scan (prefill), step (decode) | none | desc: chunked scan with intra-chunk mma and an inter-chunk recurrence. state: scan state `[heads][head_dim][d_state]`, plus conv state |
| | Hybrid layer pattern | spec (`layer_types`) | none |
| Llama 4 | Chunked local attention | `window` sliding only | desc: a mask kind `chunked(n)` next to `window(n)` |
| | NoPE layers | rope is per layer from `rope_parameters` | spec: a layer kind with rope disabled |
| | Attention temperature tuning on long context | none | desc: query scale as an expression of position |
| Sigmoid group-limited MoE | | see DeepSeek | one routing description serves DeepSeek V3, Kimi K2, GLM-4.5 |

The recurring needs are therefore: (1) an epilogue/pointwise expression language, (2) attention described by mask, score modifiers, split QK/V dims and cache layout, (3) scan and recurrence kernels, (4) top-k and gather/scatter, (5) dequant blocks as data, (6) state kinds declared in the spec.

## 2. Kernel descriptions

### 2.1 Format

A kernel description is a TOML file, `kind = "kir-kernel"`, shipped like a spec (embedded copy for the offline default, auto-updated copy wins). One file describes one kernel family instance, parametric over shape symbols.

```toml
kind = "kir-kernel"
family = "rmsnorm"        # which kir lowering to use
name = "rmsnorm"          # emitted symbol
threads = 1024

[args]
width = "hidden"          # a string that names a shape symbol, or a literal
eps = "norm_eps"
offset = 0
```

Rules:

- `args` values are literals or a **shape symbol**. Symbols come from the `Shape` the runtime builds from the model config and spec (`specrt::codegen::kir_shape`): `hidden`, `inter`, `vocab`, `norm_eps`, `q_dim`, `kv_dim`, `qkv_row`, `mlp_act`, `residual_scale`, `logit_scale`, `logit_softcap`. New symbols are added by new spec fields, not new kernels.
- Word args (`op`, `act`) are literals or a symbol that resolves to a word.
- Unknown keys, an unknown family, a missing arg, a symbol the shape lacks, or a non-size value are errors with the kernel name in the message. This is checked when the model loads, before any emit.
- After the description is resolved to a `Kernel`, the existing `Kernel::validate` runs against the live `Target` (threads, shared memory, required features, language). Nothing is emitted for a target that cannot run it.
- The hash is the hash of the resolved kernel, so the JIT cache key tracks the resolved shape and target, not the file text.

### 2.2 Two levels

The format grows in two levels, and the proof slices cross from one to the next.

**Level 1: parametric families (slice a).** `family` names a Rust lowering that already exists. The description chooses and parameterises it. This moves the *decision* of which kernels a model uses and with what arguments out of Rust (`codegen::kir_kernels` was a hand-written list) into data. The Rust builders are the reference implementations: tests assert `description.build(...) == Kernel::<builder>(...)` for every target, so emitted source and goldens are identical by construction.

**Level 2: composed bodies (slices b onwards).** `family = "compose"` and the body is a list of stages over kir primitives, so a new kernel needs no Rust family:

```toml
kind = "kir-kernel"
family = "compose"
name = "attn_decode"
threads = 128

[grid]            # one block per (head, split)
x = "heads"
y = "splits"

[[stage]]
op = "tile_load"  # global -> shared, layout is a CuTe style (shape):(stride)
src = "k_cache"
layout = "(page, kv_heads, head_dim):(kv_heads*head_dim, head_dim, 1)"
stages = 2
async = true      # cp.async where Feature::CpAsync, plain loads otherwise

[[stage]]
op = "mma"
kind = "m16n8k16_bf16"   # must be in the Target feature set
a = "q"
b = "k_tile"
acc = "s"

[[stage]]
op = "epilogue"
expr = "s * scale"       # small expression language, section 2.3
mask = "window(window)"  # window(n), chunked(n), causal

[[stage]]
op = "online_softmax"
state = "m,l,o"
sink = "sinks"           # optional per-head logit joining the denominator
```

Primitive vocabulary (each entry lowers per dialect in Rust once, and a new hardware feature adds one entry):

| Group | Primitives |
|---|---|
| Memory | `tile_load` / `tile_store` with a `Layout`, `gather` and `scatter` by an index buffer, `lookup` (table by token id) |
| Compute | `mma` (kind from the `Target` feature list), `dot` (scalar and warp), `epilogue` (expressions) |
| Reduce | `block_sum`, `block_max`, `online_softmax`, `topk(k)` with optional groups, `scan(op)` over a chunk and across chunks |
| Quant | `dequant` blocks as data: `{ elem = "e2m1", scale = "e8m0", group = 32 }` |
| Pipeline | `stages`, `async`, `split_k`, `split_kv` plus a combine kernel |

### 2.3 Expressions

Elementwise maths (soft-cap, scale, clamp, activations, alpha, gating, query temperature as a function of position) are written in a small expression language: arithmetic, `tanh`, `exp`, `sigmoid`, `min`, `max`, `select`, shape symbols and named stage values. It lowers to both CUDA and WGSL text. `Pointwise::{Scale, SoftCap, Fill}` and `Activation::{Silu, GeluTanh}` become expression instances; their goldens pin the lowering.

### 2.4 Golden safety

The Rust builders stay as the reference implementations until every emitted kernel is covered by a description and the byte-identity test. A description is never trusted by text: the test builds both, compares the `Kernel` values, the describe string, the hash and, for emitted kernels, the CUDA and WGSL source. The existing golden files in `crates/kir/tests/golden/` do not change in any slice. A slice that has to change a golden is not allowed to land; it is split.

## 3. Runtime: op kinds and state kinds in the spec

Today `plan::Op` is a Rust enum (`Flash`, `GdnDecode`, `MoeRoute`, ...) and state is a fixed set (KV pages, delta state, conv state). The plan and the executor match on those.

Target design:

- **Op kinds:** a graph node `op` is either a built-in name or a reference to a kernel description (`op = "kernel"`, `kernel = "attn_decode"`). The node declares its buffer bindings (`in`, `out`, `state`, `weight`), launch grid symbols and the phase. The executor launches a described kernel from the compiled module by name with arguments resolved from the bindings. The built-in ops are migrated one at a time to descriptions so the enum shrinks to the genuinely host-side ones (embed splice, sampling, MTP orchestration).
- **State kinds:** a `[[state]]` table in the family spec declares a buffer: `name`, `per = "layer" | "sequence" | "token"`, `shape` as shape symbols, `dtype`, whether it is snapshotted, whether prefix caching may share it, and the rule for rollback (`positional` for KV, `checkpoint` for scan state). KV cache, latent KV, compressed KV, scan state and index buffers are all instances. The state pool, snapshot store (`sessions.rs`), prefix cache and speculative rollback already handle two kinds (positional KV, checkpointed recurrent); the declaration selects between those two behaviours and sizes the allocation from the shape. Sharing across layers (GLM shared indexer) is `owner = "layer:N"` or `shared = true`.
- **Validation:** unknown op kinds, undeclared buffers, a state whose shape symbol is missing and a kernel the target cannot run are load-time errors, as for graph specs today.
- **Memory planning:** `anatomy.rs` and the fit estimate read the `[[state]]` shapes instead of KV formulas, so a latent cache is counted correctly in the library fit check.

## 4. Slices

Each slice is its own issue under goal g4 (backlog, no goal set). The order is a dependency chain, with (a) done in this issue.

| Slice | Content | Proof |
|---|---|---|
| a | Level 1 descriptions: `kir::desc`, the embedded `kernels/*.toml` for rmsnorm and its light family, `codegen::kir_kernels` driven by them. | goldens byte-identical, description equals builder for every target. **Done in #802.** |
| a2 | Decode attention and the rope/KV-write family as level 1 descriptions (`attention.rs` families `SplitDecode`, `GroupedDecodeMma`, `RopeKvWrite`). Auto-update delivery of `kind = "kir-kernel"` files next to family specs. | attention goldens byte-identical |
| a3 | Expression language (section 2.3) and `compose` with `epilogue`, `block_sum`, `tile_load`, `tile_store`: express rmsnorm, norm_add, pointwise and swiglu as composed bodies, builders become the reference. | goldens byte-identical |
| b | New attention variant from spec only: sinks, chunked mask, soft-cap plus window, as attention bodies. Runtime op kind `kernel`. | tiny fixture against transformers, no Rust change. **Done in #806.** |
| c | State kinds from the spec (section 3), split QK/V dims, MLA absorbed decode and latent cache as spec only. Absorbs #641. | DeepSeek-V2-Lite tiny fixture against transformers |
| d | Scan and recurrence primitives, state checkpoint rule: Mamba2 SSD, KDA. Absorbs #643. | tiny Mamba2 hybrid fixture |
| e | `topk` with groups, `gather`/`scatter`, `lookup`, dequant blocks as data (MXFP4): group-limited sigmoid MoE, gpt-oss, hash layers, DSA indexer. Absorbs #640 and #642. | tiny gpt-oss and GLM MoE fixtures |
| f | Allowlist test: per-architecture names appear nowhere in Rust. | grep test in CI |

Issues: a2 #804, a3 #805, b #806, c #807, d #808, e #809, f #810.

Fixtures and GPU runs follow `docs/testing.md`: CPU and tiny tiers first, the Spark for the real models, never the desktop GPUs while the user's app is running.

## 5. Status of slice (a)

- `crates/kir/src/desc.rs`: `Description`, `Shape`, `build_builtin`. Families: `rmsnorm`, `add_rmsnorm`, `norm_add`, `rmsnorm_segment`, `add`, `pointwise`, `gated_activation`.
- `crates/kir/kernels/*.toml`: ten descriptions embedded in the binary.
- `specrt::codegen::kir_kernels` now selects names by spec conditions and builds every kernel from its description through `kir_shape`.
- `crates/kir/tests/desc.rs`: description equals builder (kernel value and hash) on sm_120, sm_80, wgpu and wgpu with subgroups; error cases; target validation still applies.
- Emitted CUDA and WGSL are unchanged: the kir goldens and the specrt lib tests pass untouched.

## 6. Status of slice (a3)

- `crates/kir/src/expr.rs`: the expression language (arithmetic, comparisons, `tanh exp log rsqrt sqrt abs min max select`, macros `sigmoid silu gelu_tanh`, shape constants, stage names) lowered to CUDA and WGSL text.
- `crates/kir/src/compose.rs`: `Body::Compose`, stages `tile_load`, `epilogue`, `block_sum`, `tile_store`, emitted for CUDA and WGSL. `family = "compose"` descriptions take `[rows]`, `[[param]]`, `[[const]]` and `[[stage]]` tables; `${word}` in an expr takes an `args` word such as `mlp_act`.
- `crates/kir/kernels/compose_*.toml`: rmsnorm, norm_add, add_rmsnorm, swiglu, residual_add, logit_scale, logit_softcap, clear_branch as composed bodies.
- `crates/kir/tests/compose.rs`: emitted source equals the Rust builders byte for byte. rmsnorm and swiglu on sm_120, sm_80, wgpu and wgpu with subgroups (swiglu for silu and gelu); norm_add, add_rmsnorm and the pointwise kernels on sm_120 and sm_80 (the builders have no WGSL emitter). A new clamped swiglu is built from TOML alone and validated with naga. Goldens are unchanged.

## 7. Status of slice (a2)

- `crates/kir/kernels/{qk_norm_rope,kv_norm_rope,attention_split,attention_reduce,flash_attention,flash_mma,batch_decode_attention,batch_decode_combine}.toml`: all eight attention kernels of the five `Family` variants (RopeKvWrite, SplitDecode, FlashScalar, FlashMma, GroupedDecodeMma) as level 1 descriptions. Each carries a `[shape]` table whose values are shape symbols (`heads`, `kv_heads`, `head_dim`, `qkv_row`, the rope constants, `kv_planes`, `nope`, `local_rope`, mrope) or literals.
- `crates/kir/src/desc.rs`: `Description::build_attention` resolves `[shape]` into a `kir::attention::AttentionShape` and builds through `AttentionKernel::new`, so validation, threads and smem still come from the family builder; `kir::desc::ATTENTION` lists the eight embedded descriptions.
- `specrt::codegen::attention_kernels` now builds all eight from descriptions through `attention_symbols`, which turns the spec, kv planes and max context into the symbol table. FlashScalar and FlashMma are included so the legacy plan path keeps `flash_attention`.
- Delivery: the eight files ship in `crates/specrt/specs/` (embedded copy, auto-updated copy wins). `Spec::check` returns `Checked::Kernel(name)` for them, `family::is_kernel` identifies them, `crates/core::specs::install_families` also installs a `specrt::kir_kernel::Registry` from the spec layers (first text for a name wins, layers in priority order), and the registry falls back to the embedded set.
- `crates/kir/tests/desc.rs` and `crates/specrt/tests/kernel_desc.rs`: description equals builder per target (sm_120, sm_80) for every kernel, error cases named, and the delivered files parse. The attention goldens in `crates/kir/tests/golden/attention` are byte-identical.

## 8. Status of slice (b)

Plumbing (commit "Scan attention bodies and kernel graph nodes"), generic and variant free:

- `kir::expr`: `==`, `!=`, `&&`, `||` and `floor`.
- `kir::Scalar` gains `Bf16` and `I32`, `kir::Role` gains `Value` (a launch scalar), and `param` tables take `type`.
- `kir::scan` and `family = "scan_attention"`: a query-row scan over the paged KV cache. Stages `epilogue`, `mask` and `online_softmax` (optional `sink`). Built-in names in expressions: `s` (raw dot), `scale`, `q_pos`, `k_pos`, `head`, `kv_head`, plus every `value` param. CUDA only. A mask is an expression, so `window(n)` and `chunked(n)` are one line each.
- `[[attention.node]]` and `[attention.params]` in a model spec or family spec (`SPEC_FORMAT` 9). Nodes are `qk_rope`, `decode_rope` and `kernel`; a `kernel` node names a description, a launch `grid` (`heads`, `kv_heads`, `rows` or numbers) and `bind`s each extra param to a weight (`{ weight, len }`), a number, `window`, `nope` or an `[attention.params]` entry. `[attention.params]` values come from `config.json` keys in a family spec.
- `graphs/decoder.toml`: `attention_nodes` expands to the spec nodes, and the stock attention ops carry `when = "stock"`. A spec with no nodes lowers exactly as before.
- specrt: `Op::Kernel`, `Op::DecodeRope`, `codegen::described_kernels`, per-layer extra weights (`self_attn.<weight>`), the executor launch of a described kernel from its bindings. Load-time errors name the node and kernel.

The variant (no Rust): `specs/attn_sinks.toml` (per-head sink in the softmax denominator, window mask, soft-cap), `specs/attn_chunked.toml` (chunked mask, NoPE temperature `1 + attn_scale * ln(1 + floor((pos + 1) / floor_scale))`), and the family files `specs/family-sinks-qwen3.toml` and `specs/family-chunked-smollm3.toml`.

Proof (`tests/variants.rs`, fixtures `tiny-sinks` and `tiny-chunked`, reference `scripts/tiny_variants.py`, transformers with the eager attention swapped for the gpt-oss sink softmax and the Llama 4 chunked mask with temperature tuning), Spark GB10, release:

```text
tiny-chunked prefill max_abs 0.00037956238, greedy decode tokens equal
tiny-sinks   prefill max_abs 0.00030136108, greedy decode tokens equal
```

Not covered: WGSL for scan bodies, the single stream `step` path (a spec with nodes runs through prefill, `generate_batch` and the scheduler), MTP together with nodes, and tensor core (mma) scan bodies. The scan is a scalar warp kernel, so these descriptions are correct, not fast.

## 9. Status of slice (c)

Plumbing (commits "Spec-declared state tables" and "State-bound scan, row prims, split QK/V dims"), variant free:

- `specrt::state`: `[[state]]` tables (`name`, `per = token | sequence | layer`, `layers = attention | linear`, `shape` as size expressions over model symbols and `[attention.params]`, `dtype`, `snapshot`, `prefix_share`, `rollback = positional | checkpoint`, `owner = "layer:N"`). `family-base.toml` declares the stock `k`, `v`, `conv` and `ssm` states, so stock specs serialise as before and every golden is unchanged. The runtime allocates one paged cache per per-token state; copy, park, resume and session save/load walk the declared caches. `kv_block_bytes`, the anatomy fit, `slot_state_bytes` and the placement cost model read the state shapes.
- `kir::scan` takes an optional `[geometry]` with `[[query]]` segments: K and V come from named state bindings with their own row width, value offset and width (split QK and V head dims), and the result goes to an f32 buffer. Stock scan emission is unchanged.
- `kir::rows`: `span_rope` (rope over a span of each head, `style = "halves" | "pairs"`), `state_write` (rows into a paged state) and `head_matvec` (`y[h] = W[h] · x[h]` with the weight addressed by head, inner and out strides plus an offset).
- Node bindings `{ buffer = "qkv" | "attn" | <scratch> }` and `{ state = <name> }`; `[attention.scratch]` rows; `[[attention.projection]]` for the fused input linear and `o_cols` for the output linear; `[attention.params]` entries can be `{ expr = "..." }` over config keys (dotted paths) and other params. `SPEC_FORMAT` 10.

The variant (no Rust): `specs/family-deepseek_v2.toml` and seven `specs/mla_*.toml` descriptions. The q and kv_a projections are one linear, then latent rmsnorm (`rmsnorm_segment`), pair rope on q and k rope slices, the latent write, W_UK absorption of q_nope, scan attention over the 1 x (kv_lora_rank + rope) latent (K = whole row, V = first kv_lora_rank), W_UV expansion into the attention planes, o_proj. The softmax scale includes the YaRN `mscale_all_dim` correction as a param expression.

Proof (`tests/variants.rs`, fixture `tiny-deepseek-v2`, reference `scripts/tiny_variants.py`, transformers `DeepseekV2ForCausalLM` with random weights), Spark GB10, release:

```text
tiny-deepseek-v2 prefill max_abs 0.00039076805, greedy decode tokens equal
```

Not covered: the fixture's MLP is dense, because V2 MoE (two shared experts wider than the routed ones) does not load yet (#830); the real DeepSeek-V2-Lite is about 31 GB in bf16, over the Spark budget. The MLA kernels are scalar, correct rather than fast.

## 10. Status of slice (e)

Done, variant free:

- **Group-limited sigmoid routing with score bias.** `kir::ops::MoeShape` gains `groups`, `topk_group` and `score_bias`. `moe_route` selects on `sigmoid + e_score_correction_bias`, keeps the best `topk_group` groups (a group scores the sum of its two best), takes the top-k inside them, and weights by the unbiased sigmoid, with `norm_topk` and `routed_scale`. With `groups = 0` the kernel computes exactly what it did before. The family spec declares `[moe] router = "sigmoid"` and `score_bias = true`; `n_group` and `topk_group` come from `config.json`; the bias tensor is `mlp.gate.e_score_correction_bias`. `specs/family-glm4_moe.toml` is the first user, so GLM-4.5/4.6/Air load from a family file (absorbs the GLM half of #642).
- **Gated activation as an expression.** `[mlp.gated]` in a family spec holds an `expr` over `g` (gate), `u` (up) and `params` read from `config.json`. It lowers into one `gated_act(g, u)` device function used by the dense, batched, MoE gemv and MoE mma kernels. A spec without it emits `activation(g) * u`, so stock kernels are unchanged. `tests/fixtures/family-clamped.toml` expresses the gpt-oss clamp and alpha SwiGLU.
- **Dequant blocks as data.** `specrt::dequant::BlockFormat { elem, scale, group }` (`e2m1 | e4m3 | int8` elements, `e8m0 | e4m3` scales). `MXFP4` is `e2m1`, `e8m0`, 32. `[moe] expert_block` plus `gate_up_interleaved` in a family spec make the loader decode `gate_up_proj_blocks/_scales` and `down_proj_blocks/_scales` into the bf16 expert stacks. gpt-oss rows are interleaved gate/up; the flag picks the order. Decode is exact against the e2m1 table and power-of-two scale.

Proof (Spark GB10, release, `tests/variants.rs`, fixture `tiny-glm4-moe`, reference `scripts/tiny_variants.py` with transformers `Glm4MoeForCausalLM`: 8 experts, 2 groups, topk_group 1, bias, scale 1.5, norm_topk, partial rotary, qk-norm, one shared expert):

```text
group_limited_sigmoid_routing_with_score_bias_matches_transformers ... ok
```

Second pass (same slice):

- **Biases.** `[attention] o_bias`, `[moe] router_bias` and `expert_bias` (router module name via `router_module`). The expert biases enter the gemv and grouped kernels before the gated expression; the router bias is added to the logits in `moe_route_logits`. The mma path is NVFP4 only and does not take expert biases.
- **Hash layers.** `[moe] hash_table = "tid2eid"` with `num_hash_layers` from `config.json`: the first N sparse layers read experts from an `[vocab, top_k]` table indexed by the token id (`lookup`), and weight them by the router scores. `MoeShape.hash` selects the code; the table is a per layer checkpoint tensor. `tests/moe_route.rs` runs the real kernel on the GPU against a host reference.
- **gather, scatter, lookup stages** in `family = "compose"`: `index` names a read i32 param; rows are addressed by the index buffer instead of the block id. CUDA only, errors name the param. `crates/kir/tests/compose.rs`.
- **DSA indexer.** `family = "index_topk"` (`specs/dsa_index_topk.toml`): relu index scores weighted per head, then top-k per query row into an i32 buffer (`-1` for unused slots). A later layer shares the selection by binding the same buffer. `tests/index_topk.rs` checks the order against a host reference. Wiring it in front of the MLA scan (indexer projections, key cache, masked scan) is the part of GLM-5 left.
- **tiny gpt-oss** (`tests/fixtures/tiny-gpt-oss`, `specs/family-gpt_oss.toml`): sinks with sliding window, biased attention, router bias, top-2 softmax weights, clamped SwiGLU with alpha, MXFP4 experts. Transformers `GptOssForCausalLM` is the reference; its experts are dequantised from the same blocks.

```text
gpt_oss_sinks_clamped_swiglu_and_mxfp4_experts_match_transformers ... ok   (Spark GB10)
```

`plan.rs` now asks the spec for the expert tensors (`Spec::expert_tensors`), so `plan_rs_has_no_architecture_or_op_sequence_knowledge` passes unchanged.

Not done, filed separately: V4 compressed sparse attention and native block-scaled mma for MXFP4 experts.
