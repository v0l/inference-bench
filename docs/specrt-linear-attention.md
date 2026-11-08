# specrt: gated delta-net linear attention

Issue #308. The first engine step toward Qwen3.8 27B (see `qwen3.8-27b-vllm-baseline.md`): the
gated delta-net layers that make up 48 of its 64 layers, plus the changes the 16 full-attention
layers need, in bf16 and proven on a tiny random-weight model against `transformers`.

## What a hybrid spec looks like

```toml
min_engine = 2

[layers]
count = 64
pattern = ["linear", "linear", "linear", "full"]

[norm]
kind = "rmsnorm"
eps = 1e-6
unit_offset = true

[attention]
kind = "gqa"
heads = 24
kv_heads = 4
head_dim = 256
qkv_bias = false
qk_norm = true
output_gate = true
rotary_dim = 64

[linear_attention]
kind = "gated_delta_net"
key_heads = 16
value_heads = 48
key_head_dim = 128
value_head_dim = 128
conv_kernel = 4
```

The full file is `crates/specrt/tests/fixtures/qwen3.8-27b.toml`. It lives in the fixtures, not in
`specs/`, because everything in `specs/` ships to the app as a runnable model and the 27B cannot run
until FP8 or NVFP4 weights load. `Spec::from_hf_config` reads the real `config.json` (the nested
`text_config`, `rope_parameters`, `layer_types` reduced to its shortest repeating period).
`SPEC_FORMAT` is now 2; older apps list a hybrid spec as "needs a newer inference-bench".

New fields, all optional so the dense specs are unchanged on disk:

| field | meaning |
| --- | --- |
| `layers.pattern` `"linear"` / `"full"` | a gated delta-net layer, or a full GQA layer beside them (`"dense"` stays for pure dense models and cannot be mixed with `"linear"`) |
| `norm.unit_offset` | RMSNorm weights are stored as `w - 1` (Qwen3.5 `(1 + w)`); the loader adds the 1 |
| `attention.output_gate` | `q_proj` is twice as wide: per head `[query, gate]`; output is `attn * sigmoid(gate)` |
| `attention.rotary_dim` | partial rotary: only the first `rotary_dim` of each head rotate (64 of 256 here) |
| `[linear_attention]` | heads, head dims and conv kernel of the delta-net layers |

`validate` rejects: a linear layer without `[linear_attention]` and the reverse, value heads not a
multiple of key heads, key and value head dims that differ or are not a multiple of 32 up to 128, a
conv kernel outside 2..8, a layer count that is not a multiple of the pattern, and a rotary dim that
is odd or above the head dim.

## The layer, as implemented

Per token, for each linear layer (shapes for the 27B):

1. `in_proj` = one GEMM over `[qkv 10240 | z 6144 | b 48 | a 48]` rows, on the normed hidden state.
2. Causal depthwise conv over the 10240 qkv channels, kernel 4, then SiLU. The state is the last 3
   inputs per channel.
3. `q`, `k` L2-normalised per head (eps 1e-6), `q` scaled by `1/sqrt(128)`. Each of the 16 key heads
   serves 3 value heads.
4. `g = -exp(A_log) * softplus(a + dt_bias)`, `beta = sigmoid(b)`.
5. Delta rule with a 128 x 128 f32 state per value head: `S = exp(g) S; S += k (beta (v - S^T k))^T;
   o = S^T q`.
6. Gated RMSNorm per head: `norm_w * rmsnorm(o) * silu(z)`, then `out_proj` (6144 to 5120) added to
   the residual.

Decode (`gdn_conv_step`, `gdn_step`) is the recurrent form: one block of 512 threads per value head
holds a column slice of the state in registers, does the decay, the `S^T k` read, the rank-1 update
and the `S^T q` read in one pass, then the gated norm.

Prefill is the chunked form, chunk 64, with the WY representation (the same algorithm as
`torch_chunk_gated_delta_rule` and the flash-linear-attention `chunk_gated_delta_rule`):

- `gdn_chunk_prep`, one block per (value head, chunk), all chunks in parallel: cumulative `g`,
  `A = -beta_i (k_i . k_j) exp(g_i - g_j)` strictly below the diagonal, the forward substitution
  that turns it into `T = (I - A)^-1`, then `u = T (beta v)`, `w = T (beta k exp(g))` and the masked
  intra-chunk `q k^T exp(g_i - g_j)`.
- `gdn_chunk_scan`, one block per (value head, 16 state columns), sequential over chunks:
  `v' = u - w S`, `o = (q exp(g)) S + Aqk v'`, `S = exp(g_last) S + (k exp(g_last - g))^T v'`.

Both run their matrix products on tensor cores through `mma_bf16` with the same hi/lo bf16 split
the dense prefill GEMMs use (three MMAs per tile), so they keep about f32 accuracy. The forward
substitution stays scalar (64 dependent rows). The state is f32 throughout, as in the checkpoint
(`mamba_ssm_dtype = float32`).

The conv window and the delta-rule state for every linear layer of every sequence sit in two flat
buffers beside the KV cache (`GdnBuffers` in `runtime.rs`); the KV cache now holds only the full
attention layers, so the 27B pays KV memory for 16 layers, not 64. The recurrent state is reset at
the start of every `generate`, `prompt_logits` and batch prefill, because unlike KV it is not keyed by
position. Batched decode (`generate_batch`) works for hybrid specs too.

## Tests and results

The oracle is a tiny Qwen3.5 model: 3 layers (`linear, linear, full`), hidden 256, 2 key heads and 6
value heads of 128 (the 27B's 1:3 ratio and head dim), full attention 4 heads of 128 with partial
rotary 32 and the output gate, vocab 1024. Weights are random with seed 0, scaled like the real
checkpoint (`A_log` in -5.5..-1, `dt_bias` around 2.8), stored as bf16 in
`crates/specrt/tests/fixtures/tiny-qwen3.5/` (7 MB). `scripts/reference.py tiny <fixture> <out>`
builds it the first time and dumps per-layer hidden states, logits and the final conv and delta
states from `transformers` 5.15 (the `~/.venv-vllm` one; the system 4.57 has no `qwen3_5`), on CPU in
f32.

`crates/specrt/tests/linear.rs`, on the RTX PRO 6000 (shared with the live vLLM 27B), max abs:

| test | per layer | logits | delta state | conv state |
| --- | --- | --- | --- | --- |
| decode 32 tokens, recurrent | 1.9e-6, 3.3e-6, 4.8e-6 | 2.6e-6 | 8.3e-7 | 1.5e-6 |
| prefill 1024 tokens, chunked on tensor cores | 3.3e-5, 5.1e-5, 6.6e-5 | 4.5e-5 | 1.2e-5 | 2.1e-5 |
| prefill 1024 then compare with 1024 recurrent decode steps | | | 1.5e-5 (state max 1.40) | 2.1e-5 |
| prefill 1024 + decode 16 vs one 1040-token prefill | | 3.0e-5 | | |
| whole model, 48-token prompt | | 4.0e-5, top-1 48/48 | | |

Greedy decode of 64 tokens is identical to `transformers.generate`, and 4-sequence batch decode is
identical to single-sequence greedy. The bound in every assertion is 1e-3 for activations and logits
and 1e-4 for states.

The issue asked for a bit-identical state hand-off between prefill and decode. It is not
bit-identical and cannot be without giving up the chunked form: the chunked algorithm sums the same
terms in a different order (and through the WY inverse) than the token-by-token recurrence, so the
f32 results differ in the last bits. The conv window differs too, by 2e-5, because the prefill path
gets its inputs from the tensor-core `in_proj` GEMM and the decode path from the scalar GEMV. What
the test proves instead is that the hand-off is exact in meaning: prefill then decode gives the same
logits as one long prefill to 3e-5, and the chunked state equals the recurrent state to 1.5e-5.

The dense suites still pass unchanged: `logits` (all 8, `SPECRT_REF_DIR=/tmp/ref273`), `generate`
(6, including batch decode and contexts over 4096), `chat`, `jit`, and the workspace tests.

## Kernel times

`nsys` kernel sums for the tiny model on the shared GPU (vLLM at about 95% busy beside it), so read
them as relative numbers.

Prefill 1024 tokens, per linear layer:

| kernel | scalar f32 | tensor cores, 32 columns | tensor cores, 16 columns |
| --- | --- | --- | --- |
| `gdn_chunk_scan` | 1260 us | 316 us | 202 us |
| `gdn_chunk_prep` | 193 us | 83 us | 80 us |

For comparison the full-attention `flash_mma` layer is 138 us for the same 1024 tokens. The scan is
latency bound on the tiny shape (6 heads x 8 column blocks = 48 CTAs on 188 SMs, 16 chunks in
sequence); the 27B has 48 value heads, so 384 CTAs, and fills the card.

Decode, per token per linear layer: `gdn_step` 5.2 us, `gdn_conv_step` 1.2 us, `lin_in_norm` 2.5 us,
`lin_out_proj` 1.7 us (medians).

## head_dim 256

The 27B's full-attention layers use `head_dim = 256`. `attention_split` keeps one float4 column per
lane and loops over `ceil(head_dim / 128)` columns, so 256 needs no extra block size.
`flash_mma` stages Q as 128 rows x head_dim x 4 B, which is 128 KB at 256 and over the 99 KB
per-block limit, so above 128 it runs 64 query rows and 16-key tiles (64 KB). `Module::generate`
fails with the byte count if a shape would exceed 99 KB.

The second oracle is the same tiny model with `head_dim = 256`
(`tests/fixtures/tiny-qwen3.5-hd256/`, 8 MB, references named `tiny256-*`). Decode, prefill, logits
and batched decode all match to the same 1e-3 as the 128 fixture (logits max abs 4.9e-5 at 1024
tokens, 4.0e-5 over the 48-token prompt). `qwen38_27b_module_compiles_through_nvrtc` in `tests/jit.rs`
compiles the real 27B spec.

Prefill cost of the smaller tiles, nsys time of one `flash_mma` call over the same 1024-token tiny
prefill (4 heads, one full layer): 137 us at head_dim 128, 284 us at 256. That is 2.1x for 2x the
head dim, so per-FLOP the 64-row tiling costs close to nothing; the 27B's 16 full layers will pay
the 2x of the wider heads, not an extra tiling penalty.

Dense models are unchanged. Qwen3-0.6B (head_dim 128), before/after this change, 6 alternating
`bench` runs, medians, GPU shared with the live vLLM:

| run | prefill tok/s before | after | decode tok/s before | after |
| --- | --- | --- | --- | --- |
| 128 prompt / 128 new | 12629 | 12708 | 450.5 | 448.6 |
| 1024 prompt / 128 new | 31752 | 34889 | 380.3 | 380.8 |

nsys, same runs, median per call (us, before -> after): at 128 tokens `attention_split` 5.09 ->
5.02, `attention_reduce` 2.24 -> 2.24, `flash_mma` 36.3 -> 36.4; at 1024 `attention_split` 9.92 ->
9.73, `attention_reduce` 4.45 -> 4.38, `flash_mma` 164.1 -> 161.4. Means differ because of
warm-up outliers on the shared card.

## Regenerating every reference

The tests read one folder, `SPECRT_REF_DIR` (default `crates/specrt/out`). Everything below writes
into it. `/tmp` is deleted on reboot, so rerun this after one.

```sh
uv venv ~/.venvs/specrt-ref
uv pip install -p ~/.venvs/specrt-ref "transformers>=5.1" torch numpy safetensors
export SPECRT_REF_DIR=/tmp/ref273
cd crates/specrt
PY=~/.venvs/specrt-ref/bin/python
for s in specs/*.toml; do $PY scripts/reference.py $s $SPECRT_REF_DIR; done
$PY scripts/reference.py tiny tests/fixtures/tiny-qwen3.5 $SPECRT_REF_DIR
$PY scripts/reference.py tiny tests/fixtures/tiny-qwen3.5-hd256 $SPECRT_REF_DIR 256
cargo test --release -- --include-ignored --test-threads=1
```

The per-spec call needs the HF weights for each spec's `repo`; the `tiny` calls need only the
fixtures and write `tiny-*` (128) and `tiny256-*` (256) files. Run the suite with
`--test-threads=1` because the GPU is shared and `contexts_above_4096` needs most of the free memory.

`scripts/reference.py` writes `rope_parameters` (with `rope_theta` from the model's own config) for
rope-scaled specs, so it needs transformers 5. On this machine use `PY=~/.venv-vllm/bin/python`.

## What the next issue needs (FP8 and NVFP4 weights for the real 27B)

The 27B in bf16 is 54 GB of weights and does not fit beside vLLM; the FP8 checkpoint
`Qwen/Qwen3.8-27B-FP8` is the next target. What it takes, from the checkpoint as it is on disk:

- Sharded safetensors: `model.safetensors.index.json` with `layers-N.safetensors`,
  `outside.safetensors` and `mtp.safetensors`. `find_safetensors` only finds one
  `model.safetensors`; the loader must open every shard named in the index.
- FP8 e4m3 block-scaled weights: `weight` is `F8_E4M3` and `weight_scale_inv` is bf16 with one scale
  per 128 x 128 block (`weight_block_size [128, 128]`, `activation_scheme dynamic`). Every
  projection is FP8: `in_proj_qkv`, `in_proj_z`, `out_proj`, `q/k/v/o_proj`, `gate/up/down_proj`.
  Kept in bf16: `in_proj_a`, `in_proj_b` (48 rows each), `conv1d`, `A_log`, `dt_bias`, the norms,
  `embed_tokens` and `lm_head`. So `in_proj` (one fused GEMM here) splits into an FP8 part (16384
  rows) and a bf16 part (96 rows), or the 96 rows get upcast to the FP8 block layout at load.
- An FP8 GEMM on sm_120: `mma.sync ... e4m3` with per-token dynamic activation scales and the
  128 x 128 weight block scales applied per K block, the same way `Q8` applies its group scales in
  `gemm_tc` now. Decode needs an FP8 GEMV with the same scales.
- The NVFP4 checkpoint (`orcarouter/Qwen3.8-27B-Uncensored-NVFP4`, compressed-tensors
  mixed precision) has FP4 e2m1 MLPs with group-16 scales; that is its own issue after FP8.
- Memory at 27B: about 27 GB of FP8 weights plus 5 GB for the bf16 `embed_tokens` and `lm_head`.
  KV for 16 layers x 4 heads x 256 x 2 (K, V) in the current f16 hi/lo pair is 128 KB per token;
  the linear state is 48 layers x 48 heads x 128 x 128 x 4 bytes = 151 MB per sequence plus 5.9 MB
  of conv window, independent of context.
- The vision tower and `mtp.*` are ignored by this loader; MTP is a later issue.
- `head_dim` 256 on the full layers: the decode `attention_split` kernel asserts `HEAD_DIM <= 128`,
  so a 27B module does not compile yet. The tiny fixture uses 128 to stay inside it; the 27B needs
  that kernel extended to 256, and `flash_mma` and `batch_decode_attention` checked at 256 (their
  shared memory and register use grow with the head dim).
