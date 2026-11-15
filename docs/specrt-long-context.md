# Long context with YaRN

Qwen3.8 is trained for 262144 tokens. Qwen documents 1,048,576 with YaRN: `rope_type: "yarn"`, `factor: 4.0`, `original_max_position_embeddings: 262144`. specrt reads that from a checkpoint config, and also offers it as an opt-in load option.

## The maths

`spec::yarn_inv_freq` and `Spec::rope_constants` follow transformers' `_compute_yarn_parameters`:

- Each rotated pair `i` blends the plain frequency with the frequency divided by `factor`. The blend is a linear ramp over the pair index between the correction dims for `beta_fast` (32) and `beta_slow` (1), found from `original_max_position_embeddings`. `truncate` (default true) floors and ceils them.
- Cos and sin are multiplied by the attention factor: `attention_factor` if given, else `mscale / mscale_all_dim` ratios when both are given, else `0.1 * ln(factor) + 1`.
- Only the rotated dims are affected (`partial_rotary_factor` 0.25 gives 64 of 256 on Qwen3.8). The interleaved mrope sections only choose which position axis feeds a pair, so they compose with it unchanged.
- The CUDA (`ops.cu`, ROPE_KIND 3) and WGSL kernels bake the constants at JIT time, nothing is computed per model in the kernel source.

`dynamic` YaRN is not implemented: Qwen's documentation only uses the static form, and a frequency table that changes with the sequence length does not fit a baked kernel and a paged cache.

## Load option

```toml
[runtime]
rope_extension = { type = "yarn", factor = 4 }
```

Also in the server config `[runtime]` table and the load dialog (advanced, runtime: context extension, extension factor). It multiplies the model's own `max_position_embeddings`, which is what `model_max_context`, the serving `max_model_len` check and the dialog slider follow. It is off by default because YaRN changes short context numerics slightly. A model that already declares rope scaling refuses the option. The prefix cache namespace includes it, so KV computed with and without extension is never shared.

## Memory at 1M tokens (Qwen3.8 27B, 16 attention layers, 4 KV heads, head dim 256)

| KV type | per token | at 1,048,576 tokens |
|---|---|---|
| f16 | 64 KiB | 64 GiB |
| fp8 | 33 KiB | about 33 GiB |

The load dialog shows both numbers at the model maximum, prices the pool for the chosen type, and disables load when the estimate does not fit the free GPU memory or the pool cannot hold one request.

## Tests

- `tests/yarn.rs`: tables against `tests/fixtures/yarn_reference.json` (transformers 5.22.2, made by `scripts/yarn_reference.py`) at positions 0, 1, 4096, 262143, 262144, 262145, 524288 and 1,000,000, three parameter sets; option parsing, limits, prefix key, kernel constants.
- `tests/yarn.rs` (ignored, GPU): the tiny hybrid fixture prefills and decodes at positions 1,048,000 to 1,048,140 with YaRN, with a straight and a reversed block table giving identical logits.
- `tests/logits.rs` `yarn_rope_scaling_logits_match_transformers`: Qwen2.5-0.5B with YaRN forced in both engines (spec `qwen2.5-0.5b-yarn4`).
- `tests/real27b.rs` (ignored, GPU): `real_27b_finds_a_needle_at_500k_with_yarn_extension` and `real_27b_finds_a_needle_at_1m_with_yarn_extension`:

```sh
flock /tmp/gpu27.lock systemd-run --user --wait --collect --pipe -p MemoryMax=16G -p MemorySwapMax=0 \
  cargo test -p inference-bench-specrt --release --test real27b -- --ignored --test-threads=1 needle
```

They need the whole 96 GB card (weights, 33 GiB fp8 KV at 1M, recurrent state, activations), so they run in a quiet GPU window.
