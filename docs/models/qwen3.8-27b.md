# Qwen3.8 27B

The model the project is measured on. Two checkpoints: `Qwen/Qwen3.8-27B-FP8` (block FP8) and `orcarouter/Qwen3.8-27B-Uncensored-NVFP4` (compressed-tensors mixed precision, the one the user's vLLM served). 64 layers, hidden 5120, MLP 17408, vocab 248,320, 262,144 positions; 48 gated delta-net layers and 16 full attention layers (one in four, 24 query and 4 KV heads of 256, output gate, qk norm, partial rotary 64); one MTP layer; a vision tower. How each part runs is in the design docs: [linear-attention.md](../design/linear-attention.md), [quantisation.md](../design/quantisation.md), [mtp.md](../design/mtp.md), [vision.md](../design/vision.md).

## Specs

| Spec | Checkpoint | Use |
|---|---|---|
| `crates/specrt/tests/fixtures/qwen3.8-27b.toml` | FP8, `quant = "fp8_block128"`, `act_quant = "fp8"` | serving |
| `crates/specrt/tests/fixtures/qwen3.8-27b-nvfp4.toml` | NVFP4, `quant = "fp8_channel"`, `act_quant = "fp8"`, `[mlp] quant = "nvfp4_g16"` with `quant_until = 56`, `[mlp] act_quant = "fp4"` | serving, the default for benches |
| `qwen3.8-27b-exact.toml`, `qwen3.8-27b-nvfp4-exact.toml` | the same, weight only | the exact reference tests (`SPECRT_27B_SPEC` defaults to them) |
| `qwen3.8-27b-nvfp4-head-fp8.toml`, `-head-nvfp4.toml` | quantised output heads | head tests |

They live in the fixtures, not `crates/specrt/specs/`, because specs there ship to the app as runnable models. `Spec::from_hf_config` derives the same quant from each checkpoint's `quantization_config` and `Spec::from_hf_config_serving` adds the activation quant from compressed-tensors `input_activations`; TOML specs are completed from the snapshot's `config.json` at load.

## Checkpoint layouts

FP8: every projection is e4m3 with a bf16 `weight_scale_inv` per 128 x 128 block; `in_proj_a`, `in_proj_b`, the conv, norms, `embed_tokens` and `lm_head` stay bf16 (the 96 gate rows are their own bf16 linear, `lin_ba`). On the device the grid stays as stored, with one u32 per fused row giving the offset of its grid row (21.5 MB for the model), so head splits, gate/up interleave and qkv fusion only remap rows.

NVFP4 (`quantization_config.config_groups`):

| Group | Format | Targets |
|---|---|---|
| `group_0` | e4m3 weights per channel, e4m3 activations per token, dynamic | attention q, k, v, o; delta-net `in_proj_qkv`, `in_proj_z`, `out_proj`; the MLPs of layers 56 to 63 |
| `group_1` | NVFP4: e2m1 packed, e4m3 scales per 16, global weight and input scales | the MLPs of layers 0 to 55 |

Not quantised: the vision tower, `lm_head`, `in_proj_a`, `in_proj_b`, the gated norms and convs, everything under `mtp.`. Shards: `model-00001` to `00005` plus `model-extra-00001-of-00001.safetensors` (MTP, bf16). Device weights 23.2 GB (MTP and vision not loaded), 24.8 GB with the NVFP4 draft head; decode reads about 20 GB per token.

## Memory on one RTX PRO 6000

| Item | Size |
|---|---|
| weights, NVFP4 checkpoint | 24.8 GB |
| weights, FP8 checkpoint | 29.5 GB |
| delta-net and conv state per sequence | 157 MB |
| KV per token, f16 one plane | 64 KiB (64 GiB at 1M tokens) |
| KV per token, fp8 | 33 KiB |

The context limit is the model's `max_position_embeddings` (262,144) and nothing in the engine caps it lower (`engine::tests::kv_paging_and_flat_offsets_do_not_overflow_i32_at_262144`); YaRN factor 4 opens 1,048,576. A 153,055 token prompt with a code word at its start was answered correctly (#580).

## Measured state

NVFP4 spec with MTP k = 2, server defaults, desktop GPU1 with the card otherwise idle, 27 to 28 Nov 2026, commits 0810eb2e and cdb6353e, against vLLM on the same harness (`crates/specrt/scripts/specrt_baseline.sh`; every number with its window and card state is in [../generated/qwen27-baseline.md](../generated/qwen27-baseline.md)):

| Prompt tokens | vLLM decode | specrt decode, fp8 KV | specrt decode, f16 KV | vLLM prefill | specrt prefill, fp8 KV |
|---|---|---|---|---|---|
| 128 | 112.0 | 99.6 | 108.1 | 2,195 | 1,760 |
| 1,024 | 105.6 | 89.0 | 108.0 | 7,912 | 4,496 |
| 4,096 | 118.2 | 80.2 | 92.0 | 8,781 | 5,441 |
| 32,768 | 106.0 | 64.3 | | 6,849 | 4,288 |
| 131,072 | 98.7 | 30.2 | | 3,764 | 2,346 |

Load: NVFP4 weights in 2.85 s cold and 2.34 s warm, a both caches cold start in 8.4 s ([engine.md](../design/engine.md)).

Accuracy gates (`crates/specrt/tests/real27b.rs`):

| Test | Gate | Measured |
|---|---|---|
| `real_27b_greedy_and_logits_match_the_fp8_reference` (FP8, against a f32 CPU oracle, 5 prompts) | weight only: greedy identical for 32 tokens; act spec: KL within 1.5x of the act reference's own gap | logits 2.0e-4 of max, 32 of 32; act KL 0.00108 against 0.00086 |
| `real_27b_nvfp4_agrees_with_vllm_greedy_and_logprobs` (20 prompts, 64 tokens, teacher forced on vLLM's tokens) | top-1 at least 90 percent, mean KL at most 0.05 | top-1 1,219 of 1,280, mean KL 0.0133 |
| `real_27b_per_layer_trace_matches_the_cpu_oracle` | every layer under 1e-3 | 7.6e-7 to 1.4e-5 |
| `real_27b_mtp_acceptance_on_vllms_own_greedy_tokens` | | acceptance length 2.34 |
| `real_27b_finds_a_needle_at_500k_with_yarn_extension`, `_at_1m_` | found | needs the whole card |

Greedy against the served vLLM cannot reach 90 percent token agreement: vLLM disagrees with itself at that level (MTP and batch dependent kernels), and every specrt divergence is at a near tie in vLLM's own distribution, so the gate is teacher forced.

## Tests and references

References: `crates/specrt/scripts/reference.py real-full | real-prompts | real-slice | real-mtp | nvfp4-dequant`, `vllm_logprobs.py`, `vllm_mtp_dump.py`, `vllm_vision.py`, kept in `~/.cache/specrt-ref-375`. Run one test at a time on GPU1 ([../guides/testing.md](../guides/testing.md), tier 5). The tiny stand in with every 27B code path is `tests/fixtures/tiny-qwen3.5-27b` ([../guides/test-fixtures.md](../guides/test-fixtures.md)). Scripts: `crates/specrt/scripts/specrt_baseline.sh`, `kvgate27b.sh`, `spec27b.sh`, `bench27b-nvfp4.sh`, `serve27b.sh`, `coldstart.sh`, `vllm27b_serve.sh` (a vLLM on port 8011 with the service's flags, for references).
