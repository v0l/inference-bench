# qwen3_moe FP8 prefill, grouped tensor core experts (#885)

Spark GB10, `Qwen/Qwen3-30B-A3B-FP8`, `bench --runs 3 <model> 1024,8192 16`, built from this branch.

| | prefill 1024 | prefill 8192 |
|---|---|---|
| before (2026-11-19 pass, per-expert GEMV path) | 306.5 | 273.7 |
| after (`moe_mma_up_f8` / `moe_mma_down_f8`) | 1453.5 | 1191.9 |
| vLLM (same pass) | 3959.2 | 6064.5 |

The FP8 experts now run on the tensor cores for 8 or more tokens: tokens are sorted per expert, activations are quantised to E4M3 with a block-128 scale (`moe_quant_f8`), and `moe_gemm_f8` (m16n8k32 e4m3 MMA, 128-wide K blocks, weight and activation block scales applied per K block) does gate+up+SwiGLU and down. It follows the NVFP4 grouped path. Serving specs set `[mlp] act_quant = "fp8"` for FP8 MoE checkpoints; `from_hf_config` (reference path) keeps it off.

nsys prefill 1024 after: `moe_mma_up_f8` 29%, `moe_mma_down_f8` 18%, `qkv_mma_big` 12%, `o_mma_big` 10%, `flash_mma` 10%, `moe_route_logits` 9% (about 520 us per layer, next target), `moe_combine` 5%.

Tests: `crates/specrt/tests/moe_mma_fp8.rs` (CPU reference on sm_89+, run on the Spark: max err 1.4e-3 and 4.3e-2 of 1079 and 1330).
