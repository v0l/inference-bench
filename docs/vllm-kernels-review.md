# vLLM's kernels on our 27B, and what specrt should take from them

Research for #411. Sources: the `local-llm.service` unit and journal, the launcher
`~/qwen3.8-27B-NVFP4.sh`, the FlashInfer JIT cache (`~/.cache/flashinfer/0.6.16.post3/120f`), the
FlashInfer autotune cache (`~/.cache/vllm/flashinfer_autotune_cache/.../autotune_configs.json`), the
Triton cache (`~/.triton/cache`, 622 entries), vLLM's torch.compile cache
(`~/.cache/vllm/torch_compile_cache`), the installed packages in `~/.venv-vllm` (vLLM 0.27.1,
FlashInfer 0.6.16.post3, Triton 3.7.1, torch 2.13.0+cu130, CUTLASS DSL 4.6.0) and vLLM's C++
sources at the v0.27.1 tag (`csrc/`, sparse checkout in `/tmp/vllm-csrc`). Nothing was run on
the GPU and nothing imported vLLM or torch.

The card here is the RTX PRO 6000 Blackwell Max-Q (sm_120, 96 GB, 300 W, streaming read 1,640 to
1,660 GB/s beside vLLM, `docs/specrt-nvfp4.md`). The GB10 (sm_121) runs the same sm_120 kernels in
both engines (FlashInfer builds `Arch = Sm120` for both).

Every specrt number below was taken beside the live vLLM at 92 to 100 percent utilisation and
the 300 W cap. vLLM's numbers are its own 10 s journal averages over real traffic. Neither is a
clean benchmark, so per-kernel `nsys` times are the only fair comparison, and vLLM has none yet
(see "Measured").

## What vLLM runs for this model

From the 9 Nov start of `local-llm.service` (EngineCore pid 36931 and later restarts, same config):

| piece | what vLLM picked | where it shows |
|---|---|---|
| NVFP4 MLP GEMM (layers 0 to 55) | `FlashInferCutlassNvFp4LinearKernel`, FlashInfer `fp4_gemm_cutlass_sm120` (CUTLASS 3.x, `KernelTmaWarpSpecializedCooperative`, `OpClassBlockScaledTensorOp`, cluster 1x1x1) | journal `__init__.py:1077`; JIT cache `cached_ops/fp4_gemm_cutlass_sm120` with 16 instantiations (`fp4_gemm_cutlass_{half,bf16}_{128_32_128 ... 256_128_128}.cu`) |
| activation quant for NVFP4 | `torch.ops._C.scaled_fp4_quant` (vLLM CUDA, `csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu`), 224 calls per forward graph | `computation_graph.py` of the backbone |
| FP8 projections (qkv, o, in_proj_qkvz, out_proj, MLP 56 to 63) | `CutlassFP8ScaledMMLinearKernel`: `torch.ops._C.cutlass_scaled_mm`, sm120 dispatch in `csrc/.../c3x/scaled_mm_sm120_fp8_dispatch.cuh`, 288 calls per graph; activations quantised per token by Inductor-generated Triton | journal `__init__.py:665`; graph |
| `in_proj_ba` and lm_head | bf16 `torch._C._nn.linear` (cuBLAS) | graph |
| full attention, 16 layers, head dim 256 | FlashInfer, FA2 template, `batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_..._head_dim_qk_256` (the only attention module in the JIT cache), decode through the same tensor-core kernel (`use_tensor_cores=True`, `decode_backend=flashinfer-native`), fp8 e4m3 KV | journal `flashinfer.py:824`, JIT cache, `v1/attention/backends/flashinfer.py:1031` |
| gated delta-net prefill | Triton FLA chunked kernels vendored in `vllm/third_party/flash_linear_attention/ops` | journal `qwen_gdn_linear_attn.py:150` ("Triton/FLA GDN prefill kernel"); Triton cache: `chunk_local_cumsum_scalar_kernel`, `chunk_scaled_dot_kkt_fwd_kernel`, `merge_16x16_to_64x64_inverse_kernel`, `recompute_w_u_fwd_kernel`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_fwd_kernel_o`, `_fused_post_conv_kernel` |
| gated delta-net decode | `fused_recurrent_gated_delta_rule_packed_decode_kernel` (no speculation in the step) and `fused_sigmoid_gating_delta_rule_update_kernel` (speculative verify) | Triton cache, `qwen_gdn_linear_attn.py:1180-1496` |
| causal conv1d | Triton `_causal_conv1d_fwd_kernel` and `_causal_conv1d_update_kernel` (`model_executor/layers/mamba/ops/causal_conv1d.py`) | Triton cache |
| RMSNorm, residual add, SwiGLU, gates, FP8 quant | `custom_ops: ['none']`: native torch ops compiled by Inductor into fused Triton kernels | compile config, Inductor cache names (below) |
| rope | Triton `rotary_kernel` / `_triton_mrope_forward` | Triton cache |
| sampling | FlashInfer top-k/top-p sampling (`cached_ops/sampling`), rejection sampler in Triton (`rejection_greedy_sample_kernel`, `rejection_random_sample_kernel`, `sample_recovered_tokens_kernel`) | journal `topk_topp_sampler.py:62`, Triton cache |
| MTP drafter | `Qwen3_5MTP`, own torch.compile graph (`eagle_head`), shares embedding and lm_head, padded drafter batch prepared by Triton (`eagle_prepare_inputs_padded_kernel`, `eagle_prepare_next_token_padded_kernel`) | journal `llm_base_proposer.py:1478,1558`, Triton cache |
| CUDA graphs | `FULL_AND_PIECEWISE` requested, downgraded to `PIECEWISE` because FlashInfer's FULL graphs do not support spec decode; 27 sizes `[1, 2, 4, 8, 16, 24, ..., 192]`; attention, `qwen_gdn_attention_core` and KV update run eagerly between graph pieces | journal `compilation.py:1458`, `gpu_model_runner.py:6681` |
| torch.compile | Inductor, one dynamic range `(1, 16384)`, `combo_kernels: True` (horizontal fusion of independent small kernels), every `pass_config` fusion off (`fuse_norm_quant`, `fuse_act_quant`, `fuse_attn_quant`, `enable_qk_norm_rope_fusion` all `False`), allreduce fusions irrelevant at TP 1 | journal engine config |
| autotune | FlashInfer autotuner times every CUTLASS fp4 config for M in powers of two from 1 to 16384 for both MLP shapes at startup (44 entries, 20 s cold, reused from the json afterwards); Triton `@autotune` for the FLA kernels keyed by `(H, K, V, BT)` | journal `autotuner.py`, json, `*.autotune.json` in the Triton cache |
| PDL | the fp4 GEMM launches with `enablePDL=true`; DeepGEMM PDL enabled (DeepGEMM is not used by this model) | `fp4_gemm_template_sm120.h:188`, journal `deep_gemm.py:202` |

### FlashInfer's NVFP4 tile choices on this card

The autotune json records a config index per M. `CutlassFp4GemmRunner::getConfigs`
(`flashinfer/data/include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h:187`) lists 8 tiles,
each in the order (swap A/B data-parallel, plain data-parallel, swap A/B stream-K, plain
stream-K), so the index decodes as below (my decoding of that order; the json itself stores only
the index):

| M (tokens) | gate_up 34816 x 5120 | down 5120 x 17408 |
|---|---|---|
| 1 to 32 | 4: CTA 128 x 32 x 256, A/B swapped, persistent data-parallel | 4 |
| 64 | 12: 128 x 64 x 256, swapped | 4 |
| 128 | 12 | 12 |
| 256 | 16: 128 x 128 x 128, swapped | 12 |
| 512 | 25: 256 x 128 x 128, plain | 16 |
| 768 to 16384 | 24 to 27: 256 x 128 x 128, swapped or plain, data-parallel or stream-K | 19, 24, 26 |

So for decode and MTP verify (M up to 32) vLLM puts the weight rows on the MMA's M side in
128-row tiles and the tokens on the 32-wide N side: the same layout as specrt's `gemv_f4b`, but
it runs up to 32 tokens, and up to 256 with a 64- or 128-wide N tile, before it changes kernel.

### What Inductor fuses

Kernel names in the AOT compile cache (`torch_aot_compile/*/inductor_cache`):

- `triton_red_fused__to_copy_abs_add_clamp_cutlass_scaled_mm_div_fused_add_rms_norm_max_mul_reciprocal_unsqueeze_*`:
  residual add, RMSNorm with the `1 + w` weight, per-token absmax, FP8 e4m3 quantisation and the
  scale, in one reduction kernel feeding `cutlass_scaled_mm`. This is the norm-quant fusion that
  `fuse_norm_quant` would do with a CUDA kernel; here Inductor gets it because the FP8 quant is
  written as native torch (`input_quant_fp8.py:199`, `forward_native`).
- `..._cutlass_scaled_mm_div_max_mul_reciprocal_silu_slice_unsqueeze_*`: SwiGLU plus FP8 quant
  before `down` on the FP8 MLP layers.
- `..._sigmoid_unsqueeze_view_*`: the attention output gate `o * sigmoid(gate)` plus FP8 quant
  before `o_proj`.
- `triton_poi_fused_mul_scaled_fp4_quant_silu_slice_zeros_*` and
  `..._fused_add_rms_norm_scaled_fp4_quant_zeros_*`: SwiGLU or add-norm plus the zero fill of
  the fp4 scale buffer. The fp4 quantisation itself stays a separate CUDA kernel
  (`scaled_fp4_quant`), so every NVFP4 linear costs two extra launches in vLLM.
- `combo_kernels` packs small independent pointwise kernels into one launch.

## Kernel by kernel

### NVFP4 GEMV and small-M GEMM (decode, MTP verify)

| | |
|---|---|
| vLLM | FlashInfer `fp4_gemm_cutlass_sm120`, `fp4_gemm_template_sm120.h`. Activations first quantised to e2m1 by `scaled_fp4_quant` (separate kernel). |
| how it is fast | TMA loads of A, B and both ue4m3 scale tensors, warp-specialised producer/consumer with an mbarrier ring whose depth is `StageCountAutoCarveout` (all shared memory left after the epilogue), block-scaled `mma` (`OpClassBlockScaledTensorOp`), scales in CUTLASS's swizzled 128 x 4 layout so one TMA box carries them, persistent tile scheduler (`StaticPersistentScheduler`) or stream-K, A/B swap for small M, TMA epilogue with alpha (`1 / (wgs * igs)`) folded in, PDL so the next kernel's prologue overlaps the tail, per-M config chosen by measurement at startup. |
| specrt | `gate_up_act_f4` / `down_proj_f4` (`templates/fp4.cu`): one warp per row (two rows for gate/up), `ld.global.nc.L1::no_allocate`, f32 activations transposed in shared memory, norm fused into the prologue, SwiGLU or residual in the epilogue. Batch 2 to 16: `gemv_f4b` (`templates/fp4_batch.cu`): weight rows on the mma M side, tokens on N (8 or 16), 3-stage `cp.async` ring for the activation planes, split-K across `blockIdx.y` from `f4b_splits`, separate reduce kernel. Above 16 rows: the `_tiny` tile GEMM. |
| gap | Single row: 1,278 GB/s median, 1,470 min after #362, 78 to 89 percent of the 1,640 ceiling. vLLM's per-kernel number is unknown. Batch 8: 160 us for the pair against 121.5 us single row (1.32x). Above 16 rows specrt drops to the tile GEMM: 251 to 261 us for gate_up, about 400 GB/s, and reads the weights once per 16 tokens. That is why MTP at batch 8 (24 verify rows) is 0.81x of plain decode (115.9 against 143.7 tok/s, `docs/specrt-mtp.md`). vLLM keeps one weight read up to 32 rows and stays on a swapped small-N tile up to 256. |
| port | Extend `gemv_f4b` (and `gemv_f8b`) to 32 and 64 tokens per weight read: more N tiles per warp, or the native `mxf4nvf4` mma at m16n8k64 with fp4 activations for 17 to 64 rows, keeping weights on the M side. |
| expected win | MTP at batch 8 from 0.81x to 1.1 to 1.3x of plain (115.9 to 160 to 190 tok/s, guess from batch 4's 1.17x with 12 rows), and decode at 17 to 32 slots stops reading the weights twice. |
| cost | 2 to 3 days: the template, `codegen.rs` (`F4B_MAX_TOKENS`, `f4b_tokens`, `f4b_smem`), `runtime.rs` (`tc_f4b`, `tc_f8b`), tests in `tests/fp4_batch.rs` and `tests/fp8_batch.rs`. |

TMA, the mbarrier ring and a persistent grid might lift the single-row GEMV from 78 to about 90
percent of the ceiling (guess). #362 already tried more loads in flight and larger blocks; the
measurement issue below decides whether vLLM's GEMM is actually faster at M = 1 before anyone
writes a TMA GEMV.

### FP8 GEMM (prefill) and FP8 GEMV

| | |
|---|---|
| vLLM | `cutlass_scaled_mm` sm120 (`csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh`). Per-token activation scale from the fused Inductor kernel, per-channel weight scale. |
| how it is fast | Static dispatch by M: up to 16 rows `16 x 64 x 128` Pingpong, up to 32 `32 x 64 x 128` Pingpong, up to 256 `64 x 64 x 128` Pingpong, above that `128 x 128 x 128` with `KernelScheduleAuto` (TMA warp-specialised Cooperative). Stage count is auto-carved from shared memory. Both scales are applied once in the epilogue (EVT `ScaledEpilogue`: `acc * scale_a[row] * scale_b[col]`), so the main loop is pure e4m3 mma. No autotune for FP8. |
| specrt | `gemm_fp8` (`templates/ops.cu:702`), `[runtime] act_quant = "fp8"`: 2-stage `cp.async` (`FP8_ACT_STAGES = 2`), plain 32-bit `ld.shared` fragment loads (no `ldmatrix`), a padded 144-byte smem row, activation scale per 128 columns and weight scale per 128 columns applied to a fresh accumulator after every k tile (`acc += c * sa * sw` per k step), tiles 64 x 128 (`_big`), 64 x 64, 64 x 32. Activations quantised by `quant_rows_fp8` from the hi/lo planes. Decode: `gemv_f8b` (2 to 16 rows) and the per-row GEMVs. |
| gap | Native FP8 `gate_up` 1024 x 34816 x 5120: 4.9 ms, 74 TFLOP/s; `lin_in`: 0.95 ms, 1024 x 16384 x 5120, about 180 TFLOP/s; NVFP4 on the same shape runs at 315 TFLOP/s (`docs/specrt-nvfp4.md`). The FP8 projections are 37 percent of native prefill time. vLLM's per-call time on this card is unknown. |
| port | Per-token activation scale (one scale per row, as vLLM and the checkpoint do; `docs/specrt-nvfp4.md` already lists this as the one known difference), which lets both scales move to the epilogue and drops the per-k-tile rescale; 3 or 4 `cp.async` stages; `ldmatrix` for the A and B fragments; a 128 x 128 tile with 8 warps for M above 256; a Pingpong-style split of producer and consumer warps is a second step. |
| expected win | 1.5 to 2x on the FP8 GEMMs (guess from the 4x gap to the fp4 kernel and the per-k-tile FMA work removed), so 12 to 18 percent on native prefill. |
| cost | 3 to 4 days: `templates/ops.cu` (`gemm_fp8`, `quant_rows_fp8`), `codegen.rs` (`fp8_act_smem`, `FP8_ACT_STAGES`, tiles), `runtime.rs` (`tc_fp8`, the xscale layout), the native-path accuracy tests. |

### Gated delta-net decode (recurrent step and speculative verify)

| | |
|---|---|
| vLLM | `fused_recurrent_gated_delta_rule_packed_decode_kernel` (`third_party/flash_linear_attention/ops/fused_recurrent.py:256`), `fused_sigmoid_gating_delta_rule_update_kernel` (`fused_sigmoid_gating.py`) for verify rows. |
| how it is fast | Grid `(V / 32, B * HV)`: 4 programs per value head per sequence, one warp each (`num_warps = 1`, `BV = 32`, `BK = 128`), so the 27B has 192 programs per sequence; the 32 x 128 f32 state slice lives in registers; q, k, v read straight out of the packed conv output; softplus/sigmoid gating and the q/k L2 norm fused in; state read once and written once in place through `ssm_state_indices`. For verify, one program walks the k + 1 rows in order and writes the state after each row to its own slot, so rejection is an index choice (`num_accepted_tokens`), not a recompute. State stays f32 (`mamba_ssm_dtype = float32`). |
| specrt | `gdn_step` (`templates/gdn.cu:103`): one 512-thread block per value head (`LA_D * LA_KSPLIT`), so 48 blocks per sequence on 188 SMs at batch 1; q/k L2 norm, gates, rank-1 update and the gated RMSNorm with `silu(z)` fused in one kernel (vLLM runs the gated norm separately); `gdn_verify` and `gdn_commit` with a deferred commit. |
| gap | Unknown on the 27B: the only per-kernel number is 5.2 us per layer on the tiny fixture (6 heads). The floor is the state traffic: 48 heads x 64 KiB read and written per layer, 302 MB per token per sequence over 48 layers, 0.19 ms at 1,640 GB/s, about 1 percent of a 21 ms decode token at batch 1 and 10 percent at batch 8. 48 blocks of 512 threads may not reach it (guess). |
| port | vLLM's grid (V split in 32-column slices, one warp per slice, more CTAs), keeping specrt's fused gated norm (it needs a cross-slice reduction: a second tiny kernel or a cluster). |
| expected win | Only if `nsys` shows `gdn_step` well above its floor. Guess: 0.3 ms per token at batch 1 (1 to 2 percent), more at batch 8. |
| cost | 1 to 2 days after the measurement. |

### Gated delta-net chunked prefill

| | |
|---|---|
| vLLM | FLA `chunk_gated_delta_rule_fwd` (`third_party/flash_linear_attention/ops/chunk.py:23`): `chunk_local_cumsum`, `chunk_scaled_dot_kkt_fwd`, `solve_tril`, `recompute_w_u_fwd`, `chunk_gated_delta_rule_fwd_h` (`chunk_delta_h.py`), `chunk_fwd_o`; `fused_post_conv_prep` before it. |
| how it is fast | Chunk 64. The `(I - A)^-1` of each chunk is built from four 16 x 16 inverses merged into 64 x 64 (`merge_16x16_to_64x64_inverse_kernel`, `solve_tril.py:238`), so the inverse is a few small matmuls instead of 64 dependent rows. All products are bf16 `tl.dot` with f32 accumulate. The state recurrence (`..._h_blockdim64`) splits K into 64-wide blocks held as separate register tiles and V into `BV` slices, autotuned over `BV`, `num_warps` 2 or 4 and `num_stages` 2 to 4, keyed by `(H, K, V, BT)` (cache entries for H = 32, 48, 64). Conv, l2norm and gating are fused in one post-conv kernel. |
| specrt | `gdn_chunk_prep` (scalar forward substitution over 64 rows) and `gdn_chunk_scan` (one block per value head and 16 state columns, sequential over chunks), every product through `mma_bf16` with the hi/lo split (three mma per product) to keep f32 accuracy. |
| gap | On the 27B unknown per kernel; the native prefill profile puts MLPs at 32 percent and FP8 projections at 37 percent, so delta-net, attention and the rest share 31 percent. On the tiny fixture `gdn_chunk_scan` is 202 us and `gdn_chunk_prep` 80 us per layer at 1,024 tokens. |
| port | The block-merged 16 x 16 inverse in `gdn_chunk_prep`; drop the hi/lo split on the delta-net products when `act_quant` is on (that path already accepts the activation-quantisation envelope). |
| expected win | Guess: 5 to 10 percent of native prefill, decided by the measurement. |
| cost | 2 to 3 days. Porting the algorithm is fine; copying `solve_tril.py` needs the Apache-2.0 notice with vLLM's and Songlin Yang and Yu Zhang's copyright lines. |

### Causal conv1d

| | |
|---|---|
| vLLM | Triton `_causal_conv1d_update_kernel` (decode, with `num_accepted_tokens` for verify) and `_causal_conv1d_fwd_kernel` (prefill), `model_executor/layers/mamba/ops/causal_conv1d.py`. Separate launches from the delta rule. |
| specrt | `gdn_conv_step`, `gdn_conv_prefill`, `gdn_conv_keep`, `gdn_conv_verify`, `gdn_conv_commit`: 1.2 us per layer on the tiny fixture. |
| gap | None known; 10,240 channels x 4 taps is tiny next to the projections. Fusing the conv into `gdn_step` (vLLM does not) would save 48 launches per token; that belongs with PDL below. |

### Full attention, head dim 256, decode and prefill

| | |
|---|---|
| vLLM | FlashInfer FA2-template `BatchPrefillWithPagedKVCache` JIT-built for bf16 q, e4m3 KV, head dim 256, causal; decode runs the same tensor-core kernel (`use_tensor_cores=True`). |
| how it is fast | GQA packing: the 6 query heads of a KV head become 6 rows of the MMA's M dimension (`packed_qo_len = qo_len * group`), so decode and MTP verify (3 rows x 6 heads = 18) use tensor cores; `cta_tile_q` from `FA2DetermineCtaTileQ` (16 for 16 or fewer packed rows, 64 up to 64, 128 above for head dim under 256); KV split chosen on the host by a binary search for the chunk size that fills the SMs (`PrefillBinarySearchKVChunkSize`, `scheduler.cuh`), planned once per batch outside the graph; fp8 KV halves the bytes; swizzled shared memory with `ldmatrix` and `cp.async`; page size 16 to 64 (with a 1,600-token logical block to line up with the Mamba page). |
| specrt | Decode: `attention_split` + `attention_reduce` (single sequence, 64 splits, CUDA cores, f32), `batch_decode_attention` + `batch_decode_combine` (all sequences, kv head x sequence x 8 splits, the GQA group shares each K/V load in registers, CUDA cores). Prefill: `flash_mma` (64 query rows, 16-key tiles at head dim 256, f16 hi/lo planes, double-buffered swizzled `cp.async`). KV: bf16 single plane in serving (`Layout::serving`), f16 hi/lo planes (4 bytes per element) in `Layout::contiguous`, which `bench` uses. |
| gap | Bytes per KV element: specrt 2 (serving) or 4 (bench) against vLLM's 1. At 32k context one sequence reads 2 GB (bf16) against 1 GB (fp8) of KV per token, 10 against 5 percent of the 21 GB of weights; at batch 8 that is 16 against 8 GB per step. No per-kernel number at 256 on the 27B. |
| port | fp8 KV (already #324); tensor-core decode with GQA packed on M for verify rows (3 to 4 rows x 6 heads). |
| expected win | KV halving matters from about 8k context up and dominates batched long-context decode; tensor-core decode matters only for verify rows at long context (guess). |
| cost | #324 covers it. |

### RMSNorm, rope, SwiGLU, activation quantisation

| | |
|---|---|
| vLLM | Inductor-fused Triton reductions (add + RMSNorm + FP8 quant; SwiGLU + FP8 quant; gate + FP8 quant), Triton `rotary_kernel`, CUDA `scaled_fp4_quant` (512 threads, 16 elements per thread, `cvt` e2m1 pairs, scales written swizzled for TMA). |
| specrt | Decode: the norm is folded into the GEMV prologue (`lin_in_norm`, `qkv_proj_norm`, `gate_up_act`, `lm_head_norm`), SwiGLU into the gate/up epilogue, the residual into the down epilogue, q/k norm and rope into `qk_norm_rope` / `kv_norm_rope`, the gated norm into `gdn_step`. Prefill: `rms_split` writes the hi/lo planes once; `quant_rows_fp8` and `quant_rows_fp4` read them again. |
| gap | specrt fuses more than vLLM in decode: vLLM runs a norm-quant kernel plus, for NVFP4, a zero fill and a quant kernel before each GEMM. In prefill specrt pays one extra read of the activation planes per quantised linear: about 30 MB per linear at 1,024 tokens, under 1 percent of prefill. |
| port | Quantise straight from `rms_split`, the SwiGLU epilogue and `gdn_gated_norm` when `act_quant` is on. Low value; not filed. |

### Sampling

| | |
|---|---|
| vLLM | FlashInfer `top_k_top_p_sampling_from_logits` (rejection sampling with pivots, no sort), Triton rejection sampler for MTP; the served config sets `top_k 20, top_p 0.95, temperature 1.0`. |
| specrt | `sample_rows` (`templates/ops.cu:1707`): temperature plus Gumbel-max over the whole vocabulary, one block per row; `spec_accept` for MTP. No top-k or top-p: `crates/core` and `crates/server` parse `top_p` and the engine does not use it. |
| gap | Not speed: specrt samples from a different distribution than the one vLLM serves (full vocabulary against top-20 and nucleus 0.95). Sampling time is small either way. |
| port | top-k and top-p in `sample_rows` and in `spec_accept`'s acceptance and recovery, with the Gumbel keys of #407 kept. |

### MTP drafter

| | |
|---|---|
| vLLM | `Qwen3_5MTP` compiled as its own graph, piecewise CUDA graphs, inputs of the next round built on the device by `eagle_prepare_inputs_padded_kernel` and `eagle_prepare_next_token_padded_kernel` (padded drafter batch), drafts by argmax of a full bf16 lm_head. |
| specrt | `runtime/mtp.rs`: chained MTP steps, each with its own bf16 lm_head over 248,320 rows; host uploads synchronise the stream (`up_i32`), the round is not graph-captured (#401). |
| gap | vLLM single stream: 162 tok/s at 2.79 tokens per step, about 17 ms per speculative step. specrt: 31.4 tok/s at 2.43, 77 ms per step, 1.8 plain steps against a byte ratio of about 1.32 (21.3 GB plain; plus 2 x (0.85 + 2.54) GB for two drafts). Acceptance gap tracked in #387. |
| port | The device-side round (#401, vLLM's padded drafter batch is the reference). New, not a vLLM technique: draft with a quantised head. Drafts only need an argmax, so the NVFP4 head that failed the target's accuracy gate (95.8 percent top-1 against vLLM) changes only acceptance, never output. It cuts 2 x 1.82 GB from each k = 2 step. |
| expected win | Guess: 8 to 12 percent on speculative tok/s, minus whatever acceptance drops (to be measured). |
| cost | 1 to 2 days: `runtime/mtp.rs` (`mtp_chain`, the head used by drafts), `runtime.rs` (a second head copy, 0.72 GB), `spec.rs` (`[mtp] head_quant`). |

### Launch overhead and PDL

| | |
|---|---|
| vLLM | PDL on the CUTLASS fp4 GEMMs (`gemm.run(..., enablePDL=true)`), `combo_kernels` to merge small kernels, piecewise graphs. |
| specrt | Whole decode step in one CUDA graph (better than vLLM's piecewise graphs); no PDL; on the order of 9 kernels per layer (guess from the kernel list), about 600 graph nodes per token. |
| gap | Unknown. Guess: 1 to 2 us of drain and fill per node, 0.6 to 1.2 ms of a 21 ms token. |
| port | Launch with `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` (`cuLaunchKernelEx`, in cudarc's `sys`), `griddepcontrol.wait` before the first dependent load and `griddepcontrol.launch_dependents` after the last store in the GEMV, norm and gdn templates; graph capture keeps the edges programmatic. |
| expected win | 3 to 6 percent of decode kernel time (guess). |
| cost | 2 to 3 days: a launch helper in `runtime.rs`, the PTX in `templates/*.cu`, a test that graph and eager output stay bit identical. |

## Where specrt is already ahead

- Decode fusion: norm in the GEMV prologue, SwiGLU and residual in the epilogues, the gated norm
  in `gdn_step`. vLLM runs separate norm, quant and fp4 scale-fill kernels around each GEMM.
- One CUDA graph for the whole plain decode step; vLLM is piecewise with speculation on.
- Native fp4 prefill GEMM at 315 TFLOP/s per call, and 1,875 to 1,926 tok/s at a 1,024-token
  prompt on the native path beside vLLM, against vLLM's logged 1,782 tok/s single-request median
  (different prompts, not a like-for-like number).
- Batched NVFP4/FP8 GEMV up to 16 rows reads each weight byte once: the same layout as vLLM's
  swapped 128 x 32 CUTLASS tile.
- The weight-only default is exact to the dequantised checkpoint (KL 1.2e-9); vLLM always
  quantises activations.
- Start-up: 5 to 14 s against vLLM's 153 to 185 s (`docs/vllm-jit-warmup-review.md`).

## Measured

#413, 2026-11-09 15:05 to 16:10. Both engines ran in one window beside the served vLLM (92 to 100
percent utilisation, 42 GB free, 0 errors in `journalctl -u local-llm.service` for the window).
The isolated vLLM 0.27.1 ran one after the other with specrt (not interleaved) under
`systemd-run -p MemoryMax=24G -p MemorySwapMax=0`, `nsys profile --cuda-graph-trace=node`, and
was stopped after each run. It differs from the served command in `--max-model-len 8192`,
`--max-num-seqs 8`, `--kv-cache-memory-bytes 3G`, `--gpu-memory-utilization 0.38` (0.75 does not
fit beside the served one), `--skip-mm-profiling --mm-processor-cache-gb 0
--safetensors-load-strategy lazy` (without them the 24 GB cgroup OOM-killed it), and no parsers,
middleware or prompt-detail flags. Requests were greedy, 128-token prompts, 64 new tokens.
specrt is `qwen3.8-27b-nvfp4-act.toml` (`act`) and `qwen3.8-27b-nvfp4.toml` (`default`),
three runs per `bench`.

```
flock /tmp/gpu27.lock env MTP=on  crates/specrt/scripts/vllm27b-nsys.sh vllm
flock /tmp/gpu27.lock env MTP=off crates/specrt/scripts/vllm27b-nsys.sh vllm
flock /tmp/gpu27.lock crates/specrt/scripts/vllm27b-nsys.sh specrt
python3 crates/specrt/scripts/nsys_steps.py crates/specrt/out/nsys413/<run>.sqlite <steps>
python3 crates/specrt/scripts/nsys_grid.py  crates/specrt/out/nsys413/<run>.sqlite
```

vLLM's FlashInfer GEMMs all show as `device_kernel`, so they are told apart by grid and by the
e2m1 or e4m3 element type in the demangled name. Times are medians over all calls, with the 10th
percentile beside it: both engines share the GPU, so some calls run at half speed while the
served vLLM holds a time slice, and the p10 is the unshared time. Bytes and the specrt names are
from `docs/specrt-nvfp4.md`.

### Decode kernels, batch 1, plain (MTP off), per call

| kernel | bytes | calls per token | vLLM med / p10 | specrt med / p10 | GB/s at p10, vLLM / specrt |
|---|---|---|---|---|---|
| NVFP4 `gate_up` (vLLM grid 1x188, `gate_up_act_f4`) | 100 MB | 56 | 67.1 / 66.2 us | 80.8 / 71.5 us | 1,510 / 1,399 |
| NVFP4 `down` (1x40, `down_proj_f4`) | 50 MB | 56 | 40.7 / 39.6 | 43.7 / 38.9 | 1,263 / 1,285 |
| FP8 `lin_in` (1x256, `lin_in_norm`) | 84 MB | 48 | 57.2 / 56.5 | 78.9 / 70.5 | 1,487 / 1,191 |
| FP8 `qkv` (1x224, `qkv_proj_norm`) | 73 MB | 16 | 51.7 / 51.2 | 72.2 / 64.7 | 1,426 / 1,128 |
| FP8 `lin_out`, `o` (1x80, `lin_out_proj`) | 31 MB | 64 | 24.1 / 23.6 | 25.7 / 24.1 | 1,314 / 1,286 |
| FP8 `gate_up` layers 56 to 63 (1x544, `gate_up_act`) | 178 MB | 8 | 117.5 / 116.0 | 138.0 / 128.4 | 1,534 / 1,386 |
| FP8 `down` layers 56 to 63 (`down_proj`) | 89 MB | 8 | 61.8 / 60.6 | 66.6 / 62.0 | 1,469 / 1,435 |
| bf16 `lm_head` (`cutlass_80_wmma` 8x1940, `lm_head_norm`) | 2.54 GB | 1 | 4,130 / 1,990 | 4,185 / 1,763 | 1,276 / 1,441 |
| delta-net step (`fused_recurrent_..._packed_decode`, `gdn_step`) | 6.3 MB state read and write | 48 | 7.3 / 6.9 | 9.0 / 8.0 | 911 / 786 |
| conv1d update (`_causal_conv1d_update`, `gdn_conv_step`) | small | 48 | 2.3 | 2.6 | |
| decode attention (`BatchPrefillWithPagedKV`, `attention_split`) | small | 16 | 10.9 | 11.6 | |

The `lm_head` median is bimodal in both engines (about 2.0 and 4.1 ms): the slow half is the
time slice. `gdn_step` against its floor: 6.3 MB per layer at 1,640 GB/s is 3.8 us, so 0.18 ms
per token over 48 layers. specrt is at 0.43 ms (2.3x, p10 2.0x), vLLM at 0.35 ms (1.8x). At
batch 8 specrt's step is 27.5 us against vLLM's 24.0 and a nominal floor of 30.7 us (vLLM is under
it, so part of the state is served from L2).

Batched rows (vLLM runs the same CUTLASS kernel for 1 to 24 rows; specrt's batched kernels,
medians): `gate_up` at 4 or 8 rows 67.6 us against specrt `gate_up_f4b8` 102.8 and 103.6 us (1.5x);
`down` 48.8 (p10 40.4) against 51.8 and 52.8; `lin_in` 58.7 against 72.1 and 74.4. At the 12 and
24 verify rows of MTP at batch 4 and 8, vLLM's `gate_up` stays at 67.6 us and specrt's
`gate_up_f4b8` is 104 us.

### Per token, plain decode, batch 1, wall clock against kernel sum

| | vLLM | specrt |
|---|---|---|
| wall per token (no nsys) | 36.4 ms (27.5 tok/s) | 34.0 ms (29.4 tok/s) |
| kernels per token | 1,247 | 467 |
| kernel sum, median x calls | 19.4 ms (p10 16.9) | 21.1 ms (p10 17.0) |
| kernel sum, mean x calls | 31.0 ms | 34.2 ms |
| gaps between kernels, 20 us and over | 4.9 ms | 0.5 ms |
| gaps under 20 us | 0.4 ms | 0.1 ms |
| kernels under 20 us | 988 calls, 2.9 ms | 206 calls, 1.4 ms |

The 43 ms against 21.8 ms of #363 is time slicing: the mean kernel sum is 11.6 ms (vLLM) and 13.1
ms (specrt) above the median one, and the launch gaps are 0.6 ms for specrt. vLLM's graph has
2.7x the nodes and 10x the large gaps. Beside the served vLLM both engines decode a plain token
in 34 to 36 ms, and their unshared kernel sums are level (16.9 against 17.0 ms).

### Speculative decode (MTP, k = 2)

| | vLLM | specrt `act` |
|---|---|---|
| tokens per second, batch 1, MTP off / on | 27.5 / 37.6 to 47.0 (1.4x to 1.7x, four runs) | 22.5 to 23.3 / 27.8 to 28.0 (1.2x) |
| batch 4 aggregate, off / on | 96.1 / 144.8 (1.51x) | 83.6 / 98.3 (1.18x) |
| batch 8 aggregate, off / on | 157.6 to 186.8 / 123.8 to 152.2 (0.8x) | 129.1 / 148.6 (1.15x) |
| mean acceptance length | 2.65 (632 drafts, 1,040 accepted of 1,264) | 2.37 to 2.38 |
| round at batch 1: wall | 52 ms | 84 ms |
| round: kernel median sum / mean sum | 28.6 / 42.7 ms | 29.7 / 51.6 ms |
| round: idle gaps of 20 us and over | 8.2 ms | 34.4 ms |
| kernels per round | 1,559 | 918 |

specrt's round is 41 percent idle (host synchronisation after each upload and the accept
readback, #401). vLLM's own MTP loses at batch 8 (0.81x), the same as specrt's `default` spec
(125.2 against 141.5). A round spends 3.1 `lm_head` calls (1.74 ms each, p10 1.67) and
`spec_accept` 0.7 to 1.0 ms.

### Prefill, 1,024 tokens, per call (p10)

| GEMM | FLOP | vLLM | specrt | vLLM / specrt TFLOP/s |
|---|---|---|---|---|
| NVFP4 `gate_up` | 365 G | 504 us (med 576) | 1,184 us (med 1,496) | 724 / 308 |
| NVFP4 `down` | 182 G | 187 (211) | 598 (617) | 976 / 305 |
| FP8 `lin_in` | 172 G | 408 (489) | 939 (974) | 421 / 183 |
| FP8 `lin_out` | 64 G | 142 (152) | 409 (427) | 454 / 157 |
| FP8 `qkv` | 150 G | 343 (412) | 840 (869) | 438 / 179 |
| FP8 `gate_up` | 365 G | 891 (1,017) | 2,949 (5,090) | 410 / 124 |

Per request, kernel time as median x calls, including the one decode step of the 2-token
request: 1,024 tokens vLLM 166 ms (linears and head 140, delta-net 11, attention 3) against specrt
`act` 360 ms (291, 47, 11); 4,096 tokens vLLM 518 ms (412, 42, 29) against 2,058 ms (1,638, 213,
161). Request wall for vLLM: 293 ms and 988 ms (MTP off), 328 and 1,041 ms (MTP on), about 4,000
and 4,300 tok/s once one decode step is taken off. specrt `bench` prefill: `act` 1,864 and 1,808
tok/s, `default` 649 and 750 tok/s. The FLA delta-net kernels (`chunk_scaled_dot_kkt`,
`merge_16x16_to_64x64_inverse`, `recompute_w_u`, `chunk_fwd_kernel_h`, `chunk_fwd_kernel_o`) take
10.7 and 41.6 ms against specrt's `gdn_chunk_prep` and `gdn_chunk_scan` family at 46.6 and 212.9.
Attention at 4,096 tokens: vLLM `BatchPrefillWithPagedKV` 29 ms, specrt `flash_mma` 161 ms.

### What it decides

- Decode GEMVs: vLLM's best reach 1,510 GB/s (92 percent of the 1,640 ceiling). specrt is level
  on `down` and `lm_head` and 7 to 21 percent behind on `gate_up`, `lin_in` and `qkv`. Matching
  vLLM on those is about 1.2 ms per token (6 percent), less the separate norm kernel vLLM still
  runs (5.3 us x 57).
- Launch gaps in plain decode are 0.6 ms per token for specrt: PDL (#417) cannot reach 3 percent
  at batch 1.
- The speculative round is where the wall clock goes: 34 ms of 84 ms is idle. A round at the
  kernel mean sum plus a little is 52 to 55 ms, which is vLLM's round (#401).
- The delta-net decode grid (#418) is worth 0.08 ms per token at batch 1 and 0.17 ms at batch
  8 (0.5 percent). The prefill half is 4x to 5x.
- NVFP4 and FP8 prefill GEMMs are 2x to 3x slower per call than vLLM's CUTLASS at 1,024 tokens;
  the NVFP4 ones are not covered by #416.

## Ranked ports

Expected tok/s per day of work. All wins are guesses until the measurement issue lands, except
where a number is quoted.

| rank | port | expected win | cost | issue |
|---|---|---|---|---|
| 0 | Per-kernel `nsys` of an isolated vLLM and of specrt on the 27B, same sharing | decides 3, 4, 7, 8 and the TMA GEMV row | 1 day | #413 |
| 1 | Batched fp4/fp8 GEMV to 32 and 64 rows (MTP verify at batch 8, decode at 17 to 32 slots) | MTP at batch 8 from 0.81x to 1.1 to 1.3x of plain: +40 to 60 percent there | 2 to 3 days | #414 |
| 2 | NVFP4 head for MTP drafts only | +8 to 12 percent speculative tok/s, minus any acceptance loss | 1 to 2 days | #415 |
| 3 | FP8 prefill GEMM: per-token scale, scales in the epilogue, 3 to 4 stages, `ldmatrix`, 128 x 128 tile | +12 to 18 percent native prefill | 3 to 4 days | #416 |
| 4 | PDL across the decode graph | +3 to 6 percent decode | 2 to 3 days | #417 |
| 5 | Device-side MTP round (vLLM's padded drafter batch) | about 43 against 31 tok/s single stream (#401's estimate) | in #401 | #401 (comment) |
| 6 | fp8 KV with dequant in the attention kernels | half the KV bytes; 5 percent at 32k single stream, more batched | in #324 | #324 (comment) |
| 7 | FLA-style delta-net kernels: 32-column decode grid, block-merged tril inverse in prefill | 1 to 2 percent decode, 5 to 10 percent prefill | 3 to 4 days | #418 |
| 8 | Per-shape measured tile choice (FlashInfer autotune pattern) | depends on 0 | in #408 | #408 (comment) |
| 9 | top-k and top-p sampling as served | correctness of the served distribution, not speed | 1 to 2 days | #419 |
| - | Quantise directly in the norm and SwiGLU epilogues | under 1 percent of prefill | 1 day | not filed |
| - | TMA / persistent single-row GEMV | 78 to 90 percent of the ceiling, if vLLM shows it is reachable | 3 to 5 days | #362 (comment) |

## Licence

vLLM, FlashInfer and the FLA kernels vendored in vLLM are Apache-2.0; CUTLASS is BSD-3-Clause.
None of the filed issues copies code. If one does: a file from
`vllm/third_party/flash_linear_attention/ops/` keeps its Apache-2.0 header with vLLM's and Songlin
Yang and Yu Zhang's copyright lines and the repository gets the Apache `NOTICE`; a FlashInfer
header (`flashinfer/data/include/flashinfer/gemm/*.h`) keeps NVIDIA's Apache-2.0 header; any
CUTLASS code keeps NVIDIA's BSD-3-Clause notice.
