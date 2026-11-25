# Kernel timing against FlashInfer, vLLM and cuBLAS (#1158, phase 2)

Phase 1 ([kernel-static.md](kernel-static.md)) compared compiled artefacts. This phase times the 8 rows where the static data left doubt. Each of our kernels runs against the competitor's best configuration on the same shape, data and seed, in one process on one card, alternating launch by launch.

- Date 2026-11-25 16:54 UTC, commit bf1bd3a1, GPU1 of the desktop (RTX PRO 6000 Blackwell Max-Q, 188 SMs, 1792 GB/s, idle at the start, P0, 3090 MHz max SM clock). GPU0 was serving the user's app. Both cards were otherwise quiet.
- Generated tables with every number, error, register count and the `nvidia-smi -q -d CLOCK` dump: [kernel-vs-tables.md](kernel-vs-tables.md). Raw results: [kernel-vs.json](kernel-vs.json).
- Command (about 10 minutes: CPU dump and FlashInfer JIT first, then about 5 minutes holding the GPU1 lock):

```bash
KERNEL_VS_JOBS=1 CARGO_TARGET_DIR=/mnt/data/cargo-target/inference-bench-1158 crates/specrt/scripts/kernel_vs.sh /tmp/kernel-vs docs/perf/kernel-vs-tables.md
```

`KERNEL_VS_ONLY=<regex>` picks cases by name and `KERNEL_VS_ITERS` sets the sample count. The script takes GPU1 through `~/.agents/bin/gpu-run 1 --min-free 20000` and never touches GPU0.

## How it works

| Step | File | What it does |
| --- | --- | --- |
| Cases | `crates/specrt/tests/kernel_vs.rs` | Builds the real one-layer modules (`Module::generate_kv`, max_ctx 32768, sm_120 with 188 SMs) for 27B NVFP4, the 27B with head_dim 128 at GQA 8:1 and 4:1, 122B NVFP4, Qwen3.5-4B bf16 and gpt-oss-20b, plus the standalone kir `add_rmsnorm`. It writes the source, a cubin from NVRTC and the launch geometry the runtime uses (grid, block, dynamic smem, the GEMM variant `gemm::select` picks at each m) into `KERNEL_VS_OUT`. A CPU test, not ignored, checks that the 8 rows are covered and the geometry is valid. |
| Timing | `crates/specrt/scripts/kernel_vs.py run` | Loads our cubins with `cuModuleLoadData` and launches them with `cuLaunchKernel` on torch-allocated buffers, so torch only allocates and times. Competitors: FlashInfer 0.7.3 `BatchDecodeWithPagedKVCacheWrapper` (default plan, and `use_tensor_cores=True`), `BatchPrefillWithPagedKVCacheWrapper` FA2, `mm_fp4` (CUTLASS sm120 with the autotuner, and b12x), `rmsnorm` and `fused_add_rmsnorm`; vLLM 0.34 `rms_norm` and `fused_add_rms_norm`; cuBLAS via `torch.matmul`; and the Triton grouped matmul from `kernel_static_triton.py`, best of its 6 tile configs. |
| JIT prep | `kernel_vs.py prepare` | Builds the FlashInfer attention and norm modules into a scratch `FLASHINFER_WORKSPACE_BASE` with `MAX_JOBS=2`. `~/.cache/flashinfer` is read only: the CUTLASS FP4 GEMM is loaded from its existing `fp4_gemm_cutlass_sm120.so` and never rebuilt. |
| Report | `kernel_vs.py report` | Writes the tables from the JSON. |

Measurement rules:
- Inputs are identical on both sides. KV, weights and activations come from a seeded generator, and the page table is a random permutation. Ours reads f16 KV in our page layout (`[block][kv_head][16][head_dim]`), which is FlashInfer's HND layout with page size 16, so both read the same buffer. NVFP4 GEMMs use the same packed e2m1 data and e4m3 scales. Theirs go through vLLM's `swizzle_blockscale`.
- Error is max abs over the largest reference value, checked against an f64 host reference: the full output for decode and norms, and 64 sampled rows for GEMMs and prefill. Every side of every case passes. Ours is under 1e-5 everywhere (hi/lo bf16 output planes and f32 norms). Theirs is 5e-4 (f16 output) to 7e-3 (bf16 output).
- Each side gets 200 samples per mode (1000 for cold kernels under about 25 us) after 20 warmup launches. The order of the sides is random every iteration. A GPU spin before each sample keeps host enqueue time out. Warm samples chain launches up to about 200 us and divide.
- Warm: the side ran just before, so whatever fits in the 128 MB L2 is resident. Cold: 512 MB is zeroed before the launch. Cold is the DRAM number. Warm rates above the 1792 GB/s roofline mean L2 hits.
- The centre is the 10 to 90% trimmed mean. CUDA events on this card step in about 2.05 us units, so a median of single cold launches under about 50 us cannot hold a 2% control. Medians and IQRs are in the tables.
- An ours-vs-ours control side runs in every case. Every control reads 1.00 +/- 0.02; the worst is 0.7% off. The run is refused (exit 2) otherwise.
- Verdict, as our speed over theirs (their time / ours): over 1.05 ahead, 0.95 or more on par, 0.8 to 0.95 close, under 0.8 behind.

## Results

Speed is their time over ours (above 1 we are faster). These are cold numbers, with warm in brackets where it differs. The strongest competitor is listed.

| Row | Case | Ours (us) | Theirs (us) | Ours / theirs | Verdict | Our roofline share |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 27B decode, 8 seqs, 1k | 71.4 | FI decode 32.8 | 0.46 (warm 0.34 vs FI tensor-core plan) | behind | 26% |
| 1 | 27B decode, 8 seqs, 8k | 234.5 | FI decode tensor-core plan 186.1 | 0.79 (warm 0.85) | behind (warm close) | 64% |
| 1 | 27B decode, 8 seqs, 32k | 766.0 | FI decode tensor-core plan 693.7 | 0.91 (warm 0.91) | close | 78% |
| 2 | hd128 GQA 8:1, 1k | 38.9 | FI 22.5 | 0.58 (warm 0.29) | behind | 24% |
| 2 | hd128 GQA 8:1, 8k | 126.0 | FI tensor-core plan 99.4 | 0.79 (warm 0.41) | behind | 59% |
| 2 | hd128 GQA 8:1, 32k | 405.8 | FI tensor-core plan 351.8 | 0.87 (warm 0.86) | close | 74% |
| 2 | hd128 GQA 4:1, 1k | 41.0 | FI 31.9 | 0.78 (warm 0.38) | behind | 46% |
| 2 | hd128 GQA 4:1, 8k | 209.8 | FI tensor-core plan 181.7 | 0.87 | close | 71% |
| 2 | hd128 GQA 4:1, 32k | 732.3 | FI tensor-core plan 693.6 | 0.95 (0.947) | close, at the edge of on par | 82% |
| 3 | 122B attention_split, 1 seq, 1k | 37.3 | FI tensor-core plan 14.3 | 0.38 (warm 0.32) | behind | 3% |
| 3 | 122B attention_split, 8k | 126.5 | FI tensor-core plan 24.7 | 0.20 (warm 0.11) | behind | 7% |
| 3 | 122B attention_split, 32k | 432.9 | FI tensor-core plan 65.5 | 0.15 (warm 0.08) | behind | 9% |
| 4 | 27B gate_up_f4a_big, m=512 | 349.8 | CUTLASS 171.4 | 0.49 (warm 0.57) | behind | 30% |
| 4 | 27B gate_up_f4a_big, m=2048 | 1234.5 | CUTLASS 673.8 | 0.55 (warm 0.57) | behind | 34% |
| 4 | 27B gate_up_f4a_big, m=4096 | 2417.9 | CUTLASS 1385.5 | 0.57 (warm 0.59) | behind | 34% |
| 4 | 27B down_f4a (the 64x64 variant the selector picks), m=512 | 265.7 | b12x 96.7 | 0.36 | behind | 20% |
| 4 | 27B down_f4a_big, m=2048 | 637.4 | CUTLASS 345.2 | 0.54 (warm 0.61) | behind | 33% |
| 4 | 27B down_f4a_big, m=4096 | 1278.4 | CUTLASS 641.0 | 0.50 | behind | 33% |
| 5 | 4B gate_up_mma_big, m=2048 | 913.7 | cuBLAS 719.1 | 0.79 (warm 0.84) | behind (warm close) | 48% |
| 5 | 4B gate_up_mma_big, m=4096 | 1695.4 | cuBLAS 1373.2 | 0.81 (warm 0.83); Triton 0.87 | close | 52% |
| 5 | gpt-oss gate_up_mma_big, m=2048 | 519.3 | cuBLAS 225.7 | 0.43 | behind | 30% |
| 5 | gpt-oss gate_up_mma_big, m=4096 | 1053.3 | cuBLAS 456.9 | 0.43 | behind | 29% |
| 6 | 27B flash_mma, 1k tokens | 205.7 | FA2 110.2 | 0.54 (warm 0.66) | behind | 14% |
| 6 | 27B flash_mma, 4k | 1449.3 | FA2 1085.1 | 0.75 (warm 0.73) | behind | 32% |
| 6 | 27B flash_mma, 16k | 18360.3 | FA2 16891.2 | 0.92 | close | 41% |
| 7 | rmsnorm 5120 x 4096 rows | 115.7 (warm 90.4) | FI 54.4 (warm 17.3) | 0.47 (warm 0.19) | behind | 81% cold |
| 7 | rmsnorm 2560 x 4096 | 83.6 | FI 26.6 | 0.32 | behind | 56% cold |
| 7 | add_rmsnorm 5120 x 4096 | 231.9 | FI 113.3 | 0.49 (warm 0.14) | behind | 81% cold |
| 7 | add_rmsnorm 2560 x 4096 | 120.7 | FI 54.6 | 0.45 (warm 0.16) | behind | 78% cold |
| 7 | rmsnorm and add_rmsnorm, 1 row | 6.1 to 8.2 | 4.1 to 4.6 | 0.55 to 0.75 | behind | launch bound |
| 8 | 27B gate_up_act_f4, m=1 | 79.9 (warm 65.1) | none | | | 70% cold (1256 GB/s) |
| 8 | 27B down_proj_f4, m=1 | 58.7 (warm 27.1) | none | | | 48% cold (855 GB/s) |

## Reading, row by row

1. **27B batch decode (hd256, group 6).** Static analysis left this question open: does running 64 threads per SM cost bandwidth? It does, but mostly at short context. At 32k we read 1402 GB/s cold, 78% of the roofline. FlashInfer's default CUDA-core plan reaches 1460 GB/s and its tensor-core plan 1548 GB/s, so we are 0.91 of the best. At 8k the gap is 0.79 to 0.85. At 1k we are at 0.46: the kernel takes 71 us for 37 MB, where FlashInfer takes 33 us. Short context is where low occupancy shows, because too few bytes are in flight per SM to cover latency, and there is no long stream to amortise over. That is the #1160 target, and the 1k and 8k rows are where its gain should show.
2. **Decode at head_dim 128.** Not spilling did not close the gap. At 32k we are 0.87 (GQA 8:1) and 0.95 (4:1). At 1k we are 0.58 and 0.78 cold, and 0.29 to 0.38 warm. The occupancy cause is the same as row 1 (160 threads per SM against 1024), so #1160 should cover head_dim 128 as well as 256.
3. **attention_split on the 122B (hd256, group 16).** This is the clearest gap: 0.15 to 0.38 of FlashInfer's tensor-core plan, at 3 to 9% of the roofline. The grid is `[kv_heads=2, 64 splits]`, 128 blocks of 128 threads on 188 SMs, each with 255 registers and 480 B of spill. At 32k it moves 169 GB/s cold in total. FlashInfer's own default CUDA-core plan is also slow on this shape (we are 0.19 of it at 32k against 0.15 of the tensor-core plan), so the competitor's best plan is the one to compare with. This path is the single-sequence step (`attention_split` plus `attention_reduce`) that the runtime uses for one sequence without a batch. Filed as a new gap, see the follow-ups below.
4. **NVFP4 prefill GEMM.** We run at 0.49 to 0.57 of CUTLASS for gate_up and 0.50 to 0.54 for down, with a flat 577 to 672 TFLOP/s, 33 to 38% of the dense FP4 peak. CUTLASS reaches 1000 to 1146 TFLOP/s, and b12x sits between the two. This confirms the static ranking (no TMA, no warp specialisation, 2 stages) with a number: the GEMM is about 1.8x slower than it could be. At m=512 the selector picks the small 64x64 `down_f4a` variant, at 0.36 of CUTLASS. Our epilogues do more work (fused SiLU-and-mul, hi/lo bf16 planes, f32 residual add, against a plain bf16 store), but at 17408 deep that work is a few percent, not 2x. #1162 holds this, with these numbers.
5. **bf16 GEMM, #991's claim.** On the 4B shape we are 0.79 to 0.84 of cuBLAS and 0.84 to 0.88 of Triton: close, not at or above. #991 measured `lin_in`, `qkv`, `lin_out` and `o` on the 122B at 4096 rows, where we were above cuBLAS. On gate_up at these shapes we are not. On gpt-oss we are at 0.43. The gpt-oss module sets `SPLIT=true` (hi and lo activation planes, two MMAs per k step) because its spec keeps exact bf16 activations, so it does twice the tensor work for the same output. That is a numerics choice of the spec, not tile quality: 134 TFLOP/s done twice is 268, about the 4B kernel's rate. The kernel-level #991 claim is therefore not reproduced for gate_up. Filed, see below.
6. **flash_mma prefill (27B hd256).** Static analysis said matched; timing says otherwise at short context. We are at 0.54 (1k), 0.75 (4k) and 0.92 (16k). Our side also does q norm and rope inside the kernel, which FlashInfer gets precomputed. That costs a fixed amount per query tile and explains part of the 1k gap, but not 2x. With `--split-compile=8`, `flash_mma` sometimes gets 192 B of spill instead of 32 B (see below), and then 4k runs 17% slower and 16k 22% slower. That is #1164 measured.
7. **rmsnorm and add_rmsnorm.** Cold at 4096 rows, we move our bytes at 1004 to 1451 GB/s, and FlashInfer and vLLM move theirs at 1420 to 1577 GB/s. Cold, the gap is the bytes, not the loads: we read and write f32 rows, twice the bytes, so we take about 2x. At 5120 we are already at 81% of the DRAM roofline for our bytes. At 2560 we are at 56%, so vector loads would help there. Warm, where the rows sit in L2, the load width shows: our 1733 to 2049 GB/s against their 4400 to 5700 GB/s, 0.14 to 0.23. So #1161 (128 bit loads, keep the row in smem) pays when activations are L2 resident, which is the common case in a layer (the residual was just written). The dtype choice (f32 residual stream) costs 2x cold on its own. At 1 row, everything is launch bound: 2 to 8 us against 1 to 4.6 us.
8. **NVFP4 GEMV at batch 1.** No competitor. `gate_up_act_f4` reads its 95 MB at 1256 GB/s cold (70% of the roofline), and `down_proj_f4` reads 47 MB at 855 GB/s (48%). The 32 bit weight loads from the static note show in `down_proj_f4`: 320 blocks of 16 warps for 5120 rows means one warp per row with 17408 columns, a long dependent loop per warp. Warm, it reaches 1853 GB/s, so the kernel can issue fast enough when the data is near. Cold, it does not keep enough DRAM requests in flight. Filed, see below.

## Build nondeterminism measured (#1164)

The same source compiled with `--split-compile=8`, as the runtime JIT does, gives two different binaries from run to run. Four builds in a row of the 27B module gave:

| Build | `batch_decode_attention` stack | `flash_mma` stack | hd128 decode regs |
| --- | --- | --- | --- |
| split 8, a | 248 B | 192 B | 242 |
| split 8, b | 224 B | 32 B | 167 |
| split 8, c | 224 B | 32 B | 167 |
| split 8, d | 248 B | 192 B | 242 |
| split 1 (twice, identical cubins) | 224 B | 32 B | 168 |

Timed, the bad build costs 17 to 22% on `flash_mma` at 4k and 16k tokens (1709 against 1468 us, and 22333 against 18332 us, warm). The hd128 decode with 242 registers is faster at 1k and 8k (25 against 32 us). So the module's speed depends on a coin flip at JIT time, and the runtime's own cache key does not change between the two. The run above uses `KERNEL_VS_JOBS=1` so its numbers are stable. The split-8 timings are in a comment on #1164.

## What changes in the phase 1 ranking

- Rank 1 (decode attention) holds, and now has a size: 0.46 at 1k down to 0.91 at 32k on the 27B. The single-sequence `attention_split` path on hd256 with a large group is far worse (0.15 to 0.38) and needs its own fix.
- Rank 2 (NVFP4 GEMM) holds at about 0.55 of CUTLASS, and is the largest prefill gap in time.
- "Matched" in phase 1 for `flash_mma` and the bf16 GEMM was wrong at short context and small shapes. `flash_mma` is 0.54 to 0.92, and gate_up bf16 is 0.79 to 0.88 (0.43 with the split planes).
- Rank 4 (vector loads) matters warm, not cold. Cold, the f32 row is the whole gap.
