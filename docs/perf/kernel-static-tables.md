
## sm_120: RTX PRO 6000 Blackwell Max-Q (desktop)

Target `target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]`, NVRTC 13.2 (--std=c++17 no --split-compile (it makes register allocation nondeterministic)). Peak numbers: nvidia.com RTX PRO 6000 Blackwell Max-Q page: 1792 GB/s, 3511 sparse FP4 TOPS (dense FP4 1755, bf16 dense taken as a quarter of that), 110 TFLOP/s FP32; 188 SMs and 3090 MHz max clock from nvidia-smi.

- `glm4.5-air` does not compile for sm_120: ptxas error   : Entry function 'moe_route_logits' uses too much shared data (0x10080 bytes, 0xc000 max)


### bf16_gemm

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4b-bf16 | `down_mma_big` | 256 | 92 | 33792 | 0 | 2 (regs) | 512 | 129, 523 | ldgsts 4 ldsm 12 lds 3 mma 32 bar 3 | Triton 3.8.0 `matmul` | 256 | 156 | 66560 | 0 | 256 | 140, 840 | ldgsts 8 ldsm 24 lds 3 mma 64 bar 5 | matches |
| 4b-bf16 | `gate_up_mma_big` | 256 | 166 | 50176 | 0 | 1 (regs) | 256 | 177, 913 | ldgsts 6 ldsm 16 lds 3 mma 64 bar 3 | Triton 3.8.0 `matmul` | 256 | 156 | 66560 | 0 | 256 | 140, 840 | ldgsts 8 ldsm 24 lds 3 mma 64 bar 5 | matches |
| gpt-oss-20b | `down_mma_big` | 256 | 127 | 50176 | 0 | 2 (regs) | 512 | 256, 1012 | ldgsts 6 ldsm 20 lds 3 mma 64 fma 64 bar 3 | Triton 3.8.0 `matmul` | 256 | 156 | 66560 | 0 | 256 | 140, 840 | ldgsts 8 ldsm 24 lds 3 mma 64 bar 5 | matches |
| gpt-oss-20b | `gate_up_mma_big` | 256 | 128 | 50176 | 0 | 2 (regs) | 512 | 254, 986 | ldgsts 6 ldsm 20 lds 3 mma 64 fma 64 bar 3 | Triton 3.8.0 `matmul` | 256 | 156 | 66560 | 0 | 256 | 140, 840 | ldgsts 8 ldsm 24 lds 3 mma 64 bar 5 | matches |

### bf16_gemv

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4b-bf16 | `down_proj` | 256 | 41 | 1024 | 0 | 5 (regs) | 1280 | 189, 328 | ldg128 24 fma 96 | cuBLAS GEMV is closed source; no static artefact | | | | | | | | |
| 4b-bf16 | `gate_up_act` | 256 | 47 | 11392 | 0 | 5 (regs) | 1280 | 39, 82 | ldg32 8 fma 8 | cuBLAS GEMV is closed source; no static artefact | | | | | | | | |
| 4b-bf16 | `qkv_proj_norm` | 256 | 46 | 11392 | 0 | 5 (regs) | 1280 | 39, 80 | ldg32 8 fma 8 | cuBLAS GEMV is closed source; no static artefact | | | | | | | | |
| gpt-oss-20b | `down_proj` | 256 | 41 | 1024 | 0 | 5 (regs) | 1280 | 189, 328 | ldg128 24 fma 96 | cuBLAS GEMV is closed source; no static artefact | | | | | | | | |
| gpt-oss-20b | `gate_up_act` | 256 | 47 | 12672 | 0 | 5 (regs) | 1280 | 190, 387 | ldg128 8 lds 16 fma 96 | cuBLAS GEMV is closed source; no static artefact | | | | | | | | |
| gpt-oss-20b | `qkv_proj_norm` | 256 | 44 | 12672 | 0 | 5 (regs) | 1280 | 190, 326 | ldg128 8 lds 16 fma 96 | cuBLAS GEMV is closed source; no static artefact | | | | | | | | |

### decode_attention

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `attention_reduce` | 256 | 40 | 1296 | 0 | 6 (regs) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| 122b-e128 | `attention_split` | 128 | 255 | 5632 | 480 | 2 (regs) | 256 | 7854, 23602 | lds 192 fma 3967 mufu 186 shfl 320 bar 353 local 496 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 512 | 72 | 1024 | 0 | 512 | 2041, 2606 | ldg32 1 ldgsts 4 lds 72 fma 821 mufu 34 shfl 160 bar 16 | behind: spills 480 B |
| 122b-e128 | `batch_decode_attention` | 32 | 255 | 33792 | 216 | 2 (smem) | 64 | 999, 3462 | ldg32 7 ldgsts 7 ldsm 32 lds 12 mma 128 fma 190 mufu 10 shfl 8 bar 4 local 46 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 512 | 72 | 1024 | 0 | 512 | 2041, 2606 | ldg32 1 ldgsts 4 lds 72 fma 821 mufu 34 shfl 160 bar 16 | behind: spills 216 B, 64 threads/SM vs 512 |
| 122b-e128 | `batch_decode_combine` | 128 | 40 | 1024 | 0 | 12 (regs) | 1536 | 80, 178 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |
| 122b-nvfp4 | `attention_reduce` | 256 | 40 | 1296 | 0 | 6 (regs) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| 122b-nvfp4 | `attention_split` | 128 | 255 | 5632 | 480 | 2 (regs) | 256 | 7854, 23602 | lds 192 fma 3967 mufu 186 shfl 320 bar 353 local 496 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 512 | 72 | 1024 | 0 | 512 | 2041, 2606 | ldg32 1 ldgsts 4 lds 72 fma 821 mufu 34 shfl 160 bar 16 | behind: spills 480 B |
| 122b-nvfp4 | `batch_decode_attention` | 32 | 255 | 33792 | 216 | 2 (smem) | 64 | 999, 3462 | ldg32 7 ldgsts 7 ldsm 32 lds 12 mma 128 fma 190 mufu 10 shfl 8 bar 4 local 46 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 512 | 72 | 1024 | 0 | 512 | 2041, 2606 | ldg32 1 ldgsts 4 lds 72 fma 821 mufu 34 shfl 160 bar 16 | behind: spills 216 B, 64 threads/SM vs 512 |
| 122b-nvfp4 | `batch_decode_combine` | 128 | 40 | 1024 | 0 | 12 (regs) | 1536 | 80, 178 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |
| 27b-nvfp4 | `attention_reduce` | 256 | 40 | 1296 | 0 | 6 (regs) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| 27b-nvfp4 | `attention_split` | 128 | 167 | 5312 | 0 | 3 (regs) | 384 | 2656, 7503 | lds 72 fma 1396 mufu 66 shfl 120 bar 133 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 192 | 64 | 1024 | 0 | 960 | 833, 1260 | ldg32 1 ldgsts 4 lds 32 fma 321 mufu 14 shfl 60 bar 16 | behind: 384 threads/SM vs 960 |
| 27b-nvfp4 | `batch_decode_attention` | 32 | 255 | 33792 | 224 | 2 (smem) | 64 | 986, 3347 | ldg32 7 ldgsts 7 ldsm 32 lds 12 mma 128 fma 190 mufu 10 shfl 8 bar 4 local 42 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 192 | 64 | 1024 | 0 | 960 | 833, 1260 | ldg32 1 ldgsts 4 lds 32 fma 321 mufu 14 shfl 60 bar 16 | behind: spills 224 B, 64 threads/SM vs 960 |
| 27b-nvfp4 | `batch_decode_combine` | 128 | 40 | 1024 | 0 | 12 (regs) | 1536 | 80, 178 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |
| 27b-nvfp4-fp8kv | `attention_reduce` | 256 | 40 | 1296 | 0 | 6 (regs) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| 27b-nvfp4-fp8kv | `attention_split` | 128 | 204 | 5312 | 0 | 2 (regs) | 256 | 2647, 6608 | lds 72 fma 1394 mufu 66 shfl 120 bar 133 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 80 | 1024 | 0 | 768 | 2081, 2752 | ldg32 1 ldgsts 4 lds 32 fma 901 mufu 14 shfl 48 bar 16 | behind: 256 threads/SM vs 768 |
| 27b-nvfp4-fp8kv | `batch_decode_attention` | 32 | 255 | 33792 | 208 | 2 (smem) | 64 | 2200, 7413 | ldg32 10 ldgsts 13 ldsm 32 lds 214 mma 128 fma 206 mufu 10 shfl 8 bar 4 local 54 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 80 | 1024 | 0 | 768 | 2081, 2752 | ldg32 1 ldgsts 4 lds 32 fma 901 mufu 14 shfl 48 bar 16 | behind: spills 208 B, 64 threads/SM vs 768 |
| 27b-nvfp4-fp8kv | `batch_decode_combine` | 128 | 40 | 1024 | 0 | 12 (regs) | 1536 | 80, 178 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |
| 4b-bf16 | `attention_reduce` | 256 | 40 | 1296 | 0 | 6 (regs) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| 4b-bf16 | `attention_split` | 128 | 127 | 5248 | 0 | 4 (regs) | 512 | 1720, 4834 | lds 48 fma 884 mufu 42 shfl 80 bar 89 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 56 | 1024 | 0 | 1152 | 587, 957 | ldg32 1 ldgsts 4 lds 24 fma 220 mufu 10 shfl 40 bar 16 | behind: 512 threads/SM vs 1152 |
| 4b-bf16 | `batch_decode_attention` | 32 | 255 | 33792 | 208 | 2 (smem) | 64 | 980, 3341 | ldg32 7 ldgsts 7 ldsm 32 lds 12 mma 128 fma 190 mufu 10 shfl 8 bar 4 local 42 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 56 | 1024 | 0 | 1152 | 587, 957 | ldg32 1 ldgsts 4 lds 24 fma 220 mufu 10 shfl 40 bar 16 | behind: spills 208 B, 64 threads/SM vs 1152 |
| 4b-bf16 | `batch_decode_combine` | 128 | 40 | 1024 | 0 | 12 (regs) | 1536 | 80, 178 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |
| gpt-oss-20b | `attention_reduce` | 64 | 40 | 1296 | 0 | 24 (threads) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| gpt-oss-20b | `attention_split` | 128 | 157 | 2304 | 0 | 3 (regs) | 384 | 2864, 8045 | lds 96 fma 1530 mufu 90 shfl 160 bar 177 | no competitor built for this target | | | | | | | | |
| gpt-oss-20b | `batch_decode_attention` | 32 | 128 | 9216 | 0 | 10 (smem) | 320 | 583, 1428 | ldg32 7 ldgsts 7 ldsm 8 lds 12 mma 32 fma 94 mufu 10 shfl 8 bar 4 | no competitor built for this target | | | | | | | | |
| gpt-oss-20b | `batch_decode_combine` | 32 | 40 | 1024 | 0 | 32 (blocks) | 1024 | 80, 182 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |
| hd128-gqa4 | `attention_reduce` | 128 | 40 | 1296 | 0 | 12 (regs) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| hd128-gqa4 | `attention_split` | 128 | 72 | 3200 | 0 | 7 (regs) | 896 | 1345, 4217 | lds 48 fma 713 mufu 42 shfl 80 bar 89 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 63 | 1024 | 0 | 1024 | 572, 931 | ldg32 1 ldgsts 4 lds 24 fma 212 mufu 10 shfl 32 bar 16 | matches |
| hd128-gqa4 | `batch_decode_attention` | 32 | 168 | 17408 | 0 | 5 (smem) | 160 | 729, 2095 | ldg32 7 ldgsts 7 ldsm 16 lds 12 mma 64 fma 126 mufu 10 shfl 8 bar 4 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 63 | 1024 | 0 | 1024 | 572, 931 | ldg32 1 ldgsts 4 lds 24 fma 212 mufu 10 shfl 32 bar 16 | behind: 160 threads/SM vs 1024 |
| hd128-gqa4 | `batch_decode_combine` | 64 | 40 | 1024 | 0 | 24 (threads) | 1536 | 80, 178 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |
| hd128-gqa8 | `attention_reduce` | 128 | 40 | 1296 | 0 | 12 (regs) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| hd128-gqa8 | `attention_split` | 128 | 128 | 3328 | 0 | 4 (regs) | 512 | 2807, 8790 | lds 96 fma 1535 mufu 90 shfl 160 bar 177 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 64 | 1024 | 0 | 1024 | 1037, 1585 | ldg32 1 ldgsts 4 lds 41 fma 404 mufu 18 shfl 64 bar 16 | matches |
| hd128-gqa8 | `batch_decode_attention` | 32 | 168 | 17408 | 0 | 5 (smem) | 160 | 719, 2078 | ldg32 7 ldgsts 7 ldsm 16 lds 12 mma 64 fma 126 mufu 10 shfl 8 bar 4 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 64 | 1024 | 0 | 1024 | 1037, 1585 | ldg32 1 ldgsts 4 lds 41 fma 404 mufu 18 shfl 64 bar 16 | behind: 160 threads/SM vs 1024 |
| hd128-gqa8 | `batch_decode_combine` | 64 | 40 | 1024 | 0 | 24 (threads) | 1536 | 80, 178 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |

### fused_add_rmsnorm

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hidden-2560 | `add_rmsnorm_2560` | 1024 | 35 | 1152 | 0 | 1 (regs) | 1024 | 56, 153 | ldg32 8 fma 10 | FlashInfer 0.7.3 `flashinfer4norm21FusedAddRMSNormKernelILj8E13__n` | 320 | 40 | 1024 | 0 | 1280 | 299, 674 | ldg128 3 lds 7 fma 86 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `add_rmsnorm` | 256 | 40 | 1056 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm25fused_add_rms_norm_kernelIN3c108BFloat16EL` | 320 | 36 | 1192 | 0 | 1280 | 140, 255 | ldg128 4 fma 35 |  |
| hidden-5120 | `add_rmsnorm_5120` | 1024 | 32 | 1152 | 0 | 1 (threads) | 1024 | 56, 153 | ldg32 8 fma 10 | FlashInfer 0.7.3 `flashinfer4norm21FusedAddRMSNormKernelILj8E13__n` | 640 | 40 | 1024 | 0 | 1280 | 299, 674 | ldg128 3 lds 7 fma 86 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `add_rmsnorm` | 256 | 40 | 1056 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm25fused_add_rms_norm_kernelIN3c108BFloat16EL` | 640 | 36 | 1192 | 0 | 1280 | 140, 255 | ldg128 4 fma 35 |  |

### nvfp4_gemm

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `down_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 223, 860 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 122b-e128 | `gate_up_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 232, 866 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 122b-e128 | `quant_rows_fp4` | 256 | 40 | 1024 | 0 | 6 (regs) | 1536 | 1, 1 |  | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 122b-nvfp4 | `down_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 223, 860 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 122b-nvfp4 | `gate_up_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 232, 866 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 122b-nvfp4 | `quant_rows_fp4` | 256 | 40 | 1024 | 0 | 6 (regs) | 1536 | 1, 1 |  | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 27b-nvfp4 | `down_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 223, 860 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 27b-nvfp4 | `gate_up_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 232, 866 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 27b-nvfp4 | `quant_rows_fp4` | 256 | 40 | 1024 | 0 | 6 (regs) | 1536 | 1, 1 |  | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 27b-nvfp4-fp8kv | `down_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 223, 860 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 27b-nvfp4-fp8kv | `gate_up_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 232, 866 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 27b-nvfp4-fp8kv | `quant_rows_fp4` | 256 | 40 | 1024 | 0 | 6 (regs) | 1536 | 1, 1 |  | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| hd128-gqa4 | `down_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 223, 860 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| hd128-gqa4 | `gate_up_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 232, 866 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| hd128-gqa4 | `quant_rows_fp4` | 256 | 40 | 1024 | 0 | 6 (regs) | 1536 | 1, 1 |  | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| hd128-gqa8 | `down_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 223, 860 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| hd128-gqa8 | `gate_up_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 232, 866 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| hd128-gqa8 | `quant_rows_fp4` | 256 | 40 | 1024 | 0 | 6 (regs) | 1536 | 1, 1 |  | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |

### nvfp4_gemv

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `down_proj_f4` | 512 | 40 | 1024 | 0 | 3 (regs) | 1536 | 1, 1 |  | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| 122b-e128 | `gate_up_act_f4` | 512 | 40 | 13440 | 0 | 3 (regs) | 1536 | 447, 629 | ldg64 4 ldg32 4 lds 32 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| 122b-nvfp4 | `down_proj_f4` | 512 | 40 | 1024 | 0 | 3 (regs) | 1536 | 1, 1 |  | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| 122b-nvfp4 | `gate_up_act_f4` | 512 | 40 | 13440 | 0 | 3 (regs) | 1536 | 447, 629 | ldg64 4 ldg32 4 lds 32 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| 27b-nvfp4 | `down_proj_f4` | 512 | 38 | 1024 | 0 | 3 (regs) | 1536 | 503, 935 | ldg64 4 ldg32 68 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| 27b-nvfp4 | `gate_up_act_f4` | 512 | 39 | 21632 | 0 | 3 (regs) | 1536 | 447, 612 | ldg64 4 ldg32 4 lds 32 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| 27b-nvfp4-fp8kv | `down_proj_f4` | 512 | 38 | 1024 | 0 | 3 (regs) | 1536 | 503, 935 | ldg64 4 ldg32 68 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| 27b-nvfp4-fp8kv | `gate_up_act_f4` | 512 | 39 | 21632 | 0 | 3 (regs) | 1536 | 447, 612 | ldg64 4 ldg32 4 lds 32 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| hd128-gqa4 | `down_proj_f4` | 512 | 38 | 1024 | 0 | 3 (regs) | 1536 | 503, 935 | ldg64 4 ldg32 68 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| hd128-gqa4 | `gate_up_act_f4` | 512 | 39 | 21632 | 0 | 3 (regs) | 1536 | 447, 612 | ldg64 4 ldg32 4 lds 32 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| hd128-gqa8 | `down_proj_f4` | 512 | 38 | 1024 | 0 | 3 (regs) | 1536 | 503, 935 | ldg64 4 ldg32 68 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| hd128-gqa8 | `gate_up_act_f4` | 512 | 39 | 21632 | 0 | 3 (regs) | 1536 | 447, 612 | ldg64 4 ldg32 4 lds 32 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |

### prefill_attention

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `flash_mma` | 256 | 255 | 99328 | 32 | 1 (regs) | 256 | 913, 2496 | ldg32 8 ldgsts 8 ldsm 64 lds 3 mma 128 fma 206 mufu 18 shfl 8 bar 7 | FlashInfer 0.7.3 `flashinfer34BatchPrefillWithPagedKVCacheKernelIL` | 128 | 255 | 50176 | 0 | 256 | 492, 1301 | ldg32 1 ldgsts 8 ldsm 48 lds 3 mma 65 fma 156 mufu 10 shfl 4 bar 8 | behind: spills 32 B |
| 122b-nvfp4 | `flash_mma` | 256 | 255 | 99328 | 32 | 1 (regs) | 256 | 913, 2496 | ldg32 8 ldgsts 8 ldsm 64 lds 3 mma 128 fma 206 mufu 18 shfl 8 bar 7 | FlashInfer 0.7.3 `flashinfer34BatchPrefillWithPagedKVCacheKernelIL` | 128 | 255 | 50176 | 0 | 256 | 492, 1301 | ldg32 1 ldgsts 8 ldsm 48 lds 3 mma 65 fma 156 mufu 10 shfl 4 bar 8 | behind: spills 32 B |
| 27b-nvfp4 | `flash_mma` | 256 | 255 | 99328 | 32 | 1 (regs) | 256 | 916, 2506 | ldg32 8 ldgsts 8 ldsm 64 lds 3 mma 128 fma 206 mufu 18 shfl 8 bar 7 | FlashInfer 0.7.3 `flashinfer34BatchPrefillWithPagedKVCacheKernelIL` | 128 | 255 | 50176 | 0 | 256 | 492, 1301 | ldg32 1 ldgsts 8 ldsm 48 lds 3 mma 65 fma 156 mufu 10 shfl 4 bar 8 | behind: spills 32 B |
| 27b-nvfp4-fp8kv | `flash_mma` | 256 | 255 | 100096 | 32 | 1 (regs) | 256 | 1584, 3483 | ldg32 13 ldgsts 16 ldsm 64 lds 25 mma 128 fma 334 mufu 18 shfl 8 bar 9 | FlashInfer 0.7.3 `flashinfer34BatchPrefillWithPagedKVCacheKernelIL` | 128 | 255 | 50176 | 0 | 256 | 492, 1301 | ldg32 1 ldgsts 8 ldsm 48 lds 3 mma 65 fma 156 mufu 10 shfl 4 bar 8 | behind: spills 32 B |
| 4b-bf16 | `flash_mma` | 256 | 255 | 99328 | 32 | 1 (regs) | 256 | 913, 2496 | ldg32 8 ldgsts 8 ldsm 64 lds 3 mma 128 fma 206 mufu 18 shfl 8 bar 7 | FlashInfer 0.7.3 `flashinfer34BatchPrefillWithPagedKVCacheKernelIL` | 128 | 255 | 50176 | 0 | 256 | 492, 1301 | ldg32 1 ldgsts 8 ldsm 48 lds 3 mma 65 fma 156 mufu 10 shfl 4 bar 8 | behind: spills 32 B |
| gpt-oss-20b | `flash_mma` | 128 | 116 | 17408 | 32 | 4 (regs) | 512 | 521, 1037 | ldg32 4 ldgsts 4 ldsm 16 lds 3 mma 32 fma 110 mufu 18 shfl 8 bar 7 | no competitor built for this target | | | | | | | | |
| hd128-gqa4 | `flash_mma` | 128 | 166 | 33792 | 32 | 2 (smem) | 256 | 665, 1706 | ldg32 8 ldgsts 8 ldsm 32 lds 3 mma 64 fma 142 mufu 18 shfl 8 bar 8 | FlashInfer 0.7.3 `flashinfer34BatchPrefillWithPagedKVCacheKernelIL` | 128 | 246 | 50176 | 0 | 256 | 793, 2169 | ldg32 3 ldgsts 8 ldsm 48 lds 6 mma 132 fma 216 mufu 36 shfl 8 bar 8 | behind: spills 32 B |
| hd128-gqa8 | `flash_mma` | 128 | 166 | 33792 | 32 | 2 (smem) | 256 | 665, 1706 | ldg32 8 ldgsts 8 ldsm 32 lds 3 mma 64 fma 142 mufu 18 shfl 8 bar 8 | FlashInfer 0.7.3 `flashinfer34BatchPrefillWithPagedKVCacheKernelIL` | 128 | 246 | 50176 | 0 | 256 | 793, 2169 | ldg32 3 ldgsts 8 ldsm 48 lds 6 mma 132 fma 216 mufu 36 shfl 8 bar 8 | behind: spills 32 B |

### rmsnorm

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 384 | 40 | 1024 | 0 | 1536 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rmsnorm` | 256 | 39 | 1056 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 384 | 40 | 1192 | 0 | 1536 | 119, 183 | ldg128 4 fma 32 |  |
| 122b-nvfp4 | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 384 | 40 | 1024 | 0 | 1536 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rmsnorm` | 256 | 39 | 1056 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 384 | 40 | 1192 | 0 | 1536 | 119, 183 | ldg128 4 fma 32 |  |
| 27b-nvfp4 | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 640 | 40 | 1024 | 0 | 1280 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rmsnorm` | 256 | 44 | 1056 | 0 | 1280 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 640 | 40 | 1192 | 0 | 1280 | 119, 183 | ldg128 4 fma 32 |  |
| 27b-nvfp4-fp8kv | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 640 | 40 | 1024 | 0 | 1280 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rmsnorm` | 256 | 44 | 1056 | 0 | 1280 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 640 | 40 | 1192 | 0 | 1280 | 119, 183 | ldg128 4 fma 32 |  |
| 4b-bf16 | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 320 | 40 | 1024 | 0 | 1280 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rmsnorm` | 256 | 39 | 1056 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 320 | 40 | 1192 | 0 | 1280 | 119, 183 | ldg128 4 fma 32 |  |
| gpt-oss-20b | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 360 | 40 | 1024 | 0 | 1536 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 360 | 40 | 1192 | 0 | 1536 | 119, 183 | ldg128 4 fma 32 |  |
| hd128-gqa4 | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 640 | 40 | 1024 | 0 | 1280 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rmsnorm` | 256 | 44 | 1056 | 0 | 1280 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 640 | 40 | 1192 | 0 | 1280 | 119, 183 | ldg128 4 fma 32 |  |
| hd128-gqa8 | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 640 | 40 | 1024 | 0 | 1280 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rmsnorm` | 256 | 44 | 1056 | 0 | 1280 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 640 | 40 | 1192 | 0 | 1280 | 119, 183 | ldg128 4 fma 32 |  |

### rotary

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `qk_norm_rope` | 256 | 36 | 2176 | 32 | 6 (regs) | 1536 | 445, 1653 | ldg32 7 lds 12 fma 93 mufu 2 shfl 5 bar 10 local 10 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rotary` | 128 | 18 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |
| 122b-nvfp4 | `qk_norm_rope` | 256 | 36 | 2176 | 32 | 6 (regs) | 1536 | 445, 1653 | ldg32 7 lds 12 fma 93 mufu 2 shfl 5 bar 10 local 10 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rotary` | 128 | 18 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |
| 27b-nvfp4 | `qk_norm_rope` | 256 | 35 | 2176 | 32 | 6 (regs) | 1536 | 410, 1541 | ldg32 4 lds 12 fma 93 mufu 2 shfl 5 bar 10 local 10 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rotary` | 128 | 18 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |
| 27b-nvfp4-fp8kv | `qk_norm_rope` | 256 | 34 | 2176 | 32 | 6 (regs) | 1536 | 567, 2075 | ldg32 5 lds 19 fma 141 mufu 3 shfl 10 bar 12 local 10 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rotary` | 128 | 18 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |
| 4b-bf16 | `qk_norm_rope` | 256 | 36 | 2176 | 32 | 6 (regs) | 1536 | 445, 1653 | ldg32 7 lds 12 fma 93 mufu 2 shfl 5 bar 10 local 10 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rotary` | 128 | 18 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |
| gpt-oss-20b | `qk_norm_rope` | 64 | 28 | 1280 | 32 | 24 (threads) | 1536 | 11, 23 | ldg32 1 fma 1 local 1 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |
| hd128-gqa4 | `qk_norm_rope` | 128 | 35 | 1664 | 32 | 12 (regs) | 1536 | 410, 1541 | ldg32 4 lds 12 fma 93 mufu 2 shfl 5 bar 10 local 10 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rotary` | 128 | 18 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |
| hd128-gqa8 | `qk_norm_rope` | 128 | 35 | 1664 | 32 | 12 (regs) | 1536 | 410, 1541 | ldg32 4 lds 12 fma 93 mufu 2 shfl 5 bar 10 local 10 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rotary` | 128 | 18 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |

### silu_and_mul

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `silu_and_mul` | 128 | 31 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |
| 122b-nvfp4 | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `silu_and_mul` | 128 | 31 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |
| 27b-nvfp4 | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `silu_and_mul` | 128 | 31 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |
| 27b-nvfp4-fp8kv | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `silu_and_mul` | 128 | 31 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |
| 4b-bf16 | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `silu_and_mul` | 128 | 31 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |
| gpt-oss-20b | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |
| hd128-gqa4 | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `silu_and_mul` | 128 | 31 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |
| hd128-gqa8 | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `silu_and_mul` | 128 | 31 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |

### topk_routing

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `moe_route` | 32 | 21 | 2112 | 0 | 32 (smem) | 1024 | 134, 440 | lds 10 fma 2 shfl 10 | Triton 3.8.0 `softmax_topk` | 128 | 24 | 1056 | 0 | 1536 | 1, 1 |  | behind: 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm3moe10topkGatingILi8ELi128ELi4ELi16ELi32Ei13` | 128 | 40 | 0 | 0 | 1536 | 143, 429 | ldg32 1 fma 1 shfl 12 |  |
| 122b-e128 | `moe_route_logits` | 256 | 72 | 50304 | 0 | 1 (smem) | 256 | 232, 818 | ldg128 1 ldg32 14 lds 8 fma 67 mufu 1 shfl 5 bar 2 | no competitor built for this target | | | | | | | | |
| 122b-nvfp4 | `moe_route` | 32 | 26 | 3136 | 0 | 24 (smem) | 768 | 101, 333 | lds 10 fma 3 shfl 10 | Triton 3.8.0 `softmax_topk` | 128 | 28 | 1056 | 0 | 1536 | 1, 1 |  | behind: 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm3moe10topkGatingILi8ELi128ELi4ELi16ELi32Ei13` | 128 | 40 | 0 | 0 | 1536 | 143, 429 | ldg32 1 fma 1 shfl 12 |  |
| 122b-nvfp4 | `moe_route_logits` | 256 | 72 | 50304 | 0 | 1 (smem) | 256 | 232, 818 | ldg128 1 ldg32 14 lds 8 fma 67 mufu 1 shfl 5 bar 2 | no competitor built for this target | | | | | | | | |
| gpt-oss-20b | `moe_route` | 32 | 20 | 1312 | 0 | 32 (blocks) | 1024 | 1, 1 |  | Triton 3.8.0 `softmax_topk` | 128 | 22 | 0 | 0 | 1536 | 1, 1 |  | behind: 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm3moe10topkGatingILi8ELi128ELi4ELi16ELi32Ei13` | 128 | 40 | 0 | 0 | 1536 | 143, 429 | ldg32 1 fma 1 shfl 12 |  |
| gpt-oss-20b | `moe_route_logits` | 256 | 72 | 47232 | 0 | 2 (smem) | 512 | 232, 814 | ldg128 1 ldg32 14 lds 8 fma 67 mufu 1 shfl 5 bar 2 | no competitor built for this target | | | | | | | | |

### Roofline (algorithmic bytes and FLOPs, peak numbers above)

| family | spec | kernel | shape | MB | GFLOP | FLOP/B | mem us | math us | bound | blocks | waves |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bf16_gemm | 4b-bf16 | `down_mma_big` | m=4096 n=2560 k=9216 | 240.12 | 193.274 | 804.9 | 134.00 | 440.26 | compute bf16 | 640 | 1.70 |
| bf16_gemm | 4b-bf16 | `gate_up_mma_big` | m=4096 n=18432 k=2560 | 438.30 | 386.547 | 881.9 | 244.59 | 880.52 | compute bf16 | 2304 | 12.26 |
| bf16_gemm | gpt-oss-20b | `down_mma_big` | m=4096 n=2880 k=2880 | 110.96 | 67.948 | 612.4 | 61.92 | 154.78 | compute bf16 | 736 | 1.96 |
| bf16_gemm | gpt-oss-20b | `gate_up_mma_big` | m=4096 n=5760 k=2880 | 174.74 | 135.895 | 777.7 | 97.51 | 309.56 | compute bf16 | 1440 | 3.83 |
| bf16_gemv | 4b-bf16 | `down_proj` | m=1 n=2560 k=9216 | 47.23 | 0.047 | 1.0 | 26.36 | 0.43 | memory | 320 | 0.34 |
| bf16_gemv | 4b-bf16 | `gate_up_act` | m=1 n=18432 k=2560 | 94.46 | 0.094 | 1.0 | 52.71 | 0.86 | memory | 1152 | 1.23 |
| bf16_gemv | 4b-bf16 | `qkv_proj_norm` | m=1 n=10240 k=2560 | 52.48 | 0.052 | 1.0 | 29.29 | 0.48 | memory | 1280 | 1.36 |
| bf16_gemv | gpt-oss-20b | `down_proj` | m=1 n=2880 k=2880 | 16.61 | 0.017 | 1.0 | 9.27 | 0.15 | memory | 360 | 0.38 |
| bf16_gemv | gpt-oss-20b | `gate_up_act` | m=1 n=5760 k=2880 | 33.21 | 0.033 | 1.0 | 18.53 | 0.30 | memory | 360 | 0.38 |
| bf16_gemv | gpt-oss-20b | `qkv_proj_norm` | m=1 n=5120 k=2880 | 29.52 | 0.029 | 1.0 | 16.48 | 0.27 | memory | 640 | 0.68 |
| decode_attention | 122b-e128 | `attention_reduce` | seqs=1 ctx=1024 | 2.11 | 0.002 | 0.7 | 1.18 | 0.01 | memory | 32 | 0.03 |
| decode_attention | 122b-e128 | `attention_reduce` | seqs=1 ctx=32768 | 2.11 | 0.002 | 0.7 | 1.18 | 0.01 | memory | 32 | 0.03 |
| decode_attention | 122b-e128 | `attention_reduce` | seqs=1 ctx=8192 | 2.11 | 0.002 | 0.7 | 1.18 | 0.01 | memory | 32 | 0.03 |
| decode_attention | 122b-e128 | `attention_split` | seqs=1 ctx=1024 | 2.10 | 0.034 | 16.0 | 1.17 | 0.31 | memory | 128 | 0.34 |
| decode_attention | 122b-e128 | `attention_split` | seqs=1 ctx=32768 | 67.11 | 1.074 | 16.0 | 37.45 | 9.76 | memory | 128 | 0.34 |
| decode_attention | 122b-e128 | `attention_split` | seqs=1 ctx=8192 | 16.78 | 0.268 | 16.0 | 9.36 | 2.44 | memory | 128 | 0.34 |
| decode_attention | 122b-e128 | `batch_decode_attention` | seqs=8 ctx=1024 | 16.78 | 0.268 | 16.0 | 9.36 | 2.44 | memory | 1024 | 2.72 |
| decode_attention | 122b-e128 | `batch_decode_attention` | seqs=8 ctx=32768 | 536.87 | 8.590 | 16.0 | 299.59 | 78.09 | memory | 1024 | 2.72 |
| decode_attention | 122b-e128 | `batch_decode_attention` | seqs=8 ctx=8192 | 134.22 | 2.147 | 16.0 | 74.90 | 19.52 | memory | 1024 | 2.72 |
| decode_attention | 122b-e128 | `batch_decode_combine` | seqs=8 ctx=1024 | 16.91 | 0.013 | 0.7 | 9.44 | 0.11 | memory | 256 | 0.11 |
| decode_attention | 122b-e128 | `batch_decode_combine` | seqs=8 ctx=32768 | 16.91 | 0.013 | 0.7 | 9.44 | 0.11 | memory | 256 | 0.11 |
| decode_attention | 122b-e128 | `batch_decode_combine` | seqs=8 ctx=8192 | 16.91 | 0.013 | 0.7 | 9.44 | 0.11 | memory | 256 | 0.11 |
| decode_attention | 122b-nvfp4 | `attention_reduce` | seqs=1 ctx=1024 | 2.11 | 0.002 | 0.7 | 1.18 | 0.01 | memory | 32 | 0.03 |
| decode_attention | 122b-nvfp4 | `attention_reduce` | seqs=1 ctx=32768 | 2.11 | 0.002 | 0.7 | 1.18 | 0.01 | memory | 32 | 0.03 |
| decode_attention | 122b-nvfp4 | `attention_reduce` | seqs=1 ctx=8192 | 2.11 | 0.002 | 0.7 | 1.18 | 0.01 | memory | 32 | 0.03 |
| decode_attention | 122b-nvfp4 | `attention_split` | seqs=1 ctx=1024 | 2.10 | 0.034 | 16.0 | 1.17 | 0.31 | memory | 128 | 0.34 |
| decode_attention | 122b-nvfp4 | `attention_split` | seqs=1 ctx=32768 | 67.11 | 1.074 | 16.0 | 37.45 | 9.76 | memory | 128 | 0.34 |
| decode_attention | 122b-nvfp4 | `attention_split` | seqs=1 ctx=8192 | 16.78 | 0.268 | 16.0 | 9.36 | 2.44 | memory | 128 | 0.34 |
| decode_attention | 122b-nvfp4 | `batch_decode_attention` | seqs=8 ctx=1024 | 16.78 | 0.268 | 16.0 | 9.36 | 2.44 | memory | 1024 | 2.72 |
| decode_attention | 122b-nvfp4 | `batch_decode_attention` | seqs=8 ctx=32768 | 536.87 | 8.590 | 16.0 | 299.59 | 78.09 | memory | 1024 | 2.72 |
| decode_attention | 122b-nvfp4 | `batch_decode_attention` | seqs=8 ctx=8192 | 134.22 | 2.147 | 16.0 | 74.90 | 19.52 | memory | 1024 | 2.72 |
| decode_attention | 122b-nvfp4 | `batch_decode_combine` | seqs=8 ctx=1024 | 16.91 | 0.013 | 0.7 | 9.44 | 0.11 | memory | 256 | 0.11 |
| decode_attention | 122b-nvfp4 | `batch_decode_combine` | seqs=8 ctx=32768 | 16.91 | 0.013 | 0.7 | 9.44 | 0.11 | memory | 256 | 0.11 |
| decode_attention | 122b-nvfp4 | `batch_decode_combine` | seqs=8 ctx=8192 | 16.91 | 0.013 | 0.7 | 9.44 | 0.11 | memory | 256 | 0.11 |
| decode_attention | 27b-nvfp4 | `attention_reduce` | seqs=1 ctx=1024 | 1.59 | 0.001 | 0.7 | 0.88 | 0.01 | memory | 24 | 0.02 |
| decode_attention | 27b-nvfp4 | `attention_reduce` | seqs=1 ctx=32768 | 1.59 | 0.001 | 0.7 | 0.88 | 0.01 | memory | 24 | 0.02 |
| decode_attention | 27b-nvfp4 | `attention_reduce` | seqs=1 ctx=8192 | 1.59 | 0.001 | 0.7 | 0.88 | 0.01 | memory | 24 | 0.02 |
| decode_attention | 27b-nvfp4 | `attention_split` | seqs=1 ctx=1024 | 4.19 | 0.025 | 6.0 | 2.34 | 0.23 | memory | 256 | 0.45 |
| decode_attention | 27b-nvfp4 | `attention_split` | seqs=1 ctx=32768 | 134.22 | 0.805 | 6.0 | 74.90 | 7.32 | memory | 256 | 0.45 |
| decode_attention | 27b-nvfp4 | `attention_split` | seqs=1 ctx=8192 | 33.55 | 0.201 | 6.0 | 18.72 | 1.83 | memory | 256 | 0.45 |
| decode_attention | 27b-nvfp4 | `batch_decode_attention` | seqs=8 ctx=1024 | 33.55 | 0.201 | 6.0 | 18.72 | 1.83 | memory | 2048 | 5.45 |
| decode_attention | 27b-nvfp4 | `batch_decode_attention` | seqs=8 ctx=32768 | 1073.74 | 6.442 | 6.0 | 599.19 | 58.57 | memory | 2048 | 5.45 |
| decode_attention | 27b-nvfp4 | `batch_decode_attention` | seqs=8 ctx=8192 | 268.44 | 1.611 | 6.0 | 149.80 | 14.64 | memory | 2048 | 5.45 |
| decode_attention | 27b-nvfp4 | `batch_decode_combine` | seqs=8 ctx=1024 | 12.68 | 0.009 | 0.7 | 7.08 | 0.09 | memory | 192 | 0.09 |
| decode_attention | 27b-nvfp4 | `batch_decode_combine` | seqs=8 ctx=32768 | 12.68 | 0.009 | 0.7 | 7.08 | 0.09 | memory | 192 | 0.09 |
| decode_attention | 27b-nvfp4 | `batch_decode_combine` | seqs=8 ctx=8192 | 12.68 | 0.009 | 0.7 | 7.08 | 0.09 | memory | 192 | 0.09 |
| decode_attention | 27b-nvfp4-fp8kv | `attention_reduce` | seqs=1 ctx=1024 | 1.59 | 0.001 | 0.7 | 0.88 | 0.01 | memory | 24 | 0.02 |
| decode_attention | 27b-nvfp4-fp8kv | `attention_reduce` | seqs=1 ctx=32768 | 1.59 | 0.001 | 0.7 | 0.88 | 0.01 | memory | 24 | 0.02 |
| decode_attention | 27b-nvfp4-fp8kv | `attention_reduce` | seqs=1 ctx=8192 | 1.59 | 0.001 | 0.7 | 0.88 | 0.01 | memory | 24 | 0.02 |
| decode_attention | 27b-nvfp4-fp8kv | `attention_split` | seqs=1 ctx=1024 | 2.10 | 0.025 | 12.0 | 1.17 | 0.23 | memory | 256 | 0.68 |
| decode_attention | 27b-nvfp4-fp8kv | `attention_split` | seqs=1 ctx=32768 | 67.11 | 0.805 | 12.0 | 37.45 | 7.32 | memory | 256 | 0.68 |
| decode_attention | 27b-nvfp4-fp8kv | `attention_split` | seqs=1 ctx=8192 | 16.78 | 0.201 | 12.0 | 9.36 | 1.83 | memory | 256 | 0.68 |
| decode_attention | 27b-nvfp4-fp8kv | `batch_decode_attention` | seqs=8 ctx=1024 | 16.78 | 0.201 | 12.0 | 9.36 | 1.83 | memory | 2048 | 5.45 |
| decode_attention | 27b-nvfp4-fp8kv | `batch_decode_attention` | seqs=8 ctx=32768 | 536.87 | 6.442 | 12.0 | 299.59 | 58.57 | memory | 2048 | 5.45 |
| decode_attention | 27b-nvfp4-fp8kv | `batch_decode_attention` | seqs=8 ctx=8192 | 134.22 | 1.611 | 12.0 | 74.90 | 14.64 | memory | 2048 | 5.45 |
| decode_attention | 27b-nvfp4-fp8kv | `batch_decode_combine` | seqs=8 ctx=1024 | 12.68 | 0.009 | 0.7 | 7.08 | 0.09 | memory | 192 | 0.09 |
| decode_attention | 27b-nvfp4-fp8kv | `batch_decode_combine` | seqs=8 ctx=32768 | 12.68 | 0.009 | 0.7 | 7.08 | 0.09 | memory | 192 | 0.09 |
| decode_attention | 27b-nvfp4-fp8kv | `batch_decode_combine` | seqs=8 ctx=8192 | 12.68 | 0.009 | 0.7 | 7.08 | 0.09 | memory | 192 | 0.09 |
| decode_attention | 4b-bf16 | `attention_reduce` | seqs=1 ctx=1024 | 1.06 | 0.001 | 0.7 | 0.59 | 0.01 | memory | 16 | 0.01 |
| decode_attention | 4b-bf16 | `attention_reduce` | seqs=1 ctx=32768 | 1.06 | 0.001 | 0.7 | 0.59 | 0.01 | memory | 16 | 0.01 |
| decode_attention | 4b-bf16 | `attention_reduce` | seqs=1 ctx=8192 | 1.06 | 0.001 | 0.7 | 0.59 | 0.01 | memory | 16 | 0.01 |
| decode_attention | 4b-bf16 | `attention_split` | seqs=1 ctx=1024 | 4.19 | 0.017 | 4.0 | 2.34 | 0.15 | memory | 256 | 0.34 |
| decode_attention | 4b-bf16 | `attention_split` | seqs=1 ctx=32768 | 134.22 | 0.537 | 4.0 | 74.90 | 4.88 | memory | 256 | 0.34 |
| decode_attention | 4b-bf16 | `attention_split` | seqs=1 ctx=8192 | 33.55 | 0.134 | 4.0 | 18.72 | 1.22 | memory | 256 | 0.34 |
| decode_attention | 4b-bf16 | `batch_decode_attention` | seqs=8 ctx=1024 | 33.55 | 0.134 | 4.0 | 18.72 | 1.22 | memory | 2048 | 5.45 |
| decode_attention | 4b-bf16 | `batch_decode_attention` | seqs=8 ctx=32768 | 1073.74 | 4.295 | 4.0 | 599.19 | 39.05 | memory | 2048 | 5.45 |
| decode_attention | 4b-bf16 | `batch_decode_attention` | seqs=8 ctx=8192 | 268.44 | 1.074 | 4.0 | 149.80 | 9.76 | memory | 2048 | 5.45 |
| decode_attention | 4b-bf16 | `batch_decode_combine` | seqs=8 ctx=1024 | 8.45 | 0.006 | 0.7 | 4.72 | 0.06 | memory | 128 | 0.06 |
| decode_attention | 4b-bf16 | `batch_decode_combine` | seqs=8 ctx=32768 | 8.45 | 0.006 | 0.7 | 4.72 | 0.06 | memory | 128 | 0.06 |
| decode_attention | 4b-bf16 | `batch_decode_combine` | seqs=8 ctx=8192 | 8.45 | 0.006 | 0.7 | 4.72 | 0.06 | memory | 128 | 0.06 |
| decode_attention | gpt-oss-20b | `attention_reduce` | seqs=1 ctx=1024 | 1.08 | 0.001 | 0.7 | 0.60 | 0.01 | memory | 64 | 0.01 |
| decode_attention | gpt-oss-20b | `attention_reduce` | seqs=1 ctx=32768 | 1.08 | 0.001 | 0.7 | 0.60 | 0.01 | memory | 64 | 0.01 |
| decode_attention | gpt-oss-20b | `attention_reduce` | seqs=1 ctx=8192 | 1.08 | 0.001 | 0.7 | 0.60 | 0.01 | memory | 64 | 0.01 |
| decode_attention | gpt-oss-20b | `attention_split` | seqs=1 ctx=1024 | 2.10 | 0.017 | 8.0 | 1.17 | 0.15 | memory | 512 | 0.91 |
| decode_attention | gpt-oss-20b | `attention_split` | seqs=1 ctx=32768 | 67.11 | 0.537 | 8.0 | 37.45 | 4.88 | memory | 512 | 0.91 |
| decode_attention | gpt-oss-20b | `attention_split` | seqs=1 ctx=8192 | 16.78 | 0.134 | 8.0 | 9.36 | 1.22 | memory | 512 | 0.91 |
| decode_attention | gpt-oss-20b | `batch_decode_attention` | seqs=8 ctx=1024 | 16.78 | 0.134 | 8.0 | 9.36 | 1.22 | memory | 4096 | 2.18 |
| decode_attention | gpt-oss-20b | `batch_decode_attention` | seqs=8 ctx=32768 | 536.87 | 4.295 | 8.0 | 299.59 | 39.05 | memory | 4096 | 2.18 |
| decode_attention | gpt-oss-20b | `batch_decode_attention` | seqs=8 ctx=8192 | 134.22 | 1.074 | 8.0 | 74.90 | 9.76 | memory | 4096 | 2.18 |
| decode_attention | gpt-oss-20b | `batch_decode_combine` | seqs=8 ctx=1024 | 8.65 | 0.006 | 0.7 | 4.83 | 0.06 | memory | 512 | 0.09 |
| decode_attention | gpt-oss-20b | `batch_decode_combine` | seqs=8 ctx=32768 | 8.65 | 0.006 | 0.7 | 4.83 | 0.06 | memory | 512 | 0.09 |
| decode_attention | gpt-oss-20b | `batch_decode_combine` | seqs=8 ctx=8192 | 8.65 | 0.006 | 0.7 | 4.83 | 0.06 | memory | 512 | 0.09 |
| decode_attention | hd128-gqa4 | `attention_reduce` | seqs=1 ctx=1024 | 1.06 | 0.001 | 0.7 | 0.59 | 0.01 | memory | 32 | 0.01 |
| decode_attention | hd128-gqa4 | `attention_reduce` | seqs=1 ctx=32768 | 1.06 | 0.001 | 0.7 | 0.59 | 0.01 | memory | 32 | 0.01 |
| decode_attention | hd128-gqa4 | `attention_reduce` | seqs=1 ctx=8192 | 1.06 | 0.001 | 0.7 | 0.59 | 0.01 | memory | 32 | 0.01 |
| decode_attention | hd128-gqa4 | `attention_split` | seqs=1 ctx=1024 | 4.19 | 0.017 | 4.0 | 2.34 | 0.15 | memory | 512 | 0.39 |
| decode_attention | hd128-gqa4 | `attention_split` | seqs=1 ctx=32768 | 134.22 | 0.537 | 4.0 | 74.90 | 4.88 | memory | 512 | 0.39 |
| decode_attention | hd128-gqa4 | `attention_split` | seqs=1 ctx=8192 | 33.55 | 0.134 | 4.0 | 18.72 | 1.22 | memory | 512 | 0.39 |
| decode_attention | hd128-gqa4 | `batch_decode_attention` | seqs=8 ctx=1024 | 33.55 | 0.134 | 4.0 | 18.72 | 1.22 | memory | 4096 | 4.36 |
| decode_attention | hd128-gqa4 | `batch_decode_attention` | seqs=8 ctx=32768 | 1073.74 | 4.295 | 4.0 | 599.19 | 39.05 | memory | 4096 | 4.36 |
| decode_attention | hd128-gqa4 | `batch_decode_attention` | seqs=8 ctx=8192 | 268.44 | 1.074 | 4.0 | 149.80 | 9.76 | memory | 4096 | 4.36 |
| decode_attention | hd128-gqa4 | `batch_decode_combine` | seqs=8 ctx=1024 | 8.52 | 0.006 | 0.7 | 4.75 | 0.06 | memory | 256 | 0.06 |
| decode_attention | hd128-gqa4 | `batch_decode_combine` | seqs=8 ctx=32768 | 8.52 | 0.006 | 0.7 | 4.75 | 0.06 | memory | 256 | 0.06 |
| decode_attention | hd128-gqa4 | `batch_decode_combine` | seqs=8 ctx=8192 | 8.52 | 0.006 | 0.7 | 4.75 | 0.06 | memory | 256 | 0.06 |
| decode_attention | hd128-gqa8 | `attention_reduce` | seqs=1 ctx=1024 | 1.06 | 0.001 | 0.7 | 0.59 | 0.01 | memory | 32 | 0.01 |
| decode_attention | hd128-gqa8 | `attention_reduce` | seqs=1 ctx=32768 | 1.06 | 0.001 | 0.7 | 0.59 | 0.01 | memory | 32 | 0.01 |
| decode_attention | hd128-gqa8 | `attention_reduce` | seqs=1 ctx=8192 | 1.06 | 0.001 | 0.7 | 0.59 | 0.01 | memory | 32 | 0.01 |
| decode_attention | hd128-gqa8 | `attention_split` | seqs=1 ctx=1024 | 2.10 | 0.017 | 8.0 | 1.17 | 0.15 | memory | 256 | 0.34 |
| decode_attention | hd128-gqa8 | `attention_split` | seqs=1 ctx=32768 | 67.11 | 0.537 | 8.0 | 37.45 | 4.88 | memory | 256 | 0.34 |
| decode_attention | hd128-gqa8 | `attention_split` | seqs=1 ctx=8192 | 16.78 | 0.134 | 8.0 | 9.36 | 1.22 | memory | 256 | 0.34 |
| decode_attention | hd128-gqa8 | `batch_decode_attention` | seqs=8 ctx=1024 | 16.78 | 0.134 | 8.0 | 9.36 | 1.22 | memory | 2048 | 2.18 |
| decode_attention | hd128-gqa8 | `batch_decode_attention` | seqs=8 ctx=32768 | 536.87 | 4.295 | 8.0 | 299.59 | 39.05 | memory | 2048 | 2.18 |
| decode_attention | hd128-gqa8 | `batch_decode_attention` | seqs=8 ctx=8192 | 134.22 | 1.074 | 8.0 | 74.90 | 9.76 | memory | 2048 | 2.18 |
| decode_attention | hd128-gqa8 | `batch_decode_combine` | seqs=8 ctx=1024 | 8.52 | 0.006 | 0.7 | 4.75 | 0.06 | memory | 256 | 0.06 |
| decode_attention | hd128-gqa8 | `batch_decode_combine` | seqs=8 ctx=32768 | 8.52 | 0.006 | 0.7 | 4.75 | 0.06 | memory | 256 | 0.06 |
| decode_attention | hd128-gqa8 | `batch_decode_combine` | seqs=8 ctx=8192 | 8.52 | 0.006 | 0.7 | 4.75 | 0.06 | memory | 256 | 0.06 |
| fused_add_rmsnorm | hidden-2560 | `add_rmsnorm_2560` | rows=1 | 0.04 | 0.000 | 0.3 | 0.02 | 0.00 | memory | 1 | 0.01 |
| fused_add_rmsnorm | hidden-2560 | `add_rmsnorm_2560` | rows=4096 | 167.77 | 0.052 | 0.3 | 93.62 | 0.48 | memory | 4096 | 21.79 |
| fused_add_rmsnorm | hidden-5120 | `add_rmsnorm_5120` | rows=1 | 0.08 | 0.000 | 0.3 | 0.05 | 0.00 | memory | 1 | 0.01 |
| fused_add_rmsnorm | hidden-5120 | `add_rmsnorm_5120` | rows=4096 | 335.54 | 0.105 | 0.3 | 187.25 | 0.95 | memory | 4096 | 21.79 |
| nvfp4_gemm | 122b-e128 | `down_f4a_big` | m=4096 n=3072 k=1024 | 68.88 | 25.770 | 374.1 | 38.44 | 14.68 | memory | 384 | 2.04 |
| nvfp4_gemm | 122b-e128 | `gate_up_f4a_big` | m=4096 n=2048 k=3072 | 87.43 | 51.540 | 589.5 | 48.79 | 29.37 | memory | 256 | 1.36 |
| nvfp4_gemm | 122b-e128 | `quant_rows_fp4` | m=4096 n=3072 k=1024 | 19.14 | 0.017 | 0.9 | 10.68 | 0.15 | memory | 4096 | 3.63 |
| nvfp4_gemm | 122b-e128 | `quant_rows_fp4` | m=4096 n=2048 k=3072 | 57.41 | 0.050 | 0.9 | 32.04 | 0.46 | memory | 4096 | 3.63 |
| nvfp4_gemm | 122b-nvfp4 | `down_f4a_big` | m=4096 n=3072 k=1024 | 68.88 | 25.770 | 374.1 | 38.44 | 14.68 | memory | 384 | 2.04 |
| nvfp4_gemm | 122b-nvfp4 | `gate_up_f4a_big` | m=4096 n=2048 k=3072 | 87.43 | 51.540 | 589.5 | 48.79 | 29.37 | memory | 256 | 1.36 |
| nvfp4_gemm | 122b-nvfp4 | `quant_rows_fp4` | m=4096 n=3072 k=1024 | 19.14 | 0.017 | 0.9 | 10.68 | 0.15 | memory | 4096 | 3.63 |
| nvfp4_gemm | 122b-nvfp4 | `quant_rows_fp4` | m=4096 n=2048 k=3072 | 57.41 | 0.050 | 0.9 | 32.04 | 0.46 | memory | 4096 | 3.63 |
| nvfp4_gemm | 27b-nvfp4 | `down_f4a_big` | m=4096 n=5120 k=17408 | 419.23 | 730.144 | 1741.6 | 233.95 | 416.04 | compute fp4 | 640 | 3.40 |
| nvfp4_gemm | 27b-nvfp4 | `gate_up_f4a_big` | m=4096 n=34816 k=5120 | 754.58 | 1460.289 | 1935.2 | 421.08 | 832.07 | compute fp4 | 4352 | 23.15 |
| nvfp4_gemm | 27b-nvfp4 | `quant_rows_fp4` | m=4096 n=5120 k=17408 | 325.32 | 0.285 | 0.9 | 181.54 | 2.59 | memory | 4096 | 3.63 |
| nvfp4_gemm | 27b-nvfp4 | `quant_rows_fp4` | m=4096 n=34816 k=5120 | 95.68 | 0.084 | 0.9 | 53.39 | 0.76 | memory | 4096 | 3.63 |
| nvfp4_gemm | 27b-nvfp4-fp8kv | `down_f4a_big` | m=4096 n=5120 k=17408 | 419.23 | 730.144 | 1741.6 | 233.95 | 416.04 | compute fp4 | 640 | 3.40 |
| nvfp4_gemm | 27b-nvfp4-fp8kv | `gate_up_f4a_big` | m=4096 n=34816 k=5120 | 754.58 | 1460.289 | 1935.2 | 421.08 | 832.07 | compute fp4 | 4352 | 23.15 |
| nvfp4_gemm | 27b-nvfp4-fp8kv | `quant_rows_fp4` | m=4096 n=5120 k=17408 | 325.32 | 0.285 | 0.9 | 181.54 | 2.59 | memory | 4096 | 3.63 |
| nvfp4_gemm | 27b-nvfp4-fp8kv | `quant_rows_fp4` | m=4096 n=34816 k=5120 | 95.68 | 0.084 | 0.9 | 53.39 | 0.76 | memory | 4096 | 3.63 |
| nvfp4_gemm | hd128-gqa4 | `down_f4a_big` | m=4096 n=5120 k=17408 | 419.23 | 730.144 | 1741.6 | 233.95 | 416.04 | compute fp4 | 640 | 3.40 |
| nvfp4_gemm | hd128-gqa4 | `gate_up_f4a_big` | m=4096 n=34816 k=5120 | 754.58 | 1460.289 | 1935.2 | 421.08 | 832.07 | compute fp4 | 4352 | 23.15 |
| nvfp4_gemm | hd128-gqa4 | `quant_rows_fp4` | m=4096 n=5120 k=17408 | 325.32 | 0.285 | 0.9 | 181.54 | 2.59 | memory | 4096 | 3.63 |
| nvfp4_gemm | hd128-gqa4 | `quant_rows_fp4` | m=4096 n=34816 k=5120 | 95.68 | 0.084 | 0.9 | 53.39 | 0.76 | memory | 4096 | 3.63 |
| nvfp4_gemm | hd128-gqa8 | `down_f4a_big` | m=4096 n=5120 k=17408 | 419.23 | 730.144 | 1741.6 | 233.95 | 416.04 | compute fp4 | 640 | 3.40 |
| nvfp4_gemm | hd128-gqa8 | `gate_up_f4a_big` | m=4096 n=34816 k=5120 | 754.58 | 1460.289 | 1935.2 | 421.08 | 832.07 | compute fp4 | 4352 | 23.15 |
| nvfp4_gemm | hd128-gqa8 | `quant_rows_fp4` | m=4096 n=5120 k=17408 | 325.32 | 0.285 | 0.9 | 181.54 | 2.59 | memory | 4096 | 3.63 |
| nvfp4_gemm | hd128-gqa8 | `quant_rows_fp4` | m=4096 n=34816 k=5120 | 95.68 | 0.084 | 0.9 | 53.39 | 0.76 | memory | 4096 | 3.63 |
| nvfp4_gemv | 122b-e128 | `down_proj_f4` | m=1 n=3072 k=1024 | 1.79 | 0.006 | 3.5 | 1.00 | 0.06 | memory | 192 | 0.34 |
| nvfp4_gemv | 122b-e128 | `gate_up_act_f4` | m=1 n=2048 k=3072 | 3.56 | 0.013 | 3.5 | 1.99 | 0.11 | memory | 64 | 0.11 |
| nvfp4_gemv | 122b-nvfp4 | `down_proj_f4` | m=1 n=3072 k=1024 | 1.79 | 0.006 | 3.5 | 1.00 | 0.06 | memory | 192 | 0.34 |
| nvfp4_gemv | 122b-nvfp4 | `gate_up_act_f4` | m=1 n=2048 k=3072 | 3.56 | 0.013 | 3.5 | 1.99 | 0.11 | memory | 64 | 0.11 |
| nvfp4_gemv | 27b-nvfp4 | `down_proj_f4` | m=1 n=5120 k=17408 | 50.23 | 0.178 | 3.5 | 28.03 | 1.62 | memory | 320 | 0.57 |
| nvfp4_gemv | 27b-nvfp4 | `gate_up_act_f4` | m=1 n=34816 k=5120 | 100.43 | 0.357 | 3.5 | 56.04 | 3.24 | memory | 1088 | 1.93 |
| nvfp4_gemv | 27b-nvfp4-fp8kv | `down_proj_f4` | m=1 n=5120 k=17408 | 50.23 | 0.178 | 3.5 | 28.03 | 1.62 | memory | 320 | 0.57 |
| nvfp4_gemv | 27b-nvfp4-fp8kv | `gate_up_act_f4` | m=1 n=34816 k=5120 | 100.43 | 0.357 | 3.5 | 56.04 | 3.24 | memory | 1088 | 1.93 |
| nvfp4_gemv | hd128-gqa4 | `down_proj_f4` | m=1 n=5120 k=17408 | 50.23 | 0.178 | 3.5 | 28.03 | 1.62 | memory | 320 | 0.57 |
| nvfp4_gemv | hd128-gqa4 | `gate_up_act_f4` | m=1 n=34816 k=5120 | 100.43 | 0.357 | 3.5 | 56.04 | 3.24 | memory | 1088 | 1.93 |
| nvfp4_gemv | hd128-gqa8 | `down_proj_f4` | m=1 n=5120 k=17408 | 50.23 | 0.178 | 3.5 | 28.03 | 1.62 | memory | 320 | 0.57 |
| nvfp4_gemv | hd128-gqa8 | `gate_up_act_f4` | m=1 n=34816 k=5120 | 100.43 | 0.357 | 3.5 | 56.04 | 3.24 | memory | 1088 | 1.93 |
| prefill_attention | 122b-e128 | `flash_mma` | tokens=1024 | 52.43 | 17.180 | 327.7 | 29.26 | 39.13 | compute bf16 | 256 | 1.36 |
| prefill_attention | 122b-e128 | `flash_mma` | tokens=16384 | 838.86 | 4398.047 | 5242.9 | 468.11 | 10018.33 | compute bf16 | 4096 | 21.79 |
| prefill_attention | 122b-e128 | `flash_mma` | tokens=4096 | 209.72 | 274.878 | 1310.7 | 117.03 | 626.15 | compute bf16 | 1024 | 5.45 |
| prefill_attention | 122b-nvfp4 | `flash_mma` | tokens=1024 | 52.43 | 17.180 | 327.7 | 29.26 | 39.13 | compute bf16 | 256 | 1.36 |
| prefill_attention | 122b-nvfp4 | `flash_mma` | tokens=16384 | 838.86 | 4398.047 | 5242.9 | 468.11 | 10018.33 | compute bf16 | 4096 | 21.79 |
| prefill_attention | 122b-nvfp4 | `flash_mma` | tokens=4096 | 209.72 | 274.878 | 1310.7 | 117.03 | 626.15 | compute bf16 | 1024 | 5.45 |
| prefill_attention | 27b-nvfp4 | `flash_mma` | tokens=1024 | 41.94 | 12.885 | 307.2 | 23.41 | 29.35 | compute bf16 | 192 | 1.02 |
| prefill_attention | 27b-nvfp4 | `flash_mma` | tokens=16384 | 671.09 | 3298.535 | 4915.2 | 374.49 | 7513.75 | compute bf16 | 3072 | 16.34 |
| prefill_attention | 27b-nvfp4 | `flash_mma` | tokens=4096 | 167.77 | 206.158 | 1228.8 | 93.62 | 469.61 | compute bf16 | 768 | 4.09 |
| prefill_attention | 27b-nvfp4-fp8kv | `flash_mma` | tokens=1024 | 41.94 | 12.885 | 307.2 | 23.41 | 29.35 | compute bf16 | 192 | 1.02 |
| prefill_attention | 27b-nvfp4-fp8kv | `flash_mma` | tokens=16384 | 671.09 | 3298.535 | 4915.2 | 374.49 | 7513.75 | compute bf16 | 3072 | 16.34 |
| prefill_attention | 27b-nvfp4-fp8kv | `flash_mma` | tokens=4096 | 167.77 | 206.158 | 1228.8 | 93.62 | 469.61 | compute bf16 | 768 | 4.09 |
| prefill_attention | 4b-bf16 | `flash_mma` | tokens=1024 | 29.36 | 8.590 | 292.6 | 16.38 | 19.57 | compute bf16 | 128 | 0.68 |
| prefill_attention | 4b-bf16 | `flash_mma` | tokens=16384 | 469.76 | 2199.023 | 4681.1 | 262.14 | 5009.16 | compute bf16 | 2048 | 10.89 |
| prefill_attention | 4b-bf16 | `flash_mma` | tokens=4096 | 117.44 | 137.439 | 1170.3 | 65.54 | 313.07 | compute bf16 | 512 | 2.72 |
| prefill_attention | gpt-oss-20b | `flash_mma` | tokens=1024 | 27.26 | 8.590 | 315.1 | 15.21 | 19.57 | compute bf16 | 1024 | 1.36 |
| prefill_attention | gpt-oss-20b | `flash_mma` | tokens=16384 | 436.21 | 2199.023 | 5041.2 | 243.42 | 5009.16 | compute bf16 | 16384 | 21.79 |
| prefill_attention | gpt-oss-20b | `flash_mma` | tokens=4096 | 109.05 | 137.439 | 1260.3 | 60.85 | 313.07 | compute bf16 | 4096 | 5.45 |
| prefill_attention | hd128-gqa4 | `flash_mma` | tokens=1024 | 29.36 | 8.590 | 292.6 | 16.38 | 19.57 | compute bf16 | 512 | 1.36 |
| prefill_attention | hd128-gqa4 | `flash_mma` | tokens=16384 | 469.76 | 2199.023 | 4681.1 | 262.14 | 5009.16 | compute bf16 | 8192 | 21.79 |
| prefill_attention | hd128-gqa4 | `flash_mma` | tokens=4096 | 117.44 | 137.439 | 1170.3 | 65.54 | 313.07 | compute bf16 | 2048 | 5.45 |
| prefill_attention | hd128-gqa8 | `flash_mma` | tokens=1024 | 27.26 | 8.590 | 315.1 | 15.21 | 19.57 | compute bf16 | 512 | 1.36 |
| prefill_attention | hd128-gqa8 | `flash_mma` | tokens=16384 | 436.21 | 2199.023 | 5041.2 | 243.42 | 5009.16 | compute bf16 | 8192 | 21.79 |
| prefill_attention | hd128-gqa8 | `flash_mma` | tokens=4096 | 109.05 | 137.439 | 1260.3 | 60.85 | 313.07 | compute bf16 | 2048 | 5.45 |
| rmsnorm | 122b-e128 | `rmsnorm` | rows=1 | 0.02 | 0.000 | 0.5 | 0.01 | 0.00 | memory | 1 | 0.01 |
| rmsnorm | 122b-e128 | `rmsnorm` | rows=4096 | 100.66 | 0.050 | 0.5 | 56.17 | 0.46 | memory | 4096 | 21.79 |
| rmsnorm | 122b-nvfp4 | `rmsnorm` | rows=1 | 0.02 | 0.000 | 0.5 | 0.01 | 0.00 | memory | 1 | 0.01 |
| rmsnorm | 122b-nvfp4 | `rmsnorm` | rows=4096 | 100.66 | 0.050 | 0.5 | 56.17 | 0.46 | memory | 4096 | 21.79 |
| rmsnorm | 27b-nvfp4 | `rmsnorm` | rows=1 | 0.04 | 0.000 | 0.5 | 0.02 | 0.00 | memory | 1 | 0.01 |
| rmsnorm | 27b-nvfp4 | `rmsnorm` | rows=4096 | 167.77 | 0.084 | 0.5 | 93.62 | 0.76 | memory | 4096 | 21.79 |
| rmsnorm | 27b-nvfp4-fp8kv | `rmsnorm` | rows=1 | 0.04 | 0.000 | 0.5 | 0.02 | 0.00 | memory | 1 | 0.01 |
| rmsnorm | 27b-nvfp4-fp8kv | `rmsnorm` | rows=4096 | 167.77 | 0.084 | 0.5 | 93.62 | 0.76 | memory | 4096 | 21.79 |
| rmsnorm | 4b-bf16 | `rmsnorm` | rows=1 | 0.02 | 0.000 | 0.5 | 0.01 | 0.00 | memory | 1 | 0.01 |
| rmsnorm | 4b-bf16 | `rmsnorm` | rows=4096 | 83.89 | 0.042 | 0.5 | 46.81 | 0.38 | memory | 4096 | 21.79 |
| rmsnorm | gpt-oss-20b | `rmsnorm` | rows=1 | 0.02 | 0.000 | 0.5 | 0.01 | 0.00 | memory | 1 | 0.01 |
| rmsnorm | gpt-oss-20b | `rmsnorm` | rows=4096 | 94.37 | 0.047 | 0.5 | 52.66 | 0.43 | memory | 4096 | 21.79 |
| rmsnorm | hd128-gqa4 | `rmsnorm` | rows=1 | 0.04 | 0.000 | 0.5 | 0.02 | 0.00 | memory | 1 | 0.01 |
| rmsnorm | hd128-gqa4 | `rmsnorm` | rows=4096 | 167.77 | 0.084 | 0.5 | 93.62 | 0.76 | memory | 4096 | 21.79 |
| rmsnorm | hd128-gqa8 | `rmsnorm` | rows=1 | 0.04 | 0.000 | 0.5 | 0.02 | 0.00 | memory | 1 | 0.01 |
| rmsnorm | hd128-gqa8 | `rmsnorm` | rows=4096 | 167.77 | 0.084 | 0.5 | 93.62 | 0.76 | memory | 4096 | 21.79 |
| rotary | 122b-e128 | `qk_norm_rope` | rows=1 | 0.14 | 0.000 | 0.4 | 0.08 | 0.00 | memory | 36 | 0.03 |
| rotary | 122b-e128 | `qk_norm_rope` | rows=4096 | 570.43 | 0.214 | 0.4 | 318.32 | 1.94 | memory | 147456 | 130.72 |
| rotary | 122b-nvfp4 | `qk_norm_rope` | rows=1 | 0.14 | 0.000 | 0.4 | 0.08 | 0.00 | memory | 36 | 0.03 |
| rotary | 122b-nvfp4 | `qk_norm_rope` | rows=4096 | 570.43 | 0.214 | 0.4 | 318.32 | 1.94 | memory | 147456 | 130.72 |
| rotary | 27b-nvfp4 | `qk_norm_rope` | rows=1 | 0.11 | 0.000 | 0.4 | 0.06 | 0.00 | memory | 32 | 0.03 |
| rotary | 27b-nvfp4 | `qk_norm_rope` | rows=4096 | 469.76 | 0.176 | 0.4 | 262.14 | 1.60 | memory | 131072 | 116.20 |
| rotary | 27b-nvfp4-fp8kv | `qk_norm_rope` | rows=1 | 0.11 | 0.000 | 0.4 | 0.06 | 0.00 | memory | 32 | 0.03 |
| rotary | 27b-nvfp4-fp8kv | `qk_norm_rope` | rows=4096 | 469.76 | 0.176 | 0.4 | 262.14 | 1.60 | memory | 131072 | 116.20 |
| rotary | 4b-bf16 | `qk_norm_rope` | rows=1 | 0.08 | 0.000 | 0.4 | 0.05 | 0.00 | memory | 24 | 0.02 |
| rotary | 4b-bf16 | `qk_norm_rope` | rows=4096 | 335.54 | 0.126 | 0.4 | 187.25 | 1.14 | memory | 98304 | 87.15 |
| rotary | gpt-oss-20b | `qk_norm_rope` | rows=1 | 0.04 | 0.000 | 0.7 | 0.02 | 0.00 | memory | 80 | 0.02 |
| rotary | gpt-oss-20b | `qk_norm_rope` | rows=4096 | 167.77 | 0.113 | 0.7 | 93.62 | 1.03 | memory | 327680 | 72.62 |
| rotary | hd128-gqa4 | `qk_norm_rope` | rows=1 | 0.08 | 0.000 | 0.4 | 0.05 | 0.00 | memory | 48 | 0.02 |
| rotary | hd128-gqa4 | `qk_norm_rope` | rows=4096 | 335.54 | 0.126 | 0.4 | 187.25 | 1.14 | memory | 196608 | 87.15 |
| rotary | hd128-gqa8 | `qk_norm_rope` | rows=1 | 0.07 | 0.000 | 0.4 | 0.04 | 0.00 | memory | 40 | 0.02 |
| rotary | hd128-gqa8 | `qk_norm_rope` | rows=4096 | 301.99 | 0.113 | 0.4 | 168.52 | 1.03 | memory | 163840 | 72.62 |
| silu_and_mul | 122b-e128 | `swiglu` | rows=1 | 0.01 | 0.000 | 0.5 | 0.01 | 0.00 | memory | 4 | 0.00 |
| silu_and_mul | 122b-e128 | `swiglu` | rows=4096 | 50.33 | 0.025 | 0.5 | 28.09 | 0.23 | memory | 16384 | 14.52 |
| silu_and_mul | 122b-nvfp4 | `swiglu` | rows=1 | 0.01 | 0.000 | 0.5 | 0.01 | 0.00 | memory | 4 | 0.00 |
| silu_and_mul | 122b-nvfp4 | `swiglu` | rows=4096 | 50.33 | 0.025 | 0.5 | 28.09 | 0.23 | memory | 16384 | 14.52 |
| silu_and_mul | 27b-nvfp4 | `swiglu` | rows=1 | 0.21 | 0.000 | 0.5 | 0.12 | 0.00 | memory | 68 | 0.06 |
| silu_and_mul | 27b-nvfp4 | `swiglu` | rows=4096 | 855.64 | 0.428 | 0.5 | 477.48 | 3.89 | memory | 278528 | 246.92 |
| silu_and_mul | 27b-nvfp4-fp8kv | `swiglu` | rows=1 | 0.21 | 0.000 | 0.5 | 0.12 | 0.00 | memory | 68 | 0.06 |
| silu_and_mul | 27b-nvfp4-fp8kv | `swiglu` | rows=4096 | 855.64 | 0.428 | 0.5 | 477.48 | 3.89 | memory | 278528 | 246.92 |
| silu_and_mul | 4b-bf16 | `swiglu` | rows=1 | 0.11 | 0.000 | 0.5 | 0.06 | 0.00 | memory | 36 | 0.03 |
| silu_and_mul | 4b-bf16 | `swiglu` | rows=4096 | 452.98 | 0.226 | 0.5 | 252.78 | 2.06 | memory | 147456 | 130.72 |
| silu_and_mul | gpt-oss-20b | `swiglu` | rows=1 | 0.03 | 0.000 | 0.5 | 0.02 | 0.00 | memory | 12 | 0.01 |
| silu_and_mul | gpt-oss-20b | `swiglu` | rows=4096 | 141.56 | 0.071 | 0.5 | 78.99 | 0.64 | memory | 49152 | 43.57 |
| silu_and_mul | hd128-gqa4 | `swiglu` | rows=1 | 0.21 | 0.000 | 0.5 | 0.12 | 0.00 | memory | 68 | 0.06 |
| silu_and_mul | hd128-gqa4 | `swiglu` | rows=4096 | 855.64 | 0.428 | 0.5 | 477.48 | 3.89 | memory | 278528 | 246.92 |
| silu_and_mul | hd128-gqa8 | `swiglu` | rows=1 | 0.21 | 0.000 | 0.5 | 0.12 | 0.00 | memory | 68 | 0.06 |
| silu_and_mul | hd128-gqa8 | `swiglu` | rows=4096 | 855.64 | 0.428 | 0.5 | 477.48 | 3.89 | memory | 278528 | 246.92 |
| topk_routing | 122b-e128 | `moe_route` | tokens=1 | 0.00 | 0.000 | 2.0 | 0.00 | 0.00 | memory | 1 | 0.00 |
| topk_routing | 122b-e128 | `moe_route` | tokens=4096 | 2.10 | 0.004 | 2.0 | 1.17 | 0.04 | memory | 4096 | 0.68 |
| topk_routing | 122b-e128 | `moe_route_logits` | tokens=1 | 0.80 | 0.001 | 1.0 | 0.45 | 0.01 | memory | 17 | 0.09 |
| topk_routing | 122b-e128 | `moe_route_logits` | tokens=4096 | 51.12 | 3.221 | 63.0 | 28.53 | 29.28 | compute fp32 | 17408 | 92.60 |
| topk_routing | 122b-nvfp4 | `moe_route` | tokens=1 | 0.00 | 0.000 | 2.0 | 0.00 | 0.00 | memory | 1 | 0.00 |
| topk_routing | 122b-nvfp4 | `moe_route` | tokens=4096 | 4.19 | 0.008 | 2.0 | 2.34 | 0.08 | memory | 4096 | 0.91 |
| topk_routing | 122b-nvfp4 | `moe_route_logits` | tokens=1 | 1.59 | 0.002 | 1.0 | 0.88 | 0.01 | memory | 33 | 0.18 |
| topk_routing | 122b-nvfp4 | `moe_route_logits` | tokens=4096 | 51.90 | 6.442 | 124.1 | 28.96 | 58.57 | compute fp32 | 33792 | 179.74 |
| topk_routing | gpt-oss-20b | `moe_route` | tokens=1 | 0.00 | 0.000 | 2.0 | 0.00 | 0.00 | memory | 1 | 0.00 |
| topk_routing | gpt-oss-20b | `moe_route` | tokens=4096 | 0.52 | 0.001 | 2.0 | 0.29 | 0.01 | memory | 4096 | 0.68 |
| topk_routing | gpt-oss-20b | `moe_route_logits` | tokens=1 | 0.20 | 0.000 | 0.9 | 0.11 | 0.00 | memory | 5 | 0.01 |
| topk_routing | gpt-oss-20b | `moe_route_logits` | tokens=4096 | 47.37 | 0.755 | 15.9 | 26.43 | 6.86 | memory | 5120 | 13.62 |

## sm_121: DGX Spark GB10 (spark)

Target `target Cuda cc=12.1 sms=48 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]`, NVRTC 13.2 (--std=c++17 no --split-compile (it makes register allocation nondeterministic)). Peak numbers: NVIDIA DGX Spark spec: 273 GB/s LPDDR5x, 1 PFLOP sparse FP4 (dense 500), bf16 dense a quarter of that; 48 SMs.

- `glm4.5-air` does not compile for sm_121: ptxas error   : Entry function 'moe_route_logits' uses too much shared data (0x10080 bytes, 0xc000 max)


### bf16_gemm

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4b-bf16 | `down_mma_big` | 256 | 92 | 33792 | 0 | 2 (regs) | 512 | 129, 523 | ldgsts 4 ldsm 12 lds 3 mma 32 bar 3 | Triton 3.8.0 `matmul` | 256 | 156 | 66560 | 0 | 256 | 140, 840 | ldgsts 8 ldsm 24 lds 3 mma 64 bar 5 | matches |
| 4b-bf16 | `gate_up_mma_big` | 256 | 166 | 50176 | 0 | 1 (regs) | 256 | 177, 913 | ldgsts 6 ldsm 16 lds 3 mma 64 bar 3 | Triton 3.8.0 `matmul` | 256 | 156 | 66560 | 0 | 256 | 140, 840 | ldgsts 8 ldsm 24 lds 3 mma 64 bar 5 | matches |
| gpt-oss-20b | `down_mma_big` | 256 | 127 | 50176 | 0 | 2 (regs) | 512 | 256, 1012 | ldgsts 6 ldsm 20 lds 3 mma 64 fma 64 bar 3 | Triton 3.8.0 `matmul` | 256 | 156 | 66560 | 0 | 256 | 140, 840 | ldgsts 8 ldsm 24 lds 3 mma 64 bar 5 | matches |
| gpt-oss-20b | `gate_up_mma_big` | 256 | 128 | 50176 | 0 | 2 (regs) | 512 | 254, 986 | ldgsts 6 ldsm 20 lds 3 mma 64 fma 64 bar 3 | Triton 3.8.0 `matmul` | 256 | 156 | 66560 | 0 | 256 | 140, 840 | ldgsts 8 ldsm 24 lds 3 mma 64 bar 5 | matches |

### bf16_gemv

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4b-bf16 | `down_proj` | 256 | 41 | 1024 | 0 | 5 (regs) | 1280 | 189, 328 | ldg128 24 fma 96 | cuBLAS GEMV is closed source; no static artefact | | | | | | | | |
| 4b-bf16 | `gate_up_act` | 256 | 47 | 11392 | 0 | 5 (regs) | 1280 | 39, 82 | ldg32 8 fma 8 | cuBLAS GEMV is closed source; no static artefact | | | | | | | | |
| 4b-bf16 | `qkv_proj_norm` | 256 | 46 | 11392 | 0 | 5 (regs) | 1280 | 39, 80 | ldg32 8 fma 8 | cuBLAS GEMV is closed source; no static artefact | | | | | | | | |
| gpt-oss-20b | `down_proj` | 256 | 41 | 1024 | 0 | 5 (regs) | 1280 | 189, 328 | ldg128 24 fma 96 | cuBLAS GEMV is closed source; no static artefact | | | | | | | | |
| gpt-oss-20b | `gate_up_act` | 256 | 47 | 12672 | 0 | 5 (regs) | 1280 | 190, 387 | ldg128 8 lds 16 fma 96 | cuBLAS GEMV is closed source; no static artefact | | | | | | | | |
| gpt-oss-20b | `qkv_proj_norm` | 256 | 44 | 12672 | 0 | 5 (regs) | 1280 | 190, 326 | ldg128 8 lds 16 fma 96 | cuBLAS GEMV is closed source; no static artefact | | | | | | | | |

### decode_attention

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `attention_reduce` | 256 | 40 | 1296 | 0 | 6 (regs) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| 122b-e128 | `attention_split` | 128 | 255 | 5632 | 480 | 2 (regs) | 256 | 7854, 23602 | lds 192 fma 3967 mufu 186 shfl 320 bar 353 local 496 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 512 | 72 | 1024 | 0 | 512 | 2041, 2606 | ldg32 1 ldgsts 4 lds 72 fma 821 mufu 34 shfl 160 bar 16 | behind: spills 480 B |
| 122b-e128 | `batch_decode_attention` | 32 | 255 | 33792 | 216 | 2 (smem) | 64 | 999, 3462 | ldg32 7 ldgsts 7 ldsm 32 lds 12 mma 128 fma 190 mufu 10 shfl 8 bar 4 local 46 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 512 | 72 | 1024 | 0 | 512 | 2041, 2606 | ldg32 1 ldgsts 4 lds 72 fma 821 mufu 34 shfl 160 bar 16 | behind: spills 216 B, 64 threads/SM vs 512 |
| 122b-e128 | `batch_decode_combine` | 128 | 40 | 1024 | 0 | 12 (regs) | 1536 | 80, 178 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |
| 122b-nvfp4 | `attention_reduce` | 256 | 40 | 1296 | 0 | 6 (regs) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| 122b-nvfp4 | `attention_split` | 128 | 255 | 5632 | 480 | 2 (regs) | 256 | 7854, 23602 | lds 192 fma 3967 mufu 186 shfl 320 bar 353 local 496 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 512 | 72 | 1024 | 0 | 512 | 2041, 2606 | ldg32 1 ldgsts 4 lds 72 fma 821 mufu 34 shfl 160 bar 16 | behind: spills 480 B |
| 122b-nvfp4 | `batch_decode_attention` | 32 | 255 | 33792 | 216 | 2 (smem) | 64 | 999, 3462 | ldg32 7 ldgsts 7 ldsm 32 lds 12 mma 128 fma 190 mufu 10 shfl 8 bar 4 local 46 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 512 | 72 | 1024 | 0 | 512 | 2041, 2606 | ldg32 1 ldgsts 4 lds 72 fma 821 mufu 34 shfl 160 bar 16 | behind: spills 216 B, 64 threads/SM vs 512 |
| 122b-nvfp4 | `batch_decode_combine` | 128 | 40 | 1024 | 0 | 12 (regs) | 1536 | 80, 178 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |
| 27b-nvfp4 | `attention_reduce` | 256 | 40 | 1296 | 0 | 6 (regs) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| 27b-nvfp4 | `attention_split` | 128 | 167 | 5312 | 0 | 3 (regs) | 384 | 2656, 7503 | lds 72 fma 1396 mufu 66 shfl 120 bar 133 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 192 | 64 | 1024 | 0 | 960 | 833, 1260 | ldg32 1 ldgsts 4 lds 32 fma 321 mufu 14 shfl 60 bar 16 | behind: 384 threads/SM vs 960 |
| 27b-nvfp4 | `batch_decode_attention` | 32 | 255 | 33792 | 224 | 2 (smem) | 64 | 986, 3347 | ldg32 7 ldgsts 7 ldsm 32 lds 12 mma 128 fma 190 mufu 10 shfl 8 bar 4 local 42 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 192 | 64 | 1024 | 0 | 960 | 833, 1260 | ldg32 1 ldgsts 4 lds 32 fma 321 mufu 14 shfl 60 bar 16 | behind: spills 224 B, 64 threads/SM vs 960 |
| 27b-nvfp4 | `batch_decode_combine` | 128 | 40 | 1024 | 0 | 12 (regs) | 1536 | 80, 178 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |
| 27b-nvfp4-fp8kv | `attention_reduce` | 256 | 40 | 1296 | 0 | 6 (regs) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| 27b-nvfp4-fp8kv | `attention_split` | 128 | 204 | 5312 | 0 | 2 (regs) | 256 | 2647, 6608 | lds 72 fma 1394 mufu 66 shfl 120 bar 133 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 80 | 1024 | 0 | 768 | 2081, 2752 | ldg32 1 ldgsts 4 lds 32 fma 901 mufu 14 shfl 48 bar 16 | behind: 256 threads/SM vs 768 |
| 27b-nvfp4-fp8kv | `batch_decode_attention` | 32 | 255 | 33792 | 208 | 2 (smem) | 64 | 2200, 7413 | ldg32 10 ldgsts 13 ldsm 32 lds 214 mma 128 fma 206 mufu 10 shfl 8 bar 4 local 54 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 80 | 1024 | 0 | 768 | 2081, 2752 | ldg32 1 ldgsts 4 lds 32 fma 901 mufu 14 shfl 48 bar 16 | behind: spills 208 B, 64 threads/SM vs 768 |
| 27b-nvfp4-fp8kv | `batch_decode_combine` | 128 | 40 | 1024 | 0 | 12 (regs) | 1536 | 80, 178 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |
| 4b-bf16 | `attention_reduce` | 256 | 40 | 1296 | 0 | 6 (regs) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| 4b-bf16 | `attention_split` | 128 | 127 | 5248 | 0 | 4 (regs) | 512 | 1720, 4834 | lds 48 fma 884 mufu 42 shfl 80 bar 89 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 56 | 1024 | 0 | 1152 | 587, 957 | ldg32 1 ldgsts 4 lds 24 fma 220 mufu 10 shfl 40 bar 16 | behind: 512 threads/SM vs 1152 |
| 4b-bf16 | `batch_decode_attention` | 32 | 255 | 33792 | 208 | 2 (smem) | 64 | 980, 3341 | ldg32 7 ldgsts 7 ldsm 32 lds 12 mma 128 fma 190 mufu 10 shfl 8 bar 4 local 42 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 56 | 1024 | 0 | 1152 | 587, 957 | ldg32 1 ldgsts 4 lds 24 fma 220 mufu 10 shfl 40 bar 16 | behind: spills 208 B, 64 threads/SM vs 1152 |
| 4b-bf16 | `batch_decode_combine` | 128 | 40 | 1024 | 0 | 12 (regs) | 1536 | 80, 178 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |
| gpt-oss-20b | `attention_reduce` | 64 | 40 | 1296 | 0 | 24 (threads) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| gpt-oss-20b | `attention_split` | 128 | 157 | 2304 | 0 | 3 (regs) | 384 | 2864, 8045 | lds 96 fma 1530 mufu 90 shfl 160 bar 177 | no competitor built for this target | | | | | | | | |
| gpt-oss-20b | `batch_decode_attention` | 32 | 128 | 9216 | 0 | 10 (smem) | 320 | 583, 1428 | ldg32 7 ldgsts 7 ldsm 8 lds 12 mma 32 fma 94 mufu 10 shfl 8 bar 4 | no competitor built for this target | | | | | | | | |
| gpt-oss-20b | `batch_decode_combine` | 32 | 40 | 1024 | 0 | 32 (blocks) | 1024 | 80, 182 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |
| hd128-gqa4 | `attention_reduce` | 128 | 40 | 1296 | 0 | 12 (regs) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| hd128-gqa4 | `attention_split` | 128 | 72 | 3200 | 0 | 7 (regs) | 896 | 1345, 4217 | lds 48 fma 713 mufu 42 shfl 80 bar 89 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 63 | 1024 | 0 | 1024 | 572, 931 | ldg32 1 ldgsts 4 lds 24 fma 212 mufu 10 shfl 32 bar 16 | matches |
| hd128-gqa4 | `batch_decode_attention` | 32 | 168 | 17408 | 0 | 5 (smem) | 160 | 729, 2095 | ldg32 7 ldgsts 7 ldsm 16 lds 12 mma 64 fma 126 mufu 10 shfl 8 bar 4 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 63 | 1024 | 0 | 1024 | 572, 931 | ldg32 1 ldgsts 4 lds 24 fma 212 mufu 10 shfl 32 bar 16 | behind: 160 threads/SM vs 1024 |
| hd128-gqa4 | `batch_decode_combine` | 64 | 40 | 1024 | 0 | 24 (threads) | 1536 | 80, 178 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |
| hd128-gqa8 | `attention_reduce` | 128 | 40 | 1296 | 0 | 12 (regs) | 1536 | 68, 176 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| hd128-gqa8 | `attention_split` | 128 | 128 | 3328 | 0 | 4 (regs) | 512 | 2807, 8790 | lds 96 fma 1535 mufu 90 shfl 160 bar 177 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 64 | 1024 | 0 | 1024 | 1037, 1585 | ldg32 1 ldgsts 4 lds 41 fma 404 mufu 18 shfl 64 bar 16 | matches |
| hd128-gqa8 | `batch_decode_attention` | 32 | 168 | 17408 | 0 | 5 (smem) | 160 | 719, 2078 | ldg32 7 ldgsts 7 ldsm 16 lds 12 mma 64 fma 126 mufu 10 shfl 8 bar 4 | FlashInfer 0.7.3 `flashinfer33BatchDecodeWithPagedKVCacheKernelILN` | 128 | 64 | 1024 | 0 | 1024 | 1037, 1585 | ldg32 1 ldgsts 4 lds 41 fma 404 mufu 18 shfl 64 bar 16 | behind: 160 threads/SM vs 1024 |
| hd128-gqa8 | `batch_decode_combine` | 64 | 40 | 1024 | 0 | 24 (threads) | 1536 | 80, 178 | ldg32 16 fma 28 mufu 4 | no competitor built for this target | | | | | | | | |

### fused_add_rmsnorm

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hidden-2560 | `add_rmsnorm_2560` | 1024 | 35 | 1152 | 0 | 1 (regs) | 1024 | 56, 153 | ldg32 8 fma 10 | FlashInfer 0.7.3 `flashinfer4norm21FusedAddRMSNormKernelILj8E13__n` | 320 | 40 | 1024 | 0 | 1280 | 299, 674 | ldg128 3 lds 7 fma 86 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `add_rmsnorm` | 256 | 40 | 1056 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm25fused_add_rms_norm_kernelIN3c108BFloat16EL` | 320 | 36 | 1192 | 0 | 1280 | 140, 255 | ldg128 4 fma 35 |  |
| hidden-5120 | `add_rmsnorm_5120` | 1024 | 32 | 1152 | 0 | 1 (threads) | 1024 | 56, 153 | ldg32 8 fma 10 | FlashInfer 0.7.3 `flashinfer4norm21FusedAddRMSNormKernelILj8E13__n` | 640 | 40 | 1024 | 0 | 1280 | 299, 674 | ldg128 3 lds 7 fma 86 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `add_rmsnorm` | 256 | 40 | 1056 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm25fused_add_rms_norm_kernelIN3c108BFloat16EL` | 640 | 36 | 1192 | 0 | 1280 | 140, 255 | ldg128 4 fma 35 |  |

### nvfp4_gemm

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `down_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 223, 860 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 122b-e128 | `gate_up_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 232, 866 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 122b-e128 | `quant_rows_fp4` | 256 | 40 | 1024 | 0 | 6 (regs) | 1536 | 1, 1 |  | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 122b-nvfp4 | `down_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 223, 860 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 122b-nvfp4 | `gate_up_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 232, 866 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 122b-nvfp4 | `quant_rows_fp4` | 256 | 40 | 1024 | 0 | 6 (regs) | 1536 | 1, 1 |  | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 27b-nvfp4 | `down_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 223, 860 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 27b-nvfp4 | `gate_up_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 232, 866 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 27b-nvfp4 | `quant_rows_fp4` | 256 | 40 | 1024 | 0 | 6 (regs) | 1536 | 1, 1 |  | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 27b-nvfp4-fp8kv | `down_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 223, 860 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 27b-nvfp4-fp8kv | `gate_up_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 232, 866 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| 27b-nvfp4-fp8kv | `quant_rows_fp4` | 256 | 40 | 1024 | 0 | 6 (regs) | 1536 | 1, 1 |  | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| hd128-gqa4 | `down_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 223, 860 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| hd128-gqa4 | `gate_up_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 232, 866 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| hd128-gqa4 | `quant_rows_fp4` | 256 | 40 | 1024 | 0 | 6 (regs) | 1536 | 1, 1 |  | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| hd128-gqa8 | `down_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 223, 860 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| hd128-gqa8 | `gate_up_f4a_big` | 512 | 128 | 83968 | 0 | 1 (regs) | 512 | 232, 866 | ldgsts 10 ldsm 12 lds 25 mma 32 bar 3 | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |
| hd128-gqa8 | `quant_rows_fp4` | 256 | 40 | 1024 | 0 | 6 (regs) | 1536 | 1, 1 |  | FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings | | | | | | | | |

### nvfp4_gemv

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `down_proj_f4` | 512 | 40 | 1024 | 0 | 3 (regs) | 1536 | 1, 1 |  | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| 122b-e128 | `gate_up_act_f4` | 512 | 40 | 13440 | 0 | 3 (regs) | 1536 | 447, 629 | ldg64 4 ldg32 4 lds 32 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| 122b-nvfp4 | `down_proj_f4` | 512 | 40 | 1024 | 0 | 3 (regs) | 1536 | 1, 1 |  | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| 122b-nvfp4 | `gate_up_act_f4` | 512 | 40 | 13440 | 0 | 3 (regs) | 1536 | 447, 629 | ldg64 4 ldg32 4 lds 32 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| 27b-nvfp4 | `down_proj_f4` | 512 | 38 | 1024 | 0 | 3 (regs) | 1536 | 503, 935 | ldg64 4 ldg32 68 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| 27b-nvfp4 | `gate_up_act_f4` | 512 | 39 | 21632 | 0 | 3 (regs) | 1536 | 447, 612 | ldg64 4 ldg32 4 lds 32 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| 27b-nvfp4-fp8kv | `down_proj_f4` | 512 | 38 | 1024 | 0 | 3 (regs) | 1536 | 503, 935 | ldg64 4 ldg32 68 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| 27b-nvfp4-fp8kv | `gate_up_act_f4` | 512 | 39 | 21632 | 0 | 3 (regs) | 1536 | 447, 612 | ldg64 4 ldg32 4 lds 32 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| hd128-gqa4 | `down_proj_f4` | 512 | 38 | 1024 | 0 | 3 (regs) | 1536 | 503, 935 | ldg64 4 ldg32 68 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| hd128-gqa4 | `gate_up_act_f4` | 512 | 39 | 21632 | 0 | 3 (regs) | 1536 | 447, 612 | ldg64 4 ldg32 4 lds 32 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| hd128-gqa8 | `down_proj_f4` | 512 | 38 | 1024 | 0 | 3 (regs) | 1536 | 503, 935 | ldg64 4 ldg32 68 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |
| hd128-gqa8 | `gate_up_act_f4` | 512 | 39 | 21632 | 0 | 3 (regs) | 1536 | 447, 612 | ldg64 4 ldg32 4 lds 32 fma 136 | FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1 | | | | | | | | |

### prefill_attention

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `flash_mma` | 256 | 255 | 99328 | 32 | 1 (regs) | 256 | 913, 2496 | ldg32 8 ldgsts 8 ldsm 64 lds 3 mma 128 fma 206 mufu 18 shfl 8 bar 7 | FlashInfer 0.7.3 `flashinfer34BatchPrefillWithPagedKVCacheKernelIL` | 128 | 255 | 50176 | 0 | 256 | 492, 1301 | ldg32 1 ldgsts 8 ldsm 48 lds 3 mma 65 fma 156 mufu 10 shfl 4 bar 8 | behind: spills 32 B |
| 122b-nvfp4 | `flash_mma` | 256 | 255 | 99328 | 32 | 1 (regs) | 256 | 913, 2496 | ldg32 8 ldgsts 8 ldsm 64 lds 3 mma 128 fma 206 mufu 18 shfl 8 bar 7 | FlashInfer 0.7.3 `flashinfer34BatchPrefillWithPagedKVCacheKernelIL` | 128 | 255 | 50176 | 0 | 256 | 492, 1301 | ldg32 1 ldgsts 8 ldsm 48 lds 3 mma 65 fma 156 mufu 10 shfl 4 bar 8 | behind: spills 32 B |
| 27b-nvfp4 | `flash_mma` | 256 | 255 | 99328 | 32 | 1 (regs) | 256 | 916, 2506 | ldg32 8 ldgsts 8 ldsm 64 lds 3 mma 128 fma 206 mufu 18 shfl 8 bar 7 | FlashInfer 0.7.3 `flashinfer34BatchPrefillWithPagedKVCacheKernelIL` | 128 | 255 | 50176 | 0 | 256 | 492, 1301 | ldg32 1 ldgsts 8 ldsm 48 lds 3 mma 65 fma 156 mufu 10 shfl 4 bar 8 | behind: spills 32 B |
| 27b-nvfp4-fp8kv | `flash_mma` | 256 | 255 | 100096 | 32 | 1 (regs) | 256 | 1584, 3483 | ldg32 13 ldgsts 16 ldsm 64 lds 25 mma 128 fma 334 mufu 18 shfl 8 bar 9 | FlashInfer 0.7.3 `flashinfer34BatchPrefillWithPagedKVCacheKernelIL` | 128 | 255 | 50176 | 0 | 256 | 492, 1301 | ldg32 1 ldgsts 8 ldsm 48 lds 3 mma 65 fma 156 mufu 10 shfl 4 bar 8 | behind: spills 32 B |
| 4b-bf16 | `flash_mma` | 256 | 255 | 99328 | 32 | 1 (regs) | 256 | 913, 2496 | ldg32 8 ldgsts 8 ldsm 64 lds 3 mma 128 fma 206 mufu 18 shfl 8 bar 7 | FlashInfer 0.7.3 `flashinfer34BatchPrefillWithPagedKVCacheKernelIL` | 128 | 255 | 50176 | 0 | 256 | 492, 1301 | ldg32 1 ldgsts 8 ldsm 48 lds 3 mma 65 fma 156 mufu 10 shfl 4 bar 8 | behind: spills 32 B |
| gpt-oss-20b | `flash_mma` | 128 | 116 | 17408 | 32 | 4 (regs) | 512 | 521, 1037 | ldg32 4 ldgsts 4 ldsm 16 lds 3 mma 32 fma 110 mufu 18 shfl 8 bar 7 | no competitor built for this target | | | | | | | | |
| hd128-gqa4 | `flash_mma` | 128 | 166 | 33792 | 32 | 2 (smem) | 256 | 665, 1706 | ldg32 8 ldgsts 8 ldsm 32 lds 3 mma 64 fma 142 mufu 18 shfl 8 bar 8 | FlashInfer 0.7.3 `flashinfer34BatchPrefillWithPagedKVCacheKernelIL` | 128 | 246 | 50176 | 0 | 256 | 793, 2169 | ldg32 3 ldgsts 8 ldsm 48 lds 6 mma 132 fma 216 mufu 36 shfl 8 bar 8 | behind: spills 32 B |
| hd128-gqa8 | `flash_mma` | 128 | 166 | 33792 | 32 | 2 (smem) | 256 | 665, 1706 | ldg32 8 ldgsts 8 ldsm 32 lds 3 mma 64 fma 142 mufu 18 shfl 8 bar 8 | FlashInfer 0.7.3 `flashinfer34BatchPrefillWithPagedKVCacheKernelIL` | 128 | 246 | 50176 | 0 | 256 | 793, 2169 | ldg32 3 ldgsts 8 ldsm 48 lds 6 mma 132 fma 216 mufu 36 shfl 8 bar 8 | behind: spills 32 B |

### rmsnorm

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 384 | 40 | 1024 | 0 | 1536 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rmsnorm` | 256 | 39 | 1056 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 384 | 40 | 1192 | 0 | 1536 | 119, 183 | ldg128 4 fma 32 |  |
| 122b-nvfp4 | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 384 | 40 | 1024 | 0 | 1536 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rmsnorm` | 256 | 39 | 1056 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 384 | 40 | 1192 | 0 | 1536 | 119, 183 | ldg128 4 fma 32 |  |
| 27b-nvfp4 | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 640 | 40 | 1024 | 0 | 1280 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rmsnorm` | 256 | 44 | 1056 | 0 | 1280 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 640 | 40 | 1192 | 0 | 1280 | 119, 183 | ldg128 4 fma 32 |  |
| 27b-nvfp4-fp8kv | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 640 | 40 | 1024 | 0 | 1280 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rmsnorm` | 256 | 44 | 1056 | 0 | 1280 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 640 | 40 | 1192 | 0 | 1280 | 119, 183 | ldg128 4 fma 32 |  |
| 4b-bf16 | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 320 | 40 | 1024 | 0 | 1280 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rmsnorm` | 256 | 39 | 1056 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 320 | 40 | 1192 | 0 | 1280 | 119, 183 | ldg128 4 fma 32 |  |
| gpt-oss-20b | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 360 | 40 | 1024 | 0 | 1536 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 360 | 40 | 1192 | 0 | 1536 | 119, 183 | ldg128 4 fma 32 |  |
| hd128-gqa4 | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 640 | 40 | 1024 | 0 | 1280 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rmsnorm` | 256 | 44 | 1056 | 0 | 1280 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 640 | 40 | 1192 | 0 | 1280 | 119, 183 | ldg128 4 fma 32 |  |
| hd128-gqa8 | `rmsnorm` | 1024 | 34 | 1152 | 0 | 1 (regs) | 1024 | 39, 117 | ldg32 8 fma 8 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 640 | 40 | 1024 | 0 | 1280 | 323, 667 | ldg128 6 lds 1 fma 83 mufu 2 shfl 5 bar 7 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rmsnorm` | 256 | 44 | 1056 | 0 | 1280 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 640 | 40 | 1192 | 0 | 1280 | 119, 183 | ldg128 4 fma 32 |  |

### rotary

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `qk_norm_rope` | 256 | 36 | 2176 | 32 | 6 (regs) | 1536 | 445, 1653 | ldg32 7 lds 12 fma 93 mufu 2 shfl 5 bar 10 local 10 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rotary` | 128 | 18 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |
| 122b-nvfp4 | `qk_norm_rope` | 256 | 36 | 2176 | 32 | 6 (regs) | 1536 | 445, 1653 | ldg32 7 lds 12 fma 93 mufu 2 shfl 5 bar 10 local 10 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rotary` | 128 | 18 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |
| 27b-nvfp4 | `qk_norm_rope` | 256 | 35 | 2176 | 32 | 6 (regs) | 1536 | 410, 1541 | ldg32 4 lds 12 fma 93 mufu 2 shfl 5 bar 10 local 10 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rotary` | 128 | 18 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |
| 27b-nvfp4-fp8kv | `qk_norm_rope` | 256 | 34 | 2176 | 32 | 6 (regs) | 1536 | 567, 2075 | ldg32 5 lds 19 fma 141 mufu 3 shfl 10 bar 12 local 10 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rotary` | 128 | 18 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |
| 4b-bf16 | `qk_norm_rope` | 256 | 36 | 2176 | 32 | 6 (regs) | 1536 | 445, 1653 | ldg32 7 lds 12 fma 93 mufu 2 shfl 5 bar 10 local 10 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rotary` | 128 | 18 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |
| gpt-oss-20b | `qk_norm_rope` | 64 | 28 | 1280 | 32 | 24 (threads) | 1536 | 11, 23 | ldg32 1 fma 1 local 1 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |
| hd128-gqa4 | `qk_norm_rope` | 128 | 35 | 1664 | 32 | 12 (regs) | 1536 | 410, 1541 | ldg32 4 lds 12 fma 93 mufu 2 shfl 5 bar 10 local 10 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rotary` | 128 | 18 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |
| hd128-gqa8 | `qk_norm_rope` | 128 | 35 | 1664 | 32 | 12 (regs) | 1536 | 410, 1541 | ldg32 4 lds 12 fma 93 mufu 2 shfl 5 bar 10 local 10 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1536 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `rotary` | 128 | 18 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 31 | 0 | 0 | 1536 | 51, 156 | ldg32 4 fma 4 |  |

### silu_and_mul

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `silu_and_mul` | 128 | 31 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |
| 122b-nvfp4 | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `silu_and_mul` | 128 | 31 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |
| 27b-nvfp4 | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `silu_and_mul` | 128 | 31 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |
| 27b-nvfp4-fp8kv | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `silu_and_mul` | 128 | 31 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |
| 4b-bf16 | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `silu_and_mul` | 128 | 31 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |
| gpt-oss-20b | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |
| hd128-gqa4 | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `silu_and_mul` | 128 | 31 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |
| hd128-gqa8 | `swiglu` | 256 | 22 | 1024 | 0 | 6 (threads) | 1536 | 37, 117 | ldg32 2 fma 13 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 40 | 0 | 0 | 1536 | 89, 154 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `silu_and_mul` | 128 | 31 | 0 | 0 | 1536 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 27 | 1024 | 0 | 1024 | 195, 560 | ldg32 8 fma 72 mufu 8 |  |

### topk_routing

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122b-e128 | `moe_route` | 32 | 21 | 2112 | 0 | 32 (smem) | 1024 | 134, 440 | lds 10 fma 2 shfl 10 | Triton 3.8.0 `softmax_topk` | 128 | 24 | 1056 | 0 | 1536 | 1, 1 |  | behind: 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm3moe10topkGatingILi8ELi128ELi4ELi16ELi32Ei13` | 128 | 40 | 0 | 0 | 1536 | 143, 429 | ldg32 1 fma 1 shfl 12 |  |
| 122b-e128 | `moe_route_logits` | 256 | 72 | 50304 | 0 | 1 (smem) | 256 | 232, 818 | ldg128 1 ldg32 14 lds 8 fma 67 mufu 1 shfl 5 bar 2 | no competitor built for this target | | | | | | | | |
| 122b-nvfp4 | `moe_route` | 32 | 26 | 3136 | 0 | 24 (smem) | 768 | 101, 333 | lds 10 fma 3 shfl 10 | Triton 3.8.0 `softmax_topk` | 128 | 28 | 1056 | 0 | 1536 | 1, 1 |  | behind: 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm3moe10topkGatingILi8ELi128ELi4ELi16ELi32Ei13` | 128 | 40 | 0 | 0 | 1536 | 143, 429 | ldg32 1 fma 1 shfl 12 |  |
| 122b-nvfp4 | `moe_route_logits` | 256 | 72 | 50304 | 0 | 1 (smem) | 256 | 232, 818 | ldg128 1 ldg32 14 lds 8 fma 67 mufu 1 shfl 5 bar 2 | no competitor built for this target | | | | | | | | |
| gpt-oss-20b | `moe_route` | 32 | 20 | 1312 | 0 | 32 (blocks) | 1024 | 1, 1 |  | Triton 3.8.0 `softmax_topk` | 128 | 22 | 0 | 0 | 1536 | 1, 1 |  | behind: 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm3moe10topkGatingILi8ELi128ELi4ELi16ELi32Ei13` | 128 | 40 | 0 | 0 | 1536 | 143, 429 | ldg32 1 fma 1 shfl 12 |  |
| gpt-oss-20b | `moe_route_logits` | 256 | 72 | 47232 | 0 | 2 (smem) | 512 | 232, 814 | ldg128 1 ldg32 14 lds 8 fma 67 mufu 1 shfl 5 bar 2 | no competitor built for this target | | | | | | | | |

### Roofline (algorithmic bytes and FLOPs, peak numbers above)

| family | spec | kernel | shape | MB | GFLOP | FLOP/B | mem us | math us | bound | blocks | waves |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bf16_gemm | 4b-bf16 | `down_mma_big` | m=4096 n=2560 k=9216 | 240.12 | 193.274 | 804.9 | 879.57 | 1546.19 | compute bf16 | 640 | 6.67 |
| bf16_gemm | 4b-bf16 | `gate_up_mma_big` | m=4096 n=18432 k=2560 | 438.30 | 386.547 | 881.9 | 1605.51 | 3092.38 | compute bf16 | 2304 | 48.00 |
| bf16_gemm | gpt-oss-20b | `down_mma_big` | m=4096 n=2880 k=2880 | 110.96 | 67.948 | 612.4 | 406.45 | 543.58 | compute bf16 | 736 | 7.67 |
| bf16_gemm | gpt-oss-20b | `gate_up_mma_big` | m=4096 n=5760 k=2880 | 174.74 | 135.895 | 777.7 | 640.06 | 1087.16 | compute bf16 | 1440 | 15.00 |
| bf16_gemv | 4b-bf16 | `down_proj` | m=1 n=2560 k=9216 | 47.23 | 0.047 | 1.0 | 173.01 | 1.52 | memory | 320 | 1.33 |
| bf16_gemv | 4b-bf16 | `gate_up_act` | m=1 n=18432 k=2560 | 94.46 | 0.094 | 1.0 | 345.99 | 3.04 | memory | 1152 | 4.80 |
| bf16_gemv | 4b-bf16 | `qkv_proj_norm` | m=1 n=10240 k=2560 | 52.48 | 0.052 | 1.0 | 192.23 | 1.69 | memory | 1280 | 5.33 |
| bf16_gemv | gpt-oss-20b | `down_proj` | m=1 n=2880 k=2880 | 16.61 | 0.017 | 1.0 | 60.85 | 0.54 | memory | 360 | 1.50 |
| bf16_gemv | gpt-oss-20b | `gate_up_act` | m=1 n=5760 k=2880 | 33.21 | 0.033 | 1.0 | 121.66 | 1.07 | memory | 360 | 1.50 |
| bf16_gemv | gpt-oss-20b | `qkv_proj_norm` | m=1 n=5120 k=2880 | 29.52 | 0.029 | 1.0 | 108.14 | 0.95 | memory | 640 | 2.67 |
| decode_attention | 122b-e128 | `attention_reduce` | seqs=1 ctx=1024 | 2.11 | 0.002 | 0.7 | 7.74 | 0.05 | memory | 32 | 0.11 |
| decode_attention | 122b-e128 | `attention_reduce` | seqs=1 ctx=32768 | 2.11 | 0.002 | 0.7 | 7.74 | 0.05 | memory | 32 | 0.11 |
| decode_attention | 122b-e128 | `attention_reduce` | seqs=1 ctx=8192 | 2.11 | 0.002 | 0.7 | 7.74 | 0.05 | memory | 32 | 0.11 |
| decode_attention | 122b-e128 | `attention_split` | seqs=1 ctx=1024 | 2.10 | 0.034 | 16.0 | 7.68 | 1.08 | memory | 128 | 1.33 |
| decode_attention | 122b-e128 | `attention_split` | seqs=1 ctx=32768 | 67.11 | 1.074 | 16.0 | 245.82 | 34.64 | memory | 128 | 1.33 |
| decode_attention | 122b-e128 | `attention_split` | seqs=1 ctx=8192 | 16.78 | 0.268 | 16.0 | 61.46 | 8.66 | memory | 128 | 1.33 |
| decode_attention | 122b-e128 | `batch_decode_attention` | seqs=8 ctx=1024 | 16.78 | 0.268 | 16.0 | 61.46 | 8.66 | memory | 1024 | 10.67 |
| decode_attention | 122b-e128 | `batch_decode_attention` | seqs=8 ctx=32768 | 536.87 | 8.590 | 16.0 | 1966.56 | 277.09 | memory | 1024 | 10.67 |
| decode_attention | 122b-e128 | `batch_decode_attention` | seqs=8 ctx=8192 | 134.22 | 2.147 | 16.0 | 491.64 | 69.27 | memory | 1024 | 10.67 |
| decode_attention | 122b-e128 | `batch_decode_combine` | seqs=8 ctx=1024 | 16.91 | 0.013 | 0.7 | 61.94 | 0.41 | memory | 256 | 0.44 |
| decode_attention | 122b-e128 | `batch_decode_combine` | seqs=8 ctx=32768 | 16.91 | 0.013 | 0.7 | 61.94 | 0.41 | memory | 256 | 0.44 |
| decode_attention | 122b-e128 | `batch_decode_combine` | seqs=8 ctx=8192 | 16.91 | 0.013 | 0.7 | 61.94 | 0.41 | memory | 256 | 0.44 |
| decode_attention | 122b-nvfp4 | `attention_reduce` | seqs=1 ctx=1024 | 2.11 | 0.002 | 0.7 | 7.74 | 0.05 | memory | 32 | 0.11 |
| decode_attention | 122b-nvfp4 | `attention_reduce` | seqs=1 ctx=32768 | 2.11 | 0.002 | 0.7 | 7.74 | 0.05 | memory | 32 | 0.11 |
| decode_attention | 122b-nvfp4 | `attention_reduce` | seqs=1 ctx=8192 | 2.11 | 0.002 | 0.7 | 7.74 | 0.05 | memory | 32 | 0.11 |
| decode_attention | 122b-nvfp4 | `attention_split` | seqs=1 ctx=1024 | 2.10 | 0.034 | 16.0 | 7.68 | 1.08 | memory | 128 | 1.33 |
| decode_attention | 122b-nvfp4 | `attention_split` | seqs=1 ctx=32768 | 67.11 | 1.074 | 16.0 | 245.82 | 34.64 | memory | 128 | 1.33 |
| decode_attention | 122b-nvfp4 | `attention_split` | seqs=1 ctx=8192 | 16.78 | 0.268 | 16.0 | 61.46 | 8.66 | memory | 128 | 1.33 |
| decode_attention | 122b-nvfp4 | `batch_decode_attention` | seqs=8 ctx=1024 | 16.78 | 0.268 | 16.0 | 61.46 | 8.66 | memory | 1024 | 10.67 |
| decode_attention | 122b-nvfp4 | `batch_decode_attention` | seqs=8 ctx=32768 | 536.87 | 8.590 | 16.0 | 1966.56 | 277.09 | memory | 1024 | 10.67 |
| decode_attention | 122b-nvfp4 | `batch_decode_attention` | seqs=8 ctx=8192 | 134.22 | 2.147 | 16.0 | 491.64 | 69.27 | memory | 1024 | 10.67 |
| decode_attention | 122b-nvfp4 | `batch_decode_combine` | seqs=8 ctx=1024 | 16.91 | 0.013 | 0.7 | 61.94 | 0.41 | memory | 256 | 0.44 |
| decode_attention | 122b-nvfp4 | `batch_decode_combine` | seqs=8 ctx=32768 | 16.91 | 0.013 | 0.7 | 61.94 | 0.41 | memory | 256 | 0.44 |
| decode_attention | 122b-nvfp4 | `batch_decode_combine` | seqs=8 ctx=8192 | 16.91 | 0.013 | 0.7 | 61.94 | 0.41 | memory | 256 | 0.44 |
| decode_attention | 27b-nvfp4 | `attention_reduce` | seqs=1 ctx=1024 | 1.59 | 0.001 | 0.7 | 5.81 | 0.04 | memory | 24 | 0.08 |
| decode_attention | 27b-nvfp4 | `attention_reduce` | seqs=1 ctx=32768 | 1.59 | 0.001 | 0.7 | 5.81 | 0.04 | memory | 24 | 0.08 |
| decode_attention | 27b-nvfp4 | `attention_reduce` | seqs=1 ctx=8192 | 1.59 | 0.001 | 0.7 | 5.81 | 0.04 | memory | 24 | 0.08 |
| decode_attention | 27b-nvfp4 | `attention_split` | seqs=1 ctx=1024 | 4.19 | 0.025 | 6.0 | 15.36 | 0.81 | memory | 256 | 1.78 |
| decode_attention | 27b-nvfp4 | `attention_split` | seqs=1 ctx=32768 | 134.22 | 0.805 | 6.0 | 491.64 | 25.98 | memory | 256 | 1.78 |
| decode_attention | 27b-nvfp4 | `attention_split` | seqs=1 ctx=8192 | 33.55 | 0.201 | 6.0 | 122.91 | 6.49 | memory | 256 | 1.78 |
| decode_attention | 27b-nvfp4 | `batch_decode_attention` | seqs=8 ctx=1024 | 33.55 | 0.201 | 6.0 | 122.91 | 6.49 | memory | 2048 | 21.33 |
| decode_attention | 27b-nvfp4 | `batch_decode_attention` | seqs=8 ctx=32768 | 1073.74 | 6.442 | 6.0 | 3933.12 | 207.82 | memory | 2048 | 21.33 |
| decode_attention | 27b-nvfp4 | `batch_decode_attention` | seqs=8 ctx=8192 | 268.44 | 1.611 | 6.0 | 983.28 | 51.96 | memory | 2048 | 21.33 |
| decode_attention | 27b-nvfp4 | `batch_decode_combine` | seqs=8 ctx=1024 | 12.68 | 0.009 | 0.7 | 46.45 | 0.30 | memory | 192 | 0.33 |
| decode_attention | 27b-nvfp4 | `batch_decode_combine` | seqs=8 ctx=32768 | 12.68 | 0.009 | 0.7 | 46.45 | 0.30 | memory | 192 | 0.33 |
| decode_attention | 27b-nvfp4 | `batch_decode_combine` | seqs=8 ctx=8192 | 12.68 | 0.009 | 0.7 | 46.45 | 0.30 | memory | 192 | 0.33 |
| decode_attention | 27b-nvfp4-fp8kv | `attention_reduce` | seqs=1 ctx=1024 | 1.59 | 0.001 | 0.7 | 5.81 | 0.04 | memory | 24 | 0.08 |
| decode_attention | 27b-nvfp4-fp8kv | `attention_reduce` | seqs=1 ctx=32768 | 1.59 | 0.001 | 0.7 | 5.81 | 0.04 | memory | 24 | 0.08 |
| decode_attention | 27b-nvfp4-fp8kv | `attention_reduce` | seqs=1 ctx=8192 | 1.59 | 0.001 | 0.7 | 5.81 | 0.04 | memory | 24 | 0.08 |
| decode_attention | 27b-nvfp4-fp8kv | `attention_split` | seqs=1 ctx=1024 | 2.10 | 0.025 | 12.0 | 7.68 | 0.81 | memory | 256 | 2.67 |
| decode_attention | 27b-nvfp4-fp8kv | `attention_split` | seqs=1 ctx=32768 | 67.11 | 0.805 | 12.0 | 245.82 | 25.98 | memory | 256 | 2.67 |
| decode_attention | 27b-nvfp4-fp8kv | `attention_split` | seqs=1 ctx=8192 | 16.78 | 0.201 | 12.0 | 61.46 | 6.49 | memory | 256 | 2.67 |
| decode_attention | 27b-nvfp4-fp8kv | `batch_decode_attention` | seqs=8 ctx=1024 | 16.78 | 0.201 | 12.0 | 61.46 | 6.49 | memory | 2048 | 21.33 |
| decode_attention | 27b-nvfp4-fp8kv | `batch_decode_attention` | seqs=8 ctx=32768 | 536.87 | 6.442 | 12.0 | 1966.56 | 207.82 | memory | 2048 | 21.33 |
| decode_attention | 27b-nvfp4-fp8kv | `batch_decode_attention` | seqs=8 ctx=8192 | 134.22 | 1.611 | 12.0 | 491.64 | 51.96 | memory | 2048 | 21.33 |
| decode_attention | 27b-nvfp4-fp8kv | `batch_decode_combine` | seqs=8 ctx=1024 | 12.68 | 0.009 | 0.7 | 46.45 | 0.30 | memory | 192 | 0.33 |
| decode_attention | 27b-nvfp4-fp8kv | `batch_decode_combine` | seqs=8 ctx=32768 | 12.68 | 0.009 | 0.7 | 46.45 | 0.30 | memory | 192 | 0.33 |
| decode_attention | 27b-nvfp4-fp8kv | `batch_decode_combine` | seqs=8 ctx=8192 | 12.68 | 0.009 | 0.7 | 46.45 | 0.30 | memory | 192 | 0.33 |
| decode_attention | 4b-bf16 | `attention_reduce` | seqs=1 ctx=1024 | 1.06 | 0.001 | 0.7 | 3.87 | 0.03 | memory | 16 | 0.06 |
| decode_attention | 4b-bf16 | `attention_reduce` | seqs=1 ctx=32768 | 1.06 | 0.001 | 0.7 | 3.87 | 0.03 | memory | 16 | 0.06 |
| decode_attention | 4b-bf16 | `attention_reduce` | seqs=1 ctx=8192 | 1.06 | 0.001 | 0.7 | 3.87 | 0.03 | memory | 16 | 0.06 |
| decode_attention | 4b-bf16 | `attention_split` | seqs=1 ctx=1024 | 4.19 | 0.017 | 4.0 | 15.36 | 0.54 | memory | 256 | 1.33 |
| decode_attention | 4b-bf16 | `attention_split` | seqs=1 ctx=32768 | 134.22 | 0.537 | 4.0 | 491.64 | 17.32 | memory | 256 | 1.33 |
| decode_attention | 4b-bf16 | `attention_split` | seqs=1 ctx=8192 | 33.55 | 0.134 | 4.0 | 122.91 | 4.33 | memory | 256 | 1.33 |
| decode_attention | 4b-bf16 | `batch_decode_attention` | seqs=8 ctx=1024 | 33.55 | 0.134 | 4.0 | 122.91 | 4.33 | memory | 2048 | 21.33 |
| decode_attention | 4b-bf16 | `batch_decode_attention` | seqs=8 ctx=32768 | 1073.74 | 4.295 | 4.0 | 3933.12 | 138.55 | memory | 2048 | 21.33 |
| decode_attention | 4b-bf16 | `batch_decode_attention` | seqs=8 ctx=8192 | 268.44 | 1.074 | 4.0 | 983.28 | 34.64 | memory | 2048 | 21.33 |
| decode_attention | 4b-bf16 | `batch_decode_combine` | seqs=8 ctx=1024 | 8.45 | 0.006 | 0.7 | 30.97 | 0.20 | memory | 128 | 0.22 |
| decode_attention | 4b-bf16 | `batch_decode_combine` | seqs=8 ctx=32768 | 8.45 | 0.006 | 0.7 | 30.97 | 0.20 | memory | 128 | 0.22 |
| decode_attention | 4b-bf16 | `batch_decode_combine` | seqs=8 ctx=8192 | 8.45 | 0.006 | 0.7 | 30.97 | 0.20 | memory | 128 | 0.22 |
| decode_attention | gpt-oss-20b | `attention_reduce` | seqs=1 ctx=1024 | 1.08 | 0.001 | 0.7 | 3.96 | 0.03 | memory | 64 | 0.06 |
| decode_attention | gpt-oss-20b | `attention_reduce` | seqs=1 ctx=32768 | 1.08 | 0.001 | 0.7 | 3.96 | 0.03 | memory | 64 | 0.06 |
| decode_attention | gpt-oss-20b | `attention_reduce` | seqs=1 ctx=8192 | 1.08 | 0.001 | 0.7 | 3.96 | 0.03 | memory | 64 | 0.06 |
| decode_attention | gpt-oss-20b | `attention_split` | seqs=1 ctx=1024 | 2.10 | 0.017 | 8.0 | 7.68 | 0.54 | memory | 512 | 3.56 |
| decode_attention | gpt-oss-20b | `attention_split` | seqs=1 ctx=32768 | 67.11 | 0.537 | 8.0 | 245.82 | 17.32 | memory | 512 | 3.56 |
| decode_attention | gpt-oss-20b | `attention_split` | seqs=1 ctx=8192 | 16.78 | 0.134 | 8.0 | 61.46 | 4.33 | memory | 512 | 3.56 |
| decode_attention | gpt-oss-20b | `batch_decode_attention` | seqs=8 ctx=1024 | 16.78 | 0.134 | 8.0 | 61.46 | 4.33 | memory | 4096 | 8.53 |
| decode_attention | gpt-oss-20b | `batch_decode_attention` | seqs=8 ctx=32768 | 536.87 | 4.295 | 8.0 | 1966.56 | 138.55 | memory | 4096 | 8.53 |
| decode_attention | gpt-oss-20b | `batch_decode_attention` | seqs=8 ctx=8192 | 134.22 | 1.074 | 8.0 | 491.64 | 34.64 | memory | 4096 | 8.53 |
| decode_attention | gpt-oss-20b | `batch_decode_combine` | seqs=8 ctx=1024 | 8.65 | 0.006 | 0.7 | 31.69 | 0.20 | memory | 512 | 0.33 |
| decode_attention | gpt-oss-20b | `batch_decode_combine` | seqs=8 ctx=32768 | 8.65 | 0.006 | 0.7 | 31.69 | 0.20 | memory | 512 | 0.33 |
| decode_attention | gpt-oss-20b | `batch_decode_combine` | seqs=8 ctx=8192 | 8.65 | 0.006 | 0.7 | 31.69 | 0.20 | memory | 512 | 0.33 |
| decode_attention | hd128-gqa4 | `attention_reduce` | seqs=1 ctx=1024 | 1.06 | 0.001 | 0.7 | 3.90 | 0.03 | memory | 32 | 0.06 |
| decode_attention | hd128-gqa4 | `attention_reduce` | seqs=1 ctx=32768 | 1.06 | 0.001 | 0.7 | 3.90 | 0.03 | memory | 32 | 0.06 |
| decode_attention | hd128-gqa4 | `attention_reduce` | seqs=1 ctx=8192 | 1.06 | 0.001 | 0.7 | 3.90 | 0.03 | memory | 32 | 0.06 |
| decode_attention | hd128-gqa4 | `attention_split` | seqs=1 ctx=1024 | 4.19 | 0.017 | 4.0 | 15.36 | 0.54 | memory | 512 | 1.52 |
| decode_attention | hd128-gqa4 | `attention_split` | seqs=1 ctx=32768 | 134.22 | 0.537 | 4.0 | 491.64 | 17.32 | memory | 512 | 1.52 |
| decode_attention | hd128-gqa4 | `attention_split` | seqs=1 ctx=8192 | 33.55 | 0.134 | 4.0 | 122.91 | 4.33 | memory | 512 | 1.52 |
| decode_attention | hd128-gqa4 | `batch_decode_attention` | seqs=8 ctx=1024 | 33.55 | 0.134 | 4.0 | 122.91 | 4.33 | memory | 4096 | 17.07 |
| decode_attention | hd128-gqa4 | `batch_decode_attention` | seqs=8 ctx=32768 | 1073.74 | 4.295 | 4.0 | 3933.12 | 138.55 | memory | 4096 | 17.07 |
| decode_attention | hd128-gqa4 | `batch_decode_attention` | seqs=8 ctx=8192 | 268.44 | 1.074 | 4.0 | 983.28 | 34.64 | memory | 4096 | 17.07 |
| decode_attention | hd128-gqa4 | `batch_decode_combine` | seqs=8 ctx=1024 | 8.52 | 0.006 | 0.7 | 31.21 | 0.20 | memory | 256 | 0.22 |
| decode_attention | hd128-gqa4 | `batch_decode_combine` | seqs=8 ctx=32768 | 8.52 | 0.006 | 0.7 | 31.21 | 0.20 | memory | 256 | 0.22 |
| decode_attention | hd128-gqa4 | `batch_decode_combine` | seqs=8 ctx=8192 | 8.52 | 0.006 | 0.7 | 31.21 | 0.20 | memory | 256 | 0.22 |
| decode_attention | hd128-gqa8 | `attention_reduce` | seqs=1 ctx=1024 | 1.06 | 0.001 | 0.7 | 3.90 | 0.03 | memory | 32 | 0.06 |
| decode_attention | hd128-gqa8 | `attention_reduce` | seqs=1 ctx=32768 | 1.06 | 0.001 | 0.7 | 3.90 | 0.03 | memory | 32 | 0.06 |
| decode_attention | hd128-gqa8 | `attention_reduce` | seqs=1 ctx=8192 | 1.06 | 0.001 | 0.7 | 3.90 | 0.03 | memory | 32 | 0.06 |
| decode_attention | hd128-gqa8 | `attention_split` | seqs=1 ctx=1024 | 2.10 | 0.017 | 8.0 | 7.68 | 0.54 | memory | 256 | 1.33 |
| decode_attention | hd128-gqa8 | `attention_split` | seqs=1 ctx=32768 | 67.11 | 0.537 | 8.0 | 245.82 | 17.32 | memory | 256 | 1.33 |
| decode_attention | hd128-gqa8 | `attention_split` | seqs=1 ctx=8192 | 16.78 | 0.134 | 8.0 | 61.46 | 4.33 | memory | 256 | 1.33 |
| decode_attention | hd128-gqa8 | `batch_decode_attention` | seqs=8 ctx=1024 | 16.78 | 0.134 | 8.0 | 61.46 | 4.33 | memory | 2048 | 8.53 |
| decode_attention | hd128-gqa8 | `batch_decode_attention` | seqs=8 ctx=32768 | 536.87 | 4.295 | 8.0 | 1966.56 | 138.55 | memory | 2048 | 8.53 |
| decode_attention | hd128-gqa8 | `batch_decode_attention` | seqs=8 ctx=8192 | 134.22 | 1.074 | 8.0 | 491.64 | 34.64 | memory | 2048 | 8.53 |
| decode_attention | hd128-gqa8 | `batch_decode_combine` | seqs=8 ctx=1024 | 8.52 | 0.006 | 0.7 | 31.21 | 0.20 | memory | 256 | 0.22 |
| decode_attention | hd128-gqa8 | `batch_decode_combine` | seqs=8 ctx=32768 | 8.52 | 0.006 | 0.7 | 31.21 | 0.20 | memory | 256 | 0.22 |
| decode_attention | hd128-gqa8 | `batch_decode_combine` | seqs=8 ctx=8192 | 8.52 | 0.006 | 0.7 | 31.21 | 0.20 | memory | 256 | 0.22 |
| fused_add_rmsnorm | hidden-2560 | `add_rmsnorm_2560` | rows=1 | 0.04 | 0.000 | 0.3 | 0.15 | 0.00 | memory | 1 | 0.02 |
| fused_add_rmsnorm | hidden-2560 | `add_rmsnorm_2560` | rows=4096 | 167.77 | 0.052 | 0.3 | 614.55 | 1.69 | memory | 4096 | 85.33 |
| fused_add_rmsnorm | hidden-5120 | `add_rmsnorm_5120` | rows=1 | 0.08 | 0.000 | 0.3 | 0.30 | 0.00 | memory | 1 | 0.02 |
| fused_add_rmsnorm | hidden-5120 | `add_rmsnorm_5120` | rows=4096 | 335.54 | 0.105 | 0.3 | 1229.10 | 3.38 | memory | 4096 | 85.33 |
| nvfp4_gemm | 122b-e128 | `down_f4a_big` | m=4096 n=3072 k=1024 | 68.88 | 25.770 | 374.1 | 252.30 | 51.54 | memory | 384 | 8.00 |
| nvfp4_gemm | 122b-e128 | `gate_up_f4a_big` | m=4096 n=2048 k=3072 | 87.43 | 51.540 | 589.5 | 320.24 | 103.08 | memory | 256 | 5.33 |
| nvfp4_gemm | 122b-e128 | `quant_rows_fp4` | m=4096 n=3072 k=1024 | 19.14 | 0.017 | 0.9 | 70.10 | 0.54 | memory | 4096 | 14.22 |
| nvfp4_gemm | 122b-e128 | `quant_rows_fp4` | m=4096 n=2048 k=3072 | 57.41 | 0.050 | 0.9 | 210.29 | 1.62 | memory | 4096 | 14.22 |
| nvfp4_gemm | 122b-nvfp4 | `down_f4a_big` | m=4096 n=3072 k=1024 | 68.88 | 25.770 | 374.1 | 252.30 | 51.54 | memory | 384 | 8.00 |
| nvfp4_gemm | 122b-nvfp4 | `gate_up_f4a_big` | m=4096 n=2048 k=3072 | 87.43 | 51.540 | 589.5 | 320.24 | 103.08 | memory | 256 | 5.33 |
| nvfp4_gemm | 122b-nvfp4 | `quant_rows_fp4` | m=4096 n=3072 k=1024 | 19.14 | 0.017 | 0.9 | 70.10 | 0.54 | memory | 4096 | 14.22 |
| nvfp4_gemm | 122b-nvfp4 | `quant_rows_fp4` | m=4096 n=2048 k=3072 | 57.41 | 0.050 | 0.9 | 210.29 | 1.62 | memory | 4096 | 14.22 |
| nvfp4_gemm | 27b-nvfp4 | `down_f4a_big` | m=4096 n=5120 k=17408 | 419.23 | 730.144 | 1741.6 | 1535.65 | 1460.29 | memory | 640 | 13.33 |
| nvfp4_gemm | 27b-nvfp4 | `gate_up_f4a_big` | m=4096 n=34816 k=5120 | 754.58 | 1460.289 | 1935.2 | 2764.03 | 2920.58 | compute fp4 | 4352 | 90.67 |
| nvfp4_gemm | 27b-nvfp4 | `quant_rows_fp4` | m=4096 n=5120 k=17408 | 325.32 | 0.285 | 0.9 | 1191.65 | 9.20 | memory | 4096 | 14.22 |
| nvfp4_gemm | 27b-nvfp4 | `quant_rows_fp4` | m=4096 n=34816 k=5120 | 95.68 | 0.084 | 0.9 | 350.49 | 2.71 | memory | 4096 | 14.22 |
| nvfp4_gemm | 27b-nvfp4-fp8kv | `down_f4a_big` | m=4096 n=5120 k=17408 | 419.23 | 730.144 | 1741.6 | 1535.65 | 1460.29 | memory | 640 | 13.33 |
| nvfp4_gemm | 27b-nvfp4-fp8kv | `gate_up_f4a_big` | m=4096 n=34816 k=5120 | 754.58 | 1460.289 | 1935.2 | 2764.03 | 2920.58 | compute fp4 | 4352 | 90.67 |
| nvfp4_gemm | 27b-nvfp4-fp8kv | `quant_rows_fp4` | m=4096 n=5120 k=17408 | 325.32 | 0.285 | 0.9 | 1191.65 | 9.20 | memory | 4096 | 14.22 |
| nvfp4_gemm | 27b-nvfp4-fp8kv | `quant_rows_fp4` | m=4096 n=34816 k=5120 | 95.68 | 0.084 | 0.9 | 350.49 | 2.71 | memory | 4096 | 14.22 |
| nvfp4_gemm | hd128-gqa4 | `down_f4a_big` | m=4096 n=5120 k=17408 | 419.23 | 730.144 | 1741.6 | 1535.65 | 1460.29 | memory | 640 | 13.33 |
| nvfp4_gemm | hd128-gqa4 | `gate_up_f4a_big` | m=4096 n=34816 k=5120 | 754.58 | 1460.289 | 1935.2 | 2764.03 | 2920.58 | compute fp4 | 4352 | 90.67 |
| nvfp4_gemm | hd128-gqa4 | `quant_rows_fp4` | m=4096 n=5120 k=17408 | 325.32 | 0.285 | 0.9 | 1191.65 | 9.20 | memory | 4096 | 14.22 |
| nvfp4_gemm | hd128-gqa4 | `quant_rows_fp4` | m=4096 n=34816 k=5120 | 95.68 | 0.084 | 0.9 | 350.49 | 2.71 | memory | 4096 | 14.22 |
| nvfp4_gemm | hd128-gqa8 | `down_f4a_big` | m=4096 n=5120 k=17408 | 419.23 | 730.144 | 1741.6 | 1535.65 | 1460.29 | memory | 640 | 13.33 |
| nvfp4_gemm | hd128-gqa8 | `gate_up_f4a_big` | m=4096 n=34816 k=5120 | 754.58 | 1460.289 | 1935.2 | 2764.03 | 2920.58 | compute fp4 | 4352 | 90.67 |
| nvfp4_gemm | hd128-gqa8 | `quant_rows_fp4` | m=4096 n=5120 k=17408 | 325.32 | 0.285 | 0.9 | 1191.65 | 9.20 | memory | 4096 | 14.22 |
| nvfp4_gemm | hd128-gqa8 | `quant_rows_fp4` | m=4096 n=34816 k=5120 | 95.68 | 0.084 | 0.9 | 350.49 | 2.71 | memory | 4096 | 14.22 |
| nvfp4_gemv | 122b-e128 | `down_proj_f4` | m=1 n=3072 k=1024 | 1.79 | 0.006 | 3.5 | 6.54 | 0.20 | memory | 192 | 1.33 |
| nvfp4_gemv | 122b-e128 | `gate_up_act_f4` | m=1 n=2048 k=3072 | 3.56 | 0.013 | 3.5 | 13.04 | 0.41 | memory | 64 | 0.44 |
| nvfp4_gemv | 122b-nvfp4 | `down_proj_f4` | m=1 n=3072 k=1024 | 1.79 | 0.006 | 3.5 | 6.54 | 0.20 | memory | 192 | 1.33 |
| nvfp4_gemv | 122b-nvfp4 | `gate_up_act_f4` | m=1 n=2048 k=3072 | 3.56 | 0.013 | 3.5 | 13.04 | 0.41 | memory | 64 | 0.44 |
| nvfp4_gemv | 27b-nvfp4 | `down_proj_f4` | m=1 n=5120 k=17408 | 50.23 | 0.178 | 3.5 | 183.97 | 5.75 | memory | 320 | 2.22 |
| nvfp4_gemv | 27b-nvfp4 | `gate_up_act_f4` | m=1 n=34816 k=5120 | 100.43 | 0.357 | 3.5 | 367.87 | 11.50 | memory | 1088 | 7.56 |
| nvfp4_gemv | 27b-nvfp4-fp8kv | `down_proj_f4` | m=1 n=5120 k=17408 | 50.23 | 0.178 | 3.5 | 183.97 | 5.75 | memory | 320 | 2.22 |
| nvfp4_gemv | 27b-nvfp4-fp8kv | `gate_up_act_f4` | m=1 n=34816 k=5120 | 100.43 | 0.357 | 3.5 | 367.87 | 11.50 | memory | 1088 | 7.56 |
| nvfp4_gemv | hd128-gqa4 | `down_proj_f4` | m=1 n=5120 k=17408 | 50.23 | 0.178 | 3.5 | 183.97 | 5.75 | memory | 320 | 2.22 |
| nvfp4_gemv | hd128-gqa4 | `gate_up_act_f4` | m=1 n=34816 k=5120 | 100.43 | 0.357 | 3.5 | 367.87 | 11.50 | memory | 1088 | 7.56 |
| nvfp4_gemv | hd128-gqa8 | `down_proj_f4` | m=1 n=5120 k=17408 | 50.23 | 0.178 | 3.5 | 183.97 | 5.75 | memory | 320 | 2.22 |
| nvfp4_gemv | hd128-gqa8 | `gate_up_act_f4` | m=1 n=34816 k=5120 | 100.43 | 0.357 | 3.5 | 367.87 | 11.50 | memory | 1088 | 7.56 |
| prefill_attention | 122b-e128 | `flash_mma` | tokens=1024 | 52.43 | 17.180 | 327.7 | 192.05 | 137.44 | memory | 256 | 5.33 |
| prefill_attention | 122b-e128 | `flash_mma` | tokens=16384 | 838.86 | 4398.047 | 5242.9 | 3072.75 | 35184.37 | compute bf16 | 4096 | 85.33 |
| prefill_attention | 122b-e128 | `flash_mma` | tokens=4096 | 209.72 | 274.878 | 1310.7 | 768.19 | 2199.02 | compute bf16 | 1024 | 21.33 |
| prefill_attention | 122b-nvfp4 | `flash_mma` | tokens=1024 | 52.43 | 17.180 | 327.7 | 192.05 | 137.44 | memory | 256 | 5.33 |
| prefill_attention | 122b-nvfp4 | `flash_mma` | tokens=16384 | 838.86 | 4398.047 | 5242.9 | 3072.75 | 35184.37 | compute bf16 | 4096 | 85.33 |
| prefill_attention | 122b-nvfp4 | `flash_mma` | tokens=4096 | 209.72 | 274.878 | 1310.7 | 768.19 | 2199.02 | compute bf16 | 1024 | 21.33 |
| prefill_attention | 27b-nvfp4 | `flash_mma` | tokens=1024 | 41.94 | 12.885 | 307.2 | 153.64 | 103.08 | memory | 192 | 4.00 |
| prefill_attention | 27b-nvfp4 | `flash_mma` | tokens=16384 | 671.09 | 3298.535 | 4915.2 | 2458.20 | 26388.28 | compute bf16 | 3072 | 64.00 |
| prefill_attention | 27b-nvfp4 | `flash_mma` | tokens=4096 | 167.77 | 206.158 | 1228.8 | 614.55 | 1649.27 | compute bf16 | 768 | 16.00 |
| prefill_attention | 27b-nvfp4-fp8kv | `flash_mma` | tokens=1024 | 41.94 | 12.885 | 307.2 | 153.64 | 103.08 | memory | 192 | 4.00 |
| prefill_attention | 27b-nvfp4-fp8kv | `flash_mma` | tokens=16384 | 671.09 | 3298.535 | 4915.2 | 2458.20 | 26388.28 | compute bf16 | 3072 | 64.00 |
| prefill_attention | 27b-nvfp4-fp8kv | `flash_mma` | tokens=4096 | 167.77 | 206.158 | 1228.8 | 614.55 | 1649.27 | compute bf16 | 768 | 16.00 |
| prefill_attention | 4b-bf16 | `flash_mma` | tokens=1024 | 29.36 | 8.590 | 292.6 | 107.55 | 68.72 | memory | 128 | 2.67 |
| prefill_attention | 4b-bf16 | `flash_mma` | tokens=16384 | 469.76 | 2199.023 | 4681.1 | 1720.74 | 17592.19 | compute bf16 | 2048 | 42.67 |
| prefill_attention | 4b-bf16 | `flash_mma` | tokens=4096 | 117.44 | 137.439 | 1170.3 | 430.19 | 1099.51 | compute bf16 | 512 | 10.67 |
| prefill_attention | gpt-oss-20b | `flash_mma` | tokens=1024 | 27.26 | 8.590 | 315.1 | 99.86 | 68.72 | memory | 1024 | 5.33 |
| prefill_attention | gpt-oss-20b | `flash_mma` | tokens=16384 | 436.21 | 2199.023 | 5041.2 | 1597.83 | 17592.19 | compute bf16 | 16384 | 85.33 |
| prefill_attention | gpt-oss-20b | `flash_mma` | tokens=4096 | 109.05 | 137.439 | 1260.3 | 399.46 | 1099.51 | compute bf16 | 4096 | 21.33 |
| prefill_attention | hd128-gqa4 | `flash_mma` | tokens=1024 | 29.36 | 8.590 | 292.6 | 107.55 | 68.72 | memory | 512 | 5.33 |
| prefill_attention | hd128-gqa4 | `flash_mma` | tokens=16384 | 469.76 | 2199.023 | 4681.1 | 1720.74 | 17592.19 | compute bf16 | 8192 | 85.33 |
| prefill_attention | hd128-gqa4 | `flash_mma` | tokens=4096 | 117.44 | 137.439 | 1170.3 | 430.19 | 1099.51 | compute bf16 | 2048 | 21.33 |
| prefill_attention | hd128-gqa8 | `flash_mma` | tokens=1024 | 27.26 | 8.590 | 315.1 | 99.86 | 68.72 | memory | 512 | 5.33 |
| prefill_attention | hd128-gqa8 | `flash_mma` | tokens=16384 | 436.21 | 2199.023 | 5041.2 | 1597.83 | 17592.19 | compute bf16 | 8192 | 85.33 |
| prefill_attention | hd128-gqa8 | `flash_mma` | tokens=4096 | 109.05 | 137.439 | 1260.3 | 399.46 | 1099.51 | compute bf16 | 2048 | 21.33 |
| rmsnorm | 122b-e128 | `rmsnorm` | rows=1 | 0.02 | 0.000 | 0.5 | 0.09 | 0.00 | memory | 1 | 0.02 |
| rmsnorm | 122b-e128 | `rmsnorm` | rows=4096 | 100.66 | 0.050 | 0.5 | 368.73 | 1.62 | memory | 4096 | 85.33 |
| rmsnorm | 122b-nvfp4 | `rmsnorm` | rows=1 | 0.02 | 0.000 | 0.5 | 0.09 | 0.00 | memory | 1 | 0.02 |
| rmsnorm | 122b-nvfp4 | `rmsnorm` | rows=4096 | 100.66 | 0.050 | 0.5 | 368.73 | 1.62 | memory | 4096 | 85.33 |
| rmsnorm | 27b-nvfp4 | `rmsnorm` | rows=1 | 0.04 | 0.000 | 0.5 | 0.15 | 0.00 | memory | 1 | 0.02 |
| rmsnorm | 27b-nvfp4 | `rmsnorm` | rows=4096 | 167.77 | 0.084 | 0.5 | 614.55 | 2.71 | memory | 4096 | 85.33 |
| rmsnorm | 27b-nvfp4-fp8kv | `rmsnorm` | rows=1 | 0.04 | 0.000 | 0.5 | 0.15 | 0.00 | memory | 1 | 0.02 |
| rmsnorm | 27b-nvfp4-fp8kv | `rmsnorm` | rows=4096 | 167.77 | 0.084 | 0.5 | 614.55 | 2.71 | memory | 4096 | 85.33 |
| rmsnorm | 4b-bf16 | `rmsnorm` | rows=1 | 0.02 | 0.000 | 0.5 | 0.08 | 0.00 | memory | 1 | 0.02 |
| rmsnorm | 4b-bf16 | `rmsnorm` | rows=4096 | 83.89 | 0.042 | 0.5 | 307.28 | 1.35 | memory | 4096 | 85.33 |
| rmsnorm | gpt-oss-20b | `rmsnorm` | rows=1 | 0.02 | 0.000 | 0.5 | 0.08 | 0.00 | memory | 1 | 0.02 |
| rmsnorm | gpt-oss-20b | `rmsnorm` | rows=4096 | 94.37 | 0.047 | 0.5 | 345.68 | 1.52 | memory | 4096 | 85.33 |
| rmsnorm | hd128-gqa4 | `rmsnorm` | rows=1 | 0.04 | 0.000 | 0.5 | 0.15 | 0.00 | memory | 1 | 0.02 |
| rmsnorm | hd128-gqa4 | `rmsnorm` | rows=4096 | 167.77 | 0.084 | 0.5 | 614.55 | 2.71 | memory | 4096 | 85.33 |
| rmsnorm | hd128-gqa8 | `rmsnorm` | rows=1 | 0.04 | 0.000 | 0.5 | 0.15 | 0.00 | memory | 1 | 0.02 |
| rmsnorm | hd128-gqa8 | `rmsnorm` | rows=4096 | 167.77 | 0.084 | 0.5 | 614.55 | 2.71 | memory | 4096 | 85.33 |
| rotary | 122b-e128 | `qk_norm_rope` | rows=1 | 0.14 | 0.000 | 0.4 | 0.51 | 0.00 | memory | 36 | 0.12 |
| rotary | 122b-e128 | `qk_norm_rope` | rows=4096 | 570.43 | 0.214 | 0.4 | 2089.47 | 6.90 | memory | 147456 | 512.00 |
| rotary | 122b-nvfp4 | `qk_norm_rope` | rows=1 | 0.14 | 0.000 | 0.4 | 0.51 | 0.00 | memory | 36 | 0.12 |
| rotary | 122b-nvfp4 | `qk_norm_rope` | rows=4096 | 570.43 | 0.214 | 0.4 | 2089.47 | 6.90 | memory | 147456 | 512.00 |
| rotary | 27b-nvfp4 | `qk_norm_rope` | rows=1 | 0.11 | 0.000 | 0.4 | 0.42 | 0.00 | memory | 32 | 0.11 |
| rotary | 27b-nvfp4 | `qk_norm_rope` | rows=4096 | 469.76 | 0.176 | 0.4 | 1720.74 | 5.68 | memory | 131072 | 455.11 |
| rotary | 27b-nvfp4-fp8kv | `qk_norm_rope` | rows=1 | 0.11 | 0.000 | 0.4 | 0.42 | 0.00 | memory | 32 | 0.11 |
| rotary | 27b-nvfp4-fp8kv | `qk_norm_rope` | rows=4096 | 469.76 | 0.176 | 0.4 | 1720.74 | 5.68 | memory | 131072 | 455.11 |
| rotary | 4b-bf16 | `qk_norm_rope` | rows=1 | 0.08 | 0.000 | 0.4 | 0.30 | 0.00 | memory | 24 | 0.08 |
| rotary | 4b-bf16 | `qk_norm_rope` | rows=4096 | 335.54 | 0.126 | 0.4 | 1229.10 | 4.06 | memory | 98304 | 341.33 |
| rotary | gpt-oss-20b | `qk_norm_rope` | rows=1 | 0.04 | 0.000 | 0.7 | 0.15 | 0.00 | memory | 80 | 0.07 |
| rotary | gpt-oss-20b | `qk_norm_rope` | rows=4096 | 167.77 | 0.113 | 0.7 | 614.55 | 3.65 | memory | 327680 | 284.44 |
| rotary | hd128-gqa4 | `qk_norm_rope` | rows=1 | 0.08 | 0.000 | 0.4 | 0.30 | 0.00 | memory | 48 | 0.08 |
| rotary | hd128-gqa4 | `qk_norm_rope` | rows=4096 | 335.54 | 0.126 | 0.4 | 1229.10 | 4.06 | memory | 196608 | 341.33 |
| rotary | hd128-gqa8 | `qk_norm_rope` | rows=1 | 0.07 | 0.000 | 0.4 | 0.27 | 0.00 | memory | 40 | 0.07 |
| rotary | hd128-gqa8 | `qk_norm_rope` | rows=4096 | 301.99 | 0.113 | 0.4 | 1106.19 | 3.65 | memory | 163840 | 284.44 |
| silu_and_mul | 122b-e128 | `swiglu` | rows=1 | 0.01 | 0.000 | 0.5 | 0.05 | 0.00 | memory | 4 | 0.01 |
| silu_and_mul | 122b-e128 | `swiglu` | rows=4096 | 50.33 | 0.025 | 0.5 | 184.37 | 0.81 | memory | 16384 | 56.89 |
| silu_and_mul | 122b-nvfp4 | `swiglu` | rows=1 | 0.01 | 0.000 | 0.5 | 0.05 | 0.00 | memory | 4 | 0.01 |
| silu_and_mul | 122b-nvfp4 | `swiglu` | rows=4096 | 50.33 | 0.025 | 0.5 | 184.37 | 0.81 | memory | 16384 | 56.89 |
| silu_and_mul | 27b-nvfp4 | `swiglu` | rows=1 | 0.21 | 0.000 | 0.5 | 0.77 | 0.00 | memory | 68 | 0.24 |
| silu_and_mul | 27b-nvfp4 | `swiglu` | rows=4096 | 855.64 | 0.428 | 0.5 | 3134.21 | 13.80 | memory | 278528 | 967.11 |
| silu_and_mul | 27b-nvfp4-fp8kv | `swiglu` | rows=1 | 0.21 | 0.000 | 0.5 | 0.77 | 0.00 | memory | 68 | 0.24 |
| silu_and_mul | 27b-nvfp4-fp8kv | `swiglu` | rows=4096 | 855.64 | 0.428 | 0.5 | 3134.21 | 13.80 | memory | 278528 | 967.11 |
| silu_and_mul | 4b-bf16 | `swiglu` | rows=1 | 0.11 | 0.000 | 0.5 | 0.41 | 0.00 | memory | 36 | 0.12 |
| silu_and_mul | 4b-bf16 | `swiglu` | rows=4096 | 452.98 | 0.226 | 0.5 | 1659.29 | 7.31 | memory | 147456 | 512.00 |
| silu_and_mul | gpt-oss-20b | `swiglu` | rows=1 | 0.03 | 0.000 | 0.5 | 0.13 | 0.00 | memory | 12 | 0.04 |
| silu_and_mul | gpt-oss-20b | `swiglu` | rows=4096 | 141.56 | 0.071 | 0.5 | 518.53 | 2.28 | memory | 49152 | 170.67 |
| silu_and_mul | hd128-gqa4 | `swiglu` | rows=1 | 0.21 | 0.000 | 0.5 | 0.77 | 0.00 | memory | 68 | 0.24 |
| silu_and_mul | hd128-gqa4 | `swiglu` | rows=4096 | 855.64 | 0.428 | 0.5 | 3134.21 | 13.80 | memory | 278528 | 967.11 |
| silu_and_mul | hd128-gqa8 | `swiglu` | rows=1 | 0.21 | 0.000 | 0.5 | 0.77 | 0.00 | memory | 68 | 0.24 |
| silu_and_mul | hd128-gqa8 | `swiglu` | rows=4096 | 855.64 | 0.428 | 0.5 | 3134.21 | 13.80 | memory | 278528 | 967.11 |
| topk_routing | 122b-e128 | `moe_route` | tokens=1 | 0.00 | 0.000 | 2.0 | 0.00 | 0.00 | memory | 1 | 0.00 |
| topk_routing | 122b-e128 | `moe_route` | tokens=4096 | 2.10 | 0.004 | 2.0 | 7.68 | 0.14 | memory | 4096 | 2.67 |
| topk_routing | 122b-e128 | `moe_route_logits` | tokens=1 | 0.80 | 0.001 | 1.0 | 2.93 | 0.03 | memory | 17 | 0.35 |
| topk_routing | 122b-e128 | `moe_route_logits` | tokens=4096 | 51.12 | 3.221 | 63.0 | 187.25 | 103.91 | memory | 17408 | 362.67 |
| topk_routing | 122b-nvfp4 | `moe_route` | tokens=1 | 0.00 | 0.000 | 2.0 | 0.00 | 0.00 | memory | 1 | 0.00 |
| topk_routing | 122b-nvfp4 | `moe_route` | tokens=4096 | 4.19 | 0.008 | 2.0 | 15.36 | 0.27 | memory | 4096 | 3.56 |
| topk_routing | 122b-nvfp4 | `moe_route_logits` | tokens=1 | 1.59 | 0.002 | 1.0 | 5.81 | 0.05 | memory | 33 | 0.69 |
| topk_routing | 122b-nvfp4 | `moe_route_logits` | tokens=4096 | 51.90 | 6.442 | 124.1 | 190.13 | 207.82 | compute fp32 | 33792 | 704.00 |
| topk_routing | gpt-oss-20b | `moe_route` | tokens=1 | 0.00 | 0.000 | 2.0 | 0.00 | 0.00 | memory | 1 | 0.00 |
| topk_routing | gpt-oss-20b | `moe_route` | tokens=4096 | 0.52 | 0.001 | 2.0 | 1.92 | 0.03 | memory | 4096 | 2.67 |
| topk_routing | gpt-oss-20b | `moe_route_logits` | tokens=1 | 0.20 | 0.000 | 0.9 | 0.72 | 0.01 | memory | 5 | 0.05 |
| topk_routing | gpt-oss-20b | `moe_route_logits` | tokens=4096 | 47.37 | 0.755 | 15.9 | 173.52 | 24.35 | memory | 5120 | 53.33 |

## sm_75: RTX 2060 (windows)

Target `target Cuda cc=7.5 sms=30 warp=32 threads=1024 smem=65536 graphs=true pdl=false features=[WarpShuffle]`, NVRTC 13.2 (--std=c++17 no --split-compile (it makes register allocation nondeterministic)). Peak numbers: TU106 RTX 2060: 336 GB/s, 30 SMs, 6.5 TFLOP/s FP32, no bf16 or FP4 tensor cores.

- `27b-nvfp4` does not compile for sm_75: flash_attention: FlashScalar: needs 84224 bytes of shared memory, the target allows 65536
- `27b-nvfp4-fp8kv` does not compile for sm_75: flash_attention: FlashScalar: needs 84224 bytes of shared memory, the target allows 65536
- `4b-bf16` does not compile for sm_75: flash_attention: FlashScalar: needs 84224 bytes of shared memory, the target allows 65536
- `122b-nvfp4` does not compile for sm_75: flash_attention: FlashScalar: needs 84224 bytes of shared memory, the target allows 65536
- `122b-e128` does not compile for sm_75: flash_attention: FlashScalar: needs 84224 bytes of shared memory, the target allows 65536
- `glm4.5-air` does not compile for sm_75: ptxas error   : Entry function 'moe_route_logits' uses too much shared data (0x10080 bytes, 0xc000 max)
- `hd128-gqa8` does not compile for sm_75: GdnChunk: needs Mma16816, which the target lacks
- `hd128-gqa4` does not compile for sm_75: GdnChunk: needs Mma16816, which the target lacks


### bf16_gemv

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-oss-20b | `down_proj` | 256 | 66 | 0 | 0 | 3 (regs) | 768 | 193, 307 | ldg128 24 fma 96 | cuBLAS GEMV is closed source; no static artefact | | | | | | | | |
| gpt-oss-20b | `gate_up_act` | 256 | 62 | 11648 | 0 | 4 (regs) | 1024 | 190, 274 | ldg128 8 lds 16 fma 96 | cuBLAS GEMV is closed source; no static artefact | | | | | | | | |
| gpt-oss-20b | `qkv_proj_norm` | 256 | 61 | 11648 | 0 | 4 (regs) | 1024 | 190, 279 | ldg128 8 lds 16 fma 96 | cuBLAS GEMV is closed source; no static artefact | | | | | | | | |

### decode_attention

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-oss-20b | `attention_reduce` | 64 | 41 | 272 | 0 | 16 (threads) | 1024 | 43, 147 | ldg32 16 lds 4 fma 16 | no competitor built for this target | | | | | | | | |
| gpt-oss-20b | `attention_split` | 128 | 128 | 1280 | 16 | 4 (regs) | 512 | 2956, 7247 | lds 96 fma 1342 mufu 90 bar 17 | no competitor built for this target | | | | | | | | |

### fused_add_rmsnorm

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hidden-2560 | `add_rmsnorm_2560` | 1024 | 26 | 128 | 0 | 1 (threads) | 1024 | 25, 88 | lds 4 fma 16 | FlashInfer 0.7.3 `flashinfer4norm21FusedAddRMSNormKernelILj8E13__n` | 320 | 41 | 0 | 0 | 960 | 211, 406 | ldg128 1 lds 3 fma 30 mufu 2 bar 2 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `add_rmsnorm` | 256 | 45 | 32 | 0 | 1024 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm25fused_add_rms_norm_kernelIN3c108BFloat16EL` | 320 | 27 | 168 | 0 | 960 | 27, 90 | ldg32 2 fma 2 |  |
| hidden-5120 | `add_rmsnorm_5120` | 1024 | 26 | 128 | 0 | 1 (threads) | 1024 | 25, 88 | lds 4 fma 16 | FlashInfer 0.7.3 `flashinfer4norm21FusedAddRMSNormKernelILj8E13__n` | 640 | 41 | 0 | 0 | 640 | 211, 406 | ldg128 1 lds 3 fma 30 mufu 2 bar 2 | behind: 32-bit loads only |
| | | | | | | | | | | Triton 3.8.0 `add_rmsnorm` | 256 | 57 | 32 | 0 | 1024 | 1, 1 |  |  |
| | | | | | | | | | | vLLM prebuilt `vllm25fused_add_rms_norm_kernelIN3c108BFloat16EL` | 640 | 27 | 168 | 0 | 640 | 27, 90 | ldg32 2 fma 2 |  |

### rmsnorm

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-oss-20b | `rmsnorm` | 1024 | 26 | 128 | 0 | 1 (threads) | 1024 | 25, 88 | lds 4 fma 16 | FlashInfer 0.7.3 `flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat` | 360 | 45 | 0 | 0 | 768 | 219, 418 | ldg128 2 lds 1 fma 30 mufu 2 bar 2 | behind: 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0` | 360 | 27 | 168 | 0 | 768 | 78, 144 | ldg128 1 fma 8 |  |

### rotary

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-oss-20b | `qk_norm_rope` | 64 | 25 | 256 | 32 | 16 (threads) | 1024 | 11, 28 | ldg32 1 local 1 | FlashInfer 0.7.3 `flashinfer56BatchQKApplyRotaryPosIdsCosSinCacheH` | 128 | 40 | 0 | 0 | 1024 | 1, 1 |  | behind: spills 32 B, 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm23rotary_embedding_kernelIN3c108BFloat16ES2_` | 512 | 33 | 0 | 0 | 1024 | 74, 193 | ldg32 4 fma 4 |  |

### silu_and_mul

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-oss-20b | `swiglu` | 256 | 19 | 0 | 0 | 4 (threads) | 1024 | 40, 148 | ldg32 2 fma 12 mufu 2 | FlashInfer 0.7.3 `flashinfer10activation18act_and_mul_kernelI13__n` | 256 | 33 | 0 | 0 | 1024 | 180, 380 | ldg128 2 fma 32 mufu 16 | behind: 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_b` | 1024 | 24 | 0 | 0 | 1024 | 92, 313 | ldg32 2 fma 18 mufu 2 |  |

### topk_routing

| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-oss-20b | `moe_route` | 32 | 38 | 288 | 0 | 16 (blocks) | 512 | 231, 781 | lds 32 fma 1 shfl 10 | Triton 3.8.0 `softmax_topk` | 128 | 19 | 0 | 0 | 1024 | 1, 1 |  | behind: 32-bit loads only |
| | | | | | | | | | | vLLM prebuilt `vllm3moe10topkGatingILi8ELi128ELi4ELi16ELi32Ei13` | 128 | 35 | 0 | 32 | 1024 | 368, 1163 | ldg32 6 fma 12 mufu 1 shfl 12 local 4 |  |
| gpt-oss-20b | `moe_route_logits` | 256 | 98 | 46208 | 0 | 1 (smem) | 256 | 509, 1585 | ldg128 31 ldg32 28 lds 10 fma 227 mufu 1 shfl 5 bar 2 | no competitor built for this target | | | | | | | | |

### Roofline (algorithmic bytes and FLOPs, peak numbers above)

| family | spec | kernel | shape | MB | GFLOP | FLOP/B | mem us | math us | bound | blocks | waves |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bf16_gemv | gpt-oss-20b | `down_proj` | m=1 n=2880 k=2880 | 16.61 | 0.017 | 1.0 | 49.44 | 2.55 | memory | 360 | 4.00 |
| bf16_gemv | gpt-oss-20b | `gate_up_act` | m=1 n=5760 k=2880 | 33.21 | 0.033 | 1.0 | 98.85 | 5.10 | memory | 360 | 3.00 |
| bf16_gemv | gpt-oss-20b | `qkv_proj_norm` | m=1 n=5120 k=2880 | 29.52 | 0.029 | 1.0 | 87.87 | 4.54 | memory | 640 | 5.33 |
| decode_attention | gpt-oss-20b | `attention_reduce` | seqs=1 ctx=1024 | 1.08 | 0.001 | 0.7 | 3.22 | 0.12 | memory | 64 | 0.13 |
| decode_attention | gpt-oss-20b | `attention_reduce` | seqs=1 ctx=32768 | 1.08 | 0.001 | 0.7 | 3.22 | 0.12 | memory | 64 | 0.13 |
| decode_attention | gpt-oss-20b | `attention_reduce` | seqs=1 ctx=8192 | 1.08 | 0.001 | 0.7 | 3.22 | 0.12 | memory | 64 | 0.13 |
| decode_attention | gpt-oss-20b | `attention_split` | seqs=1 ctx=1024 | 2.10 | 0.017 | 8.0 | 6.24 | 2.58 | memory | 512 | 4.27 |
| decode_attention | gpt-oss-20b | `attention_split` | seqs=1 ctx=32768 | 67.11 | 0.537 | 8.0 | 199.73 | 82.60 | memory | 512 | 4.27 |
| decode_attention | gpt-oss-20b | `attention_split` | seqs=1 ctx=8192 | 16.78 | 0.134 | 8.0 | 49.93 | 20.65 | memory | 512 | 4.27 |
| fused_add_rmsnorm | hidden-2560 | `add_rmsnorm_2560` | rows=1 | 0.04 | 0.000 | 0.3 | 0.12 | 0.00 | memory | 1 | 0.03 |
| fused_add_rmsnorm | hidden-2560 | `add_rmsnorm_2560` | rows=4096 | 167.77 | 0.052 | 0.3 | 499.32 | 8.07 | memory | 4096 | 136.53 |
| fused_add_rmsnorm | hidden-5120 | `add_rmsnorm_5120` | rows=1 | 0.08 | 0.000 | 0.3 | 0.24 | 0.00 | memory | 1 | 0.03 |
| fused_add_rmsnorm | hidden-5120 | `add_rmsnorm_5120` | rows=4096 | 335.54 | 0.105 | 0.3 | 998.64 | 16.13 | memory | 4096 | 136.53 |
| rmsnorm | gpt-oss-20b | `rmsnorm` | rows=1 | 0.02 | 0.000 | 0.5 | 0.07 | 0.00 | memory | 1 | 0.03 |
| rmsnorm | gpt-oss-20b | `rmsnorm` | rows=4096 | 94.37 | 0.047 | 0.5 | 280.87 | 7.26 | memory | 4096 | 136.53 |
| rotary | gpt-oss-20b | `qk_norm_rope` | rows=1 | 0.04 | 0.000 | 0.7 | 0.12 | 0.00 | memory | 80 | 0.17 |
| rotary | gpt-oss-20b | `qk_norm_rope` | rows=4096 | 167.77 | 0.113 | 0.7 | 499.32 | 17.42 | memory | 327680 | 682.67 |
| silu_and_mul | gpt-oss-20b | `swiglu` | rows=1 | 0.03 | 0.000 | 0.5 | 0.10 | 0.00 | memory | 12 | 0.10 |
| silu_and_mul | gpt-oss-20b | `swiglu` | rows=4096 | 141.56 | 0.071 | 0.5 | 421.30 | 10.89 | memory | 49152 | 409.60 |
| topk_routing | gpt-oss-20b | `moe_route` | tokens=1 | 0.00 | 0.000 | 2.0 | 0.00 | 0.00 | memory | 1 | 0.00 |
| topk_routing | gpt-oss-20b | `moe_route` | tokens=4096 | 0.52 | 0.001 | 2.0 | 1.56 | 0.16 | memory | 4096 | 8.53 |
| topk_routing | gpt-oss-20b | `moe_route_logits` | tokens=1 | 0.20 | 0.000 | 0.9 | 0.58 | 0.03 | memory | 5 | 0.17 |
| topk_routing | gpt-oss-20b | `moe_route_logits` | tokens=4096 | 47.37 | 0.755 | 15.9 | 140.98 | 116.15 | memory | 5120 | 170.67 |
