# GGUF in specrt

specrt loads a llama.cpp GGUF file directly. `model` is the path of the `.gguf` file, in the server config or the Models page. The spec, tokenizer, chat template and stop tokens all come from the file, so no `config.json` or `tokenizer.json` is needed.

## Supported

| Item | Supported |
| --- | --- |
| Architectures | `llama`, `qwen2`, `qwen3` (dense). Others fail with "GGUF architecture X is not supported". |
| Tensor types | F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K. Any other type fails at load and in the library with the tensor name and type. |
| Tokenizer | `tokenizer.ggml.model = gpt2` with `pre` of `qwen2` or `llama-bpe` |
| Files | one file. Split GGUF (`-0000N-of-0000M`) is listed but marked unsupported for specrt. |

## Layout

- `crates/specrt/src/gguf.rs`: parser (header, metadata, tensor infos, alignment, mmap), block dequantisation ported from `ggml-quants.c`, `Spec` from metadata, tokenizer and chat template from metadata, and `DeviceRows`, the device layout for K-quants.
- `weights.rs`: `Checkpoint` opens a `.gguf` and answers the HF tensor names (`model.layers.N.self_attn.q_proj.weight`) by mapping them to `blk.N.attn_q.weight`.
- GGUF stores shapes as `[cols, rows]`; the reader reverses them.
- `llama` GGUF files store Q and K rows permuted for interleaved rope. specrt uses the half-split (neox) rope, so the reader undoes the permutation by reordering whole rows, which works on quantised rows too.
- Llama 3 rope scaling is stored as a `rope_freqs.weight` tensor. The reader takes `factor` from its maximum, uses the converter defaults (low 1, high 4, original 8192) and checks the recomputed factors against the tensor, failing if they differ.
- `general.file_type` is not trusted: the spec quant is chosen from the tensor types in the blocks.

## Quant paths

| File | Spec quant | Path |
| --- | --- | --- |
| F16, BF16, F32 | `bf16` | converted to bf16 at load |
| Q8_0 | `q8` | Q8_0 is int8 with one scale per 32, the same as the existing `q8` path, so it loads without loss (the f16 scale becomes f32) |
| any K-quant (Q4_K_M, Q5_K_M, Q6_K) | `gguf` | blocks are kept as they are on the GPU |

For `gguf`, every matrix carries a row table: each row has its own byte offset and type (Q4_K, Q5_K, Q6_K, Q8_0, BF16), so a fused matrix (qkv, gate and up interleaved) can mix types the way Q4_K_M does (`attn_v` and some `ffn_down` are Q6_K). Q6_K blocks are padded from 210 to 224 bytes so every row starts 16-byte aligned.

Every linear goes through one GEMM template (`gemm_tc`) and one GEMV template (`linear_w`, `linear_norm`, `gate_up_act_norm`) with a weight decoder as a template parameter. A decoder says how its weights reach shared memory and how they become bf16 MMA fragments (prefill), and how a row block becomes a dot product (decode); the multiply, pipeline and epilogue are shared. Decoders: `DecBf16`, `DecQ8`, `DecFp8`, `DecFp4` and `DecGgml`, chosen per tensor by `kernel_for`. The engine tab shows the decoder of each linear in brackets.

- Decode: `DecGgml::rows` reads Q4_K and Q6_K blocks with 16-byte streaming loads and dequantises in registers (8 lanes per 256-value superblock). When the rows a warp handles share a type it runs them together, so both rows' loads are in flight before any math: gate and up rows always pair, and the single-output GEMVs take two rows per warp. The grid is capped at 2 blocks per SM with a grid-stride loop, so each block normalises the input once and keeps its x in shared memory for many rows. Mixed-type row pairs fall back to the per-row dot. Results are bit identical to the one-row kernel.
- Prefill: `kernel_for` picks `GemmPath::Ggb` for more than 32 tokens. Each matrix is dequantised to a bf16 scratch (`gg_dequant_bf16`) and `gemm_tc<DecBf16>` runs on it. Short batches use `gemm_tc<DecGgml>` with an in-kernel dequant.
- The GGUF GEMM grid runs the token tiles fastest, so the CTAs that share a weight tile run together and read it from L2. With the derived `[runtime] big_tile = [256, 128, 4, 2]` this took Qwen2.5-7B Q4_K_M prefill from 7.1k to 9.5k tok/s.
- The spec derived from a K-quant file sets `[runtime] unsplit` for the four dense linears. The weights are already 4 to 6 bit, so the split-activation GEMM costs time without changing accuracy: unsplit prefill on Qwen2.5-7B Q4_K_M runs at 7.3k tok/s against 4.9k split.

Tried and dropped (#462), on Qwen2.5-7B Q4_K_M at 1,024 tokens:

| Change | prefill | why dropped |
| --- | ---: | --- |
| Q4_K tile decoder: raw blocks to shared memory by `cp.async` (48 bytes per row per 64 values), decoded per fragment | 6.1k | the decode runs once per warp per fragment, 4x redundant |
| the same, decoded once per stage into a bf16 tile, then `ldmatrix` | 9.7k against 10.1k scratch | bit identical, but the extra sync and decode cost more than the scratch pass (`gg_dequant_bf16` is about 4% of prefill) |
| bk 64, 3 or 4 stages, 128x128 or 128x256 tiles for the scratch GEMM | 9.0k to 9.9k | the 256x128 tile with bk 32 and 2 stages stays best |

The scratch GEMM itself is the limit: `gate_up` at 1,024 x 37,888 x 3,584 takes 1.48 ms (188 TFLOP/s) where torch's bf16 GEMM takes 1.03 ms and its int8 GEMM 0.59 ms on the same card. llama.cpp's MMQ multiplies Q4_K weights with Q8_1 activations on int8 tensor cores, which is how it reaches 11.9k.

The bf16 dequantisation on the GPU is bit exact against ggml's `to_float`, and the native kernels give the same logits as loading the same file dequantised to bf16 (KL 0 on Qwen3-0.6B Q4_K_M).

## Config

No environment variables. A spec derived from a GGUF takes the same `[runtime]` table as any other; the server config passes it through.

## Tests

| Test | What |
| --- | --- |
| `gguf.rs` unit tests | tiny generated GGUF files: every metadata type, arrays, alignment 32 and 64, bad magic, bad versions, truncation, unknown tensor type, unknown architecture, metadata to spec |
| `tests/gguf.rs` | Q8_0, Q4_K, Q5_K and Q6_K dequantised bit exact against ggml on `fixtures/ggml-golden.bin` (written by `scripts/gguf_golden.c` with libggml); the same through `DeviceRows` with mixed types; the same on the GPU; spec and tokenizer of the real Qwen3-0.6B and Llama-3.2-1B files against the HF specs and `llama-tokenize` |
| `tests/gguf_logits.rs` | logits and greedy generation against llama.cpp; native kernels against bf16; engine chat on a GGUF |
| `core::library` | specrt support for a dense qwen3 GGUF, and the reason for an unsupported type |
| `ui` | the detail view reads GGUF metadata |

The reference logits come from `scripts/gguf_reference.c` (links libllama): all prompt logits and 64 greedy tokens for the 18-token prompt in the test. `scripts/gguf_kl.py` compares two dumps.

### KL bounds against llama.cpp

Measured on Qwen3-0.6B, 18 prompt positions, KL(llama.cpp || specrt):

| File | mean KL | max KL | top-1 agree | greedy 64 |
| --- | --- | --- | --- | --- |
| BF16 (converted with `convert_hf_to_gguf.py`) | 4.8e-5 | 1.1e-4 | 18/18 | 64/64 |
| Q8_0 | 2.2e-2 | 0.38 | 16/18 | 64/64 |
| Q4_K_M | 1.2e-2 | 6.9e-2 | 14/18 | diverges |

On BF16 the two engines agree to 5e-5, the same as llama.cpp against transformers on that file. The larger gap on the quantised files comes from llama.cpp, not from reading the weights: llama.cpp quantises activations to Q8_1 for its quantised matmuls, so its own CPU and GPU builds differ by up to KL 0.40 on Q4_K_M. Measured against the BF16 GGUF as ground truth, specrt on Q8_0 is closer than llama.cpp (mean KL 4.2e-3 against 2.6e-2) and on Q4_K_M about the same (0.48 against 0.51). The tests therefore use:

- BF16: mean KL at most 1e-3, max 5e-3, every top-1 equal.
- Q8_0 and Q4_K_M: mean KL at most 5e-2, max 0.5; top-1 at least 16/18 (Q8_0) and 14/18 (Q4_K_M).
- Q8_0 greedy: all 64 tokens equal to llama.cpp.
- Native K-quant kernels against the same file dequantised to bf16: mean KL at most 1e-4.

## Bench

RTX PRO 6000 Blackwell Max-Q, beside the resident vLLM (about 38 GB free), master after #462 (`48e2e54`). llama.cpp `7bd8282c3` (build 10267), CUDA, `-ngl 99 -fa 1`, 3 runs per point; llama.cpp decode uses `-d 1024` and `-d 4096` so both engines decode 128 tokens at the same depth. specrt `bench <file.gguf> 1024,4096 128`, median of 3. Both engines ran in the same window, alternating, each step started only below 10% GPU utilisation: `flock /tmp/gpu27.lock crates/specrt/scripts/vsllama.sh <file.gguf> <bench> 2`. The table is the mean of the two rounds.

| Model | Engine | prefill 1024 | prefill 4096 | decode at 1024 | decode at 4096 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B Q8_0 | llama.cpp | 55,630 | 51,410 | 782 | 640 |
| | specrt | 78,560 | 56,960 | 810 | 565 |
| Qwen3-0.6B Q4_K_M | llama.cpp | 53,110 | 50,140 | 806 | 658 |
| | specrt | 86,230 | 63,470 | 880 | 603 |
| Llama-3.2-1B-Instruct Q4_K_M | llama.cpp | 58,750 | 54,710 | 909 | 834 |
| | specrt | 62,600 | 52,600 | 875 | 674 |
| Qwen2.5-7B-Instruct Q4_K_M | llama.cpp | 12,130 | 11,240 | 241 | 225 |
| | specrt | 9,760 | 9,940 | 213 | 187 |

All in tokens per second. Before #462 the 7B ran at 7,092 prefill and 168 decode (llama.cpp then measured 11,024 and 222 in a busier window).

specrt is ahead on prefill for the small models and on short-context decode for Qwen3-0.6B. On the 7B it reaches 80% of llama.cpp's prefill and 88% of its decode at 1,024 context; the prefill gap is the bf16 GEMM against llama.cpp's int8 MMQ (#499). Decode at 4,096 context trails on every model; there the attention is the slower part.

The GEMV numbers in #462 were measured with paired runs of two builds (`crates/specrt/scripts/ab7b.sh`), checked bit identical with `crates/specrt/scripts/dumpall.sh` (logit hashes of 9 models across bf16, q8, fp8, nvfp4 and GGUF), and the 27B NVFP4 speed gate is `crates/specrt/scripts/gate27b.sh`.

## Decode profile (#502)

Qwen2.5-7B-Instruct Q4_K_M, one decode step, nsys with `--cuda-graph-trace=node`, median kernel time per call. The GPU was shared with other jobs, so llama.cpp's 4,096 rows are noisy.

| Kernel (per layer unless noted) | specrt 1024 | specrt 4096 | llama.cpp 1024 | llama.cpp 4096 |
| --- | ---: | ---: | ---: | ---: |
| gate and up (76 MB, floor 46 us) | 61.6 us | 63.7 us | | |
| down (38 MB, floor 23 us; llama.cpp `mul_mat_vec_q` K=18944) | 46.6 us | 47.3 us | 54.1 us | 54.7 us |
| qkv, o (llama.cpp K=3584 and N=512 calls) | 15.9 + 11.1 us | 16.3 + 11.9 us | 14.9 + 4.5 us | 19.1 + 4.6 us |
| decode attention (split plus reduce) | 11.3 + 5.2 us | 31.2 + 4.6 us | 7.5 + 8.5 us | 16.0 + 19.2 us |
| lm_head (Q6_K) | 353 us | 363 us | 2,829 us at 152064 rows in one launch, 326 us at 4,096 | |
| llama.cpp extra launches (`quantize_q8_1`, `rms_norm_f32`, rope, set_rows) | | | about 1.5 ms per step | about 2.5 ms per step |

Decode with the K-quant GEMV launch shape swept (block size 32 to 512, grid cap 1 to 8 blocks per SM, gate/up and down separately): no setting beats the shipped 256 threads and 2 blocks per SM by more than noise, and 64 threads on gate/up loses 12%. The shipped shape stays. A faster GEMV needs a different dot (int8 on Q8_1 activations), which changes the logits and is not done yet.

### Results of #502

Decode attention now runs one block per KV head (all GQA query heads share each K and V load, 4 keys in flight per warp). `dumpall.sh` shows prompt logits and greedy tokens unchanged on all 9 models; the decode-step logit hashes change because the softmax sums in a different order (greedy tokens still match). On the 7B the A/B at 4,096 context is 189 to 203 tok/s (`ab7b.sh`, 2 rounds).

Tried and dropped: an int8 `dp4a` K-quant dot on activations quantised to 8 bits per 32 values. It was slower, 201 against 217 tok/s at 1,024 and 189 against 203 at 4,096, and changes the logits, so it is not in the tree. The GEMV is limited by weight reads and load issue, not by the multiply.

Final alternating run (`vsllama.sh`, 2 rounds; llama.cpp round 2 at 1,024 was disturbed by another job, 154 tok/s):

| Context | llama.cpp | specrt |
| --- | ---: | ---: |
| decode at 1,024 | 240 | 216 and 213 |
| decode at 4,096 | 224.5 and 222.2 | 202.3 and 199.6 |

specrt is at 90% of llama.cpp on the 7B at both depths. 27B NVFP4 (`gate27b.sh`, 5 runs): decode 56.9 and 55.4 tok/s at 1,024, 54.6 and 54.0 at 4,096, against 56.3 and 55.2, 53.2 and 52.8 before; logits and greedy tokens identical.

Next lever: the K-quant GEMV reads gate/up at about 75% and down at about 50% of the wall. llama.cpp's Q4_K `mul_mat_vec_q` takes 54 us for the down matrix against 47 us here, so the remaining gap is spread over the smaller launches (qkv, o) and the attention reduce, not one kernel. Fusing the per-layer launches (attention reduce into the o projection, qkv into the norm) is the next step.

Fusing the attention reduce into `attention_split` (last block per KV head reduces its 4 query heads, atomic ticket) was slower, 195 against 216 tok/s at 1,024 and 187 against 202 at 4,096, because one block then serialises four reductions after the split; it is not in the tree. The prompt logits and greedy tokens were unchanged. The qkv GEMV already fuses the RMSNorm (`qkv_proj_norm`), so nothing is left to fuse there.

Remaining per-step gap on the 7B (llama.cpp 239 and 222 tok/s, specrt 216 and 202): about 0.45 ms per step at 1,024 and 0.5 ms at 4,096. The gate/up GEMV at 61.6 us per layer against a 46 us read floor accounts for 0.4 ms of it.

### Tried and dropped (#502)

| Attempt | Result |
| --- | --- |
| Gate/up GEMV with 4 rows per warp (two gate/up pairs, 8 16-byte loads in flight), bit identical | 210.6 against 210.5 tok/s at 1,024 and 197.2 against 197.4 at 4,096, 3 alternating pairs: no change, the loads were already in flight |
| Attention reduce fused into `attention_split` | 195 against 216 tok/s at 1,024 |
| int8 `dp4a` K-quant dot on quantised activations | 201 against 217 tok/s at 1,024 |
| GEMV block size 32 to 512, grid cap 1 to 8 blocks per SM | within noise of the shipped 256 threads and 2 blocks per SM |
