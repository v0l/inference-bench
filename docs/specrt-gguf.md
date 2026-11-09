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

- Decode: `linear_ggml` and `row_dot<WF_GGML>` read the blocks and dequantise in registers. Q4_K and Q6_K have their own dot products (8 lanes per 256-value superblock); the other types use the shared 8-value dequantiser.
- Prefill: `kernel_for` picks the `GemmPath::Ggb` path for more than 32 tokens. Each matrix is dequantised to a bf16 scratch (`gg_dequant_bf16`) and the existing bf16 tensor-core GEMM runs on it. Short batches use the MMA GEMM with an in-kernel dequant.
- The spec derived from a K-quant file sets `[runtime] unsplit` for the four dense linears. The weights are already 4 to 6 bit, so the split-activation GEMM costs time without changing accuracy: unsplit prefill on Qwen2.5-7B Q4_K_M runs at 7.3k tok/s against 4.9k split.

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

RTX PRO 6000 Blackwell Max-Q, beside the resident vLLM (about 38 GB free). llama.cpp `7bd8282c3` (build 10267), CUDA, `-ngl 99 -fa 1`, 5 runs. specrt `bench <file.gguf> 1024,4096 128` (median of 5; `SPECRT_BENCH_RUNS=5`). Decode is batch 1, 128 tokens after the prompt; llama.cpp decode uses `-d 1024` and `-d 4096` so both engines decode at the same depth.

| Model | Engine | prefill 1024 | prefill 4096 | decode at 1024 | decode at 4096 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B Q8_0 | llama.cpp | 50,680 | 46,914 | 739 | 612 |
| | specrt | 74,202 | 54,373 | 821 | 557 |
| Qwen3-0.6B Q4_K_M | llama.cpp | 48,816 | 46,379 | 779 | 636 |
| | specrt | 85,818 | 62,311 | 864 | 571 |
| Llama-3.2-1B-Instruct Q4_K_M | llama.cpp | 54,336 | 49,841 | 865 | 793 |
| | specrt | 59,098 | 54,389 | 729 | 572 |
| Qwen2.5-7B-Instruct Q4_K_M | llama.cpp | 11,024 | 10,341 | 222 | 209 |
| | specrt | 7,092 | 7,071 | 168 | 150 |

All in tokens per second. specrt is ahead on prefill for the small models and on short-context decode for Qwen3-0.6B. It is behind on the 7B (prefill pays for dequantising each weight to bf16 per chunk, and decode reads the blocks at about 73% of llama.cpp's rate) and on decode at 4096, where its attention is the slower part.
