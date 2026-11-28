# GGUF

specrt loads a llama.cpp GGUF file directly: `model` is the `.gguf` path in the server config or the Models page. Spec, tokenizer, chat template and stop tokens come from the file.

## Supported

| Item | Supported |
|---|---|
| Architectures | the `gguf = [...]` list of every family TOML: `llama`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35` (delta-net hybrids with their MTP block), `qwen35moe`. Others fail naming the architecture |
| Tensor types | F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, and on CUDA the IQ types (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS). Anything else fails with the tensor name and type |
| Tokenizer | `tokenizer.ggml.model = gpt2` with a `pre` named in the family `[gguf.pretokenizers]`; `tokenizer.ggml.normalizer.nfc` adds NFC |
| Files | one file; split GGUF is listed as unsupported |

## Layout

- `crates/specrt/src/gguf.rs`: parser, block dequantisation ported from `ggml-quants.c`, `Spec` from metadata, tokenizer and template, `DeviceRows`. `crates/specrt/src/gguf_iq.rs` and `crates/kir/src/iq_tables.rs` for the IQ types.
- `crates/specrt/src/weights.rs`: `Checkpoint` answers HF tensor names by mapping them to `blk.N.*`. Shapes are stored `[cols, rows]` and reversed.
- `llama` files store Q and K rows permuted for interleaved rope; the reader undoes it by reordering whole rows (works on quantised rows). Llama 3 rope scaling comes from `rope_freqs.weight`, checked against the recomputed factors.
- `general.file_type` is not trusted: the spec quant comes from the tensor types.
- qwen35: `family-qwen3_5.toml` maps metadata keys (`[gguf.keys]` in `family-base.toml`), tensor names (`[gguf.layer]`, `[gguf.mtp]`) and converter changes (`a_stored_negated`, norms already carrying `+1`, output gate, interleaved mrope). llama.cpp tiles the value heads; specrt keeps the tensors as stored and sets `[linear_attention] v_heads = "tiled"`, so the kernels read the key head of value head `h` as `h % key_heads`.

## Quant paths

| File | Spec quant | Path |
|---|---|---|
| F16, BF16, F32 | `bf16` | converted at load |
| Q8_0 | `q8` | loads without loss into the q8 path |
| Q4_0, Q5_0 | `gguf` | repacked to Q8_0 blocks at load (exact) |
| K-quants, mixed files, any qwen35 file | `gguf` | blocks kept on the GPU with a row table per matrix: each row has its own offset and type, so fused matrices can mix Q4_K and Q6_K; Q6_K blocks padded to 224 bytes for 16 byte alignment |
| MoE experts | `gguf` or `q8` | expanded to bf16 at load; with `[moe] quantized_experts = true` kept as ggml blocks in fixed size slots per expert, read by `moe_up_gg`, `moe_down_gg` and the grouped pair (`kir::moe::Family::ExpertsGgml`), also through the expert cache and pointer table |

Every linear uses `DecGgml` in the shared GEMM and GEMV templates ([gemm.md](gemm.md)). Decode: `DecGgml::rows` streams 16 byte loads and dequantises in registers, pairing rows of the same type; the grid is capped at 2 blocks per SM with a grid stride (a sweep of 32 to 512 threads and 1 to 8 blocks per SM found nothing better). Prefill above 32 tokens dequantises each matrix to a bf16 scratch (`gg_dequant_bf16`, bit exact against ggml `to_float`) and runs the bf16 tile GEMM with token tiles fastest; shorter batches dequantise in the kernel. K-quant specs set `[runtime] unsplit` for the dense linears and `big_tile = [256, 128, 4, 2]`.

`quantized_experts` stays off: on Qwen3.5-122B Q4_K_M cut to 12 layers it decodes 5 percent faster at batch 1 with 22.8 against 63.3 GB, but prefill is 25 to 29 percent slower and max KL is 4.9e-3 (#1184). An int8 MMQ prefill on Q8_1 activations was slower than the bf16 scratch GEMM (#499, branch `archive/499-mmq`); a `dp4a` decode dot and fusing the attention reduce into the split were slower too (#502).

## Correctness bounds

Qwen3-0.6B, 18 prompt positions, KL(llama.cpp || specrt), asserted in `crates/specrt/tests/gguf_logits.rs`:

| File | Bound | Measured |
|---|---|---|
| BF16 | mean 1e-3, max 5e-3, every top-1 equal | 4.8e-5, 1.1e-4, 18 of 18, greedy 64 of 64 |
| Q8_0 | mean 5e-2, max 0.5, top-1 at least 16 of 18, greedy 64 equal | 2.2e-2, 0.38, 16 of 18 |
| Q4_K_M | mean 5e-2, max 0.5, top-1 at least 14 of 18 | 1.2e-2, 6.9e-2, 14 of 18 |
| native K-quant kernels against the same file dequantised to bf16 | mean 1e-4 | 0 |

The quantised gap is llama.cpp's own Q8_1 activation rounding (its CPU and GPU builds differ by up to KL 0.40 on Q4_K_M).

IQ files, Qwen3-0.6B, each `llama-quantize --pure --imatrix <imatrix> --token-embedding-type q8_0 --output-tensor-type q8_0` from the BF16 GGUF (IQ2_S needs `--tensor-type 'blk\..*\.weight=iq2_s'`). `qwen3_iq_gguf_logits_stay_within_the_kl_bound_of_llama_cpp` bounds mean and max KL at 1.5x these:

| File | mean KL | max KL | top-1 |
|---|---|---|---|
| IQ4_NL | 9.9e-3 | 0.14 | 17/18 |
| IQ4_XS | 2.1e-2 | 0.21 | 16/18 |
| IQ3_S | 6.3e-3 | 1.8e-2 | 18/18 |
| IQ3_XXS | 1.2e-2 | 6.9e-2 | 14/18 |
| IQ2_S | 1.7e-2 | 4.6e-2 | 15/18 |
| IQ2_XS | 4.6e-2 | 0.18 | 17/18 |
| IQ2_XXS | 0.40 | 2.4 | 14/18 |
| IQ1_S | 2.4 | 6.2 | 3/18 |
| IQ1_M | 2.1 | 7.0 | 11/18 |

Four real `unsloth/Qwen3-0.6B-GGUF` files (IQ4_XS, IQ4_NL, UD-IQ3_XXS, UD-IQ2_XXS) over the 20 prompts of `tests/fixtures/iq_prompts.txt` are bounded at 1.5x KL and 0.9x agreement (`real_qwen3_iq_files_agree_with_llama_cpp_on_20_prompts`, references from `scripts/iq_real_reference.sh`).

## Measured state

RTX PRO 6000, llama.cpp `7bd8282c3` CUDA, `crates/specrt/scripts/vsllama.sh <file.gguf> <bench> 2`, tok/s (#462, #502):

| Model | Engine | prefill 1024 | decode at 1024 | decode at 4096 |
|---|---|---|---|---|
| Qwen3-0.6B Q4_K_M | llama.cpp / specrt | 53,110 / 86,230 | 806 / 880 | 658 / 603 |
| Llama-3.2-1B Q4_K_M | llama.cpp / specrt | 58,750 / 62,600 | 909 / 875 | 834 / 674 |
| Qwen2.5-7B Q4_K_M | llama.cpp / specrt | 12,130 / 9,760 | 240 / 216 | 224 / 202 |

The 7B prefill gap is the bf16 scratch GEMM against llama.cpp's int8 MMQ.

## Tests

`gguf.rs` unit tests (metadata types, alignment, truncation), `crates/specrt/tests/gguf.rs` (dequant bit exact against libggml on `fixtures/ggml-golden.bin` from `scripts/gguf_golden.c`; specs and tokenizers of real files), `gguf_iq.rs` (from `scripts/gen_iq_reference.c`), `gguf_logits.rs` (against llama.cpp logits from `scripts/gguf_reference.c`, `SPECRT_GGUF_DIR`), `gguf_qwen35.rs` and `gguf_qwen35_gpu.rs` (llama.cpp converted tiny fixtures, `scripts/gate676.sh`), `gguf_moe.rs`, `gguf_inventory.rs`. Bit identity across builds: `crates/specrt/scripts/dumpall.sh`; paired speed: `ab7b.sh`.
