# GGUF in specrt

specrt loads a llama.cpp GGUF file directly. `model` is the path of the `.gguf` file, in the server config or the Models page. The spec, tokenizer, chat template and stop tokens all come from the file, so no `config.json` or `tokenizer.json` is needed.

## Supported

| Item | Supported |
| --- | --- |
| Architectures | the `gguf = [...]` list of every family TOML: `llama`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35` (Qwen3.5/3.6/3.8 delta-net hybrids, with their MTP block) and `qwen35moe`. Others fail with "GGUF architecture X is not supported (list, each once)". |
| Tensor types | F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K and the IQ types (CUDA only, see i-quants). Any other type fails at load and in the library with the tensor name and type. |
| Tokenizer | `tokenizer.ggml.model = gpt2` with a `pre` named in the family TOML `[gguf.pretokenizers]` (`qwen2`, `qwen35`, `llama-bpe`, ...); `tokenizer.ggml.normalizer.nfc` adds an NFC normaliser |
| Files | one file. Split GGUF (`-0000N-of-0000M`) is listed but marked unsupported for specrt. |

## Layout

- `crates/specrt/src/gguf.rs`: parser (header, metadata, tensor infos, alignment, mmap), block dequantisation ported from `ggml-quants.c`, `Spec` from metadata, tokenizer and chat template from metadata, and `DeviceRows`, the device layout for K-quants.
- `weights.rs`: `Checkpoint` opens a `.gguf` and answers the HF tensor names (`model.layers.N.self_attn.q_proj.weight`) by mapping them to `blk.N.attn_q.weight`.
- GGUF stores shapes as `[cols, rows]`; the reader reverses them.
- `llama` GGUF files store Q and K rows permuted for interleaved rope. specrt uses the half-split (neox) rope, so the reader undoes the permutation by reordering whole rows, which works on quantised rows too.
- Llama 3 rope scaling is stored as a `rope_freqs.weight` tensor. The reader takes `factor` from its maximum, uses the converter defaults (low 1, high 4, original 8192) and checks the recomputed factors against the tensor, failing if they differ.
- `general.file_type` is not trusted: the spec quant is chosen from the tensor types in the blocks.

## qwen35 (delta-net hybrids)

`qwen35` and `qwen35moe` are mapped by `family-qwen3_5.toml`; no Rust names the architecture.

- Metadata: `[gguf.keys]` in `family-base.toml` names the `{arch}.ssm.*`, `full_attention_interval`, `attention.recurrent_layers`, `rope.dimension_sections` and `nextn_predict_layers` keys. The layer pattern comes from which blocks have the family's `[gguf.tensors] linear_layer` tensor (`ssm_a`), the MTP block is the layers past `block_count - nextn_predict_layers`.
- Tensor names: `[gguf.layer]` maps each HF tail (`linear_attn.in_proj_qkv.weight`) to its GGUF tail (`attn_qkv.weight`), `[gguf.mtp]` the `mtp.*` tensors to the `nextn.*` ones on the layer after the main stack.
- Converter changes that the family states: `a_stored_negated` (`ssm_a` is `-exp(A_log)` already), RMSNorm weights already carry the `+1`, `output_gate`, interleaved mrope.
- Value heads: llama.cpp reorders the value heads of every linear-attention tensor from grouped (HF) to tiled so ggml can broadcast. specrt does not permute the tensors, because that would break the K-quant block layout of the out projection (its columns are the permuted axis). `[linear_attention] v_heads = "tiled"` (from the family `[gguf] v_heads`) makes the kernels read the key head of value head `h` as `h % key_heads` instead of `h / ratio`. Tests: `gguf_qwen35.rs`.
- The hybrid linears `lin_in` and `lin_out` run the `DecGgml` decoder; the delta-net kernels are unchanged.
- The vocabulary is not trimmed: `token_embd` has the full vocab rows, so `dims.vocab` comes from it.

## Quant paths

| File | Spec quant | Path |
| --- | --- | --- |
| F16, BF16, F32 | `bf16` | converted to bf16 at load |
| Q8_0 | `q8` | Q8_0 is int8 with one scale per 32, the same as the existing `q8` path, so it loads without loss (the f16 scale becomes f32) |
| Q4_0, Q5_0 | `gguf` | repacked to Q8_0 blocks at load (the int values and scale are exact) |
| any K-quant (Q4_K_M, Q5_K_M, Q6_K) | `gguf` | blocks are kept as they are on the GPU |
| Q8_0 mixed with F16 or BF16 tensors (for example a Q8_K_P recipe), or any hybrid qwen35 file | `gguf` | row table per matrix, so each row keeps its own type |
| MoE experts (`ffn_*_exps`, `ffn_*_shexp`) | `gguf` or `q8` | expanded to bf16 at load by default. With `[moe] quantized_experts = true` they stay ggml blocks: one fixed size slot per expert (its `DeviceRows` padded to the largest expert) and a row table of `rows + 1` words per expert whose last word is the slot size in 16 byte units. `moe_up_gg`, `moe_down_gg` and the grouped pair (`kir::moe::Family::ExpertsGgml`) dot those rows with `DecGgml::dot`, the decoder the dense GEMV uses, against f32 activations. The staged expert cache passes slot ids and the per slot bias of #1173, which these kernels read by slot like the other formats. With `runtime.offload.pointer_table` on, the same family adds `moe_up_t_gg` and `moe_down_t_gg`, built by the shared table builder: the table holds slot plus one, the kernel resolves the slot and reads the bias by expert id. With offload on, every layer's experts get one slot size (the largest packed expert of that matrix in the file), so a Q4_K_M file whose `ffn_down` is Q6_K in some layers still shares one cache; a layer whose slots differ is refused at load. The expert cache copies a slot and its table as two parts, so host and NVMe pages shrink with the blocks |

Keeping experts quantised on the tiny qwen3moe fixture (`scripts/tiny_qwen3moe_gguf.py`, 4 experts, hidden 256): Q4_K_M experts take 0.99 MB against 3.15 MB expanded (3.2x, one `ffn_down` is Q6_K), Q8_0 1.70 MB against 3.15 MB (1.85x); total device weights 2.00 MB against 4.15 MB and 2.97 MB against 4.42 MB (GPU1). Every kept row decodes to the same bf16 as the expanded path. Logits against the expanded path: max KL 7.3e-6 (Q4_K_M) and 8.8e-6 (Q8_0), so the bound is 2e-5; the remaining difference is the bf16 rounding the expanded path adds. Swapping two block rows in one expert's table gives max KL 4.3e-2. Through the expert cache (2 of 4 slots), the pointer table with every expert resident, and the table falling back to staging, decode logits and greedy tokens equal the resident run bit for bit. Speed on the tiny fixture means nothing; measure it on a real MoE GGUF:

```
bench <file.gguf> --ab --b-set spec.moe.quantized_experts=true 128 256
```

`--a-set` and `--b-set` take `spec.<path>=<value>` for any spec key besides the `runtime.*` and `speculative.*` options. The default stays off until that run shows decode within 5 percent.

For `gguf`, every matrix carries a row table: each row has its own byte offset and type (Q4_K, Q5_K, Q6_K, Q8_0, BF16), so a fused matrix (qkv, gate and up interleaved) can mix types the way Q4_K_M does (`attn_v` and some `ffn_down` are Q6_K). Q6_K blocks are padded from 210 to 224 bytes so every row starts 16-byte aligned.

Every linear goes through one GEMM template (`gemm_tc`) and one GEMV template (`linear_w`, `linear_norm`, `gate_up_act_norm`) with a weight decoder as a template parameter. A decoder says how its weights reach shared memory and how they become bf16 MMA fragments (prefill), and how a row block becomes a dot product (decode); the multiply, pipeline and epilogue are shared. Decoders: `DecBf16`, `DecQ8`, `DecFp8`, `DecFp4` and `DecGgml`, chosen per tensor by `select`. The engine tab shows the decoder of each linear in brackets.

- Decode: `DecGgml::rows` reads Q4_K and Q6_K blocks with 16-byte streaming loads and dequantises in registers (8 lanes per 256-value superblock). When the rows a warp handles share a type it runs them together, so both rows' loads are in flight before any math: gate and up rows always pair, and the single-output GEMVs take two rows per warp. The grid is capped at 2 blocks per SM with a grid-stride loop, so each block normalises the input once and keeps its x in shared memory for many rows. Mixed-type row pairs fall back to the per-row dot. Results are bit identical to the one-row kernel.
- Prefill: `select` picks `GemmPath::Ggb` for more than 32 tokens. Each matrix is dequantised to a bf16 scratch (`gg_dequant_bf16`) and `gemm_tc<DecBf16>` runs on it. Short batches use `gemm_tc<DecGgml>` with an in-kernel dequant.
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
| `tests/gguf.rs` | Q8_0, Q4_K, Q5_K, Q6_K, Q4_0 and Q5_0 dequantised bit exact against ggml on `fixtures/ggml-golden.bin` (written by `scripts/gguf_golden.c` with libggml); the same through `DeviceRows` with mixed types; the same on the GPU; spec and tokenizer of the real Qwen3-0.6B and Llama-3.2-1B files against the HF specs and `llama-tokenize` |
| `tests/gguf_logits.rs` | logits and greedy generation against llama.cpp; native kernels against bf16; engine chat on a GGUF |
| `tests/gguf_qwen35.rs` | tiny qwen35 GGUFs made by llama.cpp `convert_hf_to_gguf.py` and `llama-quantize` from `tiny-qwen3.5-hd256` (`scripts/qwen35_gguf_fixture.py`, `qwen35_mtp_fixture.py`): the spec equals the safetensors spec, every value-head tensor lands in the same tiled order, the MTP block equals the safetensors MTP, Q8_0, Q5_K_M, Q5_0, Q4_K_M and Q4_0 load inside their error bound, the module compiles with NVRTC, the tokenizer uses the family pre-tokenizer |
| `tests/gguf_qwen35_gpu.rs` | ignored GPU tests: tiny Q8_0 against the safetensors logits, a real file against llama.cpp logits, KL and greedy, and native kernels against bf16. `scripts/gate676.sh <file.gguf>` builds the llama.cpp reference and runs them |
| `tests/gguf_moe.rs` | tiny qwen3moe Q4_K_M and Q8_0: kept expert rows decode to the expanded bf16, the size ratio, fixed expert slots, the anatomy bytes, the option only appends the ggml expert family to the module, NVRTC compile. Ignored GPU test: KL, greedy prefix, weight bytes and prefill and decode tok/s, kept against expanded |
| `core::library` | specrt support for a dense qwen3 GGUF, a qwen35 GGUF, and the reason for an unsupported type |
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

### i-quants

F32, F16, BF16 and the nine IQ types (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS) load on the CUDA engine. The wgpu engine refuses them ("IQ quant types need the CUDA engine"), and Discover marks IQ quants unsupported when every GPU is a wgpu one. Dequant is bit-exact against libggml (`tests/gguf_iq.rs`, blocks from `scripts/gen_iq_reference.c`); tables come from `crates/kir/scripts/gen_iq_tables.py`.

Qwen3-0.6B, 18 prompt positions, KL(llama.cpp || specrt). Each file is `llama-quantize --pure --imatrix <imatrix of ~300 KB of text> --token-embedding-type q8_0 --output-tensor-type q8_0` from the BF16 GGUF; IQ2_S needs `--tensor-type 'blk\..*\.weight=iq2_s'` because `--pure` maps it to IQ2_XS tensors. Every layer weight is the named type; norms stay F32.

| File | mean KL | max KL | top-1 agree | greedy 64 |
| --- | --- | --- | --- | --- |
| IQ4_NL | 9.9e-3 | 0.14 | 17/18 | 64/64 |
| IQ4_XS | 2.1e-2 | 0.21 | 16/18 | 19/64 |
| IQ3_S | 6.3e-3 | 1.8e-2 | 18/18 | 4/64 |
| IQ3_XXS | 1.2e-2 | 6.9e-2 | 14/18 | 64/64 |
| IQ2_S | 1.7e-2 | 4.6e-2 | 15/18 | 3/64 |
| IQ2_XS | 4.6e-2 | 0.18 | 17/18 | 1/64 |
| IQ2_XXS | 0.40 | 2.4 | 14/18 | 2/64 |
| IQ1_S | 2.4 | 6.2 | 3/18 | 1/64 |
| IQ1_M | 2.1 | 7.0 | 11/18 | 1/64 |

IQ1 and IQ2_XXS on a 0.6B model are close to unusable in llama.cpp too, so large KL there is the quantisation, not the reader. `qwen3_iq_gguf_logits_stay_within_the_kl_bound_of_llama_cpp` bounds mean and max KL at 1.5 times these numbers, and `qwen3_iq_native_kernels_match_the_same_weights_dequantised_to_bf16` checks the native kernels against the same file dequantised to bf16. The GPU-side checks need `~/.cache/specrt-gguf/Qwen3-0.6B-IQ*.gguf` and `q3-IQ*.logits.bin` (`scripts/gguf_reference.c`).

#### Real IQ files, 20 prompts

Files from `unsloth/Qwen3-0.6B-GGUF` (saved as `real-<name>` in `~/.cache/specrt-gguf`), llama.cpp `cb26014d9` CUDA. `scripts/iq_real_reference.sh <file> <dir>` tokenizes the 20 prompts in `tests/fixtures/iq_prompts.txt` and dumps llama.cpp logits and 32 greedy tokens each; test `real_qwen3_iq_files_agree_with_llama_cpp_on_20_prompts` (ignored, skips with a message when files are missing). 169 prompt positions, greedy prefix is the sum over prompts of tokens equal before the first difference, out of 640.

| File | sha256 | mean KL | max KL | top-1 | greedy prefix | bound (mean, max, top-1, prefix) |
| --- | --- | --- | --- | --- | --- | --- |
| IQ4_XS | dacc1fe1dc6f4799f366ac3385299f71c7e02c58b8702448a174f14d91b1c64d | 9.3e-3 | 7.1e-2 | 149/169 | 494/640 | 1.4e-2, 0.11, 135, 440 |
| IQ4_NL | f1b14e28a6de64f21672ddf2c8f24736c389da4c64ebf024b1204f4f43a4fd71 | 9.4e-3 | 0.13 | 151/169 | 439/640 | 1.4e-2, 0.2, 135, 395 |
| UD-IQ3_XXS | 0ed6dab036349ba8198324c70002a15397e8bdeb06748eee5c631f872e1eaf2f | 7.0e-3 | 0.11 | 160/169 | 508/640 | 1.1e-2, 0.17, 144, 455 |
| UD-IQ2_XXS | 55e0d0b88ae021edf15d2f0cec5ab2c47552b6a5f388dfab869f1e67240263c3 | 9.1e-2 | 1.25 | 147/169 | 348/640 | 0.14, 1.9, 132, 310 |

The numbers match the self-quantised files above, and the reader shows no format-specific error. Bounds are 1.5x the KL and 0.9x the agreement counts.

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

### Tried and dropped: MMQ int8 on Q8_1 activations (#499)

Built as a second multiply kind for GGUF prefill: activations quantised per 32 values to int8 with an f32 scale and sum (`quant_rows_q81`), Q4_K and Q6_K blocks copied raw by `cp.async`, `mma.sync m16n8k32 s8` per 32-wide sub-block with the scale and min applied in fp32, behind `act_quant = "q8_1"` and `[mlp] act_quant = "q8_1"`. The code lives on branch `archive/499-mmq`, commit `d3bef6b` (kernels in `crates/kir/src/cuda/mmq.cu` and `quant_q81.cu`, selection in `kir/src/gemm.rs`, `GemmPath::Mmq`).

Qwen2.5-7B-Instruct Q4_K_M, 1,024 token prefill, GB10 (Spark), alternating runs, tok/s:

| Path | prefill |
| --- | ---: |
| bf16 scratch GEMM | 2,666 and 2,676 |
| q8_1 MMQ | 2,575 and 2,422 |
| llama.cpp (`llama-bench` pp1024) | 3,015 |

Numerics: Qwen3-0.6B Q4_K_M, 160 token prompt, q8_1 against bf16 activations: mean KL 1.3e-3, max KL 2.6e-2, top-1 148/160. Not compared against llama.cpp logits.

Why it is slower (nsys of one 1,024 token prefill on the Spark, 56 layer calls each): `gate_up_mmq_big` 7.5 ms, `down_mmq_big` 3.8 ms, `qkv` 0.93 ms, `o` 0.84 ms, and the quantise pass 0.18 ms per call (4.9% of the GPU time). The two big GEMMs are 76% of the time, so the multiply loop is the cost, not the quantise pass. Likely causes, not isolated: the sub-block scale and min are applied per mma in fp32 with scalar shared-memory loads of the packed Q4_K header and scales, and a 2-stage pipeline at bk 128 with a 128x128 tile leaves little to hide the raw-block decode. Occupancy on sm_121 and the tile choice were not swept.
