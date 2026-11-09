# GGUF in specrt

specrt loads a llama.cpp GGUF file directly. `model` is the path of the `.gguf` file; the spec, tokenizer, chat template and stop tokens all come from the file, so there is no `config.json` or `tokenizer.json` to find.

## Supported

| Item | Supported |
| --- | --- |
| Architectures | `llama`, `qwen2`, `qwen3` (dense). Others fail with "GGUF architecture X is not supported". |
| Tensor types | F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K. Other types fail with the tensor name and type. |
| Tokenizer | `tokenizer.ggml.model = gpt2` with `pre` of `qwen2` or `llama-bpe` |
| Files | one file. Split GGUF (`-0000N-of-0000M`) is not read yet. |

## Layout

- `crates/specrt/src/gguf.rs`: parser (header, metadata, tensor infos, alignment, mmap), block dequantisation ported from `ggml-quants.c`, `Spec` from metadata, tokenizer and chat template from metadata.
- `weights.rs`: `Checkpoint` opens a `.gguf` and answers the same HF tensor names (`model.layers.N.self_attn.q_proj.weight`) by mapping them to `blk.N.attn_q.weight`.
- GGUF stores shapes as `[cols, rows]`; the reader reverses them.
- `llama` GGUF files store Q and K rows permuted for interleaved rope. specrt uses the half-split (neox) rope, so the reader undoes the permutation by reordering rows (whole blocks, so it works on quantised data).
- Llama 3 rope scaling is stored as a `rope_freqs.weight` tensor. The reader recovers `factor` from its maximum, assumes the converter defaults (low 1, high 4, original 8192) and checks the recomputed factors against the tensor.

## Quant handling

| Tensor type | Path |
| --- | --- |
| F32, F16, BF16 | converted to bf16 at load |
| Q8_0 | the existing `q8` path: Q8_0 blocks are int8 values with an f16 scale per 32, the same as `Matrix::Q8`, so it is lossless |
| Q4_K, Q5_K, Q6_K | see the kernels section |

## Config

Nothing is set by environment variable. The spec derived from the file takes the same `[runtime]` table as any other spec; the server config passes it through.

## Tests

- `gguf.rs` unit tests build tiny GGUF files (all metadata types, arrays, alignment, bad magic, bad version, truncation).
- `tests/fixtures/gguf-dequant.gguf` is written by `scripts/gguf_fixture.py` with gguf-py; each quantised tensor is stored next to its f32 expansion and the Rust dequantiser must match bit for bit.
