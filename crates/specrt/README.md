# specrt

A spike for issue #235: a CUDA runtime for dense decoder models where a new model is a TOML file,
not a new binary.

It is a standalone crate with its own `[workspace]`, not a member of the bench-suite app. The app
is an egui launcher built for macOS, Windows and Linux in CI. specrt needs CUDA at run time and only
targets Linux, so keeping it separate leaves the app's lockfile, CI matrix and release builds
untouched. cudarc uses `dynamic-loading`, so building needs no CUDA toolkit. Running needs
`libcuda` and `libnvrtc`.

## How it fits together

- `specs/*.toml` describe a model: dims, layer pattern, norm, attention variant (GQA, head dim,
  QKV bias, QK norm), RoPE, activation and quant. `Spec::from_hf_config` derives one from an HF
  `config.json`, and `tests/spec.rs` checks the checked-in specs against the real configs.
- `templates/ops.cu` holds the op templates (embedding, RMSNorm, fused QK-norm and RoPE with the
  paged KV write, paged GQA attention, SwiGLU, bf16 and Q8 GEMV, argmax). `templates/model.cu`
  instantiates the five linears for the spec. `codegen::Module::generate` fills the `{{SLOTS}}`
  from the spec, so every shape is a compile-time constant.
- `jit::Jit` compiles the source with NVRTC for the device's `compute_XY` and caches the PTX in
  `~/.cache/specrt/<sha256(source, arch, options)>.ptx` (`SPECRT_CACHE` to move it).
- `runtime::Runtime` loads `model.safetensors` from the HF hub cache, fuses q/k/v and gate/up,
  quantises to Q8 (per 32-weight group, f32 scale) when the spec asks, and runs single-stream
  decode. The KV cache is paged (16-token pages through a block table). Each decode step, argmax
  included, is captured once as a CUDA graph and replayed.

Qwen2.5-0.5B was added after Qwen3-0.6B with only `specs/qwen2.5-0.5b.toml`. It differs in head
dim (64 vs 128), GQA group (7 vs 2), QKV bias and no QK norm. No code or template changed.

## Tests

```
cargo test
```

runs the CPU tests: spec parsing and derivation, template rendering, cache keys. The GPU tests are
`#[ignore]` and need the weights in the hub cache plus reference logits:

```
python3 scripts/reference.py specs/qwen3-0.6b.toml out
python3 scripts/reference.py specs/qwen2.5-0.5b.toml out
cargo test -- --include-ignored --test-threads 1
```

`scripts/reference.py` runs `transformers` in fp32 on the CPU over a fixed 43-token prompt and
saves the logits for every position, plus 32 greedy tokens. `tests/logits.rs` feeds the same tokens
through specrt and compares all 43 x 151936 logits.

Measured on spark (GB10, torch 2.9.1, transformers 4.57.3):

| spec | max abs diff | mean abs diff | top-1 agreement | 32 greedy tokens |
|---|---|---|---|---|
| qwen3-0.6b (bf16) | 1.0e-4 | 6e-6 | 43/43 | identical |
| qwen2.5-0.5b (bf16) | 1.6e-4 | 1.1e-5 | 43/43 | identical |
| qwen3-0.6b-q8 | 2.27 | 0.057 | 43/43 | not checked |

The bf16 tests assert max abs diff at most 1e-3 plus identical greedy output. The Q8 test asserts
mean abs diff at most 0.1 and top-1 agreement of at least 95%.

## Speed

`cargo run --release --bin bench -- specs/<spec>.toml 128 128` reports the median of 5 runs.
Single stream, batch 1, 128-token prompt, 128 new tokens, decode tokens/s. Spark, GPU clock capped
at 2200 MHz, the production vLLM server resident but idle. All three engines were run back to back:

| model | specrt | llama.cpp `7bd8282c3` | vLLM 0.32.0 |
|---|---|---|---|
| Qwen3-0.6B bf16 | 117 | 164 | 117 |
| Qwen3-0.6B Q8 (llama.cpp q8_0) | 160 | 237 | not run |
| Qwen2.5-0.5B bf16 | 160 | 190 | 55 |

llama.cpp numbers are from `llama-bench -p 128 -n 128 -ngl 99 -r 5` on GGUFs converted from the
same safetensors. vLLM numbers are from `vllm bench latency --batch-size 1 --input-len 128`, with
decode time taken as the output-len 128 latency minus the output-len 1 latency. While vLLM was
serving requests (GPU 96% busy), every engine dropped to about half these numbers.

specrt prefill runs one token at a time through the decode path (about 130-210 tokens/s). llama.cpp
prefills at 9-12k tokens/s, so prefill is out of the spike's scope.

An `nsys` profile of Qwen3-0.6B decode puts 24% in `lm_head`, 24% in `gate_up_proj`, 14% in
`qkv_proj`, 13% in `down_proj` and 12% in `attention`. The GEMVs run at about 100-130 GB/s, against
roughly 273 GB/s of LPDDR5x bandwidth. Launch overhead is already gone behind the CUDA graph, so a
megakernel tape scheduler or a green-context SM split is not justified by these numbers yet. The
gap to llama.cpp is GEMV bandwidth and the f32 activations and KV cache.
