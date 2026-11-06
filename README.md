<img src="assets/icon.svg" width="64" alt="">

# inference-bench

Run the same prompt through several LLM inference backends and compare them: prefill and decode
tokens per second, time to first token, peak memory, the generated text, and the real error when a
backend cannot run a model (including "unsupported architecture").

Part of the [bench suite](https://github.com/v0l/bench-suite), a launcher that installs, updates
and starts it alongside the other bench tools.

## Backends

- `openai`: any OpenAI-compatible server (vLLM, llama.cpp server, mistral.rs, YALR, Ollama).
- `transformers`: a Python subprocess running Hugging Face `transformers`, the accuracy reference.
- `specrt`: the native spec-driven CUDA runtime in `crates/specrt`, where a new model is a TOML file.

## Command line

```
inference-bench run --backend openai --url http://localhost:8080 --model qwen3 --prompt-file p.txt
inference-bench matrix matrix.toml
inference-bench logit-diff --backend specrt --model specs/qwen3-0.6b.toml
```

Results are saved as JSON in the data directory (`INFERENCE_BENCH_DIR` to move it) and shown by
`inference-bench-ui`.
