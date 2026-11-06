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

## Window

`inference-bench-ui` is a local LLM app in the style of LM Studio, with a rail of five tabs. Every
slow call (disk scan, Hugging Face, model load, generation, bench, GPU poll) runs off the UI thread.

- Chat: pick a backend and model, load or unload, streaming markdown replies, system prompt,
  temperature, top-p, seed and max tokens, a context bar, chat template toggle, saved conversations,
  regenerate, stop and copy. TTFT, prefill and decode tok/s sit under each reply and in the status bar
  with VRAM and GPU load (`nvidia-smi`).
- Models: everything found in the Hugging Face cache, GGUF folders and `crates/specrt/specs`, with
  size, architecture, quant, context length and a supported or unsupported architecture badge per
  backend. Search and download from Hugging Face with resume, reveal in folder, delete.
- Bench: pick models, backends and prompt lengths, run with live progress, results table and charts,
  compare two runs, export JSON or CSV.
- Server: an OpenAI-compatible server (`/v1/models`, `/v1/chat/completions`, `/v1/completions`) for
  the loaded model, with port, API key, LAN toggle and a request log.
- Settings: endpoint URLs and API keys (`settings.json`, mode 0600), model folders, Hugging Face token.

![chat](docs/gui/chat.png)
![models](docs/gui/models.png)
![bench](docs/gui/bench-results.png)

## Layout

- `crates/core`: `Backend` trait, `RunResult`, the OpenAI-compatible and worker backends, matrix runner, logit diff, result store.
- `crates/cli`: the `inference-bench` command.
- `crates/ui`: the `inference-bench-ui` window.
- `crates/specrt`: the native CUDA runtime and its `inference-bench-specrt` worker. See its README.

A measured run on a DGX Spark is in `docs/spark-matrix.md`.
