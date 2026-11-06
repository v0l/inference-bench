# Server

`inference-bench-server` is a headless OpenAI-compatible HTTP server. It has no egui or wgpu dependencies, so it builds on a GPU box without a display. The same library (`crates/server`) backs the `inference-bench serve` subcommand and the GUI Server tab.

```sh
cargo build --release -p inference-bench-server -p inference-bench-specrt
INFERENCE_BENCH_API_KEYS=sk-local ./target/release/inference-bench-server \
    --model crates/specrt/specs/qwen3-0.6b.toml --bind 127.0.0.1:8080
```

`inference-bench serve ...` (alias `server`) takes the same flags.

## Endpoints

| Path | Notes |
| --- | --- |
| `GET /v1/models` | Configured models, with a `loaded` flag |
| `POST /v1/chat/completions` | Stream (SSE, `stream_options.include_usage`) and non-stream |
| `POST /v1/completions` | Stream and non-stream |
| `POST /v1/embeddings` | 501: no configured backend supports embeddings |
| `POST /v1/bench` | `{"model", "prompts": [...], "max_tokens", "repeat"}` returns per-run and summary stats (tokens/s, TTFT and decode p50/p95) |
| `GET /health` | Always open |
| `GET /metrics` | Prometheus text. Open unless `protect_metrics = true` |
| `GET /admin/models`, `POST /admin/models/{id}/load`, `POST /admin/models/{id}/unload` | Model management (needs the API key when auth is on) |

Chat messages are flattened to a prompt: a single user message is sent as is, longer conversations as `role: text` lines. The specrt backend applies the model's own chat template to a single user message. The specrt worker does not stream, so its SSE responses arrive as one content chunk when generation ends; the `openai` upstream backend streams token by token.

Every response carries `x-request-id` (a client supplied one is kept). Errors use the OpenAI shape `{"error": {"message", "type", "param", "code"}}`.

## Config

Pass `--config server.toml` (see `packaging/server.example.toml`). Unknown keys are rejected.

```toml
bind = "127.0.0.1:8080"
api_keys = ["sk-a"]                  # plus comma separated keys in $INFERENCE_BENCH_API_KEYS
api_keys_env = "INFERENCE_BENCH_API_KEYS"
protect_metrics = false
cors_origins = ["https://app.example"]   # "*" allows any origin
tls = { cert = "cert.pem", key = "key.pem" }
models_dir = "specs/"                # every *.toml becomes a model named after the file
default_model = "qwen3-0.6b"
max_concurrent = 1                   # generations running at once
queue_depth = 16                     # waiting requests before 429
retry_after_secs = 1
idle_unload_secs = 900
max_loaded_models = 1                # N concurrent models
vram_budget_bytes = 20000000000      # evict idle models to stay inside
default_max_tokens = 256
shutdown_grace_secs = 30
access_log = "/var/log/inference-bench/access.log"
log_prompts = false

[default_backend]                    # specrt, transformers or openai (llama.cpp, vLLM, Ollama proxy)
kind = "specrt"

[[models]]
id = "qwen3-0.6b"
model = "specs/qwen3-0.6b.toml"
vram_bytes = 4000000000              # used for the budget until the measured value is known
preload = true

[[models]]
id = "llama"
[models.backend]
kind = "openai"
url = "http://127.0.0.1:8081"
```

Models load on first request. A model with requests in flight is never evicted; if nothing can be evicted the request gets 503. A client that disconnects cancels its generation (queued or running). A full queue answers 429 with `Retry-After`. SIGINT and SIGTERM stop accepting connections and drain in-flight requests for up to `shutdown_grace_secs`.

For a `specrt` model `max_concurrent` above 1 queues requests on the single GPU worker; for `openai` upstreams each slot gets its own connection.

## Metrics

`inference_requests_total{endpoint,status}`, `inference_prompt_tokens_total{model}`, `inference_completion_tokens_total{model}`, `inference_ttft_seconds` (histogram), `inference_decode_tokens_per_second{window="last|mean"}`, `inference_queue_depth`, `inference_in_flight`, `inference_models_loaded`, `inference_model_vram_bytes{model}`, `inference_gpu_memory_used_bytes` (from `nvidia-smi`, absent where it reports nothing, such as unified memory boxes), `inference_requests_cancelled_total`, and model load, failure and unload counters.

## Logging

All crates use `tracing`. Flags (global on `inference-bench` and `inference-bench-server`):

| Flag | Effect |
| --- | --- |
| `--log-level` | Filter such as `info` or `specrt=debug,info`. Falls back to `RUST_LOG`, then `info` |
| `--log-format human\|json` | Human on a TTY, JSON lines with `json` |
| `--log-dir DIR` / `--log-file` | Rolling JSON log file in `DIR` or the platform state dir (`~/.local/state/inference-bench/logs`). Rotates at 20 MiB, keeps 10 files, drops files older than 14 days |
| `--log-prompts` | Log prompts at info (otherwise only at `trace`) |

Each request runs in a `request` span carrying `request_id`, `method`, `path`, `model`, `backend`, `prompt_tokens`, `completion_tokens`, `ttft_ms`, `decode_tok_s`, `finish_reason` and `status`, so every line logged while serving it has the id. Model loads log the spec hash, JIT cache hit and compile time, and VRAM before and after. Upstream HTTP errors log the status and a body snippet. Panics are logged with a backtrace. API keys are only ever logged redacted (`sk...ey (19 chars)` form), never in full.

GUI logs view: `inference_bench_core::logging::LogBuffer` is a `tracing` layer holding a bounded ring of entries with `snapshot(level, search)`, `text(...)` for copy and `save(path, ...)`. Add it through `LogConfig { buffer: Some(buffer), .. }`.

## Packaging

* `packaging/systemd/inference-bench-server.service`: put the config in `/etc/inference-bench/server.toml` and keys in `/etc/inference-bench/server.env`.
* `packaging/docker/Dockerfile`: CUDA image, run with `--gpus all`; mount a config and the Hugging Face cache.

## Spark run (Qwen3-0.6B, specrt, GB10)

16 concurrent `openai` python clients, 64 new tokens each, `max_concurrent = 1`: 16/16 ok, 117.5 tokens/s aggregate, p50 4.9 s, p95 8.2 s (requests queue behind one GPU worker; latency is queue wait). First load 2.7 s including a cold NVRTC compile.
