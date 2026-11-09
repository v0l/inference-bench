# Server

`inference-bench-server` is a headless OpenAI-compatible HTTP server for the native engine: every model it serves runs on `specrt` in the server process. It has no egui or wgpu dependencies, so it builds on a GPU box without a display. The same library (`crates/server`) backs the `inference-bench serve` subcommand and the GUI Server tab.

```sh
cargo build --release -p inference-bench-server
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
| `POST /v1/embeddings` | 501: the engine has no embeddings |
| `POST /v1/bench` | `{"model", "prompts": [...], "max_tokens", "repeat"}` returns per-run and summary stats (tokens/s, TTFT and decode p50/p95) |
| `GET /health` | Always open |
| `GET /metrics` | Prometheus text. Open unless `protect_metrics = true` |
| `GET /admin/models`, `POST /admin/models/{id}/load`, `POST /admin/models/{id}/unload` | Model management (needs the API key when auth is on) |
| `GET /admin/specs` | Spec sources, the last update check and where each spec came from. See `docs/specs-update.md` |

Chat requests apply the model's chat template and stop at the end-of-turn ids from generation_config. Replies stream in chunks of up to 16 decode steps. The engine runs on its own thread: a CUDA error or out of memory fails that request, and a panic unloads the model and fails the request, so the server keeps running.

Every response carries `x-request-id` (a client supplied one is kept). Errors use the OpenAI shape `{"error": {"message", "type", "param", "code"}}`.

## Config

Pass `--config server.toml` (see `packaging/server.example.toml`). Unknown keys are rejected. Older configs with `[default_backend]` or `[models.backend]` tables still load: a `specrt` table gives its `max_ctx`, and a model on another engine is dropped with a warning in the log.

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
# top_k = 20
# top_p = 0.95
# min_p = 0.0
shutdown_grace_secs = 30
access_log = "/var/log/inference-bench/access.log"
log_prompts = false
max_ctx = 4096                       # context the engine allocates, per model with max_ctx

[[models]]
id = "qwen3-0.6b"
model = "specs/qwen3-0.6b.toml"
vram_bytes = 4000000000              # used for the budget until the measured value is known
preload = true

[specs]                              # spec updates from GitHub, see docs/specs-update.md
auto_update = true
repo = "v0l/inference-bench"
branch = "master"
tag = ""                             # pin to a tag, e.g. "v0.2.0"
update_hours = 24                    # 0 checks only at start
folders = ["/etc/inference-bench/specs"]   # your own specs, they win over downloaded ones
```

A model's `model` can be a spec name such as `qwen3-0.6b`: it is looked up in `[specs].folders`, then the updated cache, then the specs built into the binary.

Models load on first request. A model with requests in flight is never evicted; if nothing can be evicted the request gets 503. A client that disconnects cancels its generation (queued or running). A full queue answers 429 with `Retry-After`. SIGINT and SIGTERM stop accepting connections and drain in-flight requests for up to `shutdown_grace_secs`.

Each model runs one request at a time on the GPU; `max_concurrent` above 1 queues the rest.

## Metrics

`inference_requests_total{endpoint,status}`, `inference_prompt_tokens_total{model}`, `inference_completion_tokens_total{model}`, `inference_ttft_seconds` (histogram), `inference_decode_tokens_per_second{window="last|mean"}`, `inference_queue_depth`, `inference_in_flight`, `inference_models_loaded`, `inference_model_vram_bytes{model}`, `inference_gpu_memory_used_bytes` (from `nvidia-smi`, falling back to the CUDA driver's free/total memory where nvidia-smi reports N/A, as on the GB10), `inference_requests_cancelled_total`, model load, failure and unload counters, `inference_specs_checks_total{result="updated|unchanged|failed"}` and `inference_specs_last_success_timestamp_seconds`.

## Logging

All crates use `tracing`. Flags (global on `inference-bench` and `inference-bench-server`):

| Flag | Effect |
| --- | --- |
| `--log-level` | Filter such as `info` or `specrt=debug,info`. Falls back to `RUST_LOG`, then `info` |
| `--log-format human\|json` | Human on a TTY, JSON lines with `json` |
| `--log-dir DIR` / `--log-file` | Rolling JSON log file in `DIR` or the platform state dir (`~/.local/state/inference-bench/logs`). Rotates at 20 MiB, keeps 10 files, drops files older than 14 days |
| `--log-prompts` | Log prompts at info (otherwise only at `trace`) |

Each request runs in a `request` span carrying `request_id`, `method`, `path`, `model`, `backend`, `prompt_tokens`, `completion_tokens`, `ttft_ms`, `decode_tok_s`, `finish_reason` and `status`, so every line logged while serving it has the id. Model loads log the spec hash, JIT cache hit and compile time, and VRAM before and after. Upstream HTTP errors log the status and a body snippet. Panics are logged with a backtrace. API keys are only ever logged redacted (`sk...ey (19 chars)` form), never in full.

The GUI Server tab runs this same library in-process over the loaded model, and its Logs tab reads the `LogBuffer`. `inference_bench_core::logging::LogBuffer` is a `tracing` layer holding a bounded ring of entries with `snapshot(level, search)`, `text(...)` for copy and `save(path, ...)`. Add it through `LogConfig { buffer: Some(buffer), .. }`.

## Packaging

* `packaging/systemd/inference-bench-server.service`: put the config in `/etc/inference-bench/server.toml` and keys in `/etc/inference-bench/server.env`.
* `packaging/docker/Dockerfile`: CUDA image, run with `--gpus all`; mount a config and the Hugging Face cache.

## Spark run (Qwen3-0.6B, specrt, GB10)

16 concurrent `openai` python clients, 64 new tokens each, `max_concurrent = 1`: 16/16 ok, 143.2 tokens/s aggregate, p50 3.5 s, p95 6.5 s (requests queue behind the one engine; latency is queue wait). A streamed reply arrives as 9 chunks, first at 0.03 s and last at 0.73 s. Loaded model VRAM reads 3.9 GB, device used 99 GB.

## Unsupported request fields

A field the server does not handle returns 400 naming it: `n` other than 1, `logprobs`, `top_logprobs`, `response_format` other than `text`, non-zero `frequency_penalty` or `presence_penalty`, named or `required` `tool_choice`, and any unknown top-level key. `user`, `parallel_tool_calls`, `store` and `metadata` are accepted and ignored.

## Tool loop parity with vLLM (Qwen3.8-27B)

`crates/server/scripts/tool_parity.py URL MODEL OUT` sends 20 temperature 0 turns (one prior tool exchange, then a user turn, with `get_weather` and `add`) through the `openai` client, plus a 3 turn loop with canned tool results. specrt ran with `tool_parser = "qwen3_coder"`, `reasoning_parser = "qwen3"`, `reasoning_effort = "medium"`; the reference was the vLLM server on port 8001.

| Check | Result |
|---|---|
| 20 single turns, tool name, arguments and finish reason | 20 identical |
| 3 turn loop (two calls in one reply, then a final answer) | identical |
| `n=2` through the client | 400 `n: value is not supported` |

Qwen3-0.6B with the hermes parser ran the same client loop (call, tool result, answer) correctly.
