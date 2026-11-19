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
| `POST /v1/messages` | Anthropic Messages, stream and non-stream: tool_use/tool_result, thinking blocks, images, `thinking: {type: disabled|enabled, budget_tokens}`, `output_config.effort`, `x-api-key` auth. A request with no `thinking` field gets no thinking (Claude Code helper calls), switch off with `anthropic_thinking_on_request = false` |
| `POST /v1/messages/count_tokens` | Rough token estimate |
| `POST /v1/responses` | OpenAI Responses, stateless (`previous_response_id` is refused), stream and non-stream, function tools, reasoning items |
| `GET /props`, `GET /slots` | llama.cpp style model properties (chat template, `chat_template_caps`, generation defaults, context) and slots |
| `POST /v1/embeddings` | 501: the engine has no embeddings |
| `POST /v1/bench` | `{"model", "prompts": [...], "max_tokens", "repeat"}` returns per-run and summary stats (tokens/s, TTFT and decode p50/p95) |
| `GET /health` | Always open |
| `GET /metrics` | Prometheus text. Open unless `protect_metrics = true` |
| `POST /load`, `POST /unload` | Load the default model (or `?model=ID`); unload it to give the GPU back. Unload is 409 while a request is running, `?wait=1` waits for them to finish |
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
# default_max_tokens = 4096   # unset: until the context is full
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
served_names = ["qwen3.8-27b"]       # extra names a request may use, like vLLM's --served-model-name

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

`vllm:` series sit beside ours so vLLM Grafana dashboards work, labelled `engine="0",model_name=...`: `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, `vllm:num_preemptions_total`, `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, `vllm:request_success_total{finished_reason}`, `vllm:time_to_first_token_seconds` and `vllm:inter_token_latency_seconds` (histograms), `vllm:request_queue_time_seconds`, `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total`, `vllm:spec_decode_num_draft_tokens_total` and `vllm:spec_decode_num_accepted_tokens_total`. The vLLM series with no counterpart (per-request token histograms, e2e latency, iteration tokens) are not emitted.

## Sharing the GPU

For running beside vLLM or other GPU jobs, in the server config:

| Key | Effect |
| --- | --- |
| `idle_unload_secs` | Unload a model after this many idle seconds. Works in the app too: the app engine is released (`suspended`) and the next request reloads it with the same settings |
| `lazy_load = true` | Ignore `preload`: load on the first request |
| `any_model_name = true` | When exactly one model is loaded, answer to any requested model name and echo that name in the response (all endpoints, including `/v1/messages` and `/v1/responses`). With none or several loaded, unknown names still get 404. Default false; in the app it is the server tab switch |
| `min_free_vram_mib` | Refuse a load with 503 and `Retry-After` (`retry_after_secs`) while the GPU has less free than this. A loaded model keeps serving |
| `before_load` | Shell command run before every load (`INFERENCE_BENCH_MODEL` is set); a non-zero exit refuses the load with 503. Use it to stop or wait for another job |

`/health` has `status: "ok" | "loading" | "unloaded"` and `unloaded_models`; `/v1/models` entries carry `status: "loaded" | "loading" | "unloaded"`.

## Launch and connect

```sh
inference-bench launch Qwen/Qwen3-0.6B --plan                  # what it would download and run, changes nothing
inference-bench launch Qwen/Qwen3-0.6B --yes --no-tui          # download, load, serve, print addresses and key
inference-bench launch crates/specrt/specs/qwen3-0.6b.toml --yes --access lan --detach --output json
inference-bench launch --status                                # also --stop, --output json
inference-bench connect claude-code --write                    # or codex, opencode, pi, agentbench; no app prints all
inference-bench connect --check                                # one prompt and one tool call per app, exit 1 on a failure
```

`launch` uses the load dialog's estimate (`inference_bench_core::fit`, the same code the app calls), refuses a plan that does not fit unless `--force`, and with `--access lan` binds `0.0.0.0` and generates an API key. State is in `launch.json` in the data dir. `connect` takes the address and key from the launched server, or `--url` and `--api-key`; `--write` merges into the app's existing config and keeps a `.bak`. Claude Code is checked over `/v1/messages` and Codex over `/v1/responses`, the others over chat completions.

## App engine and the server tab

In the desktop app the load dialog has a parallel requests slider (1 to 64, default 4, remembered per model). With more than one, the app engine loads through `specrt::serve` (the batched scheduler, `ServeOptions.max_num_seqs`), so chat and the app's OpenAI server share one engine and API requests run alongside chat. With 1 it is the plain single-request engine. The memory estimate counts the KV cache for that many sequences at the chosen context, plus the recurrent state per sequence on hybrid models. If the batched engine cannot load, the app falls back to one sequence and says so on the server tab.

### Several models at once

The app keeps one model loaded by default, so loading a model replaces the current one. Raise `max loaded models` on the Server tab (saved as `server.max_loaded_models` in the app settings, the same name as the server config key) and the load dialog offers "load alongside" next to "replace current". Alongside is disabled, with the reason shown, when the limit is reached or the memory estimate says the model does not fit in free VRAM. Nothing is ever unloaded to make room: unload one first. Each loaded model has its own scheduler, its own KV pool and its own stats.

The app's OpenAI server routes by model id, display name, short alias or served name, and `/v1/models` and `/health` list every loaded model. A request with no `model` goes to the first loaded one. `any_model_name` still applies only while exactly one model is loaded; with two loaded an unknown name is a 404. The Models and Server tabs list every loaded model with its VRAM, context, parallel slots and live load, and an unload button per model. The model bar switches which loaded model the chat talks to, and the same pick decides whose stats the serving section shows. Each loaded model has its own request counts, per-request table and 15 minute history (`docs/gui/server-two-models.png`), and the serving section says which model it is showing. Idle unload and the `/unload` endpoint park one model without touching the others.

Checked on the Spark with `cargo test --release -p inference-bench-server --test app_mode two_real_models -- --ignored` (Qwen3-0.6B and Qwen2.5-0.5B loaded together, both answering by name).

The server tab samples the engine once a second off the UI thread (`inference_bench_core::live`, 15 minutes kept) and shows requests per minute, running and waiting sequences, KV use, prefix hit rate, draft acceptance, preemptions, charts of generation and prompt tok/s, TTFT and inter-token p50 and p95, running sequences and KV, and a per-request table. Measured on Qwen3-0.6B (RTX PRO 6000, 4 concurrent requests of 400 tokens): 843 tok/s aggregate at 1 sequence, 2124 tok/s at 4 (`docs/gui/server-live.png`).

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

## Request fields

`/v1/models` lists each model's own id. A request may name it by id, repo id, short name or any of its `served_names` (server.toml, or the load dialog in the app).

`logprobs: true` with `top_logprobs` up to 20 (chat) or `logprobs: N` (completions) return per-token log probabilities, streamed or not, computed on the host from the sampled row's logits. `n` up to 16 runs that many sequences on the batched engine; usage sums their completion tokens. Both need the batched engine (`max_num_seqs` above 1, or more than one parallel request in the app) and return 400 saying so otherwise. A `developer` message is templated as `system`, and a chat template error from client input is a 400.

A field the server does not handle returns 400 naming it: `response_format` other than `text` (structured output is #591), non-zero `frequency_penalty` or `presence_penalty`, named or `required` `tool_choice`, and any unknown top-level key. `user`, `parallel_tool_calls`, `store` and `metadata` are accepted and ignored.

## Tool loop parity with vLLM (Qwen3.8-27B)

`crates/server/scripts/tool_parity.py URL MODEL OUT` sends 20 temperature 0 turns (one prior tool exchange, then a user turn, with `get_weather` and `add`) through the `openai` client, plus a 3 turn loop with canned tool results. specrt ran with `tool_parser = "qwen3_coder"`, `reasoning_parser = "qwen3"`, `reasoning_effort = "medium"`; the reference was the vLLM server on port 8001.

| Check | Result |
|---|---|
| 20 single turns, tool name, arguments and finish reason | 20 identical |
| 3 turn loop (two calls in one reply, then a final answer) | identical |
| `n=2` through the client | 2 choices (since #585) |

Qwen3-0.6B with the hermes parser ran the same client loop (call, tool result, answer) correctly.

## Thinking levels and tool-call recovery

Every chat dialect takes `reasoning_effort` (`none|low|medium|high`), `reasoning: {effort}` or `chat_template_kwargs.enable_thinking`. Anthropic budgets map to an effort (under 2K low, under 8K medium, else high) and are also the hard cap.

`reasoning_budget_tokens` (request, or `reasoning_budget_tokens` in the server config) caps the thinking: when the model has produced that many tokens without closing `<think>`, the server appends `reasoning_wrapup` plus `</think>` and the model answers from there. With `answer_room = true` (default) a request with `max_tokens` of 1024 or more and no budget gets one, so thinking is wrapped up when a quarter of the cap (at least 512 tokens) is left. The wrap-up count is `inference_thinking_wrapups_total`.

The tool parser only emits declared tools. It also recovers `<parameter=...>` without a function, JSON inside `<tool_call>`, a `<function=...>` without `<tool_call>`, several calls in one block and a missing `</function>`; repairs trailing commas, single quotes and `True/False/None` in array and object arguments; and maps collapsed namespace names (`mcp_a_b` for `mcp__a__b`) when unambiguous. Anything else stays text. Repairs are counted in `inference_tool_call_repairs_total{kind}`.

## Real client check (Qwen3-0.6B, specrt, hermes parser)

Server: `tool_parser = "hermes"`, `reasoning_parser = "qwen3"`, one model on `crates/specrt/specs/qwen3-0.6b.toml`.

Claude Code (`ANTHROPIC_BASE_URL=http://127.0.0.1:18629 claude -p "Read the file note.txt with the Read tool and tell me what it says." --allowedTools Read`):

```
assistant thinking  "Okay, let's see. The user wants me to read the file note.txt ... I should use the Read tool ..."
assistant tool_use  {"name": "Read", "input": {"file_path": "/tmp/t629/work/notes/notes.txt"}}
user      tool_result "File does not exist. Note: your current working directory is /tmp/t629/work." (is_error)
assistant thinking  "... the file doesn't exist ..."
assistant text      "The file `notes.txt` does not exist in your current working directory ..."
result    completed
```

Codex CLI (`wire_api = "responses"`, `codex exec`, shell tool):

```
command_execution  /bin/bash -lc cat   (function_call item from /v1/responses, executed, output sent back as function_call_output)
agent_message      "The note file has been successfully accessed. ..."
turn.completed     input_tokens 22413, output_tokens 342
```

The 0.6B model picks a wrong path or drops an argument, so the answers are poor; the check is that both clients complete the tool round trip over the new endpoints. The ignored test `qwen3_0_6b_serves_a_tool_turn_over_messages_and_responses` asserts a correct `get_weather` call on both APIs.
