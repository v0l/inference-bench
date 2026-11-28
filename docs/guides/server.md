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
| `GET /v1/models` | Configured models, with a `loaded` flag and `type` (`text`, `image`, `video`, `duplex`, `transcription`, `speech`) so clients can filter |
| `POST /v1/audio/transcriptions` | OpenAI speech to text, multipart: `file` (WAV PCM16 or float32, any rate, resampled to the model's; stereo is downmixed), `model` (`whisper-1` or an unknown id picks the loaded speech to text model), `response_format` (`json` default `{"text"}`, `text` the plain string, `verbose_json` adds `language`, `duration`, `segments`), `language` (skips detection), `prompt` (previous text), `temperature` (0 to 1, 0 greedy). Unknown fields are ignored. A file over `[media] max_audio_bytes` is a 413; jobs share the image job slots and queue (`[images] jobs_per_gpu`, `queue_depth`), so a full queue is a 429 with `Retry-After` |
| `GET /v1/audio/models` | `{"object": "list", "data": [{"id", "object", "label", "size_mb", "installed", "default"}]}` of the configured speech to text models, the shape agentbench's speech model picker reads |
| `GET /v1/audio/status` | `{"device", "reading_on", "speaking_on"}` (agentbench's `AudioStatus`): `GPU` or `CPU`, and where the loaded speech to text and text to speech models run |
| `POST /v1/audio/speech` | OpenAI text to speech, JSON: `input` (required, 1 to 20,000 characters, 400 otherwise), `voice` (a name from `/v1/audio/voices`, case insensitive; an unknown name such as `alloy` gets the model's default voice, `Hadley` for Phonon), `model` (`kokoro`, `tts-1` or an unknown id picks the loaded text to speech model), `response_format` (`pcm` streams raw PCM16 little endian mono frame by frame with `content-type: audio/L16`, `x-sample-rate: 24000`, `cache-control: no-store`; anything else answers one `audio/wav`), `seed` (extension, the same seed gives the same samples), `speed` accepted and ignored. A client that hangs up stops the work after the frame in hand. Jobs share the media slots and queue (429 with `Retry-After` when full); no speech model configured is a 404 |
| `GET /v1/audio/voices` | `{"voices": [{"id", "label", "installed"}]}` from the model's voice catalogue (`[tts.voices]` in the family spec: `gradium/phonon-voices` `voices.jsonl`, fetched when missing), labels as agentbench's `Voice::label` (`Hadley, American`, `Alfie, British man`) |
| `POST /v1/images/generations` | OpenAI images: `prompt`, `n`, `size`, `seed`, `response_format` (`b64_json` default, or `url`), plus `negative_prompt`, `steps`, `guidance`, `sampler`, `partial_images`. `stream: true` sends `image_generation.progress` (`step` of `steps`), `image_generation.partial_image` every `partial_images` steps and `image_generation.completed`. A size off the model's pixel step is a 400 naming the nearest valid size; a full image queue is a 429 with `Retry-After`; a dropped client cancels the job between steps. Image i of n uses `seed + i`; the same seed gives the same bytes. See `[images]` below |
| `GET /v1/images/files/{id}.png` | The image a `url` response named, behind the same auth, kept `images.file_ttl_secs` |
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
| `GET /admin/specs` | Spec sources, the last update check and where each spec came from. See `docs/guides/specs-update.md` |

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
# temperature = 0.7           # unset: the model's generation_config.json, else greedy
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

[specs]                              # spec updates from GitHub, see docs/guides/specs-update.md
auto_update = true
repo = "v0l/inference-bench"
branch = "master"
tag = ""                             # pin to a tag, e.g. "v0.2.0"
update_hours = 24                    # 0 checks only at start
folders = ["/etc/inference-bench/specs"]   # your own specs, they win over downloaded ones

[images]                             # POST /v1/images/generations
jobs_per_gpu = 1                     # image jobs running at once
queue_depth = 4                      # waiting image jobs before 429
file_ttl_secs = 3600                 # how long a response_format=url image is served
# file_dir = "/var/lib/inference-bench/images"   # default: a folder in the temp dir
partial_images = 0                   # preview every k steps on stream when the request gives none

[media]
max_audio_bytes = 67108864           # POST /v1/audio/transcriptions file limit, 64 MB: a 30 minute 16 kHz PCM16 WAV is 57.6 MB
```

A speech to text model is a Whisper checkpoint folder (`config.json`, `generation_config.json`, `model.safetensors`, `tokenizer.json`), matched to `specs/family-whisper.toml` through `Spec::from_hf_config_speech`; point `model` at the folder or the hub repo id. It runs on the kir GPU engine (`speech::gpu`) and falls back to the host engine when no CUDA device opens. Audio over one 30 s window is decoded long form with timestamps, as `docs/specrt-speech.md` section 2.7 describes.

An image model is a diffusers pipeline folder (`model_index.json` at the top), matched to a family file by its pipeline class (`specs/family-flux2_klein.toml`). Point `model` at the folder or a hub repo id. Before a job runs, the engine checks the denoiser weights plus its activation arena against free GPU memory: when the VAE does not fit it decodes in the largest tile that does, when guidance does not fit it runs the two passes one after the other in one arena, otherwise the request is a 400 naming the bytes. The text encoder is loaded per prompt and released before the steps unless the load page or the request keeps it.

A model's `model` can be a spec name such as `qwen3-0.6b`: it is looked up in `[specs].folders`, then the updated cache, then the specs built into the binary.

Models load on first request. A model with requests in flight is never evicted; if nothing can be evicted the request gets 503. A client that disconnects cancels its generation (queued or running). A full queue answers 429 with `Retry-After`. SIGINT and SIGTERM stop accepting connections and drain in-flight requests for up to `shutdown_grace_secs`.

Each model runs one request at a time on the GPU; `max_concurrent` above 1 queues the rest.

### Renamed ids

Parser and rope ids are named by format. Old ids still load and log a deprecation warning; the mappings live in `crates/server/parser_ids.toml` and `crates/specrt/spec-aliases.toml`.

| Old | New |
|---|---|
| `tool_parser = "qwen3_coder"` or `"qwen3_xml"` | `"xml_function"` |
| `reasoning_parser = "qwen3"` | `"think_tags"` |
| `tool_parser = "granite"` | `"tool_call_list"` (a marker tag, then a JSON list of calls) |
| `tool_parser = "llama3_json"` | `"bare_json"` (the reply is a JSON call, or `;` separated calls) |
| `tool_parser = "phi4_mini_json"` | `"functools_list"` (`functools[...]` JSON list) |
| spec `[rope.scaling] kind = "llama3"` | `"banded"` |

`/props` reports the new ids. There is no `API_VERSION` in this repo to bump.

`tool_call_list`, `bare_json` and `functools_list` are inferred from the chat template (`ToolFormat`). A template that only lists tools with `<|tool|>` and never shows a call (the Phi-4-mini one) is inferred as `functools_list`, but nothing in its prompt tells the model to write `functools[...]`. That parser is unverified on a real checkpoint; set `tool_parser` explicitly if a model needs another format.

## Metrics

`inference_requests_total{endpoint,status}`, `inference_prompt_tokens_total{model}`, `inference_completion_tokens_total{model}`, `inference_ttft_seconds` (histogram), `inference_decode_tokens_per_second{window="last|mean"}`, `inference_queue_depth`, `inference_in_flight`, `inference_models_loaded`, `inference_model_vram_bytes{model}`, `inference_gpu_memory_used_bytes` (from `nvidia-smi`, falling back to the CUDA driver's free/total memory where nvidia-smi reports N/A, as on the GB10), `inference_requests_cancelled_total`, model load, failure and unload counters, `inference_specs_checks_total{result="updated|unchanged|failed"}` and `inference_specs_last_success_timestamp_seconds`. Image jobs: `inference_image_jobs_total{model}`, `inference_images_total{model}`, `inference_image_steps_total`, `inference_image_step_seconds_total`, `inference_image_job_seconds_total`, `inference_image_steps_per_second` (last job), `inference_image_queue_depth`, `inference_image_jobs_running`; requests count under `endpoint="images_generations"`. Transcriptions: `inference_transcriptions_total{model}`, `inference_transcription_audio_seconds_total{model}`, `inference_transcription_seconds_total{model}`; requests count under `endpoint="audio_transcriptions"`, `audio_models` and `audio_status`. Speech: `inference_speech_total{model}`, `inference_speech_audio_seconds_total{model}`, `inference_speech_seconds_total{model}`; requests count under `endpoint="audio_speech"` and `audio_voices`.

Scheduler admission, per loaded specrt model: `inference_admission_blocked_steps_total{reason="slots|budget|long_cap|kv|state|resume"}` counts steps whose admission stopped for that reason while something waited (all slots in use, the step's prefill budget already planned, the long prompt cap, KV blocks short after eviction, no recurrent state, a parked sequence that cannot resume). Admission scans past requests held for the long cap or KV (#1518), so a step that reached the end of the queue counts the first reason that held a request. TTFT splits into three histograms that add up to it within one step: `inference_admission_wait_seconds{reason}` (arrival to admission, labelled by the reason that held the sequence longest, `none` when it was admitted on arrival), `inference_prefill_wait_seconds` (admission to the first prefill step, time spent behind shorter prompts) and `inference_prefill_seconds` (first prefill step to the first token). `inference_queue_seconds` stays as arrival to first prefill step. Gauges `inference_waiting_seqs`, `inference_free_slots` and `inference_long_prompts_in_flight` (running sequences with more than 2048 prompt tokens left), and the traffic mix as `inference_prompt_tokens` and `inference_cached_prompt_tokens` histograms of finished requests. The server tab's per request table shows the reason (`held by`), the admission wait (`admit`) and the prefill wait (`wait`) as width allows; at phone width it keeps the reason.

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

The per-request table splits a request's time into `queue` (arrival to its first prefill step, hidden below 700 px width), `prefill` (first prefill step to the end of the last prompt chunk, measured by the scheduler) and `ttft` (arrival to first token). `new` is the uncached prompt tokens. The prefill rate is new tokens over measured prefill time and is shown only when `new` is at least 256; below that it is a dash, because a few tokens over one scheduler step is step latency, not speed. The rate is never computed from TTFT.

On a saturated host (load average above the core count) the scheduler thread is starved, so decode and TTFT degrade and the table cannot tell. The serving strip adds a warning `host busy` reading with the 1 minute load from `/proc/loadavg` when it is above 0.9 x cores.

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

The image has no GPU at build time, so it ships no kernels. `XDG_CACHE_HOME=/cache` makes the `[runtime] cache_dir` default `/cache/specrt`, so one named volume on `/cache` carries the kernel cache. Once per host, then serve with the same volume:

```bash
docker run --rm --gpus all -v specrt-cache:/cache --entrypoint inference-bench <image> warmup --spec <spec> --max-ctx 8192 --slots 8
docker run --gpus all -v specrt-cache:/cache -p 8080:8080 <image> --model <spec>
```

Pass the same `--max-ctx` and `--slots` the server runs with (`--max-model-len`, `--max-num-seqs`). A different layout is a different cache key and compiles again ([../design/engine.md](../design/engine.md), kernel cache).

## Request fields

`/v1/models` lists each model's own id. A request may name it by id, repo id, short name or any of its `served_names` (server.toml, or the load dialog in the app).

`logprobs: true` with `top_logprobs` up to 20 (chat) or `logprobs: N` (completions) return per-token log probabilities, streamed or not, computed on the host from the sampled row's logits. `n` up to 16 runs that many sequences on the batched engine; usage sums their completion tokens. Both need the batched engine (`max_num_seqs` above 1, or more than one parallel request in the app) and return 400 saying so otherwise. A `developer` message is templated as `system`, and a chat template error from client input is a 400.

A field the server does not handle returns 400 naming it: non-zero `frequency_penalty` or `presence_penalty`, and any unknown top-level key. `response_format` (`json_object`, `json_schema`) and `tool_choice` `required` or named compile a grammar ([../design/scheduler.md](../design/scheduler.md)); the two cannot be combined. `user`, `parallel_tool_calls`, `store` and `metadata` are accepted and ignored.

A request without `temperature` samples at the server config's `temperature`, else at the model's `generation_config.json` (`do_sample: false` is greedy), else greedy. A request without `seed` draws a new one, so the same body can give a different reply; `seed` makes it repeatable, and with `n` choice i uses `seed + i` (#1469; `crates/server/tests/fixtures/chat-tab-thinking-off-loop.json` is the request that looped when an unset seed was 0).

## Tool clients

`crates/server/scripts/tool_parity.py URL MODEL OUT` sends 20 temperature 0 turns (one prior tool exchange, then a user turn, with `get_weather` and `add`) plus a 3 turn loop with canned tool results through the `openai` client, against any OpenAI compatible server; on the 27B with `tool_parser = "xml_function"`, `reasoning_parser = "think_tags"` all 20 turns and the loop matched vLLM.

## Thinking levels and tool-call recovery

Every chat dialect takes `reasoning_effort` (`none|low|medium|high`), `reasoning: {effort}` or `chat_template_kwargs.enable_thinking`. Anthropic budgets map to an effort (under 2K low, under 8K medium, else high) and are also the hard cap.

`reasoning_budget_tokens` (request, or `reasoning_budget_tokens` in the server config) caps the thinking: when the model has produced that many tokens without closing `<think>`, the server appends `reasoning_wrapup` plus `</think>` and the model answers from there. With `answer_room = true` (default) a request with `max_tokens` of 1024 or more and no budget gets one, so thinking is wrapped up when a quarter of the cap (at least 512 tokens) is left. The wrap-up count is `inference_thinking_wrapups_total`.

The tool parser only emits declared tools. It also recovers `<parameter=...>` without a function, JSON inside `<tool_call>`, a `<function=...>` without `<tool_call>`, several calls in one block and a missing `</function>`; repairs trailing commas, single quotes and `True/False/None` in array and object arguments; and maps collapsed namespace names (`mcp_a_b` for `mcp__a__b`) when unambiguous. Anything else stays text. Repairs are counted in `inference_tool_call_repairs_total{kind}`.

Claude Code (`ANTHROPIC_BASE_URL=<server> claude -p ...`, `/v1/messages`) and the Codex CLI (`wire_api = "responses"`, `/v1/responses`) complete a tool round trip against the server; the ignored test `qwen3_0_6b_serves_a_tool_turn_over_messages_and_responses` asserts a correct `get_weather` call on both APIs.
