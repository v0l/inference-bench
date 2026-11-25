# Changelog

All notable changes to Inference Bench are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-25

### Added
- New models: Gemma 4 (dense, 26B-A4B mixture of experts, E2B and E4B), Gemma 3n, Kimi Linear, Llama 4 (with its tiled vision tower), Qwen3 VL (with deepstack), Qwen3.8 Flash Next (four stream residual, n-gram embeddings and its multi token prediction layer), GLM-5.3 Flash (served from its mixed precision quant map, across two cards), GLM-4.5-Air, and GGUF i-quants IQ1 to IQ4.
- Image generation: `POST /v1/images/generations` with progress, a queue, previews and cancellation, an image load page and a Generate page, and FLUX.2 klein as the first model. `/v1/models` carries each model's type.
- A load page replaces the load dialog, with the common options first. The planner and the dialog use the same two GiB reserve per card, and a model too big for one card is shown as needing offload, with the cards, host memory and NVMe it needs.
- Offload: an NVMe expert tier, scored refill of the expert cache (now the default), pointer table kernels, KV swap and snapshots on the disk pool, and a read counter.
- The compiler: kernels are built only through kir, passes run from a typed plan (norm GEMVs from the prologue slot, store epilogues), a cost model with a dry run, a pass contract checked after every pass, and a Metal plan.
- Per request table: prefill rate is measured by the engine and shown only when 256 uncached tokens were prefilled; new queue and first token columns and a host busy flag.
- `/v1/capabilities`, a `doctor` command, a `warmup` command with a CUBIN cache, and a long prompt prefill that no longer stalls decode (default burst 16384 tokens, four long prompts in flight).
- A 5 percent bar for every speed change, a paired A/B tool, and one scoreboard.

### Changed
- Scored expert refill is the default offload policy.
- Chat templates render `generation` blocks, `break`, `strftime_now` and `| string` on maps as the Python templates do.
- Spec format 20.
## [0.1.1] - 2026-11-17

### Added
- Windows: when nvrtc is missing, the app downloads the two CUDA compiler DLLs from NVIDIA (checked against pinned checksums) the first time a model loads, with a progress bar.
- The Hardware tab has an in use switch for every GPU and every paired machine. Switched-off GPUs are left out of fit checks and loads.
- The load dialog shows the load plan per GPU, and Load stays available when the planner thinks a model will not fit.
- A Mac running the app can share its Metal GPU in a cluster.

### Changed
- The top and bottom bars sum VRAM and average utilisation over every GPU.
- Discover keeps Hugging Face's order, checks repos as they arrive, connects over IPv4 and fetches GGUF headers in one pass; a 25 repo search takes about 10 s instead of 33 s.
- Fit checks in Discover and the load dialog use the same multi-GPU planner.
- The Low memory profile changes ten options instead of four.
- The Models page header is one compact toolbar; library cards no longer show spec lines.
- Model folders save and rescan when changed.
- The load dialog scrolls on small windows, and long status lines are shortened.

### Fixed
- Pairing, the wire handshake and API key generation read `/dev/urandom` and failed on Windows.
- A missing nvrtc now gives one short message instead of a long library search list.

## [0.1.0] - 2026-11-17

### Added
- `inference-bench launch` (`--plan`, `--yes --no-tui`, `--status`, `--stop`, `--access local|lan`, `--output json`) and `inference-bench connect [app] [--check|--write]` for Claude Code, Codex, OpenCode, pi and agentbench. See docs/server.md.
- Sharing the GPU: `lazy_load`, `min_free_vram_mib` (503 with `Retry-After`), `before_load`, `POST /load` and `POST /unload`, idle unload in app mode, `unloaded` in `/health` and `/v1/models`.
- vLLM named Prometheus series (`vllm:num_requests_running`, `vllm:kv_cache_usage_perc`, TTFT and inter-token histograms, spec decode counters) beside ours.
- Spec files update themselves from GitHub: the app and `inference-bench-server` check at start and every 24 hours (ETag, size caps, parse check, atomic swap), keep the last 3 generations, and work offline from the cache or the snapshot built into every release. `inference-bench specs status|update|rollback`, a "check specs now" button, spec provenance on the Models page, spec update settings with a tag pin, `/admin/specs` and two metrics. See docs/specs-update.md.
- `min_engine` in spec files: a spec for a newer engine is listed as needing a newer inference-bench instead of failing.
- Models and the server accept a spec name such as `qwen3-0.6b` where a spec path was needed.
- `run`, `matrix`, `logit-diff` and `results` commands, with results saved as JSON.
- OpenAI-compatible, `transformers` and specrt backends.
- The `inference-bench-ui` window with a matrix view, run history and charts.
- `crates/specrt`, the spec-driven CUDA runtime moved in from bench-suite.
- An LM Studio style window: a model library, a chat page on the shared egui_bench chat module, bench and server tabs, settings and logs.
- `inference-bench-server`, a headless OpenAI-compatible server with metrics, logging and a Dockerfile.
- Tensor-core prefill, split-K decode attention and batched paged decode attention in specrt, with measurements against llama.cpp and vLLM in `docs/spark-matrix.md`.

### Changed
- Models run on the native engine in process, on a dedicated thread with cancel and panic containment; the `inference-bench-specrt` worker binary is gone. Other engines are benchmark targets only (`bench --target openai:URL`), and the transformers backend is removed. Old settings and server configs with backend entries still load.
