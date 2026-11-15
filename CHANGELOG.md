# Changelog

All notable changes to Inference Bench are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `inference-bench launch` (`--plan`, `--yes --no-tui`, `--status`, `--stop`, `--access local|lan`, `--output json`) and `inference-bench connect [app] [--check|--write]` for Claude Code, Codex, OpenCode, pi and agentbench. See docs/server.md.
- Sharing the GPU: `lazy_load`, `min_free_vram_mib` (503 with `Retry-After`), `before_load`, `POST /load` and `POST /unload`, idle unload in app mode, `unloaded` in `/health` and `/v1/models`.
- vLLM named Prometheus series (`vllm:num_requests_running`, `vllm:kv_cache_usage_perc`, TTFT and inter-token histograms, spec decode counters) beside ours.
- Spec files update themselves from GitHub: the app and `inference-bench-server` check at start and every 24 hours (ETag, size caps, parse check, atomic swap), keep the last 3 generations, and work offline from the cache or the snapshot built into every release. `inference-bench specs status|update|rollback`, a "check specs now" button, spec provenance on the Models page, spec update settings with a tag pin, `/admin/specs` and two metrics. See docs/specs-update.md.
- `min_engine` in spec files: a spec for a newer engine is listed as needing a newer inference-bench instead of failing.
- Models and the server accept a spec name such as `qwen3-0.6b` where a spec path was needed.

## [0.1.0] - 2026-11-07

### Changed
- Models run on the native engine in process, on a dedicated thread with cancel and panic containment; the `inference-bench-specrt` worker binary is gone. Other engines are benchmark targets only (`bench --target openai:URL`), and the transformers backend is removed. Old settings and server configs with backend entries still load.

### Added
- `run`, `matrix`, `logit-diff` and `results` commands, with results saved as JSON.
- OpenAI-compatible, `transformers` and specrt backends.
- The `inference-bench-ui` window with a matrix view, run history and charts.
- `crates/specrt`, the spec-driven CUDA runtime moved in from bench-suite.
- An LM Studio style window: a model library, a chat page on the shared egui_bench chat module, bench and server tabs, settings and logs.
- `inference-bench-server`, a headless OpenAI-compatible server with metrics, logging and a Dockerfile.
- Tensor-core prefill, split-K decode attention and batched paged decode attention in specrt, with measurements against llama.cpp and vLLM in `docs/spark-matrix.md`.
