# Changelog

All notable changes to Inference Bench are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
