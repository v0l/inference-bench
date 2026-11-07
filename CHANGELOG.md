# Changelog

All notable changes to Inference Bench are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Models run on the native engine in process, on a dedicated thread with cancel and panic containment; the `inference-bench-specrt` worker binary is gone. Other engines are benchmark targets only (`bench --target openai:URL`), and the transformers backend is removed. Old settings and server configs with backend entries still load.

### Added
- `run`, `matrix`, `logit-diff` and `results` commands, with results saved as JSON.
- OpenAI-compatible, `transformers` and specrt backends.
- The `inference-bench-ui` window with a matrix view, run history and charts.
- `crates/specrt`, the spec-driven CUDA runtime moved in from bench-suite.
