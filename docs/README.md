# docs index (inventory before the reorganisation, #1561)

Kinds: reference (how it works now), guide (how to run, test, bench or add a model), generated (written or read by a script or test), history (measurement logs, dated reports, finished reviews).

## Top level

| file | kind | what it is |
|---|---|---|
| bare-metal.md | history | #1523 analysis of a bare-metal OS for the server, no code |
| haru-validation.md | guide | Haru-4B against the reference implementation, trace commands |
| jitllm-review.md | history | #393 review of jitllm |
| perf-pass.md | guide | the per-architecture speed loop |
| qwen3.8-27b-vllm-baseline.md | generated + history | vLLM baseline for the 27B; `baseline_report.py --update-doc` rewrites its last section |
| server.md | guide | the headless server, config and API |
| spark-matrix.md | history | 6 Nov Spark matrix against llama.cpp and vLLM |
| specrt-27b.md | reference + history | the real 27B FP8, memory, cold start |
| specrt-defaults.md | reference | engine option schema and profiles |
| specrt-diffusion.md | reference + history | text to image and video |
| specrt-distributed.md | reference + history | pipeline stages, placement, discovery, RDMA |
| specrt-family-specs.md | reference | family TOMLs |
| specrt-gemma3.md | reference | Gemma 3 and 3n |
| specrt-gguf.md | reference + history | GGUF loading and quant paths |
| specrt-glm5.md | reference + history | GLM-5 gap list and real checkpoint |
| specrt-gpu-direct.md | reference | GPU-direct stage hops |
| specrt-graph-spec.md | reference | graph specs grammar |
| specrt-haru.md | reference + history | Haru-4B duplex speech |
| specrt-kir-compiler.md | reference + history | kir pass pipeline; `fusion_exhaustive` reads section 10 |
| specrt-kir-metal.md | reference + history | MSL target plan |
| specrt-kir-runtime-spec.md | reference + history | kernel descriptions and runtime spec |
| specrt-kir.md | reference | kir IR and emitters |
| specrt-kv-quant.md | reference + history | fp8 and int8 KV; named in `spec.rs` |
| specrt-linear-attention.md | reference + history | gated delta-net |
| specrt-long-context.md | reference | YaRN |
| specrt-long-decode.md | history | #646 long-context batched decode attention |
| specrt-model-coverage.md | generated | `popular_models` writes and checks it |
| specrt-moe.md | reference + history | MoE |
| specrt-mtp.md | reference + history | MTP speculation |
| specrt-nvfp4.md | reference + history | NVFP4 27B |
| specrt-offload.md | reference + history | tiered weights and expert cache |
| specrt-prefix-cache.md | reference + history | prefix caching |
| specrt-quantize.md | reference + history | quantise at load |
| specrt-qwen4exp.md | reference + history | Qwen3.8-Flash-Next |
| specrt-requant.md | history | requantise at load measurements |
| specrt-scheduler.md | reference + history | serving core |
| specrt-session-store.md | reference | session snapshots on disk |
| specrt-speech.md | reference + history | Whisper and TTS |
| specrt-stage-server.md | reference + history | stage server over TCP |
| specrt-test-fixtures.md | guide | tiny 27B fixture and 4B bench |
| specrt-tp.md | reference + history | tensor parallel |
| specrt-vision.md | reference + history | image and video input |
| specrt-wgpu.md | reference + history | wgpu backend and Mac speed work |
| specs-update.md | guide | spec updates from GitHub |
| testing.md | guide | test tiers and commands |
| vllm-jit-warmup-review.md | history | #394 review |
| vllm-kernels-review.md | history | #411 review |

## Folders

| folder | kind | what it is |
|---|---|---|
| fixtures/ | generated | `crates/server/tests/api.rs` includes the chat fixture |
| perf/ | generated + history | `scoreboard.md` (perfpass), `gemm-select/` (`gemm_select` test), `kernel-static-tables.md`, `kernel-vs-tables.md` (scripts), per family dated reports |
| qwen27-run/ | generated | vLLM baseline JSON read by `baseline_report` |
| spark-run/, glm5-real/, qwen4exp-real/ | history | raw run output |
| gui/, shots/, img/ | history | screenshots; README links four from `gui/` |
