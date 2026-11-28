# docs

Every doc, what it covers and the code it describes. Docs describe the current code only: for how a doc or a default got here, use `git log -p -- docs/<file>` and the issue. A file in `docs/` without a line here fails `crates/kir/tests/docs_links.rs`, as does a broken link or a named path that does not exist.

## design: how each area works now

| Doc | Covers | Code |
|---|---|---|
| [design/engine.md](design/engine.md) | crate map, backends, option schema and profiles, kernel cache, weight loading, cold start | `crates/specrt/src/runtime.rs`, `engine.rs`, `jit.rs`, `loader.rs`, `options.rs` |
| [design/kir.md](design/kir.md) | kir: targets, kernel kinds, emitters, the kir only guards, kernel descriptions, graph specs, state kinds | `crates/kir/src/`, `crates/specrt/src/codegen.rs`, `graph.rs`, `plan.rs`, `state.rs` |
| [design/kir-compiler.md](design/kir-compiler.md) | plan signatures, the fusion engine and cost model, compile passes, the pass contract, the greedy gap table read by `fusion_exhaustive`, open kernel gaps | `crates/specrt/src/signature.rs`, `fusion.rs`, `compile.rs`, `crates/kir/src/heavy.rs` |
| [design/kir-metal.md](design/kir-metal.md) | plan for a Metal target and backend (#1371 to #1381), not built | `crates/kir/src/wgsl.rs`, `crates/specrt/src/backends/` |
| [design/gemm.md](design/gemm.md) | weight GEMM and GEMV families, tiles and splits, measured GEMM selection | `crates/kir/src/gemm.rs`, `crates/kir/src/cuda/gemm/`, `crates/specrt/src/gemm.rs`, `crates/specrt/src/runtime/tune.rs` |
| [design/quantisation.md](design/quantisation.md) | weight formats, activation quant, output and draft heads, KV dtype, requant, quantise at load, quantcheck, quantised weight cache | `crates/specrt/src/nvfp4.rs`, `quantize.rs`, `quantcheck.rs`, `quant_cache.rs`, `spec.rs` |
| [design/scheduler.md](design/scheduler.md) | paged KV, steps, prefill budget, admission, preemption and swap, structured output, defaults and why | `crates/specrt/src/sched.rs`, `serve.rs`, `statepool.rs`, `grammar.rs`, `crates/core/src/serving.rs` |
| [design/mtp.md](design/mtp.md) | MTP speculative decoding, the round, delta-net deferred commit | `crates/specrt/src/mtp.rs`, `drafts.rs`, `crates/specrt/src/runtime/mtp.rs` |
| [design/linear-attention.md](design/linear-attention.md) | gated delta-net, attention kernels, YaRN context extension | `crates/kir/src/cuda/linear/`, `crates/kir/src/attention.rs`, `crates/specrt/src/spec.rs` |
| [design/prefix-cache.md](design/prefix-cache.md) | prefix cache keys, blocks and snapshots; session snapshots to disk | `crates/specrt/src/prefix.rs`, `sessions.rs` |
| [design/moe-offload.md](design/moe-offload.md) | MoE kernels and routing; layer streaming, expert cache, NVMe tier | `crates/specrt/src/runtime/moe/`, `crates/kir/src/moe.rs`, `crates/specrt/src/offload.rs`, `diskio.rs` |
| [design/distributed.md](design/distributed.md) | placement and solve, tensor parallel, pipeline stages, discovery, hosts, RDMA and GPU direct | `crates/specrt/src/placement.rs`, `solve.rs`, `tp.rs`, `allreduce.rs`, `stage.rs`, `wire.rs`, `rdma.rs`, `host.rs` |
| [design/vision.md](design/vision.md) | image and video preprocessing, encoder, LM splice, media limits | `crates/specrt/src/vision.rs`, `vision_encoder.rs`, `media_encode.rs`, `crates/core/src/media.rs` |
| [design/speech.md](design/speech.md) | Whisper, Phonon text to speech, the audio routes, duplex sessions | `crates/specrt/src/speech/`, `crates/specrt/src/duplex/`, `crates/specrt/src/codec/`, `crates/server/src/audio.rs`, `duplex.rs` |
| [design/diffusion.md](design/diffusion.md) | text to image and video: denoiser, scheduler, VAE, admission, routes | `crates/specrt/src/denoiser.rs`, `sampler.rs`, `crates/specrt/src/vae/`, `media.rs`, `crates/server/src/images.rs`, `videos.rs` |
| [design/gguf.md](design/gguf.md) | GGUF loading, quant paths, bounds against llama.cpp | `crates/specrt/src/gguf.rs`, `gguf_iq.rs`, `weights.rs` |
| [design/wgpu.md](design/wgpu.md) | the wgpu backend, its kernels and refusals | `crates/specrt/src/backends/`, `crates/specrt/src/wgsl.rs` |
| [design/specs.md](design/specs.md) | family specs: matching, strict and describe modes, the decision table, family features | `crates/specrt/src/family.rs`, `crates/specrt/specs/` |
| [design/app.md](design/app.md) | the desktop app: pages, load dialog, hardware and clusters | `crates/ui/src/` |
| [design/bare-metal.md](design/bare-metal.md) | plan for a bare metal appliance (#1525 to #1532), not built | none yet |

## models: per model notes

| Doc | Covers | Code |
|---|---|---|
| [models/qwen3.8-27b.md](models/qwen3.8-27b.md) | the 27B FP8 and NVFP4 checkpoints, specs, memory, current speed and gates | `crates/specrt/tests/fixtures/qwen3.8-27b-nvfp4.toml`, `crates/specrt/tests/real27b.rs` |
| [models/glm5.md](models/glm5.md) | GLM-5.3 Flash: mHC, KDA, pooled DSA, ModelOpt mixed precision | `crates/specrt/specs/family-glm5_next.toml`, `crates/specrt/src/quant_map.rs`, `csf.rs` |
| [models/qwen4exp.md](models/qwen4exp.md) | Qwen3.8-Flash-Next: gated residual, QSA, PLE, MTP | `crates/specrt/specs/family-qwen4_exp.toml` |
| [models/gemma3.md](models/gemma3.md) | Gemma 3 and 3n text, the SigLIP image path | `crates/specrt/specs/family-gemma3.toml`, `family-gemma3n.toml` |
| [models/llama4.md](models/llama4.md) | Llama 4: chunked attention, NoPE layers, top 1 MoE, tiled vision | `crates/specrt/specs/family-llama4.toml` |
| [models/haru-4b.md](models/haru-4b.md) | Haru-4B: tensors, streams, MeanFlow head, codec, prompt, feasibility | `crates/specrt/specs/family-haru.toml`, `crates/specrt/graphs/codec.toml`, `meanflow.toml` |

## guides: how to run, test, bench and add models

| Doc | Covers |
|---|---|
| [guides/testing.md](guides/testing.md) | test tiers, setup, every test file, reference data, the GPU rules, speed measurement |
| [guides/test-fixtures.md](guides/test-fixtures.md) | the tiny 27B shaped fixture and the Qwen3.5-4B NVFP4 bench |
| [guides/perf-pass.md](guides/perf-pass.md) | the per architecture speed loop and `bench --ab` |
| [guides/server.md](guides/server.md) | `inference-bench-server`: endpoints, config, metrics, packaging, request fields |
| [guides/specs-update.md](guides/specs-update.md) | where specs come from and how they update from GitHub |
| [guides/haru-validation.md](guides/haru-validation.md) | checking the Haru engine against its reference package |

## generated: written or read by scripts and tests

| File | Written by | Read by |
|---|---|---|
| [generated/model-coverage.md](generated/model-coverage.md) | `crates/specrt/scripts/model_coverage.sh` (`POPULAR_WRITE=1`) | `crates/specrt/tests/popular_models.rs` |
| [generated/engine-options.md](generated/engine-options.md) | `crates/specrt/tests/options_doc.rs` (`OPTIONS_WRITE=1`) | the same test |
| [generated/qwen27-baseline.md](generated/qwen27-baseline.md) | `crates/specrt/scripts/specrt_baseline.sh` through `baseline_report.py --update-doc` | |
| [generated/qwen27-run/](generated/qwen27-run/) | `crates/specrt/scripts/specrt_baseline.sh`, `vllm_baseline.py` (vLLM and specrt run JSON and the vLLM logs) | `crates/specrt/tests/baseline_report.rs`, `baseline_report.py` |
| [generated/perf/scoreboard.md](generated/perf/scoreboard.md) | `crates/specrt/scripts/perfpass.sh` through `perfpass_report.py merge`; the paired A/B claims by hand | |
| [generated/perf/gemm-select-sweep-sm120.tsv](generated/perf/gemm-select-sweep-sm120.tsv) | `crates/specrt/tests/gemm_select_speed.rs` (`SPECRT_SWEEP_OUT`) | `crates/specrt/tests/gemm_select.rs` |
| [generated/perf/kernel-static-tables.md](generated/perf/kernel-static-tables.md) | `crates/specrt/scripts/kernel_static.sh` | |
| [generated/perf/kernel-vs-tables.md](generated/perf/kernel-vs-tables.md), [generated/perf/kernel-vs.json](generated/perf/kernel-vs.json) | `crates/specrt/scripts/kernel_vs.sh`, `kernel_vs.py report` | |

## img

[img/app/](img/app/): app screenshots used by the top level `README.md`, [design/app.md](design/app.md) and [guides/specs-update.md](guides/specs-update.md). UI tests write fresh ones to `SHOTS=<dir>`, outside the repo.
