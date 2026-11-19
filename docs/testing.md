# Testing

One rule first: never wait hours for a 27B window. Run the cheapest tier that exercises your change, hand back its results, and give the exact 27B command for the supervisor to run. The real 27B is the last gate before a merge, not the first thing you run.

## Tiers

Cheapest first. Stop at the lowest tier that can fail because of your change.

| Tier | What | GPU | Time | Required when |
|---|---|---|---|---|
| 1. CPU | Unit tests, spec parsing, codegen text, scheduler, wire formats, server, core, cli, ui | none | about 4 min cold build, 30 s warm | every change |
| 2. Tiny fixtures on GPU | Qwen3-0.6B, Qwen2.5-0.5B variants, Llama-3.2-1B, tinymistral, and the `tiny-qwen3.5*` fixtures, with references in `SPECRT_REF_DIR` | under 3 GB | seconds to 2 min per file | any change to a kernel, the engine, the loader, serving or a spec field |
| 3. Synthetic 27B-shaped layer | `gemm_speed`, `flash_speed`, `gdn_speed`, `kir_speed`, `fp4_batch`, `fp8_batch`: one layer with the 27B's shapes and random weights | under 4 GB | seconds | kernel speed and kernel correctness at 27B shapes |
| 4. Mid-size benchmark | Qwen3.5-4B in NVFP4 in the 27B's layout (`scripts/bench4b.sh`, 4.9 GB on disk, 7.1 GB on the device), fits beside the user's app | about 8 GB | 5 min | end-to-end speed checks: prefill, decode, batch 4/8/16, MTP |
| 5. Real 27B | `tests/real27b.rs`, `vision_lm.rs`, `vision_encoder.rs`, `nvfp4.rs`, the `*27b*.sh` scripts | 25 to 96 GB | 15 to 30 min per run | final gate before merge, or a change specific to that checkpoint |

The tiny27b fixture (`tests/fixtures/tiny-qwen3.5-27b`, #635) is the tier 2 stand-in for every 27B code path: 4 layers in the 27B's pattern, FP8 channel plus NVFP4 act-quant config, an MTP head, a vision tower with mrope 11/11/10, and transformers references beside it. Its tests run in about 100 s:

```bash
flock /tmp/gpu27.lock systemd-run --user --wait --collect -P -q \
  -p MemoryMax=16G -p MemorySwapMax=0 --working-directory=$PWD \
  -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS \
  cargo test -p inference-bench-specrt --release --test tiny27b -- --include-ignored
```

Regenerate with `~/.venv-vllm/bin/python crates/specrt/scripts/tiny27b.py all`. Gates and detail: `docs/specrt-test-fixtures.md`.

Specs for tier 2 are in `crates/specrt/specs/` (`qwen3-0.6b`, `qwen3-0.6b-q8`, `qwen2.5-0.5b`, `-geglu`, `-linear4`, `-yarn4`, `llama-3.2-1b`, `tinymistral-248m`). The Qwen3.5 architecture fixtures (delta-net, fp8, NVFP4, MTP, head dim 256) are checked in under `crates/specrt/tests/fixtures/tiny-qwen3.5*`.

## Setup

Every command below assumes this, run once per shell:

```bash
export CARGO_TARGET_DIR=/mnt/data/cargo-target/inference-bench-<issue>
export CARGO_BUILD_JOBS=4
export SPECRT_REF_DIR=$HOME/.cache/specrt-ref-375
```

- The default target dir is shared by every worktree. Parallel sessions overwrite each other's binaries there, so always use a per-issue one. Delete it when the issue is done.
- Cargo goes under `systemd-run --user --wait --collect -P -q -p MemoryMax=10G --working-directory=$PWD -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS`. Anything that loads weights goes under `-p MemoryMax=16G -p MemorySwapMax=0`. Python that loads the 27B goes under the same 16G cap, never more than one such job at a time.
- Any GPU run goes under `flock /tmp/gpu27.lock` (or `crates/specrt/scripts/gpulock.sh`, which skips the lock when `quiet_window.sh` already holds it). Two GPU jobs must never overlap.
- `/mnt/data` also backs `/tmp` and the user's vLLM compile cache. Keep it under 90% full.

## Tier 1: CPU

```bash
systemd-run --user --wait --collect -P -q -p MemoryMax=10G --working-directory=$PWD \
  -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS cargo test -p inference-bench-specrt
```

Ignored tests are skipped. `fp4_gemv.rs` is not ignored and opens the GPU, so run it on a box with a card or skip it with `--test` selection. Result lines from a run on this branch:

```text
test result: ok. 144 passed; 0 failed; 0 ignored   (specrt lib)
test result: ok. 38 passed; 0 failed; 0 ignored    (tests/spec.rs)
test result: ok. 23 passed; 0 failed; 2 ignored    (tests/prefix.rs)
test result: ok. 21 passed; 0 failed; 0 ignored    (tests/vision.rs)
test result: ok. 14 passed; 0 failed; 0 ignored    (tests/wire.rs)
```

The other crates:

```bash
cargo test -p inference-bench-server -p inference-bench-core -p inference-bench-cli -p inference-bench-kir
cargo test -p inference-bench-ui
```

The last run gave server 147, 24, 59, 46, 9, 16, 5 and 13 passed, ui 73 passed with 9 ignored, and no failures. Five server tests in `crates/server/tests/serving.rs` and two in `crates/core/tests/live_hf.rs` are `#[ignore]` (GPU and Qwen3-0.6B, or the network).

The ui race from #590 is fixed on master.

## Tier 2 and 3: GPU tests

Ignored tests run with `--include-ignored`. Build in release, since the engine is JIT-heavy and slow in debug.

```bash
flock /tmp/gpu27.lock systemd-run --user --wait --collect -P -q \
  -p MemoryMax=16G -p MemorySwapMax=0 --working-directory=$PWD \
  -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS -E SPECRT_REF_DIR \
  cargo test -p inference-bench-specrt --release --test linear --test mtp -- --include-ignored
```

The non-27B suite as one command, and what it gave on this branch (GPU free, 92 GB):

```bash
flock /tmp/gpu27.lock systemd-run --user --wait --collect -P -q \
  -p MemoryMax=16G -p MemorySwapMax=0 --working-directory=$PWD \
  -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS -E SPECRT_REF_DIR \
  cargo test -p inference-bench-specrt --release \
  --test linear --test mtp --test fused --test link --test loader --test statepool \
  --test kv_dtype --test lin_ba_rows --test jit --test stage --test truncation \
  --test fp8_act --test fp4_act --test serving --test generate --test head_quant --test logits \
  -- --include-ignored
```

```text
fp4_act      1 passed    12.9 s      kv_dtype   5 passed   0.7 s
fp8_act      2 passed    15.3 s      lin_ba_rows 2 passed  15.3 s
fused        6 passed     0.3 s      linear    24 passed   28.6 s
jit          4 passed    13.0 s      link       4 passed    4.1 s
loader       4 passed    20.8 s      mtp       16 passed   88.9 s
stage        7 passed     7.1 s      statepool  2 passed    0.2 s
truncation   5 passed    45.3 s      serving   20 passed   15.5 s
generate     6 passed     2.9 s      head_quant 8 passed   88.2 s
logits       9 passed     2.8 s (with the four reference sets below)
```

The Qwen3-0.6B and Qwen2.5-0.5B tests need the HF weights in the HF cache (`generate`, `head_quant`, `logits`, `pipeline`, `serving`, `wgsl`).

### Reference data

References come from transformers (`scripts/reference.py`) or vLLM and are checked against by the engine. They are not in git. If a file is missing the test panics on `unwrap`, so a panic at `logits.rs:20` means a missing `<name>.ref.json`.

| Dir | Holds | Used by |
|---|---|---|
| `~/.cache/specrt-ref-375` | everything: tiny and tiny-fp8/nvfp4/mtp outputs, all 0.5B, 0.6B, llama, tinymistral references, `qwen3.8-27b-slice`, `qwen3.8-27b-nvfp4-slice`, `real27b-mtp-*`, `vllm-*.json`, `vllm-mtp-*.pt`, `nvfp4-dequant.*` | the whole suite incl. real27b |
| `/mnt/data/tmp/inference-bench-498/vis498` | vLLM vision references (`vllm-image.json`, `hf-pos-video.json`, `plp-*.json`) | `vision_lm` via `SPECRT_VISION_REF_DIR` |
| `~/.cache/specrt-gguf` (or `SPECRT_GGUF_DIR`) | Qwen3-0.6B and Llama-3.2-1B GGUF files and llama.cpp logits | `gguf`, `gguf_logits`, `wgsl` |

Regenerate with the vLLM venv python. Run it from the repo root, under the 16G cap. These load full models, so for the 27B ones the one-job-at-a-time rule applies.

```bash
PY="systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0 --working-directory=$PWD $HOME/.venv-vllm/bin/python"
$PY crates/specrt/scripts/reference.py crates/specrt/specs/qwen3-0.6b.toml $SPECRT_REF_DIR
$PY crates/specrt/scripts/reference.py tiny crates/specrt/tests/fixtures/tiny-qwen3.5 $SPECRT_REF_DIR 256
$PY crates/specrt/scripts/reference.py tiny-fp8 crates/specrt/tests/fixtures/tiny-qwen3.5-fp8 $SPECRT_REF_DIR
$PY crates/specrt/scripts/reference.py tiny-nvfp4 crates/specrt/tests/fixtures/tiny-qwen3.5-nvfp4 $SPECRT_REF_DIR
$PY crates/specrt/scripts/reference.py tiny-mtp crates/specrt/tests/fixtures/tiny-qwen3.5-mtp $SPECRT_REF_DIR
```

- The first form takes a spec TOML and writes `<name>.ref.bin` and `.ref.json`. Run it for each spec in `crates/specrt/specs/` the test wants. Run on this branch it regenerated `qwen2.5-0.5b-yarn4`.
- Do not point `tiny-fp8` at the hd256 fixture dir. It writes FP8 shards into the fixture you pass.
- The 27B ones: `reference.py real-full | real-prompts | real-slice | real-mtp | nvfp4-dequant`, `vllm_logprobs.py`, `vllm_mtp_dump.py` (needs the vLLM server on :8001, read only), `vllm_vision.py`. These are slow and rarely need redoing. Copy from `~/.cache/specrt-ref-375` first.
- `yarn_reference.json` in the fixtures dir serves `yarn.rs` and is checked in.
- bf16 activation envelope: `scripts/envelope.sh` measures vLLM, llama.cpp and specrt against fp32 transformers; the table is in [perf/bf16-activation-envelope.md](perf/bf16-activation-envelope.md). The 1e-3 fp32 tests (`logits.rs`, `families.rs`, `gemma3.rs`, `sliding.rs`) set `split_activations = true` so the exact path stays proven; the serving default for the measured families is single plane.
- GGUF: `scripts/gguf_reference.c` builds the llama.cpp logit reference, `gguf_kl.py` and `dumpall.sh` compare.
- Never build, modify or rebuild vLLM, FlashInfer or anything in `~/.venv-vllm`, `~/.cache/flashinfer`, `~/.cache/vllm`. Read only.

### Test files in `crates/specrt/tests/`

"CPU" means it runs in tier 1 with no GPU. "ign" is the count of `#[ignore]` tests. Memory is GPU memory.

| File | Covers | Tier | ign | Env, data | Memory, time |
|---|---|---|---|---|---|
| `chat.rs` | chat templates | CPU | 0 | none | none, 1 s |
| `codegen.rs` | generated CUDA source text | CPU | 0 | none | none, 1 s |
| `discovery.rs` | LAN discovery (mdns) | CPU, 1 ign | 1 | none; the ignored one needs a GPU | 2 GB, 4 s |
| `spec.rs` | spec TOML parsing, `Spec::from_hf_config` for every fixture config | CPU | 0 | none | none |
| `wire.rs` | stage wire formats | CPU | 0 | none | none |
| `remote.rs` | remote stage over loopback | CPU | 0 | none | none |
| `vision.rs` | image and video preprocessing, mrope positions | CPU | 0 | none | none, 6 s |
| `fp4_gemv.rs` | NVFP4 GEMV against a CUDA reference | not ignored, but it opens the GPU, so it needs one in tier 1 | 0 | none | small, 10 s |
| `prefix.rs` | prefix cache, snapshots, block tables | CPU, 2 ign | 2 | none | none, 0.4 s |
| `yarn.rs` | YaRN rope against `yarn_reference.json` | CPU, 1 ign | 1 | none | 2 GB |
| `gguf.rs` | GGUF parse, spec from GGUF | CPU, 3 ign | 3 | GGUF files in `~/.cache/specrt-gguf` | 2 GB |
| `gguf_qwen35.rs` | qwen35 GGUF fixtures (llama.cpp converted) against the safetensors weights, spec, MTP, quant bounds, NVRTC compile | CPU, 1 ign | 1 | checked in (`fixtures/qwen35-gguf`) | none, 1 s |
| `gguf_qwen35_gpu.rs` | qwen35 GGUF logits against safetensors and llama.cpp | 3 | 3 | `SPECRT_GGUF_FILE`, `SPECRT_GGUF_REF` (`scripts/gate676.sh`) | 3 to 40 GB |
| `jit.rs` | NVRTC compile and launch, cache | 2 | 2 | none | 1 GB, 13 s |
| `fused.rs` | fused kernels vs unfused | 2 | 6 | none | 1 GB, 1 s |
| `link.rs`, `loader.rs`, `statepool.rs`, `kv_dtype.rs`, `lin_ba_rows.rs` | linking, weight loading, state pool, KV dtypes, the lin_ba rows fix (#604) | 2 | all | none | 1 to 3 GB, under 30 s each |
| `stage.rs` | pipeline stages | 2 | 6 | none | 2 GB, 7 s |
| `truncation.rs` | context truncation and shifting | 2 | 5 | none | 3 GB, 45 s |
| `linear.rs` | every linear path (bf16, fp8, nvfp4, act-quant) vs references | 2 | 20 | `SPECRT_REF_DIR` with tiny, tiny-fp8, tiny-nvfp4 | 2 GB, 30 s |
| `mamba.rs` | described mixers (#808): tiny Bamba and tiny granitemoehybrid (`scripts/tiny_granitehybrid.py`) from spec files vs transformers, the KDA recurrence vs a host reference, checkpoint state reset and restore, NVRTC compile of the module | CPU (4), 2 (4) | 4 | `SPECRT_REF_DIR` from `scripts/tiny_mamba.py` (CPU only, seconds) | 1 GB, 8 s |
| `granite_h_tiny.rs` | real Granite 4.0 H-Tiny (Mamba2 hybrid with MoE, scaled residuals) vs transformers: prefill argmax agreement and greedy decode | 2 (1) | 1 | `GRANITE_H_TINY_DIR` snapshot and `SPECRT_REF_DIR` from `scripts/granite_h_tiny_ref.py ... float32` (CPU, 28 GB; the bf16 reference is itself noisy at 40 layers) | 14 GB, 80 s |
| `fp8_act.rs`, `fp4_act.rs` | act-quant GEMM end to end | 2 | 2, 1 | `SPECRT_REF_DIR` with `tiny-fp8`, `tiny-nvfp4` | 2 GB, 15 s |
| `fp8_gemm.rs`, `fp8_batch.rs`, `fp4_batch.rs` | block-scaled GEMMs at 27B shapes, random weights | 3 | all | sm_120 GPU | 2 to 4 GB, seconds |
| `mtp.rs` | MTP drafting, acceptance, greedy identity, chi-square sampling | 2 | 15 | `SPECRT_REF_DIR` with `tiny-mtp` | 3 GB, 90 s |
| `head_quant.rs` | `head_quant` fp8 and nvfp4 lm_head | 2 | 8 | Qwen3-0.6B HF weights | 3 GB, 90 s |
| `envelope.rs` | single plane bf16 activations against vLLM's recorded KL from fp32 for llama, gemma3, granite, smollm3, phi3 (#821); the exact `split_activations = true` path at fp32 distance | CPU (2), 2 (2) | 4 | `SPECRT_ENVELOPE_DIR` (`scripts/envelope.sh` output), `SPECRT_ENVELOPE_MODELS` | 6 GB, 85 s |
| `logits.rs` | logit KL against transformers for all small models | 2 | 9 | `SPECRT_REF_DIR`; HF weights | 3 GB, 3 s |
| `generate.rs` | seeded sampling, greedy | 2 | 6 | HF weights | 2 GB, 3 s |
| `pipeline.rs` | prefill/decode pipeline, one prints a speed table | 2 | 7 | HF weights | 3 GB |
| `serving.rs` | scheduler, batching, slots, preemption | 2 | 17 | none (3 CPU tests); GPU for the rest | 3 GB, 16 s |
| `gguf_logits.rs` | GGUF Q4_K, Q8_0, BF16 logits vs llama.cpp | 2 | 7 | `SPECRT_GGUF_DIR` | 3 GB |
| `wgsl.rs` | wgpu backend, needs the `wgpu` feature | 2 | 18 | GPU adapter; `--features wgpu` | 3 GB |
| `gemm_speed.rs` | GEMM timing at 27B shapes (`SPECRT_GEMM_TOKENS`, `SPECRT_BK`, `SPECRT_FP8_BK`) | 3, timing | 4 | none | 4 GB; print only |
| `flash_speed.rs` | attention timing (`SPECRT_FLASH_*`) | 3, timing | 1 | none | 2 GB |
| `gdn_speed.rs` | delta-net timing (`SPECRT_GDN_TOKENS`) | 3, timing | 1 | none | 2 GB |
| `kir_speed.rs` | kir kernel timing | 3, timing | 1 | none | 1 GB |
| `tiny27b.rs` | tiny Qwen3.5 27B-shaped fixture vs transformers: weight-only and act-quant logits KL, greedy identity, MTP drafts, speculative greedy, vision embeddings and mrope logits | CPU (1 spec test), 2 | 6 | none, references are checked in | 1 GB, 100 s |
| `real27b.rs` | 12 real-checkpoint gates: greedy and logits vs transformers and vLLM, MTP acceptance, per-layer trace, needle at 500k and 1M | 5 | 12 | `SPECRT_REF_DIR=~/.cache/specrt-ref-375`; `SPECRT_27B_SPEC`, `SPECRT_27B_NVFP4_SPEC`, `SPECRT_27B_NEEDLE_MODEL` | 25 to 96 GB; 1 to 30 min each |
| `gemma3.rs`, `gemma3_tiny_vision.rs`, `gemma3_vision.rs`, `gemma3_27b.rs` | Gemma 3 text, SigLIP, pan and scan, 27B FP8 vs vLLM (see `specrt-gemma3.md`) | CPU (4), 2, 5 | 12 | `SPECRT_REF_DIR`, `GEMMA3_1B_DIR`, `GEMMA3_4B_DIR`, `GEMMA3_27B_DIR` | 3 to 36 GB |
| `vision_encoder.rs`, `vision_lm.rs` | vision tower and image/video chat vs vLLM | 5 | 4, 10 | `SPECRT_VISION_REF_DIR`, `SPECRT_VISION_PRECISION` | 28 to 36 GB |
| `nvfp4.rs` | real NVFP4 dequant vs `nvfp4-dequant` reference | 5 | 1 | the NVFP4 checkpoint | 26 GB |

Every `#[ignore = "..."]` carries its reason in the attribute, so `cargo test -- --list --ignored` plus a grep of the file tells you what one needs.

## Server, core, cli, ui

```bash
cargo test -p inference-bench-server      # api, dialects, tools, logging, app_mode (CPU); serving.rs needs a GPU
cargo test -p inference-bench-core        # live_hf.rs is ignored (network)
cargo test -p inference-bench-cli
cargo test -p inference-bench-ui
```

GPU server tests: `cargo test -p inference-bench-server --release --test serving -- --include-ignored`, under the same flock and systemd-run wrapper.

## Tier 4: Qwen3.5-4B NVFP4

`crates/specrt/scripts/bench4b.sh` quantises `Qwen/Qwen3.5-4B-Base` to the 27B's layout on first use (`quantize4b.py`, into `~/.cache/specrt-models/Qwen3.5-4B-NVFP4`), then runs the bench binary under `flock /tmp/gpu27.lock`: single stream at 128, 1024 and 4096 tokens for the weight-only and act-quant specs, batch 4, 8 and 16, and MTP. Needs about 12 GB free on the card. Baseline (RTX PRO 6000 Max-Q, median of 3, commit 1655db2):

| | weight-only spec | act-quant spec (as served) |
|---|---|---|
| prefill 128 / 1,024 / 4,096 tok/s | 5,709 / 9,556 / 9,397 | 11,947 / 23,086 / 21,470 |
| decode, single, same prompts, tok/s | 255 / 253 / 241 | 253 / 250 / 240 |
| decode aggregate, batch 4 / 8 / 16 tok/s | | 870 / 1,586 / 2,509 |
| MTP k=2 / without, tok/s | | 356 (acceptance length 2.27) / 231 |

Predicts reliably: act-quant against weight-only prefill (#612), whether a GEMV or head change helps bandwidth-bound decode, batch scaling shape (#604), MTP identity. Still needs the 27B: kernel dispatch at a row count (#604 b32 GEMV against tile GEMM), long-context decode attention and split counts (#617), memory and KV pool limits, vision timing, and any absolute number. Detail: `docs/specrt-test-fixtures.md`.

## Tier 5: the real 27B

Check it compiles without running it:

```bash
systemd-run --user --wait --collect -P -q -p MemoryMax=10G --working-directory=$PWD \
  -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS \
  cargo test -p inference-bench-specrt --release \
  --test real27b --test vision_lm --test vision_encoder --test nvfp4 --no-run
```

This builds on this branch. To run it, one test at a time:

```bash
flock /tmp/gpu27.lock systemd-run --user --wait --collect -P -q \
  -p MemoryMax=16G -p MemorySwapMax=0 --working-directory=$PWD \
  -E CARGO_TARGET_DIR -E SPECRT_REF_DIR=$HOME/.cache/specrt-ref-375 \
  cargo test -p inference-bench-specrt --release --test real27b -- --include-ignored --test-threads=1 real_27b_nvfp4_agrees_with_vllm_greedy_and_logprobs
```

- The test names start `real_27b_`. A skip filter of `real27b` matches nothing, use `--skip real_27b` to leave them out.
- With the user's vLLM holding about 51 GB, some tests OOM. Rerun those alone when the card is quiet.
- Scripts, all taking `<out> <bin dir>` or a spec and reading `nvidia-smi` first: `gate27b.sh` (nvfp4 exact gate), `kvgate27b.sh <out> <bin> [kl|needle|speed|mtp|all]` (KV cache gates), `head27b-gates.sh`, `head27b.sh`, `spec27b.sh` (speculative), `bench27b.sh`, `bench27b-nvfp4.sh`, `seqs598.sh`, `serve27b.sh` (8 slots, 3 GB KV, fits beside vLLM), `vllm27b-nsys.sh` (vLLM side, read only).
- `quiet_window.sh <command>` takes the lock, stops `local-llm.service`, runs the command, restarts the service and proves it with a chat request. It is for the supervisor's batched window only (under 30 min, max 60). Do not run it yourself unless your brief says so, and never stop the user's app by hand.

## GPU etiquette

- The user's app on :8001 (local-llm, vLLM) may hold about 51 GB of the card, and the box has two cards. Check first: `nvidia-smi --query-gpu=memory.used,memory.free --format=csv`.
- Tier 2 and 3 fit beside it. Do not wait for it to go away.
- Never stop, restart or signal the user's app, local-llm.service or anything on the Spark without being told to. The Spark rule: `free -g` first, stay under 20 GB, lock file, per-issue target dir.
- If a tier 5 run needs memory you do not have, do not queue behind the lock for hours. Run tiers 1 to 3, then report.
- Evidence enough to hand back: tier 1 green, every tier 2 or 3 file your change touches green with the `test result:` lines pasted, and the exact tier 5 command (test names, ref dir, spec) for the supervisor's window. A change specific to the 27B checkpoint still has a tier 2 proof if the code path exists in the tiny Qwen3.5 fixtures.
- Config rule: behaviour comes from the spec TOML or server config. There are no env vars for behaviour. The variables the tests read (`SPECRT_REF_DIR`, `SPECRT_GGUF_*`, `SPECRT_VISION_*`, `SPECRT_27B_*`, and the `SPECRT_GEMM_*` and `SPECRT_FLASH_*` lists in `gemm_speed.rs` and `flash_speed.rs`) are test inputs and parameters, not engine knobs: nothing under `src/` reads them.

## Measuring speed

- Paired A/B only. Never compare a number from today with one from last week, since the card clock, the user's app and other jobs move it. Run base and branch back to back, interleaved, 3 rounds, on the same card, under the lock.
- Build each side with its own `CARGO_TARGET_DIR`, and look in the PTX (`[runtime] cache_dir = "<tmpdir>"` in the spec) for the JIT source you changed before trusting a number.
- `scripts/ab517.sh`: `BASE=<prefixbench> BRANCH=<prefixbench> ./scripts/ab517.sh`, long-context prefill A/B (`LENGTHS`, `ROUNDS`, `KV_GB`, `BASE_SPEC`, `BRANCH_SPEC`).
- `scripts/prefill555.sh`: `BIN=<prefixbench> [ALT_BIN=...] ./scripts/prefill555.sh`, prefill tables plus nsys per length. Set `NSYS=0` to skip the profile.
- `scripts/batch8_604.sh`: serves the NVFP4 27B with 8 slots, runs `loadtest` with 8 clients and takes an nsys of the round. Refuses to start under `NEED_MIB` of free memory (36000).
- `scripts/bench601.sh`, `flashprof.sh`, `vsllama.sh <model.gguf> <bench>` (against llama.cpp, set `LLAMA_BENCH`), `spec27b.sh` (MTP speed).
- nsys helpers: `nsys_top.py <csv|sqlite>` top kernels, `nsys_steps.py <sqlite> <steps>` per-step sums and gaps, `nsys_grid.py`, `prefill_timeline.py`.
- `scripts/quiet_window.sh` exists for numbers that need the card alone. Mark such numbers "quiet GPU" in docs. Numbers taken beside the user's app say "shared".
- Judge by our own A/B and nsys, not by user-facing comparisons.

## Known flaky tests

| Test | Issue | Note |
|---|---|---|
| Real 27B tests with vLLM resident | none | OOM, not a bug. Rerun alone when quiet. |

## Perf passes

Speed per architecture, with nsys tables and llama.cpp and vLLM side by side, is `docs/perf-pass.md` and `crates/specrt/scripts/perfpass.sh`. The scoreboard is `docs/perf/scoreboard.md`.
