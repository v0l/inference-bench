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

Without `SPECRT_REF_DIR` the tiny-fixture tests in `linear` (the ten `moe_*` ones among them: six fail, with the missing reference files as the error) look in `crates/specrt/out` and fail on master too.

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
| `~/.local/share/inference-bench/refs/vision` (never `/tmp`; on the Spark the gate scripts read the same path under `~`) | vLLM vision references (`vllm-*.json`, `hf-pos-*.json`, `plp-*.json`), made by `vllm_vision.py` (vLLM on GPU1 via `docs/qwen27-run/launch.sh kvauto` under `gpu-run 1`, then `VLLM_URL=http://127.0.0.1:8011 VLLM_MODEL=orcarouter/Qwen3.8-27B-Uncensored-NVFP4`) and `CUDA_VISIBLE_DEVICES= vision_positions.py`. vLLM 0.34 with transformers 5.23 needs `PYTHONPATH=crates/specrt/scripts/vllm_compat` on the server (its `sitecustomize.py`) drops `do_sample_frames` from `VideoMetadata` and ignores the text-only video-pad check | `vision_lm` via `SPECRT_VISION_REF_DIR` |
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
| `plan_check.rs` | kir compiler slice 1 (#1053): `signature::validate_plan` on every fixture and spec plan (tensor core, legacy, TP, MTP, every phase); broken plans fail naming the node (read before write, shape, plane form); `kernels --plan` bytes and launches on tiny-mistral against hand counts | CPU | 0 | none | none, 5 s |
| `fusion_plan.rs` | kir compiler slice 2 (#1054): the fusion dry run (`fusion::fusion_report`) on qwen3-0.6b and tiny27b picks the hand fusions (norm into GEMV, gate_up act, residual epilogue, qk norm rope KV write, allreduce plus norm, head norm) and rejects `add_rmsnorm`, attention reduce into split and the wgpu GEMV norm fold; smem and register budgets with small `Costs`; slice 3 (#1055): every GEMM of three specs lifts to `kir::heavy::Heavy` and lowers to the same kernel, and a 40 register target fails naming the pass; slice 4 (#1056): the decode norm GEMVs (`qkv_proj_norm`, `gate_up_act`, `lm_head_norm`, `lin_in_norm`, `lin_ba_norm`, `gate_up_act_f4`, `lm_head_norm_f4`) come out of the `prologue_attach` pass and a small smem target fails naming it; slice 6 (#1058): `Rule::Fill` puts the branch `clear` with the residual linear on tiny-olmo2 and tiny-granite | CPU | 0 | none | none, 1 s |
| `compile_plan.rs` | kir compiler slice 5 (#1057): the pass runner (`compile::compile`) gives the tensor parallel plans the old hand rewrite produced for every spec, leaves single rank plans alone, and its one row decode plan (`compile::single`) launches the kernels the hand `step_traced` did, all of them in the module; slice 6 (#1058): with `[runtime] store_epilogue` every branch spec launches `o_proj_store` and `down_proj_store` and no `clear_branch`, and specs without a branch keep the same plan; slice 7 (#1059): every single step spec launches fewer kernels than its unfused plan and exactly what the fusion engine predicts with the kernel library (table in the failure message), and the three spec-only mixes (`tiny-sandwich-scaled`, `tiny-sandwich-pre-qknorm`, `tiny-scaled-qknorm`) get prologue, epilogue and, with `store_epilogue`, fill fusions from the passes | CPU | 0 | none | none, 1 s |
| `spec_mix.rs` | kir compiler slice 7 (#1059): the three spec-only mix fixtures (spec TOMLs in `tests/fixtures`, weights from `scripts/tiny_families.py`) parse to existing parameters, a scaled sandwich folds `residual_scale` into its branch norm weights at load; the ignored GPU test checks prompt, decode and greedy against the torch reference (2e-3) and bit identity with `store_epilogue` on and off | CPU (2), 2 (1) | 1 | `SPECRT_REF_DIR` from `scripts/tiny_families.py <dir> tiny-sandwich-scaled tiny-sandwich-pre-qknorm tiny-scaled-qknorm` (CPU, seconds; `check-mix` first compares the hand reference with transformers on tiny-olmo2 and tiny-granite) | 1 GB, 10 s |
| `prologue_bits.rs` | kir compiler slice 4 (#1056): each fused norm GEMV against the unfused chain (`rmsnorm`, then a plain GEMV, then `swiglu`) on real weights, bit for bit, on tiny-qwen3.5 (bf16, fp8, nvfp4, nvfp4 head), tiny27b, the tiny Q8_0 GGUF and the Qwen3.5-4B NVFP4 (skipped when `~/.cache/specrt-models/Qwen3.5-4B-NVFP4` or `SPECRT_4B_DIR` is missing). `PROLOGUE_COMPILE_ONLY=1` builds the PTX into `PROLOGUE_PTX` without a GPU, then run the binary under `gpu-run 1` | 2 | 1 | none | 3 GB, 50 s |
| `source_dump.rs` | writes every fixture and spec module source (sm_120, sm_80) to `SOURCE_DUMP`, and with `PTX_DUMP=1` the NVRTC PTX of the 27B, 4B and 0.6B modules, for `diff -r` against another commit. Pin `LD_LIBRARY_PATH` to one CUDA: NVRTC 13.2 and 13.5 give different PTX for the same source | CPU | 1 | none | none, 2 min |
| `wire.rs` | stage wire formats | CPU | 0 | none | none |
| `remote.rs` | remote stage over loopback | CPU | 0 | none | none |
| `vision.rs` | image and video preprocessing, mrope positions | CPU | 0 | none | none, 6 s |
| `fp4_gemv.rs` | NVFP4 GEMV against a CUDA reference | not ignored, but it opens the GPU, so it needs one in tier 1 | 0 | none | small, 10 s |
| `quant_search.rs` | load time scale search (#1417) for nvfp4_g16 and fp8_channel, bit for bit against `scripts/quant_search_reference.py` | CPU | 0 | none | none, 0.1 s |
| `quant_search_gpu.rs` | the kir scale search kernels (`kir::weight_quant::ScaleSearch`) bit for bit against the CPU search on the numpy case, mixed and zero rows and the tiny fixture head | 2, 3 ign, run with `-- --ignored` under `~/.agents/bin/gpu-run 1 --min-free 2000 --` | 3 | none | small, 1 s |
| `quantize_load.rs` | `[runtime] quantize` at load (#1417): each linear and expert stack goes through the scale search bit for bit, the keep list and sensitive tensors stay bf16, fp8_channel row scales, off leaves the spec and every matrix as it was, stages count the keep list in whole model layers; a per layer `quant_plan` gives each layer its own format and rounding, plans the `_f8w`/`_f8b` and `_f4w`/`_f4b` kernels its layers need, and refuses a wrong layer count, a non bf16 checkpoint or fp8 on MoE | CPU | 0 | none | none, 0.1 s |
| `quantize_gpu.rs` | the tiny dense and MoE fixtures loaded with `quantize` nvfp4_g16 and fp8_channel: fewer device bytes, mean logits KL under 0.05 (nvfp4) and 0.01 (fp8) against bf16, batch decode agrees with single | 2, 3 ign, `-- --ignored --test-threads=1` under `gpu-run 1 --min-free 2000` | 3 | none | 1 GB, 5 s |
| `quantize_qwen3.rs` | real Qwen3-0.6B loaded with `quantize` fp8_channel and nvfp4_g16 on the shipped prompt set `quantcheck/prompts.toml`: mean KL, max KL and top-1 per prompt against bf16, held to `FP8_CHANNEL_MEAN_KL_BOUND`, `NVFP4_G16_MEAN_KL_BOUND` and the top-1 floors in `quantize.rs`; prints load time and bytes read. `a_quantcheck_recipe_beats_blanket_nvfp4_*` runs quantcheck at KL budget 0.1 and holds the recipe to at least 3 points more top-1 than blanket nvfp4 within 5% of its size, a fresh load of the recipe reproducing the check, and batch decode equal to single; `quantised_prefill_and_decode_speed_*` prints speed per format; `quantize_cache_load_time_*` prints cold and warm load time, source bytes read and cache size with and without `quantize_cache`, and holds the logits equal | 2, 4 ign, `-- --ignored --nocapture --test-threads=1` under `gpu-run 1 --min-free 2000` | 3 | Qwen/Qwen3-0.6B in the HF cache | 4 GB, 2.5 min |
| `quantcheck.rs` | quantcheck (#1417): per layer options keep the rounding (search or plain) with the better measured logits, greedy moves the most damaging layer up one format until the KL or memory budget holds, the recipe TOML round trips into a `quant_plan` and refuses another recipe version; `quantize = "recipe"` reads the recipe file (relative paths beside the spec) into the plan, refuses another model, layer count or format version (also through `Runtime::load`, before any GPU work) and loads with a warning on another engine commit or prompt set | CPU | 0 | none | none, 0.1 s |
| `quantize_cache.rs` | `quantize_cache` (#1417): the first load writes the quantised matrices under `models--*/specrt-quant/<key>/` (nothing beside the snapshot files), the second reads them back bit for bit on the tiny dense and MoE fixtures and converts nothing; the key changes with the plan, keep list and source weights but not serving options; outside the hub it goes under `cache_dir`, and off, bf16 or a stage writes nothing. The ignored GPU test checks a cached start gives the logits of a converting start and reads fewer source bytes | 3, 1 ign, `-- --ignored` under `gpu-run 1 --min-free 2000` | 1 | none | 1 GB, 10 s |
| `quantcheck_gpu.rs` | swapping layers in place (`Runtime::requantize_layer`) gives the logits and bytes of a fresh load of that plan, bit for bit, on the tiny dense and MoE fixtures; quantcheck on the tiny fixture meets its budget when checked whole and a fresh recipe load reproduces the check | 2, 2 ign, `-- --ignored --test-threads=1` under `gpu-run 1 --min-free 2000` | 2 | none | 1 GB, 5 s |
| `prefix.rs` | prefix cache, snapshots, block tables | CPU, 2 ign | 2 | none | none, 0.4 s |
| `yarn.rs` | YaRN rope against `yarn_reference.json` | CPU, 1 ign | 1 | none | 2 GB |
| `gguf.rs` | GGUF parse, spec from GGUF | CPU, 3 ign | 3 | GGUF files in `~/.cache/specrt-gguf` | 2 GB |
| `qwen4_exp.rs` | Qwen3.8-Flash-Next (`family-qwen4_exp.toml`, #1280) on `tiny-qwen4-exp` (3 delta-net layers, 1 QSA indexed attention layer, 8 experts top 2 plus a shared expert, 4 stream gated residual, PLE on layer 2): CPU checks that the family serves the tiny and the real configs (FP8 and NVFP4 copies too, the MTP layer refused by name), the `[prelude]` and `[gated_residual]` validation, plan order, module kernels and loader names; on GPU1 the hidden streams of layers 0, 1 and 3 (all 30 rows, the indexer selects from row 19), prefill logits and 4 greedy decode steps against transformers at 1e-3 (measured 2.5e-4 and 2.0e-4), the in engine PLE against #1279's capture, two served slots side by side against greedy and against the same prompt alone, and a state snapshot carried to another slot. Regenerate with `~/.venv-vllm/bin/python crates/specrt/scripts/tiny_qwen4_exp.py` | CPU (12), 2 (5) | 5 | checked in (`fixtures/tiny-qwen4-exp/reference`) | under 1 GB, 2 s |
| `qwen4_exp_checkpoint.rs` | Qwen3.8-Flash-Next real checkpoint tensors (#1349), from `fixtures/qwen4-exp-real/headers.json` (the three repos' shard headers): the 128 n-gram table shards join through the loader name map into the table the family reads, the folded hash constants equal the real `ngram_heads_offsets`, `ngram_heads_vocab_sizes` and `layer_multipliers`, an fp8 sharded tiny table loads in its stored bytes, fused bf16 experts with no `.weight` suffix load like per expert names; on GPU1 the fp8 and bf16 gathers against the f32 one, and the engine on bf16 shards, fp8 shards and fused experts against the original fixture and the decoded table (all bit exact) | CPU (6), 2 (2) | 2 | checked in | under 1 GB, 7 s |
| `qwen4_exp_mtp.rs` | Qwen4Exp MTP layer (#1350) on `tiny-qwen4-exp-mtp` (the `tiny-qwen4-exp` shard plus `mtp.safetensors`, shared through its index): CPU checks of the streamed `[mtp.layout]` (fc_hidden, mixer), the draft plan (expand, fc_hidden into the four streams, gated attention and MoE sites, draft mixer before the head), the loader and the kernels; on GPU the chained draft logits against the reference from `scripts/tiny_qwen4_exp.py mtp` at 1e-3 (k = 2 and 3, drafts equal), and speculative greedy equal to plain greedy for k = 1 to 3, batch 1 and 2, with MTP, oracle and rotating wrong drafts | 2 | 6 | none | 1 GB, 20 s |
| `qwen4_exp_ple.rs` | Qwen4Exp n-gram hash and PLE layer (#1279): the PLE params from the tiny and real configs, every `specs/ple_*` description builds, the `ngram_history` and `ple_conv` state sizes, the host reference against the transformers PLE capture; on GPU the kir kernels against it at 1e-3 for prefill and decode steps 1 to 4, chunked prefill with slot reuse, and two sequences side by side (ids bit exact). Hash kernels also covered by `crates/kir/tests/ple.rs` (CPU). Regenerate with `~/.venv-vllm/bin/python crates/specrt/scripts/tiny_qwen4_exp.py ple` | CPU (4), 2 (3) | 3 | checked in (`fixtures/tiny-qwen4-exp/reference`) | under 100 MB, 2 s |
| `glm5_next.rs` | GLM-5 stage 2 (#1189): the `tiny-glm5-next` fixture (KDA, KDA, KDA, DSA; 8 experts top 2 plus a shared expert; dense layer 0; mHC 4 streams; key pool 4; one MTP layer) describes with hyper-connections, no rope, the pooled indexer, `kda` and `dsa` kinds, and is refused by serving until #1192; its checkpoint keeps the real tensor names and F32 tensors; its references (prefill logits, 4 decode steps, mHC streams after layers 0, 1, 3, MTP draft and chained draft) load and are finite. Regenerate with `~/.venv-vllm/bin/python crates/specrt/scripts/tiny_glm5_next.py all` (CPU torch, transformers 5.23.0). #1191: a host f32 model of the fixture (`tests/glm5_host`) matches prefill, 4 decode steps and the saved streams within 3.8e-4 and goes red on a wrong Sinkhorn count or a dropped tail pool; the engine family (`specs/family-glm5_next.toml` since #1192, the library family) plans mHC, no rope, KDA and the pooled indexer and its module compiles through NVRTC (ignored); the ignored GPU test runs the same comparison on the engine. #1192: `glm5_mixed.rs` makes a MIXED_PRECISION copy of the fixture (MXFP8, FP8_PB_WO, NVFP4 experts) and a bf16 twin of the decoded values; the loader gives identical matrices on CPU and the engine identical logits on GPU (ignored) | CPU (9), NVRTC (1), 2 (1) | 2 | checked in (`fixtures/tiny-glm5-next/reference`) | GPU: under 1 GB, seconds |
| `diffusion.rs` | text to image and video analysis (#1281, `docs/specrt-diffusion.md`): the `tiny-flux2-klein` (Qwen3 text encoder, FLUX.2 MMDiT with one double and one single block, `AutoencoderKLFlux2` with 2 levels, 32 by 32 image, 16 by 16 latent) and `tiny-wan21` (UMT5 encoder, Wan space-time DiT, causal 3D `AutoencoderKLWan`, 5 frames) diffusers folders keep the real component classes, configs and tensor names; the Qwen3 text encoder is already a supported spec; the references (text embeddings, noise, velocity and latent after each of 3 steps, the flow Euler update between them, guided runs, decoded image and frames) load and are finite; every component config is still refused. Regenerate with `/tmp/diffusers-1281/bin/python crates/specrt/scripts/tiny_diffusion.py` (CPU torch, diffusers 0.43.0, transformers 5.24.0, in a private venv: `uv venv /tmp/diffusers-1281 -p 3.13 && VIRTUAL_ENV=/tmp/diffusers-1281 uv pip install --index-url https://download.pytorch.org/whl/cpu torch && VIRTUAL_ENV=/tmp/diffusers-1281 uv pip install diffusers transformers safetensors accelerate`) | CPU | 0 | checked in (`fixtures/tiny-flux2-klein/reference`, `fixtures/tiny-wan21/reference`) | none, 1 s |
| `haru.rs` | Haru-4B full duplex speech model (#1344, `docs/specrt-haru.md`): the real checkpoint's header (`fixtures/haru/haru-4b.header.json`) has the five parts at their sizes and the Qwen3.5 text layout under `lm.`; the `lm` sub-config is a spec the engine accepts while `model_type = "haru"` is refused; the tiny fixture (`fixtures/haru/tiny`, every real tensor kind) and its references from the reference package; in plain Rust from the fixture weights: the three stream input sum and the user latent noise (1e-5), the tied text logits and greedy tokens (1e-5), the one step MeanFlow head at guidance 1, 1.5, 2 and two steps (1e-4); the codec streaming state sizes from the config; the codec graph (`graphs/codec.toml`) plans every real codec tensor and shape, the parameter counts and the state (159,622 encoder, 147,904 decoder values per session in the reference form, 139,200 in ours); the host codec (`specrt::codec::host`) reproduces `codec.clip_mean`, `session.user_mean`, `codec.clip_audio`, `session.audio` and `codec.silence` (1e-4) and frame by frame equals whole clip (1e-5); through the engine (#1427): `specs/family-haru.toml` reads the nested `lm` section (`text_config = "lm"`), renames the `lm.` root to `model.` through `[tensors]`, declares the two latent streams and masks in `[inputs]`, and the input embedding is six `[[inputs.node]]` plan ops (`latent_proj_in`, `latent_proj_norm`, `latent_proj_out` kir descriptions) after the token gather; every input weight loads bit for bit; the family is refused with the missing slices named until #1430; folder under 3 MB. Regenerate with `crates/specrt/scripts/tiny_haru.py fixture --haru <Haru checkout>/haru --commit 7fe6b766` (CPU torch 2.15, transformers 5.24, private venv, commands in the doc) | CPU | 0 | checked in (`fixtures/haru/tiny/reference`) | none, 0.02 s |
| `speech.rs` | speech to text and text to speech analysis (#1474, `docs/specrt-speech.md`): the Whisper configs (tiny fixture, tiny.en, base, large-v3-turbo) are refused by every spec path; the gap list's "exists" rows still have their engine source and the 9 missing pieces are printed (`-- --nocapture`); the tiny Whisper fixture (`fixtures/speech/tiny-whisper`) has the real tensor kinds and tied head; a plain f64 DFT log-mel matches transformers within 8.3e-6; suppress lists on the recorded logits give the greedy and `generate` tokens and the detected language; Phonon's parameter split and layout; VibeVoice-Realtime's acoustic decoder equals Haru's codec decoder in all 276 shapes; folder under 2 MB. #1477: the log-mel op and conv stem from `graphs/speech.toml` (front end parameters read from `preprocessor_config.json`): mel bank equal to transformers' filters, host `mel.full` and `mel.short` within 1e-4 absolute (1.1e-5) and `encoder.stem` within 1e-4 relative (9.0e-7), typed signatures and the GELU fused into both padded convs (11 launches unfused, 9 fused), a padded line conv keeps causal convs at `PAD -1`; ignored: every speech kernel through nvrtc, and on GPU1 the same references (mel 1.08e-5 abs, stem 2.4e-6 rel). Regenerate with `~/.venv-vllm/bin/python crates/specrt/scripts/tiny_whisper.py all --phonon ~/.local/share/agentbench/models/phonon` (CPU, transformers 5.23, 4 s, bit identical weights) | CPU (14), nvrtc (1), GPU1 (1): `~/.agents/bin/gpu-run 1 --min-free 2000 -- <speech binary> --ignored` | 2 | checked in (`fixtures/speech/tiny-whisper/reference`) | none, 0.2 s |
| `haru_gpu.rs` | Haru codec on CUDA (#1429): every line conv kernel of the real config compiles with nvrtc (ignored, no GPU); on the GPU the tiny fixture's `codec.clip_mean`, `session.user_mean`, `codec.clip_audio`, `session.audio` and `codec.silence` within 1e-4 relative, frame by frame equal to whole clip within 1e-5, and two batched sessions equal to each run alone | GPU1, `~/.agents/bin/gpu-run 1 --min-free 2000 -- <haru_gpu binary> --ignored` | under 1 GB | checked in | none, 0.3 s |
| `haru_head.rs` | the MeanFlow audio head as a plan (#1428): `[head]` in `family-haru.toml` binds the real config to `graphs/meanflow.toml` (RMS `Modulate` with the layer norm weight, `Glu`, `GatedAdd`, `TimestepEmbed`, the update `z + h_step * u` and the unscale as `Map` expressions) and the plan reads exactly the 32 `head.` tensors of the real header; the hoist (`FlowHead::levels`) puts the four scalar embeddings and their sum once per session and guidance change (57.8 MB, 17 launches), `cond_proj` per frame, and with two steps moves `t` into the step loop; the fusion pass puts the gated residual into the down projection rows GEMV (35 to 27 launches per frame) at 1 and 8 sessions; the host runner gives `session.latent`, `head.guidance1.0`, `head.guidance2.0`, `head.steps2` within 1e-4 relative (measured 4e-7 to 8e-7), frame by frame equal to batched; a guidance change reruns only the hoisted ops; frame noise is `sampler::noise` | CPU | 0 | checked in | none, 0.05 s |
| `haru_head_gpu.rs` | the same head on CUDA: every head kernel of the real config compiles with nvrtc (ignored, no GPU); on the GPU the four references within 1e-4 relative (measured 7.6e-6 to 8.0e-6) fused and unfused, 7 sessions batched equal to one at a time, 21 launches per frame fused (23 unfused) on the tiny head | GPU1, `~/.agents/bin/gpu-run 1 --min-free 2000 -- <haru_head_gpu binary> --ignored` | under 1 GB | checked in | none, 1 s |
| `haru_duplex_gpu.rs` | Haru duplex sessions (#1430) on the tiny fixture: the six frame greedy session gives every `session.token` eager and through the frame graph; with the reference feedback each frame the user latent (1.3e-6), hidden (1.7e-4), latent (1.8e-4) and audio (2.4e-3) hold the backbone bound and the audio equals a fresh decode of the emitted latents; free running drift stays at the backbone floor; two sessions batched equal one alone; snapshot and restore through the session store gives the same next frames; relay, say queue order and interrupt; `max_frames` and `window_frames`; a voice clip ending mid frame encodes like `codec.clip_mean` (1.0e-6); the session seed changes sampled text. Numbers in `docs/specrt-haru.md` section 13 | GPU1, `~/.agents/bin/gpu-run 1 --min-free 2000 -- <haru_duplex_gpu binary> --ignored --test-threads=1` | under 1 GB | checked in | none, 4 s |
| `haru_real_gpu.rs` | the real Haru-4B against the reference package (#1489, `docs/haru-validation.md`): teacher forced from saved traces (greedy, sampled, speak first: top 1 1.0, KL under 3e-9, cosine 0.999996 or better at layers 0, 8, 16, 24, 31 and the final hidden, latent mean under 4e-5) and greedy free running token for token on the user clip and on silence with and without a voice. Traces from `scripts/haru_trace.py` (`HARU_TRACE_DIR`, default `/tmp/haru1489`) | GPU1, `~/.agents/bin/gpu-run 1 --min-free 20000 -- <haru_real_gpu binary> --ignored --test-threads=1` | 12 GB | not in git (437 MB, `haru_trace.py`) | none, 50 s |
| `haru_backbone_gpu.rs` | the tiny Haru backbone and three stream input through the engine (#1427), refusal lifted in the test: prompt prefill with per position masks (embeds 2.0e-7 relative under `EMBED_BOUND` 1e-5; layers 2.5e-5, 3.5e-5, 1.4e-4, 1.8e-4; final hidden 2.1e-4; top 1 identical at all 15 positions), the prompt fed one embedding row at a time through scheduler decode (2.7e-4), and the six session frames replayed through batched decode from the recorded tokens and latents (embeds 1.4e-7, hidden worst frame 1.6e-4, every greedy token equal). The backbone bound is `BACKBONE_BF16_TOLERANCE` = 3e-4 relative, the engine's split bf16 activations: the control with the audio streams off against `text.layers` and `text.hidden` (from `tiny_haru.py`) gives the same scale (layers 3.1e-5, 4.2e-5, 1.4e-4, 2.1e-4, hidden 2.0e-4). Wrong input paths fail far past it: the streams swapped give prompt.hidden 1.25 and session.hidden 1.75, the model mask dropped gives 1.26 (floor 1e-2) | GPU1, `~/.agents/bin/gpu-run 1 --min-free 2000 -- <haru_backbone_gpu binary> --ignored --test-threads=1` | under 1 GB | checked in | none, 7 s |
| `vae.rs` | VAE decoder (#1283): `graphs/vae.toml` expands both tiny VAE configs into plans whose tensors exist; the host conv matches numpy on every VAE conv shape (`fixtures/conv-cases`, regenerate with `python3 crates/specrt/scripts/conv_reference.py`, numpy only); the host decode matches `vae.decoded` (2.8e-6 and 2.2e-7 max abs); the fusion pass rediscovers norm apply and conv prologues, group stats in conv epilogues, residual adds and upsample folds (56 to 21 launches on the KL VAE); fused and unfused schedules decode bit identically; arena offsets never overlap; the tile size is the largest whose arena fits; tiled and untiled agree within the seam tolerance | CPU | 0 | checked in | none, 1 s |
| `vae_gpu.rs` | the same VAE on the GPU: every VAE kernel compiles through NVRTC (no GPU), every conv shape against numpy, both tiny decodes against diffusers (1e-4 max abs, PSNR over 60 dB, fused against unfused 1e-5 relative), tiled GPU against tiled host | NVRTC (1), 2 (3) | 3 | checked in | under 1 GB, seconds |
| `sampler.rs` | prompt to latent on the host (#1284): `graphs/pipeline.toml` binds the tiny FLUX.2 klein configs; the schedule (linspace, exponential shift, empirical mu) gives the diffusers sigmas to 1e-6; the flow Euler rule from the graph reproduces the recorded latents to 1e-5 and a wrong rule fails; the CFG combine; the splitmix64 Box-Muller noise equals the reference script's noise; the prompt is chat templated, truncated and padded like transformers (`fixtures/prompt-ids/qwen3.json`, regenerate with `HF_HUB_OFFLINE=1 /tmp/diffusers-1281/bin/python crates/specrt/scripts/prompt_ids.py`, needs `Qwen/Qwen3-0.6B` in the hub cache, else it skips) | CPU | 0 | checked in | none, 1 s |
| `text_encoder_gpu.rs` | the Qwen3 text encoder of `tiny-flux2-klein` on the engine: `Runtime::hidden_states` with pad keys masked gives `text.hidden`, `text.embeds` and `negative.embeds` within 1e-4 (measured 7e-6 and 9e-6); unmasked pad keys leave the prompt rows equal and move the padded rows | 2 | 2 | checked in | under 1 GB, 1 s |
| `sampler_gpu.rs` | the whole loop on the engine: 3 steps from the seed, unguided and guided (CFG 4, both prompts as 2 sequences of one forward), latents within 1e-4 of the bf16 weight diffusers run (and 2e-2 of the f32 run) and the decoded image within 1e-3 and over 60 dB PSNR; the Euler epilogue on `proj_out` is bit identical to the unfused step; batched sequences equal single forwards; `offload_text_encoder` and `offload_vae` decide what is resident and change no pixel; the arena stays under 1 MB; the admission fallbacks (guidance as two passes in one arena, half the bytes, within 1e-3; a tiled VAE with a smaller arena, over 20 dB) and a cancel after the first step | 2 | 6 | checked in | under 1 GB, 3 s |
| server `images.rs` | `POST /v1/images/generations` on `FakeImage` (the tiny klein family's real `MediaInfo` and checks, seeded pixels): size, same seed same bytes, n=2 differs, invalid size 400 before any step, a full queue 429 with `Retry-After` and the image metrics, SSE progress and previews, a dropped stream cancels between steps, `url` files behind auth, `/v1/models` and `/v1/capabilities` mark `type`; a text model refuses an image job and an image model refuses chat | CPU | 0 | none | 2 s |
| server `images_gpu.rs` | the tiny klein pipeline served end to end by the real engine through `NativeFactory`: the PNG is within one 8 bit level of the bf16 weight diffusers image, repeatable, n=2, 48x32, a 400, guided. `~/.agents/bin/gpu-run 1 -- <images_gpu binary> --ignored` | 2 | 1 | checked in | under 1 GB, 4 s |
| ui `tests_images.rs` | the image load page (size, steps, guidance, text encoder outside Advanced, components), the Generate page (result grid, save, copy, reuse, request row with steps per second), stop cancels, phone width keeps keys on screen; `SHOTS=dir` renders both pages at 1280x800 and 390x844 (`docs/shots/images/`); `the_real_engine_...` loads the tiny pipeline through the page on GPU1 and checks `nvidia-smi` | CPU (one test GPU1) | 0 | none | 1 s |
| `gguf_qwen35.rs` | qwen35 GGUF fixtures (llama.cpp converted) against the safetensors weights, spec, MTP, quant bounds, NVRTC compile | CPU, 1 ign | 1 | checked in (`fixtures/qwen35-gguf`) | none, 1 s |
| `gguf_qwen35_gpu.rs` | qwen35 GGUF logits against safetensors and llama.cpp | 3 | 3 | `SPECRT_GGUF_FILE`, `SPECRT_GGUF_REF` (`scripts/gate676.sh`) | 3 to 40 GB |
| `jit.rs` | NVRTC compile and launch, cache | 2 | 2 | none | 1 GB, 13 s |
| `fused.rs` | fused kernels vs unfused | 2 | 6 | none | 1 GB, 1 s |
| `link.rs`, `loader.rs`, `statepool.rs`, `kv_dtype.rs`, `lin_ba_rows.rs` | linking, weight loading, state pool, KV dtypes, the lin_ba rows fix (#604) | 2 | all | none | 1 to 3 GB, under 30 s each |
| `stage.rs` | pipeline stages | 2 | 6 | none | 2 GB, 7 s |
| `truncation.rs` | context truncation and shifting | 2 | 5 | none | 3 GB, 45 s |
| `linear.rs` | every linear path (bf16, fp8, nvfp4, act-quant) vs references | 2 | 20 | `SPECRT_REF_DIR` with tiny, tiny-fp8, tiny-nvfp4 | 2 GB, 30 s |
| `mamba.rs` | described mixers (#808): tiny Bamba and tiny granitemoehybrid (`scripts/tiny_granitehybrid.py`) from spec files vs transformers, the KDA recurrence vs a host reference, checkpoint state reset and restore, NVRTC compile of the module; the tensor-core chunked scan (#891) against the row-by-row recurrence and its 4k/16k kernel timing | CPU (4), 2 (6) | 6 | `SPECRT_REF_DIR` from `scripts/tiny_mamba.py` (CPU only, seconds) | 1 GB, 15 s |
| `nemotron_nano_9b.rs` | real NVIDIA-Nemotron-Nano-9B-v2 (bf16) vs an fp32 transformers reference: prefill argmax agreement and greedy decode (`NEMOTRON_NANO_9B_DIR`, `scripts/nemotron_nano_ref.py`; Spark, 24 GB) | none | 1 | `SPECRT_REF_DIR` from `scripts/nemotron_nano_ref.py <snapshot> <dir> float32` | 24 GB, 60 s |
| `nemotron_h.rs` | nemotron_h family (#903): `hybrid_override_pattern` strings (M, *, -, E) into mixer, attention and mlp-only layers on real Nemotron Nano 9B v2 and 3 Nano 30B configs, the split-layer plans, the relu2 mapped up projection (`EPI_MAP`), non-gated relu2 experts (E layers, `MOE_MAPPED`) on the Nemotron 3 Nano config and the Mamba anatomy; tiny dense and tiny MoE Nemotron-H (`scripts/tiny_nemotronh.py`, `backbone.*` names) vs transformers | CPU (9), 2 (3) | 3 | `SPECRT_REF_DIR` from `scripts/tiny_nemotronh.py <dir>` (CPU only, seconds) | 1 GB, 5 s |
| `granite_h_tiny.rs` | real Granite 4.0 H-Tiny (Mamba2 hybrid with MoE, scaled residuals) vs transformers: prefill argmax agreement and greedy decode; `[runtime] chunked_scan` on and off at 4k and 16k prompts (same first token, prefill tok/s) | 2 (3) | 3 | `GRANITE_H_TINY_DIR` snapshot and `SPECRT_REF_DIR` from `scripts/granite_h_tiny_ref.py ... float32` (CPU, 28 GB; the bf16 reference is itself noisy at 40 layers) | 14 GB, 11 min |
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
| `vision_multi_gpu.rs` | tiny27b image logits on TP=2 virtual ranks and 2 pipeline stages vs the single GPU path (KL), rank and stage specs keep vision (#910); `scripts/gate910_spark.sh` | CPU (2 spec tests), 2 | 4 | none, uses the tiny27b references | 1 GB, 60 s |
| `real27b.rs` | 12 real-checkpoint gates: greedy and logits vs transformers and vLLM, MTP acceptance, per-layer trace, needle at 500k and 1M | 5 | 12 | `SPECRT_REF_DIR=~/.cache/specrt-ref-375`; `SPECRT_27B_SPEC`, `SPECRT_27B_NVFP4_SPEC`, `SPECRT_27B_NEEDLE_MODEL` | 25 to 96 GB; 1 to 30 min each |
| `real_glm.rs` | `Firworks/GLM-4.5-Air-nvfp4` (#999): logits vs vLLM, 8 prompts x 32 tokens (mean KL 0.0118, max 0.119, teacher-forced top-1 242/256 against vLLM 0.34 marlin), and the export has no nextn layer so it serves without MTP. Reference: `systemd-run --user --wait --collect -p MemoryMax=24G -p MemorySwapMax=0 ~/.venv-vllm/bin/python scripts/vllm_logprobs_offline.py <snapshot> --out $SPECRT_REF_DIR/vllm-glm-air.json --util 0.85 --moe-backend marlin --linear-backend marlin` | 5 | 2 | `SPECRT_GLM_DIR` (the snapshot), `SPECRT_REF_DIR` | 74 GB on GPU1, 40 s |
| `gemma3.rs`, `gemma3_tiny_vision.rs`, `gemma3_vision.rs`, `gemma3_27b.rs` | Gemma 3 text, SigLIP, pan and scan, 27B FP8 vs vLLM (see `specrt-gemma3.md`) | CPU (4), 2, 5 | 12 | `SPECRT_REF_DIR`, `GEMMA3_1B_DIR`, `GEMMA3_4B_DIR`, `GEMMA3_27B_DIR` | 3 to 36 GB |
| `gemma3n.rs` | Gemma 3n: the E2B and E4B configs map AltUp (4 streams), Laurel, per-layer sparsity, per-layer inputs and KV sharing; plan order, kernels and weight names; tiny-gemma3n streams after every layer, logits, greedy and two-slot serving vs transformers (`scripts/tiny_gemma3n.py <dir>`), red checks without sparsity or Laurel | CPU (6), 2 (4) | 10 | `SPECRT_REF_DIR_3N` | 1 GB, 4 s |
| `gemma4.rs` | Gemma 4: 31B, 12B and 26B-A4B configs map, the E2B and E4B configs map their per-layer inputs, KV sharing and wide MLP, 26B FP8 and NVFP4 copies held back by quant tile sizes (#1207), per kind KV bytes, kernels build, tiny-gemma4-moe weights load through the family renames; tiny-gemma4, tiny-gemma4-e and tiny-gemma4-moe logits, greedy, layer states, two-slot serving and a KV sharing red proof vs transformers (`scripts/tiny_gemma4.py <dir>`, `G4_FIXTURE`, `G4_STYLE=e`, `G4_MOE=1`); a red check that a router fed the normed input misses the logit bound; MoE modules compile to PTX with nvrtc | CPU (14), nvrtc (1), 2 (10) | 25 | `SPECRT_REF_DIR`, `SPECRT_REF_DIR_E` | 1 GB, 4 s |
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

Speed per architecture, with nsys tables and llama.cpp and vLLM side by side, is `docs/perf-pass.md` and `crates/specrt/scripts/perfpass.sh`. The scoreboard is `docs/perf/scoreboard.md`. Keep/drop decisions use paired `bench --ab`; its method and claims are in the same file.
