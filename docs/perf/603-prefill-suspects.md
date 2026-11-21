# #603: why serving prefill is half of vLLM (suspects, before profiling)

Written without the GPU: the user's app held GPU0 and their trainer GPU1, so nothing here was profiled. It ranks the suspects from the code, the earlier measurements in `docs/qwen3.8-27b-vllm-baseline.md` and `docs/vllm-kernels-review.md`, and one read-only scrape of the user's live `/metrics` on :8001 (21 Nov, 27B NVFP4, 8 running).

## The number that reframes it

Single-stream `prefixbench long` on the same NVFP4 spec already reaches 5,065 tok/s at 32k and 5,884 at 4k (#555, quiet GPU, 74% and 67% of vLLM). The serving loadtest gets 3,509 tok/s at 15-18k. So about a third of the loss is in the serving path, not the kernels, and the kernels add the rest.

Live `/metrics` of the user's app, lifetime counters:

| metric | value |
|---|---|
| prefill steps | 42,118 |
| prefill steps with at most 1,024 tokens | 42,117 |
| prefill steps with at most 256 tokens | 579 |
| mean prefill tokens per step | 532 (22.4M / 42,118) |
| mean step time | 98.6 ms (9,012 s / 91,402) |
| mean decode batch | 7.1 |
| prefix cache hit rate | 28.6% |
| snapshots held | 32 (5.0 GB, 157 MB each) |

Under load, prefill is read 512 tokens at a time. vLLM fills its 16,384-token step budget.

## Ranked suspects

### 1. The scheduler gives prefill 512 tokens per step whenever anything decodes

`Scheduler::prefill_budget` (`crates/specrt/src/sched.rs:1353`): with decoders running, `room = itl_target - decode_secs` is about 0 (an 8-row MTP round fills the 100 ms target), so the budget falls to the `min_prefill_tokens` floor of 512 (`ServeOptions::default`, `crates/specrt/src/serve.rs:74-78`). `whole_prompt_tokens` (1,024) never applies to a 17k prompt. A 17k prompt then needs about 33 steps of decode plus 512 tokens, about 150-200 ms each, so its TTFT is 5-7 s even when it's first in line. 512 tokens per 150 ms is about 3,400 tok/s, which is the measured 3,509.

vLLM with `--max-num-batched-tokens 16384` puts `16384 - decodes` prefill tokens into each step and has no inter-token target.

Measure:
- CPU first: `sched.rs` `Fake` engine with the loadtest shape (8 clients, 15-18k prompts at 1/s, 27B step costs from the #655 table). Report TTFT p50 and prefill tok/s at `min_prefill_tokens` 512, 2,048, 4,096 and 16,384, and at `itl_target_ms` 100, 400 and 1,000.
- GPU: the loadtest below with `--min-prefill-tokens 4096` and `16384` against the default. The trade is per-stream decode ITL, so record ITL p95 too.

Fix candidates: a prefill floor that scales with the waiting prompt tokens, or vLLM's policy (fill the budget, no ITL target) as the default once a prompt is over `whole_prompt_tokens`. This is a policy change in `sched.rs`, with no kernel work.

### 2. Only one long prompt is admitted at a time

`Scheduler::admit` (`sched.rs:1131-1138`): with `shortest_first` on, a prompt over `LONG_PROMPT` (2,048) is not admitted while another long prompt is still prefilling. With every request at 15-18k, the 8 clients queue behind each other. That matches "mean running 2.2": one prefilling plus about one decoding.

Measure: `inference_waiting_seqs` and `inference_running_seqs` sampled every second during the loadtest. On CPU, run the `Fake` sim with the gate on and off.

Fix candidates: admit long prompts up to the step budget's worth of KV, and prefill several in one step (see 3).

### 3. Prefill is one sequence per call, and with MTP on it never shares a weight read with decode

In `Scheduler::step` (`sched.rs:1444-1510`), each prefill chunk is a separate `exec.prefill(slot, ...)` for one sequence. Two long prompts in one step run their GEMMs at their own row counts and read every weight twice. When the decoders speculate (`sched.rs:1408-1440`), `pending` is cleared, so the chunk never goes through `fused` (`sched.rs:1492`, `GpuExecutor::fused` at `sched.rs:307`). A step reads the 21 GB of weights twice: once for the MTP verify rows and once for the prefill chunk. vLLM runs decode rows and the prefill chunks of several requests as one forward.

Measure: nsys `--cuda-graph-trace=node` of the 8-client run. Count `gate_up_f4a*` launches per step and their M. Compare a step's kernel time against a prefill-only step with the same token count.

### 4. Kernels at 512-token chunks run below their 2-4k efficiency

All prefill tuning (#551, #555, #441) was done at 2,048-4,096-token chunks. At 512:
- `flash_mma` (`crates/kir/src/cuda/attention/flash_mma.rs`, grid `(tokens/128, 24 heads)` at `crates/specrt/src/runtime.rs:2065`) launches 4 x 24 = 96 CTAs on 188 SMs, so at most half the GPU works on attention over 16k keys. #551 rejected split-KV only because chunks were at least 2,048; at 512 it is the fix (`attention_split`/`attention_reduce` style partial softmax, or 64-row CTAs).
- The NVFP4 and FP8 GEMMs at M = 512: `variant_for` (`crates/kir/src/gemm.rs:704`) picks the big tile only when it gives at least 2 x 188 blocks. Per-call TFLOP/s at 512 has never been measured on the 27B shapes.
- `gdn_chunk_prep` and `gdn_chunk_scan` have a fixed per-launch cost and run sequentially over chunks, 48 layers per step.

Measure: `tests/flash_speed.rs` with `SPECRT_FLASH_TOKENS=512,2048 SPECRT_FLASH_BASES=0,16384,30720 SPECRT_FLASH_FP8=1`. `tests/gemm_speed.rs` with `SPECRT_GEMM_TOKENS=512,2048,4096`. `tests/gdn_speed.rs` at 512 and 4,096. Report ms and TFLOP/s per shape. Each fits beside nothing else, on a synthetic layer with no 27B load.

### 5. The prefix cache's snapshot points cap every chunk at 2,048

The max-performance profile turns the prefix cache on (`crates/specrt/src/options.rs:33`). `snapshot_interval` defaults to 2,048 (`crates/specrt/src/spec.rs:248`). `Scheduler::step` cuts each chunk at the next snapshot point (`sched.rs:1472-1476`), so even with suspect 1 fixed a chunk is never larger than 2,048. The spec's `prefill_chunk = 4096` was measured 8-10% faster than 2,048 at 32k (#551). Each point also copies the 157 MB recurrent state. A 17k prompt takes 8 copies, and with `max_snapshots` 32 the pool churns at 8 clients.

Measure: `prefixbench long` at 16k and 32k, cache on with interval 2,048 against 4,096 and 8,192, plus cache off. nsys on the D2D memcpy time per prompt.

### 6. fp8 KV in `flash_mma` is unmeasured at long context

The app runs `kv_dtype = "fp8"`. Every long-context prefill number (#551, #555) used bf16 KV. With fp8, every K/V tile runs `convert_kv` (`flash_mma.rs`, `convert_kv` lambda): all 256 threads widen 2 x 32 x 256 e4m3 values to f16 and write them back to shared memory, with two extra `__syncthreads` per tile. #551 showed K/V loads were hidden behind the MMAs with bf16. With fp8 the convert sits on the critical path between `cp_async_wait` and the MMAs. It grows linearly with keys, so it weighs most at 32k and above.

Measure: `tests/flash_speed.rs` with and without `SPECRT_FLASH_FP8=1` at bases 0, 16,384 and 30,720 (2,048 query tokens). If fp8 is over 10% slower, move the convert into the fragment load, as #651 did for decode.

### 7. Known kernel gaps from #555 (single stream, still open)

At 4k, against vLLM's kernels:
- delta-net prefill is 185 ms against 42 (4.4x);
- the NVFP4 MLP GEMMs reach 1.4-1.6x of CUTLASS's time on real data;
- the FP8 projections take 177 ms.

At 32k, `flash_mma` is 18.7% of kernel time and `gdn_chunk_scan` 12.2%. These set the single-stream ceiling (74% of vLLM at 32k). The #603 target of 80% at 16k and 32k needs some of this as well as 1-5.

### 8. Host-side gaps in the serving loop

#555 found the single-stream prefill 99.9% GPU-busy. The serving step adds things the single-stream bench does not do:
- `exec.flush()`, a full synchronise, every step (`sched.rs:1525`);
- `set_masks`;
- `cache_progress` hashing per block;
- the snapshot copies;
- `set_table` uploads at admission.

Measure: the nsys timeline of the 8-client run with `crates/specrt/scripts/prefill_timeline.py`, reporting every gap over 20 us and the total idle per step.

## Order to work in

1 and 2 are scheduler policy, now built as options (below). Then 5 (a config default). Then 4 and 6, which matter only at the chunk sizes the scheduler ends up using. Then 3 and 7, the larger kernel and executor work.

## Suspects 1 and 2: new options and the CPU sim

Two serving options, both off by default so behaviour is unchanged until a GPU run decides:

| option | default | effect |
|---|---|---|
| `serve.prefill_burst_tokens` (`prefill_burst_tokens` in server.toml, `--prefill-burst-tokens`) | 0 | While a prompt over 2,048 tokens is waiting or being read, a step carries at least this many prompt tokens (capped by `max_batched_tokens`), even when decoders fill `itl_target`. With no long prompt pending, the budget is exactly as before, so chat and short prompts keep the `itl_target` behaviour. |
| `serve.max_long_in_flight` (`max_long_in_flight`, `--max-long-in-flight`) | 1 | Prompts over 2,048 tokens admitted for prefill at the same time. 1 is the old hard gate. |

Tests in `crates/specrt/src/sched.rs` (both failed on the old code, then passed with the options set):
- `a_long_prompt_beside_decoders_is_read_in_burst_sized_steps`: 8 MTP decoders and one 16k prompt. Default: at least 30 steps to the first token (512 per step). `prefill_burst_tokens = 16384`: at most 6.
- `several_long_prompts_may_prefill_at_once`: 4 waiting 16k prompts. Default: one at a time. `max_long_in_flight = 4`: all four.

Sim (`print_long_prompt_table`, ignored, run with `--ignored --nocapture`): the user's shape. 24 requests arrive 1 per second, 8 clients, 15-18k prompts, 256-511 output tokens, MTP k = 2 on every decode step, `max_batched_tokens` 16,384, `prefill_chunk` 4,096. Costs are modelled on the 27B: 12.5 ms per decode row, so 100 ms for 8 rows, and 0.2 ms per prompt token, about 5,000 tok/s, the single-stream rate. The Fake engine runs at 1/10 of that and the clock is scaled back up. `itl_target` is 100 ms (scaled). The stream decode column is plain tokens over time: the Fake accepts 0-2 drafts at random, so compare only within the table.

| policy | TTFT p50 | TTFT p95 | prefill tok/s | mean running | stream decode p50 | token gap p50 / p99 / max |
|---|---|---|---|---|---|---|
| default (floor 512, 1 long) | 10.5 s | 20.4 s | 3,043 | 6.0 | 9.6 tok/s | 177 / 242 / 278 ms |
| `min_prefill_tokens` 4096 | 3.8 s | 15.5 s | 4,554 | 6.9 | 8.3 | 100 / 907 / 907 ms |
| `min_prefill_tokens` 16384 | 3.6 s | 16.6 s | 4,849 | 6.8 | 8.4 | 100 / 3,363 / 3,363 ms |
| `prefill_burst_tokens` 16384 | 3.6 s | 16.6 s | 4,838 | 6.8 | 8.4 | 100 / 3,363 / 3,363 ms |
| `max_long_in_flight` 4 | 7.6 s | 23.9 s | 3,029 | 7.2 | 9.0 | 177 / 230 / 278 ms |
| burst 16384 + long 4 | 3.6 s | 10.0 s | 4,847 | 6.9 | 8.6 | 100 / 3,351 / 3,363 ms |
| burst 8192 + long 4 | 3.7 s | 13.4 s | 4,730 | 6.9 | 8.7 | 100 / 1,726 / 1,726 ms |

What the sim says:
- The default reproduces the field numbers: 3,043 tok/s of prefill against the measured 3,509, and 512-token steps.
- The burst cuts TTFT p50 about 3x (10.5 to 3.6 s) and takes prefill to the single-stream rate. That meets the issue's 2x TTFT target on policy alone, if the GPU agrees.
- The cost is the tail of the token gap. A burst step that reads 16k tokens stalls every running stream for about 3.4 s once per long prompt. The median gap gets shorter, because steps without prefill are pure 100 ms decode steps. 8,192 halves the stall.
- `max_long_in_flight` alone does not help: the 512 floor is the limit. With the burst it brings TTFT p95 from 16.6 to 10.0 s, because later arrivals stop queueing behind one prompt at a time.
- Here `prefill_burst_tokens` and a higher `min_prefill_tokens` give the same numbers, because every prompt is long. They differ for chat: `min_prefill_tokens` also stretches every step that carries a short prompt, while the burst applies only when a prompt over 2,048 tokens is pending, so interactive traffic keeps the `itl_target` budget.

## The GPU runs that decide the defaults

Run all in one window on a quiet GPU0, with the same binary, spec and loadtest, one server at a time. Each arm is one `serve` start and one loadtest.

1. default: no extra flags;
2. `--min-prefill-tokens 4096`;
3. `--min-prefill-tokens 16384`;
4. `--prefill-burst-tokens 8192 --max-long-in-flight 4`;
5. `--prefill-burst-tokens 16384 --max-long-in-flight 4`;
6. a chat-shape control for arms 1 and 5: `--mix 1:500-1500 --output 256-512 --rate 4 --clients 16`, to show the burst leaves short-prompt ITL unchanged.

For each arm, record TTFT p50 and p95, prompt tok/s, ITL p50, p95 and max (the loadtest report), and `inference_prefill_tokens_per_step` from `/metrics`.

What decides it:
- Make burst 16384 + long 4 the default if it reaches TTFT p50 at most 16.4 s (2x better than 32.8) on the long shape, and the chat control's ITL p95 is within 5% of the default.
- If the 3 s ITL max is judged too long, take burst 8192 as long as it still halves TTFT.
- Arms 2 and 3 show what the flat floor costs chat traffic, for comparison.

## GPU commands for the profiling window

On a quiet GPU0 only, after checking `nvidia-smi --query-compute-apps=pid,process_name --format=csv` shows no `inference-bench-ui`. Build with `CARGO_TARGET_DIR=/mnt/data/cargo-target/inference-bench-603`. Write the spec as a copy of `tests/fixtures/qwen3.8-27b-nvfp4.toml` with `kv_dtype = "fp8"` added under `[runtime]`, at `/tmp/603/nvfp4-fp8kv.toml`.

```sh
B=/mnt/data/cargo-target/inference-bench-603/release
~/.agents/bin/gpu-run 0 -- $B/inference-bench serve --bind 127.0.0.1:18603 --model /tmp/603/nvfp4-fp8kv.toml \
  --max-num-seqs 26 --kv-cache-gb 22.5 --max-batched-tokens 16384 --max-model-len 65536 &
$B/inference-bench loadtest --url http://127.0.0.1:18603 --model nvfp4-fp8kv --requests 40 --rate 1 --clients 8 \
  --mix 1:12000-14000 --output 256-512 --json > /tmp/603/load-default.json
```

Repeat the server for each arm above (for example `--prefill-burst-tokens 16384 --max-long-in-flight 4`). For the kernel capture, put the server under `nsys profile --cuda-graph-trace=node --trace=cuda -o /tmp/603/serve8` and send 8 co-arriving 16k prompts with `--requests 8 --rate 0`. Then 1 prompt. Then the same at 4k and 32k.

For vLLM, run `docs/qwen27-run/launch.sh mtp` alone on GPU0 (port 8011, under `systemd-run --user -p MemoryMax=24G`, `MAX_JOBS=3 NVCC_THREADS=2`) with the same loadtest. Stop it before starting specrt.

The kernel microbenches (suspects 4 and 6) need no 27B load:

```sh
cargo test --release -p inference-bench-specrt --test flash_speed --test gemm_speed --test gdn_speed --no-run
SPECRT_FLASH_TOKENS=512 SPECRT_FLASH_FP8=1 ~/.agents/bin/gpu-run 0 -- cargo test --release -p inference-bench-specrt --test flash_speed -- --ignored --nocapture
SPECRT_GEMM_TOKENS=512,2048,4096 ~/.agents/bin/gpu-run 0 -- cargo test --release -p inference-bench-specrt --test gemm_speed prefill_gemm_speed_fp8_channel_and_fp4 -- --ignored --nocapture
```

## #973 FP8 huge tile at 16,384 rows

Isolated `_huge` GEMM on GPU1 (RTX PRO 6000), m = 16384, median of 10. The tile is set with `[runtime] fp8_tile = [bm, bn, wm, wn]` (`SPECRT_F8_TILE=128x128x2x4` in `tests/gemm_speed.rs`). TFLOP/s for gate_up / down / lin_in, block-128 FP8 spec:

| tile | gate_up | down | lin_in |
|---|---|---|---|
| 128x128 2x4 (default) | 355 | 373 | 395 |
| 128x128 4x2 | 366 | 390 | 397 |
| 256x128 2x4 | 355 | 385 | 363 |
| 128x256 4x2 | 254 | 315 | 339 |
| 128x256 2x4 | 224 | 312 | 311 |
| 256x128 4x2 | 279 | 294 | 294 |
| 64x256 2x4 | 194 | 370 | 381 |

Per-channel spec (the NVFP4 model's FP8 layers): 128x128 2x4 gives 484 / 493 / 497, 128x128 4x2 gives 468 / 487 / 491. The best alternative is +3% on gate_up and +4.5% on down for one spec only, so no tile reaches the 5% bar. The default stays and no loadtest was run.
