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

1 and 2 are scheduler policy. They can be built and proven on CPU with the `Fake` sim, and are likely most of the 2x TTFT target on their own. Then 5 (a config default). Then 4 and 6, which matter only at the chunk sizes the scheduler ends up using. Then 3 and 7, the larger kernel and executor work.

## GPU commands for the profiling window

On a quiet GPU0 only, after checking `nvidia-smi --query-compute-apps=pid,process_name --format=csv` shows no `inference-bench-ui`. Build with `CARGO_TARGET_DIR=/mnt/data/cargo-target/inference-bench-603`. Write the spec as a copy of `tests/fixtures/qwen3.8-27b-nvfp4.toml` with `kv_dtype = "fp8"` added under `[runtime]`, at `/tmp/603/nvfp4-fp8kv.toml`.

```sh
B=/mnt/data/cargo-target/inference-bench-603/release
~/.agents/bin/gpu-run 0 -- $B/inference-bench serve --bind 127.0.0.1:18603 --model /tmp/603/nvfp4-fp8kv.toml \
  --max-num-seqs 26 --kv-cache-gb 22.5 --max-batched-tokens 16384 --max-model-len 65536 &
$B/inference-bench loadtest --url http://127.0.0.1:18603 --model nvfp4-fp8kv --requests 40 --rate 1 --clients 8 \
  --mix 1:12000-14000 --output 256-512 --json > /tmp/603/load-default.json
```

Repeat the server with `--min-prefill-tokens 4096`, then `16384`, for suspect 1. For the kernel capture, put the server under `nsys profile --cuda-graph-trace=node --trace=cuda -o /tmp/603/serve8` and send 8 co-arriving 16k prompts with `--requests 8 --rate 0`. Then 1 prompt. Then the same at 4k and 32k.

For vLLM, run `docs/qwen27-run/launch.sh mtp` alone on GPU0 (port 8011, under `systemd-run --user -p MemoryMax=24G`, `MAX_JOBS=3 NVCC_THREADS=2`) with the same loadtest. Stop it before starting specrt.

The kernel microbenches (suspects 4 and 6) need no 27B load:

```sh
cargo test --release -p inference-bench-specrt --test flash_speed --test gemm_speed --test gdn_speed --no-run
SPECRT_FLASH_TOKENS=512 SPECRT_FLASH_FP8=1 ~/.agents/bin/gpu-run 0 -- cargo test --release -p inference-bench-specrt --test flash_speed -- --ignored --nocapture
SPECRT_GEMM_TOKENS=512,2048,4096 ~/.agents/bin/gpu-run 0 -- cargo test --release -p inference-bench-specrt --test gemm_speed prefill_gemm_speed_fp8_channel_and_fp4 -- --ignored --nocapture
```
