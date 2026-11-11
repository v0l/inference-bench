# Stage server over TCP

Slice 3 of the pipeline work (#538, #540). A `stage` process loads a layer range of a model and serves it over TCP. A coordinator chains the stages: it connects downstream first, checks that the ranges are contiguous and cover the model, then sends each request through the chain.

## Pieces

- One stage-runtime path: `Runtime::load_stage` with a `StageRange` (from #538, `runtime/stage.rs`) loads the layer range, and `Pipeline` is its in-process twin. The stage server calls `load_stage`, then `import_hidden`, `prefill_slot` or `decode`, and `export_hidden`; the last stage samples greedily and returns a token id.
- `wire.rs`: length-prefixed binary frames (`Hello`, `Ready`, `Forward`, `Output`, `Token`, `Free`, `Cancel`, `Ack`, `Error`), capped at 256 MiB, strict decoding (truncated, trailing or unknown frames are errors). Authentication is a pre-shared key: the server sends a 32 byte random nonce, the client answers with HMAC-SHA256(key, nonce), compared in constant time. The key is required (`key` or `key_file`), there is no unauthenticated mode. TCP is plaintext, so run it on a trusted link or a VPN. QUIC with TLS is not built; the transport is behind `StageServer` and `StageClient`, so it can be swapped.
- `stage.rs`: `StageConfig` / `PipelineConfig` (TOML, no env knobs), `RuntimeStage` (the handler), `generate` (the coordinator loop with chunked prefill).
- `bin/stage.rs`: `stage serve <config>`, `stage ping <config>`, `stage run <config> [prompts] [new_tokens] [runs] [chunk]`.

```toml
[stage]
spec = "crates/specrt/tests/fixtures/qwen3.8-27b-nvfp4.toml"
layers = [0, 48]
bind = "10.100.50.1:7701"
key_file = "/path/to/key"
max_ctx = 4352
slots = 1

[pipeline]
stages = ["10.100.50.1:7701", "10.100.50.2:7700"]
key_file = "/path/to/key"
```

A stage loss makes the next call on that connection fail with the stage address in the error; `Chain::free` and `Chain::cancel` release a slot on every stage.

## Tests

- `tests/wire.rs` (CPU): codec round trip and rejection of malformed frames, HMAC against the RFC 4231 vectors, chain order and hidden hand-off with fake stages, gap detection, wrong key, replayed answer, version mismatch, dead stage, empty key.
- `tests/stage.rs` (GPU, ignored): 64 greedy tokens over real TCP stages equal the single runtime on the tiny hybrid (2 and 3 stages, cut at every layer), the FP8 hybrid and the head_dim 256 hybrid; a second slot with different chunking; a killed stage fails the request; stage spec cutting of the 27B NVFP4 spec including the `quant_until` boundary.

## Measured: 27B NVFP4, desktop RTX PRO 6000 + DGX Spark

Stage 0 (layers 0-48, embedding) on the desktop, stage 1 (layers 48-64, final norm, lm_head, sampling) on the Spark over the direct 100 GbE cable (10.100.50.x, 0.13 ms ping RTT). Both load the NVFP4 checkpoint from their own HF cache. Desktop VRAM was shared with the served vLLM (37 GB free), Spark had about 24 GB of free host memory and vLLM and Frigate running; they were left alone. Greedy, batch 1, 64 new tokens, median of 3 (single device: median of 6). `compute` is the time inside the stage handler per step, `hop` is the stage round trip minus its compute (wire, framing, the 20 KB hidden copy).

| prompt | single device prefill | pipeline prefill | single decode | pipeline decode | step | stage compute (desktop, Spark) | hop (to Spark) |
|---|---|---|---|---|---|---|---|
| 128 | 480 tok/s | 445 tok/s | 25.9 tok/s | 12.5 tok/s | 80.3 ms | 32.3, 40.7 ms | 0.23 ms |
| 1024 | 751 tok/s | 657 tok/s | 25.2 tok/s | 11.9 tok/s | 84.1 ms | 32.2, 42.2 ms | 0.23 ms |
| 4096 | 791 tok/s | 442 tok/s | 25.4 tok/s | 6.8 tok/s | 148.2 ms | 35.3, 100.0 ms | 0.37 ms |

- Hop latency: TCP ping round trip 0.006 ms to a local stage and 0.023 ms to the Spark (500 pings, median). A decode hop as the coordinator sees it (stage 0 to coordinator to stage 1, hidden row of 20 KB each way) costs about 0.23 ms, 0.3 percent of the step. The network is not the cost.
- A split never speeds up a single stream: the token visits both stages in turn, so the step is the sum of both, and the Spark (273 GB/s) is the slower device. Its 16 layers read 7.8 GB (28 ms floor at the memory wall) and took 40 ms. The desktop's 48 layers took 32 ms for 15.4 GB. The gain is capacity: the 27B needs 24.8 GB of weights on one device, and here 15.4 GB sits on the desktop and 7.8 GB on the Spark.
- Prefill is sequential through the stages (no micro-batch overlap yet, slice 4), so its rate is the harmonic mix of both stages.
- **The Spark stage's 42 to 100 ms jump is GPU sharing, not context length.** A second run of the same two prompts gave 90.4 ms of Spark compute at 128 tokens (first run: 40.7 ms) and 93.5 ms at 4096 (first run: 100 ms). So the 128 and 4096 steps cost the same within a run, and the same prompt moved by 2x between runs. The Spark GPU showed 96 percent utilisation from the resident vLLM during the window. An nsys profile of the Spark stage (layers 48-64, 273 decode and prefill steps, `--cuda-graph-trace=node`) shows the kernel sum per decode step between 86 and 92 ms at both the start and the end of the run, with `batch_decode_attention` at 0.1 s in total over 1032 launches (about 0.1 ms each), so attention over the longer KV is not the cost. The stage path adds no copy inside a step: the decode graph is captured once per row count and the only host transfers are the 20 KB hidden row in and out. Kernel times are inflated by time slicing with vLLM, for example `lm_head_mma_tiny` ran 21 to 32 ms (2.5 GB of bf16 head, about 9 ms at the 273 GB/s floor). No code change is needed in the stage path; a clean number needs the Spark GPU free of vLLM.
- The one-row head runs `lm_head_mma_tiny`, the largest kernel in the profile (about 25 ms of an 87 ms step here); whether a bandwidth-bound GEMV would beat it under sharing is unmeasured.
- The Spark's `compute_121` and the desktop's `compute_120` each compiled their own kernels (JIT 11.5 and 12.6 s cold).
- Single-device numbers are from `bench` with the same spec on the desktop under the same sharing. The pipeline numbers are not clean device numbers: the GPU of both hosts was shared with other services. Re-run in a quiet window before quoting them as engine speed.

## Reproduce

```
stage serve spark.toml        # on the Spark, downstream first
stage serve local.toml        # on the desktop, under flock /tmp/gpu27.lock
stage ping local.toml
stage run local.toml 128,1024,4096 64 3 1024
```

## Serving a model over remote stages

`inference-bench-server` can use stage servers as its executor (#543). Set `[placement.remote]` next to the usual `[placement]` table; the server then runs the normal scheduler (continuous batching, preemption) over a `wire::Chain` instead of local GPUs. Every setting is in the config, none in env vars.

```toml
max_num_seqs = 4
max_model_len = 512

[placement]
context = 512
slots = 4
discovery = "mdns"
mdns_port = 15353

[[placement.stages]]
device = "a"
layers = [0, 14]

[[placement.stages]]
device = "b"
layers = [14, 28]

[placement.remote]
key_file = "/path/to/key"
timeout_ms = 120000
member_ttl_ms = 30000
connect_wait_ms = 15000
```

Each stage runs `stage serve` with no `layers`, a `[stage.device]` table and `[placement] discovery = "mdns"` (same port); the coordinator assigns the layer ranges when it connects. With `discovery = "static"`, list `address` on each `[[placement.devices]]` instead.

- **KV mapping:** stage servers keep one contiguous KV region per slot, addressed by slot and position. The scheduler's block tables are only used for admission and accounting, so any block order is served (`out_of_order_scheduler_blocks_decode_the_same_over_the_chain`). `set_table` rejects a block id outside `slots * ceil(max_ctx / 16)`, and a forward past the blocks a sequence owns is an error.
- **Stage loss:** a failed forward drops the chain and fails every in-flight request with the stage address in the error (streams end with an error event, then `[DONE]`). The next request reconnects, so a restarted stage is picked up with no server restart.
- **Replan:** the serve loop polls the coordinator each step. An mDNS join or leave that changes the placement or stage addresses fails in-flight requests and rebuilds the chain on the next request.
- MTP and the prefix cache are off on this path.

Checked on this machine with Qwen3-0.6B split 14/14 over two real `stage serve` processes found over loopback mDNS: four concurrent greedy chat completions gave text identical to a single-runtime server; killing stage b mid-stream ended the stream with `a stage was lost ... Connection reset by peer`; requests while it was down failed with `stage 127.0.0.1:7802 is not ready`; after restarting it the same four requests matched again.

## Desktop + Spark end to end (#544)

The 27B NVFP4 (`qwen3.8-27b-nvfp4.toml`) split over the desktop (RTX PRO 6000, `compute_120`) and the DGX Spark (GB10, `compute_121`) on the direct 100 GbE cable, served through `inference-bench-server` with `[placement.remote]`. Both hosts built from the same tree, release, each for its own arch. Keys were mode 600 files, deleted at the end. Work files lived in `~/e544` because `/tmp` filled up (1.7 TB used by other processes).

### Bugs found and fixed on the way

- **Prompts were cut to 2048 tokens.** The checkpoint's `tokenizer.json` carries `"truncation": {"max_length": 2048}`, so `Tokenizer::encode` silently dropped everything past 2048 tokens, in every server, not only the split. A 1024-word and a 4096-word loadtest prompt both reported `prompt_tokens: 2048`. `engine::without_truncation` clears it when the tokenizer is loaded (test `a_tokenizer_json_truncation_limit_does_not_cut_prompts`).
- **Remote decode was one round trip per row.** `ChainExecutor::decode` sent every row as its own forward, so 8 streams cost 8 steps and aggregate decode stayed flat at 12 to 13 tok/s for 1, 4 and 8 streams. New `DecodeBatch` and `Tokens` frames (wire protocol version 2) carry all rows in one hop per stage (`Chain::forward_batch`; tests `a_decode_batch_makes_one_round_trip_per_stage_and_returns_a_token_per_row`, codec round trip and truncation).
- **The remote path exported no `inference_pipeline_*` metrics** and ignored `micro_batch`. `ChainExecutor` now records per-stage busy time, calls, tokens, hop time and bytes (`the_remote_chain_reports_stage_and_hop_metrics`), and prefill can be micro-batched over the chain with one thread per stage (`Chain::forward_pipelined`; `a_pipelined_prefill_sends_every_micro_batch_through_every_stage_in_order`, `a_pipelined_prefill_fails_cleanly_when_a_stage_is_gone`, `a_micro_batched_prefill_over_the_chain_ends_on_the_same_token_as_a_plain_one`).
- **A non-last stage under-reported compute for prefill.** `compute_us` was taken before `export_hidden` waited for the GPU, so stage 0 showed 61 us for a 2.6 ms call and the wait landed in the hop (test `a_first_stage_reports_its_gpu_time_for_prefill_not_just_the_launch`, which failed before the fix).
- **mDNS had no interface choice.** New `[placement] mdns_interface` (name or IP), used by both the stage announcer and the coordinator's browser (tests `mdns_can_be_bound_to_one_named_interface_and_still_finds_the_stage`, `mdns_bound_to_a_missing_interface_never_sees_a_stage_on_another_one`).

### Sharing during the runs

Not a quiet window on the Spark: vLLM (`VLLM::EngineCore` 50 GB plus a 16.6 GB python process) held the Spark GPU at 96 percent utilisation for the whole session, plus Frigate's detector and ffmpeg. The desktop GPU carried the served vLLM (52 GB used, 100 percent) until the middle of the session; after that it was idle (2.3 GB used, 0 to 14 percent) and the second set of numbers below is from that quiet desktop. Nothing was stopped on either host. The Spark's unified memory was tight (about 19 GB free, 4 GB at the end), which briefly refused a stage reload right after the old process exited (see failure test).

### Placement

```
inference-bench plan --spec crates/specrt/tests/fixtures/qwen3.8-27b-nvfp4.toml --config plan.toml --objective latency|throughput
```

`plan.toml`: context 4352, 8 slots, desktop 40 GiB (what vLLM left free), 1792 GB/s, 500 TFLOP/s; Spark 19 GiB free, 273 GB/s, 100 TFLOP/s, link `latency_ms = 0.08` (ping on the cable: 0.103 / 0.164 / 0.211 ms min/avg/max) and 12.5 GB/s (100 Gb/s).

| objective | stages | step | notes |
|---|---|---|---|
| latency | desktop 0..64 | 14.0 ms | 21.3 GiB of weights fit on the desktop, so no split |
| throughput | desktop 0..63, spark 63..64 | 23.6 ms, bottleneck 12.3 ms | one layer on the Spark |

`latency` keeps the 27B on one device, and `throughput` moves one layer, so I pinned the split #540 used with `[[placement.stages]]`: **desktop 0..48, Spark 48..64** (15.4 GB and 7.8 GB of weights).

### 1. Correctness

`stage run` and `inference-bench-server` with `[placement.remote]` (static addresses), 8 concurrent greedy chat completions (`temperature = 0`, 96 tokens, thinking off) against a single-runtime 27B server on the desktop with the same spec:

```
cmp single/0.txt split/0.txt ... cmp single/7.txt split/7.txt
cmp 0 identical  cmp 1 identical  cmp 2 identical  cmp 3 identical
cmp 4 identical  cmp 5 identical  cmp 6 identical  cmp 7 identical
```

The same 8 files were identical again with batched decode, with micro-batched prefill, over mDNS, and after the stage restart below.

### 2. Discovery

- `discovery = "static"` with `address` on each `[[placement.devices]]`: works (above).
- `discovery = "mdns"`, port 15353, stages started empty with a `[stage.device]` table, the coordinator assigning 0..48 and 48..64: works with the default of all interfaces, 8 of 8 texts identical. Multicast does cross the cable: both ends have `MULTICAST` set, are joined to 224.0.0.251, the Spark's INPUT policy is ACCEPT, and the two hosts are on one /30 with no router in between. mDNS was likely also running over the 10.100.2.x LAN (all interfaces), so I proved which path carries it: with `mdns_interface = "enp129s0np0"` on the desktop and the Spark announcing on its LAN port `enP7s7`, the server failed with `device spark has no stage address` (the announcement never reached it); with the Spark also on `enp1s0f1np1` (the cable) the 8 texts matched again. So binding mDNS to the 10.100.50.x interface works, and keeps discovery off the LAN. The Spark has two ports on the cable subnet (`enp1s0f1np1` and `enP2p1s0f1np1`, both 10.100.50.2/30); either works, as the stage binds `10.100.50.2:7700`.

### 3. Throughput

Aggregate decode tok/s through the HTTP server (`inference-bench loadtest`, 64 word prompt, 128 new tokens, N concurrent clients). The 27B split 0..48 / 48..64:

| streams | single, quiet desktop | split (Spark under vLLM, quiet desktop) | split (both GPUs under vLLM) | single (desktop under vLLM) | split before batched decode |
|---|---|---|---|---|---|
| 1 | 90.5 | 8.0 | 13.8 | 40.3 | 13.0 |
| 4 | 221.9 | 30.0 | 44.6 | 99.4 | 13.5 |
| 8 | 300.1 | 51.1 | 72.1 | 116.9 | 12.1 |

The split scales with streams now (a step carries all rows) but never beats single-device. The Spark stage is the limit: its 16 layers took 149 ms per step at batch 1 under vLLM's 96 percent, against 14 ms for all 64 layers on the quiet desktop. A split pays for capacity (the 27B in 15.4 + 7.8 GB), not speed, and only a quiet Spark could narrow that gap (its 273 GB/s floor for 7.8 GB is 28 ms).

TTFT, one client, p50 (single-device, quiet desktop: 0.60 s at 1019 tokens, 2.24 s at 4078 tokens; under vLLM 1.44 s and 5.67 s at 4212 tokens):

| prompt | split `micro_batch = 0` | 256 | 64 |
|---|---|---|---|
| 1019 tokens, under vLLM both | 1.37 s | 1.35 s | 1.45 s |
| 4078 tokens, under vLLM both (2 runs) | 5.34, 6.60 s | 4.92, 5.14 s | 5.81, 8.22 s |
| 1019 tokens, quiet desktop (2 runs) | 1.63, 2.09 s | 1.50, 1.88 s | 2.55, 2.34 s |
| 4078 tokens, quiet desktop (2 runs) | 6.63, 6.37 s | 7.44, 6.06 s | 11.04, 9.26 s |

Overlap across two real devices: 256 is 6 percent faster than 0 on average over the four 4096 runs (5.89 s against 6.24 s), inside the run to run noise (one run 12 percent slower), and 64 is clearly worse (8.58 s, 37 percent slower: each micro-batch adds a frame and a sync). The Spark stage is the slower one, so there is little to hide on the desktop side. The default for remote placements is now `micro_batch = 256` (`[placement] micro_batch` unset), 0 stays the default for in-process stages on a shared GPU, and an explicit value wins either way.

### 4. Hop cost

From `/metrics` (`inference_pipeline_stage_busy_seconds_total`, `..._calls_total`, `inference_pipeline_hop_seconds_total`, `..._hops_total`; hop is the stage round trip minus its compute), deltas over a loadtest of N streams, 128 new tokens, so the counts include a few prefill calls:

| batch | desktop compute per call | Spark compute per call | hop per stage call | share of a step |
|---|---|---|---|---|
| 1 | 14.1 ms | 149 ms | 0.25 ms | 0.3 percent (2 hops, 0.5 ms) |
| 8 | 16.7 ms | 128 ms | 0.39 ms | 0.5 percent |

`stage ping` on the cable: 0.006 ms to the local stage and 0.026 ms to the Spark. A hop is not the cost at either batch size; the Spark's share of the compute is.

### 5. Failure

- Killed the Spark stage (SIGTERM) 6 s into a streaming request: the stream had 56 chunks, then `data: {"error":{"message":"a stage was lost, the request is failed: stage 1 at 10.100.50.2:7700 was lost: failed to fill whole buffer", ...}}` and `data: [DONE]`. A request while it was down got a 500 with `stage 1 at 10.100.50.2:7700 was lost: failed to fill whole buffer`.
- Restarted it (empty, mDNS): the first retries failed with `the GPU ran out of memory: stage 10.100.50.2:7700 could not load layers 48..64 ... cuda init` for about a minute while the Spark's unified memory was still short, and then recovered with no server restart. The 8 texts matched the single-runtime files (8 of 8 identical).
- Link pull: the Spark is reached over the LAN (10.100.2.32), not the cable, so it was safe. During a streaming request `ip link set ... down` then `up` on both Spark cable ports (`enp1s0f1np1`, `enP2p1s0f1np1`) for 2 s: the stream finished (402 chunks, `finish_reason: length`, no error) because TCP retransmission covered the gap. One port came back DOWN because my command line was wrong; I brought it up again and verified both ports and the route.

Stages and servers on both hosts were stopped at the end.
