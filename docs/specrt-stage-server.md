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
