# Distributed: tensor parallel, pipeline stages, hosts and RDMA

A model spreads over several GPUs in two ways: tensor parallel (`serve.tp`, every rank holds a slice of every layer, one host) and pipeline stages (`[placement]`, contiguous layer ranges per device or host). `specrt::solve` picks between them for every load. The app's Hardware tab, pairing and the load dialog are in [app.md](app.md).

## Placement and solve

`crates/specrt/src/placement.rs` (pure, unit tested). Inputs: the spec through `ModelAnatomy`, devices in pipeline order, a workload (context, slots, hop bytes).

- Device: memory, reserve, bandwidth, TFLOP/s, optional host offload budget, `host` (devices with the same host can form a TP group), `wgpu` (a wgpu device is refused for a model with a wgpu blocker), `address` for static remote members. Link: latency and bandwidth to the previous device; `peer` for the TP link.
- Per layer: weight bytes, active bytes per token, KV bytes per token, state bytes per sequence. Stage memory is layers plus embedding (stage 0) or head (last stage) plus KV and state for `context x slots` plus reserve.
- Stage time: `max(bytes / bandwidth, 2 x params x slots / FLOPs)` per layer; a hop is latency plus `slots x hidden x hop bytes / bandwidth`. A TP group divides bytes and FLOPs by tp and adds two allreduces per layer.
- Objectives: `latency` (default, minimises the sum, so a model that fits one device stays there) and `throughput` (minimises the slowest stage plus its hop). Both are an exact DP over (last device, layers covered).
- `[[placement.stages]]` pins ranges by hand. An infeasible plan names the bytes needed and available.

`specrt::solve::solve(PlacementRequest { mode, target, pins, devices, spec })` returns `Solved`, the one plan a load uses: stages, tp, KV dtype and pool, slots, context, MTP and the option map. The engine does not replan and has no fallback; a stage that cannot load fails with `stage N (layers a..b) on GPU g: ...`. Max performance prefers TP across matched GPUs on one host even when one GPU fits, then a pipeline; f16 KV and MTP when they fit. Low memory takes the smallest GPU that fits, fp8 KV, two slots, MTP off. The load dialog shows the `Solved` and hands exactly that to the engine (`EngineChoice.solved`); server.toml loads call `solve::for_options`.

```
inference-bench plan --spec <spec.toml> --config <server.toml> [--objective latency|throughput] [--probe]
```

`--probe` measures peer copies between local GPUs and round trips to remote devices.

## Tensor parallel

`serve.tp = N` (default 1). No NCCL: a kir allreduce joins the ranks twice per layer.

- Shards are declared on every `linear` node of `graphs/decoder.toml`: `column` (by head: qkv, gate_up, lin_in, lin_ba), `row` (input columns, then an allreduce: o, down, lin_out), `replicate` (`mtp_fc`), `vocab` (lm_head and the MTP head: each rank computes a vocabulary slice, rounded to 16 rows, and `ar_gather_f32` rebuilds the full row on every rank). `Graph::shards()` refuses a linear without a shard.
- `tp::RankCut` cuts checkpoint tensors by head, segment (fused qkv and gate_up, the delta-net q, k, v, z, a, b) and block format (NVFP4 packed and group scales, FP8 channel scales, FP8 128 blocks on block edges). `Checkpoint::for_rank` applies them in `view()`, so every loader path reads only its slice. `tp::rank_spec` divides heads and widths; a rank runs the kernels of a smaller model.
- MoE: `[moe] tp_split = "auto" | "experts" | "columns"` (`tp::plan_moe`; auto picks columns when each rank keeps at least 256 columns per expert).
- `tp::supports` refuses with the reason: head counts tp does not divide, GGUF, sandwich or scaled residual norms, full width qk norms, tiled delta-net value heads, widths not a multiple of 16.
- `kir::allreduce` modes `OneShot` (up to 1 MiB), `TwoShot` (reduce scatter and allgather), `AddRmsNorm` (sum, add into x, RMSNorm into the next GEMM's planes), `Gather` (vocab heads). Sums are f32 in rank order, so ranks hold identical results. Barriers are per block epochs in device memory with release and acquire, so CUDA graphs replay them; a barrier that waits about 10 s traps.
- `allreduce::Fabric` transports: `Peer`, `SameDevice` (virtual ranks for tests) and `HostPinned`, used when peer access is off, the IOMMU translates (`iommu_group/type` `DMA` or `DMA-FQ`), or the timed probe `peer_writes_land` sees a peer store not land within 200 ms.
- `TpExecutor` runs one thread per rank (`Runtime::load_rank`) behind the scheduler's `Executor` trait and stops with an error if ranks ever disagree on a token. `compile::partition` writes the rank partials ([kir-compiler.md](kir-compiler.md)). MTP is sharded like a layer. Not covered: the fused decode plus prefill step, TP together with pipeline stages.

## Pipeline stages

A stage owns the weights, KV and state of its layer range (`Spec::stage`, `Runtime::load_stage`, `runtime/stage.rs`); stage 0 owns the embedding, the last stage the head and sampling. The hand off is the f32 residual, `rows x hidden x 4` bytes (20 KB a token on the 27B), which keeps a pipeline bit identical to one runtime.

- In one process: `Pipeline` and `PipelineExecutor`; the hand off is a device copy, `cuMemcpyPeerAsync` or host staging (`specrt::link::DeviceLink`). `[placement] micro_batch` cuts prefill into micro batches with one thread per stage; unset is 0 in process (on one GPU it was 1.7 to 2.8x slower) and 256 over remote stages.
- Over the network: `stage serve <config>` runs a range; `wire.rs` frames (`Hello`, `Ready`, `Assign`, `Forward`, `DecodeBatch`, `Output`, `Tokens`, `Free`, `Cancel`, `DirectOpen`, `DirectInfo`, `DirectConnect`, `DirectMode`, `Pushed`, ...) capped at 256 MiB with strict decoding. Authentication is HMAC-SHA256 over a server nonce with a pre-shared key (`key` or `key_file`, required); TCP is plaintext, so use a trusted link. The server runs its normal scheduler over a `wire::Chain` with `[placement.remote]` (`key_file`, `timeout_ms`, `member_ttl_ms`, `connect_wait_ms`, `transport`, `rdma_library`, `gpu_direct`). Stages keep one contiguous KV region per slot; a lost stage fails in flight requests with its address and the next request reconnects.
- Pipelined serving runs without MTP and without the prefix cache (`serve.rs` warns at start).

## Discovery and hosts

`[placement] discovery = "static"` (default) or `"mdns"` with `mdns_port` (5353) and `mdns_interface`. Stages announce `_specrt-stage._tcp.local.`; the app and `inference-bench host` announce `_inference-bench._tcp` with a versioned TXT record. `cluster.rs` (`Membership`, `replan`, `diff`) and `coordinator.rs` turn joins and leaves into replans; a changed placement fails in flight requests and rebuilds the chain.

`inference-bench host --config host.toml` (`specrt::host::Host`) shares GPUs headless: `bind`, `mdns_port`, `interface`, `key_file`, `control_socket`, `gpus`, `reserve_gb` (4), `max_memory_gb`. It refuses an assignment when the estimate plus `reserve_gb` is over the free memory. `inference-bench host pair` prints a pairing code over the local socket. Units: `packaging/systemd/inference-bench-host.service` (with `OOMScoreAdjust=1000`, `MemoryMax=12G`) and `packaging/macos/com.v0l.inference-bench-host.plist`. Without CUDA the host probes wgpu (Apple memory and chip specs from `hardware::apple_chip_specs`) and serves dense stages through `Dense::load_stage`.

## RDMA and GPU direct

`[placement.remote] transport = "tcp" | "rdma" | "auto"` (default tcp). libibverbs is loaded with dlopen (`crates/specrt/src/rdma.rs`); `auto` falls back to TCP when it is absent. The authenticated TCP connection stays the control channel and carries the queue pair endpoints (no rdma_cm); frames then go as RDMA writes with immediate into a ring of 8 registered 64 KiB slots with credits. The GID is taken from an active RoCE port matching the local IPv4. Each connection pins 2 MiB, so the default 8 MB `ulimit -l` holds about 4. RoCE needs PFC on the link to be lossless; that is not configured here.

`gpu_direct = true` (default false, needs rdma or auto on both ends): stages export their activation buffers as dma-buf (`cuMemGetHandleForAddressRange`), register them with `ibv_reg_dmabuf_mr`, and answer a forward with `Pushed`: one RDMA write from the outbox straight into the next stage's landing buffer. The receiver checks the byte count from the immediate against the authenticated control frame. Micro batched prefill switches direct mode off for its duration. Any refusal resets every stage to byte frames.

## Measured state

| Case | Value | Source |
|---|---|---|
| 27B NVFP4 TP2 against TP1, desktop pair, batch 1 / 8 decode | 169.9 / 686.9 tok/s against 117.3 / 465.3 (1.45x, 1.48x) | #658, 20 Nov 2026 |
| same, 4k prefill | 3,050 against 1,852 tok/s | #658 |
| allreduce bf16 5120 wide over peer, 1 / 8 / 64 rows | 7.7 / 11.3 / 37.7 us | #658 |
| 27B NVFP4 TP2 against TP1 logits | top-1 256 of 256, mean KL 1.0e-8 | #658 |
| 122B NVFP4 TP2 against TP1, batch 1 / 8 | 229.6 / 789.5 against 159.2 / 475.2 tok/s | #798 |
| 27B split desktop 0..48 and Spark 48..64 over 100 GbE | texts identical to one runtime; hop 0.25 to 0.39 ms, under 0.5 percent of a step | #544 |
| GPU direct hop against pinned ring, 4 MB | 653 us against 1.46 ms | #1033 |

A split never speeds up a single stream (the token visits each stage in turn); it buys capacity. The desktop to Spark rows were taken with the Spark's vLLM resident, so they are not clean device numbers. The vocab sharded heads (#921) and the desktop to Spark GPU direct run have not been timed on two cards.

## Tests

CPU: `cargo test -p inference-bench-kir --test allreduce`, `crates/specrt/tests/tp.rs`, `wire.rs`, `remote.rs`, `discovery.rs`, `host.rs`, `direct_fallback.rs`, the `tp::`, `allreduce::`, `placement::` and `solve::` unit tests. GPU: `tp.rs` virtual ranks, `allreduce.rs`, `link.rs`, `pipeline.rs`, `stage.rs` (64 greedy tokens over TCP stages equal one runtime), `direct_chain.rs`, `direct.rs`, `rdma.rs` and `rdma_dmabuf.rs` (skip without RoCE), `solved_load.rs`. Two GPU runs (supervisor): `crates/specrt/scripts/pair658.sh`, `gate776_two_gpu.sh`; virtual ranks: `gate658.sh virtual`.
