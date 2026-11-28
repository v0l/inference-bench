# Pipeline inference across GPUs and hosts

Issue #538. A model is cut into contiguous layer ranges (stages). Each stage is a `specrt` runtime that owns the weights, the KV cache and the delta-net state of its layers. Between stages only the residual hidden state moves.

## Prior art

**exo.** `src/exo/master/placement_utils.py` builds pipeline shard assignments with `allocate_layers_proportionally`: each node gets a share of the layers equal to its share of the cluster memory, rounded by largest remainder, with at least one layer per node. It ignores link speed and per-layer cost. exo also has tensor parallel (MLX distributed, RDMA over Thunderbolt 5), which only pays off on fast links.

**mesh-llm / Skippy.** Fit the model on one node first, else route to a peer that holds it, else split into stages. Ranges are half-open, non-empty, contiguous, cover `0..layer_count` and use each node once. A split can be locked by hand with a topology file. Downstream stages load first and stage 0 becomes routable only when every stage reports ready. Each node fetches only the layers of its stage. Activations travel as raw f32 over QUIC.

**llama.cpp** `-ts` / `--split-mode layer` splits by a fixed ratio; `--rpc` puts a remote device in the list. **vLLM** pipeline parallel cuts by layer count.

What we take: mesh-llm's fit-first policy and start order, exo's memory weighting as the feasibility rule, and a cost model on top (per-layer bytes, KV, state, device speed, link) that neither has.

## Stage model

- Stage 0 owns the embedding. The last stage owns the final norm, `lm_head` and sampling. MTP stays out of pipelined stages for now (it needs the last hidden state back on stage 0).
- A stage runtime is loaded from a stage spec (`Spec::stage`): its layer count and block pattern are those of the range, `quant_until` is shifted, and `linear_attention` is dropped when the range holds no delta-net layer. Weights are read from the checkpoint with global layer indices, so a stage never touches the other layers' tensors.
- The hand-off is the f32 residual stream, `rows x hidden x 4` bytes: 20 KB a token for the 27B (hidden 5120). It is f32 because the runtime keeps the residual in f32, and that is what makes a pipelined run bit-identical to a single runtime. bf16 hops would halve the bytes and cost exactness; they are a later opt-in.
- Prefill runs chunk by chunk through the stages. With micro-batches in flight the stages overlap (slice 4). Decode is one hop per stage per step.

## Hop cost against step time

The 27B FP8 decode step is about 12 ms on the RTX PRO 6000 (bandwidth bound). One decode token is 20 KB.

| link | latency | bandwidth | hop for 1 token | for 8 slots |
|---|---|---|---|---|
| NVLink / PCIe 5 peer copy | 5 us | 25-50 GB/s | 6 us | 10 us |
| host-staged PCIe | 20 us | 12 GB/s | 22 us | 33 us |
| 10 GbE LAN | 100-300 us | 1.1 GB/s | 0.12-0.32 ms | 0.25-0.45 ms |
| 1 GbE LAN | 200-500 us | 0.11 GB/s | 0.4-0.7 ms | 1.6 ms |
| Thunderbolt / 100 GbE RDMA | 10-30 us | 10 GB/s | 15-35 us | 30-50 us |

On a LAN the hop is 1-3 percent of a step per stage boundary, so pipelining across hosts costs little for decode. The gain is capacity (a model that does not fit one box) and, with several requests in flight, throughput. A single stream never gets faster from a split, because its token still visits every stage in turn.

## Placement planner

`crates/specrt/src/placement.rs`, pure and unit tested. Inputs are the spec (through `ModelAnatomy`), the devices in pipeline order and a workload (context, slots, hop bytes per element).

- **Device**: memory, reserve, bandwidth, TFLOP/s, optional host offload budget and bandwidth. Declared in config now; measured values are a later addition.
- **Link** between neighbouring devices: latency and bandwidth. A link across skipped devices adds latencies and takes the lowest bandwidth.
- **Per layer**: weight bytes at the spec quant, active bytes read per token, KV bytes per token (attention layers only), state bytes per sequence (delta-net layers only).
- **Stage memory** = layer weights + embedding on stage 0 + head, norm and MTP on the last + KV for `context x slots` + state x slots + device reserve. It must fit.
- **Stage time** for a decode step = max(bytes read / bandwidth, 2 x params x slots / FLOPs), summed over layers. Delta-net layers read their state, attention layers read their KV, so equal layer counts are not equal cost.
- **Hop time** = link latency + slots x hidden x hop bytes / bandwidth, charged to the receiving stage.
- **Objective** `latency` (default) minimises the sum of stage and hop times, which is what a single stream sees, and so keeps the model on one device when it fits. `throughput` minimises the slowest stage plus its incoming hop (what a full pipeline sustains), breaking ties on the sum. Both are exact: a DP over (last used device, layers covered), with a binary search on the stage cost bound for `throughput`.
- **Offload**: when `host_offload_gib` is set and a stage overflows, the last layers of the stage are marked offloaded and costed at host bandwidth. The tiered store that serves them is #323; the planner only marks them.
- An infeasible plan is an error naming the bytes needed and the memory available.

`inference-bench plan --spec <spec.toml> --config <server.toml>` prints the placement. The server config carries it:

```toml
[placement]
objective = "latency"
context = 32768
slots = 8

[[placement.devices]]
name = "rtx-pro-6000"
memory_gib = 96
bandwidth_gbs = 1792
tflops = 500

[[placement.devices]]
name = "spark"
memory_gib = 110
bandwidth_gbs = 273
tflops = 100
link = { latency_ms = 0.3, bandwidth_gbs = 1.1 }
```

`[[placement.stages]]` pins ranges by hand (`device = "spark", layers = [40, 64]`) and is evaluated and checked by the same cost model.

## Single host, several GPUs (slice 2)

One process, one runtime per device. The hand-off is `Runtime::hand_off`: a device copy when both stages share a device, `cuMemcpyPeerAsync` when peer access exists, and host staging (`export_hidden` / `import_hidden`) otherwise. Staging is also the wire path of the multi-host case, so it is tested in slice 1.

The cross-device copy is `specrt::link::DeviceLink`. `LinkMode::Peer` enables peer access both ways and runs `cuMemcpyPeerAsync` on the destination stream; `LinkMode::Staged` goes device to pinned host to device. Both order the copy with events so the destination stream waits for the source stream's earlier work. `hand_off` picks the mode with `cuDeviceCanAccessPeer`.

Tests in `crates/specrt/tests/link.rs` (ignored, need a GPU) run both modes between two contexts on ordinal 0, which emulates two devices on a one-GPU host, and check that two runtimes in one process agree. Real peer and staged bandwidth has not been measured: it needs a host with two GPUs.

## Multi host (slice 3)

A stage server process runs a layer range. The coordinator opens a stream to each stage and sends: hidden rows (or token ids to stage 0), slot ids, positions, free and cancel. Downstream stages start first and stage 0 is published when the chain is ready. QUIC (quinn) with a pre-shared key is the target; plain TCP behind config is acceptable for the first cut. A stage loss fails in-flight requests and triggers a re-plan.

## Discovery (slice 5)

A static device list first (the config above), then mDNS. Nostr and relays are out of scope.

Config: `[placement] discovery = "static"` (default) or `"mdns"`, `mdns_port` (default 5353), `mdns_interface` (an interface name or IP; default all interfaces). `micro_batch` unset means 256 over remote stages and 0 in process (measured on the desktop and the Spark in `docs/specrt-stage-server.md`). `[[placement.devices]]` takes an optional `address` for static members. With `mdns`, a `stage serve` config needs a `[stage.device]` table (same fields as a placement device) and may omit `layers`: the stage starts empty and loads whatever range the coordinator assigns.

Crate: `mdns-sd` 0.21 (pure Rust, no async runtime, maintained). Each stage registers `_specrt-stage._tcp.local.` with TXT keys from `encode_announcement` and its TCP address; `discovery::Browser` turns resolved and removed services into `cluster::Event`s.

- `cluster.rs` (transport-free): `Membership` starts from the static list, applies `Join` and `Leave`, expires discovered nodes that stop announcing (static ones never leave), `replan` reruns the planner over the members (pinned stages whose device is gone are dropped, device order follows the pins), `diff` reports moved stages.
- `coordinator.rs`: `Coordinator` owns membership and the chain. `poll` feeds mDNS events and expiries into `Membership`; every change drops the chain and replans. `generate` waits for a chain (downstream first, `Chain::connect_assigned` sends `Assign` to any stage whose loaded range differs, so a stage whose layers moved reloads them) and runs the request. A lost stage fails the in-flight request with the stage address ("stage N at ADDR was lost") and drops the chain; the next request waits for the replanned chain.
- Wire: new `Assign { layer_start, layer_end }` frame, answered with `Ready`.

Tests: `tests/discovery.rs` (announce and browse on loopback, join then leave replans, kill and rejoin of a stage, and a GPU test that kills a stage process mid-request and gets bit-identical tokens after it returns). The multicast tests skip when the host has no multicast.

Not done: the HTTP server (`serve.rs`, `PipelineExecutor`) still pipelines local runtimes only. A scheduler executor over remote stages, and a live replan of the served model, are tracked as a follow-up.

## Tensor parallel

Later, and only for fast links. It needs an allreduce per layer (NCCL class), so a LAN or PCIe pair will not pay for it. Not built here.

## Slices

1. Planner and in-process pipeline of several runtimes on one GPU, bit-identical to the single runtime (this change).
2. Multi-GPU in one process: peer or staged copies.
3. Stage server over QUIC or TCP, desktop RTX plus the Spark on the 27B.
4. Serving integration (done): `PipelineExecutor` lets the scheduler drive a `Pipeline`. With `[placement] micro_batch = N` (default 0, off) a prefill chunk is cut into micro-batches and each stage runs on its own thread, so stage k works on micro-batch i while stage k+1 works on i-1. `/metrics` carries `inference_pipeline_*` series per stage (busy seconds, calls, tokens) and for hops and micro-batches. The model detail layer strip draws the planner's stage bands from `[placement]` in the app settings. A pipelined server needs `[[placement.devices]]` and a plan with more than one stage; it runs without MTP and without the prefix cache, and logs a warning at startup naming each one that the spec had on.
5. Discovery and re-planning.

Slice 3 (stage server over TCP, measured on the 27B across the desktop and the Spark) is in `docs/specrt-stage-server.md`.

## Micro-batch overlap measurement (slice 4)

Qwen3-0.6B bf16, 2 stages of 14 layers on one RTX PRO 6000, direct hop (host staged inside a device, since both stages share the GPU), prefill of one sequence, 3 timed runs after a warm-up, the three settings alternating. Test: `micro_batch_overlap_timing_on_qwen3_0_6b` in `crates/specrt/tests/pipeline.rs`.

| prompt | micro_batch 0 | 256 | 128 |
|---|---|---|---|
| 1024 tokens | 30.9 ms | 52.9 ms | 73.2 ms |
| 4096 tokens | 121.1 ms | 217.8 ms | 343.6 ms |

The overlap does not pay on one GPU: both stages compete for the same SMs and memory bandwidth, so there is nothing to hide, and each micro-batch adds a hop and stream syncs. It is 1.7-2.4x slower at 256 and 2.4-2.8x at 128, so the default is 0. The setting exists for stages on separate devices or hosts, where it is untested here (needs a second GPU or the Spark).

## Real server check

`inference-bench-server` with `[placement]` splitting Qwen3-0.6B into stages 0..14 and 14..28 (two planner devices on CUDA ordinal 0, `ordinal = 0` in `[[placement.devices]]`), `kv_cache_gb = 1.0`, `micro_batch = 32`. Four concurrent greedy chat completions (one with a 146 token prompt) gave text identical to a non-pipelined server. `/metrics` showed `inference_pipeline_stages 2`, 56 calls and 374 tokens per stage, 58 hops, 1531904 hop bytes and 4 micro-batches in 2 overlapped prefills. The non-pipelined server exports no `inference_pipeline_*` series.

## Tensor parallel groups in the planner

A `[[placement.devices]]` entry may set `host = "name"`. Neighbouring devices with the same host can form a tensor parallel group: the planner tries every tp that divides the host's run of GPUs and the q, kv and delta-net head counts of the spec. Per-layer bytes and FLOPs divide by tp, weights, KV and state are split per rank, and each layer adds two allreduces of `slots x hidden x 2` bytes over the peer link (`[placement.devices.peer]`, default the device `link`). Between hosts the plan stays a pipeline.

`inference-bench plan --probe` measures peer copies between local GPUs (P2P, or host staged when P2P is off) and the round trip to devices with an `address`, and uses them unless `peer` is declared. Without `--objective` it prints both objectives; each stage shows `host / GPU 0+1 (tp 2)`, memory, step time, allreduce share and what bounds it. `Placement` and `Stage` are `Serialize` for the app's cluster preview. The serving path refuses a plan with tp above 1 until the engine can run one.

## App cluster check (issue #657)

CPU coverage: `cargo test -p inference-bench-specrt --test discovery --test host`, `cargo test -p inference-bench-core --lib cluster`, `cargo test -p inference-bench-ui`.

GPU end to end (needs a free GPU, run under `flock /tmp/gpu27.lock`): two app instances on this machine with different data folders, sharing on in the second, Qwen3-0.6B split across both over loopback, text identical to a single runtime.

```bash
flock /tmp/gpu27.lock sh -c '
A=$(mktemp -d); B=$(mktemp -d)
echo "{\"cluster\":{\"share\":true}}" > $B/settings.json
inference-bench-ui $B & PB=$!
# in the first app: Hardware tab, wait for the peer, enter the code shown in the second, tick the peer, pick Qwen3-0.6B, split 14/14, Load
inference-bench-ui $A
kill $PB'
```

Then ask both the cluster and a single-runtime load of the same model the same greedy prompt (`temperature=0`, same `max_tokens`) and diff the text.

## One plan for the load dialog and the engine (issue #776)

The app load dialog and the engine now share one plan. `core::cluster::Planned` is built once from the dialog's devices (free memory per GPU times 0.9, in HF order), the dialog's weight bytes, KV pool and slots, through `PlacementConfig { weight_bytes, kv_pool_bytes, .. }`. The same `PlacementConfig` (with the stage ranges filled in) goes into the `ClusterSlot`, and `ServeOptions::plan` re-places it with the pool size from `kv_cache_bytes`, so the engine's per-stage bytes equal what the dialog showed. The KV pool is a total across GPUs: each stage allocates `kv_blocks` blocks of its own layers only (`Spec::stage` narrows `kv_f16_layers` to the stage), so the per-stage bytes sum to the pool and none holds all of it.

A placed load never falls back to one GPU: `engine::open` returns the stage error (`stage 1 (layers 36..64) on GPU 1: ... out of memory`). The dialog says "will span N GPUs" when the memory it sizes is more than the largest GPU and blocks the load when the planner says it does not fit.

Real check (needs both GPUs idle): `crates/specrt/scripts/gate776_two_gpu.sh`, defaults are the user's settings (27B NVFP4, fp8 KV, 21 seqs, ctx 262144, 95 GB KV pool). It checks both cards are empty, prints the plan, loads through `[placement]`, and reports VRAM per GPU and `inference_running_seqs` under 21 parallel requests.

## Profile-driven placement (issue #911)

`specrt::solve::solve(PlacementRequest { mode, target, pins, devices, spec })` returns a `Solved`: the one plan every load uses. `mode` is the profile (`MaxPerformance` or `LowMemory`), `target` is a hard constraint (context, memory budget, slots), `pins` are the user's overrides from the load dialog or server.toml, and `Profile::values()` are the defaults the search starts from. If nothing meets the target, `solve` returns the closest plan and names the miss in `Solved::shortfall`.

`Solved` is the load contract. It holds the placement stages, `tp`, KV dtype and pool bytes, slots, context, MTP and the sparse option map. `ServeOptions::from_solved` turns it into serve options and `Core::start` builds from it with no `PlacementConfig`. The engine does not re-plan and has no fallback: if it cannot honour the plan it fails with the stage error. The plan is logged at load as `solved placement`, with the full JSON at debug level.

Mode preferences:

- Max performance: tensor parallel across matched GPUs on one host even when one GPU would fit, because that is faster. When the model does not fit, a pipeline. On mixed GPUs, a pipeline or the best subset by the throughput cost model. f16 KV and MTP when they fit, then fp8 KV, then MTP off. Slots are the most the pool serves with typical requests.
- Low memory: the smallest GPU the model fits on (bandwidth breaks ties), fp8 KV, two slots, MTP off. It spans GPUs only when nothing fits alone.

Entry points:

- The load dialog solves on every change and shows the strategy, the reasons and any shortfall. Moving the slots or KV controls pins them. Loading hands the engine exactly the `Solved` the dialog showed (`EngineChoice.solved`).
- server.toml and CLI loads call `solve::for_options` from `start_engine`. `[placement] devices` without stages is solved for max performance with slots, tp, pool, KV dtype and MTP as pins. Explicit `[[placement.stages]]` become a pinned `Solved`. With no placement and `tp = 1` there is nothing to solve.
- `inference-bench plan` prints the max performance and low memory `Solved` after the stage tables.

Real check on the desktop pair (2 x RTX PRO 6000, 27B NVFP4, context 32768), `crates/specrt/tests/solved_load.rs`, run with `SOLVE_MODE=max` or `low`, `HF_HUB_OFFLINE=1`, `--ignored`:

| mode | solved | engine | GPU memory |
|---|---|---|---|
| max | TP 2 across 2, f16 KV 68 GiB pool, 32 slots, MTP on | `tensor parallel serving layout`, `max_num_seqs=32`, `kv_blocks=65536` | 53 GB and 51 GB |
| low | one GPU, fp8 KV 1.77 GiB pool, 2 slots, MTP off | `serving layout`, `max_num_seqs=2`, `kv_blocks=2400` | 28 GB and 0 |

## RDMA transport (issue #660)

Config: `[placement.remote] transport = "tcp" | "rdma" | "auto"` (default `tcp`), optional `rdma_library` (default `libibverbs.so.1`). Both the stage (`serve`) and the coordinator read it.

How it works (`crates/specrt/src/rdma.rs`, `wire.rs`):
- libibverbs is loaded with dlopen, so the binary runs where rdma-core is absent. `auto` then uses TCP, `rdma` fails with the load error.
- The TCP connection stays the control channel. It does the HMAC handshake and hello as before, then the client sends a `Message::Rdma` (tag 13: qpn, psn, gid, buffer address, rkey). The server answers with its own, or with an error (the client falls back to TCP under `auto`). RC queue pairs are brought up from this exchange over the already authenticated TCP link, not through rdma_cm.
- After the upgrade every frame is the same encoded `Message` bytes, sent with RDMA write with immediate into a ring of 8 registered 64 KiB receive slots on the peer. The immediate carries the chunk length and a last-chunk flag. The sender writes chunk i into slot i mod 8 and holds 8 credits; the receiver returns one credit (a zero-length write with a credit flag) after copying a slot out, so up to 8 chunks are in flight. TCP stays open and is peeked to detect a dead peer, so it fails in about 20 ms instead of hanging.
- Queue pairs are set up from endpoint info (qpn, psn, gid, buffer address, rkey) exchanged over the already authenticated TCP connection instead of rdma_cm. That avoids a second port and a second authentication path, since the HMAC handshake already covers the peer.
- The GID is picked from an active Ethernet RoCE port, preferring the one equal to the connection's local IPv4 (RoCE v2 mapped GID).
- Stage frames pass through pinned host ring buffers. GPU memory registers fine through dma-buf (below), but the stage handlers still hand over host byte vectors, so the engine does not use GPU-direct hops yet.

Measured on the desktop (ConnectX-4 to its own address 10.100.50.1, echo stage, 8 chunks in flight; three runs, TCP varies 1.4 to 3.6 ms at 4 MB):

| payload | TCP loopback | RDMA |
|---|---|---|
| 20 KB round trip | 10-14 us | 18 us |
| 4 MB round trip | 1.4-3.6 ms | 1.8-3.0 ms |
| 64 MB round trip | 141-158 ms | 124-153 ms |

The ring took 64 MB from 149 ms (one chunk in flight) to 124-153 ms across 13 runs; against TCP loopback in the same runs it is level to slightly ahead, with run-to-run noise as large as the gap. The cable run is the real comparison. The 20 KB decode hop stays slower than loopback, as expected: it pays the completion polling and a copy through the ring.

GPU memory over dma-buf (`crates/specrt/tests/rdma_dmabuf.rs`, ignored, run under `gpu-run 1`): `cuMemGetHandleForAddressRange(..., DMA_BUF_FD)` on a `cuMemAlloc` buffer plus `ibv_reg_dmabuf_mr` works on GPU1 (RTX PRO 6000, `CUDA_VISIBLE_DEVICES=1` under `gpu-run 1`; the tests skip on GPU0 unless `SPECRT_DMABUF_ALLOW_GPU0=1`) with the open driver and the ConnectX-4, no nvidia_peermem needed. Single RDMA write between two GPU buffers on the same card through the NIC loopback, against staging through a 1 MiB pinned host buffer (copy out, write, copy in):

| size | GPU direct | pinned host staging |
|---|---|---|
| 20 KB | 7.6 us | 17.4 us |
| 4 MB | 653 us (6.4 GB/s) | 1.21 ms (3.5 GB/s) |
| 64 MB | 10.3 ms (6.5 GB/s) | 19.4 ms (3.5 GB/s) |

GPU-direct is about twice as fast. The stage handlers now use it behind `[placement.remote] gpu_direct = true`; see `docs/specrt-gpu-direct.md`.

These are same-host numbers. TCP here is kernel loopback and does not touch the NIC, so they say nothing about the cable. The desktop to Spark numbers need the supervisor window: run `SPECRT_RDMA_TEST_IP=<ip> cargo test -p inference-bench-specrt --test rdma hop_latency_rdma_vs_tcp -- --ignored --nocapture` with a stage on the far end.

Host requirements:
- `ulimit -l`: each connection pins a send ring and a receive ring of 8 x 64 KiB (2 MiB together), so the default 8 MB holds about 4 connections in one process (the tests serialise for this reason). Raise it in `/etc/security/limits.d` only if larger rings are added.
- PFC/ECN: RoCE is only lossless with priority flow control. On the direct cable set PFC on the RoCE priority (`mlnx_qos --pfc 0,0,0,1,0,0,0,0` or `dcb pfc`) and DSCP trust on both ends. Without it, drops show up as RC retries and then a failed completion, which is reported as a lost stage. Not configured here; the user must approve it.

Cross-host allreduce: not built. The desktop has 1.8 TB/s of memory bandwidth and the Spark 273 GB/s, so TP across them runs at the Spark's pace and every layer adds two network round trips. Pipeline pays one hop per token, so pipeline is the decision. Revisit only if the planner finds a model that fits neither card alone and is compute bound.

Tests: `crates/specrt/tests/rdma.rs` skips with a message when there is no RoCE device or libibverbs. It covers round trip identical to TCP from 0 B to 5 MB, auto, wrong key, dead peer, library-absent fallback and a TCP-only server.
