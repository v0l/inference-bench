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
