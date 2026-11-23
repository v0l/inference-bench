# GPU-direct stage hops (issue #1033)

Config: `[placement.remote] gpu_direct = true` (default false), next to `transport`. It only has an effect when `transport` is `rdma` or `auto` and both ends can export device memory as dma-buf and register it with `ibv_reg_dmabuf_mr`. Everything else uses the existing frames over TCP or the pinned host ring, unchanged and bit-identical.

## How a served chain uses it

1. `Chain::connect_assigned_with` connects and assigns the stages as before, then calls `try_direct(link)`. It does nothing unless `gpu_direct` is on, the transport is not `tcp` and there are at least two stages.
2. For each hop i to i+1 the coordinator sends `DirectOpen { next: true }` to stage i and `DirectOpen { next: false }` to stage i+1. Each stage allocates its two device buffers once (outbox and landing, sized to its activation buffer), exports them as dma-buf, registers them, and answers `DirectInfo` (qpn, psn, gid, landing address, rkey, capacity). The coordinator relays each side's info to the other with `DirectConnect`. This all travels over the authenticated control connection.
3. `DirectMode { enabled: true }` is sent to every stage. A stage with a next link now answers a hidden-state forward with `Pushed { rows, compute_us }` instead of `Output { data }`: it copies its rows device to device into the outbox and posts one RDMA write with immediate (imm = byte count) into the next stage's landing buffer. The coordinator turns `Pushed` into `Input::Direct { rows }` for the next call. The next stage waits for the immediate (checking the byte count equals `rows * hidden * 4`), flushes GPU-direct writes, and copies landing to its activation rows with `import_hidden_device`. No hidden state touches host memory or the coordinator.
4. Micro-batched pipelined prefill sends several chunks in flight, which one landing buffer cannot hold, so `forward_pipelined` switches direct mode off for its duration and back on after.

`HiddenView` (in `wire.rs`) is the receiver's view of `Input`: `Host { rows, data }` for byte frames or `Device { rows }` for a payload already in the landing buffer.

## Authentication of the payload

The TCP control connection is authenticated with the pre-shared key exactly as before (HMAC over a server nonce, hello with protocol version). The direct queue pairs are set up only from endpoint info sent over that authenticated connection, so a peer that fails the handshake never learns a rkey, and the landing buffer cannot be written by anyone who has not passed the handshake (RC queue pairs only accept packets from the QP they were connected to, with the agreed psn).

The payload itself is not hashed on the GPU path: hashing 4 MB per hop on the CPU would need the host copy this path exists to avoid. What is authenticated is the header: the control frame (slot, position, rows) travels over the authenticated channel and states the expected size, the immediate carries the length, and the receiver rejects a mismatch before it reads the landing buffer. This matches what the host ring already guaranteed (integrity from RC and the authenticated QP, no per-chunk MAC). A per-hop digest on the receiving GPU is left out until a threat model asks for it.

## Fallback

Each case below ends in the unchanged byte frames, and each has a test in `crates/specrt/tests/direct_fallback.rs` (no GPU needed):

- `gpu_direct` off: no direct message is ever sent.
- TCP transport: ignored the same way.
- A stage that refuses the offer (gpu-direct not enabled on it, no libibverbs, no `ibv_reg_dmabuf_mr`, no CUDA engine, dma-buf export or registration failure): the coordinator resets direct mode on every stage and the chain reports `is_direct() == false`; the token is identical.
- An unloaded stage refuses `Input::Direct`.

## Tests

- `tests/direct_fallback.rs`: the fallbacks above.
- `tests/wire.rs`: codec round trip of the new messages and `Input::Direct`.
- `tests/direct_chain.rs` (ignored, `~/.agents/bin/gpu-run 1 --`, loopback on the 10.100.50.1 RoCE address):
  - `a_served_two_stage_chain_matches_one_stage_with_direct_hops_on_and_off`: two `RuntimeStage` servers of the tiny fixture on one GPU; tokens equal one runtime over the rdma ring, over direct hops, with direct switched off and on again on the same chain.
  - `pinned_ring_and_gpu_direct_hops_timed_in_one_run_at_20kb_and_4mb`: both hops timed in the same run.
- `tests/direct.rs` (ignored): an in-process two-runtime pipeline and raw link timing.

## Results (GPU1, loopback on the card's own address, tiny-qwen3.5)

Tokens: identical to one stage for rdma ring, direct, direct off and direct on again.

Same run, same buffers (4 MB device buffers, the ring path includes the device to host copy before and host to device copy after, as the stage handlers do):

| hop | gpu-direct | pinned ring through StageClient |
|---|---|---|
| 20 KB | 7.8 us (2.6 GB/s) | 23.7 us (0.86 GB/s) |
| 4 MB | 653 us (6.42 GB/s) | 1.46 ms (2.86 GB/s) |

In the served tiny chain (1 KB hidden decode hop, 128 KB prefill chunk) direct is level with the ring (decode first-stage round trip 107 us direct against 87 us ring, prefill chunk 561 us against 646 us): at 256 wide the hop is fixed-cost bound, and direct adds a flush and a device copy per hop. The gain shows at real hidden sizes.

## Remaining

- Spark run: desktop to Spark #544 split with `[placement.remote] transport = "rdma"` and `gpu_direct = true`, 8 of 8 identical texts, hop timings across the cable. Needs the Spark.
