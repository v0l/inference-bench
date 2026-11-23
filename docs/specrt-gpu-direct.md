# GPU-direct stage hops (issue #1033)

Config: `[placement.remote] gpu_direct = true` (default false), next to `transport`. It only has an effect when `transport` is `rdma` or `auto` and both ends can export device memory as dma-buf and register it with `ibv_reg_dmabuf_mr`. Everything else uses the existing frames over TCP or the pinned host ring, unchanged and bit-identical.

## Pieces

- `direct::DirectRegion`: a `cuMemAlloc` buffer (rounded up to 2 MiB) exported once with `cuMemGetHandleForAddressRange(DMA_BUF_FD)`. A stage owns two: an outbox (its output hidden state is copied device to device into it) and a landing buffer (the previous stage writes into it, and the stage copies from it into its input rows). Both are registered once, at link set-up, never per hop.
- `direct::DirectLink`: one RC queue pair with the outbox registered as the local source and the landing buffer registered for remote write. `push(bytes)` posts an RDMA write with immediate (imm = byte count) straight from GPU memory. `wait(bytes)` completes when the immediate arrives, which means the payload is in the peer's GPU memory.
- `Runtime::export_hidden_device` and `import_hidden_device` move hidden rows between the stage's activation buffer and a device pointer with no host copy.
- `HiddenView` is the device-side counterpart of `Input::Hidden { data }`: host bytes, or a registered device range plus length.

## Authentication of the payload

The TCP control connection is authenticated with the pre-shared key exactly as before (HMAC over a server nonce, hello with protocol version). The direct queue pair is set up only from endpoint info (qpn, psn, gid, address, rkey) sent as `Message::Direct` over that authenticated connection, so a peer that fails the handshake never learns a rkey, and the landing buffer cannot be written by anyone who has not passed the handshake (RC queue pairs only accept packets from the QP they were connected to, with the agreed psn).

The payload itself is not hashed on the GPU path: hashing 4 MB per hop on the CPU would need the host copy this path exists to avoid. What is authenticated is the header: the immediate carries the length, the control frame (slot, position, rows, versioning) still travels over the authenticated channel and states the expected byte count, and the receiver checks the immediate equals that count before it reads the landing buffer. This matches what the host ring already guaranteed (integrity comes from RC and the authenticated QP, not a per-frame MAC; the ring does not MAC chunks either). A per-hop digest computed on the receiving GPU is left out until a threat model asks for it, since the host ring has no equivalent.

## Fallback

`gpu_direct` off, TCP transport, no libibverbs, no `ibv_reg_dmabuf_mr`, a device without `DMA_BUF_SUPPORTED`, or any export or registration failure: `DirectLink::offer` returns an error and the hop uses `Input::Hidden` bytes as before. Results are identical because the same f32 bytes move.

## Scope

Chain hops currently pass through the coordinator (`Chain::forward` takes the reply of stage i and sends it to stage i+1). A coordinator that does not hold the GPU cannot forward device memory, so the direct link is a stage to stage link. Wiring the coordinator to tell stage i to push to stage i+1 (and stage i+1 to expect the immediate instead of the `Hidden` payload) is the remaining step, listed in the issue, and the cross-host numbers need the Spark.

## Tests

`crates/specrt/tests/direct.rs` (ignored, run under `~/.agents/bin/gpu-run 1 --`): two runtimes of the tiny fixture on one GPU as two threads, hidden state passed host-bytes versus GPU-direct, tokens equal to one runtime; decode (20 KB scale) and 4 MB hops timed both ways.
