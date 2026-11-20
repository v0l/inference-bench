# Tensor parallel inside one machine (#658)

Our own tensor-parallel mode, with no NCCL. `serve.tp = N` splits one model across N GPUs of one host. Each rank holds a slice of every layer, by attention head and by MLP column, and a kir allreduce joins the ranks twice per layer. The spec declares how each weight splits, so nothing is written per architecture.

## State

| Piece | State |
| --- | --- |
| Shard declarations in `graphs/decoder.toml` | done, `shard = ...` on every `linear` node |
| Per-rank cut table (`tp::RankCut`) and loader slicing (`Checkpoint::for_rank`) | done, CPU tests on the tiny 27B fixture and on the real 27B NVFP4 checkpoint |
| Allreduce kir family (`kir::allreduce`, `kir::cuda::emit_allreduce`) | done, built from `cuda/allreduce/prim.rs` builders, goldens, nvcc builds for sm_80 and sm_120 |
| Per-rank communicator (`allreduce::Fabric`, `Comm`) | done, virtual ranks on one GPU and on two GPUs |
| Plan rewrite: `Partial` buffer, `AllReduce`, `AllReduceNorm` ops (`Plan::for_tp`) | done |
| Rank runtime (`Runtime::load_rank`) and executor, one thread per rank (`tp_exec::TpExecutor`) | done |
| MTP under tp | done, the draft layer is sharded like a decoder layer, the draft head is replicated |
| `serve.tp` in the options schema and the server config | done, default 1 |
| Load dialog choice | left to #776, which owns `load.rs` |
| MoE | two splits (#798), chosen by `tp::plan_moe` with the reason; `[moe] tp_split = "auto" / "experts" / "columns"` overrides. `columns`: every expert cut like the dense MLP. `experts`: rank r owns experts r*E/tp.. (the shared expert lives on rank 0), every rank routes all tokens, `moe_route` writes idx -1 and coef 0 for experts it does not own, so the rank's `moe_combine` writes a zero contribution into `Buf::Partial`. Both end in `Op::AllReduce`. Auto picks columns when each rank keeps at least 256 columns per expert, else experts; it falls back to whichever divides. Virtual-rank tests pass on the Spark for both |
| 27B TP=2 against TP=1 numbers | measured 20 Nov, see Measurements: batch 1 decode 1.45x, short of the 1.6x target |

## Sharding in the spec

Each `linear` node in `graphs/decoder.toml` has a `shard`:

| shard | meaning | weights |
| --- | --- | --- |
| `column` | split the output rows by head | `qkv`, `gate_up`, `lin_in`, `lin_ba`, `mtp_qkv`, `mtp_gate_up` |
| `row` | split the input columns; the result is a partial that needs an allreduce | `o`, `down`, `lin_out`, `mtp_o`, `mtp_down` |
| `replicate` | every rank holds the whole matrix | `lm_head`, `mtp_fc`, `mtp_head` |

`Graph::shards()` returns the map and refuses a linear node without a shard, or a weight declared two ways. The embedding, norms and the head stay whole on every rank, so every rank computes the same logits and samples the same token. `TpExecutor` checks that the ranks agree on every token and stops with an error if they ever differ.

`tp::RankCut::new(spec, graph, tp, rank)` turns the shards into cuts of checkpoint tensors:

- `q_proj` (with the output gate interleaved per head), `k_proj` and `v_proj` cut by head. A fused `qkv_proj` (phi3) cuts its q, k and v segments separately. A fused `gate_up_proj` cuts gate and up separately.
- The delta-net `in_proj_qkv` and `conv1d` cut the q, k and v segments by key and value head. `in_proj_z`, `in_proj_a`, `in_proj_b`, `A_log` and `dt_bias` cut by value head. `out_proj` cuts its input columns.
- Block formats follow their weight: NVFP4 `weight_packed` (2 values per byte) and the e4m3 group scales (16 columns each), FP8 per-channel scales (by row), FP8 128-block `weight_scale_inv` (the cut must sit on a block edge or loading fails with a message). Global scales are left whole.

`Checkpoint::for_rank` applies the cuts inside `view()`, so every existing loader path (`attention`, `gated_delta_net`, `fp4`, `raw`, MTP) reads only its slice with no per-format code. A contiguous row cut is a borrowed slice of the mmap. Column cuts and multi-segment cuts are copied once per load group and freed with it.

`tp::rank_spec(spec, tp)` gives the per-rank spec: heads, KV heads, delta-net key and value heads and the intermediate size divided by tp. The kernels are generated from that spec as for any model, so a rank runs the same JIT kernels as a smaller model.

`tp::supports` refuses, with the reason: head counts tp does not divide, expert widths tp does not divide (a multiple of 256 per rank for NVFP4 experts, of 128 for FP8 blocks), GGUF weights, sandwich or scaled residual norms (gemma3, olmo2, granite), full-width qk norms, tiled delta-net value heads, and per-rank widths that are not multiples of 16.

## Allreduce family

`kir::allreduce::AllReduceKernel { ranks, dtype, mode, blocks }`. Sums accumulate in f32 in rank order on every rank, so all ranks hold bit-identical results.

| mode | use | shape |
| --- | --- | --- |
| `OneShot` | messages up to 1 MiB (decode) | each rank copies its partial to its own exchange buffer, flag barrier, each rank sums every buffer |
| `TwoShot` | larger (prefill) | reduce-scatter (each rank sums one chunk), barrier, allgather |
| `AddRmsNorm` | the allreduce after `o` | one-shot sum, add into the residual `x`, RMSNorm, write the bf16 hi and lo planes the next GEMM reads |

Both plain modes take `add` (sum into the output instead of overwriting) and `clear` (zero the partial after copying it out, so the next row-parallel GEMM can accumulate into it). The engine uses `add = clear = 1`: the partial goes into `x`, which every rank holds whole.

Barriers are per block. Each block stores an epoch into its peers' flag words with `st.release.sys` after a `__threadfence_system`, then spins on its own with `ld.acquire.sys`. The epoch lives in device memory, so the kernels need no host step and a captured CUDA graph replays them. A barrier that waits about 10 seconds traps instead of hanging forever. Buffers are read with `ld.cg`, so a sum never sees a stale L1 line.

`Fabric` owns the exchange buffers and the pointer table:

- `Transport::Peer`: device buffers with peer access enabled between contexts.
- `Transport::SameDevice`: virtual ranks, every rank on one GPU, through device memory. Used for tests.
- `Transport::HostPinned`: pinned host memory mapped into every context. Used when `cuDeviceCanAccessPeer` is false either way, or when the GPUs sit behind a translating IOMMU (the sysfs `iommu_group/type` is `DMA` or `DMA-FQ`). On this desktop before the `iommu=pt` reboot, the driver reported peer access as OK but kernel stores to the peer never landed (copies read zeros, loads faulted), which is the same fault that hung NCCL with P2P on. The check reads sysfs instead of trusting `canAccessPeer`.

Whenever the transport would be `Peer` or `SameDevice`, `Fabric::new` then runs a timed probe for every ordered GPU pair (`allreduce::peer_writes_land`). A one-thread kernel on the target GPU spins on a flag in its own memory against `%globaltimer`. A kernel on the source GPU stores a magic word there with `st.release.sys`, and the target reports whether it arrived within `PROBE_TIMEOUT` (200 ms). One missed write sends the whole group through pinned host memory, with a warning naming the pair. A machine where peer access fails some other way therefore falls back at init in about 200 ms, instead of trapping at the first barrier. `the_peer_probe_waits_on_a_system_flag_with_a_timer` pins the probe source, and `a_peer_write_that_never_lands_falls_back_to_host_memory_within_the_timeout` drops the store on virtual ranks and checks the fallback (213 ms on the GB10).

## Runtime

`TpExecutor::load(spec, layout, devices)` builds one `Fabric`, then starts one thread per rank. Each thread loads `Runtime::load_rank` (its sliced weights, its per-rank kernels, its KV and recurrent-state pools, which are split by head because they are sized from the rank spec) and keeps its own `Stepper`. The executor implements the scheduler's `Executor` trait: every call is sent to all ranks, each rank runs the same plan on its own stream, and the allreduces inside the plan keep them in lockstep. There is one scheduler, and block tables, state slots, masks and snapshots go to every rank. Prefix-cache state snapshots are stacked per rank (`StateSnapshot::stacked` and `unstack`).

`Plan::for_tp` rewrites each plan: a `row` linear that wrote `x` writes `Partial` instead and is followed by `AllReduce`. When `[runtime] fuse` is on and the next op is the norm into the GEMM planes, the pair becomes one `AllReduceNorm`. Decode, prefill, verify and the MTP plans are all rewritten.

MTP: the draft layer is sharded like any layer (it is attention plus MLP), and the draft head is replicated. Each rank drafts, verifies and accepts the same tokens, and the executor checks they agree. A separate broadcast from rank 0 is not needed.

CUDA graphs: each rank captures its own graph per row count. Virtual ranks on one GPU run without graphs. Two ranks' graphs on one device can each fill the work queue, leaving one rank's barrier waiting on the other's kernel that never starts (seen on the GB10 as a hang in the graphed phi3 run). Separate GPUs do not share queues.

Not covered: the fused decode-plus-prefill step (`[runtime] fuse` stays on for the norm fusion, but the scheduler's fused step runs as decode then prefill on tp ranks), the pipeline stages of `[placement]` together with tp, vision towers (dropped from the rank spec).

## Tests

CPU:

```bash
cargo test -p inference-bench-kir --test allreduce
cargo test -p inference-bench-specrt --lib tp:: allreduce:: plan:: options::
cargo test -p inference-bench-specrt --test tp
```

`the_real_27b_nvfp4_checkpoint_cuts_cleanly_at_tp_2` checks every cut of the real 27B NVFP4 checkpoint (968 tensors per rank) when it is in the HF cache.

Virtual ranks on one GPU (the Spark, or a free desktop card with `CUDA_VISIBLE_DEVICES`):

```bash
GATE_SUITES="allreduce tp" GATE_MIN_AVAIL_GB=40 GATE_MEMORY_MAX=60G \
  CARGO_TARGET_DIR=~/ib-gates-target/658 crates/specrt/scripts/gate658.sh virtual
```

Spark GB10, 18 Nov:

```
test odd_sizes_do_not_leave_a_ragged_tail ... ok
test the_fused_add_rmsnorm_sums_exactly_and_normalises ... ok
test the_host_pinned_fallback_gives_the_same_sums ... ok
test virtual_ranks_sum_random_data_from_one_row_to_16k_rows ... ok
test result: ok. 4 passed; 0 failed

tiny-mistral tp2: logits max |d| 2.970e-3 (of 25.73), top-1 48/48, greedy streams identical 6/6
tiny-smollm3 tp2: logits max |d| 3.175e-3 (of 30.25), top-1 48/48, greedy streams identical 6/6
tiny-phi3 tp2: logits max |d| 2.909e-3 (of 24.97), top-1 48/48, greedy streams identical 6/6
tiny-phi3 tp4: logits max |d| 2.237e-3 (of 24.97), top-1 48/48, greedy streams identical 6/6
tiny-qwen3.5 tp2: logits max |d| 2.317e-4 (of 4.47), top-1 48/48, greedy streams identical 6/6
tiny-qwen3.5-mtp tp2 with mtp: greedy streams identical to tp1 without mtp 6/6
tiny-qwen3.5-27b tp2 with mtp: greedy streams identical to tp1 without mtp 6/6
tiny-qwen3.5-27b tp2 (host pinned): logits max |d| 2.580e-4 (of 4.74), top-1 48/48, greedy streams identical 6/6
tiny-qwen3.5-27b tp2: logits max |d| 2.580e-4 (of 4.74), top-1 48/48, greedy streams identical 6/6
test result: ok. 6 passed; 0 failed
```

Each stream is 64 greedy tokens. The logits are not bit-identical to TP=1: a row-parallel GEMM sums half the K dimension per rank and the allreduce adds the halves, which rounds differently from one K loop. The drift is below 1.2e-4 of the logit range, and top-1 and every greedy token match. The tiny qwen3.5 fixture has 2 key heads, so it runs at tp 2 only; phi3 runs at tp 4.

Two GPUs, both free (about 25 minutes: 2 min build, 1 min P2P and allreduce, 4 x 27B loads at about 2 min each plus the runs, 8 min for vLLM to start and run):

```bash
crates/specrt/scripts/pair658.sh                       # p2p allreduce tp27b vllm
PAIR_STEPS="p2p allreduce" crates/specrt/scripts/pair658.sh
```

It refuses to start while any compute app is on either GPU or anything answers on :8001. It writes results to `/tmp/pair658`. The vLLM step starts `vllm serve ... --tensor-parallel-size 2` read-only from `~/.venv-vllm` under `MemoryMax=24G` and stops it on exit.

## Measurements

Desktop, 2x RTX PRO 6000 Max-Q, PCIe 5 through the host bridge (`nvidia-smi topo -m`: NODE), before the `iommu=pt` reboot (IOMMU translating, `DMA-FQ`):

| measurement | value |
| --- | --- |
| `cudaDeviceCanAccessPeer` | 1 both ways |
| peer copy GB/s (256 MiB), p2p on and off | 43.6 / 42.4 on, 44.2 / 44.3 off |
| bidirectional peer copy | 84.8 GB/s |
| kernel stores and loads to the peer | do not land (loads return garbage or fault) |
| allreduce bf16, 5120 wide, host pinned: 1 / 8 / 64 rows | 31.7 / 34.8 / 74.0 us |
| allreduce correctness on the pair (host pinned, 1 to 16k rows, f32 and bf16) | bit-exact |

After the reboot with `iommu=pt`, NCCL reaches about 35 GB/s busbw over P2P (supervisor's measurement). Measured 20 Nov 2026 with `pair658.sh` (iommu group type `identity`), results in `/tmp/pair658`:

| measurement | value |
| --- | --- |
| peer write probe | lands both ways, auto transport is Peer |
| peer copy GB/s (256 MiB) 0 to 1 / 1 to 0, p2p off | 43.5 / 43.6 |
| peer copy GB/s, p2p on | 53.9 / 54.4 |
| bidirectional peer copy | 105.3 GB/s |
| 4 KiB peer copy | 1.2 GB/s |
| flag ping-pong round trip | 150.9 us |
| allreduce bf16 5120 wide, peer: 1 / 8 / 64 rows | 7.7 / 11.3 / 37.7 us |
| allreduce bf16 5120 wide, host pinned: 1 / 8 / 64 rows | 33.7 / 36.2 / 75 us |
| `two_gpu_peer_group_sums_exactly` | bit-exact |
| 27B NVFP4 tp2 vs tp1, 256 positions | top-1 256/256, mean KL 1.0e-8, max KL 4.5e-7 |

27B NVFP4 (`qwen3.8-27b-nvfp4-exact.toml`, `tpbench`, 512 token prompt, 128 decode, CUDA graphs on), after the fused norm fix below:

| config | prefill tok/s | decode tok/s | per stream | ms per step |
| --- | --- | --- | --- | --- |
| tp1 batch 1 | 1668 | 117.3 | 117.3 | 8.52 |
| tp2 batch 1 | 2476 | 169.9 | 169.9 | 5.89 |
| tp1 batch 8 | 1662 | 465.3 | 58.2 | 17.19 |
| tp2 batch 8 | 2567 | 686.9 | 85.9 | 11.65 |
| tp1, 4096 prompt | 1852 | 53.0 | 53.0 | 18.85 |
| tp2, 4096 prompt | 3050 | 81.7 | 81.7 | 12.24 |

tp2 against tp1: 1.45x at batch 1 (target 1.6x, not met), 1.48x at batch 8, 1.65x on 4k prefill. Before the fix tp2 was slower than tp1 (88.0 against 119.9 tok/s at batch 1).

The cause, from nsys: `ar_add_rmsnorm_f32` took 68% of GPU time at about 270 us a call. Its residual loop read each rank's partial in a dependent chain per quad, so every load paid the full peer round trip (about 1.5 us over PCIe) one after another. The kernel now issues all `AR_RANKS` loads for a row before any arithmetic. Same bit-exact sums. The remaining gap to 1.6x is not profiled yet.

vLLM 0.x tp2 on the same pair (`vllm serve --tensor-parallel-size 2`, fp8 KV, prefix caching, max-model-len 8192, no MTP and none of the other flags of `~/qwen3.8-27B-NVFP4-tp2.sh`; run under `MemoryMax=40G` because `pair658.sh`'s 24G cap OOM-killed the engine): batch 1 decode 104.5 tok/s, batch 8 669.7 tok/s total (83.7 per stream), 4k prefill 11994 tok/s (342 ms). Our tp2 is ahead of it at batch 1 and 8 on decode and about 4x behind on prefill.

Vision on real TP=2 (#910): the server with `tp = 2` loaded the 27B on both cards, an image request (`vision/shapes.png`, 192 vision tokens, greedy, 60 tokens) returned text identical to the tp = 1 server.

MoE (#798) on the real Qwen3.5-122B-A10B NVFP4 (76 GB, fits in 2x96 GB, tp1 fits one card): `tp2 vs tp1` top-1 436/512, mean KL 1.76e-2, max KL 0.13 (fp4 activation quantisation amplifies the different sum order; the 122B against vLLM is 0.0106). Decode batch 1: 159.2 tok/s tp1, 229.6 tok/s tp2 (1.44x). Batch 8: 475.2 against 789.5 tok/s (1.66x). Prefill 512: 4011 against 5203 tok/s.

#811 A/B on the 122B, batch 8, 128 prompt, 128 new, three runs, two rounds, master before the merge against after it: decode 297.1 / 299.9 / 291.9 / 294.4 / 296.6 / 302.5 tok/s before, 298.6 / 292.7 / 294.0 / 295.3 / 292.5 / 288.0 after. Both about 295 tok/s, the PTX change from the port costs nothing measurable (under 1%).
