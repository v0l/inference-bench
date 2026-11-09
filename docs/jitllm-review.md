# jitllm review: what specrt can take from it

Issue #393. jitllm is at https://github.com/jitllm/jitllm (Go, Apache-2.0), read at commit `4de5fe20` and cloned to
`/tmp/jitllm`. Paths below are relative to that checkout. Nothing was built against a GPU. Two model-free gates
were run to check that the code described is real: `go test ./engine/grammar/` and
`go test ./engine/model/ -run TestLookupSearch` both pass.

Our target is Qwen3.x 27B (48 gated delta-net layers, 16 full-attention layers, one MTP layer, NVFP4 MLPs and FP8
projections) on Blackwell (the RTX PRO 6000 at sm_120 beside vLLM, and the GB10 at sm_121), served to agents through
crates/server.

## How far their numbers carry

Every number on their scoreboard (`docs/performance.md`, `docs/perf/current.md`) is jitllm against llama.cpp, vLLM
0.18.1 or MLX, on V100 (sm_70), T4, a Xeon E5-2680 v4 and an Apple M4, on GGUF k-quant weights. None of it is
Blackwell, NVFP4, FP8 or a gated-delta hybrid on CUDA, and the vLLM rows are a V100 where vLLM is weak. So none of
their ratios predicts a ratio for us. What does carry over:

- the **mechanisms**, which are model- and card-independent (admission, budgets, speculation control, parsers, grammar masks);
- their **negative results**, which are cheap lessons (a persistent decode block composed from per-op kernels is 11%
  slower on V100 and 4% on T4, `docs/engineering-history/gpu-kernels.md` "A persistent decode block");
- their **measurement discipline** (`AGENTS.md` RULE 1 at line 62, RULE 2 at line 112).

They publish no speedup figure for speculative decoding at all. `docs/perf/current.md` has no speculation row.

## Idea by idea

### 1. Kernels emitted at load for the model's shapes

**jitllm.** A small typed IR (`jit/gpu/ir/ir.go`, about thirty ops, shape constants baked, not passed: line 241) that
is lowered to PTX, SPIR-V, MSL and AMDGPU (`jit/gpu/ptx`, `spirv`, `msl`). Kernels per quant format and shape are
written against it (`jit/gpu/kernels/matvec.go`, `gemmint8.go`, `flash80.go`). The highest tensor-core path is sm_80
`mma.m16n8k16`/`m16n8k32` (`kernels/gemmint8.go:61`, `kernels/flash80.go:54`). There is no FP8, NVFP4 or sm_90+/sm_120
code anywhere in `jit/`.

**specrt.** Already shape-specialised at load. `codegen.rs` `Module::generate_kv` (line 632) bakes every dimension
from the spec into the CUDA templates as constants, picks kernel sets per weight format, and `jit.rs` compiles with
NVRTC and caches the PTX on disk under `sha256(source, arch, options)`. We already have native e4m3 and e2m1 block-scaled
mma on sm_120, which jitllm does not.

**What is worth taking** is not the emitter but what sits beside it: **measured per-shape choices at load,
persisted**. `jit/gpu/tier/split.go:18-39` times split widths against the real resident weights, refuses a measurement
with IQR/median above 0.10 (line 27) and keeps the table's answer, and stores the winner in a tune cache keyed by
device, driver and executable (`tier/choose.go:230` `lookupCache`, `WithTuneCache`). Our equivalent knobs are formulas
(`codegen.rs` `f4b_splits`, `f8b_splits`) and env vars (`codegen.rs:324` `env_usize`: `SPECRT_TINY_WN`,
`SPECRT_FP8_BK`, `SPECRT_BK`, `SPECRT_STAGES`), which also breaks the config-first rule.

**Win.** Unknown until measured; their own finding is that the fitted rule "missed the optimum on most shapes, even on
the card it was fitted on" (split.go:19). **Cost.** Medium: a timing harness in the runtime, a cache file, a spec
`[runtime]` field to force or disable. Overlaps #394's kernel cache and dispatch (#398, #399), so #408 covers only
the measured choice and stores it in #398's cache.

### 2. Int8 KV cache

**jitllm.** `engine/model/kvtype.go:9-39`: KV f32, f16 or q8_0 (llama.cpp's 32-element blocks). The q8 cache runs on
the **host tiers only**; a device holds float32 pages and a q8 history is widened on its way to the card
(`kvpage.go:775-790`, "a q8 cache resident on a device is not implemented"). q4_0 is refused.

**specrt.** One f16 plane, 64 KiB per token on the 27B (`docs/specrt-scheduler.md`). vLLM serves this model with
`--kv-cache-dtype fp8`. #324 already covers fp8, int4 and TurboQuant on the device with a quality harness.

**Verdict.** Nothing to take: their int8 KV is a CPU feature. No new issue; #324 stands.

### 3. Prefix caching across sessions

**jitllm.** Content-addressed pages: page n of a layer is stored under a cumulative hash of the tokens up to its end
(`kvpage.go:2504-2509`, LMCache's chunk hash, because RoPE is applied before the write). A namespace set by
`SetCacheKey` (line 1102) carries model, quant and KV width and doubles as the tenancy dial
(`"model/q4/f32/tenant-42"`), with the timing side channel stated. `geomDigest` (line 2516) hashes the layout into
every key so two models with equal page bytes cannot alias. A hybrid restores only at a point where its recurrent
summaries were stored (`kvCache.restore`, line 1142); restore points at chat-message boundaries are on their roadmap
and not built (`ROADMAP.md` section 26). Measured: 5.8x faster on a repeated 1181-token prompt (`ROADMAP.md` section 5,
host tier).

**specrt.** Not built; #325 is the issue and already asks for a hash chain, recurrent-state snapshots and a salt.

**Verdict.** No new issue. Three details from jitllm worth adding to #325 (left as a comment there): the geometry
digest in the key, the namespace as the one tenancy mechanism, and snapshot cuts on chat-template message boundaries
so a resumed turn and a cold prompt cut identically.

### 4. Speculative decoding: prompt lookup beside MTP

**jitllm.** One `Speculator` with two drafters (`engine/model/spec.go:97-125`): the model's MTP head, or prompt
lookup (`speclookup.go`): the tokens that followed the latest earlier occurrence of the last 2 to 4 tokens
(`appendLookup`, line 80), 5 drafts a round (line 33), nothing run to draft, and a round with no match is a plain
decode step. It is greedy only (`lookupCount`, line 48). Gated by `TestSpecLookupMatchesDecode` and a discriminating
gate that corrupts every fifth draft (`docs/engineering-history/model-correctness.md` "engine/model/speclookup.go").

**specrt.** MTP only. The verify path is drafter-agnostic: `Runtime::set_drafts` (`runtime/mtp.rs:192`) already feeds
arbitrary drafts and the tests use it. Our `spec_accept` (`templates/mtp.cu:70`) treats a draft as one-hot, as vLLM's
`NO_DRAFT_PROBS` does, so lookup drafts would also be exact under **sampling**, which jitllm does not have.

**Win.** Agent traffic copies a lot (tool outputs, code being edited, file contents). Lookup drafts are free to make
and can be longer than MTP's 2. Unknown size: it has to be measured as acceptance on real agent transcripts.
**Cost.** Small.

### 5. Draft count chosen from measured cost

**jitllm.** `Speculator.choose` (`spec.go:1006`): keeps the median verify time per row count, acceptance per draft
position with Laplace's rule, and picks the k that maximises expected tokens per second
(`1 + sum prod alpha_j` over verify + draft + catch-up time); re-tests a "no draft" choice on a doubling schedule and
retires speculation after 3 failed re-tests (line 1122). Depth at most 4 (line 1128).

**specrt.** Fixed `[mtp] draft_tokens` and a static `max_batch` above which speculation turns off. Measured on the 27B
(`docs/specrt-mtp.md`): k=2 is 1.36x at batch 1, 1.17x at batch 4 and **0.81x at batch 8** (verify leaves the batched
GEMV path past 16 rows). The right k and the crossover move with the verify kernels (#361, #375), the context and the
traffic, so a fixed number is always stale.

**Win.** Removes the 0.81x loss at batch 8 (at least back to 1.0x) and picks k=1 or 3 where they win. **Cost.**
Medium: timings per verify-row count in the scheduler, per-sequence acceptance, per-step k. Our kernels take k at
compile time today (`MTP_KERNELS`), so k varies by choosing among compiled k=1..3 variants or by padding.

### 6. Speculative round kept on the device

**jitllm.** `ROADMAP.md` section 28 (research): run accept, KV compaction and the next round's arguments as kernels so
the host's wait between submissions does not dominate. Not built.

**specrt.** This is our measured problem, not theirs: "the step is not graph-captured: every host-to-device index
upload synchronises the stream" (`docs/specrt-mtp.md`, Real 27B NVFP4; `runtime/mtp.rs:213`
`up_i32` calls `stream.synchronize()` after every upload). 2.43 tokens per step give only 1.36x.

**Win.** The gap between 1.36x and the acceptance length is the target: at plain decode 23 tok/s, 2.43 tokens per
step with a verify step costing 1.3 plain steps would be about 43 tok/s against 31. **Cost.** Medium to large: move
index building to device kernels, capture the round per batch size in a graph like `Runtime::decode` does.

### 7. Chunked prefill beside decodes: budget relative to the decode step

**jitllm.** `server/batchbudget.go:5-30`: a step's prompt budget is sized so the step costs at most `StepCost` decode
steps (default 4, line 58). The loop fits step time against prompt tokens carried as a line (intercept: a decode
step, slope: a prompt token), with decay 0.1, and doubles the budget while the slope is not yet resolved.

**specrt.** Budget = `itl_target` (100 ms) times the measured prefill rate (`docs/specrt-scheduler.md`). On the 27B
eight decoders alone take 95 ms, so a 100 ms absolute target leaves almost nothing: a 29k prompt prefilled at
220 tok/s beside 7 decoders against 550 to 1,300 alone, and "the 100 ms target cannot be met with any prefill at all".
A budget relative to the decode step scales with the batch and with the model.

**Win.** Prefill throughput beside decoders, the 133 s for a 29k prompt. **Cost.** Small: `sched.rs` budget
computation and two config fields.

### 8. KV-aware admission with a lookahead, and parking instead of recompute

**jitllm.** `server/batch.go:460` `admit`: a row is admitted when the card's KV room covers its prompt plus
`min(max_tokens, 512)` (`admitLookahead`, line 550), counting what admitted rows are still promised (`kvRoom`, line
581); the promise goes to the tier so scratch cannot take it. A full card **parks** sessions to the host
(`engine/model/park.go:29` `Park`, `:91` `Resume`: KV and recurrent state home, byte for byte), never recomputes.
Above fairness 0 (`server/batchfair.go:10-54`) decoders past a token quantum (256 at level 1 down to 8 at level 100)
are parked for waiting requests, prompt budget is shared among the newest prompts (`allot`, `batch.go:873`), and
priority is a head start in wait time. Measured on a V100 at 64 concurrent (their numbers, their engine): level 100
against 0, 0 errors against 144 of 256, TTFT p99 78 s against 575 s, at a cost of 1.7 s p99 inter-token gap.

**specrt.** Admission needs the prompt plus one block plus a 1% reserve (`sched.rs:416-425`). When the pool runs dry
the newest sequence is **preempted by recompute** (`preempt_latest`, line 378): its KV and 157 MB of delta-net state
are dropped and the whole prompt plus generated tokens prefilled again. On the 27B a 29k-token context costs 50 to
130 s to recompute; swapping it costs 1.8 GB of KV plus 157 MB of state over PCIe 5 (well under a second; on the GB10's
unified memory less).

**Win.** Under pressure: no multi-minute recompute stalls, and the lookahead stops admitting rows that will preempt
each other. Their fairness numbers come from a V100 that failed requests under eviction; they do not carry, but the
mechanism (swap first, time slices as an option) does. **Cost.** Medium: pinned host pool, a park and resume on the
runtime (block copies and one state copy), scheduler policy. Shares the pinned pool with #326 and #323.

### 9. Constrained decoding and tool calls

**jitllm.** `engine/grammar`: GBNF and a JSON Schema subset compiled to a grammar (`schema.go:76` `FromJSONSchema`;
unsupported keywords refused by name), a token mask from a byte trie over the vocabulary (`mask.go:47`
`IndexTokens`, `:205` `Mask`, allocation-free by test), compiled grammars cached per model with their allowed sets
(`server/grammar.go:38-40`). Tool calls are read in each family's own syntax, chosen from the chat template rather
than by trying parsers (`engine/model/toolsyntax.go:10-40`; Qwen3.5 and Qwen3-Coder are `ToolSyntaxQwenXML`, line
34, `<tool_call><function=NAME><parameter=KEY>`). `tool_choice` required or named is enforced by a grammar of the
model's own wrapper around the tool's schema (`engine/grammar/compose.go`, `server/tools.go:92`).

**specrt.** The server reads `model`, `messages` (role and text only), `max_tokens`, `temperature`, `top_p`, `seed`,
`stream`, `ignore_eos` (`crates/server/src/app.rs:358` `parse_params`). `tools`, `tool_choice`, tool-role messages,
`tool_calls` in assistant turns, `response_format`, `stop` and reasoning are silently dropped; the template always
renders with `enable_thinking = false` (`crates/specrt/src/chat.rs:61`). vLLM serves this model with the `qwen3_coder`
tool parser and the `qwen3` reasoning parser (`docs/qwen3.8-27b-vllm-baseline.md`), and the traffic we must carry is
agents. **Today specrt cannot serve an agent turn at all**, whatever its speed.

**Win.** The precondition for replacing vLLM. **Cost.** Two pieces: (a) the API surface and the Qwen XML parser
(medium, host-only), (b) grammar masks under batching with device sampling (medium to large: a mask buffer per row
fed to `sample_rows` and `spec_accept`).

### 10. Sampling that speculation and batching cannot change

**jitllm.** `ROADMAP.md` section 29 (not built): Gumbel-max sampling keyed by (seed, position, token), so a drafted
run emits exactly the undrafted sequence at any temperature.

**specrt.** Plain decode already samples this way (`templates/ops.cu:1632` `gumbel(seed, step, i)`, used by
`sample_rows`). But `spec_accept` (`templates/mtp.cu:70`) uses a uniform draw against p(draft) and a Gumbel race for
the recovered token, so speculative sampled output has the right distribution (the chi-square test) but is not the
plain sampled sequence. Accepting a draft iff it equals `argmax(logits / T + gumbel(seed, step + j, i))` accepts with
probability exactly p(draft), the same rate as now for one-hot drafts, and makes speculative sampling byte-identical
to plain sampling.

**Win.** Not speed: a byte-identity test at temperature, reproducible seeds across `max_batch` switches, and a simpler
kernel. **Cost.** Small.

### 11. `/v1/capabilities` and `doctor`

**jitllm.** `server/capabilities.go:17-38`: the route table the server mounts is the manifest, each request's fields
read off its struct, with tests holding the notes and refusals to the handlers in both directions
(`TestAPIManifestCoversEveryField`, `TestRefusalsAreRefused`). `cmd/jitllm/doctor.go:23-60`: inspects host, devices,
backends and a model file without loading it, exit code 0/1/2, each problem with a next action, `-json`.

**specrt.** No equivalent. **Win.** Agents and the UI can ask what is served instead of finding out by a silent drop
(see 9). **Cost.** Small to medium.

### 12. Zero allocation on a warm decode token

**jitllm.** `engine/model/decodealloc_test.go:30` `TestDecodeDoesNotAllocate` and siblings. The motive is Go's
collector: with page frames on the heap it took 42.5% of the CPU on Qwen3-Next-80B (`AGENTS.md` RULE 2f, line 165).

**specrt.** `runtime/step.rs:134-142` builds five small `Vec`s a step, the scheduler a few more. In Rust with no
collector that costs microseconds against a 45 to 95 ms step. **Verdict.** Does not carry. Our real host overhead is
stream synchronisation (item 6), not allocation. Not filed.

### 13. Scoreboard and measurement discipline

**jitllm.** Every perf claim is a ratio against a baseline taken in the same pass (RULE 1); A/B/A/B within one
process, n >= 20, median of per-round ratios, refused above IQR/median 0.10, an A/A self-control first, the whole
comparison run twice (RULE 2); tok/s reported with bytes per token and the fraction of a wall measured in the same
process. The scoreboard (`docs/perf/current.md`) holds every row with its command, and retracts rows in place.

**specrt.** We share the card with a live vLLM at 100% and the 300 W cap; most of our tables say "one run each,
indicative" (`docs/specrt-mtp.md`) or "medians of 6" without a control. Paired interleaved runs with an A/A control are
exactly what makes a shared card usable for A/B decisions.

**Win.** Fewer wrong decisions from noise. **Cost.** Small to medium: an A/B mode in `bench`, a scoreboard doc.

### 14. Not mapped

- **Paged weights from disk, each MoE expert a page, layers relocating between devices with their state**
  (`README.md` "The engine", `docs/placement.md`). The 27B fits; #323 covers offload if a larger model ever matters.
- **Joint step against sequential, measured per row count** (`server/batchchoose.go:24-33`). Their batched kernels
  are ragged and sometimes lose to sequential; our batched GEMVs and graphs always win at 2+ rows.
- **Persistent decode kernels** (`jit/gpu/ir/persist.go`). Measured slower (11% V100, 4% T4); their conclusion,
  "what would win is fusion that keeps an intermediate on chip", matches what we already do in `ops.cu`.
- **Warm-up at load** so the first request pays no JIT (`server/ttft.go:15-35`). Left to #394.
- **Model-free check suite, skip is not a pass** (`docs/checks.md`). A process rule, not specrt work.

## Ranking

Ranked by expected win for the 27B agent workload over cost.

| rank | idea | specrt today | win | cost | issue |
|---|---|---|---|---|---|
| 1 | Agent API surface: tools, tool-role messages, Qwen XML tool calls, reasoning, stop, refusal of unknown fields (9a) | drops them silently | precondition for agent traffic | medium | #400 |
| 2 | Speculative round on the device, graph-captured (6) | sync per upload, 1.36x at 2.43 tokens/step | about 31 to 43 tok/s single stream, estimated | medium-large | #401 |
| 3 | Swap preemption to pinned host and admission lookahead (8) | recompute, 50 to 130 s for 29k | no multi-minute stalls under load | medium | #402 |
| 4 | Adaptive draft count from measured verify cost (5) | fixed k, static `max_batch` | batch 8 from 0.81x to at least 1.0x | medium | #403 |
| 5 | Prefill budget relative to the decode step (7) | absolute 100 ms target | prefill beside decoders (220 tok/s now) | small | #404 |
| 6 | Prompt-lookup drafter beside MTP (4) | MTP only | unknown, measure on agent transcripts | small | #405 |
| 7 | Constrained decoding: JSON schema, `tool_choice` (9b) | none | forced calls are calls | medium-large | #406 |
| 8 | Gumbel-keyed acceptance: speculative sampling equals plain (10) | rejection with uniform draw | determinism, a byte-identity test | small | #407 |
| 9 | Measured per-shape tuning at load, persisted, env knobs to config (1) | formulas and env vars | unknown, measure | medium | #408 |
| 10 | Paired A/B bench mode and a scoreboard (13) | single runs on a shared card | trustworthy decisions | small-medium | #409 |
| 11 | `/v1/capabilities` and `doctor` (11) | none | discoverability | small-medium | #410 |
| - | int8 KV (2) | f16 | carried by #324 | - | #324 |
| - | prefix caching (3) | none | carried by #325, comment added | - | #325 |
| - | zero-alloc decode (12), paging, joint-vs-sequential, persistent kernels (14) | | does not carry | - | not filed |

## Licence

jitllm is Apache-2.0 (`LICENSE`, `COPYRIGHT`). Every issue above ports ideas, not code. If an implementation does
copy code, the obvious candidates are the Qwen XML tool-call reader (`engine/model/toolsyntax.go`) and the JSON Schema
to grammar converter (`engine/grammar/schema.go`). A Rust port of either is a derivative work: keep the Apache-2.0
notice and a NOTICE line naming jitllm in the file that carries it.
