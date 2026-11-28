# Engine options

Generated from `specrt::options::schema()` by `crates/specrt/tests/options_doc.rs`; do not edit by hand. Rewrite with `OPTIONS_WRITE=1 cargo test -p inference-bench-specrt --test options_doc`.

Every key can be set in the `[engine]` table of a server model, in the load dialog, or in a spec TOML under the section named by its prefix (`serve.`, `speculative.`, `runtime.`, `prefix_cache.`, `loader.`). An empty default means unset: the engine picks a value at load.

## basic

| key | default | kind | what it does |
|---|---|---|---|
| `serve.max_model_len` |  | int, from 1 | Tokens one conversation can hold. Bigger contexts keep more and cost more memory. |
| `serve.kv_cache_bytes` |  | int, from 0 | GPU memory to use. What is left after the weights becomes the KV cache that all requests share. |

## parallel requests

| key | default | kind | what it does |
|---|---|---|---|
| `serve.max_num_seqs` | `8` | int, 1 to 64 | Requests decoded together. They share one KV cache, and a single request still runs at full speed. |
| `serve.tp` | `1` | int, 1 to 8 | GPUs one model is split across, by attention head and MLP column, with an allreduce after each half layer. One stream decodes faster when the GPUs have peer access. 1 keeps the model on one GPU. |

## mtp

| key | default | kind | what it does |
|---|---|---|---|
| `speculative.enabled` | `true` | bool | Draft tokens with the model's own MTP head and verify them in one step. Output is unchanged. |
| `speculative.draft_tokens` |  | int, 1 to 3 | Tokens drafted per step. Auto (the default) picks the count per step from measured step time and acceptance. |
| `speculative.max_batch` |  | int, 1 to 64 | Largest batch that still drafts. Auto uses the model spec. |
| `speculative.head_quant` |  | bf16, fp8_channel, nvfp4_g16 | Precision of the extra lm_head copy used for drafts. Auto is nvfp4_g16, bf16 makes no copy. |

## prefix cache

| key | default | kind | what it does |
|---|---|---|---|
| `runtime.prefix_cache.enabled` | `false` | bool | Reuse the KV of a prompt prefix seen before, so a long shared prompt is read once. |
| `runtime.prefix_cache.snapshot_interval` | `2048` | int, 16 to 1048576 | Tokens between recurrent state snapshots. A multiple of the 16 token block. |
| `runtime.prefix_cache.max_snapshots` | `32` | int, 0 to 4096 | Recurrent state snapshots kept in host memory. 0 keeps none. |
| `runtime.prefix_cache.max_blocks` | `0` | int, 0 to 16777216 | KV blocks the cache may hold. 0 means no limit beyond the pool. |
| `runtime.prefix_cache.async_snapshots` | `true` | bool | Copy snapshots to the host on a side stream instead of stalling the step. |

## serving

| key | default | kind | what it does |
|---|---|---|---|
| `serve.max_batched_tokens` | `8192` | int, 64 to 1048576 | Token budget of one scheduler step. Larger prefills faster, smaller keeps decode steps short. |
| `serve.itl_target_ms` | `0` | int, 0 to 10000 | Optional cap in milliseconds on how long a decode stream may wait between tokens while prompts are being read. 0 leaves it to the step cost. |
| `serve.step_cost` | `3.0` | int, 0 to 64 | A step that reads prompts beside decoders may take this many times a plain decode step. Higher reads prompts faster, running streams wait longer between tokens. 1 or less turns it off. |
| `serve.whole_prompt_tokens` | `1024` | int, 0 to 1048576 | A new prompt up to this many tokens is read in a single step even while others decode. 0 keeps the inter-token target in charge. |
| `serve.min_prefill_tokens` | `512` | int, 0 to 1048576 | Prompt tokens read every step even when decoders already fill the inter-token target. Keeps time to first token bounded under load. 0 leaves prefill to the target alone. |
| `serve.swap_space_gb` | `0.0` | int, 0 to 1048576 | Gigabytes of pinned host memory where a preempted request parks its KV cache and state, instead of recomputing its prompt. 0 always recomputes. |
| `serve.admit_lookahead` | `0` | int, 0 to 1048576 | Tokens of generation a new request and each running one reserve KV room for before another request is admitted. Fewer preemptions, later admission. 0 reserves the prompt only. |
| `serve.prefill_burst_tokens` | `16384` | int, 0 to 1048576 | Prompt tokens read per step while a prompt over 2048 tokens is waiting or being read, even when decoders fill the inter-token target. Long prompts start sooner, running streams wait longer between tokens. 0 leaves long prompts to the guaranteed prefill. |
| `serve.max_long_in_flight` | `4` | int, 1 to 64 | Prompts over 2048 tokens read at the same time. More starts long prompts together instead of one after another. |
| `serve.max_overtakes` | `32` | int, 0 to 65536 | A waiting request that does not fit the KV cache yet lets later requests that do fit start ahead of it, at most this many times. Then nothing behind it starts until it does. 0 keeps strict arrival order. |
| `serve.stall_budget_ms` | `0` | int, 0 to 60000 | Once a running stream has waited this many milliseconds since its last token, each step reads only the prompt tokens that fit in the budget. Below it prefill uses the full burst. 0 turns it off. |
| `serve.min_stall_chunk_tokens` | `0` | int, 0 to 1048576 | Smallest prompt chunk a step reads while the decoder stall cap is active, so the cap never shrinks chunks to where prefill slows down. 0 has no floor. |

## runtime

| key | default | kind | what it does |
|---|---|---|---|
| `runtime.backend` | `"auto"` | auto, cuda, wgpu | Which compute backend runs the model. Auto picks CUDA when there is a card. |
| `runtime.graphs` | `true` | bool | Capture decode steps as CUDA graphs to cut launch overhead. |
| `runtime.pdl` | `false` | bool | Let each kernel of a decode graph start launching while the one before it finishes (Hopper and newer). |
| `runtime.fuse` | `true` | bool | Fuse neighbouring operations into single kernels. |
| `runtime.store_epilogue` |  | bool | In single stream decode, write a branch projection straight to its buffer instead of clearing the buffer first and adding into it. Same results, one launch fewer per branch. |
| `runtime.pdl_hoist` |  | bool | With programmatic launch, prefetch a kernel's weights into L2 before it waits for the kernel ahead. Off until measured faster. |
| `runtime.tile` | `"auto"` | auto, small, big, narrow | Tile shape for matrix multiplies. Auto picks per shape. |
| `runtime.gemm_select` |  | measured, formula, fixed | How a weight matrix multiply picks its kernel at decode and verify row counts. measured times each candidate once per card and shape at load and keeps the fastest in the kernel cache; formula estimates from bytes and occupancy; fixed keeps the old row thresholds. |
| `runtime.big_tile` |  | list | Explicit tile [m, n, k, stages] for the big choice. Auto uses the built in one. |
| `runtime.fp4_tile` |  | list | Explicit tile [m, n, k, stages] for the NVFP4 matrix multiply. Auto uses the built in one. |
| `runtime.fp8_tile` |  | list | Explicit tile [m, n, wm, wn] for the large FP8 prefill matrix multiply. Auto uses the built in one. |
| `runtime.fp4_stages` |  | int, 1 to 16 | Pipeline stages of the NVFP4 matrix multiply. Auto uses the built in count. |
| `runtime.kv_f16_layers` |  | int, 0 to 256 | With fp8 KV, how many of the first attention layers keep f16 KV. The early layers are the most sensitive. Auto keeps the first half up to 32k tokens of context and three quarters above it, where fewer f16 layers drift from f16. |
| `runtime.prefill_chunk` |  | int, 16 to 1048576 | Tokens read per prefill pass. Auto sizes it from memory. |
| `runtime.unsplit` | `[]` | list | Names of fused operations to keep whole instead of splitting across the batch. |
| `runtime.split_activations` |  | bool | Run each bf16 matrix multiply on a high and a low half of the activations, which matches an fp32 reference to 1e-3 at the cost of prefill speed. Off runs plain bf16 activations like other engines. Auto follows the model family. |
| `runtime.chunked_scan` |  | bool | Read long prompts through recurrent mixers in tensor core chunks instead of one row at a time. Same results, faster prefill. |
| `runtime.gemm_fp32` | `false` | bool | Accumulate matrix multiplies in fp32 on the plain path. Slower, for debugging. |
| `runtime.vision.precision` |  | fast, exact | Arithmetic of the image encoder. fast runs f16 tensor cores; exact splits every value into two halves and matches transformers in f32 at about twice the time. |
| `runtime.vision.cache_mb` |  | int, 0 to 1048576 | Host memory for encoded images, keyed by the image bytes, so a repeated image is not encoded again. 0 turns it off. |
| `runtime.vision.max_batch` |  | int, 1 to 64 | Images encoded together in one pass of the image encoder. |
| `runtime.vision.pan_and_scan` |  | bool | For tiled image models, also read crops of wide or tall images so small details survive the downscale. Costs one more tile of tokens per crop. |
| `runtime.kv_dtype` | `"f16"` | f16, fp8 | Precision of the KV cache. fp8 stores one byte per value with a scale per head and token, halving KV memory and attention reads against f16. |
| `runtime.rope_extension.type` |  | yarn | How the context is stretched past what the model was trained on. YaRN changes short context numerics slightly, so leave it off unless you need the length. |
| `runtime.rope_extension.factor` |  | int, 1 to 16 | Multiplies the model's own maximum context, so 4 takes 262144 tokens to 1048576. 1 or auto keeps the trained length. |
| `runtime.requant.lin_in` |  | fp8_channel, nvfp4_g16 | Reformat the delta-net input projections at load. Auto keeps the checkpoint format. |
| `runtime.requant.lin_out` |  | fp8_channel, nvfp4_g16 | Reformat the delta-net output projections at load. Auto keeps the checkpoint format. |
| `runtime.requant.qkv` |  | fp8_channel, nvfp4_g16 | Reformat the attention q, k and v projections at load. Auto keeps the checkpoint format. |
| `runtime.requant.o` |  | fp8_channel, nvfp4_g16 | Reformat the attention output projections at load. Auto keeps the checkpoint format. |
| `runtime.requant.mlp_tail` |  | fp8_channel, nvfp4_g16 | Reformat the MLPs the checkpoint stores as FP8 (the last 8 layers of the 27B) at load. Auto keeps the checkpoint format. |
| `runtime.requant.lm_head` |  | bf16, fp8_channel, nvfp4_g16 | Reformat the output head at load. Auto keeps the checkpoint format. |
| `runtime.quantize` |  | off, auto, nvfp4_g16, fp8_channel, recipe | Quantise the large linear and expert weights of a bf16 checkpoint while they load, with a small scale search per group. Embeddings, the output head, routers, norms and the first and last two layers stay bf16. Off keeps the checkpoint format; auto picks nvfp4_g16; recipe follows the per layer recipe quantcheck wrote (quantize recipe). |
| `runtime.quantize_recipe` |  | text | The recipe file `inference-bench quantcheck` wrote (<model>.quantrecipe.toml). Used when quantize is recipe. A recipe for another model, layer count or format version is refused. |
| `runtime.quantize_cache` |  | bool | Write the weights quantised at load to a cache beside the model (not beside its files) so the next start reads them instead of converting again. Costs disk, saves load time. |
| `runtime.cache_dir` |  | text | Where compiled kernels are kept. Auto uses the user cache folder. |
| `runtime.cache_max_bytes` |  | int, from 1 | Most bytes of compiled kernels to keep. The oldest are deleted after a new one is written. Auto keeps 4 GB. |
| `runtime.compile_jobs` |  | int, 1 to 64 | Threads NVRTC may use to compile one kernel module. Auto uses the CPU count, at most 8. |

## loader

| key | default | kind | what it does |
|---|---|---|---|
| `runtime.loader.enabled` | `true` | bool | Read weights with parallel readers straight to the card. |
| `runtime.loader.direct` | `true` | bool | Bypass the page cache with O_DIRECT reads. |
| `runtime.loader.chunk_mb` | `32` | int, 1 to 1024 | Megabytes per read. |
| `runtime.loader.readers` | `8` | int, 1 to 64 | Parallel read threads. |
| `runtime.loader.ahead` | `2` | int, 1 to 64 | Chunks queued ahead of the copy. |
| `runtime.loader.workers` | `4` | int, 1 to 64 | Threads that convert and upload tensors. |
| `runtime.loader.lanes` | `1` | int, 1 to 16 | Parallel upload streams. |
| `runtime.loader.skip_cached` | `true` | bool | Read files already in the page cache the plain way. |
| `runtime.loader.pinned` | `false` | bool | Use pinned host memory for staging. |
| `runtime.loader.host_budget_mb` | `3584` | int, 64 to 1048576 | Megabytes of host memory the loader may use for staging. |

## offload

| key | default | kind | what it does |
|---|---|---|---|
| `runtime.offload.vram_budget` | `0` | int, from 0 | How much of the weights stay on the GPU, in bytes. Layers beyond it stay in pinned host memory and stream in while earlier layers compute. 0 keeps every weight on the GPU. |
| `runtime.offload.host_budget` | `0` | int, from 0 | Bytes of pinned host memory for streamed layers. 0 means no cap. |
| `runtime.offload.nvme_path` |  | text | Folder on a fast drive for layers that fit neither tier. |
| `runtime.offload.prefetch_depth` | `1` | int, 1 to 8 | Layers copied ahead of the one computing. |
| `runtime.offload.expert_slots` | `0` | int, 0 to 1048576 | Experts kept on the card for a mixture of experts model, shared by every layer. Missing experts copy in after the router picks them. 0 sizes the cache from the gpu offload setting. |
| `runtime.offload.pointer_table` |  | bool | Decode a layer whose experts are all on the card from a device table of expert addresses, with no wait on the host. Layers with a missing expert use the staged copy path. |
| `runtime.offload.hotlist` |  | text | File of expert use counts from a calibration run. The most used experts are loaded first and guide prefetch before the router has history. |
| `runtime.offload.policy` | `"score"` | lru, frequency, score | Which resident page leaves first when a tier is full. Score keeps a decaying use score per expert and refills the expert cache between decode steps, off the token's path. |
| `runtime.offload.promote_margin` | `1.15` | int, 1 to 16 | With the score policy, a resident expert is replaced only when the newcomer's score is this many times higher, so the cache does not thrash. |
| `runtime.offload.score_half_life` | `64` | int, 1 to 1048576 | Forward passes after which an expert's use score has halved, for the score policy. One decode step is one pass. |
| `runtime.offload.promote_per_step` | `16` | int, 0 to 1048576 | Most experts the score policy copies in between two decode steps. |
