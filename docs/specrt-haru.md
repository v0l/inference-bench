# Haru-4B: full duplex speech to speech on specrt (#1344)

Analysis only, no engine code. Read on 25 Nov from `higurashi-ai/Haru-4B` (`config.json`, `tokenizer.json`, the 111,704 byte safetensors header fetched with an HTTP range, no weights) and the reference code `github.com/higurashi-ai/Haru` at commit `7fe6b766` (`haru/haru/models/*.py`, `session.py`, `prompt.py`, `sampling.py`, `server.py`). The card's facts hold; the numbers below are from the files.

Fixture and test: `crates/specrt/tests/fixtures/haru/` and `crates/specrt/tests/haru.rs` (section 7).

## 1. What it is

Apache 2.0, 5,270,000,899 parameters, all bf16 in one `model.safetensors` (10.54 GB). A pretrained base model, not tuned for instructions or tools. One language model position is one 133 ms frame (3200 samples at 24 kHz, 7.5 per second). At every position the input is the sum of three streams, and the hidden state at frame t predicts frame t + 1 on two outputs: a text token through the tied embedding and a 64 wide audio latent through a small MeanFlow head. The audio codec turns latents back into PCM (and the user's PCM into latents) one frame at a time.

| Part | Prefix | Parameters | bf16 |
|---|---|---|---|
| Qwen3.5-4B text backbone | `lm.` | 4,205,751,296 (embedding 635,699,200) | 8.41 GB |
| MeanFlow audio head | `head.` | 363,407,360 | 0.73 GB |
| Codec encoder (user audio in) | `codec.encoder.` | 343,696,032 | 0.69 GB |
| Codec decoder (model audio out) | `codec.decoder.` | 343,695,969 | 0.69 GB |
| Model audio projector | `model_audio_proj.` | 6,725,120 | 13 MB |
| User audio projector | `user_audio_proj.` | 6,725,120 | 13 MB |
| Latent normalisation | `latent_scale`, `latent_bias` | 2 scalars | |

## 2. Tensors and shapes

`config.json` has `model_type = "haru"`, `architectures = ["HaruForConditionalGeneration"]`, top level `sample_rate`, `frame_samples`, `latent_dim`, `pad_token_id` 248076 (`<|audio_pad|>`), `eop_token_id` 248071 (`<|audio_end|>`), and three nested sections `lm`, `codec`, `head`.

### Backbone, `lm.*`

`lm` is a plain `qwen3_5_text` config: 32 layers, hidden 2560, MLP 9216, vocab 248320, 16 query heads and 4 KV heads of 256 with the output gate and qk norm, partial rotary 0.25, theta 1e7, `mrope_interleaved` with sections 11, 11, 10, gated delta net layers with 16 key heads and 32 value heads of 128, conv kernel 4, `layer_types` three linear then one full, eight times. `tie_word_embeddings = true`: there is no `lm_head`, logits are `h @ embed_tokens.T`.

The tensors under `lm.` are exactly the `Qwen3_5TextModel` names with `lm.` in place of `model.` (`haru.rs` pins the 22 kinds and the 24 linear plus 8 full layer split):

```
lm.embed_tokens.weight                         [248320, 2560]
lm.norm.weight                                 [2560]
lm.layers.N.input_layernorm.weight             [2560]
lm.layers.N.post_attention_layernorm.weight    [2560]
lm.layers.N.mlp.{gate,up}_proj.weight          [9216, 2560]
lm.layers.N.mlp.down_proj.weight               [2560, 9216]
lm.layers.N.linear_attn.in_proj_qkv.weight     [8192, 2560]
lm.layers.N.linear_attn.in_proj_z.weight       [4096, 2560]
lm.layers.N.linear_attn.in_proj_{a,b}.weight   [32, 2560]
lm.layers.N.linear_attn.conv1d.weight          [8192, 1, 4]
lm.layers.N.linear_attn.{A_log,dt_bias}        [32]
lm.layers.N.linear_attn.norm.weight            [128]
lm.layers.N.linear_attn.out_proj.weight        [2560, 4096]
lm.layers.N.self_attn.q_proj.weight            [8192, 2560]   (query and gate)
lm.layers.N.self_attn.{k,v}_proj.weight        [1024, 2560]
lm.layers.N.self_attn.o_proj.weight            [2560, 4096]
lm.layers.N.self_attn.{q,k}_norm.weight        [256]
```

### Audio projectors, `model_audio_proj.*` and `user_audio_proj.*`

`fc1` Linear 64 to 2560 with bias, `norm` RMSNorm 2560 (eps 1e-6, weight), `fc2` Linear 2560 to 2560 with bias. `y = fc2(norm(fc1(x)))`, no activation.

### MeanFlow head, `head.*`

```
head.cond_proj.weight                     [2560, 2560]
head.latent_proj.weight                   [2560, 64]
head.{t,dt,h,omega}_embed.fc1.weight      [2560, 256]
head.{t,dt,h,omega}_embed.fc2.weight      [2560, 2560]
head.layers.{0..3}.norm.weight            [2560]
head.layers.{0..3}.ada.weight             [7680, 2560]
head.layers.{0..3}.ffn.{gate,up}_proj.weight [7680, 2560]
head.layers.{0..3}.ffn.down_proj.weight   [2560, 7680]
head.final.ada.weight                     [5120, 2560]
head.final.proj.weight                    [64, 2560]
```

No biases anywhere in the head.

### Codec, `codec.*`

`base_channels` 32, `ratios` [8, 5, 5, 4, 2, 2] (product 3200), `encoder_depths` [3, 3, 3, 3, 3, 3, 8], `decoder_depths` [8, 3, 3, 3, 3, 3, 3], eps 1e-5, `latent_std` 0.5.

Every stage is a stack of the same `Block` at width C (ConvNeXt shaped, channels last inside):

```
x = x + gamma * dwconv7(rmsnorm(x))           conv.weight [C, 1, 7], conv.bias [C], norm.weight [C], gamma [C]
x = x + ffn_gamma * fc2(gelu(fc1(rmsnorm(x)))) fc1 [4C, C] + bias, fc2 [C, 4C] + bias, ffn_norm.weight [C], ffn_gamma [C]
```

`gelu` is `F.gelu` with the exact erf form. RMS norms are computed in f32 over the channels of one time step.

| Encoder layer | Weight | Time steps per frame | State per layer |
|---|---|---|---|
| `stem` causal conv 1 to 32, k 7 | [32, 1, 7] | 3200 | 1 x 6 input samples |
| 3 blocks at 32 | | 3200 | 32 x 6 each |
| `down.0` causal conv 32 to 64, k 4, stride 2 | [64, 32, 4] | 1600 | 32 x 2 |
| 3 blocks at 64 | | 1600 | 64 x 6 each |
| `down.1` 64 to 128, k 4, s 2 | [128, 64, 4] | 800 | 64 x 2 |
| 3 blocks at 128 | | 800 | |
| `down.2` 128 to 256, k 8, s 4 | [256, 128, 8] | 200 | 128 x 4 |
| 3 blocks at 256 | | 200 | |
| `down.3` 256 to 512, k 10, s 5 | [512, 256, 10] | 40 | 256 x 5 |
| 3 blocks at 512 | | 40 | |
| `down.4` 512 to 1024, k 10, s 5 | [1024, 512, 10] | 8 | 512 x 5 |
| 3 blocks at 1024 | | 8 | |
| `down.5` 1024 to 2048, k 16, s 8 | [2048, 1024, 16] | 1 | 1024 x 8 |
| 8 blocks at 2048 | fc1 [8192, 2048] | 1 | 2048 x 6 each |
| `head` causal conv 2048 to 64, k 7 | [64, 2048, 7] | 1 | 2048 x 6 |

The encoder output is the latent mean (no variance tensor). The decoder mirrors it: `stem` causal conv 64 to 2048 (k 7), 8 blocks at 2048, then six `up.i` transposed convs with kernel 2s and stride s (`[2048, 1024, 16]`, `[1024, 512, 10]`, `[512, 256, 10]`, `[256, 128, 8]`, `[128, 64, 4]`, `[64, 32, 4]`), each followed by 3 blocks, and a `head` causal conv 32 to 1 (k 7).

Streaming state. A causal conv with kernel k and stride s keeps its last k minus s input columns; a transposed conv with kernel 2s keeps its overlap tail of s output columns per channel (the reference adds it to the next call's first s outputs, then adds the bias). Every state has a fixed shape, so one frame in gives exactly one frame out with the same numbers as a whole clip (fixture: 9e-7 encode, 3e-6 decode, relative). For the real config the state is 159,622 values for the encoder and 147,904 for the decoder (0.6 MB per session in bf16), computed in `haru.rs` from the config and checked against the reference on the tiny one.

In specrt (#1429) the codec is `graphs/codec.toml`, a macro graph over `base_channels`, `ratios`, `encoder_depths` and `decoder_depths` picked by `model_type`, with no Rust naming the codec. `specrt::codec` expands it, runs it on the host and on CUDA, one frame or a whole clip per call, batched over sessions. Each transposed conv is lowered to a 2 tap causal conv to s x Cout channels whose output phases are interleaved in the store, with the previous input column as its state, so the decoder keeps 139,200 values per session instead of 147,904. Convs and the 1x1 `fc` layers are the line form of the `kir::conv` family (stride, phases, exact GELU and a scaled residual in the epilogue), the depthwise k 7 conv is a per channel stencil with the `gamma` residual folded in, and each causal layer's tail is a double buffered state per session. `inference-bench kernels --spec haru-4b.config.json --plan --graph codec` prints the launches per frame: 172 each way, 1.42 GB read per frame at f32 weights (1.37 GB of weights), 3.64 GFLOP.

### Latent spaces

Three spaces, two scalars. The codec works on raw latents. The projectors and the head work on normalised latents: `norm = (raw + latent_bias) * latent_scale`, `raw = norm / latent_scale - latent_bias`. The user stream adds noise before it is used, as in training: `user = mean + r * (latent_std / 0.8) * n` with one `r ~ N(0, 1)` per frame and `n ~ N(0, 1)` per dimension. The voice reference is encoded and noised the same way.

## 3. The three streams in one position

```
e_t = embed_tokens[text_t]
    + model_mask_t * model_audio_proj(norm(model_raw_t))
    + user_mask_t  * user_audio_proj(norm(user_raw_t))
h_t = lm(e_0..e_t)                     one position per frame, KV and delta net state as for text
text_{t+1}  = sample(h_t @ embed_tokens.T)
model_{t+1} = unscale(meanflow(h_t, noise_t, guidance))
```

The masks exist only for the prompt (section 5); in a live session both audio streams are always on. `text_t` and `model_raw_t` are the model's own outputs of the previous step, fed back; `user_raw_t` is the newest user frame. The rope positions are the plain position index on all three mrope axes, so the interleaved sections reduce to ordinary partial rope, which is what the engine already does for text only Qwen3.5.

One step of a live session, in order:

1. Encode the user's 3200 samples with the streaming encoder: mean (64), plus the noise above.
2. Sum the three streams into `e_t` (2560).
3. Run the 32 layer backbone for one position, final norm, `h_t`.
4. Text: full vocab logits; PAD and EOP logits recomputed in f32 (the reference does it because bf16 cannot resolve the silence decision; our GEMV accumulates in f32 anyway); DRY repetition penalty (multiplier 0.8, base 1.75, allowed length 1, lookback 1024, max match 32, cap 32, PAD, EOP and EOS exempt); `eop_bias` on EOP; temperature 0.7, top k 25, top p 1.
5. Audio: one MeanFlow step from `h_t` and fresh noise, unscale.
6. Decode the latent with the streaming decoder: 3200 samples out.

The step consumes user frame t and returns model frame t + 1, so the model is always one frame ahead: the frame length is the only structural latency. PAD on a frame means no text; EOP marks the frame before the model starts to talk. The detokeniser renders a run of PAD or EOP as one space and holds back a partial UTF-8 character.

## 4. The MeanFlow head

The head predicts the mean velocity `u(z, t, h | c, w)` from noise (t = 0) to data (t = 1) over an interval h; sampling is `z += h * u`. The guidance scale w is an input of the network, so guided sampling is one pass, not two.

```
cond = cond_proj(h_t)
c    = silu(cond + T(t) + DT(interval_embed = 7.0) + H(h) + OMEGA(1 - 1 / w))
x    = latent_proj(z)
for each of 4 layers:
    shift, scale, gate = ada(c)                       three 2560 slices of 7680
    x = x + gate * ffn(rmsnorm(x) * norm.weight * (1 + scale) + shift)    ffn = SwiGLU 2560 to 7680 to 2560
shift, scale = final.ada(c)
u = final.proj(rmsnorm(x) * (1 + scale) + shift)      64
```

`T`, `DT`, `H`, `OMEGA` are each a sinusoid of the scalar (128 frequencies `exp(-ln(10000) i / 128)`, cos first then sin) followed by Linear 256 to 2560, SiLU, Linear 2560 to 2560. With one step (`num_steps = 1`), t = 0 and h = 1, so all four scalar embeddings are constant for a given guidance: their sum is one 2560 vector computed once per session (or per guidance change), which removes 28.8M of the 363M parameters from the per frame read. Per frame the head is one 2560 square GEMV, five modulation GEMVs, four SwiGLU MLPs and a 64 row projection, all on one row: decode shaped and memory bound, unlike the DiT where modulation was compute bound.

`haru.rs` reproduces the head in plain Rust from the fixture weights: guidance 1.5 (the session latents), 1.0 and 2.0, and a two step run, all within 1e-4 relative of the reference, and guidance 1 and 2 give different latents.

In specrt (#1428) the head is `graphs/meanflow.toml`, a flow head graph bound to the config by `[head]` in `family-haru.toml` (sizes, `num_steps`, `interval_embed`, the guidance range). Its nodes are the diffusion ops: `TimestepEmbed` and two linears per scalar, `Modulate` with `norm = "rms"` and the layer's `norm.weight`, `Glu`, `GatedAdd`, and `Map` for the embedding sum, `silu(cond + emb)`, the update `z + h_step * u` and the unscale, each a kir expression lowered to a compose kernel. The four scalars come from `[scalars]` expressions over `step`, `num_steps`, `interval_embed` and `guidance`. `FlowHead::levels` is the H1 hoist: an op is per session when nothing it reads changes per frame or per step, so with one step the four embeddings and their sum (57.8 MB, 17 launches) run when a session starts or its guidance changes, and with two steps `t` moves into the step loop. Every linear is the decode shaped rows GEMV (`kir::gemv::Gemv::Rows`, bf16 weights, split bf16 activations, all sessions of a batch as rows, an expression epilogue), and the fusion pass folds the gated residual into the down projection's epilogue. `inference-bench kernels --spec haru-4b.config.json --plan --graph head [--tokens sessions]` prints it: per frame 0.67 GB read and 35 launches, 27 after fusion. Frame noise comes from `flow_head::frame_noise`, the `sampler::noise` generator keyed by seed and frame; the tests feed the fixture's recorded noise instead. On GPU1 the tiny head gives the four references within 8e-6 relative (`tests/haru_head_gpu.rs`).

## 5. The prompt and the character card

```
position:  [0, P)            [P, S)                    [S, ...)
text:      card tokens       PAD ... PAD EOP           live text stream
model:     none (masked)     voice reference latents   model audio
user:      none (masked)     none (masked)             user audio
```

- The card is a ChatML system turn, `<|im_start|>system\n{card}<|im_end|>\n`, tokenised without special tokens added, text only. Its text is built from fields (persona, name, gender, backchannel frequency, the other person's name, relationship, situation) under a lead sentence and bracketed headers per language (`assets/card_templates.json`; English lead: "You are the person described below, talking with the other person by voice.").
- The voice reference is 3 s to 180 s of one speaker, silence trimmed, encoded and noised, one position per frame on the model stream, text PAD on each and EOP on the last.
- With `speak_first`, two more positions `[PAD, EOP]` carry the reference's quietest frame (room tone) as model audio and the encoded digital silence as user audio, and the first sampled token is forced to be text. Without a card, reference or opening, the prompt is one PAD position.
- The prompt is prefilled in chunks of 512; the first frame comes from its last hidden state. A card of a few hundred tokens and a 30 s reference is about 500 positions, one prefill call.

The card cannot change mid session: it sits at position 0. A new card means a new session.

## 6. What maps onto existing specrt pieces

| Piece | Status |
|---|---|
| Backbone, `lm.*` | **specrt** (#1427): `family-haru.toml` extends `family-qwen3_5`, reads `lm` as the text config (`text_config = "lm"`) and maps the `lm.` root to `model.` through `[tensors]`. The family stays refused until #1430 |
| Tied logits, top k, top p, temperature | **specrt** sampler |
| DRY penalty, f32 PAD and EOP logits, `eop_bias`, forced text | **specrt** (#1430): `[duplex.sampling]` in the family, host plans per row (`duplex::sampling`) applied by the `kir::duplex` kernels inside the frame graph |
| Input sum with projectors | **specrt** (#1427): `[inputs]` in the family declares the streams and masks; six `[[inputs.node]]` plan ops (`Op::Input`, typed signatures, kir `group_map` descriptions) add `mask * fc2(rmsnorm(fc1((raw + latent_bias) * latent_scale)))` into `x` after the token gather, for prefill rows and decode rows alike (`Runtime::set_inputs`) |
| MeanFlow head | **specrt** (#1428): `graphs/meanflow.toml` and `[head]`, `specrt::flow_head` (plan, hoist, report, host) and `flow_head_cuda` |
| Codec 1D convs | partly from #1283: `kir::conv` implicit GEMM (the 1x1 `fc1` and `fc2` are its 1x1 case or the tile GEMM), the per pixel channel RMS norm (`VaeOp::RmsNorm`), the causal frame cache of the Wan VAE as streaming state, one latent frame per call. **new**: strided causal 1D conv (down), the transposed conv (lowers to a 2 tap causal conv to s x Cout channels plus an interleave, the generalisation of Wan's `time_up`, with the previous input column as state instead of the overlap tail), depthwise k 7 conv (a per channel stencil, a compose or rows kernel, not a GEMM), exact GELU (`erf` is not in the `Expr` unary table, `kir/src/expr.rs:69`) |
| Session KV and state | **specrt** (#1430): paged KV for the 8 full layers, delta net state for the 24 linear layers, codec state, sampler state and the frame counter, declared as `[duplex] state`; snapshots through the #326 store with three extra sections |
| Frame clock scheduler | **specrt** (#1430): `duplex::clock`, a fixed 133 ms tick per session, silence for a missing input frame, late frames sent and marked, the input backlog cut so latency comes back |
| Audio websocket | **new** (axum is built without its `ws` feature today) |

## 7. Fixture

`crates/specrt/scripts/tiny_haru.py` builds the fixture on CPU in about 2 seconds with the reference package itself (`haru` at `7fe6b766`, torch 2.15 CPU, transformers 5.24, in a private venv). Deterministic: two runs give identical bytes.

```
uv venv /tmp/haru-1344 -p 3.13
VIRTUAL_ENV=/tmp/haru-1344 uv pip install --index-url https://download.pytorch.org/whl/cpu torch
VIRTUAL_ENV=/tmp/haru-1344 uv pip install "transformers>=5.16" tokenizers safetensors numpy huggingface-hub
git clone https://github.com/higurashi-ai/Haru /tmp/Haru && git -C /tmp/Haru checkout 7fe6b766
cd crates/specrt && systemd-run --user --wait --collect -P -q -p MemoryMax=8G --working-directory=$PWD \
  /tmp/haru-1344/bin/python scripts/tiny_haru.py fixture --haru /tmp/Haru/haru --commit 7fe6b766
```

`tests/fixtures/haru/` (2.98 MB):

- `haru-4b.config.json`: the real config. `haru-4b.header.json`: the real checkpoint's 1022 tensor names, dtypes and shapes (from the header, `tiny_haru.py header`).
- `tiny/config.json` and `tiny/model.safetensors` (3.3M parameters, bf16): the same structure in miniature. Backbone 4 layers (three linear, one full), hidden 64, 2 query heads and 1 KV head of 64, gated delta net with 2 key and 4 value heads of 32, vocab 512 with PAD 508, EOP 509, EOS 510, tied embedding, mrope sections [3, 3, 2]. Head 2 layers at 64. Codec with base 2 channels and every real stride, one block per stage, so a frame is still 3200 samples and every layer type is present. `haru.rs` checks the tiny file has exactly the real tensor kinds.
- `tiny/reference.json` and `tiny/reference/*.bin` (f32 little endian, shapes in the JSON), computed in f32 from the bf16 rounded weights:
  - `prompt.*`: a 12 token card and a 3 frame voice reference (a clip that ends mid frame): tokens, both latent streams and masks, the summed embeddings, the residual stream before layer 0 and after every layer, the final hidden states and logits.
  - `session.*`: six live frames: user PCM, the streaming encoder means, the noise and its scale, the noised user latents, the summed embeddings, the hidden state, logits, greedy token, head noise and latent, and the 7 x 3200 decoded samples, all from the reference's own step order with greedy text so it replays exactly.
  - `head.guidance{1.0,1.5,2.0}`, `head.steps2`: the head on the session hidden states.
  - `codec.*`: the reference clip, its whole clip encode and decode, and the silence latent.
  - `checks`: chunked prefill against one call (5e-6), streaming encode and decode against whole clip (under 3e-6 relative), and the codec state sizes.

`tests/haru.rs` (CPU, no GPU, 0.02 s): the real parameter split and shapes, the backbone layout, the `lm` sub-config accepted as a Qwen3.5 spec while the `haru` wrapper is refused, the tiny tensor kinds, the three stream sum and the user noise rebuilt in Rust from the fixture weights (1e-5), the tied text logits and greedy tokens (1e-5), the MeanFlow head (1e-4), the codec state sizes from the config, and the size cap.

## 8. Feasibility on one RTX PRO 6000 (1,640 GB/s read ceiling, about 1.5 TB/s reached by our decode GEMVs)

Bytes read per frame for one session, everything else is small:

| Part | bf16 | Quantised | Time bf16 | Time quantised |
|---|---|---|---|---|
| Backbone, 3.57 GB of layers plus the 1.27 GB tied head | 8.41 GB | 5.9 GB in the bench4b layout (NVFP4 MLPs, FP8 projections, bf16 head): **3.9 ms measured** (255 tok/s single stream, `docs/specrt-test-fixtures.md`); 3.3 GB with every linear in NVFP4 | 5.6 ms | 3.9 ms (2.2 ms) |
| MeanFlow head, constant embeddings hoisted | 0.67 GB | 0.19 GB NVFP4 | 0.45 ms | 0.13 ms |
| Codec encoder, nearly all weights in the 8 blocks at 2048 that run on one time step | 0.69 GB | | 0.46 ms | |
| Codec decoder | 0.69 GB | | 0.46 ms | |
| Projectors, embedding row | 27 MB | | 0.02 ms | |
| Launches, about 600 per step in a CUDA graph | | | about 1 ms | |

So one frame costs about **8 ms at bf16** against a 133 ms budget (6%), and about 6 ms quantised. Compute is not the limit: the codec does 3.6 GFLOP per frame each way, the backbone about 8 GFLOP. Quantisation is not needed to run it; it only matters for many sessions per card. Weights are read once per step for all sessions in the batch, so the same card serves on the order of ten sessions before the step nears the frame.

Memory:

| Item | Size |
|---|---|
| Weights, bf16 | 10.54 GB (about 5.5 GB with an NVFP4 backbone and head) |
| KV, 8 full layers x 4 heads x 256 x K and V x bf16 | 32 KiB per frame, **844 MiB per hour** (422 MiB in fp8); prompt of 500 positions 16 MB |
| Delta net state, 24 layers x 32 x 128 x 128 f32, plus conv state | 51.5 MB per session, constant |
| Codec state | 0.6 MB per session |
| Sampler state (DRY history of 1057 tokens, last token, last latent, RNG) | a few KB |
| Activations, prompt chunk of 512 | under 0.3 GB |

One hour long session: about **11.8 GB**, which matches the card's 12 GB. The reference caps a session at 27,000 frames (60 minutes). `max_position_embeddings` is 262,144 positions, 9.7 hours; how far the model holds up past the length it was trained on is unknown.

Beside the user's 27B app: GPU0 showed 96,491 of 97,887 MiB used when this was written, so it does not fit there. GPU1 had 5 MiB used: Haru fits on GPU1 with room for many sessions. When the app spreads the 27B over both cards it fits wherever the planner finds 12 GB free; the load dialog decides, as for any model.

## 9. Endless sessions

The 24 delta net layers carry a constant size state, so only the 8 full layers grow, 844 MiB an hour. Plan:

1. Default: no window. A four hour conversation is 3.4 GB of KV. The session ends at the position limit or at `[duplex] max_frames` from the family file (default 27,000, the reference's value).
2. `[duplex] window_frames` (off by default): keep the prompt (card and voice reference) as sink positions plus the last W frames in the full layers, positions keep counting. The delta net state still sees everything. Off until measured on the real model, because Haru was not trained with a window.
3. Idle sessions use the #326 snapshot store unchanged, with three extra payload sections: codec state, sampler state and the frame counter. A paused call resumes without prefill.

## 10. Supervisor bridge

The model has no tools, and its text stream is **its own words**, not a transcript of the user (`session.py`: the token is sampled from the model's hidden state and fed back as the model's text). So the bridge needs a separate recogniser for the user's speech. Shape:

- **User to supervisor**: the client keeps running speech to text (the existing Whisper path in `agent-voice`) on the same microphone frames it streams to the engine, resampled to 16 kHz, cut into utterances by its silence detector. Each finished utterance is a text turn to the supervisor chat.
- **Supervisor to user**: the engine gets a per session **say queue**. While the queue is not empty, the text sampler only allows PAD, EOP and the next queued token: the model keeps its own timing, voice and pauses, but the words are the supervisor's. Audio follows the text stream in this design (the model learnt to voice what its text says), so the spoken words follow the queue. When the queue is empty the session is in one of two modes: `relay` (default for the supervisor: only PAD and EOP are allowed, so the model listens, backchannels and stays quiet rather than invent facts) or `free` (plain sampling, for chit chat).
- **Barge in**: the bridge sends `interrupt` when the user speaks for more than about 400 ms while the queue is not empty; the engine drops the rest of the queue and reports how much was said, which the bridge passes to the supervisor as the part the user heard.
- **Card**: a fixed card for the supervisor voice (name, calm persona, "relays answers from an assistant", relationship and situation) and a voice reference the user records once. It is set when the session starts.

Whether constrained text keeps the speech intelligible is unmeasured: the D slice measures it on the real model (Whisper word error rate of the produced audio against the queued text). If it fails, the fallback is the existing text to speech path for supervisor replies with Haru muted to `relay` for turn taking and backchannels.

Echo: headphones first. With speakers the model hears itself, which a full duplex model treats as the user talking; an echo canceller in the client is its own slice (agentbench #1405 slice 2).

## 11. Gap list

Each slice follows `docs/specrt-kir-compiler.md`: new ops are typed signature entries with kir descriptions, new state is declared in the spec, and nothing in Rust names the architecture.

| Slice | What | Reuses | Issue |
|---|---|---|---|
| A | Haru family file, `lm` weight root, the multi stream input embedding (projectors, masks, latent scale and bias as data), a decode step fed by an embedding row, prompt prefill with masks | `family-qwen3_5`, `Op::Splice` / `add_rows` | #1427 |
| B | MeanFlow head as a plan section: RMS modulate, gated add, SwiGLU, sinusoid embeddings hoisted per guidance, the step rule as an `Expr` | #1282 and #1284 ops, H1 hoist | #1428 (after A) |
| C | Streaming codec encoder and decoder: strided causal conv, transposed conv as a 2 tap causal conv plus interleave, depthwise k 7 stencil, exact GELU (`erf` in `Expr`), channel RMS norm, per layer state declared in the spec | #1283 conv family, frame cache, channel RMS norm | #1429 (parallel to A and B) |
| D | Duplex sessions: the frame step (encode, sum, backbone, text and head, decode) batched over sessions, sampler options (DRY, f32 silence logits, `eop_bias`, the say queue and modes), session state and snapshots, `max_frames` and the optional window, real model checks on GPU1 | scheduler, #326 store | #1430 (after A, B, C) |
| E | Audio API: a websocket per session (PCM16 24 kHz frames in, frames and text out), session start with card and voice reference, `say`, `mode`, `interrupt`; the OpenAI Realtime subset if cheap; `/v1/models` type `duplex` (additive); a CLI client; the model library and load dialog report it from the engine | server, media job pattern from #1285 | #1431 (after D) |
| F | App: hands free conversation mode and the supervisor bridge client | `crates/app/src/voice`, `agent-voice` | agentbench, after #1431 and agentbench #1405 slices 1 and 2; body in section 12 |

The diffusion chain covers most of B (modulation, gated residual, SwiGLU, sinusoid, the step as an expression, hoisting) and the conv core and streaming cache of C. A, D and E are new: diffusion has no token loop, no per session state and no streaming transport.

The OpenAI Realtime API sends PCM16 at 24 kHz, the codec's own rate, so `input_audio_buffer.append` and `response.output_audio.delta` map directly onto frames. Its turn events (`response.create`, `conversation.item.*`) do not fit a model that never takes turns, so E builds the native stream first and offers the Realtime shape as a subset: audio in, audio and transcript deltas out, `session.update` with `instructions` as the card.

## 12. App issue (agentbench, goal g3, backlog)

To be filed in agentbench (this session had no permission to file there), depending on inference-bench #1431 and agentbench #1405, whose slices 1 and 2 (hands free mode with the headphones check, and the echo canceller) stay there; this is its slice 3 made concrete.

**Title:** Hands free conversation with the Haru duplex engine and the supervisor bridge

- Hands free conversation mode with a `duplex model` setting naming the server and model: the microphone stays open, 133 ms frames stream as they fill, reply frames play as they arrive through a jitter buffer of two frames (silence when a frame is late, never a growing delay), reconnect with backoff. Reuse `crates/app/src/voice` (capture, devices, band and hang logic) and `agent-voice` (audio helpers, resampling to 24 kHz).
- Talk indicator in the chat header: listening, model speaking, supervisor thinking, at 1280 and 390 wide.
- Echo: headphones first (the #1405 slice 1 refusal); speakers only once the #1405 slice 2 canceller is on.
- The bridge of section 10: Whisper on the same microphone frames turns each user utterance into a supervisor text turn; supervisor replies go to the engine as `say` in short sentences (from `Converse`) with the session in `relay` mode; a barge in sends `interrupt` and passes the `said` part back as what the user heard. A fixed supervisor card and a voice clip the user records once, kept in settings.
- Fallback: with no duplex model set, hands free uses today's speech to text, supervisor, text to speech path.
- Tests with the app's own hooks (`voice_test_*`) against a fake duplex server: frames go out every 133 ms at 6400 bytes, replies play in order, a late frame plays silence, an utterance becomes a supervisor turn, a reply becomes `say` messages, a barge in sends `interrupt`. Pictures of the indicator at 1280 and 390. The real conversation with the real model is the user's check.

## 13. Duplex sessions in specrt (#1430)

`specrt::duplex` runs Haru sessions on the engine: one user frame in, one model frame out, batched over sessions. Everything that is model specific is data in `family-haru.toml` under `[duplex]`: frame and latent sizes read from the config, the user noise divisor (0.8), the PAD, EOP and EOS ids, the ChatML card template, the declared session state (`kv`, `recurrent`, `codec`, `sampler`, `frame`), `max_frames` (27,000), `window_frames` (0, off) and `[duplex.sampling]` (temperature 0.7, top k 25, top p 1, `eop_bias` 0, one forced text frame with PAD, EOP and EOS banned, f32 PAD and EOP logits, and DRY 0.8, 1.75, 1, 1024, 32, 32 with PAD, EOP and EOS exempt). `Spec::from_hf_config_duplex` reads it; the text serving path stays refused (`[[refuse]] part = "text"`) until the audio API of #1431.

| Piece | Where |
|---|---|
| Prompt: card, voice reference (trimmed at -45 dBFS with 0.2 s margin, at most 180 s, encoded and noised with one scale for the clip), `speak_first` grid with room tone and silence, forced first text | `duplex::prompt`, `Engine::start` |
| Prefill in chunks of `prefill_chunk` with per position masks, first frame from the last hidden state and the prompt logits | `Engine::start_prompt` |
| Frame step: encode user PCM, noise the mean, sum the streams, one backbone position for every session as decode rows, exact PAD and EOP logits from the joined f32 hidden state, logit edits, allow list, sampling, MeanFlow head, decode | `Engine::step`, one CUDA graph per padded batch size (1, 2, 4, 8, 16, 32) |
| Text controls: DRY history, `eop_bias`, forced text, say queue (only PAD, EOP and the next queued token), `relay` (only PAD and EOP) and `free`, `interrupt` (drops the queue, returns said and unsaid tokens) | `duplex::sampling::TextState`, host plans uploaded per frame, applied by `duplex_edit_logits` and `duplex_allow` |
| Frame clock: 133 ms tick per session, silence for a missing frame, a late step is sent and marked late, the input side never waits, the backlog is cut by as many frames as the step was late | `duplex::clock::run` over the `FrameEngine` trait |
| Codec state per session in a pool, moved into the batch before the graph and back after it | `duplex_state_move`, `CudaCodec::launch_frame` with fixed parity sessions |
| `window_frames`: keep the prompt pages as a sink and the last W frames; the oldest page past the sink is reused for the newest positions, the rope position keeps counting through the mrope offset | `Engine::window`, `Runtime::set_position_offset` |
| Snapshots: KV blocks in use, delta net state, codec state, sampler state and frame counter, seven sections in the #326 `SessionStore` | `Engine::snapshot`, `restore`, `park`, `resume` |

The codec convs now pick a third `kir::conv` family, `Rows`, when the conv has few output rows and a long reduction (the 2048 wide blocks and the first and last up and down convs of a frame): one block per output channel, the reduction split over the block, eight rows at a time. On the real model this took the codec from 69.5 ms to 11.0 ms per frame at one session; the tile and direct families stay for long clips.

### Tiny fixture on GPU1

`tests/haru_duplex_gpu.rs` (`~/.agents/bin/gpu-run 1 --min-free 2000 -- <haru_duplex_gpu binary> --ignored --test-threads=1`, 4 s):

- The six frame greedy session from the fixture prompt and `session.user_audio` with the recorded noise gives `session.token` exactly on all seven frames, eager and through the frame graph, and a second session on the same slot repeats it.
- Fed the reference feedback each frame (its token and latent), the engine gives `session.user_latent` within 1.3e-6, `session.hidden` within 1.7e-4, `session.latent` within 1.8e-4 (per frame 2.3e-4, 6e-5, 1.8e-4, 1.8e-5, 2.2e-5, 2.9e-5, 2.1e-5) and `session.audio` within 2.4e-3, and its audio is bit identical to a fresh decode of the latents it emits. The codec alone gives `session.audio` within 8.5e-7 from `session.latent` (`haru_gpu.rs`).
- Free running, the latent error grows from 2.3e-4 to 3.2e-2 over six frames: the bf16 backbone floor (3e-4 relative, the same as the text only control in `haru_backbone_gpu.rs`) is fed back through the model's own latent. The issue's 1e-4 bound on `session.latent` and `session.audio` sits below that floor, so the test holds tokens exactly, the latent and user stream at the backbone bound with reference feedback, and the decode exactly.
- Two sessions batched in one step each give the same tokens and match the same session alone within 1e-5, with one captured graph for the batch of two.
- A snapshot after three frames, written to the store, resumed in a fresh engine, gives the same next three tokens, frame indices, latents and audio (within 1e-6).
- `relay` with an empty queue gives only PAD and EOP; a queued list comes out in order with only PAD and EOP between; `interrupt` returns what was said and what was not.
- `max_frames` 5 ends the session on frame 4 and refuses another step; `window_frames` 32 with a 64 position context runs 120 frames with finite audio and keeps at most 64 positions.

### Real Haru-4B on GPU1 (bf16, quiet GPU1, the user's app on GPU0)

Frame step time, `haru bench <dir> --batches 1,4,16 --frames 60`, every session fed silence after a card prompt:

| Sessions | Median ms | p95 ms | Budget ms |
|---|---|---|---|
| 1 | 20.3 | 20.4 | 133.3 |
| 4 | 30.2 | 30.2 | 133.3 |
| 16 | 54.5 | 54.7 | 133.3 |

At one session nsys gives 20.3 ms of kernel time per frame: backbone 7.6 ms (325 launches, close to the 8 ms estimate), codec 11.0 ms (344 launches), head 1.0 ms, duplex glue 0.7 ms. The codec is the remaining gap to the 8 ms of section 8; its largest kernels are the direct and rows convs of the 2048 wide blocks. Sixteen sessions fit in 41% of the frame.

Offline run of a recorded user clip (19.8 s, three TTS sentences with pauses, Kokoro Zoey), the English chat card from the reference package and a 13.8 s voice reference (Kokoro Harper), 149 frames each, through `haru run` and through the reference `haru-inference` (commit 7fe6b766, torch 2.11 CUDA in a private venv, `MemoryMax=16G`); Whisper large v3 turbo transcripts and RMS envelope metrics (100 ms windows, voiced above -40 dBFS) from `scripts/haru_eval.py`:

| Run | Whisper transcript | Voiced | Voiced over the user |
|---|---|---|---|
| specrt, sampled | Hi, I'm Haru. I'm doing great. How about you? Oh, that sounds wonderful. Did you hike? Yes, I have done it. It's very fun. | 34% | 2% |
| reference, sampled | Hello. I'm doing well, thank you. How are you? How was ba- How was ba- | 19% | 0% |
| specrt, greedy text | Hi there, I'm Haru. I'm doing great. How are you? | 17% | 2% |
| reference, greedy text | Hi, I'm Heru. I'm doing well, thank you. How are you? No, I haven't. What was it like? | 27% | 2% |

Both engines greet, answer the question in the user's pause and keep quiet while the user talks. The RMS envelope correlation between specrt and the reference is 0.32 sampled and 0.44 greedy, against 0.48 between the reference's own sampled and greedy runs: the audio noise stream differs between the engines, so frame for frame identity is not expected and the two engines differ about as much as one engine with two sampling settings.

Say queue, `haru say <dir> --sentences tests/fixtures/haru/say_sentences.txt --card card.txt --voice voice.wav`: 20 sentences, each queued after 8 frames of a relay session fed silence, up to 600 frames per sentence, Whisper on the produced audio:

- 9 of 20 queues were spoken to the end; in the other 11 the model kept choosing PAD and EOP, which the queue allows, and said 0 to 8 of the queued tokens in 600 frames (80 s).
- Word error rate over all 20 against the queued text: 3.07 (one sentence where the model laughed through 450 frames gives 44.5); median 1.0. Over the 9 completed: median 0.67, best 0.09 ("I set a reminder to call your sister on a Sunday. Evening.") and 0.18.
- The words that come out are the queued words, but the model wraps them in its own sounds ("Oh, God.", laughter, "Thank you." repeated in long silences).

So constrained text alone does not keep supervisor speech intelligible: the fallback of section 10 holds (text to speech for supervisor replies, Haru in `relay` for turn taking and backchannels) until the say path gets a stronger control, for example `eop_bias` while the queue is not empty or a card that tells the model it relays an assistant.
