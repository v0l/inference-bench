# Haru-4B

Full duplex speech to speech: `higurashi-ai/Haru-4B` and its reference code `github.com/higurashi-ai/Haru` at commit `7fe6b766`. How the engine runs it (sessions, the frame clock, the websocket, measured step times) is in [../design/speech.md](../design/speech.md); validation against the reference is [../guides/haru-validation.md](../guides/haru-validation.md). Fixture and tests: `crates/specrt/tests/fixtures/haru/`, `crates/specrt/tests/haru.rs` and the `haru_*_gpu.rs` suites.

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

