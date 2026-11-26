# Speech in the engine: Whisper and text to speech (#1474)

Analysis and slice plan. CPU only, no engine code. Read on 26 Nov from:

- transformers 5.23 `models/whisper/` (`modeling_whisper.py`, `feature_extraction_whisper.py`, `generation_whisper.py`) and `generation/logits_process.py`, run on the CPU;
- `openai/whisper-tiny.en` (config, generation config, safetensors header), `openai/whisper-base` and `openai/whisper-large-v3-turbo` (config and generation config);
- agentbench `crates/agent-voice` (`speech/whisper.rs`, `transcriber.rs`, `voicer.rs`, `speaker.rs`, `phonon/*`, `speak/*`), `crates/server/src/{api.rs,speech.rs,voice.rs}`, `crates/app/src/voice/{native.rs,reader.rs}`, `crates/app/src/client.rs`, the workspace `Cargo.toml`, `.github/workflows/release.yml`, `packaging/android/build.sh`;
- `gradium/phonon-31m-en` at the revision agentbench pins (`8c64b55e`, the checkpoint in `~/.local/share/agentbench/models/phonon`) and the hub revision `92741c5b` that carries the transformers config, plus a voice file;
- model cards and configs of Kokoro-82M, Orpheus 3B, Pocket TTS, VibeVoice-Realtime-0.5B (config and the safetensors header by HTTP range, no weights) and SNAC 24 kHz.

Fixture and test: `crates/specrt/tests/fixtures/speech/` and `crates/specrt/tests/speech.rs` (section 6).

## 1. What runs today

agentbench `agent-voice` (feature `speech`, 6,460 lines under `src/speech`, through candle 0.11 with the `v0l/candle` fork at `a3ebb4c8` patched in the workspace manifest so CUDA loads at run time):

| Part | Files | Model | What it does |
|---|---|---|---|
| Speech to text | `whisper.rs` 212, `transcriber.rs` 79, `melfilters*.bytes` | 13 Whisper checkpoints (`voice/models.rs`), default `whisper-base.en` | `candle_transformers::models::whisper`; 30 s windows cut back to back with no overlap, zero padded; prefix SOT, `<\|en\|>` on multilingual models, transcribe, no timestamps; greedy, `suppress_tokens` only (no begin suppress), stops at EOT or `max_target_positions / 2`; no language detection, no fallback, no timestamps |
| Text to speech, current | `phonon/*` 3,700, `voicer.rs` 105 | `gradium/phonon-31m-en`, 102 voices in `gradium/phonon-voices` (`Harper`, default `Hadley`) | flow LM plus Mimi decoder, frame by frame, sentence chunks; seed 4242424242424242, temperature 0.3 |
| Text to speech, older | `speak/*` 1,880, `speaker.rs` 89 | `hexgrad/Kokoro-82M` | still compiled; only the `say` example and `Speaker` use it, the server speaks with Phonon |

The server routes are `POST /v1/audio/transcriptions`, `POST /v1/audio/speech`, `GET /v1/audio/voices`, `GET /v1/audio/models`, `GET /v1/audio/status` (section 4).

inference-bench: no speech to text, no text to speech. `crates/core/src/library.rs` `is_text_generation` lists `Whisper`, `TTS`, `Speech`, `ASR` as non text; `Spec::from_hf_config` refuses `model_type whisper` ("model_type whisper is not supported", checked by `speech.rs`).

## 2. Whisper

### 2.1 The model

`WhisperForConditionalGeneration` is an encoder and a decoder, all f32 in the releases, `proj_out` tied to `model.decoder.embed_tokens` (absent from the checkpoint).

```
mel [80 or 128, 3000]
conv1 k3 p1 + bias, gelu (erf)             [d, 3000]
conv2 k3 s2 p1 + bias, gelu                [d, 1500]
+ embed_positions [1500, d] (sinusoids at init, stored as a learned table)
N x { x += out(attn(ln1(x)))  x += fc2(gelu(fc1(ln2(x)))) }   bidirectional, LayerNorm with bias
layer_norm                                  encoder states [1500, d]

tokens -> embed_tokens + embed_positions[448]  (learned)
M x { x += self_attn(ln1(x)) causal with KV cache
      x += cross_attn(ln2(x), encoder states)
      x += fc2(gelu(fc1(ln3(x)))) }
layer_norm, logits = x @ embed_tokens.T
```

Attention: q, v and out projections have a bias, k does not; scale `head_dim ** -0.5`; no rope anywhere. Sizes:

| | d | enc / dec layers | heads | mel bins | vocab |
|---|---|---|---|---|---|
| tiny.en | 384 | 4 / 4 | 6 | 80 | 51,864 |
| base | 512 | 6 / 6 | 8 | 80 | 51,865 |
| large-v3-turbo | 1280 | 32 / 4 | 20 | 128 | 51,866 |

The turbo model is 32 encoder layers and 4 decoder layers: the encoder is most of the work.

### 2.2 Front end

`WhisperFeatureExtractor`: pad or cut to 30 s at 16 kHz (480,000 samples), `torch.stft(n_fft 400, hop 160, hann periodic, center reflect)`, drop the last frame (3000 frames), power `|X|^2`, slaney mel bank 201 x n (0 to 8 kHz, slaney norm), `log10(max(m, 1e-10))`, `max(v, global max - 8)`, `(v + 4) / 4`. The clamp is over the whole window, so the front end is a reduction over all frames, not a per frame map.

### 2.3 Token logic (`generation_whisper.py`)

- Prefix: `<|startoftranscript|>`, language (multilingual only), task (`transcribe` or `translate`), `<|notimestamps|>` unless timestamps are asked for. English only models have no language or task tokens in the prefix (`forced_decoder_ids [[1, 50362]]`).
- Language detection: one decoder step on SOT alone, argmax over the 99 language token logits.
- `suppress_tokens` every step, `begin_suppress_tokens` (space and EOT) on the first sampled step.
- Timestamps (`WhisperTimeStampLogitsProcessor`): timestamp tokens come in pairs, the first is at most `max_initial_timestamp_index` (50, 1 s), timestamps never go back, and if the summed probability of all timestamps beats the best text token, a timestamp is forced.
- No speech: `WhisperNoSpeechDetection` keeps the probability of `<|nospeech|>` at the SOT position; a window is skipped when it is above `no_speech_threshold` (OpenAI 0.6) and the average log probability is under `logprob_threshold`.
- Temperature fallback: given a list of temperatures (OpenAI's defaults, which transformers leaves to the caller: 0.0, 0.2, ... 1.0), a window is redone at the next temperature when its compression ratio (token ids as little endian bytes, raw length over `zlib.compress` length) is above `compression_ratio_threshold` (OpenAI 2.4) or its average log probability is under `logprob_threshold` (OpenAI -1.0).
- Long audio: seek through the audio by 30 s windows; with timestamps, the next window starts at the last complete timestamp; with `condition_on_prev_tokens`, the previous window's text goes in after `<|startofprev|>`.

### 2.4 Gap list against specrt

Printed by `cargo test -p inference-bench-specrt --test speech -- --nocapture the_engine_refuses`; each "exists" line is checked against the engine source, so a rename breaks the test instead of the list going stale.

| Piece | In specrt | Kind | Slice |
|---|---|---|---|
| Log-mel front end (STFT, mel bank, log10, global clamp, scale, 30 s pad) | landed (#1477): `kir::mel::LogMel` (window, the DFT table and slaney bank built from `preprocessor_config.json`), `graphs/speech.toml` `[[front]]`, the DFT and the bank as constant line conv GEMMs, `kir::cuda::emit_mel` for framing, power, log, the global max and the scale; host `speech::host`, CUDA `speech::cuda` | op | a |
| Conv stem, k3 p1 and k3 s2 p1 with GELU | landed (#1477): `graphs/speech.toml` `[[stem]]` nodes over `kir::conv` with `ConvShape::padded` (causal convs keep `PAD -1`), GELU fused into the conv epilogue by the fusion engine | graph entry | a |
| Exact GELU | `kir::expr` macro `gelu` | exists | |
| Learned position tables added to rows | landed (#1478): `table_add` in the `[[encoder]]` graph, `position_add` in the decoder graph from `[embedding] positions` | op | b |
| LayerNorm with bias | landed (#1478): `layer_norm` nodes in the encoder graph, `[norm] kind = "layer_norm"` (with `eps`) for the decoder | op | b |
| Linear with bias, k without | landed (#1478): `unbiased = ["self_attn.k_proj"]` on an encoder linear, `[attention] k_bias = false` and `[mlp] bias = true` for the decoder | op | b |
| Bidirectional encoder attention | landed (#1478): `graphs/speech.toml` `[[encoder]]`, a denoiser IR graph (`pack` with no norms, `attention`) | op | b |
| **Cross attention** | landed (#1478): decoder graph ops `cross_attention` (prefill, `vis_flash_kv` family) and `cross_attention_rows` (decode, one row per sequence: `vis_flash_kv_rows`, emitted beside `vis_flash_kv` in the joint vision module) | op | b |
| Cross K and V once per window, kept per sequence | landed (#1478): the decoder graph's `[[window]]` section runs `cross_kv` once per window; the family's `[cross]` names the per sequence `[[state]] cross_kv` (`2 x cross_rows x hidden` f32 per layer, tiny 51 KB) that every decode step reads | state | b |
| Decoder self attention with a KV cache, no rope | paged KV and `nope_all` (Nemotron-H) | exists | b |
| Tied head | `tie_word_embeddings` | exists | |
| Suppress, begin suppress, forced prefix | landed (#1479): `speech::decode::Rules::row` gives duplex `RowPlan { edits, allow }` rows, run on GPU1 by the duplex `duplex_edit_logits` and `duplex_allow` kernels (`speech::cuda::CudaRowPlans`) bit identical to the host; lists and ids read from `generation_config.json` through `[speech.decode]` | done | c |
| Language detection | landed (#1479): `Rules::detect`, one decoder step on SOT through the language allow row | done | c |
| Timestamps, no speech, temperature fallback, compression ratio, average log probability | landed (#1479): `Rules::row` (pairs, monotonic, `max_initial_timestamp_index`), `Rules::timestamp_rule` (summed timestamp probability), `no_speech_prob`, `compression_ratio` (zlib through `flate2` on the system zlib, byte equal to Python's `zlib.compress`), `avg_logprob`, `decide`; defaults in `[speech.decode]` | done | c |
| Long audio seek and previous text prompt | landed (#1479): `speech::transcribe::HostTranscriber::transcribe_mel` (windows of `mel.frames`, seek by the last timestamp pair, `<\|startofprev\|>` plus the last `max_target_positions / 2 - 1` tokens, conditioning dropped above `condition_below`) | done | c |

So two new ops (the mel front end, the decoder cross attention), one new state kind (cross KV), a new graph file and family for the encoder and decoder, and host decoding rules. Everything else is an op the denoiser IR already runs on CUDA (`denoiser_cuda.rs`) and on the host (`denoiser_host.rs`).

Where it lives. The encoder is a whole window at once with no state, which is the denoiser IR's shape (`DenoiserGraph`, `[[block.node]]` lists of `linear`, `layer_norm`, `attention`, `table_add`, `map`). The decoder is a token loop with a KV cache and needs the scheduler. The plan: `graphs/speech.toml` with an `encoder` graph in the denoiser node vocabulary plus the two new nodes (`log_mel`, `conv`), and a `family-whisper.toml` whose decoder uses `graphs/decoder.toml` with a `layer_norm` norm kind, biases, learned positions and the new `cross_attention` node. No Rust names the architecture (`no_arch_names` stays green); `model_type = "whisper"` appears only in the TOML.

What slice b built (#1478):

- Encoder: `graphs/speech.toml` `[[encoder]]` (`kind = "encoder-graph"`, `classes = ["whisper"]`), run by the host denoiser (`speech::host::HostEncoder`, `HostDenoiser::run` with named inputs and output). `encoder.out` within 9.9e-7 relative.
- Decoder: `specs/family-whisper.toml` maps the config onto the decoder names with `[remap]`, the checkpoint onto them with `[tensors]`, and declares `[norm] kind = "layer_norm"`, `[attention] k_bias = false`, `[mlp] bias = true` with the mapped GELU, `[embedding] positions`, `[rope] nope_all` and the tied head. `[[refuse]] part = "text"` keeps Whisper out of text chat; `Spec::from_hf_config_speech` loads it. The decoder graph block `cross_attention` (when `cross`) runs self attention, then `cross_attn_layernorm`, `cross_q`, cross attention and `cross_o`, then the dense MLP.
- Cross state: `[cross] source, rows, state`; the `[[state]] cross_kv` is per sequence; `Plan::window` holds the ops run once per audio window (`cross_kv`), so no decode step recomputes cross K and V.
- Host runner: `speech::decoder::HostDecoder` walks the lowered prefill and decode plans with per sequence self KV and `cross_kv`. Tiny fixture: `prefix.logits` 1.4e-6 and `greedy.logits` 1.5e-6 relative in f32 (bf16 weights 5.3e-3 and 9.5e-3), the 12 greedy tokens identical, two sequences batched bit identical to each alone.
- CUDA: `vis_flash_kv_rows` (one block per head and sequence, keys and values read straight from each sequence's `cross_kv`), `speech::cuda::CudaCrossRows`; on GPU1 within 2.1e-7 of the host for two sequences, a batched sequence bit identical to it alone. The token runtime (`Runtime::load`) refuses a spec with `[cross]` by name; running the whole Whisper decoder on CUDA through the scheduler is slice c onwards.

### 2.5 Accuracy bar

- Tiny fixture, CPU and GPU1: mel within 1e-4 absolute of `reference/mel.*` (the test's f64 reference DFT gets 8.3e-6), encoder states within 1e-4 relative of `encoder.out`, prefix and greedy logits within 1e-3 relative (bf16 weights: 2e-2) of `prefix.logits` and `greedy.logits`, the 12 greedy tokens and the detected language identical.
- Real models: word error rate against transformers `generate` (same model, f32, greedy, no timestamps) on a fixed clip set: LibriSpeech test-clean, the first 200 utterances (about 25 minutes, CC BY 4.0), plus 5 long clips over 30 s from the same set joined, normalised with `EnglishTextNormalizer` from the transformers whisper folder. Gate: WER within 0.3 points absolute of transformers on `whisper-base.en` and `whisper-large-v3-turbo`, and at least 98% of utterances with an identical transcript. The same clips through agentbench's candle path give the parity line for slice h.

### 2.6 Decoding rules (#1479)

`specs/family-whisper.toml` `[speech.decode]` holds what is not in the checkpoint: `time_precision` 0.02, `top_k` 50 (transformers' sampling default), `condition_below` 0.5, OpenAI's temperatures and thresholds as defaults, the names of the `generation_config.json` keys (`suppress_tokens`, `begin_suppress_tokens`, `lang_to_id`, `task_to_id`, `forced_decoder_ids`, `is_multilingual`) and `[speech.decode.params]` for the token ids (`no_speech = no_timestamps_token_id - 1`, `timestamp_begin = no_timestamps_token_id + 1`, `prev_cutoff = floor(max_target_positions / 2) - 1`). `speech::decode::Rules` merges config and generation config; `Options::from_rules` gives the family defaults, a request overrides them.

Order of a step, as transformers applies its processors: the row plan (suppress, begin suppress, timestamp masks) as logit edits plus an allow range, then the summed timestamp probability rule, then temperature and top k. Greedy picks the argmax; sampling draws from `sampler::uniform` (seeded splitmix), so sampled tokens are not torch's, and the fallback test feeds the sampled tokens transformers drew and checks that our rules allow each one.

Tiny fixture, CPU, f32 host decoder: greedy `generate` (12 tokens), the timestamp run with language detection (2 windows, 122 tokens, segments and times), the 1.5 s two window clip with previous text (seek 78, the second prompt starts with `<|startofprev|>`), and the fallback run (temperatures 0, 0.4, 0.8: a window redone and kept at 0.4, a window skipped as silence) are token identical to transformers, every compression ratio equal, average log probability within 2e-3 when we decode (1e-5 on the recorded scores), and every fallback and skip decision identical.

## 3. Text to speech

### 3.1 Phonon, what agentbench runs

`gradium/phonon-31m-en`, Apache 2.0, 41,096,035 parameters (31M without the codec), English, 24 kHz mono, 12.5 frames a second. Builds on Kyutai's Pocket TTS and Mimi. The card's English (US) table (Elo from 468 raters, 8,290 votes): phonon-31m-en 2098 Elo, 1.5% hard set LLM WER, 0.51% Seed-TTS LLM WER; Kokoro 2043, 3.5%, 0.45%; NeuTTS 2034, 8.7%, 0.62%. transformers 5.23 ships it as `gradium_phonon_tts` (a `pocket_tts` subclass), so a CPU reference exists for the language model; the codec is `transformers.models.mimi`.

| Part | Tensors | Parameters | dtype |
|---|---|---|---|
| Flow LM backbone, 6 layers, d 512, 8 heads of 64, MLP 2048 GELU, LayerNorm with bias, packed `in_proj` qkv, rope pairs, no biases | `flow_lm.transformer.*` | 18,886,656 | bf16 |
| Text conditioner (4001 x 512 table) and the fixed condition vectors | `flow_lm.condition_provider.*` | 2,386,944 | bf16 |
| LSD flow head: `input_proj`, `cond_embed`, two timestep embedders (sinusoid 256, MLP, RMS var norm, `alpha`), 6 adaLN res blocks (LayerNorm, SiLU MLP, gated residual), final adaLN layer | `flow_lm.flow_net.*` | 9,496,864 | bf16 |
| EOS head, out norm, input linear, latent mean and std, BOS | | 18,017 | bf16 |
| Mimi decoder transformer, 2 layers, d 512, layer scale, sliding context 250 | `mimi.decoder_transformer.*` | 6,297,600 | f32 |
| Mimi SEANet decoder: conv k7, 3 x (ELU, transposed conv ratio 6, 5, 4, one residual block with k3 and k1 convs), ELU, conv k3 | `mimi.decoder.*` | 3,974,945 | f32 |
| Mimi 12.5 Hz to 200 Hz depthwise transposed conv (kernel 32, stride 16, groups 512, no bias), quantizer output projection 32 to 512 | `mimi.upsample.*`, `mimi.quantizer.output_proj` | 32,768 | f32 |

There are no RVQ codebooks in use: the flow LM emits a continuous 32 wide latent per frame that goes straight to `quantizer.output_proj`. A voice is a `speaker_wavs` tensor `[1, 512, 125]` (10 s of the speaker Mimi encoder's output, a different 8 layer Mimi listed under `speaker_mimi` in the config) passed through `speaker_proj` and prefilled ahead of the text.

One frame: input linear of the previous latent (BOS first) plus the summed default conditions, one backbone position, out norm, EOS logit (`> eos_threshold 4.0` starts the end count), one flow head step from noise `N(0, temperature)` (`lsd_decode_steps = 1`), denormalise, Mimi decode to 1920 samples. Text is normalised, split into chunks of at most `MAX_TOKENS_PER_CHUNK` sentencepiece tokens, and each chunk is a fresh state primed with the voice.

### 3.2 Gap list against specrt

| Piece | In specrt (Haru) | Kind | Slice |
|---|---|---|---|
| Causal convs k7, k3 and k1 with constant pad and a streaming tail | `graphs/codec.toml` `conv` (causal, stride, depthwise), `kir::conv` line and rows families, double buffered tail per session | exists; **ELU** is a new `kir::expr` macro (`select(x > 0, x, exp(x) - 1)`), and a dilation attribute on `conv` (Phonon's `dilation_base 2` with one residual layer gives dilation 1, so not needed for this checkpoint) | e |
| Transposed conv with an overlap tail (ratios 6, 5, 4, kernel 2s) | `conv_up` lowered to a 2 tap causal conv plus interleave | exists | e |
| Depthwise transposed conv k32 s16, no bias | `conv_up` is dense only | **new**: a depthwise flag on `conv_up` (the lowering is the same 2 tap form per channel) | e |
| Mimi decoder transformer (LayerNorm, layer scale, rope pairs, a 250 frame window, Mimi's interleaved qkv packing) | the codec stack has no attention | **new graph**: the codec macro graph gains an `attention` node over a ring KV of `transformer_context` frames per session, or the stack is split and the transformer runs as decoder graph layers | e |
| Flow LM backbone: LayerNorm, packed qkv, rope pairs, dense GELU MLP, no bias | `NormKind` is `Rmsnorm` only; `fused_qkv` exists (Phi-3); rope `Pairs` exists; MLP is gated or `mapped` | **new**: a `layer_norm` norm kind and a dense (ungated) GELU MLP in the decoder graph, a `family-gradium_phonon_tts.toml` | f |
| LSD flow head | `graphs/meanflow.toml` and `flow_head.rs` (Haru): RMS modulate, SwiGLU, gated add, timestep embed, hoisting | close: **new** LayerNorm modulate with bias, SiLU MLP with bias, the RMS var timestep norm with `alpha`, the two embedder average; all `Map`, `Modulate` and `Linear { bias }` nodes of the denoiser IR | f |
| EOS head and frames after EOS | nothing | **new** sampler rule | f |
| Voice prefill (`speaker_proj` of 125 frames) | `[inputs]` nodes and embedding row prefill (Haru) | exists as a mechanism; **new** input node | f |
| Text normalisation, sentence chunking, sentencepiece | tokenizers in specrt | **new** host code, ported from `phonon/text` and `plan.rs` (1,640 lines, plain Rust, no candle) | f |
| Frame noise from a seed | `sampler::noise` (splitmix64 Box-Muller) | exists; numbers differ from `rand` `StdRng`, so audio is not bit identical to the candle path, only statistically | f |

What is reusable from Haru: the causal conv and transposed conv lowering, the streaming tail state, the codec batching over sessions, the flow head machinery and its hoisting, the input node prefill, the frame loop shape of `specrt::duplex`. What differs: Mimi's 200 Hz inner transformer with a 250 step window, ELU instead of ConvNeXt blocks, LayerNorm everywhere, and the 12.5 Hz frame (80 ms) against Haru's 7.5 Hz (133 ms).

### 3.3 Alternatives, from the cards

| Model | Licence | Size | Quality, from the card | Fit |
|---|---|---|---|---|
| `hexgrad/Kokoro-82M` | Apache 2.0 | 82M | "comparable quality to larger models"; Phonon's card puts it at 2043 Elo, 3.5% hard set WER against Phonon's 2098 and 1.5% | StyleTTS 2 plus ISTFTNet, a non autoregressive duration and pitch predictor, an LSTM text encoder, AdaIN 1D conv decoder, inverse STFT; needs G2P (agentbench `speak/g2p.rs`). None of it is in specrt (LSTM, AdaIN, iSTFT, alignment expansion); poor fit, and the quality is below Phonon |
| `canopylabs/orpheus-3b-0.1-ft` | Apache 2.0 (HF metadata), Llama 3.2 3B base | 3.3B, "4B params" F32 in the metadata | "~200ms streaming latency"; no numbers | The LM is plain Llama, which specrt already runs; audio is SNAC 24 kHz (19.8M, 3 codebooks at 12, 23, 47 Hz, 7 tokens per frame, licence not stated on its card), a small conv decoder. Good engine fit, but 40 times Phonon's size for a voice reader, and the SNAC decoder is new work anyway |
| `kyutai/pocket-tts` | not stated on the card, gated, with a prohibited use policy and per voice licences | 100M | "~200ms to get the first audio chunk", voice cloning from a wav | Phonon is built on it: same flow LM and Mimi. No gain over Phonon, and an unclear licence |
| `microsoft/VibeVoice-Realtime-0.5B` | MIT, but "limited to research purposes" and "We do not recommend using VibeVoice in commercial or real-world applications" | 1.02B in the safetensors (Qwen2.5-0.5B, 4 plus 20 layers) | LibriSpeech test-clean WER 2.00, SEED test-en 2.05, "~300 ms" first audible; English only | Its acoustic decoder **is the Haru codec decoder** tensor for tensor (276 tensors, 343,695,969 parameters, same strides and stages, `speech.rs` maps one onto the other), and the LM is Qwen2, which specrt runs. The best engine fit of all. But: no acoustic encoder is shipped ("Removed acoustic tokenizer to avoid users creating embedding on their own"), voices are a fixed embedded set, an audible AI disclaimer and a watermark are added in their pipeline, the DDPM head needs 20 steps a frame, and the card says research only |

### 3.4 Recommendation

Port **Phonon** (slices e and f), in the engine, as spec data.

- It is what agentbench speaks with today, so every voice the user picked (`Hadley`, `Harper` and the other 100) keeps working, and the parity gate in slice h compares like with like.
- Its card scores best of the open models it lists (2098 Elo, 1.5% hard set WER), Apache 2.0, 41M parameters. By the shapes in 3.1, a second of speech is about 7 GFLOP (flow LM and head 0.7 at 12.5 Hz, Mimi transformer 2.5 at 200 Hz, SEANet 4.0): real time on any card and on a CPU.
- Of the ten rows in 3.2, three exist outright in specrt from Haru and the diffusion work (causal codec convs with state, the transposed conv lowering, seeded noise) and three are close (the flow head IR, input node prefill, packed qkv and rope pairs); the new ones (LayerNorm kind, dense GELU MLP, ELU, depthwise `conv_up`, a windowed attention node in the codec graph, an EOS rule) are general ops the kir compiler rule wants anyway, not per model code.
- VibeVoice-Realtime would reuse the most engine code (its codec is already running in specrt), but its card restricts it to research, it cannot clone a voice, and it adds an audible disclaimer to every clip: wrong for a desktop app's voice. Kokoro and Orpheus need more new engine work than Phonon for less (Kokoro) or are far larger than the job needs (Orpheus).

### 3.5 Accuracy bar

- Tiny fixture (slice e and f make it from `transformers.models.gradium_phonon_tts` and `transformers.models.mimi` on the CPU, random weights, the real tensor kinds): flow LM hidden states and EOS logits within 1e-3 relative, the flow head latent within 1e-4 for recorded noise, Mimi decode within 1e-4, frame by frame equal to whole clip within 1e-5, as the Haru codec tests do.
- Real checkpoint: fed the reference's previous latent and noise each step (the `every_step_matches_ptts_when_fed_the_same_input` test in agentbench does this for candle against `ptts`), latents and audio within 1e-3 relative of the transformers reference for 50 frames.
- Speech quality: a fixed sentence set (the 30 sentences of `fixtures/haru/say_sentences.txt` plus 20 from LibriSpeech test-clean transcripts) spoken in Harper and Hadley by the engine and by the candle path, each transcribed by `whisper-large-v3-turbo` through the engine: WER of the engine's audio within 0.5 points of the candle path's, and no sentence cut short (duration within 10% of the candle one).

## 4. Serving: compatibility with agentbench's routes

The inference-bench server adds these routes in `crates/server/src/app.rs` `api_routes()`, so they get the same `auth` middleware (bearer key), `request_scope` (the `x-request-id` header, generated when missing, echoed back), the access log line and `inference_requests_total{endpoint, status}` as chat. `endpoint_label` gains `audio_transcriptions`, `audio_speech`, `audio_voices`, `audio_models`.

What agentbench's clients send and read today, which the new routes must accept unchanged:

| Route | agentbench server today (`api.rs`) | Clients | inference-bench must |
|---|---|---|---|
| `POST /v1/audio/transcriptions` | multipart; reads `file` (WAV, any rate, resampled to 16 kHz by `agent_voice::voice::audio`) and `model`; ignores other fields; answers `{"text"}`; errors `{"error": "..."}` with 400 or 500 | desktop `voice/native.rs` (reqwest multipart `file`, `model`, `response_format=json`), `agent-voice` `Client::transcribe` (hand built multipart, same three fields, model `whisper-1` when empty), Android through the same app code | accept `file`, `model`, `response_format` (`json` default, `text` answers the plain string, `verbose_json` adds `language`, `duration`, `segments`), `language` (ISO code, skips detection), `prompt` (previous text), `temperature`; ignore unknown fields; WAV PCM16 and f32 at any rate, mono or stereo downmixed; answer `{"text"}` for `json`; `model` `whisper-1` or an unknown id picks the loaded Whisper model, as agentbench's `Speech::pick` does |
| `POST /v1/audio/speech` | JSON `input` (required, at most 20,000 chars, 400 when empty or longer), `voice` (Phonon voice name, case insensitive; unknown names such as `alloy` get the default `Hadley`), `response_format`: `pcm` streams raw PCM16 LE mono as frames are made with `content-type: audio/L16`, `x-sample-rate: 24000`, `cache-control: no-store`; anything else is one `audio/wav`. `model` is ignored | `app/voice/reader.rs` posts `{"model": "kokoro", "input", "response_format": "pcm", "voice"?}` and reads the stream with the rate from `x-sample-rate`; 404 or 405 means "cannot read aloud yet", 401 "refused the token"; `Client::speak` posts `{"model": "kokoro", "input", "voice"?}` and reads a WAV | the same fields, limits, defaults, headers and status codes; `model` `kokoro`, `tts-1` or unknown picks the loaded speech model; `response_format` `wav` (default), `pcm` (streamed), and additively `mp3`/`opus` only if cheap (not needed by any client); `speed` accepted and ignored at first |
| `GET /v1/audio/voices` | `{"voices": [{"id", "label", "installed"}]}`, fetching the catalogue first | `app/client.rs` `voices()` | the same shape, from the voice catalogue the engine has; `label` as `Voice::label` makes it (`Hadley, American`) |
| `GET /v1/audio/models` | `{"object": "list", "data": [{"id", "object": "model", "label", "size_mb", "installed", "default"}]}` of Whisper models | `app/client.rs` `speech_models()` | the same shape for the speech to text models the library has |
| `GET /v1/audio/status` | `AudioStatus { device, reading_on, speaking_on }` | app settings | the same, `device` from the placement |
| `GET /v1/models` | (agentbench has none for speech) | | add `"type": "transcription"` and `"type": "speech"` to `ModelKind` labels (additive: `ModelKind` gains two variants, the existing `text`, `image`, `video`, `duplex` are unchanged) |

Admission and queueing follow the media job pattern (`specrt::media`, #1285): a speech job holds a slot, a full queue answers 429 with `Retry-After`, and `/v1/capabilities` lists the fields above per route. Body limit: a 30 minute WAV at 16 kHz PCM16 is 57.6 MB, so the transcription route needs its own `[media] max_audio_bytes` (default 64 MB) in the server config rather than the chat limit.

## 5. Migration of agentbench

The managed engine child from agentbench #1473 serves duplex (Haru), transcription (Whisper) and speech (Phonon) from one process. agentbench's own `/v1/audio/*` routes become a proxy to the child when it is running, so the desktop app, the Android app and the web build change nothing: they already talk to the server endpoints, never to candle directly (Android builds `--features local-speech`, but `build.rs` turns `local_speech` off on Android, so the phone already only records and posts).

Order:

1. Slices a to g land in inference-bench; the engine passes the WER and quality gates of 2.5 and 3.5.
2. Slice h: agentbench's server routes `/v1/audio/*` to the engine child, and the desktop app's local reading (`voice/native.rs` `local_speech` path) posts to its own server instead of loading candle. The candle path stays behind a setting `speech.backend = "candle"` as the fallback for one release while the parity run is repeated on the user's machine.
3. When parity holds on the same clips, delete the candle code.

What the deletion removes:

| What | Where |
|---|---|
| `agent-voice/src/speech/` except `download.rs`, `phonon/voices.rs`, `phonon/text` (moved or kept for the catalogue and the voice picker), about 6,000 lines including Kokoro (`speak/`, `speaker.rs`) and both mel filter files | `crates/agent-voice` |
| Features `speech`, `speech-cuda`, `speech-metal`, and the optional deps `candle-core`, `candle-nn`, `candle-transformers`, `tokenizers`, `byteorder`, `rand`, `rand_distr`; the six examples that need `speech` (`say`, `hear`, `phonon_match`, `phonon_normalize`, `phonon_say`, `voicer`) | `crates/agent-voice/Cargo.toml` |
| `[patch.crates-io]` for `candle-core`, `candle-kernels`, `candle-nn`, `candle-transformers` at `v0l/candle` `a3ebb4c8` | workspace `Cargo.toml` |
| `local-speech`, `cuda`, `metal` features of `agentbench` (app) and `cuda`, `metal` of `agentbench_server`, and `default = ["cuda"]` on both | `crates/app/Cargo.toml`, `crates/server/Cargo.toml`, `crates/app/build.rs` (`local_speech` cfg) |
| The CUDA toolkit install (`cuda: '12.8.1'`) and `--features cuda` on the x86_64 Linux and Windows release jobs, `--features metal` on macOS, `--features agentbench/local-speech` on aarch64 Linux: all become plain builds, and the Windows 2022 pin for nvcc can go | `.github/workflows/release.yml` |
| `--features local-speech` in the Android build (a no op there already) | `packaging/android/build.sh` |
| `--features local-speech` in the wasm build line (the speech module is already `cfg(not(target_arch = "wasm32"))`) | agentbench `README.md` |
| `Speech` (`speech.rs`) and `Voice` (`voice.rs`) as model holders; they become thin clients of the engine | `crates/server/src` |

What stays in agentbench: `agent-voice::voice` (audio helpers, WAV, resampling, the OpenAI style client, the model and voice lists the pickers show) and the Android app code, unchanged.

## 6. Fixture and test

`crates/specrt/tests/fixtures/speech/` (1.56 MB of the 2 MB cap the test sets; the Haru fixture is under 3 MB and the diffusion ones under 20 MB, there is no shared cap across fixtures beyond each test's own):

- `tiny-whisper/`: a `WhisperForConditionalGeneration` with random weights (`init_std 0.2`, LayerNorm weights around 1), d 64, 2 encoder and 2 decoder layers, 2 heads, 80 mel bins, a 1 s window (50 encoder positions), vocab 589 laid out like the multilingual one (512 text tokens, EOT, SOT, 4 languages, translate, transcribe, startoflm, startofprev, nospeech, notimestamps, 64 timestamps), the real tensor kinds of `whisper-tiny.en` and its tied head. `generation_config.json` and `preprocessor_config.json` in the real format.
- `tiny-whisper/reference/`: a 1 s chirp in noise and a 0.6 s cut of it, the mel bank, both log-mel inputs, the conv stem output, the encoder states, the language logits at SOT, the prefix logits, the 12 greedy step logits with suppress lists applied, and in `reference.json` the greedy tokens, `generate`'s tokens (equal) and the detected language. #1479 adds a 1.5 s gated chirp (`audio.long`, `mel.long`, two windows) and `runs`: `timestamps` (detection, timestamps), `long` (two windows, previous text) and `fallback` (temperatures 0, 0.4, 0.8 with thresholds that make one redo and one skip), each with every window's seek, decoder prompt and attempts (tokens, compression ratio, average log probability, no speech probability, decision), and the scores of the sampled attempt as sparse rows.
- Real model data, no weights: `whisper-tiny.en`, `whisper-base`, `whisper-large-v3-turbo` configs and generation configs, the `whisper-tiny.en` safetensors header; Phonon's config at the pinned revision and at the hub revision, its header and a voice header; VibeVoice-Realtime's config and its header without the acoustic encoder.

Regenerate (CPU, 4 s, transformers 5.23, torch 2.14):

```
systemd-run --user --wait --collect -P -q -p MemoryMax=8G -p MemorySwapMax=0 -E CUDA_VISIBLE_DEVICES= --working-directory=$PWD \
  ~/.venv-vllm/bin/python crates/specrt/scripts/tiny_whisper.py all --phonon ~/.local/share/agentbench/models/phonon
```

The weights are seeded and come out bit identical.

`tests/speech.rs` (CPU, 0.2 s):

| Test | Checks |
|---|---|
| `the_engine_refuses_whisper_and_says_which_ops_are_missing` | the four Whisper configs are refused by `from_hf_config`, `from_hf_config_duplex` and `describe_hf_config`; every "exists" row of the gap list still has its engine source (a rename breaks it); prints the Whisper and Phonon gap lists and the 9 pieces with no engine counterpart |
| `whisper_configs_have_the_shape_the_gap_list_assumes` | GELU, no embed scale, equal heads, 80 or 128 bins, turbo's 32 and 4 layers, English only against multilingual generation configs, 99 languages |
| `the_tiny_fixture_keeps_every_real_tensor_kind` | the fixture's tensor kinds equal `whisper-tiny.en`'s, the tied head, no `k_proj.bias`, the stem and table shapes |
| `the_log_mel_front_end_is_specified_exactly_by_the_reference` | a plain f64 DFT front end written from section 2.2 gives transformers' log-mel within 8.3e-6 for the full and the short clip (gate 2e-4): the spec of slice a's op |
| `the_token_logic_reproduces_transformers_generate_from_its_logits` | suppress and begin suppress applied to the recorded logits give the greedy tokens and `generate`'s tokens; language detection is the argmax over language tokens |
| `phonon_tensors_match_the_gap_list` | Phonon's parameter split, the depthwise upsample, no RVQ codebooks, packed qkv, LayerNorm with bias, layer scale, ratios, context 250, one LSD step, the voice shape |
| `the_vibevoice_acoustic_decoder_is_the_haru_codec_decoder_tensor_for_tensor` | the renamed VibeVoice decoder equals Haru's `codec.decoder` in all 276 shapes, 343,695,969 parameters |
| `the_speech_fixture_stays_under_2_mb` | size |

## 7. Slices

Each follows `docs/specrt-kir-compiler.md`: new ops are typed signature entries with kir descriptions, models are graph and family TOML, no Rust names an architecture. GPU work is GPU1 only, through `~/.agents/bin/gpu-run 1 --min-free 2000 -- <binary>`, compiled first, never GPU0, never the Spark. Real model downloads and real model GPU runs are their own slices.

| | Slice | Depends on |
|---|---|---|
| a | Log-mel front end op and the conv stem on the tiny fixture (CPU host and CUDA) | |
| b | Encoder graph and decoder family with cross attention and cross KV state, tiny fixture logits | a |
| c | Decoding rules and long audio, tiny fixture tokens | b |
| c2 | Real Whisper models: download, WER run against transformers on GPU1 | c |
| d | `/v1/audio/transcriptions`, `/v1/audio/models`, `/v1/models` type `transcription` | c |
| e | TTS codec: Mimi decoder graph reusing the Haru codec ops | |
| f | TTS model graph and sampling: Phonon flow LM, LSD head, EOS, voices, chunking, tiny fixture | e |
| f2 | Real Phonon: download, step match against transformers and the quality run on GPU1 | f |
| g | `/v1/audio/speech` with streaming PCM, `/v1/audio/voices`, `/v1/models` type `speech` | f |
| h | agentbench: speech through the engine child, candle kept as fallback, then removed | d, g, c2, f2, agentbench #1473 |
