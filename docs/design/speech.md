# Speech: transcription, text to speech and duplex

Three audio paths, all spec data over graph TOML and kir kernels, with no Rust naming a model: Whisper transcription, Phonon text to speech, and Haru full duplex speech (model facts in [../models/haru-4b.md](../models/haru-4b.md)). The codec and flow head machinery is shared with [diffusion.md](diffusion.md).

## Whisper

`specs/family-whisper.toml` and `graphs/speech.toml`; `Spec::from_hf_config_speech` loads it and `[[refuse]] part = "text"` keeps it out of text chat.

- Front end (`[[front]]`): `kir::mel::LogMel` built from `preprocessor_config.json` (window, DFT table, slaney mel bank), the DFT and bank as constant line conv GEMMs, `kir::cuda::emit_mel` for framing, power, log, the global max clamp and the scale. Conv stem (`[[stem]]`) over `kir::conv` with GELU fused in the epilogue.
- Encoder (`[[encoder]]`, `kind = "encoder-graph"`): a denoiser IR graph with `layer_norm`, linears with bias (`unbiased = ["self_attn.k_proj"]`), bidirectional attention and the position `table_add`.
- Decoder: `graphs/decoder.toml` with `[norm] kind = "layer_norm"`, `[attention] k_bias = false`, `[mlp] bias = true`, `[embedding] positions`, `[rope] nope_all` and the tied head; the `cross_attention` block runs self attention, cross attention over the per sequence `[[state]] cross_kv` (written once per window by `Plan::window`), then the MLP. `[remap]` and `[tensors]` map config and checkpoint names.
- Decoding (`speech::decode::Rules`, `[speech.decode]`): forced prefix, suppress and begin suppress, language detection (one step on SOT), timestamps (pairs, monotonic, `max_initial_timestamp_index`, the summed timestamp rule), no speech, compression ratio (zlib, byte equal to Python), average log probability and temperature fallback, with OpenAI's thresholds and temperatures as defaults; ids and lists come from `generation_config.json`. Row plans run on the GPU through the duplex `duplex_edit_logits` and `duplex_allow` kernels. Long audio (`speech::transcribe`) seeks by the last timestamp pair and conditions on previous text below `condition_below`.
- Engines: `speech::gpu::GpuEngine` (every kernel from kir; both attentions are `vis_flash_kv_rows`) and the host engine (`speech::host`, `speech::decoder::HostDecoder`) walk the same lowered plans. Weights are f32 (F16 checkpoints read exactly).

## Phonon text to speech

`specs/family-gradium_phonon_tts.toml`, `graphs/lsd.toml` and the `mimi-decoder` codec in `graphs/codec.toml`; `Spec::from_hf_config_tts`.

- Flow LM: `graphs/decoder.toml` with LayerNorm, `fused_qkv`, `[rope] interleaved`, a mapped GELU MLP and a named one row EOS head (`[embedding] head`). Inputs are `[[inputs.node]]` `group_map` descriptions: `stream_replace` puts `input_linear(previous latent) + conditions` at the latent placeholder (the learnt BOS latent first), `stream_replace_unbiased` puts `speaker_proj(voice frame)` at each voice placeholder. Default conditions are folded at load from `[tts.condition]`.
- LSD flow head (`graphs/lsd.toml`, bound by `[head]`): timestep embedders from the stored `freqs` table, RMS norm with `alpha`, LayerNorm modulate res blocks with a gated residual (folded into fc2 by the fusion pass), the step `z + h_step * u` as an expression, so `lsd_decode_steps` sets the step count; outputs the normalised latent and the codec input.
- Mimi decoder: the output projection as a k1 conv, a depthwise `conv_up` (lowered to a 2 tap form per channel), two 200 Hz transformer layers with an `attention` node over a ring KV of `transformer_context` steps per session, then SEANet with ELU as a conv input prologue (`kir::expr` `elu`). 1920 samples a frame; streaming is per session and batched like the Haru codec.
- Sampling and text (`[tts.params]`, `[tts.tokens]`): EOS threshold from the config, frames after EOS (1, or 3 for short chunks), temperature 0.3, 50 tokens a chunk, frame budget `ceil((tokens / 3 + 2) * frame_rate)`; `speech::tts::text` normalises numbers, currency, dates, times and URLs and `speech::tts::plan` splits sentences. Frame noise is `flow_head::frame_noise` (seeded), so audio is statistically, not sample, equal to other implementations.
- Voices: `[tts.voices]` names the catalogue repo, file and default (`gradium/phonon-voices`, `voices.jsonl`, `Hadley`); `speech::tts::voices::Catalogue` looks in the model folder, then the hub cache, then fetches. A voice is prefilled once per speaker and its KV kept.
- `speech::tts::service::Speaker` runs the flow LM on the host decoder and the LSD head and Mimi through kir kernels on the GPU (or all on the host), handing each frame to the caller as it is made.

## Routes

In `api_routes()` (`crates/server/src/audio.rs`) behind the same auth, `x-request-id`, access log and `inference_requests_total` as chat. They accept agentbench's clients unchanged.

| Route | Behaviour |
|---|---|
| `POST /v1/audio/transcriptions` | multipart `file` (WAV PCM16 or f32, any rate, stereo downmixed), `model` (`whisper-1` or unknown picks the loaded Whisper), `response_format` (`json`, `text`, `verbose_json` with language, duration, segments), `language`, `prompt`, `temperature`; body limit `[media] max_audio_bytes` (64 MB, 413 above) |
| `POST /v1/audio/speech` | JSON `input` (1 to 20,000 chars), `voice` (case insensitive, unknown names get the default), `model` (`kokoro`, `tts-1` or unknown pick the loaded speech model), `response_format` `pcm` (streamed PCM16, `content-type: audio/L16`, `x-sample-rate: 24000`) or WAV, `seed`, `speed` (ignored) |
| `GET /v1/audio/voices`, `/v1/audio/models`, `/v1/audio/status` | the agentbench shapes |
| `/v1/models` | `type: "transcription"`, `"speech"`, `"duplex"` |

Jobs take the media gate's slots (`[images] jobs_per_gpu`, `queue_depth`); a full queue is a 429 with `Retry-After`. `/v1/capabilities` lists every field.

## Duplex sessions

`specrt::duplex` runs sessions of a duplex model: one 133 ms user frame in, one model frame out, batched over sessions. Everything model specific is in the family's `[duplex]`: frame and latent sizes from the config, noise divisor, PAD, EOP and EOS ids, the card template (`[duplex.card]`), declared session state (`kv`, `recurrent`, `codec`, `sampler`, `frame`), `max_frames` (27,000), `window_frames` (0, off) and `[duplex.sampling]` (temperature 0.7, top k 25, `eop_bias` 0, one forced text frame, f32 PAD and EOP logits, DRY 0.8, 1.75, 1, 1024, 32, 32). `Spec::from_hf_config_duplex` reads it.

| Piece | Where |
|---|---|
| Prompt: card, voice reference (trimmed at -45 dBFS, at most 180 s, encoded and noised), `speak_first`, forced first text | `duplex::prompt`, `Engine::start` |
| Frame step: encode, noise, sum the streams, backbone decode rows, logit edits and allow list, sampling, MeanFlow head, decode; one CUDA graph per padded batch size | `Engine::step` |
| Text controls: say queue (only PAD, EOP and the next queued token), `relay` (only PAD and EOP) and `free`, `interrupt` | `duplex::sampling::TextState` |
| Clock: fixed tick per session, silence for a missing frame, late steps marked, backlog cut | `duplex::clock::run` |
| Window: prompt pages kept as a sink plus the last W frames, positions keep counting | `Engine::window` |
| Snapshots: KV blocks, delta-net state, codec, sampler and frame counter in the session store | `Engine::snapshot`, `park`, `resume` |
| Service: one clock thread per model, sessions through its event queue | `duplex::service::Service` |

Codec convs pick the `kir::conv` `Rows` family when a conv has few output rows and a long reduction (69.5 to 11.0 ms a frame on Haru).

Server (`crates/server/src/duplex.rs`): `GET /v1/duplex` (native), `GET /v1/realtime` (OpenAI Realtime subset: `session.update`, `input_audio_buffer.append`, `response.output_audio.delta` and transcript deltas; turn events get an error saying why), `POST /v1/duplex/voices`. The client sends `start` (`model`, `card`, `voice`, `speak_first`, `guidance`, `temperature`, `mode`, `seed`, `resume`), then binary PCM16 frames of 3200 samples at 24 kHz; the server answers `ready`, one binary frame per tick with a `frame` event naming the user frame it answers, `text`, `late`, `end`; controls `say`, `mode`, `interrupt`, `stop`. `[duplex]` server config: `sessions_per_gpu`, `queue_depth` (full answers `code: "full"`), `keep_idle_secs` (snapshot a dropped socket for resume), `max_voice_seconds` (60), `allow_replay` (test hook). CLI: `inference-bench duplex --wav in.wav --out out.wav [--card --voice --realtime --events]`.

## Measured state

| Case | Value | Source |
|---|---|---|
| whisper-base.en, 205 LibriSpeech test-clean clips, engine on GPU1 | WER 6.08, equal to transformers, 205 of 205 transcripts identical, 17.7 s for 1,357 s of audio | #1480 |
| whisper-large-v3-turbo, same clips | WER 2.26, equal to transformers, 205 of 205 identical, 170 s | #1480 |
| Phonon, 80 clips (40 sentences, Harper and Hadley), engine audio transcribed by the engine's turbo | WER 1.70, equal to the candle path; durations 0.955 to 1.063 of candle | #1484 |
| Phonon step match against transformers, 50 frames | audio within 1.4e-4 | #1484 |
| Haru-4B frame step, bf16, GPU1, 1 / 4 / 16 sessions | 20.3 / 30.2 / 54.5 ms against a 133 ms frame | #1430 |

The Phonon flow LM still runs on the host decoder, so speech is slower than real time on long text (452.7 s for 361 s of audio in the quality run). The say queue on Haru finished 9 of 20 sentences, so supervisor replies go through text to speech with Haru in `relay` (#1430).

## Tests

`crates/specrt/tests/speech.rs` (CPU and GPU1: refusals, tiny Whisper, Mimi and Phonon fixtures against transformers, decoding rules, the LSD head, the GPU transcriber), the `speech::tts` unit tests, `crates/specrt/tests/haru*.rs` (codec, head, backbone, duplex sessions, the real model against traces), `crates/server/tests/audio.rs`, `speech.rs`, `duplex.rs` (CPU, agentbench's exact bodies) and `audio_gpu.rs`, `speech_gpu.rs`, `duplex_gpu.rs` (GPU1). Fixtures: `tests/fixtures/speech/` (`scripts/tiny_whisper.py`, `tiny_mimi.py`, `tiny_phonon.py`), `tests/fixtures/haru/` (`scripts/tiny_haru.py`). Real model runs: `scripts/whisper_wer.py`, `phonon_reference.py`, `phonon_quality.py`, `agentbench_speech.py`; validation of Haru against its reference: [../guides/haru-validation.md](../guides/haru-validation.md).
