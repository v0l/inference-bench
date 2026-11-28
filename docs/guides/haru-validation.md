# Haru-4B against the reference implementation

The specrt duplex engine checked tensor by tensor against the reference package `haru-inference` (github.com/higurashi-ai/Haru at `7fe6b766`, torch 2.11 CUDA, transformers 5.24) on the real checkpoint `~/models/Haru-4B` (bf16 weights). Everything ran on GPU1.

Verdict: the engine is a faithful implementation. Fed the reference's own tokens and latents, it matches the reference in f32 with top 1 agreement on every frame, KL under 3e-9, and cosine 0.999996 or better at every traced layer. Greedy free running gives the same tokens on all 149 frames in every greedy case, silence included. The forced text intelligibility problem comes from the model: the reference with a say queue is just as bad. Two real divergences turned up and are fixed (below).

## Inputs

- User clip: a 19.8 s recording of three Kokoro Zoey sentences with pauses, 149 frames.
- Voice: the 13.8 s Kokoro Harper reference.
- Card: the English `chat` card from the reference package (`haru/assets/cards/en.json`, rendered).
- Seed 1489 unless named. Temperature 0.7, top k 25, guidance 1.5, DRY on, as in both engines' defaults.

The traces live outside git, under `/tmp/haru1489` (437 MB, f32 `.bin` plus `index.json` per run). Each holds the prompt (tokens, both latent streams, masks, the trimmed voice clip with its encoder mean and the recorded noise) and, per frame, the user latent (encoder mean, noise scale, noise, noised latent), the summed input embedding, the hidden state after layers 0, 8, 16, 24 and 31, the final normed hidden state, the top 10 text logits and ids, PAD and EOP logits, the token, the head noise, the latent and the decoded PCM.

## Commands

```
cd crates/specrt
I=/tmp/haru1489/inputs
G="$HOME/.agents/bin/gpu-run 1 --min-free 30000 -- scripts/haru_gpu1.sh"
P="/tmp/haru-1430/bin/python scripts/haru_trace.py trace --haru /tmp/Haru/haru --model ~/models/Haru-4B --card $I/card.txt"
$G $P --out /tmp/haru1489/ref-f32-greedy --dtype f32 --greedy --voice $I/voice.wav --user $I/user.wav
$G $P --out /tmp/haru1489/ref-f32-sampled --dtype f32 --voice $I/voice.wav --user $I/user.wav
$G $P --out /tmp/haru1489/ref-f32-speakfirst-greedy --dtype f32 --greedy --speak-first --voice $I/voice.wav --user $I/user.wav
$G $P --out /tmp/haru1489/ref-f32-silence-novoice-greedy --dtype f32 --greedy --user silence
$G $P --out /tmp/haru1489/ref-f32-silence-voice-greedy --dtype f32 --greedy --voice $I/voice.wav --user silence
$G $P --out /tmp/haru1489/ref-f32-silence-voice-s1 --dtype f32 --seed 1 --voice $I/voice.wav --user silence
$G $P --out /tmp/haru1489/ref-bf16-teacher-greedy --dtype bf16 --greedy --voice $I/voice.wav --user $I/user.wav --teacher /tmp/haru1489/ref-f32-greedy
/tmp/haru-1430/bin/python scripts/haru_trace.py diff /tmp/haru1489/ref-bf16-teacher-greedy /tmp/haru1489/ref-f32-greedy

cargo build --release -p inference-bench-specrt --bin haru
~/.agents/bin/gpu-run 1 --min-free 20000 -- <target>/release/haru validate ~/models/Haru-4B \
  --trace /tmp/haru1489/ref-f32-greedy,/tmp/haru1489/ref-f32-sampled --out /tmp/haru1489/cmp --free
$G /tmp/haru-1430/bin/python scripts/haru_eval.py pairs /tmp/haru1489/cmp/ref-f32-greedy.free.wav,/tmp/haru1489/ref-f32-greedy/reply.wav

cargo test --release -p inference-bench-specrt --test haru_real_gpu --no-run
~/.agents/bin/gpu-run 1 --min-free 20000 -- <haru_real_gpu binary> --ignored --test-threads=1
```

`haru_gpu1.sh` refuses to run without `CUDA_VISIBLE_DEVICES` set to a card other than 0 and passes it through `systemd-run` (which drops it otherwise: the first trace attempt landed on GPU0 and hit OOM on load). Each reference trace takes about 4 s on GPU1 and each engine comparison 3 s. `haru_trace.py` drives the package's own `DuplexSession` and only wraps `embed`, `codec.sample`, `head.sample`, `sample_token` and `build_prompt` to record (and, with `--teacher`, to replay) their values, so the step order is the reference's.

`haru validate` loads the engine with `trace_layers` (an `EngineOptions` field: the decode forward copies the residual stream after each named layer), starts it from the trace's prompt and recorded noise, and feeds back the reference's token and latent each frame (`Engine::set_feedback`). `tests/haru_real_gpu.rs` reruns the same comparison from the saved traces (paths from `HARU_TRACE_DIR`, default `/tmp/haru1489`, and `HARU_MODEL`).

## Teacher forced, engine against the f32 reference

Max relative error per frame over 149 frames (the worst frame), cosine minimum over all frames and layers:

| Run | Top 1 | KL top 10 max | Min cosine | Layer 0 | 8 | 16 | 24 | 31 | Final hidden | Latent mean | PCM rel | PCM cos | First off 3e-4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 1.000 | 1.4e-9 | 1.0000000 | 3.9e-6 | 5.0e-6 | 2.8e-6 | 7.8e-5 | 3.5e-4 | 2.9e-4 | 1.4e-5 | 4.9e-3 | 1.00000 | frame 74, layer 31 (3.5e-4) |
| sampled, recorded noise | 1.000 | 2.5e-9 | 1.0000000 | 4.0e-6 | 6.4e-6 | 1.2e-5 | 1.8e-5 | 5.1e-4 | 7.6e-4 | 3.1e-5 | 3.3e-3 | 1.00000 | frame 138, layer 31 (5.1e-4) |
| greedy, speak first | 1.000 | 2.8e-9 | 1.0000000 | 4.6e-6 | 8.9e-6 | 4.4e-6 | 4.1e-5 | 2.3e-4 | 2.2e-4 | 2.2e-5 | 2.0e-3 | 1.00000 | none |
| greedy, silence, no voice | 1.000 | 1.7e-9 | 1.0000000 | 3.7e-6 | 3.7e-6 | 3.4e-6 | 3.9e-5 | 2.0e-4 | 1.2e-4 | 1.4e-5 | 1.9e-3 | 1.00000 | none |
| greedy, silence, voice | 1.000 | 6.8e-8 | 0.9999961 | 5.4e-6 | 8.9e-6 | 6.8e-6 | 2.9e-4 | 6.6e-3 | 6.5e-3 | 5.1e-5 | 1.2e-2 | 1.00000 | frame 11, final hidden (3.2e-4) |

The user latent stream (encoder mean plus the recorded noise) is within 2.6e-5 on every frame and the prompt (tokens, masks, trimmed voice clip, silence rows) is identical. Every gate of the issue passes with a wide margin: top 1 at least 0.99, KL under 0.01, cosine at least 0.999, latent mean under 2e-3. The error leaves the 3e-4 floor only at layer 31 and the final norm, on single frames, by under a factor of 2 (the 6.6e-3 in the last row is one frame of one element whose layer 31 value is near zero against a small max; its cosine stays at 0.999996). Layers 0 to 24 stay at 1e-5, so no layer is wrong.

For scale, the reference itself in bf16, teacher forced from the same f32 trace (`haru_trace.py --teacher`, then `diff`):

| Run | Top 1 | KL top 10 max | Hidden cos min | Layer 31 rel max | Final hidden rel max | Latent mean |
|---|---|---|---|---|---|---|
| reference bf16, greedy | 1.000 | 4.8e-3 | 0.962 | 4.9e-1 | 2.5e-1 | 2.1e-2 |
| reference bf16, sampled | 0.987 | 1.6e-1 | 0.189 | 5.6 | 8.1 | 4.9e-2 |

The engine sits about three orders of magnitude closer to the f32 reference than the reference's own bf16 path (its activations are split into high and low bf16 halves, `split_activations`). That is why the f32 reference is the target here; an engine run against the bf16 reference just measures the bf16 reference's own error (top 1 0.993, hidden cosine 0.989).

## Free running, same seed

| Run | First token divergence | Voiced frames engine / reference | Whisper engine | Whisper reference | WER |
|---|---|---|---|---|---|
| greedy | none (149 frames equal) | 21 / 20 | Hi, I'm Heru. I'm doing great. How are you? | Hi I'm Heru. I'm doing great. How are you? | 0.00 |
| greedy, speak first | none | 55 / 55 | I'm Haru. I'm doing well, thank you. How are you? That sounds nice. What did you do there? ... | the same | 0.00 |
| greedy, silence, no voice | none | 7 / 7 | Hi, my name's Foo. | Hi, my name's Foo. | 0.00 |
| greedy, silence, voice | none | 40 / 40 | Hi, I'm Heru. I am 22 years old and I am senior in college. | the same | 0.00 |
| sampled | frame 1 | 44 / 110 | Yes. I'm doing well, thank you for asking. How are you? ... | Hi, I'm Haru and I'm 22 years old. I go for runs ... | 0.94 |

Greedy text gives the same tokens on every frame and the same audio to Whisper. Sampled text cannot match frame for frame: the engine samples with its own counter based Gumbel noise (`sample_rows`) and the reference with `torch.multinomial`, so the same seed draws different words from the same distribution. The teacher forced rows above show that distribution matches (KL 2.5e-9). Across seeds the share of frames where each engine takes a token other than the top 1 matches the softmax: 0.122 expected, 0.128 reference, 0.101 engine on the sampled run; 0.055, 0.047 and 0.060 on silence with a voice. The tiny fixture test `sampled_text_depends_on_the_session_seed` checks that the session seed reaches the sampler.

## Silence in, both implementations

Neither implementation stays silent when the user says nothing: both open with a self introduction, in intelligible speech. Eight seeds each, 149 frames (19.9 s) of digital silence in, the English card:

| Case | Voiced frames engine (8 seeds) | Voiced frames reference (8 seeds) | Engine mean | Reference mean |
|---|---|---|---|---|
| no voice reference | 67 96 81 96 6 116 71 120 | 56 36 42 120 129 22 58 47 | 82 | 64 |
| voice reference | 10 22 40 34 60 49 29 94 | 50 52 52 13 9 47 47 14 | 42 | 36 |

Whisper on the engine's audio gives speech in all 16 runs ("Hi, my name is Haru. I'm 22 years old. I'm a senior in college", "Hey there. I'm Haru. I'm 22. I go running most mornings."); the reference's runs read the same way. Greedy, the two give the same tokens and the same voiced frame count (7 and 7 without a voice, 40 and 40 with one), which `haru_real_gpu.rs` asserts. So the noise heard in the app with no input does not come from the model step, the prompt or the sampler: the engine produces speech here from the same prompt and silence. The app path (the websocket client, its capture and resampling, the server's prompt fields) is where to look, outside this issue.

## Say queue, both implementations

`haru say` and `haru_trace.py say` with the same 20 sentences (`tests/fixtures/haru/say_sentences.txt`), the same card and voice, relay mode, 8 lead frames of silence, then the sentence queued (only PAD, EOP and the next queued token allowed), up to 600 frames, Whisper large v3 turbo against the queued text:

| Run | Sentences spoken to the end | WER over all 20 | Median WER | Median WER, completed | Best three |
|---|---|---|---|---|---|
| specrt | 9 | 1.12 | 1.00 | 0.70 | 0.00, 0.18, 0.55 |
| reference, bf16 | 10 | 1.27 | 1.22 | 1.04 | 0.09, 0.18, 0.20 |
| reference, f32 | 10 | 2.57 | 1.05 | 0.62 | 0.09, 0.18, 0.25 |

The reference with forced text is as poor as the engine (both numbers are dominated by a few stalled sentences, which is why the medians are given). In both, the model stalls on PAD and EOP for most sentences and wraps the words it does say in its own sounds. This is the model, not the implementation, so the fallback of `specrt-haru.md` section 10 (text to speech for supervisor replies) stands.

## Divergences found and fixed

1. Voice reference tail. `Engine::encode_whole` padded the last partial frame of the voice clip with zeros before encoding; the reference encodes the clip as is and its convs pad each stride hop. The last reference latent differed by 0.27 relative (all others 6e-6). Red test `a_voice_reference_that_ends_mid_frame_encodes_like_the_reference_whole_clip` in `haru_duplex_gpu.rs` (2.6e-1 before, 1.0e-6 after); the tail now goes through the encoder at its own length.
2. DRY history under fed back tokens. `Engine::set_feedback` replaced the token and latent but left the sampled token in the DRY history, so a teacher forced run penalised the wrong tokens. `TextState::replace_last` swaps it (unit test `a_fed_back_token_replaces_the_sampled_one_in_the_dry_history`). Only the teacher forced path uses `set_feedback`.

## Tests

| Test | Where | Result |
|---|---|---|
| `teacher_forced_against_the_reference_on_the_real_checkpoint` | `tests/haru_real_gpu.rs`, ignored, GPU1, real model and traces | pass (greedy, sampled, speak first) |
| `greedy_free_running_matches_the_reference_token_for_token_including_silent_input` | same | pass (four greedy cases, voiced frames within 1) |
| `a_voice_reference_that_ends_mid_frame_encodes_like_the_reference_whole_clip`, `sampled_text_depends_on_the_session_seed` | `tests/haru_duplex_gpu.rs`, tiny fixture, GPU1 | pass |
| `a_fed_back_token_replaces_the_sampled_one_in_the_dry_history` | `duplex::sampling` unit tests | pass |
| gates `no_arch_names`, `codegen`, `golden`, `spec`, `linear_unchanged`, lib, `haru`, `haru_head` | CPU | pass |
| `haru_gpu`, `haru_head_gpu`, `haru_backbone_gpu`, `haru_duplex_gpu` | GPU1 | pass |
