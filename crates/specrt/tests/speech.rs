use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn dir() -> PathBuf {
    root().join("tests/fixtures/speech")
}

fn tiny() -> PathBuf {
    dir().join("tiny-whisper")
}

fn json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap()
}

fn header(name: &str) -> BTreeMap<String, (String, Vec<usize>)> {
    let rows: Vec<(String, String, Vec<usize>)> = serde_json::from_str(&std::fs::read_to_string(dir().join(name)).unwrap()).unwrap();
    rows.into_iter().map(|(n, d, s)| (n, (d, s))).collect()
}

fn kinds<'a>(names: impl Iterator<Item = &'a String>) -> BTreeSet<String> {
    names
        .map(|n| n.split('.').map(|p| if p.parse::<usize>().is_ok() { "N" } else { p }).collect::<Vec<_>>().join("."))
        .collect()
}

fn reference(key: &str) -> (Vec<f32>, Vec<usize>) {
    let meta = json(&tiny().join("reference.json"));
    let entry = &meta["files"][key];
    let shape: Vec<usize> = entry["shape"].as_array().unwrap_or_else(|| panic!("{key}")).iter().map(|d| d.as_u64().unwrap() as usize).collect();
    let bytes = std::fs::read(tiny().join("reference").join(entry["file"].as_str().unwrap())).unwrap();
    let values: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    assert_eq!(values.len(), shape.iter().product::<usize>(), "{key}");
    (values, shape)
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect()
}

struct Need {
    piece: &'static str,
    state: &'static str,
    evidence: Option<(&'static str, &'static str)>,
    slice: &'static str,
}

const WHISPER: &[Need] = &[
    Need { piece: "log-mel front end: hann window 400, hop 160, power STFT, slaney mel bank 0 to 8 kHz, log10 clamp 1e-10, max - 8, (x + 4) / 4, 30 s pad", state: "exists: kir::mel description, DFT and mel bank as constant line conv GEMMs, graphs/speech.toml", evidence: Some(("kir/src/mel.rs", "pub fn dft_table")), slice: "a" },
    Need { piece: "conv stem: conv1d k3 p1 then k3 s2 p1 with exact GELU, centred padding", state: "exists: graphs/speech.toml nodes over kir::conv with a centred pad", evidence: Some(("kir/src/conv.rs", "pub fn padded(")), slice: "a" },
    Need { piece: "exact GELU", state: "exists", evidence: Some(("kir/src/expr.rs", "(\"gelu\", \"0.5 * x * (1.0 + erf(x * 0.7071067811865476))\")")), slice: "-" },
    Need { piece: "learned absolute positions added to the stem output and to token embeddings", state: "exists: table_add in the encoder graph, position_add in the decoder graph from [embedding] positions", evidence: Some(("specrt/src/graph.rs", "\"position_add\" => Op::PositionAdd")), slice: "b" },
    Need { piece: "LayerNorm with bias", state: "exists: layer_norm in the encoder graph, [norm] kind = layer_norm for the decoder", evidence: Some(("specrt/src/spec.rs", "#[serde(rename = \"layer_norm\")]")), slice: "b" },
    Need { piece: "linear with bias, k_proj without bias", state: "exists: unbiased parts on an encoder linear, [attention] k_bias = false for the decoder", evidence: Some(("specrt/src/denoiser.rs", "pub fn biased_parts")), slice: "b" },
    Need { piece: "bidirectional encoder self attention", state: "exists: graphs/speech.toml [[encoder]] over the denoiser IR", evidence: Some(("specrt/graphs/speech.toml", "kind = \"encoder-graph\"")), slice: "b" },
    Need { piece: "cross attention from decoder rows to encoder states", state: "exists: decoder graph op cross_attention (prefill) and cross_attention_rows (one row per sequence)", evidence: Some(("specrt/src/graph.rs", "\"cross_attention_rows\" => Op::CrossAttention { decode: true }")), slice: "b" },
    Need { piece: "cross attention K and V computed once per window and kept per sequence", state: "exists: the [[window]] cross_kv op and the per sequence cross_kv [[state]] named by [cross]", evidence: Some(("specrt/graphs/decoder.toml", "[[window]]\nop = \"cross_kv\"")), slice: "b" },
    Need { piece: "decoder causal self attention with a paged KV cache, no rope", state: "exists (paged KV, nope_all)", evidence: Some(("specrt/src/family.rs", "pub nope_all: bool")), slice: "b" },
    Need { piece: "tied output head", state: "exists", evidence: Some(("specrt/src/spec.rs", "\"tie_word_embeddings\"")), slice: "-" },
    Need { piece: "suppress and begin suppress lists, forced prefix", state: "exists: speech::decode::Rules rows as duplex RowPlan edits and allow lists, read from generation_config.json", evidence: Some(("specrt/src/speech/decode.rs", "pub fn row(&self, generated: &[u32], timestamps: bool) -> RowPlan")), slice: "c" },
    Need { piece: "language detection over the language token logits at step 0", state: "exists: Rules::detect, one step on SOT through the language allow row", evidence: Some(("specrt/src/speech/decode.rs", "pub fn detect(&self, sot_logits: &[f32]) -> u32")), slice: "c" },
    Need { piece: "timestamp rules, no speech probability, temperature fallback, compression ratio, avg logprob", state: "exists: Rules::row, timestamp_rule, no_speech_prob, compression_ratio (zlib), avg_logprob, decide; thresholds in [speech.decode]", evidence: Some(("specrt/src/speech/decode.rs", "pub fn decide(")), slice: "c" },
    Need { piece: "long audio: 30 s windows with seek by the last timestamp, previous text as prompt", state: "exists: speech::transcribe::HostTranscriber::transcribe_mel", evidence: Some(("specrt/src/speech/transcribe.rs", "pub fn transcribe_mel(")), slice: "c" },
];

const TTS: &[Need] = &[
    Need { piece: "Mimi decoder SEANet: causal conv k7 and k3 with ELU, constant pad, residual blocks k3 and k1", state: "exists: graphs/codec.toml mimi-decoder, ELU as a kir::expr macro folded into the conv input (Prologue::Act)", evidence: Some(("kir/src/expr.rs", "(\"elu\", \"select(x > 0.0, x, exp(x) - 1.0)\")")), slice: "e" },
    Need { piece: "Mimi transposed conv upsampling (ratios 6, 5, 4) with an overlap tail", state: "exists as codec conv_up", evidence: Some(("specrt/src/codec/graph.rs", "\"conv_up\" =>")), slice: "e" },
    Need { piece: "Mimi depthwise transposed conv 12.5 Hz to 200 Hz upsample (groups = channels, no bias)", state: "exists: conv_up groups = channels, the line_dwup kernel", evidence: Some(("specrt/src/codec/host.rs", "pub fn lower_dw_up(")), slice: "e" },
    Need { piece: "Mimi decoder transformer at 200 Hz: 2 layers, LayerNorm, layer scale, rope pairs, sliding context 250", state: "exists: codec attention node over a per session ring KV of context steps, layer_norm node", evidence: Some(("specrt/src/codec/graph.rs", "\"attention\" =>")), slice: "e" },
    Need { piece: "flow LM backbone: 6 layer LayerNorm transformer, packed qkv, rope pairs, GELU MLP, no bias", state: "new family file and decoder graph entries (LayerNorm kind, GELU dense MLP)", evidence: Some(("specrt/src/spec.rs", "pub enum NormKind")), slice: "f" },
    Need { piece: "LSD flow head: adaLN res blocks with LayerNorm, two timestep embedders averaged, one step", state: "close to graphs/meanflow.toml: LayerNorm modulate, bias, rms-var timestep norm are new", evidence: Some(("specrt/src/denoiser.rs", "\"modulate\" =>")), slice: "f" },
    Need { piece: "EOS head with threshold and frames after EOS", state: "new sampler rule", evidence: None, slice: "f" },
    Need { piece: "voice conditioning: 125 Mimi frames through speaker_proj, prefilled before the text", state: "new input node; prefill of embedding rows exists", evidence: Some(("specrt/src/plan.rs", "Splice")), slice: "f" },
    Need { piece: "sentence chunking and normalisation, sentencepiece tokenizer", state: "new host code", evidence: None, slice: "f" },
];

fn engine_source(rel: &str) -> String {
    let path = root().parent().unwrap().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn render(title: &str, needs: &[Need]) -> String {
    let mut out = format!("== {title}\n");
    for n in needs {
        out.push_str(&format!("  [{}] {}: {}\n", n.slice, n.piece, n.state));
    }
    out
}

#[test]
fn the_engine_refuses_whisper_as_text_chat_and_says_which_ops_are_missing() {
    for name in ["tiny-whisper/config.json", "whisper-tiny.en.config.json", "whisper-base.config.json", "whisper-large-v3-turbo.config.json"] {
        let text = std::fs::read_to_string(dir().join(name)).unwrap();
        let err = specrt::spec::Spec::from_hf_config(&text).unwrap_err();
        assert!(format!("{err:#}").contains("Whisper is a speech to text model"), "{name}: {err:#}");
        assert!(specrt::spec::Spec::from_hf_config_duplex(&text).is_err(), "{name}");
        let speech = specrt::spec::Spec::from_hf_config_speech(&text).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        assert!(speech.cross.is_some(), "{name}");
    }
    let mut missing = Vec::new();
    for n in WHISPER.iter().chain(TTS) {
        match n.evidence {
            Some((file, needle)) => assert!(engine_source(file).contains(needle), "{}: {file} no longer has {needle:?}; update the gap list", n.piece),
            None => missing.push(n.piece),
        }
    }
    print!("{}", render("speech to text (Whisper) against specrt", WHISPER));
    print!("{}", render("text to speech (Phonon) against specrt", TTS));
    println!("missing ops with no engine piece yet: {}", missing.len());
    for m in &missing {
        println!("  {m}");
    }
    assert_eq!(missing.len(), 2, "{missing:#?}");
    for slice in ["a", "b", "c", "e", "f"] {
        assert!(WHISPER.iter().chain(TTS).any(|n| n.slice == slice), "slice {slice} has no gap");
    }
}

#[test]
fn whisper_configs_have_the_shape_the_gap_list_assumes() {
    for name in ["tiny-whisper/config.json", "whisper-tiny.en.config.json", "whisper-base.config.json", "whisper-large-v3-turbo.config.json"] {
        let c = json(&dir().join(name));
        assert_eq!(c["model_type"], "whisper", "{name}");
        assert_eq!(c["architectures"][0], "WhisperForConditionalGeneration", "{name}");
        assert_eq!(c["activation_function"], "gelu", "{name}");
        assert_eq!(c["scale_embedding"], false, "{name}");
        assert_eq!(c["encoder_attention_heads"], c["decoder_attention_heads"], "{name}");
        assert_eq!(c["d_model"].as_u64().unwrap() % c["encoder_attention_heads"].as_u64().unwrap(), 0, "{name}");
        assert!([80, 128].contains(&c["num_mel_bins"].as_u64().unwrap()), "{name}");
    }
    let turbo = json(&dir().join("whisper-large-v3-turbo.config.json"));
    assert_eq!((turbo["encoder_layers"].as_u64(), turbo["decoder_layers"].as_u64(), turbo["num_mel_bins"].as_u64()), (Some(32), Some(4), Some(128)));
    let en = json(&dir().join("whisper-tiny.en.generation_config.json"));
    let multi = json(&dir().join("whisper-base.generation_config.json"));
    assert_eq!(en["is_multilingual"], false);
    assert_eq!(multi["is_multilingual"], true);
    assert_eq!(multi["lang_to_id"].as_object().unwrap().len(), 99);
    assert_eq!(multi["task_to_id"]["transcribe"], 50359);
    for g in [&en, &multi] {
        assert!(g["no_timestamps_token_id"].is_u64() && g["begin_suppress_tokens"].is_array() && g["suppress_tokens"].is_array());
    }
}

#[test]
fn the_tiny_fixture_keeps_every_real_tensor_kind() {
    let real = header("whisper-tiny.en.header.json");
    let bytes = std::fs::read(tiny().join("model.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let names: Vec<String> = st.names().into_iter().map(str::to_string).collect();
    assert_eq!(kinds(names.iter()), kinds(real.keys()), "tensor kinds differ from openai/whisper-tiny.en");
    assert!(!real.contains_key("proj_out.weight"), "the head is tied to decoder.embed_tokens");
    assert!(real.keys().all(|k| !k.ends_with("k_proj.bias")), "k_proj has no bias");
    let c = json(&tiny().join("config.json"));
    let d = c["d_model"].as_u64().unwrap() as usize;
    let shape = |n: &str| st.tensor(n).unwrap().shape().to_vec();
    assert_eq!(shape("model.encoder.conv1.weight"), vec![d, 80, 3]);
    assert_eq!(shape("model.encoder.conv2.weight"), vec![d, d, 3]);
    assert_eq!(shape("model.encoder.embed_positions.weight"), vec![c["max_source_positions"].as_u64().unwrap() as usize, d]);
    assert_eq!(shape("model.decoder.embed_tokens.weight"), vec![c["vocab_size"].as_u64().unwrap() as usize, d]);
    assert_eq!(real["model.encoder.embed_positions.weight"].1, vec![1500, 384]);
}

fn hann(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()).collect()
}

fn log_mel(pcm: &[f32], filters: &[f32], bins: usize, frames: usize) -> Vec<f32> {
    let (n_fft, hop) = (400usize, 160usize);
    let freqs = n_fft / 2 + 1;
    let half = n_fft / 2;
    let mut padded = vec![0f64; frames * hop + n_fft];
    let len = pcm.len().min(frames * hop);
    let src: Vec<f64> = (0..frames * hop).map(|i| if i < len { pcm[i] as f64 } else { 0.0 }).collect();
    for (i, p) in padded.iter_mut().enumerate() {
        let j = i as isize - half as isize;
        let k = if j < 0 { -j } else if j as usize >= src.len() { 2 * (src.len() as isize - 1) - j } else { j };
        *p = src[k as usize];
    }
    let w = hann(n_fft);
    let mut mel = vec![0f64; bins * frames];
    for f in 0..frames {
        let seg = &padded[f * hop..f * hop + n_fft];
        let mut power = vec![0f64; freqs];
        for (k, pw) in power.iter_mut().enumerate() {
            let (mut re, mut im) = (0f64, 0f64);
            for (t, s) in seg.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * (k * t) as f64 / n_fft as f64;
                re += s * w[t] * a.cos();
                im += s * w[t] * a.sin();
            }
            *pw = re * re + im * im;
        }
        for m in 0..bins {
            let v: f64 = (0..freqs).map(|k| filters[k * bins + m] as f64 * power[k]).sum();
            mel[m * frames + f] = v.max(1e-10).log10();
        }
    }
    let top = mel.iter().cloned().fold(f64::MIN, f64::max);
    mel.iter().map(|v| ((v.max(top - 8.0) + 4.0) / 4.0) as f32).collect()
}

#[test]
fn the_log_mel_front_end_is_specified_exactly_by_the_reference() {
    let meta = json(&tiny().join("reference.json"));
    let window = meta["window_samples"].as_u64().unwrap() as usize;
    let (filters, fshape) = reference("mel.filters");
    assert_eq!(fshape, vec![201, 80]);
    for (audio, mel) in [("audio.full", "mel.full"), ("audio.short", "mel.short")] {
        let (pcm, _) = reference(audio);
        let (want, shape) = reference(mel);
        assert_eq!(shape, vec![80, window / 160], "{mel}: frames are the padded window over the hop");
        let got = log_mel(&pcm, &filters, 80, shape[1]);
        let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("{mel}: max abs error {err:.2e} against transformers");
        assert!(err < 2e-4, "{mel}: {err}");
    }
    let (enc, eshape) = reference("encoder.out");
    let c = json(&tiny().join("config.json"));
    assert_eq!(eshape, vec![c["max_source_positions"].as_u64().unwrap() as usize, c["d_model"].as_u64().unwrap() as usize], "the stem halves the mel frames");
    assert!(enc.iter().all(|v| v.is_finite()));
}

#[test]
fn the_token_logic_reproduces_transformers_generate_from_its_logits() {
    let meta = json(&tiny().join("reference.json"));
    let generation = json(&tiny().join("generation_config.json"));
    let suppress: BTreeSet<u32> = ids(&generation["suppress_tokens"]).into_iter().collect();
    let begin: BTreeSet<u32> = ids(&generation["begin_suppress_tokens"]).into_iter().collect();
    let eot = generation["eos_token_id"].as_u64().unwrap() as u32;
    let (rows, shape) = reference("greedy.logits");
    let vocab = shape[1];
    let mut tokens = Vec::new();
    for (step, row) in rows.chunks(vocab).enumerate() {
        let next = row
            .iter()
            .enumerate()
            .filter(|(t, _)| !suppress.contains(&(*t as u32)) && !(step == 0 && begin.contains(&(*t as u32))))
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(t, _)| t as u32)
            .unwrap();
        tokens.push(next);
        if next == eot {
            break;
        }
    }
    assert_eq!(tokens, ids(&meta["greedy"]), "greedy with suppress lists");
    let prefix = ids(&meta["prefix"]);
    let generated = ids(&meta["generate"]);
    let body: Vec<u32> = if generated.starts_with(&prefix) { generated[prefix.len()..].to_vec() } else { generated };
    assert_eq!(tokens, body, "transformers generate with language en and no timestamps");
    let (lang, _) = reference("language.logits");
    let langs: Vec<u32> = generation["lang_to_id"].as_object().unwrap().values().map(|v| v.as_u64().unwrap() as u32).collect();
    let detected = *langs.iter().max_by(|a, b| lang[**a as usize].total_cmp(&lang[**b as usize])).unwrap();
    assert_eq!(detected as u64, meta["detected_language"].as_u64().unwrap(), "language detection is the argmax over language tokens at step 0");
}

#[test]
fn phonon_tensors_match_the_gap_list() {
    let h = header("phonon-31m-en.header.json");
    let c = json(&dir().join("phonon-31m-en.config.json"));
    assert!(c.get("model_type").is_none() && c.get("architectures").is_none(), "the revision agentbench pins names no model type");
    let hub = json(&dir().join("phonon-31m-en.hub-92741c5.config.json"));
    assert_eq!(hub["model_type"], "gradium_phonon_tts");
    assert_eq!(hub["architectures"][0], "GradiumPhononTTSForConditionalGeneration");
    assert_eq!(hub["flow_lm"], c["flow_lm"]);
    assert_eq!(hub["mimi"], c["mimi"]);
    assert_eq!((hub["hidden_act"].as_str(), hub["num_hidden_layers"].as_u64(), hub["head_dim"].as_u64()), (Some("gelu"), Some(6), Some(64)));
    assert_eq!(hub["flow_head_config"]["flow_type"], "lsd");
    let total: usize = h.values().map(|(_, s)| s.iter().product::<usize>()).sum();
    assert_eq!(total, 41_096_035);
    let part = |p: &str| -> usize { h.iter().filter(|(k, _)| k.starts_with(p)).map(|(_, (_, s))| s.iter().product::<usize>()).sum() };
    assert_eq!(part("flow_lm.transformer."), 18_886_656);
    assert_eq!(part("flow_lm.flow_net."), 9_496_864);
    assert_eq!(part("mimi.decoder."), 3_974_945);
    assert_eq!(part("mimi.decoder_transformer."), 6_297_600);
    assert_eq!(h["mimi.upsample.convtr.convtr.weight"].1, vec![512, 1, 32], "depthwise transposed conv, stride 16, no bias");
    assert!(!h.contains_key("mimi.upsample.convtr.convtr.bias"));
    assert!(h.keys().all(|k| !k.contains("quantizer.vq")), "no RVQ codebooks: the decoder reads continuous latents");
    assert_eq!(h["mimi.quantizer.output_proj.weight"].1, vec![512, 32, 1]);
    assert_eq!(h["flow_lm.transformer.layers.0.self_attn.in_proj.weight"].1, vec![1536, 512], "packed qkv");
    assert!(h.contains_key("flow_lm.transformer.layers.0.norm1.bias"), "LayerNorm, not RMS");
    assert!(h.contains_key("mimi.decoder_transformer.transformer.layers.0.layer_scale_1.scale"));
    assert_eq!(c["mimi"]["ratios"], serde_json::json!([6, 5, 4]));
    assert_eq!(c["mimi"]["transformer_context"], 250);
    assert_eq!(c["lsd_decode_steps"], 1);
    let voice = header("phonon-voice.header.json");
    assert_eq!(voice["speaker_wavs"].1, vec![1, 512, 125], "a voice is 10 s of Mimi encoder output, before speaker_proj");
}

#[test]
fn the_vibevoice_acoustic_decoder_is_the_haru_codec_decoder_tensor_for_tensor() {
    let haru: Vec<(String, String, Vec<usize>)> = serde_json::from_str(&std::fs::read_to_string(root().join("tests/fixtures/haru/haru-4b.header.json")).unwrap()).unwrap();
    let haru: BTreeMap<String, Vec<usize>> = haru.into_iter().filter(|r| r.0.starts_with("codec.decoder.")).map(|r| (r.0, r.2)).collect();
    let vv = header("vibevoice-realtime-0.5b.header.json");
    let rename = |k: &str| -> Option<String> {
        let k = k.strip_prefix("model.acoustic_tokenizer.decoder.")?;
        let k = k.replace("mixer.conv.conv.conv.", "conv.").replace("ffn.linear1.", "fc1.").replace("ffn.linear2.", "fc2.").replace("head.conv.conv.", "head.");
        if let Some(rest) = k.strip_prefix("upsample_layers.0.0.conv.conv.") {
            return Some(format!("codec.decoder.stem.{rest}"));
        }
        if let Some(rest) = k.strip_prefix("upsample_layers.") {
            let (i, tail) = rest.split_once(".0.convtr.convtr.").unwrap();
            return Some(format!("codec.decoder.up.{}.{tail}", i.parse::<usize>().unwrap() - 1));
        }
        Some(format!("codec.decoder.{k}"))
    };
    let mapped: BTreeMap<String, Vec<usize>> = vv.iter().filter_map(|(k, (_, s))| Some((rename(k)?, s.clone()))).collect();
    assert_eq!(mapped.len(), 276);
    assert_eq!(mapped, haru, "VibeVoice-Realtime's acoustic decoder renamed is Haru's codec decoder");
    let params: usize = mapped.values().map(|s| s.iter().product::<usize>()).sum();
    assert_eq!(params, 343_695_969);
    let c = json(&dir().join("vibevoice-realtime-0.5b.config.json"));
    let t = &c["acoustic_tokenizer_config"];
    assert_eq!(t["decoder_ratios"], serde_json::json!([8, 5, 5, 4, 2, 2]));
    assert_eq!(t["causal"], true);
    assert_eq!(c["decoder_config"]["model_type"], "qwen2");
    assert_eq!(c["diffusion_head_config"]["diffusion_type"], "ddpm");
    let total: usize = vv.values().map(|(_, s)| s.iter().product::<usize>()).sum();
    assert_eq!(total, 1_017_626_724, "every tensor of the release: there is no acoustic encoder, so voices cannot be made from audio");
}

#[test]
fn the_speech_fixture_stays_under_2_mb() {
    fn size(p: &Path) -> u64 {
        std::fs::read_dir(p).unwrap().flatten().map(|e| if e.path().is_dir() { size(&e.path()) } else { e.metadata().unwrap().len() }).sum()
    }
    let total = size(&dir());
    println!("tests/fixtures/speech: {total} bytes");
    assert!(total < 2 * 1024 * 1024, "{total}");
}

fn mimi() -> PathBuf {
    dir().join("tiny-mimi")
}

fn mimi_ref(key: &str) -> (Vec<f32>, Vec<usize>) {
    let meta = json(&mimi().join("reference.json"));
    let entry = &meta["files"][key];
    let shape: Vec<usize> = entry["shape"].as_array().unwrap_or_else(|| panic!("{key}")).iter().map(|d| d.as_u64().unwrap() as usize).collect();
    let bytes = std::fs::read(mimi().join("reference").join(entry["file"].as_str().unwrap())).unwrap();
    let values: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    assert_eq!(values.len(), shape.iter().product::<usize>(), "{key}");
    (values, shape)
}

fn mimi_decoder_with(config: Value) -> specrt::codec::HostCodec {
    let plan = specrt::codec::graph::plan(&config).unwrap();
    let ck = specrt::weights::Checkpoint::open(&mimi()).unwrap();
    plan.check_tensors(&|n| ck.shape_of(n)).unwrap();
    specrt::codec::HostCodec::load(plan, &ck).unwrap()
}

fn mimi_host() -> specrt::codec::HostCodec {
    specrt::codec::HostCodec::open(&mimi()).unwrap()
}

#[test]
fn the_mimi_decoder_graph_plans_every_codec_tensor_of_the_real_phonon_checkpoint() {
    let hub = json(&dir().join("phonon-31m-en.hub-92741c5.config.json"));
    let plan = specrt::codec::graph::plan(&hub).unwrap();
    assert!(!plan.has_encoder(), "the TTS codec is a decoder only");
    let h = header("phonon-31m-en.header.json");
    let codec: BTreeMap<String, Vec<usize>> = h.iter().filter(|(k, _)| k.starts_with("mimi.")).map(|(k, (_, s))| (k.clone(), s.clone())).collect();
    let planned: BTreeMap<String, Vec<usize>> = plan.decoder.tensors().into_iter().collect();
    assert_eq!(planned, codec, "the graph reads every mimi tensor of the checkpoint and nothing else");
    assert_eq!(plan.decoder.params(), 3_974_945 + 6_297_600 + 32_768);
    assert_eq!((plan.frame_samples, plan.latent_dim), (1920, 32));
    let steps = plan.decoder.steps(1);
    assert_eq!(steps.last(), Some(&1920));
    assert!(steps.contains(&16), "the transformer runs at 200 Hz, 16 steps per 12.5 Hz frame");
    let ring: usize = plan.decoder.ops.iter().filter(|o| matches!(o, specrt::codec::CodecOp::Attention { .. })).map(|o| o.state_floats()).sum();
    assert_eq!(ring, 2 * (2 * 250 * 512 + 1), "two layers, K and V rings of transformer_context = 250 steps and a position");
    let p = specrt::codec::compile::plan(&plan.decoder, 1, 4).unwrap();
    let report = specrt::codec::compile::report(&plan.decoder, &p, 4);
    println!("{report}");
    assert!(report.contains("ring attention") && report.contains("dwconv_up mimi.upsample.convtr.convtr"), "{report}");
    assert!(p.kernels(&kir::Target::sm_120()).unwrap().iter().any(|k| matches!(k.prologue, kir::conv::Prologue::Act("elu"))), "ELU is a conv input prologue, not a launch");
    assert!(!p.launches.iter().any(|l| l.label(&plan.decoder).starts_with("add")), "the residual add folds into the k1 conv");
}

#[test]
fn the_host_mimi_decoder_matches_transformers_whole_and_frame_by_frame() {
    let codec = mimi_host();
    let (latents, shape) = mimi_ref("latents");
    let (want, _) = mimi_ref("audio");
    let frames = shape[0];
    assert_eq!(codec.plan.frame_samples * frames, want.len());
    let whole = codec.decode(&latents).unwrap();
    let e = max_rel(&whole, &want);
    println!("host mimi decode: max rel {e:.2e} against transformers over {} samples", want.len());
    assert!(e < 1e-4, "{e}");
    let mut state = codec.decoder.new_state();
    assert_eq!(state.floats(), codec.plan.decoder.state_floats());
    let streamed: Vec<f32> = latents.chunks(codec.plan.latent_dim).flat_map(|z| codec.decoder.run(z, &mut state).unwrap()).collect();
    let f = max_rel(&streamed, &whole);
    println!("host mimi decode: frame by frame against whole clip max rel {f:.2e}");
    assert!(f < 1e-5, "{f}");
    let mut state = codec.decoder.new_state();
    let pairs: Vec<f32> = latents.chunks(2 * codec.plan.latent_dim).flat_map(|z| codec.decoder.run(z, &mut state).unwrap()).collect();
    assert!(max_rel(&pairs, &whole) < 1e-5);
}

#[test]
fn a_ring_shorter_than_the_reference_window_changes_the_audio() {
    let mut config = json(&mimi().join("config.json"));
    let context = config["mimi"]["transformer_context"].as_u64().unwrap();
    assert!(context < 6 * 16, "the fixture's 96 steps at 200 Hz pass the ring of {context}, so the window is exercised");
    config["mimi"]["transformer_context"] = (context - 4).into();
    let short = mimi_decoder_with(config);
    let (latents, _) = mimi_ref("latents");
    let (want, _) = mimi_ref("audio");
    let e = max_rel(&short.decode(&latents).unwrap(), &want);
    println!("ring of {} instead of {context}: max rel {e:.2e}", context - 4);
    assert!(e > 1e-3, "a wrong window must not pass the 1e-4 bar: {e}");
    let mut state = short.decoder.new_state();
    let streamed: Vec<f32> = latents.chunks(short.plan.latent_dim).flat_map(|z| short.decoder.run(z, &mut state).unwrap()).collect();
    assert!(max_rel(&streamed, &short.decode(&latents).unwrap()) < 1e-5, "the ring streams like the whole clip at any window");
}

#[test]
fn the_elu_on_the_conv_inputs_is_what_makes_the_audio() {
    let codec = mimi_host();
    let (latents, _) = mimi_ref("latents");
    let (want, _) = mimi_ref("audio");
    let mut plain = specrt::codec::HostCodec { plan: codec.plan.clone(), encoder: codec.encoder, decoder: codec.decoder };
    let mut dropped = 0;
    for p in &mut plain.decoder.params {
        if let specrt::codec::host::Param::Conv(c) = p
            && c.act.is_some()
        {
            c.act = None;
            dropped += 1;
        }
    }
    assert_eq!(dropped, 3 * 3 + 1, "ELU before each up conv, both residual convs of each stage, and the final conv");
    let e = max_rel(&plain.decode(&latents).unwrap(), &want);
    println!("without ELU: max rel {e:.2e}");
    assert!(e > 1e-2, "{e}");
    let refuse = kir::conv::ConvKernel::new(&kir::Target::sm_120(), kir::conv::ConvShape::line(8, 8, 3, 1, 1), kir::conv::Prologue::Act("sigmoid"), kir::conv::Epilogue { residual: false, stats: None, gelu: false });
    assert!(format!("{}", refuse.unwrap_err()).contains("maps 0 to 0"), "an input activation that moves zero would break the zero pad and tail");
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12);
    max_abs(a, b) / scale
}

fn frames_major(mel: &[f32], bins: usize, frames: usize) -> Vec<f32> {
    (0..frames * bins).map(|i| mel[(i % bins) * frames + i / bins]).collect()
}

fn speech_plan() -> specrt::speech::SpeechPlan {
    specrt::speech::graph::plan_dir(&tiny()).unwrap()
}

#[test]
fn the_speech_plan_reads_the_front_end_from_preprocessor_config() {
    let p = speech_plan();
    let m = &p.mel;
    assert_eq!((m.samples, m.n_fft, m.hop, m.frames, m.bins, m.rate), (16000, 400, 160, 100, 80, 16000.0));
    assert_eq!(m.window, kir::mel::Window::Hann { periodic: true });
    assert_eq!((p.frames, p.width, p.positions), (50, 64, 50));
    let mut pre = json(&tiny().join("preprocessor_config.json"));
    pre["n_fft"] = 512.into();
    pre["hop_length"] = 200.into();
    let edited = specrt::speech::graph::log_mel(&pre).unwrap();
    assert_eq!((edited.n_fft, edited.hop, edited.frames, edited.freqs()), (512, 200, 80, 257));
    pre["feature_size"] = 128.into();
    let err = specrt::speech::graph::plan(&json(&tiny().join("config.json")), &pre).unwrap_err();
    assert!(format!("{err:#}").contains("reads 80 mel bins, the preprocessor makes 128"), "{err:#}");
    let real = specrt::speech::graph::plan(&json(&dir().join("whisper-large-v3-turbo.config.json")), &serde_json::json!({"feature_extractor_type": "WhisperFeatureExtractor", "feature_size": 128, "hop_length": 160, "n_fft": 400, "n_samples": 480000, "sampling_rate": 16000})).unwrap();
    assert_eq!((real.mel.frames, real.frames, real.width), (3000, 1500, 1280));
}

#[test]
fn the_mel_bank_equals_transformers_filters() {
    let p = speech_plan();
    let (want, shape) = reference("mel.filters");
    let bank = p.mel.mel_bank();
    let (f, b) = (shape[0], shape[1]);
    let got: Vec<f32> = (0..f * b).map(|i| bank[(i % b) * f + i / b]).collect();
    let err = max_abs(&got, &want);
    println!("mel.filters: max abs error {err:.2e}");
    assert!(err < 1e-7, "{err}");
}

fn encoder_states() -> Vec<f32> {
    let host = specrt::speech::HostSpeech::open(&tiny()).unwrap();
    let enc = specrt::speech::host::HostEncoder::new(&json(&tiny().join("config.json"))).unwrap();
    let ck = specrt::weights::Checkpoint::open(&tiny()).unwrap();
    let (pcm, _) = reference("audio.full");
    let (_, stem) = host.run(&pcm).unwrap();
    enc.encode(&ck, &stem).unwrap()
}

fn greedy_tokens(dec: &specrt::speech::decoder::HostDecoder, states: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
    let meta = json(&tiny().join("reference.json"));
    let generation = json(&tiny().join("generation_config.json"));
    let suppress: BTreeSet<u32> = ids(&generation["suppress_tokens"]).into_iter().collect();
    let begin: BTreeSet<u32> = ids(&generation["begin_suppress_tokens"]).into_iter().collect();
    let vocab = dec.spec.dims.vocab;
    let prefix = ids(&meta["prefix"]);
    let mut seq = dec.new_sequence();
    dec.window(&mut seq, states).unwrap();
    let prompt = dec.prefill(&mut seq, &prefix).unwrap();
    let mut row = prompt[(prefix.len() - 1) * vocab..].to_vec();
    let (mut rows, mut tokens) = (Vec::new(), Vec::new());
    for step in 0..12 {
        rows.extend_from_slice(&row);
        let next = row.iter().enumerate().filter(|(t, _)| !suppress.contains(&(*t as u32)) && !(step == 0 && begin.contains(&(*t as u32)))).max_by(|a, b| a.1.total_cmp(b.1)).map(|(t, _)| t as u32).unwrap();
        tokens.push(next);
        row = dec.step(&mut [&mut seq], &[next]).unwrap();
    }
    (prompt, rows, tokens)
}

#[test]
fn the_whisper_decoder_runs_from_the_family_spec_and_matches_transformers() {
    let states = encoder_states();
    let meta = json(&tiny().join("reference.json"));
    for (bf16, bar) in [(false, 1e-3), (true, 2e-2)] {
        let dec = specrt::speech::decoder::HostDecoder::open(&tiny(), bf16).unwrap();
        let (prompt, rows, tokens) = greedy_tokens(&dec, &states);
        let (want_prefix, _) = reference("prefix.logits");
        let (want_greedy, _) = reference("greedy.logits");
        let (pe, ge) = (max_rel(&prompt, &want_prefix), max_rel(&rows, &want_greedy));
        println!("host decoder bf16={bf16}: prefix.logits max rel {pe:.2e}, greedy.logits max rel {ge:.2e}");
        assert!(pe < bar && ge < bar, "bf16={bf16}: {pe} {ge}");
        assert_eq!(tokens, ids(&meta["greedy"]), "bf16={bf16}: the 12 greedy tokens");
    }
}

#[test]
fn two_whisper_sequences_batched_equal_each_alone() {
    let states = encoder_states();
    let dec = specrt::speech::decoder::HostDecoder::open(&tiny(), false).unwrap();
    let quiet = vec![0.25f32; states.len()];
    let prefix = ids(&json(&tiny().join("reference.json"))["prefix"]);
    let alone = |s: &[f32], prompt: &[u32], next: &[u32]| {
        let mut q = dec.new_sequence();
        dec.window(&mut q, s).unwrap();
        dec.prefill(&mut q, prompt).unwrap();
        next.iter().map(|t| dec.step(&mut [&mut q], &[*t]).unwrap()).collect::<Vec<_>>()
    };
    let (a_next, b_next) = ([501u32, 257, 121], [300u32, 12, 7]);
    let want_a = alone(&states, &prefix, &a_next);
    let want_b = alone(&quiet, &prefix[..2], &b_next);
    let (mut a, mut b) = (dec.new_sequence(), dec.new_sequence());
    dec.window(&mut a, &states).unwrap();
    dec.window(&mut b, &quiet).unwrap();
    dec.prefill(&mut a, &prefix).unwrap();
    dec.prefill(&mut b, &prefix[..2]).unwrap();
    let vocab = dec.spec.dims.vocab;
    for k in 0..3 {
        let both = dec.step(&mut [&mut a, &mut b], &[a_next[k], b_next[k]]).unwrap();
        assert_eq!(&both[..vocab], &want_a[k][..], "step {k}: sequence a batched differs from alone");
        assert_eq!(&both[vocab..], &want_b[k][..], "step {k}: sequence b batched differs from alone");
    }
    assert_ne!(want_a[0], want_b[0], "the two windows give different logits, so the batch check sees a mixup");
}

#[test]
fn the_cross_kv_state_is_declared_in_the_spec_and_needs_its_window() {
    let text = std::fs::read_to_string(tiny().join("config.json")).unwrap();
    let spec = specrt::spec::Spec::from_hf_config_speech(&text).unwrap();
    let cross = spec.cross.as_ref().unwrap();
    assert_eq!((cross.rows, cross.state.as_str()), (50, "cross_kv"));
    let state = specrt::state::named(&spec, "cross_kv").unwrap();
    assert_eq!(state.per, specrt::state::Per::Sequence);
    assert_eq!(state.sequence_bytes(&spec), 2 * 2 * 50 * 64 * 4, "2 layers x k and v x 50 encoder rows x d 64, f32");
    assert_eq!((spec.norm.kind, spec.attention.qkv_bias, spec.attention.k_bias, spec.attention.o_bias, spec.mlp.bias, spec.dims.tie_embeddings), (specrt::spec::NormKind::LayerNorm, true, false, true, true, true));
    assert_eq!(spec.rope.nope_layers, vec![0, 1], "no rope anywhere");
    let prefill = specrt::plan::Plan::build(&spec, specrt::plan::Phase::Prefill, true);
    let decode = specrt::plan::Plan::build(&spec, specrt::plan::Phase::Decode, true);
    assert_eq!(prefill.window, vec![specrt::plan::Op::CrossKv], "cross K and V are computed once per window, not per token");
    let block = spec.block(0);
    assert!(prefill.layer(block).contains(&specrt::plan::Op::CrossAttention { decode: false }));
    assert!(decode.layer(block).contains(&specrt::plan::Op::CrossAttention { decode: true }));
    assert!(!decode.layer(block).contains(&specrt::plan::Op::CrossKv), "no decode step recomputes cross K and V");
    let dec = specrt::speech::decoder::HostDecoder::open(&tiny(), false).unwrap();
    let mut seq = dec.new_sequence();
    let err = dec.prefill(&mut seq, &[513]).unwrap_err();
    assert!(format!("{err:#}").contains("has no cross_kv state; run the window first"), "{err:#}");
}

#[test]
fn the_host_encoder_graph_matches_transformers() {
    let host = specrt::speech::HostSpeech::open(&tiny()).unwrap();
    let config = json(&tiny().join("config.json"));
    let enc = specrt::speech::host::HostEncoder::new(&config).unwrap();
    let ck = specrt::weights::Checkpoint::open(&tiny()).unwrap();
    let (mel, _) = reference("mel.full");
    let stem = host.stem(&frames_major(&mel, host.plan.mel.bins, host.plan.mel.frames)).unwrap();
    let states = enc.encode(&ck, &stem).unwrap();
    let (want, _) = reference("encoder.out");
    let err = max_rel(&states, &want);
    println!("host encoder.out: max rel {err:.2e}");
    assert!(err < 1e-4, "{err}");
}

#[test]
fn the_host_log_mel_and_stem_match_transformers() {
    let host = specrt::speech::HostSpeech::open(&tiny()).unwrap();
    let (bins, frames) = (host.plan.mel.bins, host.plan.mel.frames);
    for (audio, mel) in [("audio.full", "mel.full"), ("audio.short", "mel.short")] {
        let (pcm, _) = reference(audio);
        let (want, _) = reference(mel);
        let got = specrt::speech::host::log_mel(&host.plan, &pcm).unwrap();
        let err = max_abs(&got, &frames_major(&want, bins, frames));
        println!("host {mel}: max abs {err:.2e}");
        assert!(err < 1e-4, "{mel}: {err}");
    }
    let (mel, _) = reference("mel.full");
    let (want, _) = reference("encoder.stem");
    let stem = host.stem(&frames_major(&mel, bins, frames)).unwrap();
    let err = max_rel(&stem, &want);
    println!("host encoder.stem: max rel {err:.2e}");
    assert!(err < 1e-4, "{err}");
}

#[test]
fn the_speech_ops_are_typed_and_the_gelu_fuses_into_the_conv() {
    let p = speech_plan();
    specrt::signature::validate_speech(&p).unwrap();
    let t = kir::Target::sm_120();
    let fused = specrt::speech::compile::fuse(&p, &t, true).unwrap();
    let steps = specrt::speech::compile::steps(&p, &fused).unwrap();
    print!("{}", specrt::speech::compile::report(&p, &fused, &steps));
    assert_eq!((fused.launches_before, steps.len()), (11, 9));
    let gelus: Vec<bool> = steps.iter().filter_map(|s| match s { specrt::speech::compile::Step::Gemm { gelu, .. } => Some(*gelu), _ => None }).collect();
    assert_eq!(gelus, [false, false, true, true], "dft, bank, then both stem convs with GELU in the epilogue");
    let ks = specrt::speech::compile::kernels(&steps, &t).unwrap();
    assert!(ks.iter().filter(|k| k.shape.pad == Some(1)).count() == 2);
    let unfused = specrt::speech::compile::fuse(&p, &t, false).unwrap();
    let s = specrt::speech::compile::steps(&p, &unfused).unwrap();
    assert!(specrt::speech::compile::kernels(&s, &t).is_err(), "an unfused GELU names its missing kernel");
    let mut bad = p.clone();
    if let specrt::speech::SpeechOp::Conv { t_in, .. } = &mut bad.ops[9] {
        *t_in += 1;
    }
    let err = specrt::signature::validate_speech(&bad).unwrap_err();
    assert!(format!("{err:#}").contains("reads stem"), "{err:#}");
}

#[test]
fn the_padded_line_conv_is_its_own_kernel_and_refuses_a_wide_pad() {
    let t = kir::Target::sm_120();
    let causal = kir::conv::ConvShape::line(64, 64, 3, 2, 1);
    let padded = kir::conv::ConvShape::padded(64, 64, 3, 2, 1);
    assert_eq!((causal.out_len(100), padded.out_len(100), padded.context(), causal.context()), (50, 50, 0, 1));
    let e = kir::conv::Epilogue { residual: false, stats: None, gelu: true };
    let a = kir::conv::ConvKernel::new(&t, causal, kir::conv::Prologue::None, e).unwrap();
    let b = kir::conv::ConvKernel::new(&t, padded, kir::conv::Prologue::None, e).unwrap();
    assert_ne!(a.name(), b.name());
    let src = kir::cuda::emit_line(&[a, b]).unwrap();
    assert!(src.contains("true, -1>(x, st") && src.contains("true, 1>(x, st"), "causal convs keep PAD -1, the stem gets PAD 1");
    assert!(kir::conv::ConvKernel::new(&t, kir::conv::ConvShape::padded(64, 64, 3, 1, 3), kir::conv::Prologue::None, e).is_err());
}

#[test]
fn the_log_mel_emitter_bakes_the_spec_expressions() {
    let p = speech_plan();
    let src = kir::cuda::emit_mel(&p.mel).unwrap();
    for name in p.mel.names() {
        assert!(src.contains(&format!("void __launch_bounds__(256) {name}(")), "{name}");
    }
    assert!(src.contains("fmaxf(m, 1e-10f)") || src.contains("fmaxf(m, 0.0000000001f)"), "{}", src.as_str());
    assert!(src.contains("constexpr int N_FFT = 400;") && src.contains("constexpr int HOP = 160;"));
    let mut m = p.mel.clone();
    m.log = kir::expr::Expr::parse("log(max(q, 1e-10))").unwrap();
    let err = kir::cuda::emit_mel(&m).unwrap_err();
    assert!(err.contains("names q"), "{err}");
}

#[test]
#[ignore = "needs nvrtc, no GPU: compiles the tiny and real speech modules for sm_120"]
fn every_speech_kernel_compiles_with_nvrtc() {
    let host = specrt::speech::HostSpeech::open(&tiny()).unwrap();
    let (m, _, _) = specrt::speech::cuda::module(&host, &kir::Target::sm_120()).unwrap();
    let opts = cudarc::nvrtc::CompileOptions { arch: Some("compute_120"), options: vec!["--std=c++17".into()], ..Default::default() };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(m.source.as_str(), opts).unwrap().to_src();
    for name in &m.kernels {
        assert!(ptx.contains(&format!(".entry {name}(")), "{name} missing from the PTX");
    }
    let vk = kir::vision::VisionKernel::new(&kir::Target::sm_120(), 32, true).unwrap().joint();
    let src = kir::cuda::emit_vision(&vk).unwrap();
    let opts = cudarc::nvrtc::CompileOptions { arch: Some("compute_120"), options: vec!["--std=c++17".into()], ..Default::default() };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(src.as_str(), opts).unwrap().to_src();
    for name in kir::vision::JOINT_KERNELS {
        assert!(ptx.contains(&format!(".entry {name}(")), "{name} missing from the PTX");
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_gpu_cross_attention_rows_match_the_host_on_two_sequences() {
    let states = encoder_states();
    let dec = specrt::speech::decoder::HostDecoder::open(&tiny(), false).unwrap();
    let quiet: Vec<f32> = states.iter().enumerate().map(|(i, v)| v * 0.5 + (i % 7) as f32 * 0.01).collect();
    let (mut a, mut b) = (dec.new_sequence(), dec.new_sequence());
    dec.window(&mut a, &states).unwrap();
    dec.window(&mut b, &quiet).unwrap();
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let cache = std::env::var_os("SPECRT_CACHE").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("specrt-speech-jit"));
    let jit = specrt::jit::Jit::new(ctx.clone(), cache).unwrap();
    let gpu = specrt::speech::cuda::CudaCrossRows::load(&ctx.default_stream(), &jit, &dec.spec).unwrap();
    let qd = dec.spec.q_dim();
    let q: Vec<f32> = (0..2 * qd).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.03).collect();
    for layer in 0..dec.spec.layers.count {
        let got = gpu.run(&q, &[&a.cross_kv[layer], &b.cross_kv[layer]]).unwrap();
        let mut want = dec.attend_cross(&a, layer, &q[..qd]).unwrap();
        want.extend(dec.attend_cross(&b, layer, &q[qd..]).unwrap());
        let err = max_rel(&got, &want);
        println!("gpu cross attention rows layer {layer}: max rel {err:.2e}");
        assert!(err < 1e-5, "layer {layer}: {err}");
        let alone = gpu.run(&q[qd..], &[&b.cross_kv[layer]]).unwrap();
        assert_eq!(&got[qd..], &alone[..], "layer {layer}: sequence b batched differs from alone");
    }
}

fn rules() -> specrt::speech::decode::Rules {
    let spec = specrt::spec::Spec::from_hf_config_speech(&std::fs::read_to_string(tiny().join("config.json")).unwrap()).unwrap();
    specrt::speech::decode::Rules::open(&tiny(), spec.speech_decode.as_ref().unwrap()).unwrap()
}

fn run_options(run: &Value) -> specrt::speech::decode::Options {
    let o = &run["options"];
    let f = |k: &str| o[k].as_f64();
    specrt::speech::decode::Options {
        language: o["language"].as_str().map(str::to_string),
        task: o["language"].is_string().then(|| "transcribe".into()),
        timestamps: true,
        temperatures: o["temperatures"].as_array().map_or(vec![0.0], |a| a.iter().map(|t| t.as_f64().unwrap()).collect()),
        compression_ratio_threshold: f("compression_ratio_threshold"),
        logprob_threshold: f("logprob_threshold"),
        no_speech_threshold: f("no_speech_threshold"),
        condition_on_prev: o["condition_on_prev_tokens"].as_bool().unwrap_or(false),
        max_new_tokens: None,
        seed: 0,
        prompt: Vec::new(),
    }
}

fn transcriber() -> specrt::speech::transcribe::HostTranscriber {
    specrt::speech::transcribe::HostTranscriber::open(&tiny(), false).unwrap()
}

fn frames_major_mel(name: &str) -> Vec<f32> {
    let (mel, shape) = reference(name);
    frames_major(&mel, shape[0], shape[1])
}

fn check_run(name: &str, got: &specrt::speech::transcribe::Transcript, run: &Value) {
    let windows = run["windows"].as_array().unwrap();
    assert_eq!(got.windows.len(), windows.len(), "{name}: windows");
    for (w, (g, want)) in got.windows.iter().zip(windows).enumerate() {
        assert_eq!(g.seek as u64, want["seek"].as_u64().unwrap(), "{name} window {w}: seek");
        assert_eq!(g.decoder_input, ids(&want["decoder_input"]), "{name} window {w}: prompt with the previous text");
        let attempts = want["attempts"].as_array().unwrap();
        assert_eq!(g.attempts.len(), attempts.len(), "{name} window {w}: fallback attempts");
        for (a, (ga, wa)) in g.attempts.iter().zip(attempts).enumerate() {
            assert_eq!(ga.temperature, wa["temperature"].as_f64().unwrap(), "{name} {w}.{a}");
            assert_eq!(ga.tokens, ids(&wa["tokens"]), "{name} {w}.{a}: tokens");
            let d = &ga.decision;
            assert!((d.compression_ratio - wa["compression_ratio"].as_f64().unwrap()).abs() < 1e-9, "{name} {w}.{a}: compression ratio {}", d.compression_ratio);
            assert!((d.avg_logprob - wa["avg_logprob"].as_f64().unwrap()).abs() < 2e-3, "{name} {w}.{a}: avg logprob {} against {}", d.avg_logprob, wa["avg_logprob"]);
            if let Some(p) = wa["no_speech_prob"].as_f64() {
                assert!((d.no_speech_prob.unwrap() - p).abs() < 1e-3 * p.max(1e-4), "{name} {w}.{a}: no speech {:?} against {p}", d.no_speech_prob);
            }
            assert_eq!((d.needs_fallback, d.should_skip), (wa["needs_fallback"].as_bool().unwrap(), wa["should_skip"].as_bool().unwrap()), "{name} {w}.{a}: fallback decision");
        }
    }
    let segs: Vec<(f64, f64, Vec<u32>)> = got.segments.iter().map(|s| (s.start, s.end, s.tokens.clone())).collect();
    let want: Vec<(f64, f64, Vec<u32>)> = run["segments"].as_array().unwrap().iter().map(|s| (s["start"].as_f64().unwrap(), s["end"].as_f64().unwrap(), ids(&s["tokens"]))).collect();
    assert_eq!(segs.len(), want.len(), "{name}: segments {segs:?}");
    for (g, w) in segs.iter().zip(&want) {
        assert!((g.0 - w.0).abs() < 1e-6 && (g.1 - w.1).abs() < 1e-6 && g.2 == w.2, "{name}: segment {g:?} against {w:?}");
    }
    assert_eq!(got.sequence, ids(&run["sequence"]), "{name}: generate sequence");
}

#[test]
fn the_decoding_rules_come_from_generation_config_and_the_family() {
    let r = rules();
    assert_eq!((r.sot, r.eot, r.no_timestamps, r.timestamp_begin, r.no_speech, r.prev), (513, 512, 523, 524, 522, Some(521)));
    assert_eq!((r.prev_cutoff, r.max_length, r.max_initial_timestamp), (31, 64, Some(50)));
    assert_eq!(r.forced, vec![(1, None), (2, Some(519))]);
    let meta = json(&tiny().join("reference.json"));
    let greedy = specrt::speech::decode::Options { language: Some("en".into()), task: Some("transcribe".into()), ..specrt::speech::decode::Options::greedy() };
    assert_eq!(r.init_tokens(&greedy, None).unwrap(), ids(&meta["prefix"]), "the forced prefix of generate(language=en, task=transcribe)");
    let auto = specrt::speech::decode::Options::greedy();
    assert!(r.needs_detection(&auto), "forced_decoder_ids [1, null] asks for language detection");
    assert_eq!(r.init_tokens(&auto, Some(515)).unwrap(), vec![513, 515, 519, 523]);
    let ts = specrt::speech::decode::Options { timestamps: true, ..auto };
    assert_eq!(r.init_tokens(&ts, Some(515)).unwrap(), vec![513, 515, 519], "timestamps drop no_timestamps");
    for (name, prefix) in [("whisper-tiny.en", vec![50257u32]), ("whisper-base", vec![50258, 50259, 50359]), ("whisper-large-v3-turbo", vec![50258, 50259, 50360])] {
        let c = json(&dir().join(format!("{name}.config.json")));
        let g = json(&dir().join(format!("{name}.generation_config.json")));
        let spec = specrt::spec::Spec::from_hf_config_speech(&c.to_string()).unwrap();
        let real = specrt::speech::decode::Rules::from_configs(&c, &g, spec.speech_decode.as_ref().unwrap()).unwrap();
        let o = specrt::speech::decode::Options { timestamps: true, language: real.multilingual.then(|| "en".to_string()), ..specrt::speech::decode::Options::greedy() };
        assert_eq!(real.init_tokens(&o, None).unwrap(), prefix, "{name}");
        assert_eq!(real.timestamp_begin, real.no_timestamps + 1, "{name}");
        assert_eq!(real.prev_cutoff, 223, "{name}");
    }
    let plan = r.row(&[], false);
    assert!(plan.allow.is_empty());
    let banned: BTreeSet<u32> = plan.edits.iter().map(|e| e.0).collect();
    assert!(banned.contains(&512) && banned.contains(&220) && banned.contains(&1), "begin suppress and suppress on the first step");
    assert!(!r.row(&[100], false).edits.iter().any(|e| e.0 == 512), "EOT is allowed after the first step");
}

#[test]
fn the_compression_ratio_is_zlib_of_the_token_bytes_as_transformers_computes_it() {
    let r = rules();
    let meta = json(&tiny().join("reference.json"));
    let mut n = 0;
    for run in meta["runs"].as_object().unwrap().values() {
        for w in run["windows"].as_array().unwrap() {
            for a in w["attempts"].as_array().unwrap() {
                let got = r.compression_ratio(&ids(&a["tokens"])).unwrap();
                assert_eq!(got, a["compression_ratio"].as_f64().unwrap(), "{:?}", ids(&a["tokens"]));
                n += 1;
            }
        }
    }
    assert_eq!(n, 7, "attempts");
}

#[test]
fn the_fallback_decisions_match_transformers_on_the_recorded_logits() {
    let r = rules();
    let meta = json(&tiny().join("reference.json"));
    let run = &meta["runs"]["fallback"];
    let o = run_options(run);
    let mut kinds = BTreeSet::new();
    for w in run["windows"].as_array().unwrap() {
        for a in w["attempts"].as_array().unwrap() {
            let Some(key) = a["scores"].as_str() else { continue };
            let (pos, _) = reference(&format!("{key}.ids"));
            let (vals, _) = reference(&format!("{key}.scores"));
            let tokens = ids(&a["tokens"]);
            let mut rows = vec![vec![f32::NEG_INFINITY; r.vocab]; tokens.len()];
            for (p, v) in pos.chunks(2).zip(&vals) {
                rows[p[0] as usize][p[1] as usize] = *v;
            }
            let t = a["temperature"].as_f64().unwrap();
            let d = r.decide(&o, &tokens, &rows, t, a["no_speech_prob"].as_f64()).unwrap();
            assert!((d.avg_logprob - a["avg_logprob"].as_f64().unwrap()).abs() < 1e-5, "{key}: {} against {}", d.avg_logprob, a["avg_logprob"]);
            assert_eq!((d.needs_fallback, d.should_skip), (a["needs_fallback"].as_bool().unwrap(), a["should_skip"].as_bool().unwrap()), "{key}");
        }
    }
    for w in run["windows"].as_array().unwrap() {
        for a in w["attempts"].as_array().unwrap() {
            let ratio = r.compression_ratio(&ids(&a["tokens"])).unwrap();
            let low = a["avg_logprob"].as_f64().unwrap() < o.logprob_threshold.unwrap();
            let quiet = a["no_speech_prob"].as_f64().unwrap() > o.no_speech_threshold.unwrap();
            let want = (a["needs_fallback"].as_bool().unwrap(), a["should_skip"].as_bool().unwrap());
            assert_eq!(want, (!(low && quiet) && (ratio > o.compression_ratio_threshold.unwrap() || low), low && quiet), "the transformers rule on the recorded figures");
            kinds.insert(want);
        }
    }
    assert_eq!(kinds.len(), 3, "the fixture covers a fallback, a kept attempt and a skipped window: {kinds:?}");
}

#[test]
fn greedy_through_the_transcriber_equals_generate_and_detects_the_language() {
    let t = transcriber();
    let meta = json(&tiny().join("reference.json"));
    let mel = frames_major_mel("mel.full");
    let states = t.encode(&mel, 0, t.window_frames()).unwrap();
    let (lang, logits) = t.detect(&states).unwrap();
    assert_eq!(u64::from(lang), meta["detected_language"].as_u64().unwrap());
    let (want, _) = reference("language.logits");
    assert!(max_rel(&logits, &want) < 1e-3);
    let o = specrt::speech::decode::Options { language: Some("en".into()), task: Some("transcribe".into()), max_new_tokens: Some(12), ..specrt::speech::decode::Options::greedy() };
    let got = t.transcribe_mel(&mel, &o, &mut |_, _, _, _, _| None).unwrap();
    assert_eq!(got.windows[0].decoder_input, ids(&meta["prefix"]));
    let generated = ids(&meta["generate"]);
    assert_eq!(got.windows[0].attempts[0].tokens, generated, "generate(language=en, max_new_tokens=12)");
    let (pcm, _) = reference("audio.full");
    assert_eq!(t.transcribe(&pcm, &o).unwrap().sequence, got.sequence, "from the audio through the host front end");
}

#[test]
fn timestamp_decoding_equals_transformers_generate() {
    let t = transcriber();
    let meta = json(&tiny().join("reference.json"));
    let run = &meta["runs"]["timestamps"];
    let o = specrt::speech::decode::Options { timestamps: true, ..specrt::speech::decode::Options::greedy() };
    let got = t.transcribe_mel(&frames_major_mel("mel.full"), &o, &mut |_, _, _, _, _| None).unwrap();
    assert_eq!(got.language.map(u64::from), meta["detected_language"].as_u64());
    check_run("timestamps", &got, run);
    assert!(got.sequence.iter().filter(|x| **x >= 524).count() >= 4, "timestamp pairs");
}

#[test]
fn a_two_window_clip_seeks_by_the_last_timestamp_and_conditions_on_the_previous_text() {
    let t = transcriber();
    let meta = json(&tiny().join("reference.json"));
    let mel = frames_major_mel("mel.long");
    assert!(mel.len() / 80 > t.window_frames(), "the clip is longer than one window");
    let run = &meta["runs"]["long"];
    let got = t.transcribe_mel(&mel, &run_options(run), &mut |_, _, _, _, _| None).unwrap();
    check_run("long", &got, run);
    assert_eq!(got.windows[1].decoder_input[0], 521, "the second window starts with <|startofprev|>");
    let (pcm, _) = reference("audio.long");
    let host = specrt::speech::transcribe::log_mel_clip(&t.plan, &pcm).unwrap();
    assert!(max_abs(&host, &mel) < 1e-4, "the long front end matches transformers");
}

#[test]
fn temperature_fallback_redoes_a_window_and_skips_silence_as_transformers_does() {
    let t = transcriber();
    let meta = json(&tiny().join("reference.json"));
    let run = &meta["runs"]["fallback"];
    let windows = run["windows"].as_array().unwrap().clone();
    let mut forced = |w: usize, a: usize, step: usize, row: &[f32], temperature: f64| -> Option<u32> {
        if temperature == 0.0 {
            return None;
        }
        let tok = ids(&windows[w]["attempts"][a]["tokens"])[step];
        assert!(row[tok as usize].is_finite(), "window {w} attempt {a} step {step}: the sampled token {tok} is masked by our rules");
        Some(tok)
    };
    let got = t.transcribe_mel(&frames_major_mel("mel.long"), &run_options(run), &mut forced).unwrap();
    check_run("fallback", &got, run);
}

#[test]
fn sampling_at_a_temperature_keeps_to_the_top_k_of_the_warped_row() {
    let r = rules();
    let mut row: Vec<f32> = (0..r.vocab).map(|i| (i % 97) as f32 * 0.1).collect();
    r.warp(&mut row, 0.5);
    assert_eq!(row.iter().filter(|v| v.is_finite()).count(), 54, "the top 50 and every tie of the 50th, as TopKLogitsWarper keeps them");
    for draw in 0..64 {
        let t = r.pick(&row, 0.5, 7, draw);
        assert!(row[t as usize].is_finite());
    }
    assert_eq!(r.pick(&row, 0.5, 7, 3), r.pick(&row, 0.5, 7, 3), "seeded");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_row_plans_on_the_gpu_duplex_kernels_equal_the_host_rules() {
    let r = rules();
    let spec = specrt::spec::Spec::from_hf_config_speech(&std::fs::read_to_string(tiny().join("config.json")).unwrap()).unwrap();
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let cache = std::env::var_os("SPECRT_CACHE").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("specrt-speech-jit"));
    let jit = specrt::jit::Jit::new(ctx.clone(), cache).unwrap();
    let gpu = specrt::speech::cuda::CudaRowPlans::load(&ctx.default_stream(), &jit, &spec).unwrap();
    let meta = json(&tiny().join("reference.json"));
    let seq = ids(&meta["runs"]["timestamps"]["windows"][0]["attempts"][0]["tokens"]);
    let (greedy, _) = reference("greedy.logits");
    let base = &greedy[..r.vocab];
    let histories: Vec<(Vec<u32>, bool)> = vec![(vec![], false), (vec![501], false), (vec![], true), (seq[..1].to_vec(), true), (seq[..2].to_vec(), true), (seq[..15].to_vec(), true), (seq[..16].to_vec(), true)];
    let mut plans: Vec<specrt::duplex::sampling::RowPlan> = histories.iter().map(|(h, ts)| r.row(h, *ts)).collect();
    plans.push(r.language_row());
    plans.push(specrt::speech::decode::Rules::forced_row(519));
    let logits: Vec<f32> = plans.iter().flat_map(|_| base.iter().copied()).collect();
    let got = gpu.run(&logits, &plans).unwrap();
    for (i, p) in plans.iter().enumerate() {
        let mut want = base.to_vec();
        specrt::speech::decode::apply(&mut want, p);
        assert_eq!(&got[i * r.vocab..(i + 1) * r.vocab], &want[..], "row {i}");
    }
    println!("gpu row plans: {} rows bit identical to the host rules", plans.len());
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_gpu_transcriber_matches_transformers_and_the_host_on_every_tiny_run() {
    use specrt::speech::transcribe::Engine;
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let cache = std::env::var_os("SPECRT_CACHE").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("specrt-speech-jit"));
    let jit = specrt::jit::Jit::new(ctx.clone(), cache).unwrap();
    let g = specrt::speech::gpu::GpuTranscriber::open(&ctx.default_stream(), &jit, &tiny()).unwrap();
    let meta = json(&tiny().join("reference.json"));
    let mel = frames_major_mel("mel.full");
    let states = g.encode(&mel, 0, g.window_frames()).unwrap();
    let (want, _) = reference("encoder.out");
    let got = ctx.default_stream().clone_dtoh(&states).unwrap();
    let err = max_rel(&got, &want);
    println!("gpu encoder.out: max rel {err:.2e}");
    assert!(err < 1e-4, "{err}");
    let prompt = ids(&meta["prefix"]);
    let mut seq = g.engine.start(&states).unwrap();
    let logits = g.engine.prefill(&mut seq, &prompt).unwrap();
    let (want_prefix, _) = reference("prefix.logits");
    let pe = max_rel(&logits, &want_prefix);
    let mut rows = logits[(prompt.len() - 1) * g.rules.vocab..].to_vec();
    for &t in &ids(&meta["greedy"])[..ids(&meta["greedy"]).len() - 1] {
        rows.extend(g.engine.step(&mut seq, t).unwrap());
    }
    let (want_greedy, _) = reference("greedy.logits");
    let raw: Vec<f32> = rows.clone();
    let host = transcriber();
    let mut hseq = host.engine.start(&host.encode(&mel, 0, host.window_frames()).unwrap()).unwrap();
    let mut hrows = host.engine.prefill(&mut hseq, &prompt).unwrap()[(prompt.len() - 1) * g.rules.vocab..].to_vec();
    for &t in &ids(&meta["greedy"])[..ids(&meta["greedy"]).len() - 1] {
        hrows.extend(host.engine.step(&mut hseq, t).unwrap());
    }
    let he = max_rel(&raw, &hrows);
    let ge = max_rel(&raw, &want_greedy);
    println!("gpu prefix.logits max rel {pe:.2e}, greedy.logits {ge:.2e}, greedy rows against the host {he:.2e}");
    assert!(pe < 1e-3 && ge < 1e-3 && he < 1e-3, "{pe} {ge} {he}");
    let (lang, _) = g.detect(&states).unwrap();
    assert_eq!(u64::from(lang), meta["detected_language"].as_u64().unwrap());
    let o = specrt::speech::decode::Options { language: Some("en".into()), task: Some("transcribe".into()), max_new_tokens: Some(12), ..specrt::speech::decode::Options::greedy() };
    let (pcm, _) = reference("audio.full");
    assert_eq!(g.transcribe(&pcm, &o).unwrap().windows[0].attempts[0].tokens, ids(&meta["generate"]), "generate from the audio through the GPU front end");
    let ts = specrt::speech::decode::Options { timestamps: true, ..specrt::speech::decode::Options::greedy() };
    check_run("gpu timestamps", &g.transcribe_mel(&mel, &ts, &mut |_, _, _, _, _| None).unwrap(), &meta["runs"]["timestamps"]);
    let run = &meta["runs"]["long"];
    check_run("gpu long", &g.transcribe_mel(&frames_major_mel("mel.long"), &run_options(run), &mut |_, _, _, _, _| None).unwrap(), run);
    let run = &meta["runs"]["fallback"];
    let windows = run["windows"].as_array().unwrap().clone();
    let mut forced = |w: usize, a: usize, step: usize, _: &[f32], t: f64| (t != 0.0).then(|| ids(&windows[w]["attempts"][a]["tokens"])[step]);
    check_run("gpu fallback", &g.transcribe_mel(&frames_major_mel("mel.long"), &run_options(run), &mut forced).unwrap(), run);
    println!("gpu transcriber: greedy, timestamps, long and fallback runs equal transformers, {} weight bytes", g.engine.weight_bytes);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_gpu_mimi_decoder_matches_transformers_frame_by_frame_and_batched() {
    use specrt::codec::cuda::CudaCodec;
    let host = mimi_host();
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let cache = std::env::var_os("SPECRT_CACHE").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("specrt-codec-jit"));
    let jit = specrt::jit::Jit::new(ctx.clone(), cache).unwrap();
    let (latents, shape) = mimi_ref("latents");
    let (want, _) = mimi_ref("audio");
    let frames = shape[0];
    let gpu = CudaCodec::load(&ctx, &jit, &host, &[(1, 1), (1, 2), (frames, 1)]).unwrap();
    let mut s = gpu.sessions(&gpu.decoder, 1).unwrap();
    assert_eq!(s.floats_per_session(), host.plan.decoder.state_floats());
    let whole = gpu.run(&gpu.decoder, &latents, &mut s).unwrap();
    let e = max_rel(&whole, &want);
    let host_whole = host.decode(&latents).unwrap();
    let h = max_rel(&whole, &host_whole);
    println!("gpu mimi decode: max rel {e:.2e} against transformers, {h:.2e} against the host");
    assert!(e < 1e-4 && h < 1e-4, "{e} {h}");
    let d = host.plan.latent_dim;
    let mut s = gpu.sessions(&gpu.decoder, 1).unwrap();
    let streamed: Vec<f32> = latents.chunks(d).flat_map(|z| gpu.run(&gpu.decoder, z, &mut s).unwrap()).collect();
    let f = max_rel(&streamed, &whole);
    println!("gpu mimi decode: frame by frame against whole clip max rel {f:.2e}");
    assert!(f < 1e-5, "{f}");
    let reversed: Vec<Vec<f32>> = latents.chunks(d).rev().map(<[f32]>::to_vec).collect();
    let mut pair = gpu.sessions(&gpu.decoder, 2).unwrap();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let n = host.plan.frame_samples;
    for (i, z) in latents.chunks(d).enumerate() {
        let both = gpu.run(&gpu.decoder, &[z.to_vec(), reversed[i].clone()].concat(), &mut pair).unwrap();
        a.extend_from_slice(&both[..n]);
        b.extend_from_slice(&both[n..]);
    }
    let mut solo = gpu.sessions(&gpu.decoder, 1).unwrap();
    let alone: Vec<f32> = reversed.iter().flat_map(|z| gpu.run(&gpu.decoder, z, &mut solo).unwrap()).collect();
    let (ea, eb) = (max_rel(&a, &streamed), max_rel(&b, &alone));
    println!("gpu mimi decode: two sessions batched against each alone {ea:.2e} {eb:.2e}");
    assert!(ea < 1e-5 && eb < 1e-5, "{ea} {eb}");
}

#[test]
#[ignore = "needs nvrtc, no GPU: compiles every Mimi decoder kernel of the real Phonon config for sm_120"]
fn every_mimi_kernel_of_the_real_config_compiles_with_nvrtc() {
    let plan = specrt::codec::graph::plan(&json(&dir().join("phonon-31m-en.hub-92741c5.config.json"))).unwrap();
    let target = kir::Target::sm_120();
    let mut all = Vec::new();
    for (steps, batch) in [(1, 1), (1, 8), (16, 1)] {
        all.extend(specrt::codec::compile::plan(&plan.decoder, steps, batch).unwrap().kernels(&target).unwrap());
    }
    all.sort_by_key(|k| k.name());
    all.dedup_by_key(|k| k.name());
    let m = specrt::codec::cuda::module(&all).unwrap();
    let opts = cudarc::nvrtc::CompileOptions { arch: Some("compute_120"), options: vec!["--std=c++17".into()], ..Default::default() };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(m.source.as_str(), opts).unwrap().to_src();
    for name in &m.kernels {
        assert!(ptx.contains(&format!(".entry {name}(")), "{name} missing from the PTX");
    }
    assert!(m.source.as_str().contains("if (ACT == 4) return (v > 0.0f ? v : expf(v) - 1.0f);"), "ELU is lowered from the kir macro");
    println!("{} kernels, {} conv variants", m.kernels.len(), all.len());
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_gpu_log_mel_and_stem_match_transformers() {
    let host = specrt::speech::HostSpeech::open(&tiny()).unwrap();
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let cache = std::env::var_os("SPECRT_CACHE").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("specrt-speech-jit"));
    let jit = specrt::jit::Jit::new(ctx.clone(), cache).unwrap();
    let gpu = specrt::speech::cuda::CudaSpeech::load(&ctx.default_stream(), &jit, &host).unwrap();
    let (bins, frames) = (host.plan.mel.bins, host.plan.mel.frames);
    for (audio, mel) in [("audio.full", "mel.full"), ("audio.short", "mel.short")] {
        let (pcm, _) = reference(audio);
        let (want, _) = reference(mel);
        let (got, stem) = gpu.run(&pcm).unwrap();
        let err = max_abs(&got, &frames_major(&want, bins, frames));
        println!("gpu {mel}: max abs {err:.2e}");
        assert!(err < 1e-4, "{mel}: {err}");
        if audio == "audio.full" {
            let (want, _) = reference("encoder.stem");
            let err = max_rel(&stem, &want);
            println!("gpu encoder.stem: max rel {err:.2e}, {} launches", gpu.launches);
            assert!(err < 1e-4, "{err}");
        }
    }
}
