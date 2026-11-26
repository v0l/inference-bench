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
    Need { piece: "learned absolute positions added to the stem output and to token embeddings", state: "exists as an op", evidence: Some(("specrt/src/denoiser.rs", "\"table_add\" =>")), slice: "b" },
    Need { piece: "LayerNorm with bias", state: "exists as an op", evidence: Some(("specrt/src/denoiser.rs", "\"layer_norm\" =>")), slice: "b" },
    Need { piece: "linear with bias, k_proj without bias", state: "exists as an op", evidence: Some(("specrt/src/denoiser.rs", "DitOp::Linear { weight, input: from()?, output: into()?, rows: self.rows()?, add: self.add, bias: self.bias, act }")), slice: "b" },
    Need { piece: "bidirectional encoder self attention", state: "exists as an op", evidence: Some(("specrt/src/denoiser.rs", "\"attention\" => DitOp::JointAttention")), slice: "b" },
    Need { piece: "cross attention from decoder rows to encoder states", state: "exists as a denoiser op only (vis_flash_kv); no decoder graph op", evidence: Some(("specrt/src/denoiser.rs", "\"cross_attention\" => DitOp::CrossAttention")), slice: "b" },
    Need { piece: "cross attention K and V computed once per window and kept per sequence", state: "new decoder graph op and session state", evidence: None, slice: "b" },
    Need { piece: "decoder causal self attention with a paged KV cache, no rope", state: "exists (paged KV, nope_all)", evidence: Some(("specrt/src/family.rs", "pub nope_all: bool")), slice: "b" },
    Need { piece: "tied output head", state: "exists", evidence: Some(("specrt/src/spec.rs", "\"tie_word_embeddings\"")), slice: "-" },
    Need { piece: "suppress and begin suppress lists, forced prefix", state: "new sampler plan rows (allow and ban lists exist for duplex)", evidence: Some(("specrt/src/duplex/sampling.rs", "pub allow: Vec<u32>")), slice: "c" },
    Need { piece: "language detection over the language token logits at step 0", state: "new decoding rule", evidence: None, slice: "c" },
    Need { piece: "timestamp rules, no speech probability, temperature fallback, compression ratio, avg logprob", state: "new decoding rules (host)", evidence: None, slice: "c" },
    Need { piece: "long audio: 30 s windows with seek by the last timestamp, previous text as prompt", state: "new host loop", evidence: None, slice: "c" },
];

const TTS: &[Need] = &[
    Need { piece: "Mimi decoder SEANet: causal conv k7 and k3 with ELU, constant pad, dilated residual blocks", state: "new graph entry over the codec convs; ELU is a new expression", evidence: Some(("specrt/src/codec/graph.rs", "\"conv\" =>")), slice: "e" },
    Need { piece: "Mimi transposed conv upsampling (ratios 4, 5, 6) with an overlap tail", state: "exists as codec conv_up", evidence: Some(("specrt/src/codec/graph.rs", "\"conv_up\" =>")), slice: "e" },
    Need { piece: "Mimi depthwise transposed conv 12.5 Hz to 200 Hz upsample (groups = channels, no bias)", state: "new: conv_up is dense only", evidence: None, slice: "e" },
    Need { piece: "Mimi decoder transformer at 200 Hz: 2 layers, LayerNorm, layer scale, rope pairs, sliding context 250", state: "new graph (codec stack has no attention)", evidence: None, slice: "e" },
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
fn the_engine_refuses_whisper_and_says_which_ops_are_missing() {
    for name in ["tiny-whisper/config.json", "whisper-tiny.en.config.json", "whisper-base.config.json", "whisper-large-v3-turbo.config.json"] {
        let text = std::fs::read_to_string(dir().join(name)).unwrap();
        let err = specrt::spec::Spec::from_hf_config(&text).unwrap_err();
        assert!(format!("{err:#}").contains("model_type whisper is not supported"), "{name}: {err:#}");
        assert!(specrt::spec::Spec::from_hf_config_duplex(&text).is_err(), "{name}");
        assert!(specrt::spec::Spec::describe_hf_config(&text).is_err(), "{name}");
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
    assert_eq!(missing.len(), 8, "{missing:#?}");
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
