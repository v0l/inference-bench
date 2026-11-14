use crate::chat::ChatTemplate;
use crate::runtime::{Runtime, Sampling};
use crate::spec::{BackendChoice, Spec};
use crate::weights::find_in_snapshot;
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::path::Path;
use tokenizers::Tokenizer;

pub const DEFAULT_CTX: usize = 2048;
pub const MAX_CTX: usize = 131072;

pub fn resolve_spec(model: &str) -> Result<Spec> {
    if model.ends_with(".toml") {
        let mut spec = Spec::load(Path::new(model))?;
        complete_spec(&mut spec)?;
        return Ok(spec);
    }
    if model.ends_with(".gguf") {
        let mut spec = crate::gguf::Gguf::open(Path::new(model))?.spec().with_context(|| model.to_string())?;
        spec.repo = model.to_string();
        return Ok(spec);
    }
    let config = find_in_snapshot(model, "config.json")?;
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(&config)?)
        .with_context(|| model.to_string())?;
    spec.repo = model.to_string();
    spec.name = model.rsplit('/').next().unwrap_or(model).to_lowercase();
    Ok(spec)
}

pub fn complete_spec(spec: &mut Spec) -> Result<()> {
    if spec.repo.ends_with(".gguf") || spec.vision.is_some() {
        return Ok(());
    }
    if let Ok(path) = find_in_snapshot(&spec.repo, "config.json") {
        spec.complete_from_checkpoint_config(&std::fs::read_to_string(path)?)?;
    }
    Ok(())
}

const END_OF_TURN: [&str; 6] =
    ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<end_of_turn>", "<|endoftext|>", "</s>"];

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SamplingDefaults {
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
}

impl SamplingDefaults {
    pub fn from_generation_config(text: Option<&str>) -> Self {
        let Some(v) = text.and_then(|t| serde_json::from_str::<Value>(t).ok()) else { return Self::default() };
        SamplingDefaults {
            top_k: v["top_k"].as_u64().map(|n| n as u32),
            top_p: v["top_p"].as_f64().map(|n| n as f32),
            min_p: v["min_p"].as_f64().map(|n| n as f32),
        }
    }
}

pub fn stop_ids(
    generation_config: Option<&str>,
    tokenizer_config: Option<&str>,
    lookup: &dyn Fn(&str) -> Option<u32>,
) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    if let Some(v) = generation_config.and_then(|t| serde_json::from_str::<Value>(t).ok()) {
        match &v["eos_token_id"] {
            Value::Number(n) => out.extend(n.as_u64().map(|n| n as u32)),
            Value::Array(a) => out.extend(a.iter().filter_map(|n| n.as_u64().map(|n| n as u32))),
            _ => {}
        }
    }
    if let Some(v) = tokenizer_config.and_then(|t| serde_json::from_str::<Value>(t).ok()) {
        let eos = match &v["eos_token"] {
            Value::String(s) => Some(s.clone()),
            Value::Object(o) => o.get("content").and_then(Value::as_str).map(String::from),
            _ => None,
        };
        out.extend(eos.as_deref().and_then(lookup));
        let template = v["chat_template"].to_string();
        for marker in END_OF_TURN {
            if template.contains(marker) {
                out.extend(lookup(marker));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    UnsupportedArchitecture,
    ModelNotFound,
    Failed,
}

pub fn failure_kind(err: &anyhow::Error) -> FailureKind {
    let lower = format!("{err:#}").to_lowercase();
    if lower.contains("not supported") {
        FailureKind::UnsupportedArchitecture
    } else if lower.contains("no config.json")
        || lower.contains("no model.safetensors")
        || lower.contains("no such file")
    {
        FailureKind::ModelNotFound
    } else {
        FailureKind::Failed
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Reply {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub peak_device_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PromptLogits {
    pub tokens: Vec<u32>,
    pub vocab: usize,
    pub values: Vec<f32>,
}

pub(crate) struct Assets {
    pub spec: Spec,
    pub tokenizer: Tokenizer,
    pub chat: Option<ChatTemplate>,
    pub stop: Vec<u32>,
    pub sampling: SamplingDefaults,
    pub media: Option<crate::vision::MediaAssets>,
}

pub fn without_truncation(mut tokenizer: Tokenizer) -> Result<Tokenizer> {
    tokenizer.with_truncation(None).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    Ok(tokenizer)
}

impl Assets {
    fn from_gguf(spec: Spec) -> Result<Self> {
        let g = crate::gguf::Gguf::open(Path::new(&spec.repo))?;
        let tokenizer: Tokenizer = std::str::FromStr::from_str(&g.tokenizer_json()?).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let tokenizer = without_truncation(tokenizer)?;
        let tok_cfg = g.tokenizer_config_json();
        let chat = match &tok_cfg {
            Some(c) => ChatTemplate::from_files(c, None)?,
            None => None,
        };
        let gen_cfg = g.generation_config_json();
        let stop = stop_ids(Some(&gen_cfg), tok_cfg.as_deref(), &|t| tokenizer.token_to_id(t));
        Ok(Assets { spec, tokenizer, chat, stop, sampling: SamplingDefaults::default(), media: None })
    }

    pub fn load(model: &str) -> Result<Self> {
        let spec = resolve_spec(model)?;
        if model.ends_with(".gguf") {
            return Self::from_gguf(spec);
        }
        let tokenizer = Tokenizer::from_file(find_in_snapshot(&spec.repo, "tokenizer.json")?)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let tokenizer = without_truncation(tokenizer)?;
        let chat = match find_in_snapshot(&spec.repo, "tokenizer_config.json") {
            Ok(path) => {
                let jinja = find_in_snapshot(&spec.repo, "chat_template.jinja").ok().and_then(|p| std::fs::read_to_string(p).ok());
                ChatTemplate::from_files(&std::fs::read_to_string(path)?, jinja.as_deref())?
            }
            Err(_) => None,
        };
        let read = |f: &str| {
            find_in_snapshot(&spec.repo, f).ok().and_then(|p| std::fs::read_to_string(p).ok())
        };
        let sampling = SamplingDefaults::from_generation_config(read("generation_config.json").as_deref());
        let stop = stop_ids(
            read("generation_config.json").as_deref(),
            read("tokenizer_config.json").as_deref(),
            &|t| tokenizer.token_to_id(t),
        );
        let media = media_assets(&spec, &tokenizer)?;
        Ok(Assets { spec, tokenizer, chat, stop, sampling, media })
    }

    pub fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.tokenizer.encode(text, false).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn encode_messages(&self, messages: &[Value], template: Option<&Value>) -> Result<Vec<u32>> {
        let text = render_prompt(self.chat.as_ref(), messages, template)?;
        self.tokens(&text)
    }

    pub fn encode_media_messages(
        &self,
        messages: &[Value],
        template: Option<&Value>,
        items: &[crate::vision::MediaItem],
        budget: crate::vision::MediaBudget,
    ) -> Result<(Vec<u32>, Vec<crate::sched::MediaInput>, usize)> {
        let media = self.media.as_ref().context("the model has no vision_config")?;
        let processed = media.process(items, budget)?;
        let text = render_prompt(self.chat.as_ref(), &strip_media_urls(messages), template)?;
        let expanded = media.expand(&text, &processed)?;
        let ids = self.tokens(&expanded)?;
        let vision_tokens = ids.iter().filter(|&&t| t == media.ids.image || t == media.ids.video).count();
        let inputs = processed.into_iter().zip(items).map(|(processed, item)| crate::sched::MediaInput { processed, hash: item.hash }).collect();
        Ok((ids, inputs, vision_tokens))
    }
}

fn ensure_media(ok: bool) -> Result<()> {
    if ok { Ok(()) } else { bail!("the model does not accept image or video input: its config has no vision_config") }
}

pub fn strip_media_urls(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            let mut m = m.clone();
            if let Some(parts) = m.get_mut("content").and_then(Value::as_array_mut) {
                for p in parts.iter_mut() {
                    match p["type"].as_str() {
                        Some("image_url") => *p = serde_json::json!({"type": "image"}),
                        Some("video_url") => *p = serde_json::json!({"type": "video"}),
                        _ => {}
                    }
                }
            }
            m
        })
        .collect()
}

fn media_assets(spec: &Spec, tokenizer: &Tokenizer) -> Result<Option<crate::vision::MediaAssets>> {
    use crate::vision::{MediaAssets, MediaTokens, Preprocessor};
    let Some(vision) = &spec.vision else { return Ok(None) };
    let read = |f: &str| find_in_snapshot(&spec.repo, f).ok().and_then(|p| std::fs::read_to_string(p).ok());
    let (Some(image), Some(video)) = (read("preprocessor_config.json"), read("video_preprocessor_config.json")) else { return Ok(None) };
    let text = |id: u32| tokenizer.id_to_token(id).with_context(|| format!("no token {id}"));
    let tokens = MediaTokens {
        image: text(vision.tokens.image)?,
        video: text(vision.tokens.video)?,
        start: text(vision.tokens.start)?,
        end: text(vision.tokens.end)?,
    };
    Ok(Some(MediaAssets { image: Preprocessor::from_json(&image)?, video: Preprocessor::from_json(&video)?, tokens, ids: vision.tokens.clone() }))
}

pub fn render_prompt(chat: Option<&ChatTemplate>, messages: &[Value], template: Option<&Value>) -> Result<String> {
    match (chat, template) {
        (Some(t), Some(vars)) => t.render_with(messages, vars),
        _ => Ok(messages.iter().filter_map(|m| m["content"].as_str()).collect::<Vec<_>>().join("\n\n")),
    }
}

enum Device {
    Cuda(Runtime),
    #[cfg(feature = "wgpu")]
    Wgpu(crate::backends::dense::Dense<crate::backends::wgpu::Wgpu>),
}

impl Device {
    fn load(spec: &Spec, max_ctx: usize) -> Result<Self> {
        match spec.runtime.backend {
            BackendChoice::Cuda => Ok(Device::Cuda(Runtime::load(spec, max_ctx)?)),
            BackendChoice::Wgpu => Self::wgpu(spec, max_ctx),
            BackendChoice::Auto => match Runtime::load(spec, max_ctx) {
                Ok(rt) => Ok(Device::Cuda(rt)),
                Err(cuda) => Self::wgpu(spec, max_ctx).map_err(|wgpu| cuda.context(format!("wgpu fallback failed too: {wgpu:#}"))),
            },
        }
    }

    #[cfg(feature = "wgpu")]
    fn wgpu(spec: &Spec, max_ctx: usize) -> Result<Self> {
        let be = crate::backends::wgpu::Wgpu::new(spec.runtime.cache_dir.clone().or_else(|| Some(crate::jit::default_cache_dir())))?;
        Ok(Device::Wgpu(crate::backends::dense::Dense::load(be, spec, max_ctx)?))
    }

    #[cfg(not(feature = "wgpu"))]
    fn wgpu(_: &Spec, _: usize) -> Result<Self> {
        bail!("backend wgpu was requested but this build has no wgpu support (build with --features wgpu)")
    }

    fn generate_with(
        &mut self,
        prompt: &[u32],
        new_tokens: usize,
        sampling: Sampling,
        stop: &[u32],
        chunked: bool,
        on_tokens: &mut dyn FnMut(&[u32]) -> bool,
    ) -> Result<crate::runtime::Generation> {
        match self {
            Device::Cuda(r) => r.generate_with(prompt, new_tokens, sampling, stop, chunked, on_tokens),
            #[cfg(feature = "wgpu")]
            Device::Wgpu(d) => d.generate_with(prompt, new_tokens, sampling, stop, chunked, on_tokens),
        }
    }

    fn has_vision(&self) -> bool {
        match self {
            Device::Cuda(r) => r.has_vision(),
            #[cfg(feature = "wgpu")]
            Device::Wgpu(_) => false,
        }
    }

    fn set_media(&mut self, plan: Option<&crate::vision::MediaPlan>) -> Result<()> {
        match self {
            Device::Cuda(r) => r.set_media(0, plan),
            #[cfg(feature = "wgpu")]
            Device::Wgpu(_) => bail!("the wgpu backend has no vision encoder"),
        }
    }

    fn plan_media(&mut self, prompt: &[u32], media: &[crate::sched::MediaInput]) -> Result<std::sync::Arc<crate::vision::MediaPlan>> {
        match self {
            Device::Cuda(r) => r.plan_media(prompt, media),
            #[cfg(feature = "wgpu")]
            Device::Wgpu(_) => bail!("the wgpu backend has no vision encoder"),
        }
    }

    fn prompt_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        match self {
            Device::Cuda(r) => r.prompt_logits(tokens),
            #[cfg(feature = "wgpu")]
            Device::Wgpu(d) => d.prompt_logits(tokens),
        }
    }
}

pub struct Engine {
    spec: Spec,
    tokenizer: Tokenizer,
    chat: Option<ChatTemplate>,
    runtime: Device,
    max_ctx: usize,
    stop: Vec<u32>,
    sampling: SamplingDefaults,
    media: Option<crate::vision::MediaAssets>,
}

impl Engine {
    pub fn load(model: &str, max_ctx: Option<usize>) -> Result<Self> {
        Self::load_with(model, max_ctx, None)
    }

    pub fn load_with(model: &str, max_ctx: Option<usize>, runtime: Option<&crate::spec::RuntimeOptions>) -> Result<Self> {
        let Assets { mut spec, tokenizer, chat, stop, sampling, media } = Assets::load(model)?;
        if let Some(r) = runtime {
            spec.runtime = r.clone();
        }
        let max_ctx = max_ctx.unwrap_or(DEFAULT_CTX).min(MAX_CTX);
        tracing::info!(model, spec = %spec.name, max_ctx, "loading model");
        let runtime = Device::load(&spec, max_ctx)?;
        Ok(Self { spec, tokenizer, chat, runtime, max_ctx, stop, sampling, media })
    }

    pub fn spec(&self) -> &Spec {
        &self.spec
    }

    pub fn sampling_defaults(&self) -> SamplingDefaults {
        self.sampling
    }

    pub fn thinking(&self) -> crate::chat::Thinking {
        self.chat.as_ref().map(|c| c.thinking()).unwrap_or_default()
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        let enc =
            self.tokenizer.encode(text, false).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn encode_messages(&self, messages: &[Value], template: Option<&Value>) -> Result<Vec<u32>> {
        let text = render_prompt(self.chat.as_ref(), messages, template)?;
        self.tokens(&text)
    }

    fn fits(&self, prompt: usize, max_tokens: usize) -> Result<()> {
        if prompt + max_tokens > self.max_ctx {
            bail!(
                "prompt of {prompt} tokens plus {max_tokens} new tokens exceeds the context of {}",
                self.max_ctx
            );
        }
        Ok(())
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampling: Sampling,
    ) -> Result<Reply> {
        let ids = self.tokens(prompt)?;
        self.run(ids, max_tokens, sampling, false, &mut |_| true)
    }

    pub fn chat(
        &mut self,
        messages: &[Value],
        template: Option<&Value>,
        max_tokens: usize,
        sampling: Sampling,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Reply> {
        let ids = self.encode_messages(messages, template)?;
        self.run(ids, max_tokens, sampling, true, on_text)
    }

    fn run(
        &mut self,
        ids: Vec<u32>,
        max_tokens: usize,
        sampling: Sampling,
        streaming: bool,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Reply> {
        self.fits(ids.len(), max_tokens)?;
        let mut so_far: Vec<u32> = Vec::new();
        let mut sent = String::new();
        let tokenizer = &self.tokenizer;
        let stop: &[u32] = if streaming { &self.stop } else { &[] };
        let g = self.runtime.generate_with(&ids, max_tokens, sampling, stop, streaming, &mut |fresh| {
            so_far.extend_from_slice(fresh);
            match tokenizer.decode(&so_far, true) {
                Ok(text)
                    if text.len() > sent.len()
                        && text.starts_with(&sent)
                        && !text.ends_with('\u{fffd}') =>
                {
                    let more = on_text(&text[sent.len()..]);
                    sent = text;
                    more
                }
                _ => true,
            }
        })?;
        let text =
            self.tokenizer.decode(&g.tokens, true).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        if text.len() > sent.len() && text.starts_with(&sent) {
            on_text(&text[sent.len()..]);
        }
        Ok(Reply {
            text,
            prompt_tokens: ids.len(),
            completion_tokens: g.tokens.len(),
            prefill_secs: g.prefill_secs,
            decode_secs: g.decode_secs,
            peak_device_bytes: g.peak_device_bytes,
        })
    }

    pub fn accepts_media(&self) -> bool {
        self.media.is_some() && self.runtime.has_vision()
    }

    pub fn chat_media(
        &mut self,
        messages: &[Value],
        template: Option<&Value>,
        items: &[crate::vision::MediaItem],
        budget: crate::vision::MediaBudget,
        max_tokens: usize,
        sampling: Sampling,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<(Reply, usize)> {
        ensure_media(self.accepts_media())?;
        let media = self.media.as_ref().context("the model has no vision_config")?;
        let processed = media.process(items, budget)?;
        let text = render_prompt(self.chat.as_ref(), &strip_media_urls(messages), template)?;
        let ids = self.tokens(&media.expand(&text, &processed)?)?;
        let vision_tokens = ids.iter().filter(|&&t| t == media.ids.image || t == media.ids.video).count();
        let inputs: Vec<crate::sched::MediaInput> =
            processed.into_iter().zip(items).map(|(processed, item)| crate::sched::MediaInput { processed, hash: item.hash }).collect();
        let plan = self.runtime.plan_media(&ids, &inputs)?;
        self.runtime.set_media(Some(&plan))?;
        let reply = self.run(ids, max_tokens, sampling, true, on_text);
        self.runtime.set_media(None)?;
        Ok((reply?, vision_tokens))
    }

    pub fn logits(&mut self, prompt: &str) -> Result<PromptLogits> {
        let tokens = self.tokens(prompt)?;
        let values = self.runtime.prompt_logits(&tokens)?;
        Ok(PromptLogits { tokens, vocab: self.spec.dims.vocab, values })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tokenizer_json_truncation_limit_does_not_cut_prompts() {
        let json = r#"{"version":"1.0","truncation":{"direction":"Right","max_length":8,"strategy":"LongestFirst","stride":0},"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"a":0,"[UNK]":1},"unk_token":"[UNK]"}}"#;
        let text = "a ".repeat(40);
        let cut: Tokenizer = std::str::FromStr::from_str(json).unwrap();
        assert_eq!(cut.encode(text.as_str(), false).unwrap().get_ids().len(), 8);
        let fixed = without_truncation(cut).unwrap();
        assert_eq!(fixed.encode(text.as_str(), false).unwrap().get_ids().len(), 40);
    }

    #[test]
    fn unsupported_model_types_are_reported_as_unsupported_architecture() {
        let e = anyhow::anyhow!("model_type gemma3 is not supported");
        assert_eq!(failure_kind(&e), FailureKind::UnsupportedArchitecture);
    }

    #[test]
    fn sampling_defaults_come_from_generation_config() {
        let d = SamplingDefaults::from_generation_config(Some(r#"{"top_k": 20, "top_p": 0.95, "temperature": 1.0}"#));
        assert_eq!(d, SamplingDefaults { top_k: Some(20), top_p: Some(0.95), min_p: None });
        assert_eq!(SamplingDefaults::from_generation_config(None), SamplingDefaults::default());
    }

    #[test]
    fn stop_ids_come_from_generation_config_eos_and_the_template_end_of_turn() {
        let vocab = |t: &str| match t {
            "<|im_end|>" => Some(151645),
            "<|endoftext|>" => Some(151643),
            "<|eot_id|>" => Some(9),
            _ => None,
        };
        let gen_cfg = r#"{"eos_token_id": [151645, 151643]}"#;
        let tok_cfg = r#"{"eos_token": "<|im_end|>", "chat_template": "{{ '<|im_start|>' }}{{ m }}<|im_end|>"}"#;
        assert_eq!(stop_ids(Some(gen_cfg), Some(tok_cfg), &vocab), vec![151643, 151645]);
        let only_tok =
            r#"{"eos_token": {"content": "<|endoftext|>"}, "chat_template": "<|eot_id|>"}"#;
        assert_eq!(stop_ids(None, Some(only_tok), &vocab), vec![9, 151643]);
        assert_eq!(stop_ids(Some(r#"{"eos_token_id": 2}"#), None, &vocab), vec![2]);
    }

    #[test]
    fn a_missing_checkpoint_is_model_not_found() {
        let e = anyhow::anyhow!("no model.safetensors for x/y under /tmp");
        assert_eq!(failure_kind(&e), FailureKind::ModelNotFound);
    }

    #[test]
    fn a_spec_path_that_does_not_exist_fails_to_load_without_a_gpu() {
        let e = Engine::load("/nonexistent/model.toml", None).err().unwrap();
        assert_eq!(failure_kind(&e), FailureKind::ModelNotFound, "{e:#}");
    }

    #[test]
    fn spec_files_resolve_by_path() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/specs/qwen3-0.6b.toml");
        assert_eq!(resolve_spec(path).unwrap().repo, "Qwen/Qwen3-0.6B");
    }
}

pub struct MediaPrompt {
    pub ids: Vec<u32>,
    pub media: Vec<crate::sched::MediaInput>,
    pub vision_tokens: usize,
}

pub fn encode_media_prompt(model: &str, messages: &[Value], template: Option<&Value>, items: &[crate::vision::MediaItem]) -> Result<MediaPrompt> {
    let assets = Assets::load(model)?;
    let (ids, media, vision_tokens) = assets.encode_media_messages(messages, template, items, Default::default())?;
    Ok(MediaPrompt { ids, media, vision_tokens })
}

pub struct MediaPromptAssets {
    tokenizer: Tokenizer,
    media: crate::vision::MediaAssets,
}

impl MediaPromptAssets {
    pub fn encode(&self, templated: &str, items: &[crate::vision::MediaItem]) -> Result<(Vec<u32>, Vec<crate::sched::MediaInput>, usize)> {
        let processed = self.media.process(items, Default::default())?;
        let ids = self.tokenizer.encode(self.media.expand(templated, &processed)?, false).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?.get_ids().to_vec();
        let vision = ids.iter().filter(|&&t| t == self.media.ids.image || t == self.media.ids.video).count();
        let inputs = processed.into_iter().zip(items).map(|(processed, item)| crate::sched::MediaInput { processed, hash: item.hash }).collect();
        Ok((ids, inputs, vision))
    }
}

pub fn media_prompt_assets(spec: &Spec) -> Result<MediaPromptAssets> {
    let tokenizer = without_truncation(Tokenizer::from_file(find_in_snapshot(&spec.repo, "tokenizer.json")?).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?)?;
    let media = media_assets(spec, &tokenizer)?.context("the model has no vision_config or preprocessor configs")?;
    Ok(MediaPromptAssets { tokenizer, media })
}
