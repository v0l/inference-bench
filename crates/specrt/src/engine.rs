use crate::chat::ChatTemplate;
use crate::runtime::{Runtime, Sampling};
use crate::spec::Spec;
use crate::weights::find_in_snapshot;
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::path::Path;
use tokenizers::Tokenizer;

pub const DEFAULT_CTX: usize = 2048;
pub const MAX_CTX: usize = 131072;

pub fn resolve_spec(model: &str) -> Result<Spec> {
    if model.ends_with(".toml") {
        return Spec::load(Path::new(model));
    }
    let config = find_in_snapshot(model, "config.json")?;
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(&config)?)
        .with_context(|| model.to_string())?;
    spec.repo = model.to_string();
    spec.name = model.rsplit('/').next().unwrap_or(model).to_lowercase();
    Ok(spec)
}

const END_OF_TURN: [&str; 6] =
    ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<end_of_turn>", "<|endoftext|>", "</s>"];

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

pub struct Engine {
    spec: Spec,
    tokenizer: Tokenizer,
    chat: Option<ChatTemplate>,
    runtime: Runtime,
    max_ctx: usize,
    stop: Vec<u32>,
}

impl Engine {
    pub fn load(model: &str, max_ctx: Option<usize>) -> Result<Self> {
        let spec = resolve_spec(model)?;
        let tokenizer = Tokenizer::from_file(find_in_snapshot(&spec.repo, "tokenizer.json")?)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let chat = match find_in_snapshot(&spec.repo, "tokenizer_config.json") {
            Ok(path) => ChatTemplate::from_tokenizer_config(&std::fs::read_to_string(path)?)?,
            Err(_) => None,
        };
        let read = |f: &str| {
            find_in_snapshot(&spec.repo, f).ok().and_then(|p| std::fs::read_to_string(p).ok())
        };
        let stop = stop_ids(
            read("generation_config.json").as_deref(),
            read("tokenizer_config.json").as_deref(),
            &|t| tokenizer.token_to_id(t),
        );
        let max_ctx = max_ctx.unwrap_or(DEFAULT_CTX).min(MAX_CTX);
        tracing::info!(model, spec = %spec.name, max_ctx, "loading model");
        let runtime = Runtime::load(&spec, max_ctx)?;
        Ok(Self { spec, tokenizer, chat, runtime, max_ctx, stop })
    }

    pub fn spec(&self) -> &Spec {
        &self.spec
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        let enc =
            self.tokenizer.encode(text, false).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn encode_messages(&self, messages: &[Value], template: bool) -> Result<Vec<u32>> {
        let text = match (&self.chat, template) {
            (Some(t), true) => t.render(messages)?,
            _ => messages
                .iter()
                .filter_map(|m| m["content"].as_str())
                .collect::<Vec<_>>()
                .join("\n\n"),
        };
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
        template: bool,
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
    fn unsupported_model_types_are_reported_as_unsupported_architecture() {
        let e = anyhow::anyhow!("model_type gemma3 is not supported");
        assert_eq!(failure_kind(&e), FailureKind::UnsupportedArchitecture);
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
