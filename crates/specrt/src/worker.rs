use crate::chat::ChatTemplate;
use crate::runtime::{Runtime, Sampling};
use crate::spec::Spec;
use crate::weights::find_in_snapshot;
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::io::{BufRead, Write};
use std::path::Path;
use tokenizers::Tokenizer;

const DEFAULT_CTX: usize = 2048;
const MAX_CTX: usize = 131072;

pub fn resolve_spec(model: &str) -> Result<Spec> {
    if model.ends_with(".toml") {
        return Spec::load(Path::new(model));
    }
    let config = find_in_snapshot(model, "config.json")?;
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(&config)?)
        .with_context(|| format!("{model}"))?;
    spec.repo = model.to_string();
    spec.name = model.rsplit('/').next().unwrap_or(model).to_lowercase();
    Ok(spec)
}

struct Loaded {
    spec: Spec,
    tokenizer: Tokenizer,
    chat: Option<ChatTemplate>,
    runtime: Runtime,
    max_ctx: usize,
    stop: Vec<u32>,
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

fn error(err: &anyhow::Error) -> Value {
    let message = format!("{err:#}");
    let lower = message.to_lowercase();
    let kind = if lower.contains("is not supported") || lower.contains("not supported") {
        "unsupported_architecture"
    } else if lower.contains("no config.json") || lower.contains("no model.safetensors") {
        "model_not_found"
    } else {
        "failed"
    };
    json!({"event": "error", "kind": kind, "message": message})
}

fn load(model: &str, max_ctx: Option<usize>) -> Result<Loaded> {
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
    Ok(Loaded { spec, tokenizer, chat, runtime, max_ctx, stop })
}

fn encode_messages(l: &Loaded, messages: &[Value], template: bool) -> Result<Vec<u32>> {
    let text = match (&l.chat, template) {
        (Some(t), true) => t.render(messages)?,
        _ => messages.iter().filter_map(|m| m["content"].as_str()).collect::<Vec<_>>().join("\n\n"),
    };
    let enc =
        l.tokenizer.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

fn encode(l: &Loaded, prompt: &str, chat: bool) -> Result<Vec<u32>> {
    let rendered;
    let prompt = match (&l.chat, chat) {
        (Some(template), true) => {
            rendered = template.render_user(prompt)?;
            rendered.as_str()
        }
        _ => prompt,
    };
    let enc = l.tokenizer.encode(prompt, false).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

fn generate(
    l: &mut Loaded,
    prompt: &str,
    max_tokens: usize,
    chat: bool,
    sampling: Sampling,
) -> Result<Value> {
    let ids = encode(l, prompt, chat)?;
    generate_ids(l, ids, max_tokens, sampling).map(|(_, done)| done)
}

fn generate_ids(
    l: &mut Loaded,
    ids: Vec<u32>,
    max_tokens: usize,
    sampling: Sampling,
) -> Result<(String, Value)> {
    if ids.len() + max_tokens > l.max_ctx {
        bail!(
            "prompt of {} tokens plus {max_tokens} new tokens exceeds the context of {}",
            ids.len(),
            l.max_ctx
        );
    }
    generate_streaming(l, ids, max_tokens, sampling, false, &mut |_| {})
}

fn generate_streaming(
    l: &mut Loaded,
    ids: Vec<u32>,
    max_tokens: usize,
    sampling: Sampling,
    stop_at_end_of_turn: bool,
    on_text: &mut dyn FnMut(&str),
) -> Result<(String, Value)> {
    if ids.len() + max_tokens > l.max_ctx {
        bail!(
            "prompt of {} tokens plus {max_tokens} new tokens exceeds the context of {}",
            ids.len(),
            l.max_ctx
        );
    }
    let mut so_far: Vec<u32> = Vec::new();
    let mut sent = String::new();
    let tokenizer = &l.tokenizer;
    let stop: &[u32] = if stop_at_end_of_turn { &l.stop } else { &[] };
    let g = l.runtime.generate_until(&ids, max_tokens, sampling, stop, &mut |fresh| {
        so_far.extend_from_slice(fresh);
        if let Ok(text) = tokenizer.decode(&so_far, true)
            && text.len() > sent.len()
            && text.starts_with(&sent)
            && !text.ends_with('\u{fffd}')
        {
            on_text(&text[sent.len()..]);
            sent = text;
        }
    })?;
    let text = l.tokenizer.decode(&g.tokens, true).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    if text.len() > sent.len() && text.starts_with(&sent) {
        on_text(&text[sent.len()..]);
    }
    let done = json!({"event": "done", "generation": {
        "text": text.clone(),
        "prompt_tokens": ids.len(),
        "completion_tokens": g.tokens.len(),
        "ttft_secs": g.prefill_secs,
        "prefill_secs": g.prefill_secs,
        "decode_secs": g.decode_secs,
        "peak_memory_bytes": g.peak_device_bytes,
    }});
    Ok((text, done))
}

fn chat(l: &mut Loaded, msg: &Value, out: &mut impl Write) -> Result<Value> {
    let messages = msg["messages"].as_array().cloned().unwrap_or_default();
    let ids = encode_messages(l, &messages, msg["template"].as_bool().unwrap_or(true))?;
    let max_tokens = msg["max_tokens"].as_u64().unwrap_or(64) as usize;
    let mut failed = None;
    let (_, done) = generate_streaming(l, ids, max_tokens, sampling_from(msg), true, &mut |t| {
        if let Err(e) =
            writeln!(out, "{}", json!({"event": "text", "text": t})).and_then(|_| out.flush())
        {
            failed = Some(e);
        }
    })?;
    if let Some(e) = failed {
        return Err(e.into());
    }
    Ok(done)
}

fn logits(l: &mut Loaded, prompt: &str, path: &str) -> Result<Value> {
    let ids = encode(l, prompt, false)?;
    let values = l.runtime.prompt_logits(&ids)?;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes)?;
    Ok(json!({"event": "logits", "tokens": ids, "vocab": l.spec.dims.vocab}))
}

fn sampling_from(msg: &Value) -> Sampling {
    Sampling {
        temperature: msg["temperature"].as_f64().unwrap_or(0.0).max(0.0) as f32,
        seed: msg["seed"].as_u64().unwrap_or(0) as u32,
    }
}

pub fn run(input: impl BufRead, mut out: impl Write) -> Result<()> {
    let mut loaded: Option<Loaded> = None;
    for line in input.lines() {
        let msg: Value = serde_json::from_str(&line?)?;
        let reply = match msg["op"].as_str().unwrap_or("") {
            "quit" => break,
            "load" => {
                let ctx = msg["max_ctx"].as_u64().map(|n| n as usize);
                match load(msg["model"].as_str().unwrap_or(""), ctx) {
                    Ok(l) => {
                        loaded = Some(l);
                        json!({"event": "loaded"})
                    }
                    Err(e) => error(&e),
                }
            }
            op => match loaded.as_mut() {
                None => json!({"event": "error", "kind": "failed", "message": "no model loaded"}),
                Some(l) => {
                    let prompt = msg["prompt"].as_str().unwrap_or("");
                    let r = match op {
                        "generate" => {
                            let sampling = sampling_from(&msg);
                            let chat = msg["chat"].as_bool().unwrap_or(true);
                            let max_tokens = msg["max_tokens"].as_u64().unwrap_or(64) as usize;
                            generate(l, prompt, max_tokens, chat, sampling)
                        }
                        "chat" => chat(l, &msg, &mut out),
                        "logits" => logits(l, prompt, msg["path"].as_str().unwrap_or("")),
                        other => Err(anyhow::anyhow!("unknown op {other}")),
                    };
                    r.unwrap_or_else(|e| error(&e))
                }
            },
        };
        writeln!(out, "{reply}")?;
        out.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_model_types_are_reported_as_unsupported_architecture() {
        let e = anyhow::anyhow!("model_type gemma3 is not supported");
        assert_eq!(error(&e)["kind"], "unsupported_architecture");
        assert_eq!(error(&e)["message"], "model_type gemma3 is not supported");
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
    fn sampling_defaults_to_greedy() {
        assert_eq!(sampling_from(&json!({"op": "generate"})), Sampling::GREEDY);
        let s = sampling_from(&json!({"temperature": 0.5, "seed": 7}));
        assert_eq!((s.temperature, s.seed), (0.5, 7));
    }

    #[test]
    fn a_missing_checkpoint_is_model_not_found() {
        let e = anyhow::anyhow!("no model.safetensors for x/y under /tmp");
        assert_eq!(error(&e)["kind"], "model_not_found");
    }

    #[test]
    fn requests_before_load_are_errors() {
        let input = b"{\"op\":\"generate\",\"prompt\":\"x\",\"max_tokens\":1}\n{\"op\":\"quit\"}\n";
        let mut out = Vec::new();
        run(&input[..], &mut out).unwrap();
        let reply: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(reply["event"], "error");
    }

    #[test]
    fn spec_files_resolve_by_path() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/specs/qwen3-0.6b.toml");
        assert_eq!(resolve_spec(path).unwrap().repo, "Qwen/Qwen3-0.6B");
    }
}
