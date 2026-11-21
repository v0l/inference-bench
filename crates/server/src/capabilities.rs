use serde_json::{Value, json};

pub const CHAT: u8 = 1;
pub const COMPLETIONS: u8 = 2;
pub const BENCH: u8 = 4;
const GENERATE: u8 = CHAT | COMPLETIONS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Usage {
    Read,
    Validated,
    ZeroOnly,
    Ignored,
}

impl Usage {
    pub fn label(self) -> &'static str {
        match self {
            Usage::Read => "supported",
            Usage::Validated => "supported",
            Usage::ZeroOnly => "zero_only",
            Usage::Ignored => "ignored",
        }
    }
}

pub struct Field {
    pub name: &'static str,
    pub on: u8,
    pub usage: Usage,
    pub ty: &'static str,
    pub note: &'static str,
}

const fn f(name: &'static str, on: u8, usage: Usage, ty: &'static str, note: &'static str) -> Field {
    Field { name, on, usage, ty, note }
}

pub const FIELDS: &[Field] = &[
    f("model", CHAT | COMPLETIONS | BENCH, Usage::Read, "string", "a configured model id or served name; defaults to the only model"),
    f("messages", CHAT, Usage::Read, "array", "non-empty; text content only, images and video go through content parts when the model has vision"),
    f("prompt", COMPLETIONS, Usage::Read, "string", "one prompt; batched prompts are refused"),
    f("prompts", BENCH, Usage::Read, "array", "non-empty array of strings"),
    f("repeat", BENCH, Usage::Read, "integer", "1 to 100"),
    f("max_tokens", CHAT | COMPLETIONS | BENCH, Usage::Read, "integer", "at least 1"),
    f("max_completion_tokens", GENERATE, Usage::Read, "integer", "wins over max_tokens"),
    f("temperature", GENERATE, Usage::Read, "number", "defaults to 0"),
    f("top_p", GENERATE, Usage::Read, "number", ""),
    f("top_k", GENERATE, Usage::Read, "integer", "extension"),
    f("min_p", GENERATE, Usage::Read, "number", "extension"),
    f("seed", GENERATE, Usage::Read, "integer", ""),
    f("stop", GENERATE, Usage::Read, "string or array of strings", ""),
    f("stream", GENERATE, Usage::Read, "boolean", ""),
    f("stream_options", GENERATE, Usage::Read, "object", "include_usage"),
    f("ignore_eos", GENERATE, Usage::Read, "boolean", "extension"),
    f("tools", CHAT, Usage::Read, "array", "function tools with a name"),
    f("tool_choice", CHAT, Usage::Read, "string", "auto or none; required and named choices are refused"),
    f("chat_template_kwargs", CHAT, Usage::Read, "object", "passed to the chat template"),
    f("reasoning_effort", CHAT, Usage::Read, "string", "none, minimal, low, medium, high, xhigh or max"),
    f("reasoning", CHAT, Usage::Read, "object", "effort"),
    f("reasoning_budget_tokens", GENERATE, Usage::Read, "integer", "caps thinking tokens"),
    f("thinking_budget_tokens", GENERATE, Usage::Read, "integer", "alias of reasoning_budget_tokens"),
    f("cache_salt", GENERATE, Usage::Read, "string", "isolates the prefix cache"),
    f("n", GENERATE, Usage::Read, "integer", "1 to 16; above 1 needs a model loaded with more than one parallel request"),
    f("logprobs", GENERATE, Usage::Read, "boolean or integer", "boolean on chat, integer on completions; needs the batched engine"),
    f("top_logprobs", CHAT, Usage::Read, "integer", "needs logprobs: true"),
    f("response_format", GENERATE, Usage::Validated, "object", "json_object, json_schema; needs the batched engine"),
    f("frequency_penalty", GENERATE, Usage::ZeroOnly, "number", "only 0 or null is accepted"),
    f("presence_penalty", GENERATE, Usage::ZeroOnly, "number", "only 0 or null is accepted"),
    f("user", GENERATE, Usage::Ignored, "string", "accepted and ignored"),
    f("parallel_tool_calls", GENERATE, Usage::Ignored, "boolean", "accepted and ignored"),
    f("store", GENERATE, Usage::Ignored, "boolean", "accepted and ignored"),
    f("metadata", GENERATE, Usage::Ignored, "object", "accepted and ignored"),
];

pub fn lookup(name: &str) -> Option<&'static Field> {
    FIELDS.iter().find(|f| f.name == name)
}

pub fn model_features(model: &str) -> Result<inference_bench_core::doctor::ModelFeatures, String> {
    inference_bench_core::doctor::features(model).map(|(_, f)| f)
}

fn endpoint_of(method: &str, path: &str) -> u8 {
    match (method, path) {
        ("POST", "/v1/chat/completions") => CHAT,
        ("POST", "/v1/completions") => COMPLETIONS,
        ("POST", "/v1/bench") => BENCH,
        _ => 0,
    }
}

pub fn fields_json(on: u8) -> Value {
    Value::Array(
        FIELDS
            .iter()
            .filter(|f| f.on & on != 0)
            .map(|f| json!({"name": f.name, "status": f.usage.label(), "type": f.ty, "note": f.note}))
            .collect(),
    )
}

pub fn routes_with_fields(routes: &[crate::landing::ApiRoute]) -> Value {
    Value::Array(
        routes
            .iter()
            .map(|r| {
                let mut v = json!({"method": r.method, "path": r.path, "purpose": r.purpose});
                let on = endpoint_of(r.method, r.path);
                if on != 0 {
                    v["fields"] = fields_json(on);
                }
                v
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn read_by_handlers() -> BTreeSet<String> {
        let source = include_str!("app.rs");
        let mut names = BTreeSet::new();
        let marker = "body[\"";
        for (at, _) in source.match_indices(marker) {
            let rest = &source[at + marker.len()..];
            if let Some(end) = rest.find('"') {
                names.insert(rest[..end].to_string());
            }
        }
        names
    }

    #[test]
    fn every_field_the_handlers_read_is_in_the_manifest() {
        let missing: Vec<String> = read_by_handlers().into_iter().filter(|n| lookup(n).is_none()).collect();
        assert!(missing.is_empty(), "read by app.rs but missing from capabilities::FIELDS: {missing:?}");
    }

    #[test]
    fn every_supported_field_in_the_manifest_is_read_by_a_handler() {
        let read = read_by_handlers();
        let stale: Vec<&str> = FIELDS.iter().filter(|f| f.usage == Usage::Read && !read.contains(f.name)).map(|f| f.name).collect();
        assert!(stale.is_empty(), "in capabilities::FIELDS as supported but never read: {stale:?}");
    }

    #[test]
    fn field_names_are_unique_and_every_generate_endpoint_has_fields() {
        let names: BTreeSet<&str> = FIELDS.iter().map(|f| f.name).collect();
        assert_eq!(names.len(), FIELDS.len());
        for on in [CHAT, COMPLETIONS, BENCH] {
            assert!(!fields_json(on).as_array().unwrap().is_empty());
        }
    }
}
