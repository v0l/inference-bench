use anyhow::{Context, Result, anyhow};
use minijinja::{Environment, context};
use serde_json::Value;

pub struct ChatTemplate {
    source: String,
    bos_token: String,
    eos_token: String,
}

fn token_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Object(o) => o.get("content").and_then(Value::as_str).unwrap_or("").to_string(),
        _ => String::new(),
    }
}

impl ChatTemplate {
    pub fn from_tokenizer_config(json: &str) -> Result<Option<Self>> {
        Self::from_files(json, None)
    }

    pub fn from_files(json: &str, jinja: Option<&str>) -> Result<Option<Self>> {
        let config: Value = serde_json::from_str(json).context("tokenizer_config.json")?;
        let source = match (&config["chat_template"], jinja) {
            (_, Some(s)) => s.to_string(),
            (Value::String(s), None) => s.clone(),
            (Value::Array(list), None) => {
                let pick = list
                    .iter()
                    .find(|t| t["name"] == "default")
                    .or_else(|| list.first())
                    .and_then(|t| t["template"].as_str());
                match pick {
                    Some(s) => s.to_string(),
                    None => return Ok(None),
                }
            }
            _ => return Ok(None),
        };
        Ok(Some(ChatTemplate {
            source,
            bos_token: token_text(&config["bos_token"]),
            eos_token: token_text(&config["eos_token"]),
        }))
    }

    fn iterates_content(&self) -> bool {
        ["content is iterable", "in message.content", "in message['content']"].iter().any(|p| self.source.contains(p))
    }

    pub fn render_user(&self, prompt: &str) -> Result<String> {
        self.render(&[serde_json::json!({"role": "user", "content": prompt})])
    }

    pub fn render(&self, messages: &[Value]) -> Result<String> {
        self.render_with(messages, &serde_json::json!({"enable_thinking": false}))
    }

    pub fn render_with(&self, messages: &[Value], vars: &Value) -> Result<String> {
        let (messages, vars) = prepare(messages, vars, self.iterates_content());
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        });
        env.add_filter("tojson", |v: minijinja::Value| -> Result<minijinja::Value, minijinja::Error> {
            let json = serde_json::to_value(&v)
                .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string()))?;
            Ok(minijinja::Value::from_safe_string(py_json(&json)))
        });
        env.add_template("chat", &self.source).map_err(|e| anyhow!("chat template: {e}"))?;
        let base = context! {
            messages => messages,
            add_generation_prompt => true,
            bos_token => self.bos_token,
            eos_token => self.eos_token,
        };
        let ctx = context! { ..minijinja::Value::from_serialize(vars), ..base };
        env.get_template("chat").and_then(|t| t.render(ctx)).map_err(|e| anyhow!("chat template: {e}"))
    }
}

fn text_parts(content: &Value) -> Option<Vec<String>> {
    let parts = content.as_array()?;
    parts.iter().all(|p| p["type"] == "text").then(|| parts.iter().map(|p| p["text"].as_str().unwrap_or("").to_string()).collect())
}

fn prepare(messages: &[Value], vars: &Value, iterates_content: bool) -> (Vec<Value>, Value) {
    let messages = messages
        .iter()
        .map(|m| {
            let mut m = m.clone();
            let role = m["role"].as_str().unwrap_or("").to_string();
            if let Some(r) = m.get("reasoning_content").cloned().filter(|r| !r.is_null())
                && m.get("reasoning").is_none_or(Value::is_null)
            {
                m["reasoning"] = r;
            }
            if role == "assistant" && m.get("reasoning").is_some_and(|r| !r.is_null()) {
                m["reasoning_content"] = m["reasoning"].clone();
            } else if let Some(o) = m.as_object_mut() {
                o.remove("reasoning_content");
            }
            match (&m["content"], text_parts(&m["content"])) {
                (Value::Null, _) if !iterates_content => m["content"] = Value::String(String::new()),
                (_, Some(texts)) if role == "tool" => m["content"] = Value::String(texts.join("\n")),
                (_, Some(texts)) if !iterates_content => m["content"] = Value::String(texts.join("\n")),
                _ => {}
            }
            if let Some(calls) = m.get_mut("tool_calls").and_then(Value::as_array_mut) {
                for call in calls.iter_mut() {
                    let f = &mut call["function"];
                    let args = match &f["arguments"] {
                        Value::String(s) if !s.is_empty() => serde_json::from_str(s).unwrap_or(Value::Null),
                        Value::Object(_) | Value::Array(_) => f["arguments"].clone(),
                        _ => Value::Null,
                    };
                    f["arguments"] = if args.is_null() { serde_json::json!({}) } else { args };
                }
            }
            if m.get("tool_calls").and_then(Value::as_array).is_some_and(|c| c.is_empty()) {
                m.as_object_mut().unwrap().remove("tool_calls");
            }
            m
        })
        .collect();
    let mut vars = vars.clone();
    if let Some(tools) = vars.get_mut("tools").and_then(Value::as_array_mut) {
        for t in tools.iter_mut() {
            let f = t["function"].clone();
            let mut function = serde_json::Map::new();
            function.insert("name".into(), f["name"].clone());
            function.insert("description".into(), f.get("description").cloned().unwrap_or(Value::Null));
            function.insert("parameters".into(), f.get("parameters").cloned().unwrap_or(Value::Null));
            for key in ["strict", "defer_loading"] {
                if let Some(v) = f.get(key).filter(|v| !v.is_null()) {
                    function.insert(key.into(), v.clone());
                }
            }
            *t = serde_json::json!({"type": "function", "function": function});
        }
    }
    (messages, vars)
}

pub fn py_json(v: &Value) -> String {
    let mut out = String::new();
    write_py_json(v, &mut out);
    out
}

fn write_py_json(v: &Value, out: &mut String) {
    match v {
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(k).unwrap());
                out.push_str(": ");
                write_py_json(item, out);
            }
            out.push('}');
        }
        other => out.push_str(&serde_json::to_string(other).unwrap()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHATML: &str = "{% for m in messages %}<|im_start|>{{ m['role'] }}\\n{{ m['content'] }}<|im_end|>\\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\\n{% endif %}";

    fn config(template: &str) -> String {
        format!(
            r#"{{"chat_template": "{template}", "bos_token": null, "eos_token": "<|im_end|>"}}"#
        )
    }

    #[test]
    fn a_user_prompt_is_wrapped_in_the_chatml_turns() {
        let t = ChatTemplate::from_tokenizer_config(&config(CHATML)).unwrap().unwrap();
        assert_eq!(
            t.render_user("hi").unwrap(),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn a_config_without_a_template_gives_none() {
        assert!(ChatTemplate::from_tokenizer_config("{}").unwrap().is_none());
    }

    #[test]
    fn a_template_list_prefers_the_default_entry() {
        let json = r#"{"chat_template": [{"name": "tool_use", "template": "T"}, {"name": "default", "template": "D{{ eos_token }}"}], "eos_token": {"content": "</s>"}}"#;
        let t = ChatTemplate::from_tokenizer_config(json).unwrap().unwrap();
        assert_eq!(t.render_user("x").unwrap(), "D</s>");
    }

    #[test]
    fn a_broken_template_is_an_error() {
        let t = ChatTemplate::from_tokenizer_config(&config("{% for %}")).unwrap().unwrap();
        assert!(t.render_user("x").is_err());
    }
}
