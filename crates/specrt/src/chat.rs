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
        let config: Value = serde_json::from_str(json).context("tokenizer_config.json")?;
        let source = match &config["chat_template"] {
            Value::String(s) => s.clone(),
            Value::Array(list) => {
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

    pub fn render_user(&self, prompt: &str) -> Result<String> {
        self.render(&[serde_json::json!({"role": "user", "content": prompt})])
    }

    pub fn render(&self, messages: &[Value]) -> Result<String> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        });
        env.add_template("chat", &self.source).map_err(|e| anyhow!("chat template: {e}"))?;
        env.get_template("chat")
            .and_then(|t| {
                t.render(context! {
                    messages => messages,
                    add_generation_prompt => true,
                    bos_token => self.bos_token,
                    eos_token => self.eos_token,
                    enable_thinking => false,
                })
            })
            .map_err(|e| anyhow!("chat template: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHATML: &str = "{% for m in messages %}<|im_start|>{{ m['role'] }}\\n{{ m['content'] }}<|im_end|>\\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\\n{% endif %}";

    fn config(template: &str) -> String {
        format!(r#"{{"chat_template": "{template}", "bos_token": null, "eos_token": "<|im_end|>"}}"#)
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
