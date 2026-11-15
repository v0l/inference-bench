use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const PROVIDER: &str = "inference-bench";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum App {
    ClaudeCode,
    Codex,
    Opencode,
    Pi,
    Agentbench,
}

impl App {
    pub const ALL: [App; 5] = [App::ClaudeCode, App::Codex, App::Opencode, App::Pi, App::Agentbench];

    pub fn parse(name: &str) -> Option<App> {
        match name.to_ascii_lowercase().replace(['_', ' '], "-").as_str() {
            "claude-code" | "claude" => Some(App::ClaudeCode),
            "codex" | "codex-cli" => Some(App::Codex),
            "opencode" => Some(App::Opencode),
            "pi" => Some(App::Pi),
            "agentbench" => Some(App::Agentbench),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            App::ClaudeCode => "claude-code",
            App::Codex => "codex",
            App::Opencode => "opencode",
            App::Pi => "pi",
            App::Agentbench => "agentbench",
        }
    }

    pub fn default_path(self, home: &Path) -> Option<PathBuf> {
        match self {
            App::ClaudeCode => Some(home.join(".claude/settings.json")),
            App::Codex => Some(home.join(".codex/config.toml")),
            App::Opencode => Some(home.join(".config/opencode/opencode.json")),
            App::Pi => Some(home.join(".pi/agent/models.json")),
            App::Agentbench => None,
        }
    }

    pub fn protocol(self) -> &'static str {
        match self {
            App::ClaudeCode => "Anthropic Messages (/v1/messages)",
            App::Codex => "OpenAI Responses (/v1/responses)",
            _ => "OpenAI chat completions (/v1/chat/completions)",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Endpoint {
    pub url: String,
    pub key: Option<String>,
    pub model: String,
}

impl Endpoint {
    pub fn root(&self) -> &str {
        self.url.trim_end_matches('/').trim_end_matches("/v1")
    }

    pub fn v1(&self) -> String {
        format!("{}/v1", self.root())
    }
}

pub fn render(app: App, e: &Endpoint) -> String {
    match app {
        App::ClaudeCode => serde_json::to_string_pretty(&claude_patch(e)).unwrap(),
        App::Codex => toml::to_string_pretty(&codex_patch(e)).unwrap(),
        App::Opencode => serde_json::to_string_pretty(&opencode_patch(e)).unwrap(),
        App::Pi => serde_json::to_string_pretty(&pi_patch(e)).unwrap(),
        App::Agentbench => format!(
            "Add an OpenAI compatible endpoint in the agentbench provider setup:\n  base URL  {}\n  API key   {}\n  model     {}\n",
            e.v1(),
            e.key.as_deref().unwrap_or("(none needed)"),
            e.model
        ),
    }
}

fn claude_patch(e: &Endpoint) -> Value {
    let mut env = json!({
        "ANTHROPIC_BASE_URL": e.root(),
        "ANTHROPIC_MODEL": e.model,
        "ANTHROPIC_SMALL_FAST_MODEL": e.model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": e.model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": e.model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": e.model,
    });
    env["ANTHROPIC_AUTH_TOKEN"] = json!(e.key.clone().unwrap_or_else(|| "sk-not-needed".into()));
    json!({"env": env})
}

fn codex_patch(e: &Endpoint) -> toml::Table {
    let mut provider = toml::Table::new();
    provider.insert("name".into(), "Inference Bench".into());
    provider.insert("base_url".into(), e.v1().into());
    provider.insert("wire_api".into(), "responses".into());
    if e.key.is_some() {
        provider.insert("experimental_bearer_token".into(), e.key.clone().unwrap().into());
    }
    let mut providers = toml::Table::new();
    providers.insert(PROVIDER.into(), provider.into());
    let mut root = toml::Table::new();
    root.insert("model".into(), e.model.clone().into());
    root.insert("model_provider".into(), PROVIDER.into());
    root.insert("model_providers".into(), providers.into());
    root
}

fn opencode_patch(e: &Endpoint) -> Value {
    json!({
        "$schema": "https://opencode.ai/config.json",
        "model": format!("{PROVIDER}/{}", e.model),
        "provider": {
            PROVIDER: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Inference Bench",
                "options": {"baseURL": e.v1(), "apiKey": e.key.clone().unwrap_or_else(|| "sk-not-needed".into())},
                "models": {e.model.clone(): {"name": e.model}},
            }
        }
    })
}

fn pi_patch(e: &Endpoint) -> Value {
    json!({
        "providers": {
            PROVIDER: {
                "name": "Inference Bench",
                "api": "openai-completions",
                "apiKey": e.key.clone().unwrap_or_else(|| "sk-not-needed".into()),
                "baseUrl": e.v1(),
                "models": [{"id": e.model}],
            }
        }
    })
}

fn merge_json(into: &mut Value, patch: &Value) {
    match (into, patch) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in b {
                match a.get_mut(k) {
                    Some(existing) if existing.is_object() && v.is_object() => merge_json(existing, v),
                    _ => {
                        a.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (slot, patch) => *slot = patch.clone(),
    }
}

fn merge_toml(into: &mut toml::Table, patch: &toml::Table) {
    for (k, v) in patch {
        match (into.get_mut(k), v) {
            (Some(toml::Value::Table(a)), toml::Value::Table(b)) => merge_toml(a, b),
            _ => {
                into.insert(k.clone(), v.clone());
            }
        }
    }
}

pub fn write(app: App, e: &Endpoint, path: &Path) -> Result<()> {
    if app == App::Agentbench {
        bail!("agentbench keeps its providers in its own database: use the printed values in its provider setup");
    }
    let existing = std::fs::read_to_string(path).ok();
    let text = match app {
        App::Codex => {
            let mut table: toml::Table = match &existing {
                Some(t) if !t.trim().is_empty() => toml::from_str(t).with_context(|| format!("{} is not valid TOML", path.display()))?,
                _ => toml::Table::new(),
            };
            merge_toml(&mut table, &codex_patch(e));
            toml::to_string_pretty(&table)?
        }
        _ => {
            let mut value: Value = match &existing {
                Some(t) if !t.trim().is_empty() => serde_json::from_str(t).with_context(|| format!("{} is not valid JSON", path.display()))?,
                _ => json!({}),
            };
            let patch = match app {
                App::ClaudeCode => claude_patch(e),
                App::Opencode => opencode_patch(e),
                _ => pi_patch(e),
            };
            merge_json(&mut value, &patch);
            serde_json::to_string_pretty(&value)? + "\n"
        }
    };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    if existing.is_some() {
        let backup = path.with_extension(format!("{}.bak", path.extension().and_then(|e| e.to_str()).unwrap_or("conf")));
        let _ = std::fs::copy(path, backup);
    }
    std::fs::write(path, text).with_context(|| path.display().to_string())
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Step {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CheckReport {
    pub app: App,
    pub protocol: String,
    pub passed: bool,
    pub steps: Vec<Step>,
}

impl CheckReport {
    pub fn describe(&self) -> String {
        let mut out = format!("{} over {}\n", self.app.name(), self.protocol);
        for s in &self.steps {
            out.push_str(&format!("  {}  {}  {}\n", if s.passed { "pass" } else { "FAIL" }, s.name, s.detail));
        }
        out.push_str(if self.passed { "result: pass\n" } else { "result: fail\n" });
        out
    }
}

const WEATHER: &str = "get_weather";

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(900)))
        .build()
        .into()
}

fn post(e: &Endpoint, app: App, path: &str, body: &Value) -> Result<Value, String> {
    let url = format!("{}{path}", e.root());
    let mut req = agent().post(&url);
    if let Some(k) = &e.key {
        req = match app {
            App::ClaudeCode => req.header("x-api-key", k),
            _ => req.header("Authorization", &format!("Bearer {k}")),
        };
    }
    if app == App::ClaudeCode {
        req = req.header("anthropic-version", "2023-06-01");
    }
    let mut resp = req.send_json(body).map_err(|e| format!("{url}: {e}"))?;
    let status = resp.status().as_u16();
    let text = resp.body_mut().read_to_string().map_err(|e| format!("{url}: {e}"))?;
    if status >= 400 {
        return Err(format!("{url}: HTTP {status}: {}", text.chars().take(300).collect::<String>()));
    }
    serde_json::from_str(&text).map_err(|e| format!("{url}: bad JSON: {e}"))
}

fn prompt_request(app: App, model: &str) -> (&'static str, Value) {
    let ask = "Reply with one short greeting.";
    match app {
        App::ClaudeCode => ("/v1/messages", json!({"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": ask}]})),
        App::Codex => ("/v1/responses", json!({"model": model, "max_output_tokens": 64, "input": ask})),
        _ => ("/v1/chat/completions", json!({"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": ask}]})),
    }
}

fn tool_request(app: App, model: &str) -> (&'static str, Value) {
    let ask = "What is the weather in Paris? Use the get_weather tool.";
    let schema = json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]});
    match app {
        App::ClaudeCode => (
            "/v1/messages",
            json!({"model": model, "max_tokens": 256, "messages": [{"role": "user", "content": ask}],
                "tools": [{"name": WEATHER, "description": "Current weather for a city", "input_schema": schema}]}),
        ),
        App::Codex => (
            "/v1/responses",
            json!({"model": model, "max_output_tokens": 256, "input": ask,
                "tools": [{"type": "function", "name": WEATHER, "description": "Current weather for a city", "parameters": schema}]}),
        ),
        _ => (
            "/v1/chat/completions",
            json!({"model": model, "max_tokens": 256, "messages": [{"role": "user", "content": ask}],
                "tools": [{"type": "function", "function": {"name": WEATHER, "description": "Current weather for a city", "parameters": schema}}]}),
        ),
    }
}

fn reply_text(app: App, v: &Value) -> String {
    match app {
        App::ClaudeCode => v["content"].as_array().map(|a| a.iter().filter_map(|c| c["text"].as_str()).collect::<String>()).unwrap_or_default(),
        App::Codex => v["output"]
            .as_array()
            .map(|a| a.iter().filter(|i| i["type"] == "message").flat_map(|i| i["content"].as_array().cloned().unwrap_or_default()).filter_map(|c| c["text"].as_str().map(String::from)).collect::<String>())
            .unwrap_or_default(),
        _ => v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string(),
    }
}

fn tool_call(app: App, v: &Value) -> Option<(String, Value)> {
    match app {
        App::ClaudeCode => v["content"].as_array()?.iter().find(|c| c["type"] == "tool_use").map(|c| (c["name"].as_str().unwrap_or("").to_string(), c["input"].clone())),
        App::Codex => v["output"].as_array()?.iter().find(|i| i["type"] == "function_call").map(|i| {
            let args = i["arguments"].as_str().and_then(|a| serde_json::from_str(a).ok()).unwrap_or(Value::Null);
            (i["name"].as_str().unwrap_or("").to_string(), args)
        }),
        _ => {
            let call = v["choices"][0]["message"]["tool_calls"].as_array()?.first()?.clone();
            let args = call["function"]["arguments"].as_str().and_then(|a| serde_json::from_str(a).ok()).unwrap_or(Value::Null);
            Some((call["function"]["name"].as_str().unwrap_or("").to_string(), args))
        }
    }
}

pub fn check(app: App, e: &Endpoint) -> CheckReport {
    let mut steps = Vec::new();
    let (path, body) = prompt_request(app, &e.model);
    steps.push(match post(e, app, path, &body) {
        Ok(v) => {
            let text = reply_text(app, &v);
            Step { name: "one prompt".into(), passed: !text.trim().is_empty(), detail: match text.trim().is_empty() {
                true => "the reply had no text".into(),
                false => format!("replied: {}", text.trim().chars().take(60).collect::<String>()),
            } }
        }
        Err(e) => Step { name: "one prompt".into(), passed: false, detail: e },
    });
    let (path, body) = tool_request(app, &e.model);
    steps.push(match post(e, app, path, &body) {
        Ok(v) => match tool_call(app, &v) {
            Some((name, args)) if name == WEATHER && args.is_object() => {
                Step { name: "one tool call".into(), passed: true, detail: format!("called {name} with {args}") }
            }
            Some((name, _)) => Step { name: "one tool call".into(), passed: false, detail: format!("called {name:?} instead of {WEATHER} with usable arguments") },
            None => Step {
                name: "one tool call".into(),
                passed: false,
                detail: format!("no tool call came back: {}", reply_text(app, &v).trim().chars().take(80).collect::<String>()),
            },
        },
        Err(e) => Step { name: "one tool call".into(), passed: false, detail: e },
    });
    CheckReport { app, protocol: app.protocol().into(), passed: steps.iter().all(|s| s.passed), steps }
}

pub fn first_model(url: &str, key: Option<&str>) -> Result<String> {
    let e = Endpoint { url: url.to_string(), key: key.map(String::from), model: String::new() };
    let mut req = agent().get(format!("{}/v1/models", e.root()));
    if let Some(k) = key {
        req = req.header("Authorization", &format!("Bearer {k}"));
    }
    let mut resp = req.call().with_context(|| format!("{} is not answering", e.root()))?;
    anyhow::ensure!(resp.status().as_u16() < 400, "{}/v1/models answered HTTP {}", e.root(), resp.status());
    let v: Value = resp.body_mut().read_json()?;
    v["data"][0]["id"].as_str().map(String::from).context("the server lists no models")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint() -> Endpoint {
        Endpoint { url: "http://127.0.0.1:8080/v1/".into(), key: Some("sk-x".into()), model: "qwen".into() }
    }

    #[test]
    fn urls_are_normalised_with_and_without_v1() {
        assert_eq!(endpoint().root(), "http://127.0.0.1:8080");
        assert_eq!(endpoint().v1(), "http://127.0.0.1:8080/v1");
    }

    #[test]
    fn every_app_prints_its_endpoint_key_and_model() {
        for app in App::ALL {
            let text = render(app, &endpoint());
            assert!(text.contains("qwen") && text.contains("8080"), "{app:?}: {text}");
            assert!(text.contains("sk-x"), "{app:?}: {text}");
        }
        assert!(render(App::ClaudeCode, &endpoint()).contains("\"ANTHROPIC_BASE_URL\": \"http://127.0.0.1:8080\""));
        assert!(render(App::Codex, &endpoint()).contains("wire_api = \"responses\""));
        assert!(render(App::Opencode, &endpoint()).contains("inference-bench/qwen"));
    }

    #[test]
    fn writing_merges_into_an_existing_file_and_keeps_a_backup() {
        let dir = std::env::temp_dir().join(format!("ib-connect-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        std::fs::write(&path, r#"{"theme": "dark", "env": {"KEEP": "1"}}"#).unwrap();
        write(App::ClaudeCode, &endpoint(), &path).unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["theme"], "dark");
        assert_eq!(v["env"]["KEEP"], "1");
        assert_eq!(v["env"]["ANTHROPIC_MODEL"], "qwen");
        assert!(dir.join("settings.json.bak").exists());
        let codex = dir.join("config.toml");
        std::fs::write(&codex, "approval_policy = \"never\"\n[model_providers.other]\nname = \"x\"\n").unwrap();
        write(App::Codex, &endpoint(), &codex).unwrap();
        let t: toml::Table = toml::from_str(&std::fs::read_to_string(&codex).unwrap()).unwrap();
        assert_eq!(t["approval_policy"].as_str(), Some("never"));
        assert_eq!(t["model_provider"].as_str(), Some(PROVIDER));
        assert!(t["model_providers"]["other"].is_table());
        assert!(write(App::Agentbench, &endpoint(), &path).is_err());
    }
}
