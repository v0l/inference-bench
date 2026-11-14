use crate::chat::{Message, Sampling, Stats};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    #[serde(flatten)]
    pub message: Message,
    #[serde(default)]
    pub stats: Option<Stats>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<crate::media::Attachment>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub system: String,
    pub model: String,
    pub sampling: Sampling,
    pub use_template: bool,
    #[serde(default = "thinking_on")]
    pub thinking: bool,
    pub turns: Vec<Turn>,
    pub updated: chrono::DateTime<chrono::Utc>,
}

fn thinking_on() -> bool {
    true
}

impl Conversation {
    pub fn new() -> Self {
        let now = chrono::Utc::now();
        Self {
            id: now.format("%Y%m%dT%H%M%S%3f").to_string(),
            title: "New chat".into(),
            system: String::new(),
            model: String::new(),
            sampling: Sampling::default(),
            use_template: true,
            thinking: true,
            turns: Vec::new(),
            updated: now,
        }
    }

    pub fn retitle(&mut self) {
        if let Some(first) = self.turns.iter().find(|t| t.message.role == crate::chat::Role::User) {
            let line = first.message.content.lines().next().unwrap_or("").trim();
            let title: String = line.chars().take(40).collect();
            if !title.is_empty() {
                self.title = title;
            }
        }
    }

    pub fn openai_messages(&self) -> Option<Vec<serde_json::Value>> {
        if self.turns.iter().all(|t| t.attachments.is_empty()) {
            return None;
        }
        let mut out = Vec::new();
        if !self.system.trim().is_empty() {
            out.push(serde_json::json!({"role": "system", "content": self.system.trim()}));
        }
        for t in &self.turns {
            let text = match t.message.role {
                crate::chat::Role::Assistant => crate::reasoning::answer_of(&t.message.content),
                _ => t.message.content.clone(),
            };
            out.push(serde_json::json!({
                "role": t.message.role.name(),
                "content": crate::media::content_with_attachments(&text, &t.attachments),
            }));
        }
        Some(out)
    }

    pub fn messages(&self) -> Vec<Message> {
        let mut out = Vec::new();
        if !self.system.trim().is_empty() {
            out.push(Message::new(crate::chat::Role::System, self.system.trim()));
        }
        out.extend(self.turns.iter().map(|t| match t.message.role {
            crate::chat::Role::Assistant => Message::new(t.message.role, crate::reasoning::answer_of(&t.message.content)),
            _ => t.message.clone(),
        }));
        out
    }

    pub fn chat_request(&self, thinking: specrt::chat::Thinking) -> crate::chat::ChatRequest {
        let on = self.thinking && thinking != specrt::chat::Thinking::Unsupported;
        let mut request = crate::chat::ChatRequest::new(self.messages(), self.sampling).with_thinking(on);
        request.use_template = self.use_template;
        request.openai_messages = self.openai_messages();
        request
    }

    pub fn token_estimate(&self) -> usize {
        self.messages().iter().map(|m| crate::chat::estimate_tokens(&m.content) + 4).sum()
    }
}

impl Default for Conversation {
    fn default() -> Self {
        Self::new()
    }
}

fn dir_of(data: &Path) -> PathBuf {
    data.join("chats")
}

pub fn save(data: &Path, c: &Conversation) -> std::io::Result<()> {
    let dir = dir_of(data);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(format!("{}.json", c.id)), serde_json::to_string_pretty(c)?)
}

pub fn remove(data: &Path, id: &str) {
    let _ = std::fs::remove_file(dir_of(data).join(format!("{id}.json")));
}

pub fn load_all(data: &Path) -> Vec<Conversation> {
    let Ok(entries) = std::fs::read_dir(dir_of(data)) else { return Vec::new() };
    let mut all: Vec<Conversation> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| serde_json::from_str(&std::fs::read_to_string(e.path()).ok()?).ok())
        .collect();
    all.sort_by_key(|c| std::cmp::Reverse(c.updated));
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::Role;

    #[test]
    fn conversations_survive_a_round_trip_newest_first() {
        let dir = std::env::temp_dir().join(format!("ib-chats-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut a = Conversation::new();
        a.id = "a".into();
        a.turns.push(Turn { message: Message::new(Role::User, "hello there"), stats: Some(Stats { completion_tokens: 3, ..Default::default() }), attachments: Vec::new() });
        a.retitle();
        let mut b = Conversation::new();
        b.id = "b".into();
        b.updated = a.updated + chrono::Duration::seconds(5);
        save(&dir, &a).unwrap();
        save(&dir, &b).unwrap();
        let all = load_all(&dir);
        assert_eq!(all.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(), vec!["b", "a"]);
        assert_eq!(all[1], a);
        assert_eq!(all[1].title, "hello there");
        remove(&dir, "a");
        assert_eq!(load_all(&dir).len(), 1);
    }

    #[test]
    fn a_conversation_saved_with_a_backend_still_loads() {
        let dir = std::env::temp_dir().join(format!("ib-chats-old-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut c = serde_json::to_value(Conversation::new()).unwrap();
        c["id"] = "old".into();
        c["backend"] = "vllm".into();
        std::fs::create_dir_all(dir_of(&dir)).unwrap();
        std::fs::write(dir_of(&dir).join("old.json"), c.to_string()).unwrap();
        assert_eq!(load_all(&dir)[0].id, "old");
    }

    #[test]
    fn a_turn_with_an_attachment_becomes_openai_content_parts_and_round_trips() {
        let dir = std::env::temp_dir().join(format!("ib-chats-att-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let png = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        let mut c = Conversation::new();
        c.id = "att".into();
        assert!(c.openai_messages().is_none());
        let att = crate::media::Attachment::from_bytes("a.png", &png).unwrap();
        c.turns.push(Turn { message: Message::new(Role::User, "what is this"), stats: None, attachments: vec![att.clone()] });
        let list = c.openai_messages().unwrap();
        assert_eq!(list[0]["content"][0]["type"], "image_url");
        assert_eq!(list[0]["content"][1]["text"], "what is this");
        save(&dir, &c).unwrap();
        assert_eq!(load_all(&dir)[0].turns[0].attachments, vec![att]);
    }

    #[test]
    fn the_system_prompt_leads_the_messages() {
        let mut c = Conversation::new();
        c.system = " be brief ".into();
        c.turns.push(Turn { message: Message::new(Role::User, "hi"), stats: None, attachments: Vec::new() });
        let m = c.messages();
        assert_eq!((m[0].role, m[0].content.as_str()), (Role::System, "be brief"));
        assert_eq!(m.len(), 2);
        assert!(c.token_estimate() > 8);
    }
}
