use super::*;

const END: [&str; 3] = ["<|end|>", "<|call|>", "<|return|>"];
const MESSAGE: &str = "<|message|>";

#[derive(Clone, Debug)]
pub(super) enum Harmony {
    Header,
    Body { channel: Channel, recipient: Option<String>, text: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Channel {
    Analysis,
    Final,
    Commentary,
}

fn header(text: &str) -> (Channel, Option<String>) {
    let channel = match text.split("<|channel|>").nth(1).map(|c| c.trim_start()) {
        Some(c) if c.starts_with("analysis") => Channel::Analysis,
        Some(c) if c.starts_with("commentary") => Channel::Commentary,
        _ => Channel::Final,
    };
    let recipient = text
        .split(|c: char| c.is_whitespace() || c == '<')
        .find_map(|w| w.strip_prefix("to="))
        .map(|r| r.trim().to_string())
        .filter(|r| !r.is_empty());
    (channel, recipient)
}

fn safe_len(buffer: &str) -> usize {
    let Some(i) = buffer.rfind('<') else { return buffer.len() };
    let tail = &buffer[i..];
    if END.iter().any(|e| e.starts_with(tail) && tail.len() < e.len()) { i } else { buffer.len() }
}

impl OutputParser {
    pub fn harmony(mut self) -> Self {
        self.harmony = Some(Harmony::Header);
        self.state = State::Content;
        self
    }

    pub(super) fn harmony_feed(&mut self, last: bool, out: &mut Vec<Delta>) {
        loop {
            let Some(mode) = self.harmony.clone() else { return };
            match mode {
                Harmony::Header => {
                    let Some(at) = self.buffer.find(MESSAGE) else {
                        if last {
                            let rest = std::mem::take(&mut self.buffer);
                            if !rest.trim().is_empty() {
                                self.push_content(rest, out);
                            }
                        }
                        return;
                    };
                    let head: String = self.buffer.drain(..at + MESSAGE.len()).collect();
                    let (channel, recipient) = header(&head[..at]);
                    self.harmony = Some(Harmony::Body { channel, recipient, text: String::new() });
                }
                Harmony::Body { channel, recipient, mut text } => {
                    let found = END.iter().filter_map(|e| self.buffer.find(e).map(|i| (i, e.len()))).min();
                    let (take, done) = match found {
                        Some((i, n)) => (i, Some(n)),
                        None if last => (self.buffer.len(), None),
                        None => (safe_len(&self.buffer), None),
                    };
                    let chunk: String = self.buffer.drain(..take).collect();
                    if !chunk.is_empty() {
                        match (&recipient, channel) {
                            (Some(_), _) => text.push_str(&chunk),
                            (None, Channel::Analysis) => {
                                self.reasoning.push_str(&chunk);
                                out.push(Delta::Reasoning(chunk));
                            }
                            (None, _) => self.push_content(chunk, out),
                        }
                    }
                    match done {
                        Some(n) => {
                            self.buffer.drain(..n);
                            if let Some(r) = recipient {
                                self.harmony_call(&r, &text, out);
                            }
                            self.harmony = Some(Harmony::Header);
                        }
                        None => {
                            self.harmony = Some(Harmony::Body { channel, recipient, text });
                            return;
                        }
                    }
                }
            }
        }
    }

    pub(super) fn harmony_finish(&mut self, out: &mut Vec<Delta>) {
        self.harmony_feed(true, out);
        if let Some(Harmony::Body { recipient: Some(r), text, .. }) = self.harmony.clone() {
            self.harmony_call(&r, &text, out);
        }
        self.harmony = Some(Harmony::Header);
    }

    fn harmony_call(&mut self, recipient: &str, body: &str, out: &mut Vec<Delta>) {
        let name = recipient.strip_prefix("functions.").unwrap_or(recipient).to_string();
        let args = match body.trim() {
            "" => "{}".to_string(),
            b => b.to_string(),
        };
        self.slots.push(Slot { name, ended: true, raw: body.to_string(), ..Slot::default() });
        let index = self.slots.len() - 1;
        if self.send_name(index, out) {
            self.slots[index].arguments = Some(args.clone());
            out.push(Delta::Args { index, arguments: args });
        } else {
            self.slots.remove(index);
        }
    }
}
