use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    Content(String),
    Reasoning(String),
    Call { index: usize, id: String, name: String },
    Args { index: usize, arguments: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Parsed {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Content,
    Reasoning,
    Preamble,
    Name,
    Args,
    Between,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Term {
    ThinkStart,
    ThinkEnd,
    ToolStart,
    ToolEnd,
    FuncPrefix,
    FuncEnd,
    ParamStart,
    ParamEnd,
    Close,
}

const TERMS: [(&str, Term); 8] = [
    ("<think>", Term::ThinkStart),
    ("</think>", Term::ThinkEnd),
    ("<tool_call>", Term::ToolStart),
    ("</tool_call>", Term::ToolEnd),
    ("<function=", Term::FuncPrefix),
    ("</function>", Term::FuncEnd),
    ("<parameter=", Term::ParamStart),
    ("</parameter>", Term::ParamEnd),
];

#[derive(Default)]
struct Slot {
    id: String,
    name: String,
    args: String,
    arguments: Option<String>,
    name_sent: bool,
    ended: bool,
}

pub struct OutputParser {
    tools: Vec<Value>,
    tool_parsing: bool,
    json_calls: bool,
    state: State,
    buffer: String,
    slots: Vec<Slot>,
    content_nonws: bool,
    deferred_ws: String,
    content: String,
    reasoning: String,
}

enum Lex {
    Text(String),
    Term(Term, &'static str),
}

impl OutputParser {
    pub fn new(tools: &[Value], tool_parsing: bool, reasoning: bool) -> Self {
        Self {
            tools: tools.to_vec(),
            tool_parsing,
            json_calls: false,
            state: if reasoning { State::Reasoning } else { State::Content },
            buffer: String::new(),
            slots: Vec::new(),
            content_nonws: false,
            deferred_ws: String::new(),
            content: String::new(),
            reasoning: String::new(),
        }
    }

    pub fn hermes(mut self) -> Self {
        self.json_calls = true;
        self
    }

    pub fn feed(&mut self, text: &str) -> Vec<Delta> {
        self.buffer.push_str(text);
        let mut out = Vec::new();
        self.drain(false, &mut out);
        out
    }

    pub fn finish(&mut self) -> Vec<Delta> {
        let mut out = Vec::new();
        self.drain(true, &mut out);
        if matches!(self.state, State::Preamble | State::Name | State::Args | State::Between) {
            self.end_call(&mut out);
        }
        self.state = State::Content;
        if !self.deferred_ws.is_empty() && self.slots.is_empty() {
            out.push(Delta::Content(std::mem::take(&mut self.deferred_ws)));
        }
        out
    }

    pub fn called(&self) -> bool {
        self.slots.iter().any(|s| s.name_sent)
    }

    pub fn result(&self) -> Parsed {
        let tool_calls: Vec<ToolCall> = self
            .slots
            .iter()
            .filter(|s| s.name_sent)
            .map(|s| ToolCall {
                id: s.id.clone(),
                name: s.name.clone(),
                arguments: s.arguments.clone().unwrap_or_else(|| "{}".into()),
            })
            .collect();
        let content = match tool_calls.is_empty() {
            true => self.content.clone(),
            false => self.content.trim().to_string(),
        };
        Parsed {
            content: (!content.is_empty()).then_some(content),
            reasoning: (!self.reasoning.is_empty()).then(|| self.reasoning.clone()),
            tool_calls,
        }
    }

    pub fn parse(tools: &[Value], tool_parsing: bool, reasoning: bool, text: &str) -> Parsed {
        let mut p = Self::new(tools, tool_parsing, reasoning);
        p.feed(text);
        p.finish();
        p.result()
    }

    fn drain(&mut self, last: bool, out: &mut Vec<Delta>) {
        while let Some(lex) = self.next_lex(last) {
            match lex {
                Lex::Text(t) => self.on_text(t, out),
                Lex::Term(t, lit) => self.on_term(t, lit, out),
            }
        }
    }

    fn next_lex(&mut self, last: bool) -> Option<Lex> {
        if self.buffer.is_empty() {
            return None;
        }
        let close = self.state == State::Name;
        for (i, c) in self.buffer.char_indices() {
            if c != '<' && !(close && c == '>') {
                continue;
            }
            if i > 0 {
                let text: String = self.buffer.drain(..i).collect();
                return Some(Lex::Text(text));
            }
            if c == '>' {
                self.buffer.drain(..1);
                return Some(Lex::Term(Term::Close, ">"));
            }
            if let Some((lit, t)) = TERMS.iter().find(|(lit, _)| self.buffer.starts_with(lit)) {
                self.buffer.drain(..lit.len());
                return Some(Lex::Term(*t, lit));
            }
            if !last && TERMS.iter().any(|(lit, _)| lit.starts_with(self.buffer.as_str())) {
                return None;
            }
            let n = c.len_utf8();
            let rest = &self.buffer[n..];
            let next = rest.find(|ch| ch == '<' || (close && ch == '>')).map(|j| j + n).unwrap_or(self.buffer.len());
            let text: String = self.buffer.drain(..next).collect();
            return Some(Lex::Text(text));
        }
        Some(Lex::Text(std::mem::take(&mut self.buffer)))
    }

    fn on_text(&mut self, text: String, out: &mut Vec<Delta>) {
        match self.state {
            State::Content => self.push_content(text, out),
            State::Reasoning => {
                self.reasoning.push_str(&text);
                out.push(Delta::Reasoning(text));
            }
            State::Name => self.slots.last_mut().unwrap().name.push_str(&text),
            State::Args => self.slots.last_mut().unwrap().args.push_str(&text),
            State::Preamble if self.json_calls => self.slots.last_mut().unwrap().args.push_str(&text),
            State::Preamble | State::Between => {}
        }
    }

    fn push_content(&mut self, text: String, out: &mut Vec<Delta>) {
        self.content.push_str(&text);
        if self.content_nonws {
            out.push(Delta::Content(text));
            return;
        }
        let joined = std::mem::take(&mut self.deferred_ws) + &text;
        if !joined.trim().is_empty() {
            self.content_nonws = true;
            out.push(Delta::Content(joined));
        } else if self.slots.is_empty() {
            self.deferred_ws = joined;
        }
    }

    fn on_term(&mut self, t: Term, lit: &'static str, out: &mut Vec<Delta>) {
        use State::*;
        use Term::*;
        if !self.tool_parsing && !matches!(t, ThinkStart | ThinkEnd) {
            if self.state == Reasoning && t == ToolStart {
                self.state = Content;
            }
            return self.on_text(lit.to_string(), out);
        }
        match (self.state, t) {
            (Reasoning, ThinkStart) | (Content, ThinkEnd) => {}
            (Reasoning, ThinkEnd) => self.state = Content,
            (Reasoning | Content | Between, ToolStart) => {
                self.slots.push(Slot::default());
                self.state = Preamble;
            }
            (Preamble, FuncPrefix) => self.state = Name,
            (Content | Between, FuncPrefix) => {
                self.slots.push(Slot::default());
                self.state = Name;
            }
            (Preamble, ToolEnd) => {
                self.end_call(out);
                self.state = Content;
            }
            (Name, Close) => {
                self.send_name(out);
                self.state = Args;
            }
            (Name | Args, FuncEnd) => {
                self.end_call(out);
                self.state = Between;
            }
            (Between, ToolEnd) => self.state = Content,
            _ => self.on_text(lit.to_string(), out),
        }
    }

    fn end_json_call(&mut self, index: usize, out: &mut Vec<Delta>) {
        let slot = &mut self.slots[index];
        slot.ended = true;
        let Ok(body) = serde_json::from_str::<Value>(slot.args.trim()) else { return };
        let Some(name) = body["name"].as_str().filter(|n| !n.is_empty()) else { return };
        let args = match body.get("arguments").or_else(|| body.get("parameters")) {
            Some(Value::String(s)) => s.clone(),
            Some(v) => crate::tools::py_dumps(v),
            None => "{}".to_string(),
        };
        slot.name = name.to_string();
        self.send_name(out);
        let slot = &mut self.slots[index];
        slot.arguments = Some(args.clone());
        out.push(Delta::Args { index, arguments: args });
    }

    fn send_name(&mut self, out: &mut Vec<Delta>) {
        let index = self.slots.len() - 1;
        let slot = &mut self.slots[index];
        let name = slot.name.trim().to_string();
        if slot.name_sent || name.is_empty() {
            return;
        }
        slot.name = name.clone();
        slot.name_sent = true;
        slot.id = format!("chatcmpl-tool-{}", uuid::Uuid::new_v4().simple());
        out.push(Delta::Call { index, id: slot.id.clone(), name });
    }

    fn end_call(&mut self, out: &mut Vec<Delta>) {
        let Some(index) = self.slots.len().checked_sub(1) else { return };
        if self.slots[index].ended {
            return;
        }
        if self.json_calls && self.state == State::Preamble {
            return self.end_json_call(index, out);
        }
        self.send_name(out);
        let slot = &mut self.slots[index];
        slot.ended = true;
        if !slot.name_sent {
            return;
        }
        let arguments = match slot.args.trim().is_empty() {
            true => "{}".to_string(),
            false => convert_args(&slot.args, properties(&self.tools, &slot.name)),
        };
        slot.arguments = Some(arguments.clone());
        out.push(Delta::Args { index, arguments });
    }
}

fn properties<'a>(tools: &'a [Value], name: &str) -> Option<&'a Map<String, Value>> {
    tools.iter().find_map(|t| {
        let f = &t["function"];
        match f["name"].as_str() == Some(name) {
            true => f["parameters"]["properties"].as_object(),
            false => None,
        }
    })
}

fn trim_wrapping_newlines(v: &str) -> &str {
    let v = v.strip_prefix('\n').unwrap_or(v);
    v.strip_suffix('\n').unwrap_or(v)
}

fn params(raw: &str) -> Vec<(String, String)> {
    const OPEN: &str = "<parameter=";
    const CLOSE: &str = "</parameter>";
    let mut out: Vec<(String, String)> = Vec::new();
    let mut pos = 0;
    while let Some(off) = raw[pos..].find(OPEN) {
        let start = pos + off + OPEN.len();
        let Some(gt) = raw[start..].find('>') else { break };
        let name = raw[start..start + gt].to_string();
        let vstart = start + gt + 1;
        let rest = &raw[vstart..];
        let (value, next) = match (rest.find(CLOSE), rest.find(OPEN)) {
            (Some(c), n) if n.is_none_or(|n| c < n) => (&rest[..c], vstart + c + CLOSE.len()),
            (_, Some(n)) => (&rest[..n], vstart + n),
            _ => break,
        };
        let value = trim_wrapping_newlines(value).to_string();
        match out.iter_mut().find(|(k, _)| *k == name) {
            Some(slot) => slot.1 = value,
            None => out.push((name, value)),
        }
        pos = next;
    }
    out
}

fn convert_args(raw: &str, props: Option<&Map<String, Value>>) -> String {
    let props = props.filter(|p| !p.is_empty());
    let fields: Vec<String> = params(raw)
        .into_iter()
        .map(|(k, v)| {
            let value = match props.and_then(|p| p.get(&k)).filter(|s| s.is_object()) {
                Some(schema) => coerce(&v, &schema_types(schema)),
                None => json_str(&v),
            };
            format!("{}: {value}", json_str(&k))
        })
        .collect();
    format!("{{{}}}", fields.join(", "))
}

fn json_str(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}

fn schema_types(schema: &Value) -> Vec<String> {
    let mut types: Vec<String> = Vec::new();
    let mut add = |t: &str| {
        if !types.iter().any(|x| x == t) {
            types.push(t.to_string());
        }
    };
    match &schema["type"] {
        Value::String(t) => add(t),
        Value::Array(list) => list.iter().filter_map(Value::as_str).for_each(&mut add),
        _ => {}
    }
    if let Some(values) = schema["enum"].as_array() {
        for v in values {
            add(match v {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(n) if n.is_f64() => "number",
                Value::Number(_) => "integer",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            });
        }
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(choices) = schema[key].as_array() {
            for c in choices {
                let nested = match c.is_object() {
                    true => schema_types(c),
                    false => vec!["string".to_string()],
                };
                nested.iter().for_each(|t| add(t));
            }
        }
    }
    if types.is_empty() {
        types.push("string".into());
    }
    types
}

fn normalize_type(t: &str) -> String {
    let t = t.trim().to_lowercase();
    let alias = match t.as_str() {
        "str" | "text" | "varchar" | "char" | "enum" => "string",
        "int" | "int32" | "int64" | "uint" | "uint32" | "uint64" | "long" | "short" | "unsigned" => "integer",
        "float" | "float32" | "float64" | "double" => "number",
        "bool" => "boolean",
        "dict" => "object",
        "arr" | "list" | "sequence" => "array",
        other => other,
    };
    alias.to_string()
}

fn coerce(value: &str, types: &[String]) -> String {
    let types: Vec<String> = types.iter().map(|t| normalize_type(t)).collect();
    let has = |t: &str| types.iter().any(|x| x == t);
    for t in ["null", "integer", "number", "boolean", "object", "array", "string"] {
        if !has(t) {
            continue;
        }
        let hit = match t {
            "null" => (value.to_lowercase() == "null").then(|| "null".to_string()),
            "string" => return json_str(value),
            "integer" => py_int(value),
            "number" => py_number(value),
            "boolean" => match value.trim().to_lowercase().as_str() {
                "true" | "1" => Some("true".into()),
                "false" | "0" => Some("false".into()),
                _ => None,
            },
            _ => py_json(value),
        };
        if let Some(v) = hit {
            return v;
        }
    }
    py_json(value).unwrap_or_else(|| json_str(value))
}

fn py_int(value: &str) -> Option<String> {
    let v = value.trim();
    let (neg, digits) = match v.strip_prefix('-') {
        Some(d) => (true, d),
        None => (false, v.strip_prefix('+').unwrap_or(v)),
    };
    let ok = !digits.is_empty()
        && digits.chars().all(|c| c.is_ascii_digit() || c == '_')
        && !digits.starts_with('_')
        && !digits.ends_with('_')
        && !digits.contains("__");
    if !ok {
        return None;
    }
    let digits: String = digits.chars().filter(|c| *c != '_').collect();
    let digits = digits.trim_start_matches('0');
    Some(match (digits.is_empty(), neg) {
        (true, _) => "0".to_string(),
        (false, true) => format!("-{digits}"),
        (false, false) => digits.to_string(),
    })
}

fn py_number(value: &str) -> Option<String> {
    let v: f64 = value.trim().parse().ok().filter(|v: &f64| v.is_finite())?;
    Some(match v.fract() == 0.0 {
        true => format!("{v:.0}"),
        false => format!("{v}"),
    })
}

fn py_float(v: f64) -> String {
    match v.fract() == 0.0 && v.abs() < 1e16 {
        true => format!("{v:.1}"),
        false => format!("{v}"),
    }
}

fn py_dumps(v: &Value) -> String {
    py_json(&v.to_string()).unwrap_or_else(|| v.to_string())
}

fn py_json(text: &str) -> Option<String> {
    serde_json::from_str::<Value>(text).ok()?;
    let mut out = String::new();
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        match c {
            '"' => {
                let mut end = i + 1;
                let mut escape = false;
                for (j, ch) in chars.by_ref() {
                    end = j + ch.len_utf8();
                    match (escape, ch) {
                        (true, _) => escape = false,
                        (false, '\\') => escape = true,
                        (false, '"') => break,
                        _ => {}
                    }
                }
                let s: String = serde_json::from_str(&text[i..end]).ok()?;
                out.push_str(&json_str(&s));
            }
            ',' => out.push_str(", "),
            ':' => out.push_str(": "),
            c if c.is_whitespace() => {}
            c if c == '-' || c.is_ascii_digit() => {
                let mut end = i + 1;
                while let Some(&(j, ch)) = chars.peek() {
                    if !(ch.is_ascii_digit() || matches!(ch, '.' | 'e' | 'E' | '+' | '-')) {
                        break;
                    }
                    end = j + 1;
                    chars.next();
                }
                let tok = &text[i..end];
                match tok.contains(['.', 'e', 'E']) {
                    true => out.push_str(&py_float(tok.parse().ok()?)),
                    false => out.push_str(&py_int(tok)?),
                }
            }
            c => out.push(c),
        }
    }
    Some(out)
}

#[cfg(test)]
#[path = "tools_tests.rs"]
mod tests;
