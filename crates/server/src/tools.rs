use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::sync::Mutex;

static REPAIRS: Mutex<BTreeMap<&'static str, u64>> = Mutex::new(BTreeMap::new());

pub fn note_repair(kind: &'static str) {
    *REPAIRS.lock().unwrap().entry(kind).or_default() += 1;
}

pub fn repair_counts() -> Vec<(&'static str, u64)> {
    REPAIRS.lock().unwrap().iter().map(|(k, v)| (*k, *v)).collect()
}

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
    raw: String,
}

pub struct OutputParser {
    tools: Vec<Value>,
    tool_parsing: bool,
    json_calls: bool,
    kv_calls: bool,
    raw: Option<(RawFormat, RawPhase)>,
    state: State,
    buffer: String,
    slots: Vec<Slot>,
    content_nonws: bool,
    deferred_ws: String,
    content: String,
    reasoning: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RawFormat {
    TagList,
    BareJson,
    NamedList,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RawPhase {
    Undecided,
    Collect,
    Plain,
}

impl RawFormat {
    pub fn from_parser(name: &str) -> Option<Self> {
        match name {
            "tool_call_list" => Some(Self::TagList),
            "bare_json" => Some(Self::BareJson),
            "functools_list" => Some(Self::NamedList),
            _ => None,
        }
    }

    fn markers(self) -> &'static [&'static str] {
        match self {
            Self::TagList => &["<|tool_call|>"],
            Self::BareJson => &["{", "<|python_tag|>"],
            Self::NamedList => &["functools"],
        }
    }
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
            kv_calls: false,
            raw: None,
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

    pub fn key_value(mut self) -> Self {
        self.json_calls = true;
        self.kv_calls = true;
        self
    }

    pub fn raw_json(mut self, format: RawFormat) -> Self {
        self.json_calls = true;
        self.raw = Some((format, RawPhase::Undecided));
        self
    }

    pub fn for_parser(self, name: Option<&str>) -> Self {
        match name {
            Some("hermes") => self.hermes(),
            Some("xml_kv") => self.key_value(),
            Some(n) => match RawFormat::from_parser(n) {
                Some(f) if self.tool_parsing => self.raw_json(f),
                _ => self,
            },
            None => self,
        }
    }

    pub fn feed(&mut self, text: &str) -> Vec<Delta> {
        self.buffer.push_str(text);
        let mut out = Vec::new();
        if self.raw.is_some() {
            self.raw_feed(&mut out);
            return out;
        }
        self.drain(false, &mut out);
        out
    }

    fn raw_feed(&mut self, out: &mut Vec<Delta>) {
        let Some((format, phase)) = self.raw else { return };
        match phase {
            RawPhase::Collect => {}
            RawPhase::Plain => {
                let text = std::mem::take(&mut self.buffer);
                self.push_content(text, out);
            }
            RawPhase::Undecided => {
                let t = self.buffer.trim_start();
                if t.is_empty() {
                    return;
                }
                if format.markers().iter().any(|m| t.starts_with(m)) {
                    self.raw = Some((format, RawPhase::Collect));
                } else if format.markers().iter().any(|m| m.starts_with(t)) {
                } else {
                    self.raw = Some((format, RawPhase::Plain));
                    let text = std::mem::take(&mut self.buffer);
                    self.push_content(text, out);
                }
            }
        }
    }

    fn raw_finish(&mut self, out: &mut Vec<Delta>) {
        let Some((format, phase)) = self.raw else { return };
        let whole = std::mem::take(&mut self.buffer);
        match phase {
            RawPhase::Plain => {}
            RawPhase::Undecided => {
                if !whole.is_empty() {
                    self.push_content(whole, out);
                }
            }
            RawPhase::Collect => {
                let t = whole.trim_start();
                let body = format.markers().iter().find_map(|m| t.strip_prefix(m).filter(|_| *m != "{")).unwrap_or(t);
                self.slots.push(Slot { raw: whole.clone(), args: body.to_string(), ..Slot::default() });
                let index = self.slots.len() - 1;
                self.end_json_call(index, "", out);
            }
        }
        self.raw = Some((format, RawPhase::Plain));
    }

    pub fn finish(&mut self) -> Vec<Delta> {
        let mut out = Vec::new();
        if self.raw.is_some() {
            self.raw_finish(&mut out);
        }
        self.drain(true, &mut out);
        if matches!(self.state, State::Preamble | State::Name | State::Args | State::Between) {
            self.end_call(&mut out, "");
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
            State::Name => {
                let slot = self.slots.last_mut().unwrap();
                slot.raw.push_str(&text);
                slot.name.push_str(&text);
            }
            State::Args => self.slots.last_mut().unwrap().args.push_str(&text),
            State::Preamble => {
                let slot = self.slots.last_mut().unwrap();
                slot.raw.push_str(&text);
                slot.args.push_str(&text);
            }
            State::Between => {}
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
                self.open_slot(lit);
                self.state = Preamble;
            }
            (Name | Args, ToolStart) => {
                self.end_call(out, "");
                self.open_slot(lit);
                self.state = Preamble;
            }
            (Preamble, FuncPrefix) => {
                let slot = self.slots.last_mut().unwrap();
                slot.raw.push_str(lit);
                slot.args.clear();
                self.state = Name;
            }
            (Preamble, ParamStart) if self.slots.last().is_some_and(|s| s.args.trim().is_empty()) => {
                let slot = self.slots.last_mut().unwrap();
                slot.raw.push_str(lit);
                slot.args.clear();
                note_repair("parameter_as_function");
                self.state = Name;
            }
            (Content | Between, FuncPrefix) => {
                self.open_slot(lit);
                self.state = Name;
            }
            (Args, FuncPrefix) => {
                note_repair("missing_function_end");
                self.end_call(out, "");
                self.open_slot(lit);
                self.state = Name;
            }
            (Preamble, ToolEnd) => {
                self.end_call(out, lit);
                self.state = Content;
            }
            (Name, Close) => {
                let index = self.slots.len() - 1;
                self.slots[index].raw.push('>');
                match self.send_name(index, out) {
                    true => self.state = Args,
                    false => {
                        self.drop_slot_as_text(index, "", out);
                        self.state = Content;
                    }
                }
            }
            (Name | Args, FuncEnd) => {
                self.end_call(out, lit);
                self.state = Between;
            }
            (Name | Args, ToolEnd) => {
                if self.state == Args {
                    note_repair("missing_function_end");
                }
                self.end_call(out, lit);
                self.state = Content;
            }
            (Between, ToolEnd) => self.state = Content,
            _ => self.on_text(lit.to_string(), out),
        }
    }

    fn open_slot(&mut self, lit: &str) {
        self.slots.push(Slot { raw: lit.to_string(), ..Slot::default() });
    }

    fn drop_slot_as_text(&mut self, index: usize, closer: &str, out: &mut Vec<Delta>) {
        let slot = self.slots.remove(index);
        let mut text = slot.raw;
        text.push_str(closer);
        if !text.is_empty() {
            self.push_content(text, out);
        }
    }

    fn resolve(&self, name: &str) -> Option<String> {
        let name = name.trim();
        if name.is_empty() {
            return None;
        }
        let declared: Vec<&str> = self.tools.iter().filter_map(|t| t["function"]["name"].as_str()).collect();
        if declared.is_empty() || declared.contains(&name) {
            return Some(name.to_string());
        }
        let squash = |s: &str| {
            let mut out = String::new();
            for c in s.chars() {
                if !(c == '_' && out.ends_with('_')) {
                    out.push(c);
                }
            }
            out
        };
        let wanted = squash(name);
        let hits: Vec<&&str> = declared.iter().filter(|d| squash(d) == wanted).collect();
        match hits.as_slice() {
            [one] => {
                note_repair("namespace");
                Some(one.to_string())
            }
            _ => None,
        }
    }

    fn end_json_call(&mut self, index: usize, closer: &str, out: &mut Vec<Delta>) {
        self.slots[index].ended = true;
        let text = self.slots[index].args.trim().to_string();
        if self.kv_calls {
            return self.end_kv_call(index, &text, closer, out);
        }
        let mut calls = self.json_calls_in(&text);
        if calls.is_empty() && self.raw.is_some_and(|(f, _)| f == RawFormat::BareJson) && text.contains(';') {
            let parts: Vec<Vec<(String, String)>> = text.split(';').map(|p| self.json_calls_in(p.trim())).collect();
            if parts.iter().all(|p| !p.is_empty()) {
                calls = parts.into_iter().flatten().collect();
            }
        }
        if calls.is_empty() {
            return self.drop_slot_as_text(index, closer, out);
        }
        if !self.json_calls {
            note_repair("json_call");
        }
        for (i, (name, args)) in calls.into_iter().enumerate() {
            let at = match i {
                0 => index,
                _ => {
                    self.slots.push(Slot { ended: true, ..Slot::default() });
                    self.slots.len() - 1
                }
            };
            self.slots[at].name = name;
            self.send_name(at, out);
            self.slots[at].arguments = Some(args.clone());
            out.push(Delta::Args { index: at, arguments: args });
        }
    }

    fn end_kv_call(&mut self, index: usize, text: &str, closer: &str, out: &mut Vec<Delta>) {
        let cut = [text.find('\n'), text.find("<arg_key>")].into_iter().flatten().min().unwrap_or(text.len());
        let (name, rest) = text.split_at(cut);
        let Some(name) = self.resolve(name) else { return self.drop_slot_as_text(index, closer, out) };
        let arguments = convert_kv_args(rest, properties(&self.tools, &name));
        self.slots[index].name = name;
        self.send_name(index, out);
        self.slots[index].arguments = Some(arguments.clone());
        out.push(Delta::Args { index, arguments });
    }

    fn json_calls_in(&self, text: &str) -> Vec<(String, String)> {
        if text.is_empty() {
            return Vec::new();
        }
        let strict: Result<Vec<Value>, _> = serde_json::Deserializer::from_str(text).into_iter::<Value>().collect();
        let values = match strict {
            Ok(v) => v,
            Err(_) => match repair_json(text) {
                Some(v) => {
                    note_repair("json_arguments");
                    vec![v]
                }
                None => return Vec::new(),
            },
        };
        let values = values.into_iter().flat_map(|v| match v {
            Value::Array(items) => items,
            v => vec![v],
        });
        let mut calls = Vec::new();
        for v in values {
            let Some(name) = v["name"].as_str().and_then(|n| self.resolve(n)) else { return Vec::new() };
            let args = match v.get("arguments").or_else(|| v.get("parameters")) {
                Some(Value::String(s)) => s.clone(),
                Some(v) => py_dumps(v),
                None => "{}".to_string(),
            };
            calls.push((name, args));
        }
        calls
    }

    fn send_name(&mut self, index: usize, out: &mut Vec<Delta>) -> bool {
        let name = self.slots[index].name.trim().to_string();
        if self.slots[index].name_sent {
            return true;
        }
        let Some(name) = self.resolve(&name) else { return false };
        let slot = &mut self.slots[index];
        slot.name = name.clone();
        slot.name_sent = true;
        slot.id = format!("chatcmpl-tool-{}", uuid::Uuid::new_v4().simple());
        out.push(Delta::Call { index, id: slot.id.clone(), name });
        true
    }

    fn end_call(&mut self, out: &mut Vec<Delta>, closer: &str) {
        let Some(index) = self.slots.len().checked_sub(1) else { return };
        if self.slots[index].ended {
            return;
        }
        if self.state == State::Preamble {
            return self.end_json_call(index, closer, out);
        }
        if !self.send_name(index, out) {
            self.slots[index].ended = true;
            return self.drop_slot_as_text(index, closer, out);
        }
        let slot = &mut self.slots[index];
        slot.ended = true;
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

fn convert_kv_args(raw: &str, props: Option<&Map<String, Value>>) -> String {
    const KEY: &str = "<arg_key>";
    const KEY_END: &str = "</arg_key>";
    const VALUE: &str = "<arg_value>";
    const VALUE_END: &str = "</arg_value>";
    let props = props.filter(|p| !p.is_empty());
    let mut fields: Vec<String> = Vec::new();
    let mut rest = raw;
    while let Some(k) = rest.find(KEY) {
        let after = &rest[k + KEY.len()..];
        let Some(ke) = after.find(KEY_END) else { break };
        let key = after[..ke].trim().to_string();
        let after = &after[ke + KEY_END.len()..];
        let Some(v) = after.find(VALUE) else { break };
        let after = &after[v + VALUE.len()..];
        let (value, next) = match after.find(VALUE_END) {
            Some(ve) => (&after[..ve], &after[ve + VALUE_END.len()..]),
            None => (after, ""),
        };
        let value = value.trim();
        let rendered = match props.and_then(|p| p.get(&key)).filter(|s| s.is_object()) {
            Some(schema) => coerce(value, &schema_types(schema)),
            None => json_str(value),
        };
        fields.retain(|f| !f.starts_with(&format!("{}:", json_str(&key))));
        fields.push(format!("{}: {rendered}", json_str(&key)));
        rest = next;
    }
    format!("{{{}}}", fields.join(", "))
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
            _ => py_json_repaired(value),
        };
        if let Some(v) = hit {
            return v;
        }
    }
    py_json_repaired(value).unwrap_or_else(|| json_str(value))
}

fn py_json_repaired(value: &str) -> Option<String> {
    if let Some(v) = py_json(value) {
        return Some(v);
    }
    let v = value.trim();
    if !(v.starts_with('[') || v.starts_with('{')) {
        return None;
    }
    let repaired = repair_json(v)?;
    note_repair("arguments");
    Some(py_dumps(&repaired))
}

pub fn repair_json(text: &str) -> Option<Value> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '"' | '\'' => {
                out.push('"');
                i += 1;
                while i < chars.len() && chars[i] != c {
                    match chars[i] {
                        '\\' if i + 1 < chars.len() => {
                            let next = chars[i + 1];
                            match next == '\'' {
                                true => out.push('\''),
                                false => {
                                    out.push('\\');
                                    out.push(next);
                                }
                            }
                            i += 2;
                            continue;
                        }
                        '"' => out.push_str("\\\""),
                        '\n' => out.push_str("\\n"),
                        ch => out.push(ch),
                    }
                    i += 1;
                }
                out.push('"');
                i += 1;
            }
            ',' => {
                let next = chars[i + 1..].iter().find(|c| !c.is_whitespace());
                if !matches!(next, Some('}') | Some(']')) {
                    out.push(',');
                }
                i += 1;
            }
            c if c.is_alphabetic() => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                out.push_str(match word.as_str() {
                    "True" => "true",
                    "False" => "false",
                    "None" => "null",
                    w => w,
                });
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    serde_json::from_str(&out).ok()
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
