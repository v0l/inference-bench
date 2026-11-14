use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

const MAX_DEPTH: usize = 128;
const ANY: u32 = 0;
const ANY_OBJECT: u32 = 1;

#[derive(Clone, Debug)]
enum Node {
    AnyOf(Vec<u32>),
    Object { props: Vec<Prop>, extra: Option<u32> },
    Array { items: u32 },
    Str,
    Num { int: bool },
    Lit(u32),
}

#[derive(Clone, Debug)]
struct Prop {
    key: u32,
    node: u32,
    required: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Frame {
    Value(u32),
    Lit(u32, u16),
    Str { cont: u8, esc: u8 },
    Num { phase: u8, int: bool },
    Obj { node: u32, next: u16, phase: u8 },
    Arr { node: u32, phase: u8 },
}

type Stack = Vec<Frame>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State(Vec<Stack>);

impl State {
    fn is_dead(&self) -> bool {
        self.0.is_empty()
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn num_complete(phase: u8) -> bool {
    matches!(phase, 2 | 3 | 5 | 8)
}

fn num_step(phase: u8, int: bool, b: u8) -> Option<u8> {
    let digit = b.is_ascii_digit();
    match phase {
        1 => match b {
            b'0' => Some(2),
            b'1'..=b'9' => Some(3),
            _ => None,
        },
        2 | 3 | 5 | 8 if digit && phase == 3 => Some(3),
        2 | 3 if b == b'.' && !int => Some(4),
        2 | 3 | 5 if matches!(b, b'e' | b'E') && !int => Some(6),
        4 | 5 if digit => Some(5),
        6 => match b {
            b'+' | b'-' => Some(7),
            _ if digit => Some(8),
            _ => None,
        },
        7 | 8 if digit => Some(8),
        _ => None,
    }
}

struct Compiled {
    nodes: Vec<Node>,
    lits: Vec<Vec<u8>>,
}

impl Compiled {
    fn new() -> Self {
        let mut c = Compiled { nodes: Vec::new(), lits: Vec::new() };
        c.nodes.push(Node::AnyOf(Vec::new()));
        c.nodes.push(Node::Object { props: Vec::new(), extra: Some(ANY) });
        let array = c.push(Node::Array { items: ANY });
        let string = c.push(Node::Str);
        let number = c.push(Node::Num { int: false });
        let t = c.literal(b"true");
        let f = c.literal(b"false");
        let n = c.literal(b"null");
        c.nodes[ANY as usize] = Node::AnyOf(vec![ANY_OBJECT, array, string, number, t, f, n]);
        c
    }

    fn push(&mut self, node: Node) -> u32 {
        self.nodes.push(node);
        (self.nodes.len() - 1) as u32
    }

    fn lit_id(&mut self, bytes: &[u8]) -> u32 {
        if let Some(i) = self.lits.iter().position(|l| l == bytes) {
            return i as u32;
        }
        self.lits.push(bytes.to_vec());
        (self.lits.len() - 1) as u32
    }

    fn literal(&mut self, bytes: &[u8]) -> u32 {
        let id = self.lit_id(bytes);
        self.push(Node::Lit(id))
    }
}

pub struct TokenTable {
    tokens: Vec<Vec<u8>>,
    trie: Vec<TrieNode>,
}

#[derive(Default)]
struct TrieNode {
    children: Vec<(u8, u32)>,
    tokens: Vec<u32>,
}

impl TokenTable {
    pub fn new(tokens: Vec<Vec<u8>>) -> Self {
        let mut trie = vec![TrieNode::default()];
        for (id, bytes) in tokens.iter().enumerate() {
            if bytes.is_empty() {
                continue;
            }
            let mut at = 0usize;
            for &b in bytes {
                at = match trie[at].children.iter().find(|c| c.0 == b) {
                    Some(c) => c.1 as usize,
                    None => {
                        trie.push(TrieNode::default());
                        let new = trie.len() - 1;
                        trie[at].children.push((b, new as u32));
                        new
                    }
                };
            }
            trie[at].tokens.push(id as u32);
        }
        TokenTable { tokens, trie }
    }

    pub fn from_tokenizer(tokenizer: &tokenizers::Tokenizer) -> Self {
        let vocab = tokenizer.get_vocab(true);
        let special: std::collections::HashSet<u32> =
            tokenizer.get_added_tokens_decoder().into_iter().filter(|(_, t)| t.special).map(|(id, _)| id).collect();
        let size = vocab.values().copied().max().map_or(0, |m| m as usize + 1);
        let mut tokens = vec![Vec::new(); size];
        for (text, id) in vocab {
            if !special.contains(&id) {
                tokens[id as usize] = token_bytes(&text);
            }
        }
        TokenTable::new(tokens)
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn bytes(&self, id: u32) -> &[u8] {
        self.tokens.get(id as usize).map_or(&[], |t| t.as_slice())
    }
}

fn byte_level_inverse(c: char) -> Option<u8> {
    let c = c as u32;
    if (0x21..=0x7e).contains(&c) || (0xa1..=0xac).contains(&c) || (0xae..=0xff).contains(&c) {
        return Some(c as u8);
    }
    let printable: Vec<u32> = (0x21..=0x7e).chain(0xa1..=0xac).chain(0xae..=0xff).collect();
    let mut extra = 0u32;
    for b in 0u32..256 {
        if !printable.contains(&b) {
            if 256 + extra == c {
                return Some(b as u8);
            }
            extra += 1;
        }
    }
    None
}

fn token_bytes(text: &str) -> Vec<u8> {
    if let Some(hex) = text.strip_prefix("<0x").and_then(|t| t.strip_suffix('>'))
        && let Ok(b) = u8::from_str_radix(hex, 16)
    {
        return vec![b];
    }
    let mapped: Option<Vec<u8>> = text.chars().map(byte_level_inverse).collect();
    match mapped {
        Some(bytes) => bytes,
        None => text.replace('\u{2581}', " ").into_bytes(),
    }
}

pub struct Grammar {
    compiled: Compiled,
    root: u32,
    table: Arc<TokenTable>,
    vocab: usize,
    stop: Vec<u32>,
    cache: Mutex<HashMap<State, Arc<Vec<u32>>>>,
}

impl PartialEq for Grammar {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl std::fmt::Debug for Grammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Grammar(root {})", self.root)
    }
}

impl Grammar {
    pub fn json_object(table: Arc<TokenTable>, vocab: usize, stop: Vec<u32>) -> Arc<Grammar> {
        Arc::new(Grammar { compiled: Compiled::new(), root: ANY_OBJECT, table, vocab, stop, cache: Mutex::new(HashMap::new()) })
    }

    pub fn json_any(table: Arc<TokenTable>, vocab: usize, stop: Vec<u32>) -> Arc<Grammar> {
        Arc::new(Grammar { compiled: Compiled::new(), root: ANY, table, vocab, stop, cache: Mutex::new(HashMap::new()) })
    }

    pub fn cursor(self: &Arc<Self>) -> Cursor {
        Cursor { grammar: self.clone(), state: State(vec![vec![Frame::Value(self.root)]]) }
    }

    fn node(&self, n: u32) -> &Node {
        &self.compiled.nodes[n as usize]
    }

    fn feed(&self, mut st: Stack, b: u8, out: &mut Vec<Stack>) {
        let Some(top) = st.pop() else { return };
        match top {
            Frame::Value(n) => {
                if is_ws(b) {
                    st.push(Frame::Value(n));
                    out.push(st);
                } else {
                    self.begin(n, b, st, out);
                }
            }
            Frame::Lit(id, pos) => {
                let lit = &self.compiled.lits[id as usize];
                if lit[pos as usize] == b {
                    if (pos as usize) + 1 < lit.len() {
                        st.push(Frame::Lit(id, pos + 1));
                    }
                    out.push(st);
                }
            }
            Frame::Str { cont, esc } => {
                if let Some(next) = str_step(cont, esc, b) {
                    match next {
                        None => out.push(st),
                        Some((cont, esc)) => {
                            st.push(Frame::Str { cont, esc });
                            out.push(st);
                        }
                    }
                }
            }
            Frame::Num { phase, int } => match num_step(phase, int, b) {
                Some(p) => {
                    st.push(Frame::Num { phase: p, int });
                    out.push(st);
                }
                None if num_complete(phase) => self.feed(st, b, out),
                None => {}
            },
            Frame::Obj { node, next, phase } => self.feed_object(node, next, phase, st, b, out),
            Frame::Arr { node, phase } => self.feed_array(node, phase, st, b, out),
        }
    }

    fn begin(&self, n: u32, b: u8, mut st: Stack, out: &mut Vec<Stack>) {
        if st.len() >= MAX_DEPTH {
            return;
        }
        match self.node(n) {
            Node::AnyOf(alts) => {
                for &a in alts {
                    self.begin(a, b, st.clone(), out);
                }
            }
            Node::Lit(id) => {
                let lit = &self.compiled.lits[*id as usize];
                if lit[0] == b {
                    if lit.len() > 1 {
                        st.push(Frame::Lit(*id, 1));
                    }
                    out.push(st);
                }
            }
            Node::Str => {
                if b == b'"' {
                    st.push(Frame::Str { cont: 0, esc: 0 });
                    out.push(st);
                }
            }
            Node::Num { int } => {
                let phase = match b {
                    b'-' => 1,
                    b'0' => 2,
                    b'1'..=b'9' => 3,
                    _ => return,
                };
                st.push(Frame::Num { phase, int: *int });
                out.push(st);
            }
            Node::Array { .. } => {
                if b == b'[' {
                    st.push(Frame::Arr { node: n, phase: 0 });
                    out.push(st);
                }
            }
            Node::Object { .. } => {
                if b == b'{' {
                    st.push(Frame::Obj { node: n, next: 0, phase: 0 });
                    out.push(st);
                }
            }
        }
    }

    fn closable(props: &[Prop], next: usize) -> bool {
        props[next.min(props.len())..].iter().all(|p| !p.required)
    }

    fn feed_object(&self, n: u32, next: u16, phase: u8, mut st: Stack, b: u8, out: &mut Vec<Stack>) {
        let Node::Object { props, extra } = self.node(n) else { return };
        let keep = |st: &mut Stack, next: u16, phase: u8| st.push(Frame::Obj { node: n, next, phase });
        if is_ws(b) {
            keep(&mut st, next, phase);
            out.push(st);
            return;
        }
        let next_i = next as usize;
        match phase {
            0 | 1 => {
                if b == b'}' && phase == 0 {
                    if props.is_empty() || Self::closable(props, next_i) {
                        out.push(st);
                    }
                } else if b == b'"' {
                    if props.is_empty() {
                        if extra.is_some() {
                            keep(&mut st, next, 2);
                            st.push(Frame::Str { cont: 0, esc: 0 });
                            out.push(st);
                        }
                        return;
                    }
                    for j in next_i..props.len() {
                        let mut cand = st.clone();
                        keep(&mut cand, j as u16, 2);
                        cand.push(Frame::Lit(props[j].key, 1));
                        out.push(cand);
                        if props[j].required {
                            break;
                        }
                    }
                }
            }
            2 => {
                if b == b':' {
                    let (value, after) = if props.is_empty() {
                        let Some(e) = extra else { return };
                        (*e, next)
                    } else {
                        (props[next_i].node, next + 1)
                    };
                    keep(&mut st, after, 3);
                    st.push(Frame::Value(value));
                    out.push(st);
                }
            }
            _ => {
                if b == b',' {
                    if props.is_empty() || next_i < props.len() {
                        keep(&mut st, next, 1);
                        out.push(st);
                    }
                } else if b == b'}' && (props.is_empty() || Self::closable(props, next_i)) {
                    out.push(st);
                }
            }
        }
    }

    fn feed_array(&self, n: u32, phase: u8, mut st: Stack, b: u8, out: &mut Vec<Stack>) {
        let Node::Array { items } = self.node(n) else { return };
        if is_ws(b) {
            st.push(Frame::Arr { node: n, phase });
            out.push(st);
            return;
        }
        match phase {
            0 | 2 => {
                if b == b']' && phase == 0 {
                    out.push(st);
                } else {
                    st.push(Frame::Arr { node: n, phase: 1 });
                    self.begin(*items, b, st, out);
                }
            }
            _ => match b {
                b',' => {
                    st.push(Frame::Arr { node: n, phase: 2 });
                    out.push(st);
                }
                b']' => out.push(st),
                _ => {}
            },
        }
    }

    fn step(&self, state: &State, b: u8) -> State {
        let mut out = Vec::new();
        for st in &state.0 {
            self.feed(st.clone(), b, &mut out);
        }
        out.sort();
        out.dedup();
        State(out)
    }

    fn complete(&self, state: &State) -> bool {
        state.0.iter().any(|s| match s.as_slice() {
            [] => true,
            [Frame::Num { phase, .. }] => num_complete(*phase),
            _ => false,
        })
    }

    fn walk(&self, state: &State, node: usize, words: &mut [u32]) {
        for &t in &self.table.trie[node].tokens {
            if (t as usize) < self.vocab {
                words[t as usize / 32] |= 1 << (t % 32);
            }
        }
        for &(b, child) in &self.table.trie[node].children {
            let next = self.step(state, b);
            if !next.is_dead() {
                self.walk(&next, child as usize, words);
            }
        }
    }

    fn mask(&self, state: &State) -> Arc<Vec<u32>> {
        if let Some(m) = self.cache.lock().unwrap().get(state) {
            return m.clone();
        }
        let mut words = vec![0u32; self.vocab.div_ceil(32)];
        self.walk(state, 0, &mut words);
        if self.complete(state) || words.iter().all(|&w| w == 0) {
            for &t in &self.stop {
                if (t as usize) < self.vocab {
                    words[t as usize / 32] |= 1 << (t % 32);
                }
            }
        }
        let words = Arc::new(words);
        self.cache.lock().unwrap().insert(state.clone(), words.clone());
        words
    }

    pub fn cached_states(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
}

fn str_step(cont: u8, esc: u8, b: u8) -> Option<Option<(u8, u8)>> {
    if esc == 1 {
        return match b {
            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => Some(Some((0, 0))),
            b'u' => Some(Some((0, 14))),
            _ => None,
        };
    }
    if esc > 1 {
        if !b.is_ascii_hexdigit() {
            return None;
        }
        if esc == 14 && matches!(b, b'd' | b'D') {
            return Some(Some((0, 30)));
        }
        if esc == 30 {
            return matches!(b, b'0'..=b'7').then_some(Some((0, 12)));
        }
        let left = esc - 11;
        return Some(Some((0, if left == 0 { 0 } else { 10 + left })));
    }
    if cont > 0 {
        let range = match cont >> 4 {
            1 => 0xa0..=0xbf,
            2 => 0x80..=0x9f,
            3 => 0x90..=0xbf,
            4 => 0x80..=0x8f,
            _ => 0x80..=0xbf,
        };
        return range.contains(&b).then_some(Some(((cont & 0x0f) - 1, 0)));
    }
    match b {
        b'"' => Some(None),
        b'\\' => Some(Some((0, 1))),
        0..=0x1f => None,
        0x20..=0x7f => Some(Some((0, 0))),
        0xc2..=0xdf => Some(Some((1, 0))),
        0xe0 => Some(Some((0x12, 0))),
        0xed => Some(Some((0x22, 0))),
        0xe1..=0xef => Some(Some((2, 0))),
        0xf0 => Some(Some((0x33, 0))),
        0xf4 => Some(Some((0x43, 0))),
        0xf1..=0xf3 => Some(Some((3, 0))),
        _ => None,
    }
}

#[derive(Clone)]
pub struct Cursor {
    grammar: Arc<Grammar>,
    state: State,
}

impl std::fmt::Debug for Cursor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cursor({} stacks)", self.state.0.len())
    }
}

impl PartialEq for Cursor {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.grammar, &other.grammar) && self.state == other.state
    }
}

impl Cursor {
    pub fn mask(&self) -> Arc<Vec<u32>> {
        self.grammar.mask(&self.state)
    }

    pub fn advance(&mut self, token: u32) -> bool {
        if self.grammar.stop.contains(&token) {
            return self.grammar.complete(&self.state);
        }
        let mut state = self.state.clone();
        for &b in self.grammar.table.bytes(token) {
            state = self.grammar.step(&state, b);
            if state.is_dead() {
                return false;
            }
        }
        self.state = state;
        true
    }

    pub fn complete(&self) -> bool {
        self.grammar.complete(&self.state)
    }

    pub fn finished(&self) -> bool {
        self.state.0.len() == 1 && self.state.0[0].is_empty()
    }
}

pub fn parse_response_format(value: &Value) -> Result<Option<Schema>, String> {
    if value.is_null() {
        return Ok(None);
    }
    match value["type"].as_str() {
        Some("text") => Ok(None),
        Some("json_object") => Ok(Some(Schema::Object)),
        Some("json_schema") => Err("response_format json_schema is not supported yet".into()),
        Some(other) => Err(format!("response_format type {other:?} is not supported")),
        None => Err("response_format needs a type".into()),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Schema {
    Object,
}

impl Schema {
    pub fn build(&self, table: Arc<TokenTable>, vocab: usize, stop: Vec<u32>) -> Arc<Grammar> {
        match self {
            Schema::Object => Grammar::json_object(table, vocab, stop),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOS: u32 = 0;

    fn single_bytes() -> Arc<TokenTable> {
        let mut tokens = vec![Vec::new()];
        tokens.extend((0u8..=255).map(|b| vec![b]));
        Arc::new(TokenTable::new(tokens))
    }

    fn accepts(g: &Arc<Grammar>, text: &str) -> bool {
        let mut c = g.cursor();
        text.bytes().all(|b| c.advance(b as u32 + 1)) && c.complete()
    }

    fn object() -> Arc<Grammar> {
        Grammar::json_object(single_bytes(), 257, vec![EOS])
    }

    #[test]
    fn json_object_accepts_objects() {
        let g = object();
        for ok in [
            "{}",
            " { }",
            r#"{"a":1}"#,
            r#"{"a": [1, 2.5, -3e+2, "x\n\u00e9", true, false, null, {"b": {}}], "c": ""}"#,
            "{\"é\": \"日本語\"}",
        ] {
            assert!(accepts(&g, ok), "{ok}");
        }
    }

    #[test]
    fn json_object_rejects_everything_else() {
        let g = object();
        for bad in ["[]", "1", "\"a\"", "{", "{\"a\"}", "{\"a\":}", "{\"a\":1,}", "{\"a\":01}", "{\"a\":1.}", "{\"a\":tru}", "{'a':1}", "{\"a\":\"\\x\"}", "{\"a\":1} x", "{\"a\":\"\n\"}", "{\"a\":\"\\ud800\"}", "{\"a\":\"\\ud8\"}"] {
            assert!(!accepts(&g, bad), "{bad}");
        }
    }

    #[test]
    fn strings_only_admit_valid_utf8_and_non_surrogate_escapes() {
        let g = object();
        let ok = |b: &[u8]| {
            let mut c = g.cursor();
            b.iter().all(|&x| c.advance(x as u32 + 1)) && c.complete()
        };
        let wrap = |inner: &[u8]| [b"{\"a\":\"".as_slice(), inner, b"\"}"].concat();
        assert!(ok(&wrap("é日𝄞".as_bytes())));
        assert!(ok(&wrap(br"\ud7ff\u0d00\uABCD")));
        for bad in [&[0xc0u8, 0x80][..], &[0xe0, 0x80, 0x80], &[0xed, 0xa0, 0x80], &[0xf4, 0x90, 0x80, 0x80], &[0xff], &[0xc3]] {
            assert!(!ok(&wrap(bad)), "{bad:?}");
        }
        assert!(!ok(&wrap(br"\udc00")));
    }

    #[test]
    fn json_any_accepts_scalars() {
        let g = Grammar::json_any(single_bytes(), 257, vec![EOS]);
        for ok in ["1", "-0.5e3", "\"x\"", "[]", "null", "{}"] {
            assert!(accepts(&g, ok), "{ok}");
        }
    }

    #[test]
    fn mask_only_allows_open_brace_first_and_stop_only_at_the_end() {
        let g = object();
        let mut c = g.cursor();
        let allowed = |m: &[u32], t: u32| m[t as usize / 32] >> (t % 32) & 1 == 1;
        let m = c.mask();
        let open: Vec<u32> = (0..257).filter(|&t| allowed(&m, t)).collect();
        let mut want: Vec<u32> = [b' ', b'\t', b'\n', b'\r', b'{'].iter().map(|&b| b as u32 + 1).collect();
        want.sort();
        assert_eq!(open, want);
        assert!(c.advance(b'{' as u32 + 1) && c.advance(b'}' as u32 + 1));
        assert!(allowed(&c.mask(), EOS));
        assert!(!allowed(&c.mask(), b' ' as u32 + 1));
        assert!(c.advance(EOS));
    }

    #[test]
    fn multi_byte_tokens_are_checked_through_the_trie() {
        let mut tokens: Vec<Vec<u8>> = vec![Vec::new()];
        tokens.extend([b"{\"".to_vec(), b"a".to_vec(), b"\":".to_vec(), b"1}".to_vec(), b"x}".to_vec(), b"{}".to_vec(), b"}".to_vec()]);
        let g = Grammar::json_object(Arc::new(TokenTable::new(tokens)), 8, vec![EOS]);
        let mut c = g.cursor();
        let m = c.mask();
        assert_eq!(m[0] & 0xff, 0b0100_0010);
        for t in [1, 2, 3, 4] {
            assert!(c.advance(t), "{t}");
        }
        assert!(c.complete());
        assert!(!g.cursor().clone().advance(7));
    }

    #[test]
    fn masks_are_cached_per_state() {
        let g = object();
        let mut c = g.cursor();
        c.mask();
        assert!(c.advance(b'{' as u32 + 1));
        c.mask();
        let n = g.cached_states();
        c.mask();
        assert_eq!(g.cached_states(), n);
    }

    #[test]
    fn byte_level_tokens_map_back_to_bytes() {
        assert_eq!(token_bytes("\u{120}a"), b" a");
        assert_eq!(token_bytes("\u{10a}"), b"\n");
        assert_eq!(token_bytes("<0x41>"), b"A");
        assert_eq!(token_bytes("\u{2581}the"), b" the");
        assert_eq!(token_bytes("{\""), b"{\"");
    }

    #[test]
    fn response_format_types() {
        use serde_json::json;
        assert_eq!(parse_response_format(&json!({"type": "json_object"})), Ok(Some(Schema::Object)));
        assert_eq!(parse_response_format(&json!({"type": "text"})), Ok(None));
        assert_eq!(parse_response_format(&Value::Null), Ok(None));
        assert!(parse_response_format(&json!({"type": "nope"})).is_err());
    }
}
