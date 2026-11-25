use anyhow::{Result, bail};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Num(f64),
    List(Vec<f64>),
    Text(String),
}

impl Value {
    pub fn num(&self) -> Result<f64> {
        match self {
            Value::Num(n) => Ok(*n),
            v => bail!("{v:?} is not a number"),
        }
    }

    pub fn from_json(v: &serde_json::Value) -> Option<Value> {
        match v {
            serde_json::Value::Number(n) => n.as_f64().map(Value::Num),
            serde_json::Value::Bool(b) => Some(Value::Num(f64::from(u8::from(*b)))),
            serde_json::Value::Array(a) => a.iter().map(|x| x.as_f64().or_else(|| x.as_bool().map(|b| f64::from(u8::from(b))))).collect::<Option<Vec<f64>>>().map(Value::List),
            serde_json::Value::String(s) => Some(Value::Text(s.clone())),
            _ => None,
        }
    }
}

pub type Env = BTreeMap<String, Value>;

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Id(String),
    Sym(&'static str),
}

const SYMS: [&str; 16] = ["==", "!=", "<=", ">=", "&&", "||", "<", ">", "+", "-", "*", "/", "(", ")", "[", "]"];

fn tokens(text: &str) -> Result<Vec<Tok>> {
    let c: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        if c[i].is_whitespace() {
            i += 1;
        } else if c[i].is_ascii_digit() || c[i] == '.' {
            let s = i;
            while i < c.len() && (c[i].is_ascii_digit() || c[i] == '.' || c[i] == 'e' || ((c[i] == '-' || c[i] == '+') && i > s && c[i - 1] == 'e')) {
                i += 1;
            }
            let t: String = c[s..i].iter().collect();
            out.push(Tok::Num(t.parse().map_err(|_| anyhow::anyhow!("bad number {t:?} in {text:?}"))?));
        } else if c[i].is_ascii_alphabetic() || c[i] == '_' {
            let s = i;
            while i < c.len() && (c[i].is_ascii_alphanumeric() || c[i] == '_') {
                i += 1;
            }
            out.push(Tok::Id(c[s..i].iter().collect()));
        } else if c[i] == ',' {
            out.push(Tok::Sym(","));
            i += 1;
        } else {
            let rest: String = c[i..c.len().min(i + 2)].iter().collect();
            let Some(sym) = SYMS.iter().find(|s| rest.starts_with(**s)) else { bail!("unexpected {:?} in {text:?}", c[i]) };
            out.push(Tok::Sym(sym));
            i += sym.len();
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, PartialEq)]
enum Node {
    Num(f64),
    Name(String),
    Index(Box<Node>, Box<Node>),
    Neg(Box<Node>),
    Bin(&'static str, Box<Node>, Box<Node>),
    Call(String, Vec<Node>),
}

struct Parser {
    toks: Vec<Tok>,
    at: usize,
}

impl Parser {
    fn eat(&mut self, s: &str) -> bool {
        if matches!(self.toks.get(self.at), Some(Tok::Sym(x)) if *x == s) {
            self.at += 1;
            return true;
        }
        false
    }

    fn binary(&mut self, level: usize) -> Result<Node> {
        const LEVELS: [&[&str]; 5] = [&["||"], &["&&"], &["==", "!=", "<=", ">=", "<", ">"], &["+", "-"], &["*", "/"]];
        if level == LEVELS.len() {
            return self.unary();
        }
        let mut l = self.binary(level + 1)?;
        loop {
            let Some(op) = LEVELS[level].iter().find(|op| matches!(self.toks.get(self.at), Some(Tok::Sym(x)) if x == *op)) else { return Ok(l) };
            self.at += 1;
            let r = self.binary(level + 1)?;
            l = Node::Bin(op, Box::new(l), Box::new(r));
        }
    }

    fn unary(&mut self) -> Result<Node> {
        if self.eat("-") {
            return Ok(Node::Neg(Box::new(self.unary()?)));
        }
        let mut n = self.atom()?;
        while self.eat("[") {
            let i = self.binary(0)?;
            if !self.eat("]") {
                bail!("missing ]");
            }
            n = Node::Index(Box::new(n), Box::new(i));
        }
        Ok(n)
    }

    fn atom(&mut self) -> Result<Node> {
        let tok = self.toks.get(self.at).cloned();
        self.at += 1;
        match tok {
            Some(Tok::Num(n)) => Ok(Node::Num(n)),
            Some(Tok::Sym("(")) => {
                let e = self.binary(0)?;
                if !self.eat(")") {
                    bail!("missing )");
                }
                Ok(e)
            }
            Some(Tok::Id(id)) if self.eat("(") => {
                let mut args = Vec::new();
                if !self.eat(")") {
                    loop {
                        args.push(self.binary(0)?);
                        if self.eat(")") {
                            break;
                        }
                        if !self.eat(",") {
                            bail!("missing , or ) after an argument of {id}");
                        }
                    }
                }
                Ok(Node::Call(id, args))
            }
            Some(Tok::Id(id)) => Ok(Node::Name(id)),
            t => bail!("unexpected {t:?}"),
        }
    }
}

fn parse(text: &str) -> Result<Node> {
    let mut p = Parser { toks: tokens(text)?, at: 0 };
    let n = p.binary(0)?;
    if p.at != p.toks.len() {
        bail!("trailing input in {text:?}");
    }
    Ok(n)
}

fn eval_node(n: &Node, env: &Env) -> Result<Value> {
    let num = |n: &Node| eval_node(n, env)?.num();
    Ok(match n {
        Node::Num(v) => Value::Num(*v),
        Node::Name(name) => match name.as_str() {
            "true" => Value::Num(1.0),
            "false" => Value::Num(0.0),
            _ => env.get(name).cloned().ok_or_else(|| anyhow::anyhow!("{name} is not defined"))?,
        },
        Node::Index(list, i) => {
            let Value::List(items) = eval_node(list, env)? else { bail!("{list:?} is not a list") };
            let i = num(i)?;
            if i < 0.0 || i as usize >= items.len() || i.fract() != 0.0 {
                bail!("index {i} is outside a list of {}", items.len());
            }
            Value::Num(items[i as usize])
        }
        Node::Neg(e) => Value::Num(-num(e)?),
        Node::Bin(op, a, b) => {
            let (a, b) = (num(a)?, num(b)?);
            let truth = |c: bool| f64::from(u8::from(c));
            Value::Num(match *op {
                "+" => a + b,
                "-" => a - b,
                "*" => a * b,
                "/" => a / b,
                "==" => truth(a == b),
                "!=" => truth(a != b),
                "<" => truth(a < b),
                ">" => truth(a > b),
                "<=" => truth(a <= b),
                ">=" => truth(a >= b),
                "&&" => truth(a != 0.0 && b != 0.0),
                _ => truth(a != 0.0 || b != 0.0),
            })
        }
        Node::Call(f, args) => match (f.as_str(), args.as_slice()) {
            ("select", [c, a, b]) => {
                if num(c)? != 0.0 {
                    eval_node(a, env)?
                } else {
                    eval_node(b, env)?
                }
            }
            ("len", [l]) => match eval_node(l, env)? {
                Value::List(items) => Value::Num(items.len() as f64),
                v => bail!("len of {v:?}"),
            },
            ("min", [a, b]) => Value::Num(num(a)?.min(num(b)?)),
            ("max", [a, b]) => Value::Num(num(a)?.max(num(b)?)),
            ("pow", [a, b]) => Value::Num(num(a)?.powf(num(b)?)),
            _ => bail!("unknown function {f} with {} arguments", args.len()),
        },
    })
}

pub fn eval(text: &str, env: &Env) -> Result<Value> {
    eval_node(&parse(text)?, env).map_err(|e| anyhow::anyhow!("{text:?}: {e}"))
}

pub fn number(text: &str, env: &Env) -> Result<f64> {
    eval(text, env)?.num().map_err(|e| anyhow::anyhow!("{text:?}: {e}"))
}

pub fn count(text: &str, env: &Env) -> Result<usize> {
    let v = number(text, env)?;
    if v < 0.0 || v.fract() != 0.0 {
        bail!("{text:?} is {v}, not a whole count");
    }
    Ok(v as usize)
}

pub fn interpolate(template: &str, env: &Env) -> Result<String> {
    let mut out = String::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let Some(close) = rest[open..].find('}') else { bail!("unclosed {{ in {template:?}") };
        let name = &rest[open + 1..open + close];
        match env.get(name) {
            Some(Value::Text(s)) => out.push_str(s),
            Some(Value::Num(n)) if n.fract() == 0.0 => out.push_str(&format!("{}", *n as i64)),
            v => bail!("{template:?}: {name} is {v:?}, not a name or a whole number"),
        }
        rest = &rest[open + close + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> Env {
        let mut e = Env::new();
        e.insert("dims".into(), Value::List(vec![1.0, 2.0, 4.0]));
        e.insert("n".into(), Value::Num(3.0));
        e.insert("prefix".into(), Value::Text("decoder.up".into()));
        e
    }

    #[test]
    fn expressions_index_lists_and_select_is_lazy() {
        let e = env();
        assert_eq!(number("dims[n - 1] * 8", &e).unwrap(), 32.0);
        assert_eq!(number("len(dims) + 1", &e).unwrap(), 4.0);
        assert_eq!(number("select(n == 3, 7, dims[9])", &e).unwrap(), 7.0);
        assert_eq!(number("-2 * (n - 1) >= -4 && 1", &e).unwrap(), 1.0);
        assert_eq!(number("0.000001", &e).unwrap(), 1e-6);
        assert!(number("dims[3]", &e).is_err());
        assert!(number("missing + 1", &e).is_err());
    }

    #[test]
    fn templates_fill_names_and_counts() {
        let mut e = env();
        e.insert("j".into(), Value::Num(1.0));
        assert_eq!(interpolate("{prefix}.resnets.{j}.conv1", &e).unwrap(), "decoder.up.resnets.1.conv1");
        assert!(interpolate("{dims}", &e).is_err());
    }
}
