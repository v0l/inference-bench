#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Xor,
    And,
    Or,
    Shl,
    Shr,
}

impl IntOp {
    fn prec(self) -> u8 {
        match self {
            IntOp::Or => 1,
            IntOp::Xor => 2,
            IntOp::And => 3,
            IntOp::Shl | IntOp::Shr => 4,
            IntOp::Add | IntOp::Sub => 5,
            IntOp::Mul | IntOp::Div | IntOp::Rem => 6,
        }
    }

    fn text(self) -> &'static str {
        match self {
            IntOp::Add => "+",
            IntOp::Sub => "-",
            IntOp::Mul => "*",
            IntOp::Div => "/",
            IntOp::Rem => "%",
            IntOp::Xor => "^",
            IntOp::And => "&",
            IntOp::Or => "|",
            IntOp::Shl => "<<",
            IntOp::Shr => ">>",
        }
    }

    fn apply(self, a: u64, b: u64) -> Result<u64, String> {
        Ok(match self {
            IntOp::Add => a.wrapping_add(b),
            IntOp::Sub => a.wrapping_sub(b),
            IntOp::Mul => a.wrapping_mul(b),
            IntOp::Div => a.checked_div(b).ok_or("integer division by zero")?,
            IntOp::Rem => a.checked_rem(b).ok_or("integer remainder by zero")?,
            IntOp::Xor => a ^ b,
            IntOp::And => a & b,
            IntOp::Or => a | b,
            IntOp::Shl => a.wrapping_shl(b as u32),
            IntOp::Shr => a.wrapping_shr(b as u32),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntExpr {
    Lit(u64),
    Name(String),
    Bin(IntOp, Box<IntExpr>, Box<IntExpr>),
    Call(&'static str, Vec<IntExpr>),
}

const CALLS: [(&str, usize, bool); 8] = [("rem", 2, true), ("max", 2, true), ("min", 2, true), ("ge", 2, true), ("select", 3, true), ("splitmix64", 1, false), ("prime_after", 2, false), ("prime_sum", 3, false)];

pub fn splitmix64(v: u64) -> u64 {
    let v = v.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

pub fn floor_rem(a: u64, b: u64) -> Result<u64, String> {
    let (a, b) = (a as i64, b as i64);
    if b == 0 {
        return Err("rem by zero".into());
    }
    let r = a.wrapping_rem(b);
    Ok((if r != 0 && (r < 0) != (b < 0) { r.wrapping_add(b) } else { r }) as u64)
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    let mut d = 3u64;
    while d.saturating_mul(d) <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

pub fn prime_after(start: u64, count: u64) -> Result<u64, String> {
    if count > 1 << 20 {
        return Err(format!("prime_after count {count} is too large"));
    }
    let mut p = start;
    for _ in 0..count {
        p = p.checked_add(1).ok_or("prime_after overflows")?;
        while !is_prime(p) {
            p = p.checked_add(1).ok_or("prime_after overflows")?;
        }
    }
    Ok(p)
}

pub fn prime_sum(start: u64, first: u64, count: u64) -> Result<u64, String> {
    if count > 1 << 12 {
        return Err(format!("prime_sum count {count} is too large"));
    }
    let mut total = 0u64;
    for j in 0..count {
        total = total.checked_add(prime_after(start, first + j)?).ok_or("prime_sum overflows")?;
    }
    Ok(total)
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(u64),
    Id(String),
    Sym(&'static str),
}

const SYMS: [&str; 13] = ["<<", ">>", "+", "-", "*", "/", "%", "^", "&", "|", "(", ")", ","];

fn tokens(text: &str) -> Result<Vec<Tok>, String> {
    let c: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
        } else if ch.is_ascii_digit() {
            let s = i;
            while i < c.len() && (c[i].is_ascii_alphanumeric() || c[i] == '_') {
                i += 1;
            }
            let word: String = c[s..i].iter().filter(|c| **c != '_').collect();
            let v = match word.strip_prefix("0x").or_else(|| word.strip_prefix("0X")) {
                Some(hex) => u64::from_str_radix(hex, 16),
                None => word.parse::<u64>(),
            };
            out.push(Tok::Num(v.map_err(|_| format!("integer expression {text:?}: bad number {word:?}"))?));
        } else if ch.is_ascii_alphabetic() || ch == '_' {
            let s = i;
            while i < c.len() && (c[i].is_ascii_alphanumeric() || c[i] == '_') {
                i += 1;
            }
            out.push(Tok::Id(c[s..i].iter().collect()));
        } else {
            let rest: String = c[i..c.len().min(i + 2)].iter().collect();
            let sym = SYMS.iter().find(|s| rest.starts_with(**s)).ok_or_else(|| format!("integer expression {text:?}: unexpected {ch:?}"))?;
            out.push(Tok::Sym(sym));
            i += sym.len();
        }
    }
    Ok(out)
}

struct Parser<'a> {
    text: &'a str,
    toks: Vec<Tok>,
    at: usize,
}

impl Parser<'_> {
    fn err(&self, what: &str) -> String {
        format!("integer expression {:?}: {what}", self.text)
    }

    fn eat(&mut self, s: &str) -> bool {
        if matches!(self.toks.get(self.at), Some(Tok::Sym(x)) if *x == s) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn binary(&mut self, level: u8) -> Result<IntExpr, String> {
        if level > 6 {
            return self.unary();
        }
        let mut l = self.binary(level + 1)?;
        loop {
            let op = [IntOp::Or, IntOp::Xor, IntOp::And, IntOp::Shl, IntOp::Shr, IntOp::Add, IntOp::Sub, IntOp::Mul, IntOp::Div, IntOp::Rem].into_iter().find(|op| op.prec() == level && matches!(self.toks.get(self.at), Some(Tok::Sym(x)) if *x == op.text()));
            let Some(op) = op else { return Ok(l) };
            self.at += 1;
            let r = self.binary(level + 1)?;
            l = IntExpr::Bin(op, Box::new(l), Box::new(r));
        }
    }

    fn unary(&mut self) -> Result<IntExpr, String> {
        if self.eat("-") {
            let e = self.unary()?;
            return Ok(IntExpr::Bin(IntOp::Sub, Box::new(IntExpr::Lit(0)), Box::new(e)));
        }
        match self.toks.get(self.at).cloned() {
            Some(Tok::Num(n)) => {
                self.at += 1;
                Ok(IntExpr::Lit(n))
            }
            Some(Tok::Sym("(")) => {
                self.at += 1;
                let e = self.binary(1)?;
                if !self.eat(")") {
                    return Err(self.err("missing )"));
                }
                Ok(e)
            }
            Some(Tok::Id(id)) => {
                self.at += 1;
                if !self.eat("(") {
                    return Ok(IntExpr::Name(id));
                }
                let mut args = Vec::new();
                if !self.eat(")") {
                    loop {
                        args.push(self.binary(1)?);
                        if self.eat(")") {
                            break;
                        }
                        if !self.eat(",") {
                            return Err(self.err("missing , or )"));
                        }
                    }
                }
                let (name, arity, _) = CALLS.iter().find(|(n, _, _)| *n == id).ok_or_else(|| self.err(&format!("unknown function {id:?}")))?;
                if args.len() != *arity {
                    return Err(self.err(&format!("{id} takes {arity} arguments")));
                }
                Ok(IntExpr::Call(name, args))
            }
            _ => Err(self.err("unexpected end or token")),
        }
    }
}

impl IntExpr {
    pub fn parse(text: &str) -> Result<IntExpr, String> {
        let toks = tokens(text)?;
        let mut p = Parser { text, toks, at: 0 };
        let e = p.binary(1)?;
        if p.at != p.toks.len() {
            return Err(p.err("trailing input"));
        }
        Ok(e)
    }

    pub fn names(&self, out: &mut Vec<String>) {
        match self {
            IntExpr::Name(n) if !out.contains(n) => out.push(n.clone()),
            IntExpr::Bin(_, a, b) => {
                a.names(out);
                b.names(out);
            }
            IntExpr::Call(_, args) => args.iter().for_each(|a| a.names(out)),
            _ => {}
        }
    }

    pub fn eval(&self, env: &dyn Fn(&str) -> Option<u64>) -> Result<u64, String> {
        match self {
            IntExpr::Lit(v) => Ok(*v),
            IntExpr::Name(n) => env(n).ok_or_else(|| format!("integer name {n:?} has no value")),
            IntExpr::Bin(op, a, b) => op.apply(a.eval(env)?, b.eval(env)?),
            IntExpr::Call(f, args) => {
                let v = args.iter().map(|a| a.eval(env)).collect::<Result<Vec<u64>, String>>()?;
                match (*f, v.as_slice()) {
                    ("rem", [a, b]) => floor_rem(*a, *b),
                    ("max", [a, b]) => Ok(*a.max(b)),
                    ("min", [a, b]) => Ok(*a.min(b)),
                    ("ge", [a, b]) => Ok(u64::from(a >= b)),
                    ("select", [c, a, b]) => Ok(if *c != 0 { *a } else { *b }),
                    ("splitmix64", [a]) => Ok(splitmix64(*a)),
                    ("prime_after", [a, b]) => prime_after(*a, *b),
                    ("prime_sum", [a, b, c]) => prime_sum(*a, *b, *c),
                    _ => Err(format!("integer function {f} has no evaluation")),
                }
            }
        }
    }

    pub fn subst(&self, env: &dyn Fn(&str) -> Option<IntExpr>) -> IntExpr {
        match self {
            IntExpr::Name(n) => env(n).unwrap_or_else(|| self.clone()),
            IntExpr::Bin(op, a, b) => IntExpr::Bin(*op, Box::new(a.subst(env)), Box::new(b.subst(env))),
            IntExpr::Call(f, args) => IntExpr::Call(f, args.iter().map(|a| a.subst(env)).collect()),
            IntExpr::Lit(_) => self.clone(),
        }
    }

    pub fn fold(&self) -> Result<IntExpr, String> {
        let mut names = Vec::new();
        self.names(&mut names);
        if names.is_empty() {
            return Ok(IntExpr::Lit(self.eval(&|_| None)?));
        }
        Ok(match self {
            IntExpr::Bin(op, a, b) => IntExpr::Bin(*op, Box::new(a.fold()?), Box::new(b.fold()?)),
            IntExpr::Call(f, args) => IntExpr::Call(f, args.iter().map(IntExpr::fold).collect::<Result<_, _>>()?),
            e => e.clone(),
        })
    }

    pub fn lower_cuda(&self, names: &dyn Fn(&str) -> Option<String>) -> Result<String, String> {
        Ok(match self {
            IntExpr::Lit(v) => format!("{v}ull"),
            IntExpr::Name(n) => names(n).ok_or_else(|| format!("integer name {n:?} is not bound here"))?,
            IntExpr::Bin(op, a, b) => format!("({} {} {})", a.lower_cuda(names)?, op.text(), b.lower_cuda(names)?),
            IntExpr::Call(f, args) => {
                let (_, _, device) = CALLS.iter().find(|(n, _, _)| n == f).expect("parsed calls are known");
                if !device {
                    return Err(format!("{f} runs only at build time, so its arguments must be constants"));
                }
                let a = args[0].lower_cuda(names)?;
                let b = args[1].lower_cuda(names)?;
                match *f {
                    "rem" => format!("(unsigned long long)(((long long){a} % (long long){b} + (long long){b}) % (long long){b})"),
                    "max" => format!("({a} > {b} ? {a} : {b})"),
                    "ge" => format!("({a} >= {b} ? 1ull : 0ull)"),
                    "select" => format!("({a} != 0ull ? {b} : {})", args[2].lower_cuda(names)?),
                    _ => format!("({a} < {b} ? {a} : {b})"),
                }
            }
        })
    }
}
