use crate::layout::Dialect;
use crate::lit;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

impl Op {
    fn prec(self) -> u8 {
        match self {
            Op::Or => 1,
            Op::And => 2,
            Op::Lt | Op::Gt | Op::Le | Op::Ge | Op::Eq | Op::Ne => 3,
            Op::Add | Op::Sub => 4,
            Op::Mul | Op::Div => 5,
        }
    }

    fn text(self) -> &'static str {
        match self {
            Op::Add => "+",
            Op::Sub => "-",
            Op::Mul => "*",
            Op::Div => "/",
            Op::Lt => "<",
            Op::Gt => ">",
            Op::Le => "<=",
            Op::Ge => ">=",
            Op::Eq => "==",
            Op::Ne => "!=",
            Op::And => "&&",
            Op::Or => "||",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Const {
    pub name: String,
    pub value: f32,
    pub wgsl_const: bool,
    pub fold: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Lit(String),
    Const(String, f32),
    Raw(String),
    Name(String),
    Width,
    Neg(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),
    Call(&'static str, Vec<Expr>),
}

const UNARY: [(&str, &str, &str); 9] = [
    ("floor", "floorf", "floor"),
    ("sin", "sinf", "sin"),
    ("cos", "cosf", "cos"),
    ("tanh", "tanhf", "tanh"),
    ("exp", "expf", "exp"),
    ("log", "logf", "log"),
    ("rsqrt", "rsqrtf", "inverseSqrt"),
    ("sqrt", "sqrtf", "sqrt"),
    ("abs", "fabsf", "abs"),
];

const MACROS: [(&str, &str); 4] = [
    ("sigmoid", "1.0 / (1.0 + exp(-x))"),
    ("silu", "x / (1.0 + exp(-x))"),
    ("gelu_tanh", "0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))"),
    ("softplus", "select(x > 20.0, x, log(1.0 + exp(x)))"),
];

const BINARY: [(&str, &str, &str); 2] = [("min", "fminf", "min"), ("max", "fmaxf", "max")];

const MODULE: [(&str, usize, &str); 2] = [("gated", 2, "gated_act"), ("mapped", 1, "mapped_act")];

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(String),
    Id(String),
    Sym(String),
}

fn tokens(text: &str) -> Result<Vec<Tok>, String> {
    let c: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
        } else if ch.is_ascii_digit() || ch == '.' {
            let s = i;
            while i < c.len() && (c[i].is_ascii_digit() || c[i] == '.') {
                i += 1;
            }
            if i < c.len() && (c[i] == 'e' || c[i] == 'E') {
                i += 1;
                if i < c.len() && (c[i] == '-' || c[i] == '+') {
                    i += 1;
                }
                while i < c.len() && c[i].is_ascii_digit() {
                    i += 1;
                }
            }
            out.push(Tok::Num(c[s..i].iter().collect()));
        } else if ch.is_ascii_alphabetic() || ch == '_' {
            let s = i;
            while i < c.len() && (c[i].is_ascii_alphanumeric() || c[i] == '_') {
                i += 1;
            }
            out.push(Tok::Id(c[s..i].iter().collect()));
        } else if ((ch == '<' || ch == '>' || ch == '=' || ch == '!') && c.get(i + 1) == Some(&'=')) || (ch == '&' && c.get(i + 1) == Some(&'&')) || (ch == '|' && c.get(i + 1) == Some(&'|')) {
            out.push(Tok::Sym(format!("{ch}{}", c[i + 1])));
            i += 2;
        } else if "<>+-*/(),".contains(ch) {
            out.push(Tok::Sym(ch.to_string()));
            i += 1;
        } else {
            return Err(format!("expression {text:?}: unexpected {ch:?}"));
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
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.at)
    }

    fn eat(&mut self, s: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Sym(x)) if x == s) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn err(&self, what: &str) -> String {
        format!("expression {:?}: {what}", self.text)
    }

    fn logic_or(&mut self) -> Result<Expr, String> {
        let mut l = self.logic_and()?;
        while self.eat("||") {
            let r = self.logic_and()?;
            l = Expr::Bin(Op::Or, Box::new(l), Box::new(r));
        }
        Ok(l)
    }

    fn logic_and(&mut self) -> Result<Expr, String> {
        let mut l = self.compare()?;
        while self.eat("&&") {
            let r = self.compare()?;
            l = Expr::Bin(Op::And, Box::new(l), Box::new(r));
        }
        Ok(l)
    }

    fn compare(&mut self) -> Result<Expr, String> {
        let l = self.sum()?;
        for (s, op) in [("<=", Op::Le), (">=", Op::Ge), ("==", Op::Eq), ("!=", Op::Ne), ("<", Op::Lt), (">", Op::Gt)] {
            if self.eat(s) {
                let r = self.sum()?;
                return Ok(Expr::Bin(op, Box::new(l), Box::new(r)));
            }
        }
        Ok(l)
    }

    fn sum(&mut self) -> Result<Expr, String> {
        let mut l = self.product()?;
        loop {
            let op = if self.eat("+") {
                Op::Add
            } else if self.eat("-") {
                Op::Sub
            } else {
                return Ok(l);
            };
            let r = self.product()?;
            l = Expr::Bin(op, Box::new(l), Box::new(r));
        }
    }

    fn product(&mut self) -> Result<Expr, String> {
        let mut l = self.unary()?;
        loop {
            let op = if self.eat("*") {
                Op::Mul
            } else if self.eat("/") {
                Op::Div
            } else {
                return Ok(l);
            };
            let r = self.unary()?;
            l = Expr::Bin(op, Box::new(l), Box::new(r));
        }
    }

    fn unary(&mut self) -> Result<Expr, String> {
        if self.eat("-") {
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        self.atom()
    }

    fn atom(&mut self) -> Result<Expr, String> {
        match self.toks.get(self.at).cloned() {
            Some(Tok::Num(n)) => {
                self.at += 1;
                n.parse::<f64>().map_err(|_| self.err(&format!("bad number {n:?}")))?;
                Ok(Expr::Lit(n))
            }
            Some(Tok::Sym(open)) if open == "(" => {
                self.at += 1;
                let e = self.logic_or()?;
                if !self.eat(")") {
                    return Err(self.err("missing )"));
                }
                Ok(e)
            }
            Some(Tok::Id(id)) => {
                self.at += 1;
                if self.eat("(") {
                    let mut args = Vec::new();
                    if !self.eat(")") {
                        loop {
                            args.push(self.logic_or()?);
                            if self.eat(")") {
                                break;
                            }
                            if !self.eat(",") {
                                return Err(self.err("missing , or )"));
                            }
                        }
                    }
                    return self.call(&id, args);
                }
                Ok(if id == "width" { Expr::Width } else { Expr::Name(id) })
            }
            _ => Err(self.err("unexpected end or token")),
        }
    }

    fn call(&self, id: &str, mut args: Vec<Expr>) -> Result<Expr, String> {
        let want = |n: usize| if args.len() == n { Ok(()) } else { Err(self.err(&format!("{id} takes {n} arguments"))) };
        if let Some((_, body)) = MACROS.iter().find(|(n, _)| *n == id) {
            want(1)?;
            let x = args.remove(0);
            let e = Expr::parse(body)?;
            return e.subst(&|n| (n == "x").then(|| x.clone()));
        }
        if id == "select" {
            want(3)?;
            return Ok(Expr::Call("select", args));
        }
        if id == "probit" {
            want(1)?;
            return Ok(Expr::Call("probit", args));
        }
        if let Some((n, _, _)) = UNARY.iter().find(|(n, _, _)| *n == id && *n != "neg") {
            want(1)?;
            return Ok(Expr::Call(n, args));
        }
        if let Some((n, arity, _)) = BINARY.iter().map(|(n, _, _)| (n, 2, "")).chain(MODULE.iter().map(|(n, a, c)| (n, *a, *c))).find(|(n, _, _)| **n == id) {
            want(arity)?;
            return Ok(Expr::Call(n, args));
        }
        Err(self.err(&format!("unknown function {id:?}")))
    }
}

impl Expr {
    pub fn parse(text: &str) -> Result<Expr, String> {
        let toks = tokens(text)?;
        let mut p = Parser { text, toks, at: 0 };
        let e = p.logic_or()?;
        if p.at != p.toks.len() {
            return Err(p.err("trailing input"));
        }
        Ok(e)
    }

    pub fn names(&self, out: &mut Vec<String>) {
        match self {
            Expr::Name(n) => {
                if !out.contains(n) {
                    out.push(n.clone());
                }
            }
            Expr::Lit(_) | Expr::Const(..) | Expr::Raw(_) | Expr::Width => {}
            Expr::Neg(e) => e.names(out),
            Expr::Bin(_, a, b) => {
                a.names(out);
                b.names(out);
            }
            Expr::Call(_, args) => args.iter().for_each(|a| a.names(out)),
        }
    }

    pub fn eval(&self, env: &dyn Fn(&str) -> Option<f64>) -> Option<f64> {
        Some(match self {
            Expr::Lit(t) => t.parse().ok()?,
            Expr::Const(_, v) => f64::from(*v),
            Expr::Name(n) => env(n)?,
            Expr::Neg(e) => -e.eval(env)?,
            Expr::Bin(op, a, b) => {
                let (a, b) = (a.eval(env)?, b.eval(env)?);
                let truth = |c: bool| if c { 1.0 } else { 0.0 };
                match op {
                    Op::Add => a + b,
                    Op::Sub => a - b,
                    Op::Mul => a * b,
                    Op::Div => a / b,
                    Op::Lt => truth(a < b),
                    Op::Gt => truth(a > b),
                    Op::Le => truth(a <= b),
                    Op::Ge => truth(a >= b),
                    Op::Eq => truth(a == b),
                    Op::Ne => truth(a != b),
                    Op::And => truth(a != 0.0 && b != 0.0),
                    Op::Or => truth(a != 0.0 || b != 0.0),
                }
            }
            Expr::Call("select", args) if args.len() == 3 => {
                return if args[0].eval(env)? != 0.0 { args[1].eval(env) } else { args[2].eval(env) };
            }
            Expr::Call(f, args) => {
                let v: Vec<f64> = args.iter().map(|a| a.eval(env)).collect::<Option<_>>()?;
                match (*f, v.as_slice()) {
                    ("select", [c, a, b]) => if *c != 0.0 { *a } else { *b },
                    ("min", [a, b]) => a.min(*b),
                    ("max", [a, b]) => a.max(*b),
                    ("floor", [a]) => a.floor(),
                    ("sin", [a]) => a.sin(),
                    ("cos", [a]) => a.cos(),
                    ("tanh", [a]) => a.tanh(),
                    ("exp", [a]) => a.exp(),
                    ("log", [a]) => a.ln(),
                    ("rsqrt", [a]) => 1.0 / a.sqrt(),
                    ("sqrt", [a]) => a.sqrt(),
                    ("abs", [a]) => a.abs(),
                    ("probit", [p]) => probit(*p)?,
                    _ => return None,
                }
            }
            Expr::Raw(_) | Expr::Width => return None,
        })
    }

    pub fn count(&self, name: &str) -> usize {
        match self {
            Expr::Name(n) => usize::from(n == name),
            Expr::Lit(_) | Expr::Const(..) | Expr::Raw(_) | Expr::Width => 0,
            Expr::Neg(e) => e.count(name),
            Expr::Bin(_, a, b) => a.count(name) + b.count(name),
            Expr::Call(_, args) => args.iter().map(|a| a.count(name)).sum(),
        }
    }

    pub fn bind(self, consts: &[Const]) -> Expr {
        match self {
            Expr::Name(n) => match consts.iter().find(|c| c.name == n) {
                Some(c) => Expr::Const(c.name.clone(), c.value),
                None => Expr::Name(n),
            },
            Expr::Neg(e) => Expr::Neg(Box::new(e.bind(consts))),
            Expr::Bin(op, a, b) => fold(op, a.bind(consts), b.bind(consts), consts),
            Expr::Call(f, args) => Expr::Call(f, args.into_iter().map(|a| a.bind(consts)).collect()),
            e => e,
        }
    }

    pub fn subst(&self, env: &dyn Fn(&str) -> Option<Expr>) -> Result<Expr, String> {
        Ok(match self {
            Expr::Name(n) => env(n).ok_or_else(|| format!("name {n:?} is not defined here"))?,
            Expr::Neg(e) => Expr::Neg(Box::new(e.subst(env)?)),
            Expr::Bin(op, a, b) => Expr::Bin(*op, Box::new(a.subst(env)?), Box::new(b.subst(env)?)),
            Expr::Call(f, args) => Expr::Call(f, args.iter().map(|a| a.subst(env)).collect::<Result<_, _>>()?),
            e => e.clone(),
        })
    }

    fn prec(&self) -> u8 {
        match self {
            Expr::Bin(op, ..) => op.prec(),
            Expr::Neg(_) => 6,
            _ => 7,
        }
    }

    pub fn lower(&self, d: Dialect, wgsl_consts: &[String]) -> Result<String, String> {
        let child = |e: &Expr, paren: bool| -> Result<String, String> {
            let s = e.lower(d, wgsl_consts)?;
            Ok(if paren { format!("({s})") } else { s })
        };
        Ok(match self {
            Expr::Lit(t) => {
                let t = if t.contains(['.', 'e', 'E']) { t.clone() } else { format!("{t}.0") };
                if d == Dialect::Cuda { format!("{t}f") } else { t }
            }
            Expr::Const(n, v) => match d {
                Dialect::Cuda => format!("{}f", lit(*v)),
                Dialect::Wgsl if wgsl_consts.contains(n) => n.to_uppercase(),
                Dialect::Wgsl => lit(*v),
            },
            Expr::Name(n) => return Err(format!("name {n:?} is not bound")),
            Expr::Raw(t) => t.clone(),
            Expr::Width => match d {
                Dialect::Cuda => "WIDTH".into(),
                Dialect::Wgsl => "f32(WIDTH)".into(),
            },
            Expr::Neg(e) => format!("-{}", child(e, e.prec() < 6 || matches!(**e, Expr::Neg(_)))?),
            Expr::Bin(op, a, b) => format!("{} {} {}", child(a, a.prec() < op.prec())?, op.text(), child(b, b.prec() <= op.prec())?),
            Expr::Call("select", args) => {
                let (c, a, b) = (child(&args[0], false)?, child(&args[1], false)?, child(&args[2], false)?);
                match d {
                    Dialect::Cuda => format!("({c} ? {a} : {b})"),
                    Dialect::Wgsl => format!("select({b}, {a}, {c})"),
                }
            }
            Expr::Call(f, args) if MODULE.iter().any(|(n, _, _)| n == f) => {
                let (_, _, cu) = MODULE.iter().find(|(n, _, _)| n == f).expect("checked above");
                if d != Dialect::Cuda {
                    return Err(format!("{f} is a CUDA module function with no WGSL lowering"));
                }
                let parts: Result<Vec<String>, String> = args.iter().map(|a| child(a, false)).collect();
                format!("{cu}({})", parts?.join(", "))
            }
            Expr::Call("probit", _) => return Err("probit is evaluated when the spec is read and has no device lowering".into()),
            Expr::Call(f, args) => {
                let (_, cu, wg) = UNARY.iter().chain(BINARY.iter()).find(|(n, _, _)| n == f).ok_or_else(|| format!("unknown function {f}"))?;
                let parts: Result<Vec<String>, String> = args.iter().map(|a| child(a, false)).collect();
                format!("{}({})", if d == Dialect::Cuda { cu } else { wg }, parts?.join(", "))
            }
        })
    }
}

pub fn probit(p: f64) -> Option<f64> {
    if !(p > 0.0 && p < 1.0) {
        return None;
    }
    const A: [f64; 6] = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
    const B: [f64; 5] = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const C: [f64; 6] = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const D: [f64; 4] = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    let tail = |q: f64| (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    Some(if p < 0.02425 {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - 0.02425 {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    })
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let child = |e: &Expr, paren: bool| if paren { format!("({e})") } else { e.to_string() };
        match self {
            Expr::Lit(t) | Expr::Raw(t) | Expr::Name(t) => f.write_str(t),
            Expr::Const(n, _) => f.write_str(n),
            Expr::Width => f.write_str("width"),
            Expr::Neg(e) => write!(f, "-{}", child(e, e.prec() < 6)),
            Expr::Bin(op, a, b) => write!(f, "{} {} {}", child(a, a.prec() < op.prec()), op.text(), child(b, b.prec() <= op.prec())),
            Expr::Call(name, args) => write!(f, "{name}({})", args.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", ")),
        }
    }
}

fn fold(op: Op, a: Expr, b: Expr, consts: &[Const]) -> Expr {
    let is = |e: &Expr, v: f32| matches!(e, Expr::Const(n, c) if *c == v && consts.iter().any(|k| k.name == *n && k.fold));
    match op {
        Op::Add if is(&b, 0.0) => a,
        Op::Add if is(&a, 0.0) => b,
        Op::Sub if is(&b, 0.0) => a,
        Op::Mul if is(&a, 1.0) => b,
        Op::Mul if is(&b, 1.0) => a,
        Op::Div if is(&b, 1.0) => a,
        op => Expr::Bin(op, Box::new(a), Box::new(b)),
    }
}
