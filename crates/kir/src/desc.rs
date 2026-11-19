use crate::attention::{AttentionKernel, AttentionShape, Family, MropeShape, RopeShape};
use crate::compose::{Address, Compose, Iter, Mat, Rows, Stage, Tile};
use crate::expr::{Const, Expr};
use crate::layout::Layout;
use crate::{Access, Activation, BlockSum, Kernel, Param, Pointwise, Role, Scalar, Target};
use std::collections::{BTreeMap, HashSet};
use std::sync::Mutex;
use toml::Value;

pub const KIND: &str = "kir-kernel";

pub const BUILTIN: [(&str, &str); 10] = [
    ("rmsnorm", include_str!("../kernels/rmsnorm.toml")),
    ("swiglu", include_str!("../kernels/swiglu.toml")),
    ("norm_add", include_str!("../kernels/norm_add.toml")),
    ("add_rmsnorm", include_str!("../kernels/add_rmsnorm.toml")),
    ("clear_branch", include_str!("../kernels/clear_branch.toml")),
    ("residual_add", include_str!("../kernels/residual_add.toml")),
    ("q_norm_rows", include_str!("../kernels/q_norm_rows.toml")),
    ("k_norm_rows", include_str!("../kernels/k_norm_rows.toml")),
    ("logit_scale", include_str!("../kernels/logit_scale.toml")),
    ("logit_softcap", include_str!("../kernels/logit_softcap.toml")),
];

pub const COMPOSED: [(&str, &str); 8] = [
    ("rmsnorm", include_str!("../kernels/compose_rmsnorm.toml")),
    ("norm_add", include_str!("../kernels/compose_norm_add.toml")),
    ("add_rmsnorm", include_str!("../kernels/compose_add_rmsnorm.toml")),
    ("swiglu", include_str!("../kernels/compose_swiglu.toml")),
    ("residual_add", include_str!("../kernels/compose_residual_add.toml")),
    ("logit_softcap", include_str!("../kernels/compose_logit_softcap.toml")),
    ("logit_scale", include_str!("../kernels/compose_logit_scale.toml")),
    ("clear_branch", include_str!("../kernels/compose_clear_branch.toml")),
];

pub const ATTENTION: [(&str, &str); 8] = [
    ("qk_norm_rope", include_str!("../kernels/qk_norm_rope.toml")),
    ("kv_norm_rope", include_str!("../kernels/kv_norm_rope.toml")),
    ("attention_split", include_str!("../kernels/attention_split.toml")),
    ("attention_reduce", include_str!("../kernels/attention_reduce.toml")),
    ("flash_attention", include_str!("../kernels/flash_attention.toml")),
    ("flash_mma", include_str!("../kernels/flash_mma.toml")),
    ("batch_decode_attention", include_str!("../kernels/batch_decode_attention.toml")),
    ("batch_decode_combine", include_str!("../kernels/batch_decode_combine.toml")),
];

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Description {
    pub kind: String,
    pub family: String,
    pub name: String,
    #[serde(default)]
    pub threads: Threads,
    #[serde(default)]
    pub args: BTreeMap<String, Value>,
    #[serde(default)]
    pub rows: Option<RowsDesc>,
    #[serde(default)]
    pub param: Vec<ParamDesc>,
    #[serde(default, rename = "const")]
    pub consts: Vec<ConstDesc>,
    #[serde(default)]
    pub stage: Vec<StageDesc>,
    #[serde(default)]
    pub shape: Option<AttentionShapeDesc>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(untagged)]
pub enum Threads {
    Fixed(u32),
    Symbol(String),
}

impl Default for Threads {
    fn default() -> Threads {
        Threads::Fixed(0)
    }
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AttentionShapeDesc {
    pub heads: Value,
    pub kv_heads: Value,
    pub head_dim: Value,
    pub qkv_row: Value,
    pub output_gate: Value,
    pub qk_norm: Value,
    pub eps: Value,
    pub rotary_dim: Value,
    pub theta: Value,
    pub kind: Value,
    pub factor: Value,
    pub low: Value,
    pub high: Value,
    pub original: Value,
    pub mscale: Value,
    pub freq_factors: Option<Value>,
    pub mrope_section: Option<Value>,
    pub mrope_interleaved: Option<Value>,
    pub kv_layers: Value,
    pub kv_planes: Value,
    pub kv_fp8: Value,
    pub kv_f16_layers: Value,
    pub query_scale: Value,
    pub softcap: Value,
    pub local_theta: Value,
    pub bidirectional: Value,
    pub nope: Option<Value>,
    pub local_rope: Option<Value>,
}

impl Default for AttentionShapeDesc {
    fn default() -> AttentionShapeDesc {
        AttentionShapeDesc {
            heads: Value::Integer(0),
            kv_heads: Value::Integer(0),
            head_dim: Value::Integer(0),
            qkv_row: Value::Integer(0),
            output_gate: Value::Boolean(false),
            qk_norm: Value::Boolean(false),
            eps: Value::Float(0.0),
            rotary_dim: Value::Integer(0),
            theta: Value::Float(10000.0),
            kind: Value::Integer(0),
            factor: Value::Float(1.0),
            low: Value::Float(1.0),
            high: Value::Float(1.0),
            original: Value::Float(1.0),
            mscale: Value::Float(1.0),
            freq_factors: None,
            mrope_section: None,
            mrope_interleaved: None,
            kv_layers: Value::Integer(0),
            kv_planes: Value::Integer(2),
            kv_fp8: Value::Boolean(false),
            kv_f16_layers: Value::Integer(0),
            query_scale: Value::Float(1.0),
            softcap: Value::Float(0.0),
            local_theta: Value::Float(10000.0),
            bidirectional: Value::Boolean(false),
            nope: None,
            local_rope: None,
        }
    }
}

impl AttentionShapeDesc {
    fn resolved(&self, shape: &Shape, name: &str) -> Result<AttentionShape, String> {
        let get = |v: &Value, key: &str| -> Result<Value, String> {
            match v {
                Value::String(s) => shape.get(s).cloned().ok_or_else(|| format!("{name}: {key:?} names {s:?}, which the model shape does not provide")),
                v => Ok(v.clone()),
            }
        };
        let num = |v: &Value, key: &str| -> Result<f64, String> {
            match get(v, key)? {
                Value::Integer(i) => Ok(i as f64),
                Value::Float(f) => Ok(f),
                v => Err(format!("{name}: {key:?} is {v}, not a number")),
            }
        };
        let count = |v: &Value, key: &str| -> Result<usize, String> {
            let n = num(v, key)?;
            if n < 0.0 || n.fract() != 0.0 {
                return Err(format!("{name}: {key:?} is {n}, not a size"));
            }
            Ok(n as usize)
        };
        let flag = |v: &Value, key: &str| -> Result<bool, String> {
            match get(v, key)? {
                Value::Boolean(b) => Ok(b),
                v => Err(format!("{name}: {key:?} is {v}, not a boolean")),
            }
        };
        let optional = |v: &Option<Value>, key: &str| -> Result<Vec<f64>, String> {
            let Some(v) = v else { return Ok(Vec::new()) };
            match shape.get(v.as_str().unwrap_or_default()) {
                None if v.as_str().is_some() => Ok(Vec::new()),
                _ => match get(v, key)? {
                    Value::Array(a) => a.iter().map(|x| num(x, key)).collect(),
                    v => Err(format!("{name}: {key:?} is {v}, not a list")),
                },
            }
        };
        let flags = |v: &Value, key: &str| -> Result<Vec<bool>, String> {
            match get(v, key)? {
                Value::Array(a) => a.iter().map(|x| flag(x, key)).collect(),
                v => Err(format!("{name}: {key:?} is {v}, not a list")),
            }
        };
        let mrope = match self.mrope_section.as_ref() {
            None => None,
            Some(v) if v.as_str().is_some() && shape.get(v.as_str().unwrap()).is_none() => None,
            Some(v) => {
                let section = match get(v, "mrope_section")? {
                    Value::Array(a) if a.len() == 3 => {
                        let mut s = [0usize; 3];
                        for (i, x) in a.iter().enumerate() {
                            s[i] = count(x, "mrope_section")?;
                        }
                        s
                    }
                    v => return Err(format!("{name}: mrope_section is {v}, not three sizes")),
                };
                let interleaved = match self.mrope_interleaved.as_ref() {
                    None => false,
                    Some(b) => flag(b, "mrope_interleaved")?,
                };
                Some(MropeShape { section, interleaved })
            }
        };
        let kind = num(&self.kind, "kind")?;
        if kind < 0.0 || kind.fract() != 0.0 || kind > u32::MAX as f64 {
            return Err(format!("{name}: kind is {kind}, not a rope kind"));
        }
        Ok(AttentionShape {
            heads: count(&self.heads, "heads")?,
            kv_heads: count(&self.kv_heads, "kv_heads")?,
            head_dim: count(&self.head_dim, "head_dim")?,
            qkv_row: count(&self.qkv_row, "qkv_row")?,
            output_gate: flag(&self.output_gate, "output_gate")?,
            qk_norm: flag(&self.qk_norm, "qk_norm")?,
            eps: num(&self.eps, "eps")?,
            rope: RopeShape {
                rotary_dim: count(&self.rotary_dim, "rotary_dim")?,
                theta: num(&self.theta, "theta")?,
                kind: kind as u32,
                factor: num(&self.factor, "factor")?,
                low: num(&self.low, "low")?,
                high: num(&self.high, "high")?,
                original: num(&self.original, "original")?,
                mscale: num(&self.mscale, "mscale")?,
                freq_factors: optional(&self.freq_factors, "freq_factors")?,
                mrope,
            },
            kv_layers: count(&self.kv_layers, "kv_layers")?,
            kv_planes: count(&self.kv_planes, "kv_planes")?,
            kv_fp8: flag(&self.kv_fp8, "kv_fp8")?,
            kv_f16_layers: count(&self.kv_f16_layers, "kv_f16_layers")?,
            query_scale: num(&self.query_scale, "query_scale")?,
            softcap: num(&self.softcap, "softcap")?,
            local_theta: num(&self.local_theta, "local_theta")?,
            bidirectional: flag(&self.bidirectional, "bidirectional")?,
            nope: match self.nope.as_ref() {
                None => Vec::new(),
                Some(v) => flags(v, "nope")?,
            },
            local_rope: match self.local_rope.as_ref() {
                None => Vec::new(),
                Some(v) => flags(v, "local_rope")?,
            },
        })
    }
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowsDesc {
    pub iter: String,
    pub address: String,
    pub per_row: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParamDesc {
    pub name: String,
    pub role: String,
    pub access: String,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstDesc {
    pub name: String,
    pub value: Value,
    #[serde(default)]
    pub wgsl_const: bool,
    #[serde(default)]
    pub fold: bool,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageDesc {
    pub op: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub src: Option<String>,
    #[serde(default)]
    pub dst: Option<String>,
    #[serde(default)]
    pub value: Option<String>,
    #[serde(default)]
    pub expr: Option<String>,
    #[serde(default)]
    pub layout: Option<String>,
    #[serde(default)]
    pub at: Vec<String>,
    #[serde(default)]
    pub materialize: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Shape(BTreeMap<String, Value>);

impl Shape {
    pub fn new() -> Shape {
        Shape::default()
    }

    pub fn num(mut self, name: &str, v: f64) -> Shape {
        self.0.insert(name.into(), Value::Float(v));
        self
    }

    pub fn word(mut self, name: &str, v: &str) -> Shape {
        self.0.insert(name.into(), Value::String(v.into()));
        self
    }

    pub fn flags(mut self, name: &str, v: &[bool]) -> Shape {
        self.0.insert(name.into(), Value::Array(v.iter().map(|b| Value::Boolean(*b)).collect()));
        self
    }

    pub fn flag(mut self, name: &str, v: bool) -> Shape {
        self.0.insert(name.into(), Value::Boolean(v));
        self
    }

    pub fn nums(mut self, name: &str, v: &[f64]) -> Shape {
        self.0.insert(name.into(), Value::Array(v.iter().map(|f| Value::Float(*f)).collect()));
        self
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.0.get(name)
    }
}

fn intern(name: &str) -> &'static str {
    static NAMES: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);
    let mut guard = NAMES.lock().unwrap();
    let set = guard.get_or_insert_with(HashSet::new);
    if let Some(n) = set.get(name) {
        return n;
    }
    let n: &'static str = Box::leak(name.to_string().into_boxed_str());
    set.insert(n);
    n
}

impl Description {
    pub fn parse(text: &str) -> Result<Description, String> {
        let d: Description = toml::from_str(text).map_err(|e| format!("kernel description: {e}"))?;
        if d.kind != KIND {
            return Err(format!("kernel description {}: kind {:?} is not {KIND:?}", d.name, d.kind));
        }
        Ok(d)
    }

    pub fn builtin(name: &str) -> Result<Description, String> {
        let (_, text) = BUILTIN.iter().find(|(n, _)| *n == name).ok_or_else(|| format!("no built in kernel description {name:?}"))?;
        Description::parse(text)
    }

    pub fn composed(name: &str) -> Result<Description, String> {
        let (_, text) = COMPOSED.iter().find(|(n, _)| *n == name).ok_or_else(|| format!("no composed kernel description {name:?}"))?;
        Description::parse(text)
    }

    pub fn attention(name: &str) -> Result<Description, String> {
        let (_, text) = ATTENTION.iter().find(|(n, _)| *n == name).ok_or_else(|| format!("no attention kernel description {name:?}"))?;
        Description::parse(text)
    }

    fn arg<'a>(&'a self, shape: &'a Shape, key: &str) -> Result<&'a Value, String> {
        let v = self.args.get(key).ok_or_else(|| format!("{}: missing arg {key:?}", self.name))?;
        match v {
            Value::String(sym) if self.is_symbol(key) => shape.get(sym).ok_or_else(|| format!("{}: arg {key:?} names {sym:?}, which the model shape does not provide", self.name)),
            v => Ok(v),
        }
    }

    fn is_symbol(&self, key: &str) -> bool {
        !matches!(key, "op" | "act")
    }

    fn number(&self, shape: &Shape, key: &str) -> Result<f64, String> {
        match self.arg(shape, key)? {
            Value::Integer(i) => Ok(*i as f64),
            Value::Float(f) => Ok(*f),
            v => Err(format!("{}: arg {key:?} is {v}, not a number", self.name)),
        }
    }

    fn count(&self, shape: &Shape, key: &str) -> Result<u32, String> {
        let v = self.number(shape, key)?;
        if v < 0.0 || v.fract() != 0.0 || v > u32::MAX as f64 {
            return Err(format!("{}: arg {key:?} is {v}, not a size", self.name));
        }
        Ok(v as u32)
    }

    fn word(&self, shape: &Shape, key: &str) -> Result<String, String> {
        match self.args.get(key) {
            Some(Value::String(s)) => Ok(match shape.get(s) {
                Some(Value::String(v)) => v.clone(),
                _ => s.clone(),
            }),
            _ => Err(format!("{}: missing word arg {key:?}", self.name)),
        }
    }

    fn known(&self, allowed: &[&str]) -> Result<(), String> {
        match self.args.keys().find(|k| !allowed.contains(&k.as_str())) {
            Some(k) => Err(format!("{}: family {} has no arg {k:?}", self.name, self.family)),
            None => Ok(()),
        }
    }

    fn constant(&self, shape: &Shape, c: &ConstDesc) -> Result<f32, String> {
        let v = match &c.value {
            Value::String(sym) => shape.get(sym).ok_or_else(|| format!("{}: const {:?} names {sym:?}, which the model shape does not provide", self.name, c.name))?,
            v => v,
        };
        match v {
            Value::Integer(i) => Ok(*i as f32),
            Value::Float(f) => Ok(*f as f32),
            v => Err(format!("{}: const {:?} is {v}, not a number", self.name, c.name)),
        }
    }

    fn words(&self, shape: &Shape) -> Result<Vec<(String, String)>, String> {
        let mut out = Vec::new();
        for k in self.args.keys().filter(|k| *k != "width") {
            let w = self.word(shape, k)?;
            let w = match w.as_str() {
                "swiglu" => "silu".to_string(),
                "geglu" => "gelu_tanh".to_string(),
                _ => w,
            };
            out.push((k.clone(), w));
        }
        Ok(out)
    }

    fn text(&self, words: &[(String, String)], text: &str) -> String {
        let mut out = text.to_string();
        for (k, w) in words {
            out = out.replace(&format!("${{{k}}}"), w);
        }
        out
    }

    fn tile(&self, width: u32, layout: &Option<String>, at: &[String]) -> Result<Option<Tile>, String> {
        let Some(text) = layout else {
            return if at.is_empty() { Ok(None) } else { Err(format!("{}: stage has `at` without a layout", self.name)) };
        };
        let bad = || format!("{}: layout {text:?} is not (shape):(stride)", self.name);
        let (shape, stride) = text.split_once(':').ok_or_else(bad)?;
        let list = |part: &str| -> Result<Vec<u64>, String> {
            let inner = part.trim().strip_prefix('(').and_then(|p| p.strip_suffix(')')).ok_or_else(bad)?;
            inner
                .split(',')
                .map(|x| match x.trim() {
                    "width" => Ok(width as u64),
                    n => n.parse::<u64>().map_err(|_| bad()),
                })
                .collect()
        };
        let layout = Layout::flat(&list(shape)?, &list(stride)?)?;
        Ok(Some(Tile { layout, at: at.to_vec() }))
    }

    fn mat(&self, m: &Option<String>) -> Result<Mat, String> {
        Ok(match m.as_deref() {
            None | Some("none") => Mat::default(),
            Some("both") => Mat { cuda: true, wgsl: true },
            Some("cuda") => Mat { cuda: true, wgsl: false },
            Some("wgsl") => Mat { cuda: false, wgsl: true },
            Some(o) => return Err(format!("{}: materialize {o:?} is not none, cuda, wgsl or both", self.name)),
        })
    }

    fn compose(&self, t: &Target, shape: &Shape) -> Result<Kernel, String> {
        let width = self.count(shape, "width")?;
        let words = self.words(shape)?;
        let rows = self.rows.as_ref().ok_or_else(|| format!("{}: compose needs [rows]", self.name))?;
        let iter = match rows.iter.as_str() {
            "block" => Iter::Block,
            "grid" => Iter::Grid,
            o => return Err(format!("{}: rows iter {o:?} is not block or grid", self.name)),
        };
        let address = match rows.address.as_str() {
            "advance" => Address::Advance,
            "base" => Address::Base,
            o => return Err(format!("{}: rows address {o:?} is not advance or base", self.name)),
        };
        let per_row = rows
            .per_row
            .iter()
            .map(|r| match r.split_once('*') {
                None => Ok((r.clone(), 1)),
                Some((n, m)) => m.parse::<u32>().map(|m| (n.to_string(), m)).map_err(|_| format!("{}: rows entry {r:?} is not name or name*N", self.name)),
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut params = Vec::new();
        for p in &self.param {
            let role = match p.role.as_str() {
                "weight" => Role::Weight,
                "activation" => Role::Activation,
                "scratch" => Role::Scratch,
                o => return Err(format!("{}: param role {o:?} is unknown", self.name)),
            };
            let access = match p.access.as_str() {
                "read" => Access::Read,
                "write" => Access::Write,
                o => return Err(format!("{}: param access {o:?} is unknown", self.name)),
            };
            params.push(Param { name: intern(&p.name), ty: Scalar::F32, role, access });
        }
        let mut consts = Vec::new();
        for c in &self.consts {
            consts.push(Const { name: c.name.clone(), value: self.constant(shape, c)?, wgsl_const: c.wgsl_const, fold: c.fold });
        }
        let need = |v: &Option<String>, what: &str, op: &str| v.clone().ok_or_else(|| format!("{}: stage {op} needs {what}", self.name));
        let mut stages = Vec::new();
        for s in &self.stage {
            let expr = |s: &StageDesc| -> Result<Expr, String> { Expr::parse(&self.text(&words, &need(&s.expr, "expr", &s.op)?)).map_err(|e| format!("{}: {e}", self.name)) };
            stages.push(match s.op.as_str() {
                "tile_load" => Stage::TileLoad { dst: need(&s.name, "name", &s.op)?, src: need(&s.src, "src", &s.op)?, tile: self.tile(width, &s.layout, &s.at)?, mat: self.mat(&s.materialize)? },
                "epilogue" => Stage::Epilogue { dst: need(&s.name, "name", &s.op)?, expr: expr(s)?, mat: self.mat(&s.materialize)? },
                "block_sum" => Stage::BlockSum { dst: need(&s.name, "name", &s.op)?, expr: expr(s)? },
                "tile_store" => Stage::TileStore { dst: need(&s.dst, "dst", &s.op)?, value: need(&s.value, "value", &s.op)?, tile: self.tile(width, &s.layout, &s.at)? },
                o => return Err(format!("{}: unknown stage op {o:?}", self.name)),
            });
        }
        let sum = stages.iter().any(|s| matches!(s, Stage::BlockSum { .. })).then(|| BlockSum::for_target(t));
        Kernel::compose(t, intern(&self.name), self.threads(shape)?, params, Compose { width, rows: Rows { iter, address, per_row }, consts, sum, stages }).map_err(|e| format!("{}: {e}", self.name))
    }

    fn threads(&self, shape: &Shape) -> Result<u32, String> {
        match &self.threads {
            Threads::Fixed(n) => Ok(*n),
            Threads::Symbol(s) => match shape.get(s) {
                Some(Value::Integer(i)) => u32::try_from(*i).map_err(|_| format!("{}: threads symbol {s:?} is negative", self.name)),
                Some(Value::Float(f)) if *f >= 0.0 && f.fract() == 0.0 => Ok(*f as u32),
                v => Err(format!("{}: threads symbol {s:?} is {v:?}, not a size", self.name)),
            },
        }
    }

    pub fn build(&self, t: &Target, shape: &Shape) -> Result<Kernel, String> {
        let name = intern(&self.name);
        let threads = self.threads(shape)?;
        if self.shape.is_some() {
            return Err(format!("{}: family {} takes no [shape]", self.name, self.family));
        }
        if self.family != "compose" && (self.rows.is_some() || !self.param.is_empty() || !self.consts.is_empty() || !self.stage.is_empty()) {
            return Err(format!("{}: only family compose takes rows, param, const and stage", self.name));
        }
        match self.family.as_str() {
            "compose" => self.compose(t, shape),
            "rmsnorm" => {
                self.known(&["width", "eps", "offset"])?;
                Kernel::rmsnorm_offset(t, self.count(shape, "width")?, self.number(shape, "eps")? as f32, self.number(shape, "offset")? as f32, threads)
            }
            "add_rmsnorm" => {
                self.known(&["width", "eps", "offset"])?;
                Kernel::add_rmsnorm(t, self.count(shape, "width")?, self.number(shape, "eps")? as f32, self.number(shape, "offset")? as f32, threads)
            }
            "norm_add" => {
                self.known(&["width", "eps", "offset"])?;
                Kernel::norm_add(t, self.count(shape, "width")?, self.number(shape, "eps")? as f32, self.number(shape, "offset")? as f32, threads)
            }
            "rmsnorm_segment" => {
                self.known(&["width", "stride", "start", "eps"])?;
                Kernel::rmsnorm_segment(
                    t,
                    name,
                    self.count(shape, "width")?,
                    self.count(shape, "stride")?,
                    self.count(shape, "start")?,
                    self.number(shape, "eps")? as f32,
                    threads,
                )
            }
            "add" => {
                self.known(&["width", "scale"])?;
                Kernel::add_scaled(t, name, self.count(shape, "width")?, self.number(shape, "scale")? as f32, threads)
            }
            "pointwise" => {
                self.known(&["width", "op", "value"])?;
                let v = self.number(shape, "value")? as f32;
                let op = match self.word(shape, "op")?.as_str() {
                    "scale" => Pointwise::Scale(v),
                    "softcap" => Pointwise::SoftCap(v),
                    "fill" => Pointwise::Fill(v),
                    o => return Err(format!("{}: unknown pointwise op {o:?}", self.name)),
                };
                Kernel::pointwise(t, name, self.count(shape, "width")?, op, threads)
            }
            "gated_activation" => {
                self.known(&["width", "act"])?;
                let act = match self.word(shape, "act")?.as_str() {
                    "silu" | "swiglu" => Activation::Silu,
                    "gelu_tanh" | "geglu" => Activation::GeluTanh,
                    a => return Err(format!("{}: unknown activation {a:?}", self.name)),
                };
                Kernel::gated_activation(t, name, self.count(shape, "width")?, act, threads)
            }
            f => Err(format!("{}: unknown kernel family {f:?}", self.name)),
        }
    }

    pub fn build_attention(&self, t: &Target, shape: &Shape) -> Result<AttentionKernel, String> {
        let Some(desc) = &self.shape else { return Err(format!("{}: family {} needs a [shape] table", self.name, self.family)) };
        if !self.args.is_empty() || self.rows.is_some() || !self.param.is_empty() || !self.consts.is_empty() || !self.stage.is_empty() || self.threads != Threads::default() {
            return Err(format!("{}: family {} takes only [shape]", self.name, self.family));
        }
        let resolved = desc.resolved(shape, &self.name)?;
        let family = match self.family.as_str() {
            "rope_kv_write" => Family::RopeKv,
            "split_decode" => Family::SplitDecode,
            "flash_scalar" => Family::FlashLegacy,
            "flash_mma" => Family::FlashMma,
            "grouped_decode" => Family::GroupedDecode,
            f => return Err(format!("{}: unknown attention family {f:?}", self.name)),
        };
        AttentionKernel::new(t, family, resolved).map_err(|e| format!("{}: {e}", self.name))
    }
}

pub fn build_builtin(name: &str, t: &Target, shape: &Shape) -> Result<Kernel, String> {
    Description::builtin(name)?.build(t, shape)
}

pub fn build_attention(name: &str, t: &Target, shape: &Shape) -> Result<AttentionKernel, String> {
    Description::attention(name)?.build_attention(t, shape)
}
