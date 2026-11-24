use crate::attention::{AttentionKernel, AttentionShape, Family, MropeShape, RopeShape};
use crate::compose::{Address, Compose, Iter, Mat, Rows, Stage, Tile};
use crate::expr::{Const, Expr};
use crate::layout::Layout;
use crate::scan::{Scan, Stage as ScanStage};
use crate::{Access, Activation, BlockSum, Kernel, Param, Pointwise, Role, Scalar, Target};
use std::collections::{BTreeMap, HashSet};
use std::sync::Mutex;
use toml::Value;

pub const KIND: &str = "kir-kernel";

pub const ROW_FAMILIES: [&str; 14] = ["span_rope", "state_write", "head_matvec", "index_topk", "layernorm_segment", "head_rms", "block_pool", "hyper_pre", "hyper_post", "hyper_expand", "hyper_mean", "pool_topk", "group_pool", "group_topk"];

pub const SEQ_FAMILIES: [&str; 5] = ["recurrence", "seq_conv", "group_norm", "token_hash", "group_map"];

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
    #[serde(default)]
    pub geometry: Option<BTreeMap<String, Value>>,
    #[serde(default)]
    pub query: Vec<BTreeMap<String, Value>>,
    #[serde(default)]
    pub blocks: Option<BTreeMap<String, Value>>,
    #[serde(default)]
    pub recurrence: Option<BTreeMap<String, Value>>,
    #[serde(default)]
    pub input: Vec<BTreeMap<String, Value>>,
    #[serde(default)]
    pub int: Vec<IntDesc>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntDesc {
    pub name: String,
    pub expr: String,
    #[serde(default)]
    pub per: Option<String>,
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
    pub qk_norm_unit: Value,
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
    pub rope_interleaved: Value,
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
            qk_norm_unit: Value::Boolean(false),
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
            rope_interleaved: Value::Boolean(false),
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
            qk_norm_unit: flag(&self.qk_norm_unit, "qk_norm_unit")?,
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
                interleaved: flag(&self.rope_interleaved, "rope_interleaved")?,
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
    #[serde(default, rename = "type")]
    pub ty: Option<String>,
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
    #[serde(default)]
    pub sink: Option<String>,
    #[serde(default)]
    pub key: Option<String>,
    #[serde(default)]
    pub row: Option<Value>,
    #[serde(default)]
    pub head: Option<Value>,
    #[serde(default)]
    pub offset: Option<Value>,
    #[serde(default)]
    pub index: Option<String>,
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

pub(crate) fn intern(name: &str) -> &'static str {
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

    fn boolean(&self, shape: &Shape, key: &str) -> Result<bool, String> {
        match self.arg(shape, key)? {
            Value::Boolean(b) => Ok(*b),
            v => Err(format!("{}: arg {key:?} is {v}, not a boolean", self.name)),
        }
    }

    fn number(&self, shape: &Shape, key: &str) -> Result<f64, String> {
        if let Some(Value::String(text)) = self.args.get(key)
            && self.is_symbol(key)
            && shape.get(text).is_none()
        {
            return self.size(shape, &Value::String(text.clone()), key).map(f64::from);
        }
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

    fn scalar(&self, p: &ParamDesc) -> Result<Scalar, String> {
        Ok(match p.ty.as_deref() {
            None | Some("f32") => Scalar::F32,
            Some("u32") => Scalar::U32,
            Some("bf16") => Scalar::Bf16,
            Some("i32") => Scalar::I32,
            Some(o) => return Err(format!("{}: param type {o:?} is not f32, u32, bf16 or i32", self.name)),
        })
    }

    fn scan(&self, t: &Target, shape: &Shape) -> Result<Kernel, String> {
        let words = self.words(shape)?;
        let mut params = Vec::new();
        for p in &self.param {
            let role = match p.role.as_str() {
                "weight" => Role::Weight,
                "activation" => Role::Activation,
                "value" => Role::Value,
                o => return Err(format!("{}: scan param role {o:?} is not weight, activation or value", self.name)),
            };
            let access = match p.access.as_str() {
                "read" => Access::Read,
                "write" => Access::Write,
                o => return Err(format!("{}: param access {o:?} is unknown", self.name)),
            };
            params.push(Param { name: intern(&p.name), ty: self.scalar(p)?, role, access });
        }
        let mut consts = Vec::new();
        for c in &self.consts {
            consts.push(Const { name: c.name.clone(), value: self.constant(shape, c)?, wgsl_const: false, fold: c.fold });
        }
        let need = |v: &Option<String>, what: &str, op: &str| v.clone().ok_or_else(|| format!("{}: stage {op} needs {what}", self.name));
        let expr = |s: &StageDesc| -> Result<Expr, String> { Expr::parse(&self.text(&words, &need(&s.expr, "expr", &s.op)?)).map_err(|e| format!("{}: {e}", self.name)) };
        let mut stages = Vec::new();
        for s in &self.stage {
            stages.push(match s.op.as_str() {
                "epilogue" => ScanStage::Epilogue { dst: need(&s.name, "name", &s.op)?, expr: expr(s)? },
                "mask" => ScanStage::Mask { expr: expr(s)? },
                "select_from" => ScanStage::Select { index: need(&s.index, "index", &s.op)?, row: self.size(shape, s.row.as_ref().ok_or_else(|| format!("{}: stage select_from needs row", self.name))?, "row")? },
                "online_softmax" => ScanStage::Softmax { value: need(&s.value, "value", &s.op)?, sink: s.sink.clone() },
                o => return Err(format!("{}: scan stage op {o:?} is not epilogue, mask, select_from or online_softmax", self.name)),
            });
        }
        Kernel::scan(t, intern(&self.name), self.threads(shape)?, params, Scan { consts, stages, geometry: self.scan_geometry(shape)? }).map_err(|e| format!("{}: {e}", self.name))
    }

    fn size(&self, shape: &Shape, v: &Value, key: &str) -> Result<u32, String> {
        let n = match v {
            Value::Integer(i) => *i as f64,
            Value::Float(f) => *f,
            Value::String(text) => {
                let e = Expr::parse(text).map_err(|e| format!("{}: {key}: {e}", self.name))?;
                let env = |n: &str| match shape.get(n) {
                    Some(Value::Integer(i)) => Some(*i as f64),
                    Some(Value::Float(f)) => Some(*f),
                    _ => None,
                };
                e.eval(&env).ok_or_else(|| format!("{}: {key} = {text:?} names a symbol the model shape does not provide", self.name))?
            }
            v => return Err(format!("{}: {key} is {v}, not a size", self.name)),
        };
        if n < 0.0 || n.fract() != 0.0 || n > u32::MAX as f64 {
            return Err(format!("{}: {key} is {n}, not a size", self.name));
        }
        Ok(n as u32)
    }

    fn sizes(&self, shape: &Shape, table: &BTreeMap<String, Value>, keys: &[&str], what: &str) -> Result<BTreeMap<String, u32>, String> {
        if let Some(k) = table.keys().find(|k| !keys.contains(&k.as_str()) && !matches!(k.as_str(), "src" | "style" | "name" | "space" | "act" | "input" | "output" | "eps" | "mix_eps" | "invert" | "stride" | "theta" | "angles")) {
            return Err(format!("{}: {what} has no key {k:?}", self.name));
        }
        let mut out = BTreeMap::new();
        for k in keys {
            let v = table.get(*k).ok_or_else(|| format!("{}: {what} needs {k:?}", self.name))?;
            out.insert(k.to_string(), self.size(shape, v, k)?);
        }
        Ok(out)
    }

    fn sizes_except(&self, shape: &Shape, table: &BTreeMap<String, Value>, keys: &[&str], words: &[&str], what: &str) -> Result<BTreeMap<String, u32>, String> {
        if let Some(k) = table.keys().find(|k| !keys.contains(&k.as_str()) && !words.contains(&k.as_str())) {
            return Err(format!("{}: {what} has no key {k:?}", self.name));
        }
        let mut out = BTreeMap::new();
        for k in keys {
            let v = table.get(*k).ok_or_else(|| format!("{}: {what} needs {k:?}", self.name))?;
            out.insert(k.to_string(), self.size(shape, v, k)?);
        }
        Ok(out)
    }

    fn scan_geometry(&self, shape: &Shape) -> Result<Option<crate::scan::Geometry>, String> {
        let Some(g) = &self.geometry else {
            return if self.query.is_empty() { Ok(None) } else { Err(format!("{}: [[query]] needs a [geometry]", self.name)) };
        };
        let s = self.sizes(shape, g, &["layers", "kv_rows", "k_width", "v_stride", "v_offset", "v_width", "out_row", "out_head", "out_offset"], "[geometry]")?;
        let mut query = Vec::new();
        for q in &self.query {
            let src = q.get("src").and_then(|v| v.as_str()).ok_or_else(|| format!("{}: [[query]] needs src", self.name))?;
            let n = self.sizes(shape, q, &["row", "head", "offset", "width"], "[[query]]")?;
            query.push(crate::scan::Segment { src: src.to_string(), row: n["row"], head: n["head"], offset: n["offset"], width: n["width"] });
        }
        let blocks = match &self.blocks {
            None => None,
            Some(b) => {
                let n = self.sizes_except(shape, b, &["layers", "topk"], &["cache", "index", "ratio"], "[blocks]")?;
                let word = |k: &str| b.get(k).and_then(|v| v.as_str()).map(str::to_string);
                Some(crate::scan::Blocks {
                    cache: word("cache").ok_or_else(|| format!("{}: [blocks] needs cache", self.name))?,
                    layers: n["layers"],
                    ratio: word("ratio").ok_or_else(|| format!("{}: [blocks] needs ratio", self.name))?,
                    topk: n["topk"],
                    index: word("index"),
                })
            }
        };
        Ok(Some(crate::scan::Geometry {
            blocks,
            query,
            layers: s["layers"],
            kv_rows: s["kv_rows"],
            k_width: s["k_width"],
            v_stride: s["v_stride"],
            v_offset: s["v_offset"],
            v_width: s["v_width"],
            out_row: s["out_row"],
            out_head: s["out_head"],
            out_offset: s["out_offset"],
        }))
    }

    fn params(&self) -> Result<Vec<Param>, String> {
        let mut params = Vec::new();
        for p in &self.param {
            let role = match p.role.as_str() {
                "weight" => Role::Weight,
                "activation" => Role::Activation,
                "value" => Role::Value,
                o => return Err(format!("{}: param role {o:?} is not weight, activation or value", self.name)),
            };
            let access = match p.access.as_str() {
                "read" => Access::Read,
                "write" => Access::Write,
                o => return Err(format!("{}: param access {o:?} is unknown", self.name)),
            };
            params.push(Param { name: intern(&p.name), ty: self.scalar(p)?, role, access });
        }
        Ok(params)
    }

    fn expr_arg(&self, words: &[(String, String)], table: &BTreeMap<String, Value>, key: &str, default: &str) -> Result<Expr, String> {
        let text = match table.get(key) {
            None => default.to_string(),
            Some(Value::String(s)) => s.clone(),
            Some(v) => return Err(format!("{}: {key} is {v}, not an expression", self.name)),
        };
        Expr::parse(&self.text(words, &text)).map_err(|e| format!("{}: {key}: {e}", self.name))
    }

    pub fn token_hash(&self, shape: &Shape) -> Result<crate::hash::TokenHash, String> {
        use crate::int::IntExpr;
        let n = self.sizes_except(shape, &self.args, &["context", "heads", "out_row", "layers", "blank"], &["index"], "[args]")?;
        let parse = |text: &str, what: &str| IntExpr::parse(text).map_err(|e| format!("{}: {what}: {e}", self.name));
        let symbol = |name: &str| -> Option<u64> {
            match shape.get(name)? {
                Value::Integer(i) if *i >= 0 => Some(*i as u64),
                Value::Float(f) if *f >= 0.0 && f.fract() == 0.0 && *f < 9.007_199_254_740_992e15 => Some(*f as u64),
                _ => None,
            }
        };
        let mut consts: Vec<(String, u64)> = Vec::new();
        let mut per_head: Vec<(String, Vec<u64>)> = Vec::new();
        let mut lets: Vec<(String, IntExpr)> = Vec::new();
        let heads = n["heads"] as u64;
        for d in &self.int {
            let taken = consts.iter().any(|(c, _)| *c == d.name) || per_head.iter().any(|(c, _)| *c == d.name) || lets.iter().any(|(c, _)| *c == d.name);
            if taken || d.name == "head" || !d.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err(format!("{}: [[int]] name {:?} is not usable", self.name, d.name));
            }
            let e = parse(&d.expr, &d.name)?.subst(&|m| consts.iter().find(|(c, _)| c == m).map(|(_, v)| IntExpr::Lit(*v)));
            let at = |what: String| format!("{}: [[int]] {}: {what}", self.name, d.name);
            match d.per.as_deref() {
                None | Some("const") => {
                    let v = e.eval(&|m| symbol(m)).map_err(at)?;
                    consts.push((d.name.clone(), v));
                }
                Some("head") => {
                    let mut table = Vec::new();
                    for h in 0..heads {
                        let env = |m: &str| if m == "head" { Some(h) } else { per_head.iter().find(|(c, _)| c == m).map(|(_, t)| t[h as usize]).or_else(|| symbol(m)) };
                        table.push(e.eval(&env).map_err(at)?);
                    }
                    per_head.push((d.name.clone(), table));
                }
                Some("row") => {
                    let local = |m: &str| m == "head" || (m.starts_with('t') && m[1..].parse::<u32>().is_ok()) || per_head.iter().any(|(c, _)| c == m) || lets.iter().any(|(c, _)| c == m);
                    let e = e.subst(&|m| if local(m) { None } else { symbol(m).map(IntExpr::Lit) });
                    lets.push((d.name.clone(), e.fold().map_err(at)?));
                }
                Some(o) => return Err(format!("{}: [[int]] {} per {o:?} is not const, head or row", self.name, d.name)),
            }
        }
        let index = match self.args.get("index") {
            Some(Value::String(s)) => parse(s, "index")?.subst(&|m| consts.iter().find(|(c, _)| c == m).map(|(_, v)| IntExpr::Lit(*v))),
            _ => return Err(format!("{}: token_hash needs an index expression", self.name)),
        };
        Ok(crate::hash::TokenHash { context: n["context"], heads: n["heads"], out_row: n["out_row"], layers: n["layers"], blank: n["blank"] as u64, per_head, lets, index: index.fold().map_err(|e| format!("{}: index: {e}", self.name))? })
    }

    fn group_map(&self, shape: &Shape, words: &[(String, String)]) -> Result<crate::group::GroupMap, String> {
        use crate::group::GroupStage;
        let n = self.sizes(shape, &self.args, &["width"], "[args]")?;
        let mut consts = Vec::new();
        for c in &self.consts {
            consts.push(Const { name: c.name.clone(), value: self.constant(shape, c)?, wgsl_const: false, fold: c.fold });
        }
        let need = |v: &Option<String>, what: &str, op: &str| v.clone().ok_or_else(|| format!("{}: stage {op} needs {what}", self.name));
        let size = |v: &Option<Value>, key: &str| -> Result<u32, String> { v.as_ref().map_or(Ok(0), |v| self.size(shape, v, key)) };
        let parse = |text: String| Expr::parse(&self.text(words, &text)).map_err(|e| format!("{}: {e}", self.name));
        let mut stages = Vec::new();
        for s in &self.stage {
            let e = || parse(need(&s.expr, "expr", &s.op)?);
            let dst = || need(&s.name, "name", &s.op);
            let at = || -> Result<(u32, u32, u32), String> { Ok((size(&s.row, "row")?, size(&s.head, "head")?, size(&s.offset, "offset")?)) };
            stages.push(match s.op.as_str() {
                "load" => {
                    let (row, group, offset) = at()?;
                    GroupStage::Load { dst: dst()?, src: need(&s.src, "src", &s.op)?, row, group, offset, index: None }
                }
                "gather" => {
                    let (row, group, offset) = at()?;
                    GroupStage::Load { dst: dst()?, src: need(&s.src, "src", &s.op)?, row, group, offset, index: Some(need(&s.index, "index", &s.op)?) }
                }
                "value" => GroupStage::Value { dst: dst()?, expr: e()? },
                "sum" => GroupStage::Sum { dst: dst()?, expr: e()? },
                "scalar" => GroupStage::Scalar { dst: dst()?, expr: e()? },
                "store" => {
                    let (row, group, offset) = at()?;
                    GroupStage::Store { dst: need(&s.dst, "dst", &s.op)?, value: parse(need(&s.value, "value", &s.op)?)?, row, group, offset }
                }
                o => return Err(format!("{}: group_map stage op {o:?} is not load, gather, value, sum, scalar or store", self.name)),
            });
        }
        Ok(crate::group::GroupMap { width: n["width"], consts, stages })
    }

    fn seq_kernel(&self, t: &Target, shape: &Shape) -> Result<Kernel, String> {
        use crate::recur::{Input, Recurrence, SeqBody, Space, Stage as RStage};
        let params = self.params()?;
        let words: Vec<(String, String)> = self.args.iter().filter_map(|(k, v)| v.as_str().filter(|s| s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')).map(|s| (k.clone(), s.to_string()))).collect();
        let fail = |e: String| if e.starts_with(&format!("{}:", self.name)) { e } else { format!("{}: {e}", self.name) };
        match self.family.as_str() {
            "seq_conv" => {
                let mut table = self.args.clone();
                table.entry("dilation".into()).or_insert(Value::Integer(1));
                let n = self.sizes(shape, &table, &["x_row", "x_offset", "channels", "taps", "dilation", "y_row", "y_offset", "layers"], "[args]")?;
                let expr = self.expr_arg(&words, &self.args, "act", "v")?;
                let body = SeqBody::Conv { x_row: n["x_row"], x_offset: n["x_offset"], channels: n["channels"], taps: n["taps"], dilation: n["dilation"], y_row: n["y_row"], y_offset: n["y_offset"], layers: n["layers"], expr };
                Kernel::seq(t, intern(&self.name), self.threads(shape)?, params, body).map_err(fail)
            }
            "token_hash" => {
                let body = SeqBody::TokenHash(self.token_hash(shape)?);
                Kernel::seq(t, intern(&self.name), self.threads(shape)?, params, body).map_err(fail)
            }
            "group_map" => {
                let body = SeqBody::GroupMap(self.group_map(shape, &words)?);
                Kernel::seq(t, intern(&self.name), self.threads(shape)?, params, body).map_err(fail)
            }
            "group_norm" => {
                let n = self.sizes(shape, &self.args, &["x_row", "x_offset", "g_row", "g_offset", "width", "w_group", "out_row"], "[args]")?;
                let eps = self.number(shape, "eps")? as f32;
                let input = self.expr_arg(&words, &self.args, "input", "x")?;
                let output = self.expr_arg(&words, &self.args, "output", "w * n")?;
                let body = SeqBody::GroupNorm { x_row: n["x_row"], x_offset: n["x_offset"], g_row: n["g_row"], g_offset: n["g_offset"], width: n["width"], w_group: n["w_group"], out_row: n["out_row"], eps, input, output };
                Kernel::seq(t, intern(&self.name), self.threads(shape)?, params, body).map_err(fail)
            }
            _ => {
                let r = self.recurrence.as_ref().ok_or_else(|| format!("{}: family recurrence needs a [recurrence] table", self.name))?;
                let state = r.get("state").and_then(|v| v.as_str()).ok_or_else(|| format!("{}: [recurrence] needs state", self.name))?.to_string();
                let mut table = r.clone();
                table.remove("state");
                let n = self.sizes(shape, &table, &["keys", "values", "ksplit", "layers"], "[recurrence]")?;
                if !n["ksplit"].is_power_of_two() || n["ksplit"] > 32 {
                    return Err(format!("{}: ksplit {} must be a power of two up to 32", self.name, n["ksplit"]));
                }
                let mut inputs = Vec::new();
                for i in &self.input {
                    let name = i.get("name").and_then(|v| v.as_str()).ok_or_else(|| format!("{}: [[input]] needs name", self.name))?;
                    let src = i.get("src").and_then(|v| v.as_str()).ok_or_else(|| format!("{}: [[input]] {name} needs src", self.name))?;
                    let space = match i.get("space").and_then(|v| v.as_str()) {
                        Some("key") => Space::Key,
                        Some("value") => Space::Value,
                        Some("scalar") => Space::Scalar,
                        o => return Err(format!("{}: [[input]] {name} space {o:?} is not key, value or scalar", self.name)),
                    };
                    let mut sized = i.clone();
                    sized.entry("per".into()).or_insert(Value::Integer(1));
                    let s = self.sizes(shape, &sized, &["row", "head", "per", "offset"], "[[input]]")?;
                    inputs.push(Input { name: name.to_string(), src: src.to_string(), space, row: s["row"], head: s["head"], per: s["per"], offset: s["offset"] });
                }
                let mut consts = Vec::new();
                for c in &self.consts {
                    consts.push(Const { name: c.name.clone(), value: self.constant(shape, c)?, wgsl_const: false, fold: c.fold });
                }
                let need = |v: &Option<String>, what: &str, op: &str| v.clone().ok_or_else(|| format!("{}: stage {op} needs {what}", self.name));
                let parse = |text: String| Expr::parse(&self.text(&words, &text)).map_err(|e| format!("{}: {e}", self.name));
                let size = |v: &Option<Value>, key: &str| -> Result<u32, String> { v.as_ref().map_or(Ok(0), |v| self.size(shape, v, key)) };
                let mut stages = Vec::new();
                for s in &self.stage {
                    let e = || parse(need(&s.expr, "expr", &s.op)?);
                    let dst = || need(&s.name, "name", &s.op);
                    stages.push(match s.op.as_str() {
                        "key" => RStage::Key { dst: dst()?, expr: e()? },
                        "key_sum" => RStage::KeySum { dst: dst()?, expr: e()? },
                        "scalar" => RStage::Scalar { dst: dst()?, expr: e()? },
                        "value" => RStage::Value { dst: dst()?, expr: e()? },
                        "decay" => RStage::Decay { expr: e()? },
                        "update" => RStage::Update { key: parse(need(&s.key, "key", &s.op)?)?, value: parse(need(&s.value, "value", &s.op)?)? },
                        "read" => RStage::Read { dst: dst()?, key: parse(need(&s.key, "key", &s.op)?)? },
                        "store" => RStage::Store { dst: need(&s.dst, "dst", &s.op)?, value: parse(need(&s.value, "value", &s.op)?)?, row: size(&s.row, "row")?, head: size(&s.head, "head")?, offset: size(&s.offset, "offset")? },
                        o => return Err(format!("{}: recurrence stage op {o:?} is not key, key_sum, scalar, value, decay, update, read or store", self.name)),
                    });
                }
                let body = Recurrence { state, keys: n["keys"], values: n["values"], ksplit: n["ksplit"], layers: n["layers"], consts, inputs, stages };
                Kernel::recurrence(t, intern(&self.name), n["values"] * n["ksplit"], params, body).map_err(fail)
            }
        }
    }

    fn row_kernel(&self, t: &Target, shape: &Shape) -> Result<Kernel, String> {
        use crate::rows::{RopeStyle, RowBody};
        let mut params = Vec::new();
        for p in &self.param {
            let role = match p.role.as_str() {
                "weight" => Role::Weight,
                "activation" => Role::Activation,
                "value" => Role::Value,
                o => return Err(format!("{}: param role {o:?} is not weight, activation or value", self.name)),
            };
            let access = match p.access.as_str() {
                "read" => Access::Read,
                "write" => Access::Write,
                o => return Err(format!("{}: param access {o:?} is unknown", self.name)),
            };
            params.push(Param { name: intern(&p.name), ty: self.scalar(p)?, role, access });
        }
        let body = match self.family.as_str() {
            "span_rope" => {
                let n = self.sizes(shape, &self.args, &["row", "head", "offset", "rot"], "[args]")?;
                let style = match self.args.get("style").and_then(|v| v.as_str()) {
                    None | Some("halves") => RopeStyle::Halves,
                    Some("pairs") => RopeStyle::Pairs,
                    Some(o) => return Err(format!("{}: rope style {o:?} is not halves or pairs", self.name)),
                };
                let invert = match self.args.get("invert") {
                    None => false,
                    Some(Value::Boolean(b)) => *b,
                    Some(o) => return Err(format!("{}: invert is {o}, not a boolean", self.name)),
                };
                let theta = match self.args.get("theta") {
                    None => None,
                    Some(_) => Some(self.number(shape, "theta")? as f32),
                };
                let angles = match self.args.get("angles") {
                    None => None,
                    Some(v) => Some(self.size(shape, v, "angles")?),
                };
                RowBody::SpanRope { row: n["row"], head: n["head"], offset: n["offset"], rot: n["rot"], style, invert, theta, angles }
            }
            "state_write" => {
                let n = self.sizes(shape, &self.args, &["src_row", "src_offset", "layers", "rows", "width"], "[args]")?;
                let stride = match self.args.get("stride") {
                    None => n["width"],
                    Some(v) => self.size(shape, v, "stride")?,
                };
                RowBody::StateWrite { src_row: n["src_row"], src_offset: n["src_offset"], layers: n["layers"], rows: n["rows"], width: n["width"], stride }
            }
            "index_topk" => {
                let n = self.sizes(shape, &self.args, &["heads", "dim", "topk", "max_keys", "q_row", "q_offset", "w_row", "w_offset", "out_row", "layers"], "[args]")?;
                RowBody::IndexTopk { heads: n["heads"], dim: n["dim"], topk: n["topk"], max_keys: n["max_keys"], q_row: n["q_row"], q_offset: n["q_offset"], w_row: n["w_row"], w_offset: n["w_offset"], out_row: n["out_row"], layers: n["layers"] }
            }
            "head_rms" => {
                let n = self.sizes(shape, &self.args, &["row", "head", "offset", "width"], "[args]")?;
                RowBody::HeadRms { row: n["row"], head: n["head"], offset: n["offset"], width: n["width"], eps: self.number(shape, "eps")? as f32 }
            }
            "block_pool" => {
                let n = self.sizes(shape, &self.args, &["dim", "rot", "layers", "raw_width", "gate_offset"], "[args]")?;
                RowBody::BlockPool { dim: n["dim"], rot: n["rot"], layers: n["layers"], raw_width: n["raw_width"], gate_offset: n["gate_offset"], eps: self.number(shape, "eps")? as f32 }
            }
            "group_pool" => {
                let n = self.sizes_except(shape, &self.args, &["dim", "rot", "layers", "ratio"], &["eps", "unit"], "[args]")?;
                RowBody::GroupPool { dim: n["dim"], rot: n["rot"], layers: n["layers"], ratio: n["ratio"], eps: self.number(shape, "eps")? as f32, unit: self.boolean(shape, "unit")? }
            }
            "group_topk" => {
                let n = self.sizes_except(shape, &self.args, &["heads", "dim", "rot", "topk", "ratio", "max_groups", "q_row", "q_offset", "layers"], &["eps", "unit"], "[args]")?;
                RowBody::GroupTopk {
                    heads: n["heads"],
                    dim: n["dim"],
                    rot: n["rot"],
                    topk: n["topk"],
                    ratio: n["ratio"],
                    max_groups: n["max_groups"],
                    q_row: n["q_row"],
                    q_offset: n["q_offset"],
                    layers: n["layers"],
                    eps: self.number(shape, "eps")? as f32,
                    unit: self.boolean(shape, "unit")?,
                }
            }
            "layernorm_segment" => {
                let n = self.sizes(shape, &self.args, &["width", "row", "head", "offset"], "[args]")?;
                RowBody::LayerNorm { width: n["width"], row: n["row"], head: n["head"], offset: n["offset"], eps: self.number(shape, "eps")? as f32 }
            }
            "hyper_pre" => {
                let n = self.sizes(shape, &self.args, &["streams", "hidden", "iters"], "[args]")?;
                RowBody::HyperPre { streams: n["streams"], hidden: n["hidden"], iters: n["iters"], norm_eps: self.number(shape, "eps")? as f32, eps: self.number(shape, "mix_eps")? as f32 }
            }
            "hyper_post" | "hyper_expand" | "hyper_mean" => {
                let n = self.sizes(shape, &self.args, &["streams", "hidden"], "[args]")?;
                let (streams, hidden) = (n["streams"], n["hidden"]);
                match self.family.as_str() {
                    "hyper_post" => RowBody::HyperPost { streams, hidden },
                    "hyper_expand" => RowBody::HyperExpand { streams, hidden },
                    _ => RowBody::HyperMean { streams, hidden },
                }
            }
            "pool_topk" => {
                let n = self.sizes(shape, &self.args, &["heads", "dim", "pool", "pick", "max_pools", "q_row", "q_offset", "w_row", "w_offset", "out_row", "layers", "tail"], "[args]")?;
                let tail = n["tail"] != 0;
                RowBody::PoolTopk {
                    heads: n["heads"],
                    dim: n["dim"],
                    pool: n["pool"],
                    pick: n["pick"],
                    max_pools: n["max_pools"],
                    q_row: n["q_row"],
                    q_offset: n["q_offset"],
                    w_row: n["w_row"],
                    w_offset: n["w_offset"],
                    out_row: n["out_row"],
                    layers: n["layers"],
                    tail,
                }
            }
            "head_matvec" => {
                let n = self.sizes(shape, &self.args, &["x_row", "x_head", "x_offset", "inner", "w_head", "w_inner", "w_out", "w_offset", "out", "y_row", "y_head", "y_offset"], "[args]")?;
                RowBody::HeadMatvec {
                    x_row: n["x_row"],
                    x_head: n["x_head"],
                    x_offset: n["x_offset"],
                    inner: n["inner"],
                    w_head: n["w_head"],
                    w_inner: n["w_inner"],
                    w_out: n["w_out"],
                    w_offset: n["w_offset"],
                    out: n["out"],
                    y_row: n["y_row"],
                    y_head: n["y_head"],
                    y_offset: n["y_offset"],
                }
            }
            f => return Err(format!("{}: unknown row family {f:?}", self.name)),
        };
        Kernel::rows(t, intern(&self.name), self.threads(shape)?, params, body).map_err(|e| format!("{}: {e}", self.name))
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
                "value" => Role::Value,
                o => return Err(format!("{}: param role {o:?} is unknown", self.name)),
            };
            let access = match p.access.as_str() {
                "read" => Access::Read,
                "write" => Access::Write,
                o => return Err(format!("{}: param access {o:?} is unknown", self.name)),
            };
            params.push(Param { name: intern(&p.name), ty: self.scalar(p)?, role, access });
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
                "gather" => Stage::Gather { dst: need(&s.name, "name", &s.op)?, src: need(&s.src, "src", &s.op)?, index: need(&s.index, "index", &s.op)? },
                "lookup" => Stage::Lookup { dst: need(&s.name, "name", &s.op)?, src: need(&s.src, "src", &s.op)?, index: need(&s.index, "index", &s.op)? },
                "scatter" => Stage::Scatter { dst: need(&s.dst, "dst", &s.op)?, value: need(&s.value, "value", &s.op)?, index: need(&s.index, "index", &s.op)? },
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
        if self.family != "compose" && self.family != "scan_attention" && !ROW_FAMILIES.contains(&self.family.as_str()) && !SEQ_FAMILIES.contains(&self.family.as_str()) && (self.rows.is_some() || !self.param.is_empty() || !self.consts.is_empty() || !self.stage.is_empty()) {
            return Err(format!("{}: only family compose takes rows, param, const and stage", self.name));
        }
        if self.family != "scan_attention" && (self.geometry.is_some() || !self.query.is_empty() || self.blocks.is_some()) {
            return Err(format!("{}: only family scan_attention takes [geometry] and [[query]]", self.name));
        }
        if self.family != "recurrence" && (self.recurrence.is_some() || !self.input.is_empty()) {
            return Err(format!("{}: only family recurrence takes [recurrence] and [[input]]", self.name));
        }
        if self.family != "token_hash" && !self.int.is_empty() {
            return Err(format!("{}: only family token_hash takes [[int]]", self.name));
        }
        match self.family.as_str() {
            "compose" => self.compose(t, shape),
            "scan_attention" => self.scan(t, shape),
            f if ROW_FAMILIES.contains(&f) => self.row_kernel(t, shape),
            f if SEQ_FAMILIES.contains(&f) => self.seq_kernel(t, shape),
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
