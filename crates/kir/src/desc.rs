use crate::{Activation, Kernel, Pointwise, Target};
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

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Description {
    pub kind: String,
    pub family: String,
    pub name: String,
    pub threads: u32,
    #[serde(default)]
    pub args: BTreeMap<String, Value>,
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

    pub fn build(&self, t: &Target, shape: &Shape) -> Result<Kernel, String> {
        let name = intern(&self.name);
        let threads = self.threads;
        match self.family.as_str() {
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
}

pub fn build_builtin(name: &str, t: &Target, shape: &Shape) -> Result<Kernel, String> {
    Description::builtin(name)?.build(t, shape)
}
