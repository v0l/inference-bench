use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub leaf: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl TensorInfo {
    pub fn shape_text(&self) -> String {
        let dims: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        format!("{} {}", self.dtype, dims.join(" x "))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub path: String,
    pub instances: usize,
    pub tensors: Vec<TensorInfo>,
}

impl Group {
    pub fn name(&self) -> &str {
        self.path.rsplit('.').next().unwrap_or(&self.path)
    }

    pub fn module(&self) -> &str {
        self.path.split('.').next().unwrap_or(&self.path)
    }

    pub fn tensor(&self, leaf: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.leaf == leaf)
    }

    pub fn weight(&self) -> Option<&TensorInfo> {
        self.tensor("weight").or_else(|| self.tensor("weight_packed")).or_else(|| self.tensors.first())
    }

    pub fn is_norm(&self) -> bool {
        self.name().contains("norm")
    }

    pub fn is_matrix(&self) -> bool {
        self.weight().is_some_and(|w| w.shape.len() == 2 && w.shape.iter().all(|d| *d > 1))
    }

    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.iter().map(|t| format!("{}.{}", self.path, t.leaf)).collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub index: usize,
    pub groups: Vec<Group>,
}

impl Layer {
    pub fn group(&self, path: &str) -> Option<&Group> {
        self.groups.iter().find(|g| g.path == path)
    }

    pub fn modules(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        for g in &self.groups {
            if !out.contains(&g.module()) {
                out.push(g.module());
            }
        }
        out
    }

    pub fn in_module<'a>(&'a self, module: &'a str) -> impl Iterator<Item = &'a Group> {
        self.groups.iter().filter(move |g| g.module() == module && g.path.contains('.'))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stack {
    pub prefix: String,
    pub layers: Vec<Layer>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Structure {
    pub stacks: Vec<Stack>,
    pub outside: Vec<Group>,
    pub tensors: usize,
}

type Raw = (String, String, Vec<usize>);

fn layer_split(name: &str) -> Option<(String, usize, String)> {
    let parts: Vec<&str> = name.split('.').collect();
    for i in 0..parts.len().saturating_sub(2) {
        if matches!(parts[i], "layers" | "blk" | "h" | "blocks") {
            if let Ok(index) = parts[i + 1].parse::<usize>() {
                let prefix = parts[..=i].join(".");
                return Some((prefix, index, parts[i + 2..].join(".")));
            }
        }
    }
    None
}

fn group_tensors(tensors: &[(String, String, Vec<usize>)]) -> Vec<Group> {
    struct Acc {
        indexes: Vec<Vec<String>>,
        tensors: Vec<TensorInfo>,
    }
    let mut order: Vec<String> = Vec::new();
    let mut by_path: BTreeMap<String, Acc> = BTreeMap::new();
    for (name, dtype, shape) in tensors {
        let parts: Vec<&str> = name.split('.').collect();
        let (leaf, module) = parts.split_last().map(|(l, m)| (*l, m)).unwrap_or((name.as_str(), &[][..]));
        let mut numbers = Vec::new();
        let collapsed: Vec<&str> = module
            .iter()
            .map(|p| {
                if p.parse::<usize>().is_ok() {
                    numbers.push(p.to_string());
                    "*"
                } else {
                    *p
                }
            })
            .collect();
        let path = if collapsed.is_empty() { leaf.to_string() } else { collapsed.join(".") };
        let leaf = if collapsed.is_empty() { "weight" } else { leaf };
        if !by_path.contains_key(&path) {
            order.push(path.clone());
        }
        let acc = by_path.entry(path).or_insert(Acc { indexes: Vec::new(), tensors: Vec::new() });
        if !acc.indexes.contains(&numbers) {
            acc.indexes.push(numbers);
        }
        if !acc.tensors.iter().any(|t| t.leaf == leaf) {
            acc.tensors.push(TensorInfo { leaf: leaf.to_string(), dtype: dtype.clone(), shape: shape.clone() });
        }
    }
    order
        .into_iter()
        .map(|path| {
            let acc = by_path.remove(&path).unwrap();
            Group { path, instances: acc.indexes.len(), tensors: acc.tensors }
        })
        .collect()
}

impl Structure {
    pub fn from_tensors(tensors: impl IntoIterator<Item = Raw>) -> Self {
        let mut all: Vec<Raw> = tensors.into_iter().collect();
        all.sort_by(|a, b| a.0.cmp(&b.0));
        let tensors = all.len();
        let mut stacks: BTreeMap<String, BTreeMap<usize, Vec<Raw>>> = BTreeMap::new();
        let mut outside: Vec<Raw> = Vec::new();
        for (name, dtype, shape) in all {
            match layer_split(&name) {
                Some((prefix, index, rest)) => stacks.entry(prefix).or_default().entry(index).or_default().push((rest, dtype, shape)),
                None => outside.push((name, dtype, shape)),
            }
        }
        let stacks = stacks
            .into_iter()
            .map(|(prefix, layers)| Stack {
                prefix,
                layers: layers.into_iter().map(|(index, t)| Layer { index, groups: group_tensors(&t) }).collect(),
            })
            .collect();
        Structure { stacks, outside: group_tensors(&outside), tensors }
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let rows: Vec<(String, String, Vec<usize>)> = serde_json::from_str(text)?;
        Ok(Self::from_tensors(rows))
    }

    pub fn read(files: &[PathBuf]) -> Result<Self> {
        let mut raw = Vec::new();
        for f in files {
            raw.extend(read_header(f).with_context(|| f.display().to_string())?);
        }
        if raw.is_empty() {
            bail!("no tensors in the safetensors headers");
        }
        Ok(Self::from_tensors(raw))
    }

    pub fn read_dir(dir: &Path) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| dir.display().to_string())?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        files.sort();
        Self::read(&files)
    }

    pub fn main(&self) -> Option<&Stack> {
        self.stacks.iter().max_by_key(|s| s.layers.len())
    }

    pub fn layer(&self, index: usize) -> Option<&Layer> {
        self.main()?.layers.iter().find(|l| l.index == index)
    }

    pub fn any_leaf(&self, group_name: &str, leaf: &str) -> bool {
        self.main().is_some_and(|s| s.layers.iter().any(|l| l.groups.iter().any(|g| g.name() == group_name && g.tensor(leaf).is_some())))
    }

    pub fn layer_has_module(&self, index: usize, module: &str) -> bool {
        self.layer(index).is_some_and(|l| l.modules().contains(&module))
    }
}

pub fn read_header(path: &Path) -> Result<Vec<Raw>> {
    let mut file = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len) as usize;
    if len > 1 << 28 {
        bail!("safetensors header of {len} bytes is not plausible");
    }
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf)?;
    let value: serde_json::Value = serde_json::from_slice(&buf)?;
    let map = value.as_object().context("safetensors header is not an object")?;
    let mut out = Vec::new();
    for (name, entry) in map {
        if name == "__metadata__" {
            continue;
        }
        let dtype = entry["dtype"].as_str().context("tensor without dtype")?.to_string();
        let shape = entry["shape"]
            .as_array()
            .context("tensor without shape")?
            .iter()
            .map(|d| d.as_u64().unwrap_or(0) as usize)
            .collect();
        out.push((name.clone(), dtype, shape));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(name: &str) -> Structure {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/headers").join(format!("{name}.json"));
        Structure::from_json(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn paths(l: &Layer) -> Vec<&str> {
        l.groups.iter().map(|g| g.path.as_str()).collect()
    }

    #[test]
    fn a_dense_attention_checkpoint_groups_into_its_modules() {
        let s = header("qwen3-0.6b");
        let main = s.main().unwrap();
        assert_eq!(main.prefix, "model.layers");
        assert_eq!(main.layers.len(), 28);
        let l = s.layer(0).unwrap();
        for p in [
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            assert!(paths(l).contains(&p), "{p} in {:?}", paths(l));
        }
        assert_eq!(l.group("self_attn.q_proj").unwrap().weight().unwrap().shape, vec![2048, 1024]);
        assert_eq!(l.group("self_attn.k_proj").unwrap().weight().unwrap().shape, vec![1024, 1024]);
        assert!(l.group("self_attn.q_proj").unwrap().tensor("bias").is_none());
        assert!(s.any_leaf("q_norm", "weight"));
        assert!(!s.any_leaf("q_proj", "bias"));
        assert!(s.outside.iter().any(|g| g.path.contains("embed_tokens")));
    }

    #[test]
    fn a_hybrid_checkpoint_has_two_layer_shapes_told_apart_by_their_tensors() {
        let s = header("qwen3.8-27b-fp8");
        let delta = s.layer(0).unwrap();
        let full = s.layer(3).unwrap();
        assert_eq!(delta.modules(), vec!["input_layernorm", "linear_attn", "mlp", "post_attention_layernorm"]);
        assert!(full.modules().contains(&"self_attn"));
        assert!(!full.modules().contains(&"linear_attn"));
        let qkv = delta.group("linear_attn.in_proj_qkv").unwrap();
        assert_eq!(qkv.weight().unwrap().shape, vec![10240, 5120]);
        assert_eq!(qkv.weight().unwrap().dtype, "F8_E4M3");
        assert!(qkv.tensor("weight_scale_inv").is_some());
        assert_eq!(delta.group("linear_attn.conv1d").unwrap().weight().unwrap().shape, vec![10240, 1, 4]);
        assert_eq!(full.group("self_attn.q_proj").unwrap().weight().unwrap().shape, vec![12288, 5120]);
        assert!(s.any_leaf("q_norm", "weight"));
    }

    #[test]
    fn experts_collapse_to_a_count_read_from_the_tensors() {
        let s = header("qwen3.6-35b-a3b-fp8");
        let l = s.layer(3).unwrap();
        let down = l.group("mlp.experts.*.down_proj").unwrap();
        assert_eq!(down.instances, 256);
        assert_eq!(down.weight().unwrap().shape.len(), 2);
        assert!(l.group("mlp.gate").is_some());
        assert_eq!(l.group("mlp.shared_expert.down_proj").unwrap().instances, 1);
        assert!(l.group("mlp.shared_expert_gate").is_some());
        assert!(l.groups.iter().filter(|g| g.path.starts_with("mlp.experts.")).all(|g| g.instances == 256));
    }

    #[test]
    fn a_llama_checkpoint_has_no_norm_or_bias_tensors_inside_attention() {
        let s = header("llama-3.2-1b");
        let l = s.layer(0).unwrap();
        assert!(l.in_module("self_attn").all(|g| !g.is_norm()));
        assert!(!s.any_leaf("q_norm", "weight"));
        assert!(!s.any_leaf("q_proj", "bias"));
        assert_eq!(l.in_module("self_attn").count(), 4);
        assert_eq!(l.group("mlp.gate_proj").unwrap().weight().unwrap().shape, vec![8192, 2048]);
    }

    #[test]
    fn extra_norms_and_scalars_show_up_because_they_are_in_the_checkpoint() {
        let s = header("gemma4-31b");
        let l = s.layer(0).unwrap();
        for p in ["pre_feedforward_layernorm", "post_feedforward_layernorm", "layer_scalar"] {
            assert!(l.group(p).is_some(), "{p}");
        }
        assert!(l.group("self_attn.v_proj").is_some());
        assert!(s.layer(5).unwrap().group("self_attn.v_proj").is_none());
        assert_eq!(s.layer(5).unwrap().group("self_attn.q_norm").unwrap().weight().unwrap().shape, vec![512]);
    }

    #[test]
    fn a_checkpoint_directory_is_read_from_its_headers_only() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5");
        let s = Structure::read_dir(&dir).unwrap();
        assert!(s.main().unwrap().layers.len() >= 2);
        assert!(s.tensors > 10);
    }
}
