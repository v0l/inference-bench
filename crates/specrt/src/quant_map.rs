use anyhow::{Context, Result, bail, ensure};
use safetensors::Dtype;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorQuant {
    Fp8,
    #[serde(rename = "fp8_block128")]
    Fp8Block128,
    Mxfp8,
    Nvfp4,
    #[serde(rename = "nvfp4_csf")]
    Nvfp4Csf,
    #[serde(rename = "w4a16_nvfp4")]
    W4a16Nvfp4,
}

pub const CSF_PARTS: [&str; 5] = ["bases", "codes", "page_bases", "offsets", "exceptions"];

impl TensorQuant {
    pub fn activations(self) -> Option<crate::spec::ActQuant> {
        match self {
            TensorQuant::Fp8 => Some(crate::spec::ActQuant::Fp8),
            TensorQuant::Nvfp4 | TensorQuant::Nvfp4Csf => Some(crate::spec::ActQuant::Fp4),
            TensorQuant::Fp8Block128 | TensorQuant::Mxfp8 | TensorQuant::W4a16Nvfp4 => None,
        }
    }

    pub fn is_fp4(self) -> bool {
        matches!(self, TensorQuant::Nvfp4 | TensorQuant::Nvfp4Csf | TensorQuant::W4a16Nvfp4)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct QuantMap(pub BTreeMap<String, TensorQuant>);

pub struct Tensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

fn layers_of<'a>(rule: &crate::family::FamilyQuant, q: &'a serde_json::Value) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
    if rule.layers_key.is_empty() {
        return None;
    }
    q.get(rule.layers_key.as_str()).and_then(|l| l.as_object())
}

impl QuantMap {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn from_config(rule: &crate::family::FamilyQuant, q: &serde_json::Value) -> Result<QuantMap> {
        let Some(layers) = layers_of(rule, q) else { return Ok(QuantMap::default()) };
        let mut out = BTreeMap::new();
        for (module, entry) in layers {
            let mut entry = entry.clone();
            if entry.get(rule.method_key.as_str()).is_none()
                && let Some(m) = q.get(rule.method_key.as_str())
            {
                entry[rule.method_key.as_str()] = m.clone();
            }
            match crate::family::first_match(&rule.tensor_rules, &entry) {
                Some(f) => out.insert(module.clone(), f),
                None => bail!("{module}: quantisation {} is not supported", layers[module]),
            };
        }
        Ok(QuantMap(out))
    }

    pub fn from_quant_config(q: &serde_json::Value) -> Result<QuantMap> {
        let family = crate::family::active();
        QuantMap::from_config(&family.describing()?.quantization, q)
    }

    pub fn renamed(&self, rules: &std::collections::BTreeMap<String, String>) -> QuantMap {
        QuantMap(self.0.iter().map(|(k, v)| (rules.iter().fold(k.clone(), |n, (from, to)| n.replace(from.as_str(), to)), *v)).collect())
    }

    pub fn owner<'a>(&self, tensor: &'a str) -> Option<(&'a str, TensorQuant)> {
        let mut at = tensor;
        while let Some((head, _)) = at.rsplit_once('.') {
            if let Some(&f) = self.0.get(head) {
                return Some((head, f));
            }
            at = head;
        }
        None
    }

    pub fn tensor(&self, name: &str) -> Option<TensorQuant> {
        self.owner(name).map(|(_, f)| f)
    }

    pub fn counts(&self) -> BTreeMap<TensorQuant, usize> {
        let mut out = BTreeMap::new();
        for f in self.0.values() {
            *out.entry(*f).or_insert(0) += 1;
        }
        out
    }

    pub fn mlp_fp4(&self, layer: usize) -> Option<bool> {
        let key = format!("layers.{layer}.mlp.");
        let mut hits = self.0.iter().filter(|(k, _)| k.contains(&key)).peekable();
        hits.peek()?;
        Some(hits.any(|(_, f)| f.is_fp4()))
    }

    pub fn fp4_experts(&self, layers: usize) -> bool {
        let main = |k: &str| (0..layers).any(|l| k.contains(&format!("layers.{l}.mlp.experts")));
        let mut experts = self.0.iter().filter(|(k, _)| main(k)).peekable();
        experts.peek().is_some() && experts.all(|(_, f)| f.is_fp4())
    }

    pub fn stage(&self, start: usize, end: usize) -> QuantMap {
        let mut out = BTreeMap::new();
        for (k, f) in &self.0 {
            let Some(at) = k.find("layers.").map(|i| i + "layers.".len()) else {
                out.insert(k.clone(), *f);
                continue;
            };
            let digits = k[at..].bytes().take_while(u8::is_ascii_digit).count();
            let Ok(layer) = k[at..at + digits].parse::<usize>() else {
                out.insert(k.clone(), *f);
                continue;
            };
            if (start..end).contains(&layer) {
                out.insert(format!("{}{}{}", &k[..at], layer - start, &k[at + digits..]), *f);
            }
        }
        QuantMap(out)
    }

    pub fn check_layout(&self, tensors: &BTreeMap<String, Tensor>) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        for (name, t) in tensors {
            let Some(module) = name.strip_suffix(".weight") else { continue };
            match self.owner(name) {
                Some((_, f)) => check_weight(module, t, f, tensors)?,
                None => ensure!(!matches!(t.dtype, Dtype::F8_E4M3 | Dtype::U8), "{name} is {:?} but no quantized_layers entry covers it", t.dtype),
            }
        }
        Ok(())
    }
}

fn shape_is(name: &str, t: &Tensor, dtype: Dtype, shapes: &[&[usize]]) -> Result<()> {
    ensure!(t.dtype == dtype && shapes.iter().any(|s| t.shape == *s), "{name} is {:?} {:?}, expected {dtype:?} {:?}", t.dtype, t.shape, shapes[0]);
    Ok(())
}

fn part<'a>(tensors: &'a BTreeMap<String, Tensor>, name: &str, format: TensorQuant) -> Result<&'a Tensor> {
    tensors.get(name).with_context(|| format!("{name} is missing ({format:?} needs it)"))
}

fn scalar(tensors: &BTreeMap<String, Tensor>, name: &str, format: TensorQuant) -> Result<()> {
    let t = part(tensors, name, format)?;
    ensure!(t.dtype == Dtype::F32 && t.shape.iter().product::<usize>() == 1, "{name} is {:?} {:?}, expected one F32", t.dtype, t.shape);
    Ok(())
}

fn check_weight(module: &str, w: &Tensor, format: TensorQuant, tensors: &BTreeMap<String, Tensor>) -> Result<()> {
    let weight = format!("{module}.weight");
    let scale = format!("{module}.weight_scale");
    ensure!(w.shape.len() == 2, "{weight} has shape {:?}, expected a matrix", w.shape);
    let (rows, stored) = (w.shape[0], w.shape[1]);
    let csf = CSF_PARTS.map(|p| format!("{scale}.csf32_{p}"));
    if format != TensorQuant::Nvfp4Csf
        && let Some(stray) = csf.iter().find(|n| tensors.contains_key(n.as_str()))
    {
        bail!("{stray} is a compressed scale but {module} is {format:?}");
    }
    if format.is_fp4() {
        shape_is(&weight, w, Dtype::U8, &[&[rows, stored]])?;
        let cols = stored * 2;
        ensure!(cols.is_multiple_of(16), "{weight} has {cols} columns, not a multiple of 16");
        scalar(tensors, &format!("{module}.weight_scale_2"), format)?;
        if format == TensorQuant::Nvfp4Csf {
            ensure!(!tensors.contains_key(&scale), "{scale} is plain but {module} declares compressed scales");
            ensure!(rows.is_multiple_of(32), "{weight} has {rows} rows, compressed scales need a multiple of 32");
            let groups = cols / 16;
            let want: [(Dtype, Option<usize>); 5] = [(Dtype::U8, Some(rows)), (Dtype::U8, Some(rows * groups / 2)), (Dtype::U32, None), (Dtype::U16, None), (Dtype::U16, None)];
            for (n, (dtype, len)) in csf.iter().zip(want) {
                let t = part(tensors, n, format)?;
                ensure!(t.dtype == dtype && t.shape.len() == 1 && len.is_none_or(|l| t.shape[0] == l), "{n} is {:?} {:?}, expected {dtype:?} [{}]", t.dtype, t.shape, len.map_or("n".to_string(), |l| l.to_string()));
            }
        } else {
            shape_is(&scale, part(tensors, &scale, format)?, Dtype::F8_E4M3, &[&[rows, cols / 16]])?;
        }
        return Ok(());
    }
    shape_is(&weight, w, Dtype::F8_E4M3, &[&[rows, stored]])?;
    let s = part(tensors, &scale, format)?;
    match format {
        TensorQuant::Mxfp8 => {
            ensure!(stored.is_multiple_of(32), "{weight} has {stored} columns, MXFP8 needs a multiple of 32");
            shape_is(&scale, s, Dtype::U8, &[&[rows, stored / 32]])
        }
        TensorQuant::Fp8Block128 => {
            let (rb, cb) = (rows.div_ceil(128), stored.div_ceil(128));
            shape_is(&scale, s, Dtype::F32, &[&[rb, 1, cb, 1], &[rb, cb]])
        }
        _ => {
            ensure!(s.dtype == Dtype::F32 && (s.shape.iter().product::<usize>() == 1 || s.shape == [rows, 1] || s.shape == [rows]), "{scale} is {:?} {:?}, expected one F32 or one per row", s.dtype, s.shape);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_longest_module_prefix_owns_a_tensor() {
        let map = QuantMap(BTreeMap::from([("a.b".to_string(), TensorQuant::Mxfp8), ("a.b.c".to_string(), TensorQuant::Fp8)]));
        assert_eq!(map.tensor("a.b.c.weight"), Some(TensorQuant::Fp8));
        assert_eq!(map.tensor("a.b.weight"), Some(TensorQuant::Mxfp8));
        assert_eq!(map.tensor("a.bc.weight"), None);
        assert_eq!(map.tensor("a.b"), None);
    }
}
