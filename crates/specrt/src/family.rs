use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Flag {
    Fixed(bool),
    Config { config: String, #[serde(default)] default: bool },
}

impl Flag {
    pub fn read(&self, config: &serde_json::Value) -> bool {
        match self {
            Flag::Fixed(b) => *b,
            Flag::Config { config: key, default } => config.get(key).and_then(|v| v.as_bool()).unwrap_or(*default),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockKind {
    #[default]
    Dense,
    Hybrid,
    FromConfig,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Match {
    pub model_types: Vec<String>,
    pub architectures: Vec<String>,
    pub gguf: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyAttention {
    pub qkv_bias: Flag,
    pub qk_norm: Flag,
    pub output_gate: Flag,
}

impl Default for FamilyAttention {
    fn default() -> Self {
        FamilyAttention {
            qkv_bias: Flag::Config { config: "attention_bias".into(), default: false },
            qk_norm: Flag::Fixed(false),
            output_gate: Flag::Config { config: "attn_output_gate".into(), default: false },
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyBlocks {
    pub kind: BlockKind,
    pub unit_offset: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyMoe {
    pub norm_topk_default: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Family {
    pub name: String,
    pub kind: String,
    #[serde(default)]
    pub min_engine: u32,
    #[serde(default)]
    pub extends: Option<String>,
    #[serde(default, rename = "match")]
    pub matches: Match,
    #[serde(default)]
    pub attention: FamilyAttention,
    #[serde(default)]
    pub blocks: FamilyBlocks,
    #[serde(default)]
    pub mtp: bool,
    #[serde(default)]
    pub moe: FamilyMoe,
}

pub const FAMILY_KIND: &str = "family";

pub fn is_family(table: &toml::Table) -> bool {
    table.get("kind").and_then(|k| k.as_str()) == Some(FAMILY_KIND)
}

fn merge(base: &mut toml::Table, over: toml::Table) {
    for (k, v) in over {
        match (base.get_mut(&k), v) {
            (Some(toml::Value::Table(b)), toml::Value::Table(o)) => merge(b, o),
            (_, v) => {
                base.insert(k, v);
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Registry {
    families: Vec<Family>,
}

impl Registry {
    pub fn from_texts<'a>(texts: impl IntoIterator<Item = &'a str>) -> Result<Registry> {
        let mut tables: Vec<toml::Table> = Vec::new();
        for text in texts {
            let table: toml::Table = toml::from_str(text)?;
            if is_family(&table) {
                tables.push(table);
            }
        }
        let mut families = Vec::new();
        for table in &tables {
            let resolved = resolve(table, &tables, 0)?;
            families.push(toml::Value::Table(resolved).try_into::<Family>()?);
        }
        Ok(Registry { families })
    }

    pub fn embedded() -> &'static Registry {
        static EMBEDDED_FAMILIES: OnceLock<Registry> = OnceLock::new();
        EMBEDDED_FAMILIES.get_or_init(|| {
            Registry::from_texts(crate::spec::EMBEDDED.iter().map(|(_, t)| *t)).expect("the embedded family specs parse")
        })
    }

    pub fn families(&self) -> &[Family] {
        &self.families
    }

    pub fn for_model_type(&self, model_type: &str) -> Option<&Family> {
        self.families.iter().find(|f| f.matches.model_types.iter().any(|m| m == model_type))
    }

    pub fn for_gguf(&self, arch: &str) -> Option<&Family> {
        self.families.iter().find(|f| f.matches.gguf.iter().any(|m| m == arch))
    }

    pub fn named(&self, name: &str) -> Option<&Family> {
        self.families.iter().find(|f| f.name == name)
    }

    pub fn gguf_architectures(&self) -> Vec<&str> {
        self.families.iter().flat_map(|f| f.matches.gguf.iter().map(String::as_str)).collect()
    }

    pub fn strict_for(&self, model_type: &str) -> Result<&Family> {
        match self.for_model_type(model_type) {
            Some(f) => Ok(f),
            None => bail!("model_type {model_type} is not supported"),
        }
    }

    pub fn describing(&self) -> Result<&Family> {
        self.named("generic").context("no generic family spec")
    }
}

fn resolve(table: &toml::Table, all: &[toml::Table], depth: usize) -> Result<toml::Table> {
    ensure!(depth < 8, "family specs extend each other in a loop");
    let Some(parent) = table.get("extends").and_then(|e| e.as_str()) else { return Ok(table.clone()) };
    let base = all
        .iter()
        .find(|t| t.get("name").and_then(|n| n.as_str()) == Some(parent))
        .with_context(|| format!("family extends {parent}, which does not exist"))?;
    let mut merged = resolve(base, all, depth + 1)?;
    merge(&mut merged, table.clone());
    merged.remove("extends");
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "name = \"a\"\nkind = \"family\"\n[match]\nmodel_types = [\"a\", \"a_text\"]\n[attention]\nqk_norm = true\n";
    const CHILD: &str = "name = \"b\"\nkind = \"family\"\nextends = \"a\"\n[match]\nmodel_types = [\"b\"]\n[attention]\nqkv_bias = true\n";

    #[test]
    fn a_family_parses_with_defaults() {
        let r = Registry::from_texts([BASE]).unwrap();
        let f = r.for_model_type("a_text").unwrap();
        assert_eq!(f.attention.qk_norm, Flag::Fixed(true));
        assert_eq!(f.attention.qkv_bias, Flag::Config { config: "attention_bias".into(), default: false });
        assert_eq!(f.blocks.kind, BlockKind::Dense);
        assert!(r.for_model_type("zzz").is_none());
    }

    #[test]
    fn a_child_family_overrides_only_what_it_states() {
        let r = Registry::from_texts([BASE, CHILD]).unwrap();
        let b = r.for_model_type("b").unwrap();
        assert_eq!((b.attention.qk_norm.clone(), b.attention.qkv_bias.clone()), (Flag::Fixed(true), Flag::Fixed(true)));
        assert!(r.for_model_type("a").is_some());
    }

    #[test]
    fn a_missing_parent_is_an_error() {
        assert!(Registry::from_texts([CHILD]).is_err());
    }

    #[test]
    fn plain_model_specs_are_not_families() {
        let r = Registry::from_texts(["name = \"x\"\nrepo = \"y\"\n", BASE]).unwrap();
        assert_eq!(r.families().len(), 1);
    }

    #[test]
    fn flags_read_the_config_or_the_default() {
        let f = Flag::Config { config: "k".into(), default: true };
        assert!(f.read(&serde_json::json!({})));
        assert!(!f.read(&serde_json::json!({"k": false})));
        assert!(Flag::Fixed(true).read(&serde_json::json!({"k": false})));
    }
}
