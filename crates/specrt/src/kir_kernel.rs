use kir::desc::Description;
use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::RwLock;

static ACTIVE: RwLock<Option<Registry>> = RwLock::new(None);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Registry {
    descriptions: BTreeMap<String, Description>,
}

impl Registry {
    pub fn from_texts<'a>(texts: impl IntoIterator<Item = &'a str>) -> Result<Registry> {
        let mut out = Registry::default();
        for text in texts {
            let table: toml::Table = toml::from_str(text)?;
            if table.get("kind").and_then(|k| k.as_str()) != Some(kir::desc::KIND) {
                continue;
            }
            let d: Description = toml::from_str(text)?;
            if out.descriptions.insert(d.name.clone(), d).is_some() {
                anyhow::bail!("kernel description {} appears twice", out.descriptions.keys().last().map(String::as_str).unwrap_or_default());
            }
        }
        Ok(out)
    }

    pub fn embedded() -> Registry {
        Registry::from_texts(crate::spec::EMBEDDED.iter().map(|(_, t)| *t)).expect("the embedded kernel descriptions parse")
    }

    pub fn install(registry: Registry) {
        *ACTIVE.write().unwrap() = Some(registry);
    }

    pub fn active() -> Registry {
        ACTIVE.read().unwrap().clone().unwrap_or_else(Registry::embedded)
    }

    pub fn descriptions(&self) -> impl Iterator<Item = &Description> {
        self.descriptions.values()
    }

    pub fn get(&self, name: &str) -> Option<&Description> {
        self.descriptions.get(name)
    }
}
