use crate::spec::TtsVoices;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct Entry {
    pub name: String,
    #[serde(default, deserialize_with = "or_empty")]
    pub voice_id: String,
    pub filename: String,
    #[serde(default, deserialize_with = "or_empty")]
    pub language: String,
    #[serde(default, deserialize_with = "or_empty")]
    pub locale: String,
    #[serde(default, deserialize_with = "or_empty")]
    pub gender: String,
    #[serde(default, deserialize_with = "or_empty")]
    pub description: String,
}

fn or_empty<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(<Option<String> as serde::Deserialize>::deserialize(d)?.unwrap_or_default())
}

impl Entry {
    pub fn label(&self) -> String {
        let accent = match self.locale.as_str() {
            "en-us" => "American",
            "en-gb" => "British",
            "en-au" => "Australian",
            "en-ca" => "Canadian",
            "en-in" => "Indian",
            "en-ie" => "Irish",
            "en-za" => "South African",
            _ => "",
        };
        let man = if self.gender == "male" { " man" } else { "" };
        if accent.is_empty() { self.name.clone() } else { format!("{}, {accent}{man}", self.name) }
    }
}

fn plain(s: &str) -> bool {
    !s.is_empty() && !s.contains(['/', '\\']) && !s.contains("..")
}

pub fn parse(text: &str, language: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Entry = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("voice catalogue line {}: {e}", i + 1);
                continue;
            }
        };
        let file_ok = v.filename.strip_prefix("voices/").is_some_and(plain);
        let spoken = language.is_empty() || (v.language == language && v.locale.starts_with(&format!("{language}-")));
        if spoken && file_ok && plain(&v.name) {
            out.push(v);
        }
    }
    out
}

#[derive(Clone, Debug, Default)]
pub struct Catalogue {
    pub repo: String,
    pub file: String,
    pub default: String,
    pub language: String,
    pub dirs: Vec<PathBuf>,
    pub voices: Vec<Entry>,
}

impl Catalogue {
    pub fn find(model_dir: &Path, spec: &TtsVoices, language: &str) -> Catalogue {
        let mut dirs = vec![model_dir.to_path_buf()];
        if !spec.repo.is_empty() {
            let snapshots = snapshots(&spec.repo);
            let mut found: Vec<PathBuf> = std::fs::read_dir(&snapshots).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
            found.sort();
            let main = snapshots.join("main");
            if !found.contains(&main) {
                found.push(main);
            }
            dirs.extend(found);
        }
        let mut c = Catalogue { repo: spec.repo.clone(), file: spec.catalogue.clone(), default: spec.default.clone(), language: language.to_string(), dirs, voices: Vec::new() };
        c.reload();
        c
    }

    pub fn open(model_dir: &Path) -> Result<Catalogue> {
        let json = std::fs::read_to_string(model_dir.join("config.json")).with_context(|| model_dir.display().to_string())?;
        let spec = crate::spec::Spec::from_hf_config_tts(&json)?;
        let tts = spec.tts.context("no [tts]")?;
        Ok(Catalogue::find(model_dir, &tts.voices, &tts.language))
    }

    pub fn reload(&mut self) {
        self.voices = match self.path() {
            Some(p) => std::fs::read_to_string(&p).map(|t| parse(&t, &self.language)).unwrap_or_default(),
            None => Vec::new(),
        };
    }

    pub fn path(&self) -> Option<PathBuf> {
        if self.file.is_empty() {
            return None;
        }
        self.dirs.iter().map(|d| d.join(&self.file)).find(|p| p.is_file())
    }

    pub fn home(&self) -> PathBuf {
        self.path().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_else(|| self.dirs[0].clone())
    }

    pub fn pick(&self, asked: &str) -> Option<&Entry> {
        let find = |n: &str| self.voices.iter().find(|v| v.name.eq_ignore_ascii_case(n.trim()));
        find(asked).or_else(|| find(&self.default)).or_else(|| self.voices.first())
    }

    pub fn file_of(&self, v: &Entry) -> Option<PathBuf> {
        self.dirs.iter().map(|d| d.join(&v.filename)).find(|p| p.is_file())
    }

    pub fn installed(&self, v: &Entry) -> bool {
        self.file_of(v).is_some()
    }

    pub fn url(&self, base: &str, file: &str) -> Option<String> {
        (!self.repo.is_empty()).then(|| format!("{}/{}/resolve/main/{file}", base.trim_end_matches('/'), self.repo))
    }

    pub fn fetch_catalogue(&mut self, base: &str) -> Result<()> {
        if self.path().is_some() || self.file.is_empty() {
            return Ok(());
        }
        let url = self.url(base, &self.file).context("the voice catalogue has no repo to fetch from")?;
        fetch(&url, &snapshots(&self.repo).join("main").join(&self.file))?;
        self.reload();
        Ok(())
    }

    pub fn voice_file(&self, v: &Entry, base: &str) -> Result<PathBuf> {
        if let Some(p) = self.file_of(v) {
            return Ok(p);
        }
        let url = self.url(base, &v.filename).with_context(|| format!("voice {} is not on this machine and the catalogue names no repo", v.name))?;
        let dest = snapshots(&self.repo).join("main").join(&v.filename);
        fetch(&url, &dest)?;
        Ok(dest)
    }
}

pub const HUB: &str = "https://huggingface.co";

fn snapshots(repo: &str) -> PathBuf {
    crate::weights::hub_dir().join(format!("models--{}", repo.replace('/', "--"))).join("snapshots")
}

pub fn fetch(url: &str, dest: &Path) -> Result<()> {
    use std::io::Read;
    let resp = ureq::get(url).call().map_err(|e| anyhow::anyhow!("{url}: {e}"))?;
    anyhow::ensure!(resp.status().is_success(), "{url}: HTTP {}", resp.status());
    let mut bytes = Vec::new();
    resp.into_body().into_reader().take(64 << 20).read_to_end(&mut bytes).with_context(|| url.to_string())?;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part = dest.with_extension("part");
    std::fs::write(&part, &bytes)?;
    std::fs::rename(&part, dest)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_catalogue_keeps_voices_of_the_model_language_with_safe_names() {
        let text = r#"{"filename": "voices/a1.safetensors", "name": "Hadley", "voice_id": "a1", "language": "en", "locale": "en-us", "gender": "female"}
{"filename": "voices/b2.safetensors", "name": "Lucie", "voice_id": "b2", "language": "fr", "locale": "fr-fr"}
{"filename": "../../etc/passwd", "name": "Evil", "voice_id": "c3", "language": "en", "locale": "en-us"}
{"filename": "voices/d4.safetensors", "name": "Alfie", "voice_id": "d4", "language": "en", "locale": "en-gb", "gender": "male"}
{"filename": "voices/e5.safetensors", "name": "Odd", "voice_id": "e5", "language": "en", "locale": "en-us", "gender": null, "description": null}
{"filename": "voices/f6.safetensors", "name": "Captain", "voice_id": "f6", "language": "en", "locale": "characters"}
not json
"#;
        let voices = parse(text, "en");
        let names: Vec<_> = voices.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(names, ["Hadley", "Alfie", "Odd"]);
        assert_eq!(voices.iter().map(Entry::label).collect::<Vec<_>>(), ["Hadley, American", "Alfie, British man", "Odd, American"]);
        assert_eq!(parse(text, "fr").len(), 1);
    }

    #[test]
    fn a_name_is_picked_case_insensitive_and_an_unknown_one_gets_the_default() {
        let dir = std::env::temp_dir().join(format!("specrt-voices-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("voices")).unwrap();
        std::fs::write(dir.join("voices.jsonl"), "{\"filename\": \"voices/a.safetensors\", \"name\": \"Zoey\", \"language\": \"en\", \"locale\": \"en-us\"}\n{\"filename\": \"voices/b.safetensors\", \"name\": \"Hadley\", \"language\": \"en\", \"locale\": \"en-us\"}\n").unwrap();
        std::fs::write(dir.join("voices/b.safetensors"), b"x").unwrap();
        let spec = TtsVoices { repo: String::new(), catalogue: "voices.jsonl".into(), default: "Hadley".into() };
        let c = Catalogue::find(&dir, &spec, "en");
        assert_eq!(c.pick("zOEY").unwrap().name, "Zoey");
        assert_eq!(c.pick("alloy").unwrap().name, "Hadley");
        assert_eq!(c.pick("").unwrap().name, "Hadley");
        assert_eq!((c.installed(c.pick("zoey").unwrap()), c.installed(c.pick("hadley").unwrap())), (false, true));
        assert_eq!(c.home(), dir);
        assert_eq!(c.url("https://hf.example/", "voices/a.safetensors"), None);
        assert!(c.voice_file(c.pick("zoey").unwrap(), HUB).is_err(), "no repo, nothing to fetch from");
        let none = Catalogue::find(&dir.join("missing"), &spec, "en");
        assert!(none.voices.is_empty() && none.pick("Hadley").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
