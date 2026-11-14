use crate::library::{Lineage, Relation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

pub const HF_BASE: &str = "https://huggingface.co";

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Repo {
    #[serde(rename = "id")]
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RepoFile {
    pub path: String,
    #[serde(default)]
    pub size: u64,
    #[serde(rename = "type", default)]
    pub kind: String,
}

pub struct Hub {
    base: String,
    token: Option<String>,
    agent: ureq::Agent,
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(std::time::Duration::from_secs(15)))
        .timeout_recv_body(Some(std::time::Duration::from_secs(60)))
        .build()
        .into()
}

impl Hub {
    pub fn new(base: &str, token: &str) -> Self {
        Self {
            base: base.trim_end_matches('/').to_string(),
            token: (!token.trim().is_empty()).then(|| token.trim().to_string()),
            agent: agent(),
        }
    }

    fn get(&self, url: &str) -> ureq::RequestBuilder<ureq::typestate::WithoutBody> {
        let mut req = self.agent.get(url);
        if let Some(t) = &self.token {
            req = req.header("Authorization", &format!("Bearer {t}"));
        }
        req
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Repo>, String> {
        let url = format!("{}/api/models", self.base);
        let mut resp = self
            .get(&url)
            .query("search", query)
            .query("limit", limit.to_string())
            .query("sort", "downloads")
            .query("direction", "-1")
            .query("filter", "text-generation")
            .call()
            .map_err(|e| format!("{url}: {e}"))?;
        if resp.status().as_u16() >= 400 {
            return Err(format!("{url}: HTTP {}", resp.status()));
        }
        resp.body_mut().read_json().map_err(|e| format!("bad search reply: {e}"))
    }

    pub fn base_of(&self, repo: &str) -> Result<Option<Lineage>, String> {
        let url = format!("{}/api/models/{repo}", self.base);
        let mut resp = self.get(&url).call().map_err(|e| format!("{url}: {e}"))?;
        if resp.status().as_u16() >= 400 {
            return Err(format!("{url}: HTTP {}", resp.status()));
        }
        let info: serde_json::Value = resp.body_mut().read_json().map_err(|e| format!("bad model reply: {e}"))?;
        Ok(lineage_from_info(&info))
    }

    fn child_url(&self, base: &str, relation: Relation) -> String {
        format!("{}/api/models?filter=base_model%3A{}%3A{}", self.base, relation.label(), base.replace('/', "%2F"))
    }

    pub fn children(&self, base: &str, relation: Relation, limit: usize) -> Result<Vec<Repo>, String> {
        let url = self.child_url(base, relation);
        let mut resp = self
            .get(&url)
            .query("limit", limit.to_string())
            .query("sort", "downloads")
            .query("direction", "-1")
            .call()
            .map_err(|e| format!("{url}: {e}"))?;
        if resp.status().as_u16() >= 400 {
            return Err(format!("{url}: HTTP {}", resp.status()));
        }
        resp.body_mut().read_json().map_err(|e| format!("bad children reply: {e}"))
    }

    pub fn child_count(&self, base: &str, relation: Relation) -> Result<u64, String> {
        let url = self.child_url(base, relation);
        let mut resp = self.get(&url).query("limit", "1000").call().map_err(|e| format!("{url}: {e}"))?;
        if resp.status().as_u16() >= 400 {
            return Err(format!("{url}: HTTP {}", resp.status()));
        }
        if let Some(total) = resp.headers().get("x-total-count").and_then(|v| v.to_str().ok()).and_then(|v| v.trim().parse().ok()) {
            return Ok(total);
        }
        let page: Vec<serde_json::Value> = resp.body_mut().read_json().map_err(|e| format!("bad children reply: {e}"))?;
        Ok(page.len() as u64)
    }

    pub fn files(&self, repo: &str) -> Result<Vec<RepoFile>, String> {
        let url = format!("{}/api/models/{repo}/tree/main", self.base);
        let mut resp = self.get(&url).query("recursive", "1").call().map_err(|e| format!("{url}: {e}"))?;
        if resp.status().as_u16() >= 400 {
            return Err(format!("{url}: HTTP {}", resp.status()));
        }
        let all: Vec<RepoFile> = resp.body_mut().read_json().map_err(|e| format!("bad file list: {e}"))?;
        Ok(all.into_iter().filter(|f| f.kind == "file" || f.kind.is_empty()).collect())
    }

    pub fn download(
        &self,
        repo: &str,
        file: &str,
        hub_dir: &Path,
        progress: &Progress,
    ) -> Result<PathBuf, String> {
        let target = snapshot_path(hub_dir, repo, file);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let part = PathBuf::from(format!("{}.part", target.display()));
        let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let url = format!("{}/{repo}/resolve/main/{file}", self.base);
        let mut req = self.get(&url);
        if have > 0 {
            req = req.header("Range", &format!("bytes={have}-"));
        }
        let resp = req.call().map_err(|e| format!("{url}: {e}"))?;
        let status = resp.status().as_u16();
        if status == 416 && have > 0 {
            std::fs::rename(&part, &target).map_err(|e| e.to_string())?;
            return Ok(target);
        }
        if status >= 400 {
            return Err(format!("{url}: HTTP {status}"));
        }
        let resumed = status == 206;
        let remaining = resp
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        let start = if resumed { have } else { 0 };
        progress.total.store(start + remaining, Ordering::Relaxed);
        progress.done.store(start, Ordering::Relaxed);
        let mut out = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(resumed)
            .truncate(!resumed)
            .open(&part)
            .map_err(|e| e.to_string())?;
        let mut body = resp.into_body().into_reader();
        let mut buf = vec![0u8; 256 * 1024];
        loop {
            if progress.cancel.load(Ordering::Relaxed) {
                return Err("paused".into());
            }
            let n = body.read(&mut buf).map_err(|e| format!("download broke: {e}"))?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
            progress.done.fetch_add(n as u64, Ordering::Relaxed);
        }
        out.flush().map_err(|e| e.to_string())?;
        drop(out);
        std::fs::rename(&part, &target).map_err(|e| e.to_string())?;
        Ok(target)
    }
}

pub fn lineage_from_info(info: &serde_json::Value) -> Option<Lineage> {
    let card = &info["cardData"];
    let mut bases: Vec<String> = match &card["base_model"] {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(a) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        _ => Vec::new(),
    };
    let mut relation = card["base_model_relation"].as_str().and_then(Relation::parse);
    for tag in info["tags"].as_array().into_iter().flatten().filter_map(|t| t.as_str()) {
        let mut parts = tag.splitn(3, ':');
        if let (Some("base_model"), Some(rel), Some(repo)) = (parts.next(), parts.next(), parts.next()) {
            if bases.is_empty() {
                bases.push(repo.to_string());
            }
            relation = relation.or(Relation::parse(rel));
        }
    }
    bases.retain(|b| !b.is_empty());
    (!bases.is_empty()).then(|| Lineage { bases, relation: relation.unwrap_or(Relation::Finetune) })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModelTree {
    pub chain: Vec<String>,
    pub relations: Vec<String>,
    pub counts: Vec<(String, u64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct TreeCache {
    infos: BTreeMap<String, (u64, Option<(Vec<String>, String)>)>,
    counts: BTreeMap<String, (u64, u64)>,
}

const CACHE_SECS: u64 = 86_400;
const MAX_DEPTH: usize = 6;

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn model_tree(
    hub: &Hub,
    repo: &str,
    local: &dyn Fn(&str) -> Option<Lineage>,
    cache_path: &Path,
    now: u64,
) -> Result<ModelTree, String> {
    let mut cache: TreeCache = std::fs::read_to_string(cache_path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    let mut tree = ModelTree { chain: vec![repo.to_string()], ..ModelTree::default() };
    let mut at = repo.to_string();
    for _ in 0..MAX_DEPTH {
        let found = match local(&at) {
            Some(l) => Some(l),
            None => match cache.infos.get(&at).filter(|(t, _)| now.saturating_sub(*t) < CACHE_SECS) {
                Some((_, cached)) => cached.as_ref().map(|(b, r)| Lineage { bases: b.clone(), relation: Relation::parse(r).unwrap_or(Relation::Finetune) }),
                None => {
                    let fetched = hub.base_of(&at)?;
                    cache.infos.insert(at.clone(), (now, fetched.as_ref().map(|l| (l.bases.clone(), l.relation.label().to_string()))));
                    fetched
                }
            },
        };
        let Some(l) = found else { break };
        if tree.chain.iter().any(|c| c == l.base()) {
            break;
        }
        tree.relations.push(l.relation.label().to_string());
        tree.chain.push(l.base().to_string());
        at = l.base().to_string();
    }
    let top = tree.chain[tree.chain.len() - 1].clone();
    for relation in Relation::ALL {
        let key = format!("{}:{top}", relation.label());
        let n = match cache.counts.get(&key).filter(|(t, _)| now.saturating_sub(*t) < CACHE_SECS) {
            Some((_, n)) => *n,
            None => {
                let n = hub.child_count(&top, relation)?;
                cache.counts.insert(key, (now, n));
                n
            }
        };
        tree.counts.push((relation.label().to_string(), n));
    }
    if let Some(parent) = cache_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(cache_path, serde_json::to_string(&cache).unwrap_or_default());
    Ok(tree)
}

pub fn model_tree_now(hub: &Hub, repo: &str, local: &dyn Fn(&str) -> Option<Lineage>, cache_path: &Path) -> Result<ModelTree, String> {
    model_tree(hub, repo, local, cache_path, now_secs())
}

#[derive(Clone, Default)]
pub struct Progress {
    pub done: Arc<AtomicU64>,
    pub total: Arc<AtomicU64>,
    pub cancel: Arc<AtomicBool>,
}

impl Progress {
    pub fn fraction(&self) -> Option<f32> {
        let total = self.total.load(Ordering::Relaxed);
        (total > 0).then(|| self.done.load(Ordering::Relaxed) as f32 / total as f32)
    }
}

pub fn snapshot_path(hub_dir: &Path, repo: &str, file: &str) -> PathBuf {
    hub_dir.join(format!("models--{}", repo.replace('/', "--"))).join("snapshots/main").join(file)
}

pub fn is_weight_file(path: &str) -> bool {
    path.ends_with(".gguf") || path.ends_with(".safetensors")
}

pub fn companion_files(files: &[RepoFile]) -> Vec<&RepoFile> {
    let wanted = ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"];
    files.iter().filter(|f| wanted.contains(&f.path.as_str())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    const PAYLOAD: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";

    fn server() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for mut s in listener.incoming().flatten() {
                std::thread::spawn(move || {
                    let mut r = BufReader::new(s.try_clone().unwrap());
                    let mut line = String::new();
                    r.read_line(&mut line).unwrap();
                    let path = line.split_whitespace().nth(1).unwrap().to_string();
                    let mut range = 0usize;
                    loop {
                        line.clear();
                        r.read_line(&mut line).unwrap();
                        if line.trim().is_empty() {
                            break;
                        }
                        if let Some(v) = line.to_lowercase().strip_prefix("range: bytes=") {
                            range = v.trim().trim_end_matches('-').parse().unwrap();
                        }
                    }
                    let (head, body): (String, Vec<u8>) = if path.starts_with("/api/models/acme/m/tree") {
                        let b = br#"[{"type":"file","path":"config.json","size":2},{"type":"file","path":"m-Q4.gguf","size":36},{"type":"directory","path":"sub"}]"#.to_vec();
                        ("200 OK".into(), b)
                    } else if path.starts_with("/api/models?") {
                        ("200 OK".into(), br#"[{"id":"acme/m","downloads":5,"likes":1,"tags":["gguf"]}]"#.to_vec())
                    } else if path.contains("nope") {
                        ("404 Not Found".into(), Vec::new())
                    } else if path.contains("/resolve/main/") && range > 0 {
                        ("206 Partial Content".into(), PAYLOAD[range..].to_vec())
                    } else if path.contains("/resolve/main/") {
                        ("200 OK".into(), PAYLOAD.to_vec())
                    } else {
                        ("404 Not Found".into(), Vec::new())
                    };
                    use std::io::Write;
                    let _ = write!(s, "HTTP/1.1 {head}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                    let _ = s.write_all(&body);
                });
            }
        });
        base
    }

    fn temp(n: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ib-hf-{n}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn search_and_file_listing_parse_the_hub_api() {
        let hub = Hub::new(&server(), "");
        let repos = hub.search("m", 5).unwrap();
        assert_eq!((repos[0].id.as_str(), repos[0].downloads), ("acme/m", 5));
        let files = hub.files("acme/m").unwrap();
        assert_eq!(files.len(), 2);
        assert!(is_weight_file(&files[1].path) && !is_weight_file(&files[0].path));
        assert_eq!(companion_files(&files).len(), 1);
    }

    #[test]
    fn a_download_lands_in_the_hub_cache_layout() {
        let hub = Hub::new(&server(), "");
        let dir = temp("full");
        let p = Progress::default();
        let path = hub.download("acme/m", "m-Q4.gguf", &dir, &p).unwrap();
        assert_eq!(path, dir.join("models--acme--m/snapshots/main/m-Q4.gguf"));
        assert_eq!(std::fs::read(&path).unwrap(), PAYLOAD);
        assert_eq!(p.fraction(), Some(1.0));
    }

    #[test]
    fn a_partial_file_is_resumed_with_a_range_request() {
        let hub = Hub::new(&server(), "");
        let dir = temp("resume");
        let target = snapshot_path(&dir, "acme/m", "m-Q4.gguf");
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::write(format!("{}.part", target.display()), &PAYLOAD[..10]).unwrap();
        let p = Progress::default();
        let path = hub.download("acme/m", "m-Q4.gguf", &dir, &p).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), PAYLOAD);
        assert_eq!(p.total.load(Ordering::Relaxed), PAYLOAD.len() as u64);
        assert!(!PathBuf::from(format!("{}.part", path.display())).exists());
    }

    #[test]
    fn a_missing_file_is_an_error() {
        let hub = Hub::new(&server(), "");
        let e = hub.download("acme/m", "nope.bin", &temp("x"), &Progress::default());
        assert!(e.is_err());
    }

    fn tree_server(hits: std::sync::Arc<std::sync::atomic::AtomicUsize>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for mut s in listener.incoming().flatten() {
                let hits = hits.clone();
                std::thread::spawn(move || {
                    let mut r = BufReader::new(s.try_clone().unwrap());
                    let mut line = String::new();
                    r.read_line(&mut line).unwrap();
                    let path = line.split_whitespace().nth(1).unwrap().to_string();
                    loop {
                        line.clear();
                        r.read_line(&mut line).unwrap();
                        if line.trim().is_empty() {
                            break;
                        }
                    }
                    hits.fetch_add(1, Ordering::SeqCst);
                    let (extra, body) = if path.starts_with("/api/models/org/Mid") {
                        (String::new(), br#"{"cardData":{"base_model":"Qwen/Base","base_model_relation":"finetune"}}"#.to_vec())
                    } else if path.starts_with("/api/models/Qwen/Base") {
                        (String::new(), br#"{"cardData":{}}"#.to_vec())
                    } else if path.contains("filter=base_model%3Aquantized%3AQwen%2FBase") {
                        ("x-total-count: 1794\r\n".to_string(), br#"[{"id":"unsloth/Base-GGUF"}]"#.to_vec())
                    } else if path.contains("filter=base_model%3Afinetune%3AQwen%2FBase") {
                        ("x-total-count: 902\r\n".to_string(), b"[]".to_vec())
                    } else if path.contains("filter=base_model") {
                        (String::new(), br#"[{"id":"a/b"},{"id":"c/d"}]"#.to_vec())
                    } else {
                        (String::new(), b"{}".to_vec())
                    };
                    use std::io::Write;
                    let _ = write!(s, "HTTP/1.1 200 OK\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                    let _ = s.write_all(&body);
                });
            }
        });
        base
    }

    #[test]
    fn the_model_tree_walks_up_and_counts_the_bases_children_and_caches_for_a_day() {
        let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let hub = Hub::new(&tree_server(hits.clone()), "");
        let dir = temp("tree");
        let cache = dir.join("tree.json");
        let none = |_: &str| None;
        let t = model_tree(&hub, "org/Mid", &none, &cache, 1000).unwrap();
        assert_eq!(t.chain, ["org/Mid", "Qwen/Base"]);
        assert_eq!(t.relations, ["finetune"]);
        let count = |name: &str| t.counts.iter().find(|c| c.0 == name).unwrap().1;
        assert_eq!((count("quantized"), count("finetune"), count("adapter"), count("merge")), (1794, 902, 2, 2));
        let first = hits.load(Ordering::SeqCst);
        let again = model_tree(&hub, "org/Mid", &none, &cache, 2000).unwrap();
        assert_eq!(again, t);
        assert_eq!(hits.load(Ordering::SeqCst), first);
        model_tree(&hub, "org/Mid", &none, &cache, 1000 + 86_401).unwrap();
        assert!(hits.load(Ordering::SeqCst) > first);
    }

    #[test]
    fn the_local_card_is_used_before_the_hub() {
        let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let hub = Hub::new(&tree_server(hits), "");
        let local = |r: &str| (r == "me/Quant").then(|| Lineage { bases: vec!["Qwen/Base".into()], relation: Relation::Quantized });
        let t = model_tree(&hub, "me/Quant", &local, &temp("tree-local").join("c.json"), 5).unwrap();
        assert_eq!((t.chain.as_slice(), t.relations.as_slice()), (["me/Quant".to_string(), "Qwen/Base".to_string()].as_slice(), ["quantized".to_string()].as_slice()));
    }

    #[test]
    fn hub_tags_give_the_lineage_when_the_card_has_none() {
        let info: serde_json::Value = serde_json::from_str(r#"{"tags":["gguf","base_model:quantized:Qwen/Qwen3.8-27B"],"cardData":{}}"#).unwrap();
        let l = lineage_from_info(&info).unwrap();
        assert_eq!((l.base(), l.relation), ("Qwen/Qwen3.8-27B", Relation::Quantized));
    }
}
