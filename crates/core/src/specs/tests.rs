use super::*;
use crate::fake::{FakeFile, FakeServer, Script, tarball};

const TOP: &str = "inference-bench-abc123";

fn temp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ib-specs-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn spec_text(name: &str, layers: u32) -> String {
    let base = EMBEDDED.iter().find(|(n, _)| *n == "qwen3-0.6b.toml").unwrap().1;
    base.replace("name = \"qwen3-0.6b\"", &format!("name = \"{name}\"")).replace("count = 28", &format!("count = {layers}"))
}

fn at(rel: &str) -> String {
    format!("crates/specrt/specs/{rel}")
}

struct Rig {
    server: FakeServer,
    script: Script,
    cfg: SpecsConfig,
}

fn rig() -> Rig {
    let script = Script::default();
    let server = FakeServer::start(script.clone());
    let cfg = SpecsConfig { base: server.url.clone(), ..SpecsConfig::default() };
    Rig { server, script, cfg }
}

impl Rig {
    fn publish(&self, entries: &[(String, Vec<u8>)], etag: &str, commit: &str) {
        let refs: Vec<(&str, &[u8])> = entries.iter().map(|(p, b)| (p.as_str(), b.as_slice())).collect();
        self.script.serve_file(FakeFile {
            path: format!("/v0l/inference-bench/tar.gz/{}", self.cfg.git_ref()),
            body: tarball(TOP, &refs, Some(commit)),
            etag: Some(format!("\"{etag}\"")),
            status: None,
        });
    }

    fn requests(&self) -> usize {
        self.server.requests.lock().unwrap().len()
    }
}

fn file(rel: &str, text: &str) -> (String, Vec<u8>) {
    (at(rel), text.as_bytes().to_vec())
}

fn names(dir: &Path) -> Vec<String> {
    toml_files(dir).iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect()
}

#[test]
fn an_update_takes_good_specs_and_keeps_the_old_copy_of_a_bad_one() {
    let r = rig();
    let store = SpecStore::new(&temp("bad"));
    r.publish(&[file("a.toml", &spec_text("a", 2)), file("c.toml", &spec_text("c", 2))], "v1", "1111111aaaa");
    assert!(matches!(store.update(&r.cfg).unwrap(), Outcome::Updated(_)));
    r.publish(
        &[file("a.toml", &spec_text("a", 3)), file("b.toml", &spec_text("b", 4)), file("c.toml", "name = \"c\"\nbroken = [")],
        "v2",
        "2222222bbbb",
    );
    let Outcome::Updated(changes) = store.update(&r.cfg).unwrap() else { panic!("expected an update") };
    assert_eq!(changes.added, vec!["b"]);
    assert_eq!(changes.updated, vec!["a"]);
    assert!(changes.removed.is_empty());
    assert_eq!(changes.skipped.len(), 1);
    assert_eq!(changes.skipped[0].0, "c.toml");
    assert_eq!(names(&store.current()), vec!["a.toml", "b.toml", "c.toml"]);
    assert_eq!(std::fs::read_to_string(store.current().join("c.toml")).unwrap(), spec_text("c", 2));
    assert!(std::fs::read_to_string(store.current().join("a.toml")).unwrap().contains("count = 3"));
    let g = store.generation().unwrap();
    assert_eq!((g.commit.as_deref(), g.source.as_str()), (Some("2222222bbbb"), "v0l/inference-bench@master"));
    assert!(store.state().last_ok);
}

#[test]
fn a_matching_etag_makes_no_change() {
    let r = rig();
    let data = temp("etag");
    let store = SpecStore::new(&data);
    r.publish(&[file("a.toml", &spec_text("a", 2))], "same", "c1");
    store.update(&r.cfg).unwrap();
    let before = store.generation().unwrap();
    assert_eq!(std::fs::read_to_string(data.join("specs/.etag")).unwrap(), "\"same\"");
    assert_eq!(store.update(&r.cfg).unwrap(), Outcome::Unchanged);
    assert_eq!(store.generation().unwrap(), before);
    assert!(store.history().is_empty());
    assert_eq!(store.state().last_result, "specs are up to date");
}

#[test]
fn traversal_names_nested_paths_and_non_toml_files_are_ignored() {
    let r = rig();
    let data = temp("traverse");
    let store = SpecStore::new(&data);
    let good = spec_text("good", 2);
    r.publish(
        &[
            file("good.toml", &good),
            file("../../../escape.toml", &spec_text("escape", 2)),
            file("sub/nested.toml", &spec_text("nested", 2)),
            file("notes.md", "hello"),
            file("run.sh", "rm -rf /"),
            file(".hidden.toml", &spec_text("hidden", 2)),
            file("we ird.toml", &spec_text("weird", 2)),
            ("../crates/specrt/specs/outside.toml".into(), good.as_bytes().to_vec()),
        ],
        "t",
        "c",
    );
    store.update(&r.cfg).unwrap();
    assert_eq!(names(&store.current()), vec!["good.toml"]);
    assert!(!data.join("escape.toml").exists() && !data.join("specs/escape.toml").exists());
    assert!(!store.current().join("sub").exists());
}

#[test]
fn files_and_downloads_over_the_size_caps_are_refused() {
    let r = rig();
    let store = SpecStore::new(&temp("cap"));
    let mut big = spec_text("big", 2);
    big.push_str(&format!("\n# {}\n", "x".repeat(4096)));
    r.publish(&[file("small.toml", &spec_text("small", 2)), file("big.toml", &big)], "c1", "c1");
    let cfg = SpecsConfig { max_spec: 2048, ..r.cfg.clone() };
    let Outcome::Updated(changes) = store.update(&cfg).unwrap() else { panic!() };
    assert_eq!(names(&store.current()), vec!["small.toml"]);
    assert_eq!(changes.skipped[0].0, "big.toml");
    let tiny = SpecsConfig { max_download: 64, ..r.cfg.clone() };
    let store2 = SpecStore::new(&temp("cap2"));
    assert!(store2.update(&tiny).is_err());
    assert!(!store2.current().exists());
    assert!(!store2.state().last_ok);
}

#[test]
fn a_failed_update_leaves_the_old_folder_in_place() {
    let r = rig();
    let data = temp("fail");
    let store = SpecStore::new(&data);
    r.publish(&[file("a.toml", &spec_text("a", 2))], "v1", "c1");
    store.update(&r.cfg).unwrap();
    let before = read_dir_files(&store.current());
    r.publish(&[file("a.toml", "name = \"a\"\nnot a spec = 1")], "v2", "c2");
    assert_eq!(store.update(&r.cfg).unwrap(), Outcome::Unchanged);
    assert_eq!(read_dir_files(&store.current()), before);
    r.publish(&[file("a.toml", &spec_text("a", 7))], "v2b", "c2b");
    std::fs::write(store.root().join(".history"), "in the way").unwrap();
    assert!(store.update(&r.cfg).is_err());
    assert_eq!(read_dir_files(&store.current()), before);
    std::fs::remove_file(store.root().join(".history")).unwrap();
    r.script.serve_file(FakeFile {
        path: format!("/v0l/inference-bench/tar.gz/{}", r.cfg.git_ref()),
        body: b"this is not gzip".to_vec(),
        etag: Some("\"v3\"".into()),
        status: None,
    });
    assert!(store.update(&r.cfg).is_err());
    r.script.serve_file(FakeFile { path: format!("/v0l/inference-bench/tar.gz/{}", r.cfg.git_ref()), status: Some(500), ..FakeFile::default() });
    assert!(store.update(&r.cfg).unwrap_err().to_string().contains("500"));
    assert_eq!(read_dir_files(&store.current()), before);
    assert_eq!(std::fs::read_to_string(data.join("specs/.etag")).unwrap(), "\"v2\"");
    let fresh = SpecStore::new(&temp("fail-fresh"));
    r.publish(&[file("a.toml", "name = \"a\"\nnot a spec = 1")], "v4", "c4");
    assert!(fresh.update(&r.cfg).is_err());
    assert!(!fresh.current().exists());
    let leftovers: Vec<_> = std::fs::read_dir(store.root()).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(".incoming")).collect();
    assert!(leftovers.is_empty());
}

#[test]
fn auto_update_off_makes_no_request() {
    let r = rig();
    let store = SpecStore::new(&temp("off"));
    r.publish(&[file("a.toml", &spec_text("a", 2))], "v1", "c1");
    let cfg = SpecsConfig { auto_update: false, ..r.cfg.clone() };
    assert_eq!(store.auto_update(&cfg).unwrap(), Outcome::Disabled);
    assert!(!store.due(&cfg, Utc::now()));
    assert_eq!(r.requests(), 0);
    assert!(store.generation().is_none());
}

#[test]
fn rollback_restores_the_previous_generation_and_history_keeps_three() {
    let r = rig();
    let store = SpecStore::new(&temp("rollback"));
    for n in 1..=5 {
        r.publish(&[file("a.toml", &spec_text("a", n))], &format!("v{n}"), &format!("commit{n}"));
        store.update(&r.cfg).unwrap();
    }
    assert_eq!(store.history().len(), KEEP_GENERATIONS);
    assert_eq!(store.generation().unwrap().commit.as_deref(), Some("commit5"));
    let back = store.rollback().unwrap();
    assert_eq!(back.commit.as_deref(), Some("commit4"));
    assert_eq!(store.generation().unwrap().commit.as_deref(), Some("commit4"));
    assert!(std::fs::read_to_string(store.current().join("a.toml")).unwrap().contains("count = 4"));
    store.rollback().unwrap();
    store.rollback().unwrap();
    assert_eq!(store.generation().unwrap().commit.as_deref(), Some("commit2"));
    assert!(store.rollback().is_err());
}

#[test]
fn a_removed_spec_is_reported_and_the_list_follows() {
    let r = rig();
    let store = SpecStore::new(&temp("removed"));
    r.publish(&[file("a.toml", &spec_text("a", 2)), file("b.toml", &spec_text("b", 2))], "v1", "c1");
    store.update(&r.cfg).unwrap();
    r.publish(&[file("a.toml", &spec_text("a", 2))], "v2", "c2");
    let Outcome::Updated(c) = store.update(&r.cfg).unwrap() else { panic!() };
    assert_eq!(c.removed, vec!["b"]);
    assert_eq!(c.summary(), "removed b");
}

#[test]
fn a_pinned_tag_fetches_the_tag_ref() {
    let cfg = SpecsConfig { tag: "v0.2.0".into(), base: "https://x".into(), ..SpecsConfig::default() };
    assert_eq!(cfg.url(), "https://x/v0l/inference-bench/tar.gz/refs/tags/v0.2.0");
    assert_eq!(cfg.source(), "v0l/inference-bench@v0.2.0");
    let branch = SpecsConfig { base: "https://x".into(), ..SpecsConfig::default() };
    assert_eq!(branch.url(), "https://x/v0l/inference-bench/tar.gz/refs/heads/master");
}

#[test]
fn a_check_is_due_after_the_interval_or_when_the_source_changes() {
    let r = rig();
    let store = SpecStore::new(&temp("due"));
    let now = Utc::now();
    assert!(store.due(&r.cfg, now));
    r.publish(&[file("a.toml", &spec_text("a", 2))], "v1", "c1");
    store.update(&r.cfg).unwrap();
    assert!(!store.due(&r.cfg, Utc::now()));
    assert!(store.due(&r.cfg, Utc::now() + chrono::Duration::hours(25)));
    let tagged = SpecsConfig { tag: "v9".into(), ..r.cfg.clone() };
    assert!(store.due(&tagged, Utc::now()));
    assert!(!store.due(&SpecsConfig { update_hours: 0, ..r.cfg.clone() }, Utc::now() + chrono::Duration::days(9)));
}

fn put(dir: &Path, file: &str, text: &str) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join(file), text).unwrap();
}

#[test]
fn a_folder_beats_the_cache_beats_embedded_beats_the_checkout() {
    let r = rig();
    let data = temp("priority");
    let store = SpecStore::new(&data);
    r.publish(&[file("qwen3-0.6b.toml", &spec_text("qwen3-0.6b", 5)), file("cached.toml", &spec_text("cached", 5))], "v1", "c1");
    store.update(&r.cfg).unwrap();
    let folder = data.join("mine");
    put(&folder, "q.toml", &spec_text("qwen3-0.6b", 1));
    let checkout = data.join("checkout");
    put(&checkout, "llama-3.2-1b.toml", &spec_text("llama-3.2-1b", 9));
    put(&checkout, "devonly.toml", &spec_text("devonly", 9));
    let all = list(&layers(&store, std::slice::from_ref(&folder), std::slice::from_ref(&checkout)));
    let by = |n: &str| all.iter().find(|l| l.name == n).unwrap();
    assert_eq!(all.iter().filter(|l| l.name == "qwen3-0.6b").count(), 1);
    assert_eq!(by("qwen3-0.6b").origin, SpecOrigin::Folder);
    assert!(matches!(by("cached").origin, SpecOrigin::Updated(_)));
    assert_eq!(by("llama-3.2-1b").origin, SpecOrigin::Embedded);
    assert_eq!(by("devonly").origin, SpecOrigin::Checkout);
    assert_eq!(by("qwen3-0.6b").origin.label(), "your folder");
    assert!(by("cached").origin.label().starts_with("updated from GitHub v0l/inference-bench@master (c1) at "));
    assert_eq!(find("qwen3-0.6b", &layers(&store, std::slice::from_ref(&folder), &[])), Some(folder.join("q.toml")));
    assert_eq!(std::fs::read_to_string(folder.join("q.toml")).unwrap(), spec_text("qwen3-0.6b", 1));
}

#[test]
fn a_spec_needing_a_newer_engine_is_listed_but_not_loadable_and_does_not_hide_an_older_one() {
    let data = temp("newer");
    let store = SpecStore::new(&data);
    let folder = data.join("mine");
    let future = format!("min_engine = 99\n{}", spec_text("future", 2).replace("[mlp]", "[moe]\nexperts = 8\n[mlp]"));
    put(&folder, "future.toml", &future);
    put(&folder, "q.toml", &format!("min_engine = 99\n{}", spec_text("qwen3-0.6b", 2)));
    let all = list(&layers(&store, &[folder], &[]));
    let f = all.iter().find(|l| l.name == "future").unwrap();
    assert!(!f.loadable());
    assert!(f.problem.as_deref().unwrap().contains("needs a newer inference-bench"));
    let q = all.iter().find(|l| l.name == "qwen3-0.6b").unwrap();
    assert!(q.loadable());
    assert_eq!(q.origin, SpecOrigin::Embedded);
    assert!(find("future", &layers(&store, &[], &[])).is_none());
}

#[test]
fn offline_uses_the_cache_then_the_embedded_snapshot() {
    let r = rig();
    let data = temp("offline");
    let store = SpecStore::new(&data);
    let fresh = list(&layers(&store, &[], &[]));
    assert_eq!(fresh.len(), EMBEDDED.len());
    assert!(fresh.iter().all(|l| l.origin == SpecOrigin::Embedded && l.loadable()));
    r.publish(&[file("qwen3-0.6b.toml", &spec_text("qwen3-0.6b", 3))], "v1", "c1");
    store.update(&r.cfg).unwrap();
    let dead = SpecsConfig { base: "http://127.0.0.1:9".into(), timeout: Duration::from_secs(2), ..r.cfg.clone() };
    assert!(store.update(&dead).is_err());
    let all = list(&layers(&store, &[], &[]));
    let q = all.iter().find(|l| l.name == "qwen3-0.6b").unwrap();
    assert!(matches!(q.origin, SpecOrigin::Updated(_)));
    assert!(all.iter().any(|l| l.name == "llama-3.2-1b" && l.origin == SpecOrigin::Embedded));
    let state = store.state();
    assert!(!state.last_ok && state.last_success.is_some());
}
