use inference_bench_core::discover::inspect;
use inference_bench_core::hf::{HF_BASE, Hub};
use std::time::Instant;

#[test]
#[ignore = "talks to huggingface.co, prints how long each repo takes to inspect"]
fn live_timing_of_a_qwen_search_inspection() {
    let hub = Hub::new(HF_BASE, "");
    let t = Instant::now();
    let repos = hub.search("Qwen3.5", 25).unwrap();
    println!("search: {} repos in {:?}", repos.len(), t.elapsed());
    let start = Instant::now();
    let times = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for r in &repos {
            let (hub, times) = (&hub, &times);
            s.spawn(move || {
                let t = Instant::now();
                let files = hub.files(&r.id).map(|f| f.len());
                let tree = t.elapsed();
                let out = inspect(hub, &r.id);
                times.lock().unwrap().push((t.elapsed(), tree, r.id.clone(), files.ok(), out.is_ok()));
            });
        }
    });
    let mut v = times.into_inner().unwrap();
    v.sort();
    for (total, tree, id, files, ok) in &v {
        println!("{total:>8.2?} (tree {tree:>8.2?}) {id} files={files:?} ok={ok}");
    }
    println!("all done in {:?}", start.elapsed());
}
