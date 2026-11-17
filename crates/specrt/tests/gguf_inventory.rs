use std::collections::BTreeMap;

#[test]
#[ignore = "set SPECRT_GGUF_FILE to a GGUF to list its architecture, tensor types and unmapped tensors"]
fn inventory_of_a_gguf_file() {
    let path = std::env::var("SPECRT_GGUF_FILE").expect("SPECRT_GGUF_FILE");
    let g = specrt::gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
    println!("architecture {}", g.architecture().unwrap());
    let mut by_type: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut shapes: BTreeMap<String, (String, Vec<usize>)> = BTreeMap::new();
    for t in g.tensors.values() {
        let e = by_type.entry(t.kind.name()).or_default();
        e.0 += 1;
        e.1 += t.bytes;
        let generic: String = t.name.split('.').map(|p| if p.chars().all(|c| c.is_ascii_digit()) { "N" } else { p }).collect::<Vec<_>>().join(".");
        shapes.entry(format!("{generic} {}", t.kind.name())).or_insert((t.name.clone(), t.shape.clone()));
    }
    for (k, (n, b)) in &by_type {
        println!("type {k}: {n} tensors, {b} bytes");
    }
    for (k, (n, s)) in &shapes {
        println!("{k} e.g. {n} {s:?}");
    }
    let mut keys: Vec<_> = g.metadata.keys().filter(|k| !k.starts_with("tokenizer.ggml.") || k.ends_with("pre") || k.ends_with("model")).collect();
    keys.sort();
    for k in keys {
        println!("kv {k} = {:?}", g.metadata[k]);
    }
}
