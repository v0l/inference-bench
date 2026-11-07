use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("specs");
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|n| n.ends_with(".toml"))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    let mut table = String::from("&[\n");
    for name in &names {
        let path = dir.join(name);
        println!("cargo:rerun-if-changed={}", path.display());
        table.push_str(&format!("    ({name:?}, include_str!({:?})),\n", path.display().to_string()));
    }
    table.push_str("]\n");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("embedded_specs.rs");
    std::fs::write(out, table).unwrap();
}
