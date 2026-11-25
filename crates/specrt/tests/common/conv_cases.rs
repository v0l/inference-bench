use specrt::vae::{Grid, host};
use std::path::{Path, PathBuf};

fn cases_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/conv-cases")
}

pub fn cases() -> Vec<serde_json::Value> {
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(cases_dir().join("cases.json")).unwrap()).unwrap();
    index["cases"].as_array().unwrap().clone()
}

pub fn expected(case: &serde_json::Value) -> Vec<f32> {
    let bytes = std::fs::read(cases_dir().join(format!("{}.bin", case["name"].as_str().unwrap()))).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

pub fn field(case: &serde_json::Value, key: &str) -> usize {
    case[key].as_u64().unwrap() as usize
}

pub struct Case {
    pub conv: host::ConvWeights,
    pub input: Grid,
    pub interleave: bool,
    pub upsample: bool,
}

pub fn case(c: &serde_json::Value) -> Case {
    let (cin, cout, k, kt) = (field(c, "cin"), field(c, "cout"), field(c, "k"), field(c, "kt"));
    let (t, h, w) = (field(c, "frames"), field(c, "height"), field(c, "width"));
    let seed = c["seed"].as_u64().unwrap();
    let fan = cin * k * k * kt;
    let x = host::uniform(seed, cin * t * h * w, 1.0);
    let w_ = host::uniform(seed + 1, cout * fan, 1.0 / (fan as f64).sqrt());
    let b = host::uniform(seed + 2, cout, 0.1);
    let input = Grid::from_channels_first(&x, cin, t, h, w);
    Case { conv: host::ConvWeights { w: w_, b, cin, cout, k, kt }, input, interleave: c["interleave"].as_bool().unwrap(), upsample: c["upsample"].as_bool().unwrap() }
}

pub fn host_output(c: &Case) -> Vec<f32> {
    let x = if c.upsample { host::upsample(&c.input) } else { c.input.clone() };
    let y = host::conv(&x, &c.conv, None);
    let y = if c.interleave { host::interleave(&y) } else { y };
    y.to_channels_first()
}

pub fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[allow(dead_code)]
fn main() {}
