const HIDDEN: u32 = 1024u;
const INTER: u32 = 3072u;
const VOCAB: u32 = 151936u;
const HEADS: u32 = 16u;
const KV_HEADS: u32 = 8u;
const HEAD_DIM: u32 = 128u;
const GROUP: u32 = 2u;
const QK_NORM: bool = true;
const EPS: f32 = 1e-6;
const THETA: f32 = 1e6;
const ROPE_KIND: u32 = 0u;
const ROPE_FACTOR: f32 = 4e0;
const ROPE_LOW_FREQ: f32 = 1e0;
const ROPE_HIGH_FREQ: f32 = 4e0;
const ROPE_ORIG_POS: f32 = 8.192e3;
const ROPE_MSCALE: f32 = 1e0;
const ACT_KIND: u32 = 0u;
const MAX_CTX: u32 = 4096u;
const Q_DIM: u32 = 2048u;
const QKV_ROW: u32 = 4096u;
const ROT: u32 = 128u;
const NEG_INF: f32 = -3.0e38;
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
struct P { a: vec4<u32>, b: vec4<u32> };
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 8388608u;
    if (idx >= p.a.x * p.a.y) {
        return;
    }
    y[idx] = y[idx] + bias[idx % p.a.y];
}
