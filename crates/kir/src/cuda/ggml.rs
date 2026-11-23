mod dec;
mod decode;
mod dequant;
mod iq;
mod rowmath;
mod rows;

pub fn emit_ggml() -> crate::KirSource {
    crate::KirSource::emitted(format!("{}\n{}\n{}\n", decode::body(), super::quant::dequant_bf16("gg_dequant_bf16"), dec::body()))
}
