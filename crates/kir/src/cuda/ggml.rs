mod dec;
mod decode;

pub fn emit_ggml() -> String {
    format!("{}\n{}\n{}\n", decode::body(), super::quant::dequant_bf16("gg_dequant_bf16"), dec::body())
}
