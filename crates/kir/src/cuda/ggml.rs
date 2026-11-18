const DECODE: &str = include_str!("ggml/decode.cu");
const TO_BF16: &str = include_str!("ggml/to_bf16.cu");
const DEC: &str = include_str!("ggml/dec.cu");

pub fn emit_ggml() -> String {
    format!("{DECODE}\n{TO_BF16}\n{DEC}\n")
}
