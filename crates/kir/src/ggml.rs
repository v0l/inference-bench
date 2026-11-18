use crate::{Language, Target, fnv};

pub const SUPER_BLOCK: usize = 256;
pub const BLOCK_BYTES: [(&str, usize); 4] = [("Q4_K", 144), ("Q5_K", 176), ("Q6_K", 224), ("Q8_0", 272)];
pub const TABLE_KERNELS: &[&str] = &["gg_dequant_rows", "gg_dequant_bf16"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GgmlKernel {
    pub target: Target,
}

impl GgmlKernel {
    pub fn new(t: &Target) -> GgmlKernel {
        GgmlKernel { target: *t }
    }

    pub fn describe(&self) -> String {
        format!("{}\nggml decoders Q4_K Q5_K Q6_K Q8_0 bf16 f16", self.target.describe())
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }

    pub fn emit(&self) -> Result<String, String> {
        match self.target.language {
            Language::Cuda => Ok(crate::cuda::emit_ggml()),
            Language::Wgsl => Err("ggml decoders on wgsl are emitted per kernel with wgsl::ggml_decode".into()),
        }
    }
}
