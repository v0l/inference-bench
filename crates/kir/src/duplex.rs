use crate::fnv;

pub const KERNELS: &[&str] = &["duplex_join", "duplex_exact_logits", "duplex_edit_logits", "duplex_allow", "duplex_user_latent", "duplex_state_move"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DuplexShape {
    pub hidden: usize,
    pub vocab: usize,
    pub latent: usize,
}

impl DuplexShape {
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden == 0 || self.vocab == 0 || self.latent == 0 {
            return Err(format!("duplex kernels need a hidden, vocab and latent width, got {self:?}"));
        }
        Ok(())
    }

    pub fn hash(&self) -> String {
        fnv(format!("duplex {} {} {}", self.hidden, self.vocab, self.latent).bytes())
    }
}
