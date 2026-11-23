use crate::{Language, Target, fnv};

pub const READ: &str = "read_sum";
pub const COPY: &str = "copy";
pub const KERNELS: [&str; 2] = [READ, COPY];
pub const VECTOR_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bandwidth {
    pub target: Target,
    pub threads: u32,
    pub blocks_per_sm: u32,
}

impl Bandwidth {
    pub fn probe(t: &Target) -> Result<Bandwidth, String> {
        let b = Bandwidth { target: *t, threads: 512, blocks_per_sm: 8 };
        if t.language != Language::Cuda {
            return Err(format!("bandwidth probe has no {:?} lowering", t.language));
        }
        if b.threads > t.max_threads {
            return Err(format!("bandwidth probe: {} threads does not fit the target", b.threads));
        }
        Ok(b)
    }

    pub fn grid(&self) -> u32 {
        self.target.sm_count.max(1) * self.blocks_per_sm
    }

    pub fn describe(&self) -> String {
        format!("{}\nbandwidth probe threads={} blocks_per_sm={}", self.target.describe(), self.threads, self.blocks_per_sm)
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }
}
