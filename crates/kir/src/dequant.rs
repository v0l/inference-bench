use crate::{Language, Target, fnv};

pub const KERNEL: &str = "block_dequant_bf16";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Elem {
    E2m1,
    E4m3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scale {
    E8m0,
    E4m3,
}

impl Elem {
    pub fn bits(self) -> u32 {
        match self {
            Elem::E2m1 => 4,
            Elem::E4m3 => 8,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Elem::E2m1 => "e2m1",
            Elem::E4m3 => "e4m3",
        }
    }
}

impl Scale {
    fn label(self) -> &'static str {
        match self {
            Scale::E8m0 => "e8m0",
            Scale::E4m3 => "e4m3",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockDequant {
    pub target: Target,
    pub elem: Elem,
    pub scale: Scale,
    pub group: u32,
    pub threads: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub grid: u32,
    pub threads: u32,
}

impl BlockDequant {
    pub fn new(t: &Target, elem: Elem, scale: Scale, group: u32) -> Result<BlockDequant, String> {
        let k = BlockDequant { target: *t, elem, scale, group, threads: 256 };
        k.validate()?;
        Ok(k)
    }

    pub fn validate(&self) -> Result<(), String> {
        let t = &self.target;
        if t.language != Language::Cuda {
            return Err(format!("block dequant {} has no {:?} lowering", self.label(), t.language));
        }
        if self.group == 0 || !(self.group * self.elem.bits()).is_multiple_of(8) {
            return Err(format!("block dequant {}: a group of {} does not fill whole bytes", self.label(), self.group));
        }
        if self.threads > t.max_threads || !self.threads.is_multiple_of(t.warp_size) {
            return Err(format!("block dequant {}: {} threads does not fit the target", self.label(), self.threads));
        }
        Ok(())
    }

    pub fn name(&self) -> &'static str {
        KERNEL
    }

    fn label(&self) -> String {
        format!("{} {} g{}", self.elem.label(), self.scale.label(), self.group)
    }

    pub fn group_bytes(&self) -> u32 {
        self.group * self.elem.bits() / 8
    }

    pub fn launch(&self, groups: usize) -> Launch {
        Launch { grid: groups.div_ceil(self.threads as usize).max(1) as u32, threads: self.threads }
    }

    pub fn describe(&self) -> String {
        format!("{}\nblock dequant elem={:?} scale={:?} group={} threads={}", self.target.describe(), self.elem, self.scale, self.group, self.threads)
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }
}
