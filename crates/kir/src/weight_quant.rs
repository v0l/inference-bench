use crate::{Language, Target, fnv};

pub const GROUP: u32 = 16;
pub const SCALE_HEADER: u32 = 16;
pub const AMAX: &str = "nvfp4_amax";
pub const QUANTIZE: &str = "nvfp4_quantize";
pub const KERNELS: [&str; 2] = [AMAX, QUANTIZE];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightQuant {
    pub target: Target,
    pub amax_threads: u32,
    pub quant_threads: u32,
    pub fp4_max: u32,
    pub e4m3_max: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub grid: u32,
    pub threads: u32,
}

impl WeightQuant {
    pub fn nvfp4(t: &Target) -> Result<WeightQuant, String> {
        let k = WeightQuant { target: *t, amax_threads: 256, quant_threads: 128, fp4_max: 6, e4m3_max: 448 };
        k.validate()?;
        Ok(k)
    }

    pub fn validate(&self) -> Result<(), String> {
        let t = &self.target;
        if t.language != Language::Cuda {
            return Err(format!("nvfp4 weight quant has no {:?} lowering", t.language));
        }
        for n in [self.amax_threads, self.quant_threads] {
            if n == 0 || n > t.max_threads || n % t.warp_size != 0 {
                return Err(format!("nvfp4 weight quant: {n} threads does not fit the target"));
            }
        }
        Ok(())
    }

    pub fn amax_launch(&self) -> Launch {
        Launch { grid: self.target.sm_count.max(1) * 8, threads: self.amax_threads }
    }

    pub fn quant_launch(&self, groups: usize) -> Launch {
        Launch { grid: groups.div_ceil(self.quant_threads as usize).max(1) as u32, threads: self.quant_threads }
    }

    pub fn describe(&self) -> String {
        format!("{}\nnvfp4 weight quant group={GROUP} header={SCALE_HEADER} amax_threads={} quant_threads={} fp4_max={} e4m3_max={}", self.target.describe(), self.amax_threads, self.quant_threads, self.fp4_max, self.e4m3_max)
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }
}
