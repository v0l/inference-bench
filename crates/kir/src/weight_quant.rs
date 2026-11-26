use crate::{Language, Target, fnv};

pub const GROUP: u32 = 16;
pub const SCALE_HEADER: u32 = 16;
pub const AMAX: &str = "nvfp4_amax";
pub const QUANTIZE: &str = "nvfp4_quantize";
pub const KERNELS: [&str; 2] = [AMAX, QUANTIZE];
pub const NVFP4_SEARCH: &str = "nvfp4_search";
pub const FP8_CHANNEL_SEARCH: &str = "fp8_channel_search";
pub const SEARCH_KERNELS: [&str; 3] = [AMAX, NVFP4_SEARCH, FP8_CHANNEL_SEARCH];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleSearch {
    pub quant: WeightQuant,
    pub nvfp4_steps: &'static [i32],
    pub fp8_factors: &'static [f32],
    pub row_threads: u32,
}

impl ScaleSearch {
    pub fn new(t: &Target, nvfp4_steps: &'static [i32], fp8_factors: &'static [f32]) -> Result<ScaleSearch, String> {
        let k = ScaleSearch { quant: WeightQuant::nvfp4(t)?, nvfp4_steps, fp8_factors, row_threads: 64 };
        k.validate()?;
        Ok(k)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.quant.validate()?;
        let t = &self.quant.target;
        if self.row_threads == 0 || self.row_threads > t.max_threads || self.row_threads % t.warp_size != 0 {
            return Err(format!("scale search: {} row threads does not fit the target", self.row_threads));
        }
        if self.fp8_factors.is_empty() || self.fp8_factors.iter().any(|f| !f.is_finite() || *f <= 0.0) {
            return Err(format!("scale search: fp8 factors {:?} must be positive", self.fp8_factors));
        }
        if self.nvfp4_steps.is_empty() || self.nvfp4_steps.iter().any(|s| *s == 0 || s.abs() > 0x7e) {
            return Err(format!("scale search: nvfp4 steps {:?} must be non zero code offsets", self.nvfp4_steps));
        }
        Ok(())
    }

    pub fn group_launch(&self, groups: usize) -> Launch {
        self.quant.quant_launch(groups)
    }

    pub fn row_launch(&self, rows: usize) -> Launch {
        Launch { grid: rows.div_ceil(self.row_threads as usize).max(1) as u32, threads: self.row_threads }
    }

    pub fn describe(&self) -> String {
        format!("{}\nscale search nvfp4_steps={:?} fp8_factors={:?} row_threads={}", self.quant.describe(), self.nvfp4_steps, self.fp8_factors, self.row_threads)
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }
}

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
