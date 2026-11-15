#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Language {
    Cuda,
    Wgsl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Feature {
    WarpShuffle,
    CpAsync,
    Mma16816,
    Tma,
    Subgroups,
    ShaderF16,
    Fp8Mma,
    BlockScaledFp4,
}

const FEATURES: [Feature; 8] = [
    Feature::WarpShuffle,
    Feature::CpAsync,
    Feature::Mma16816,
    Feature::Tma,
    Feature::Subgroups,
    Feature::ShaderF16,
    Feature::Fp8Mma,
    Feature::BlockScaledFp4,
];

const SMEM_RESERVED_PER_BLOCK: u32 = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    pub language: Language,
    pub compute_capability: (u32, u32),
    pub arch_specific: bool,
    pub sm_count: u32,
    pub warp_size: u32,
    pub max_threads: u32,
    pub smem_limit: u32,
    pub features: u32,
    pub graphs: bool,
    pub pdl: bool,
}

fn bit(f: Feature) -> u32 {
    1 << f as u32
}

impl Target {
    pub fn cuda(cc_major: u32, cc_minor: u32, sm_count: u32, smem_limit: u32) -> Target {
        let cc = cc_major * 10 + cc_minor;
        let mut features = bit(Feature::WarpShuffle);
        if cc >= 80 {
            features |= bit(Feature::CpAsync) | bit(Feature::Mma16816) | bit(Feature::ShaderF16);
        }
        if cc >= 89 {
            features |= bit(Feature::Fp8Mma);
        }
        if cc >= 90 {
            features |= bit(Feature::Tma);
        }
        if cc_major == 12 {
            features |= bit(Feature::BlockScaledFp4);
        }
        Target {
            language: Language::Cuda,
            compute_capability: (cc_major, cc_minor),
            arch_specific: false,
            sm_count,
            warp_size: 32,
            max_threads: 1024,
            smem_limit,
            features,
            graphs: true,
            pdl: cc >= 90,
        }
    }

    pub fn sm_120() -> Target {
        Target::cuda(12, 0, 188, 99 * 1024)
    }

    pub fn sm_80() -> Target {
        Target::cuda(8, 0, 108, 163 * 1024)
    }

    pub fn wgpu(max_threads: u32, smem_limit: u32) -> Target {
        Target {
            language: Language::Wgsl,
            compute_capability: (0, 0),
            arch_specific: false,
            sm_count: 0,
            warp_size: 1,
            max_threads,
            smem_limit,
            features: 0,
            graphs: false,
            pdl: false,
        }
    }

    pub fn wgpu_baseline() -> Target {
        Target::wgpu(256, 16 * 1024)
    }

    pub fn named(name: &str) -> Option<Target> {
        match name {
            "sm_120" => Some(Target::sm_120()),
            "sm_80" => Some(Target::sm_80()),
            "wgpu" => Some(Target::wgpu_baseline().with_subgroups(32).with_feature(Feature::ShaderF16)),
            "wgpu-nosubgroups" => Some(Target::wgpu_baseline()),
            _ => None,
        }
    }

    pub fn with_feature(mut self, f: Feature) -> Target {
        self.features |= bit(f);
        self
    }

    pub fn with_subgroups(mut self, size: u32) -> Target {
        self.warp_size = size;
        self.with_feature(Feature::Subgroups)
    }

    pub fn has(&self, f: Feature) -> bool {
        self.features & bit(f) != 0
    }

    pub fn smem_per_sm(&self) -> u32 {
        self.smem_limit + SMEM_RESERVED_PER_BLOCK
    }

    pub fn smem_reserved_per_block(&self) -> u32 {
        SMEM_RESERVED_PER_BLOCK
    }

    pub fn describe(&self) -> String {
        let features: Vec<String> = FEATURES
        .into_iter()
        .filter(|f| self.has(*f))
        .map(|f| format!("{f:?}"))
        .collect();
        format!(
            "target {:?} cc={}.{}{} sms={} warp={} threads={} smem={} graphs={} pdl={} features=[{}]",
            self.language,
            self.compute_capability.0,
            self.compute_capability.1,
            if self.arch_specific { "a" } else { "" },
            self.sm_count,
            self.warp_size,
            self.max_threads,
            self.smem_limit,
            self.graphs,
            self.pdl,
            features.join(",")
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockSum {
    WarpShuffle,
    Subgroup,
    SharedTree,
}

impl BlockSum {
    pub fn for_target(t: &Target) -> BlockSum {
        match t.language {
            Language::Cuda => BlockSum::WarpShuffle,
            Language::Wgsl if t.has(Feature::Subgroups) => BlockSum::Subgroup,
            Language::Wgsl => BlockSum::SharedTree,
        }
    }

    pub fn needs(self) -> &'static [Feature] {
        match self {
            BlockSum::WarpShuffle => &[Feature::WarpShuffle],
            BlockSum::Subgroup => &[Feature::Subgroups],
            BlockSum::SharedTree => &[],
        }
    }

    pub fn language(self) -> Language {
        match self {
            BlockSum::WarpShuffle => Language::Cuda,
            BlockSum::Subgroup | BlockSum::SharedTree => Language::Wgsl,
        }
    }
}
