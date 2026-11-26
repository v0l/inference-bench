use crate::nvfp4::{Fp4, GROUP, e2m1_to_f32, f32_to_e2m1};
use crate::weights::{bf16_to_f32, e4m3_to_f32, f32_to_e4m3, fp8_scales, nvfp4_global_scale};
use std::borrow::Cow;

pub const NVFP4_SCALE_STEPS: [i32; 5] = [-1, -2, -3, -4, 1];
pub const FP8_SCALE_FACTORS: [f32; 5] = [1.0, 0.96875, 0.9375, 0.90625, 0.875];
pub const FORMAT_VERSION: u32 = 1;
pub const FP8_CHANNEL_MEAN_KL_BOUND: f64 = 0.02;
pub const NVFP4_G16_MEAN_KL_BOUND: f64 = 0.155;
pub const FP8_CHANNEL_TOP1_FLOOR: f64 = 0.92;
pub const NVFP4_G16_TOP1_FLOOR: f64 = 0.72;

fn nvfp4_group_error(group: &[u16], code: u8, global: f32) -> f64 {
    let s = e4m3_to_f32(code) / global;
    group
        .iter()
        .map(|&v| {
            let w = bf16_to_f32(v);
            let q = if s > 0.0 { f32_to_e2m1(w / s) } else { 0 };
            let d = w as f64 - (e2m1_to_f32(q) * s) as f64;
            d * d
        })
        .sum()
}

pub fn nvfp4_group_code(group: &[u16], global: f32) -> u8 {
    let amax = group.iter().map(|&v| bf16_to_f32(v).abs()).fold(0f32, f32::max);
    let base = f32_to_e4m3((amax / 6.0 * global).min(448.0));
    if amax == 0.0 {
        return base;
    }
    let mut best = (base, nvfp4_group_error(group, base, global));
    for step in NVFP4_SCALE_STEPS {
        let c = base as i32 + step;
        if !(1..=0x7e).contains(&c) {
            continue;
        }
        let e = nvfp4_group_error(group, c as u8, global);
        if e < best.1 {
            best = (c as u8, e);
        }
    }
    best.0
}

pub fn nvfp4_search(w: &[u16], cols: usize) -> Fp4<'static> {
    nvfp4_search_with(w, cols, nvfp4_global_scale(w))
}

pub fn nvfp4_search_with(w: &[u16], cols: usize, global: f32) -> Fp4<'static> {
    let mut packed = vec![0u8; w.len() / 2];
    let mut scales = vec![0u8; w.len() / GROUP];
    let rows = w.len() / cols;
    rows_in_parallel(w, cols, &mut packed, cols / 2, &mut scales, cols / GROUP, |row, out, codes| {
        for ((g, group), code) in row.chunks_exact(GROUP).enumerate().zip(codes.iter_mut()) {
            *code = nvfp4_group_code(group, global);
            let s = e4m3_to_f32(*code) / global;
            for (i, &v) in group.iter().enumerate() {
                let q = if s > 0.0 { f32_to_e2m1(bf16_to_f32(v) / s) } else { 0 };
                let at = g * GROUP + i;
                out[at / 2] |= if at.is_multiple_of(2) { q } else { q << 4 };
            }
        }
    });
    Fp4::new(Cow::Owned(packed), Cow::Owned(scales), global, rows, cols).expect("nvfp4 search shape")
}

fn fp8_row_error(row: &[u16], s: f32) -> f64 {
    row.iter()
        .map(|&v| {
            let w = bf16_to_f32(v);
            let d = w as f64 - (e4m3_to_f32(f32_to_e4m3(w / s)) * s) as f64;
            d * d
        })
        .sum()
}

pub fn fp8_row_scale(row: &[u16]) -> f32 {
    let amax = row.iter().map(|&v| bf16_to_f32(v).abs()).fold(0f32, f32::max);
    if amax == 0.0 {
        return 1.0;
    }
    let mut best: Option<(f32, f64)> = None;
    for f in FP8_SCALE_FACTORS {
        let s = amax / 448.0 * f;
        let e = fp8_row_error(row, s);
        if best.is_none_or(|(_, b)| e < b) {
            best = Some((s, e));
        }
    }
    best.map_or(1.0, |(s, _)| s)
}

pub fn fp8_channel_search(w: &[u16], cols: usize) -> (Vec<u8>, Vec<f32>) {
    let rows = w.len() / cols;
    let kb = cols.div_ceil(crate::codegen::FP8_BLOCK);
    let mut values = vec![0u8; w.len()];
    let mut row_scales = vec![0f32; rows];
    rows_in_parallel(w, cols, &mut values, cols, &mut row_scales, 1, |row, out, scale| {
        scale[0] = fp8_row_scale(row);
        for (o, &v) in out.iter_mut().zip(row) {
            *o = f32_to_e4m3(bf16_to_f32(v) / scale[0]);
        }
    });
    let grid: Vec<f32> = row_scales.iter().flat_map(|&s| std::iter::repeat_n(s, kb)).collect();
    let map: Vec<u32> = (0..rows).map(|r| (r * kb) as u32).collect();
    (values, fp8_scales(&grid, &map))
}

pub fn squared_error_nvfp4(w: &[u16], m: &Fp4) -> f64 {
    let cols = m.cols;
    w.chunks_exact(cols).enumerate().map(|(r, row)| row.iter().zip(m.dequant_row(r)).map(|(&v, q)| (bf16_to_f32(v) as f64 - q as f64).powi(2)).sum::<f64>()).sum()
}

pub fn squared_error_fp8_channel(w: &[u16], cols: usize, values: &[u8], scales: &[f32]) -> f64 {
    w.iter()
        .zip(values)
        .enumerate()
        .map(|(i, (&v, &q))| {
            let (r, c) = (i / cols, i % cols);
            let s = scales[scales[r].to_bits() as usize + c / crate::codegen::FP8_BLOCK];
            (bf16_to_f32(v) as f64 - (e4m3_to_f32(q) * s) as f64).powi(2)
        })
        .sum()
}

pub const PROMPTS: &str = include_str!("../quantcheck/prompts.toml");

#[derive(serde::Deserialize)]
struct PromptFile {
    prompt: Vec<Prompt>,
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Prompt {
    pub name: String,
    pub text: String,
}

pub fn prompt_set() -> Vec<Prompt> {
    toml::from_str::<PromptFile>(PROMPTS).expect("quantcheck prompts").prompt
}

pub fn prompt_set_hash() -> String {
    use sha2::Digest;
    sha2::Sha256::digest(PROMPTS.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect()
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LogitStats {
    pub positions: usize,
    pub mean_kl: f64,
    pub max_kl: f64,
    pub top1: f64,
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) as f64;
    let lse = max + row.iter().map(|&v| (v as f64 - max).exp()).sum::<f64>().ln();
    row.iter().map(|&v| v as f64 - lse).collect()
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

pub fn logit_stats(reference: &[&[f32]], quantised: &[&[f32]], vocab: usize) -> LogitStats {
    let (mut positions, mut sum, mut max, mut same) = (0usize, 0f64, 0f64, 0usize);
    for (a, b) in reference.iter().zip(quantised) {
        assert_eq!(a.len(), b.len());
        for (p, q) in a.chunks_exact(vocab).zip(b.chunks_exact(vocab)) {
            let (lp, lq) = (log_softmax(p), log_softmax(q));
            let kl = lp.iter().zip(&lq).map(|(x, y)| x.exp() * (x - y)).sum::<f64>().max(0.0);
            sum += kl;
            max = max.max(kl);
            same += (argmax(p) == argmax(q)) as usize;
            positions += 1;
        }
    }
    let n = positions.max(1) as f64;
    LogitStats { positions, mean_kl: sum / n, max_kl: max, top1: same as f64 / n }
}

pub struct DeviceFp8 {
    pub values: cudarc::driver::CudaSlice<u8>,
    pub row_scales: cudarc::driver::CudaSlice<f32>,
}

fn search_kernels(stream: &std::sync::Arc<cudarc::driver::CudaStream>, cache_dir: &std::path::Path) -> anyhow::Result<(kir::weight_quant::ScaleSearch, crate::jit::Loaded)> {
    let ctx = stream.context();
    let target = crate::codegen::cuda_target(ctx)?;
    let search = kir::weight_quant::ScaleSearch::new(&target, &NVFP4_SCALE_STEPS, &FP8_SCALE_FACTORS).map_err(anyhow::Error::msg)?;
    let module = crate::codegen::Module {
        source: kir::cuda::emit_scale_search(&search).map_err(anyhow::Error::msg)?,
        kernels: kir::weight_quant::SEARCH_KERNELS.to_vec(),
        smem: std::collections::BTreeMap::new(),
        arch_features: false,
        kir_hash: search.hash(),
    };
    Ok((search, crate::jit::Jit::new(ctx.clone(), cache_dir.to_path_buf())?.load(&module)?))
}

fn launch(l: kir::weight_quant::Launch) -> cudarc::driver::LaunchConfig {
    cudarc::driver::LaunchConfig { grid_dim: (l.grid, 1, 1), block_dim: (l.threads, 1, 1), shared_mem_bytes: 0 }
}

pub fn nvfp4_search_on_device(stream: &std::sync::Arc<cudarc::driver::CudaStream>, cache_dir: &std::path::Path, w: &[u16], cols: usize) -> anyhow::Result<crate::nvfp4::DeviceFp4> {
    use cudarc::driver::PushKernelArg;
    anyhow::ensure!(cols.is_multiple_of(GROUP) && w.len().is_multiple_of(cols), "nvfp4 rows must be a multiple of {GROUP} wide");
    let (search, loaded) = search_kernels(stream, cache_dir)?;
    let amax_kernel = loaded.function(kir::weight_quant::AMAX)?;
    let search_kernel = loaded.function(kir::weight_quant::NVFP4_SEARCH)?;
    let dev_w = stream.clone_htod(w)?;
    let mut amax = stream.alloc_zeros::<u32>(1)?;
    let groups = w.len() / GROUP;
    let mut packed = stream.alloc_zeros::<u8>(w.len() / 2)?;
    let mut scales = stream.alloc_zeros::<u8>(crate::nvfp4::SCALE_HEADER + groups)?;
    let n = w.len() as u64;
    unsafe { stream.launch_builder(&amax_kernel).arg(&dev_w).arg(&n).arg(&mut amax).launch(launch(search.quant.amax_launch())) }?;
    let g = groups as u64;
    unsafe { stream.launch_builder(&search_kernel).arg(&dev_w).arg(&g).arg(&amax).arg(&mut packed).arg(&mut scales).launch(launch(search.group_launch(groups))) }?;
    Ok(crate::nvfp4::DeviceFp4 { packed, scales })
}

pub fn fp8_channel_search_on_device(stream: &std::sync::Arc<cudarc::driver::CudaStream>, cache_dir: &std::path::Path, w: &[u16], cols: usize) -> anyhow::Result<DeviceFp8> {
    use cudarc::driver::PushKernelArg;
    anyhow::ensure!(cols > 0 && w.len().is_multiple_of(cols), "fp8 channel search needs whole rows");
    let (search, loaded) = search_kernels(stream, cache_dir)?;
    let kernel = loaded.function(kir::weight_quant::FP8_CHANNEL_SEARCH)?;
    let rows = w.len() / cols;
    let dev_w = stream.clone_htod(w)?;
    let mut values = stream.alloc_zeros::<u8>(w.len())?;
    let mut row_scales = stream.alloc_zeros::<f32>(rows.max(1))?;
    let (r, c) = (rows as u64, cols as u64);
    unsafe { stream.launch_builder(&kernel).arg(&dev_w).arg(&r).arg(&c).arg(&mut values).arg(&mut row_scales).launch(launch(search.row_launch(rows))) }?;
    Ok(DeviceFp8 { values, row_scales })
}

fn rows_in_parallel<A: Send, B: Send>(w: &[u16], cols: usize, a: &mut [A], a_cols: usize, b: &mut [B], b_cols: usize, f: impl Fn(&[u16], &mut [A], &mut [B]) + Sync) {
    let rows = w.len() / cols;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = rows.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for ((src, da), db) in w.chunks(per * cols).zip(a.chunks_mut(per * a_cols)).zip(b.chunks_mut(per * b_cols)) {
            let f = &f;
            s.spawn(move || {
                for ((row, ra), rb) in src.chunks_exact(cols).zip(da.chunks_exact_mut(a_cols)).zip(db.chunks_exact_mut(b_cols)) {
                    f(row, ra, rb);
                }
            });
        }
    });
}
