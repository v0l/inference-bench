use super::compile::{Dims, compile};
use super::graph::VaePlan;
use super::host::Grid;
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tiling {
    pub tile: usize,
    pub overlap: usize,
}

impl Tiling {
    pub fn stride(&self) -> usize {
        self.tile - self.overlap
    }

    pub fn starts(&self, extent: usize) -> Vec<usize> {
        if extent <= self.tile {
            return vec![0];
        }
        let mut out: Vec<usize> = (0..).map(|i| i * self.stride()).take_while(|s| s + self.tile < extent).collect();
        out.push(extent - self.tile);
        out.dedup();
        out
    }
}

pub fn arena_bytes(plan: &VaePlan, target: &kir::Target, tile_h: usize, tile_w: usize) -> Result<u64> {
    let first = compile(plan, Dims { t: 1, h: tile_h, w: tile_w }, target, true, true)?.arena.size;
    if !plan.temporal {
        return Ok(first);
    }
    let steady = compile(plan, Dims { t: 1, h: tile_h, w: tile_w }, target, true, false)?.arena.size;
    Ok(first.max(steady))
}

pub fn choose(plan: &VaePlan, target: &kir::Target, latent_h: usize, latent_w: usize, budget: u64, align: usize, overlap_frac: f64) -> Result<Option<Tiling>> {
    let full = arena_bytes(plan, target, latent_h, latent_w)?;
    if full <= budget {
        return Ok(None);
    }
    let longest = latent_h.max(latent_w);
    let mut best = None;
    let mut tile = align;
    while tile < longest {
        let (h, w) = (tile.min(latent_h), tile.min(latent_w));
        if arena_bytes(plan, target, h, w)? > budget {
            break;
        }
        best = Some(tile);
        tile += align;
    }
    let Some(tile) = best else { anyhow::bail!("no tile of {align} latent pixels fits {budget} bytes of activations") };
    let overlap = ((tile as f64 * overlap_frac).round() as usize).min(tile - 1);
    Ok(Some(Tiling { tile, overlap }))
}

fn ramp(i: usize, extent: usize, lead: usize, trail: usize) -> f32 {
    let up = if lead == 0 { 1.0 } else { ((i as f32 + 0.5) / lead as f32).min(1.0) };
    let down = if trail == 0 { 1.0 } else { ((extent - i) as f32 - 0.5).min(trail as f32) / trail as f32 };
    up.min(down).max(1e-6)
}

pub fn decode_tiled(latent: &Grid, scale: usize, tiling: Tiling, decode: &mut dyn FnMut(&Grid) -> Result<Grid>) -> Result<Grid> {
    ensure!(tiling.overlap < tiling.tile, "overlap {} is not below the tile {}", tiling.overlap, tiling.tile);
    let (ys, xs) = (tiling.starts(latent.h), tiling.starts(latent.w));
    let mut acc: Option<Grid> = None;
    let mut weight: Vec<f32> = Vec::new();
    for (yi, &y0) in ys.iter().enumerate() {
        for (xi, &x0) in xs.iter().enumerate() {
            let (th, tw) = (tiling.tile.min(latent.h), tiling.tile.min(latent.w));
            let out = decode(&latent.crop(y0, x0, th, tw))?;
            let (oh, ow) = (th * scale, tw * scale);
            ensure!(out.h == oh && out.w == ow, "a tile decoded to {}x{}, not {oh}x{ow}", out.h, out.w);
            let total = acc.get_or_insert_with(|| Grid::zeros(out.c, out.t, latent.h * scale, latent.w * scale));
            if weight.is_empty() {
                weight = vec![0.0; total.t * total.h * total.w];
            }
            let lap = |i: usize, starts: &[usize]| -> (usize, usize) {
                let lead = if i > 0 { (starts[i - 1] + tiling.tile).saturating_sub(starts[i]) * scale } else { 0 };
                let trail = if i + 1 < starts.len() { (starts[i] + tiling.tile).saturating_sub(starts[i + 1]) * scale } else { 0 };
                (lead, trail)
            };
            let (ly, ty) = lap(yi, &ys);
            let (lx, tx) = lap(xi, &xs);
            for t in 0..out.t {
                for y in 0..oh {
                    let wy = ramp(y, oh, ly, ty);
                    for x in 0..ow {
                        let wgt = wy * ramp(x, ow, lx, tx);
                        let (gy, gx) = (y0 * scale + y, x0 * scale + x);
                        let at = (t * total.h + gy) * total.w + gx;
                        weight[at] += wgt;
                        let src = ((t * oh + y) * ow + x) * out.c;
                        for c in 0..out.c {
                            total.data[at * out.c + c] += wgt * out.data[src + c];
                        }
                    }
                }
            }
        }
    }
    let mut total = acc.expect("at least one tile");
    let c = total.c;
    for (p, w) in weight.iter().enumerate() {
        for v in &mut total.data[p * c..(p + 1) * c] {
            *v /= w;
        }
    }
    Ok(total)
}
