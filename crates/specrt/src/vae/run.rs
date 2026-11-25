use super::compile::{Compiled, Pro, Step};
use super::graph::VaeOp;
use super::host::{self, Grid, HostVae, Stream};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeMap;

pub struct Runner<'a> {
    pub vae: &'a HostVae,
    pub compiled: &'a Compiled,
}

fn apply_grid(vae: &HostVae, op: usize, x: &Grid, stats: &[(f64, f64)]) -> Grid {
    let VaeOp::GroupNorm { eps, act, .. } = &vae.plan.ops[op] else { unreachable!() };
    let (gamma, beta) = vae.norm_weights(op).expect("group norm weights");
    let mut y = x.clone();
    host::group_apply(&mut y, stats, gamma, beta, *eps, *act);
    y
}

impl Runner<'_> {
    pub fn run(&self, latent: &Grid, stream: &mut Stream) -> Result<Grid> {
        let s = &self.compiled.schedule;
        let mut grids: BTreeMap<usize, Grid> = BTreeMap::from([(s.input, latent.clone())]);
        let mut stats: BTreeMap<usize, Vec<(f64, f64)>> = BTreeMap::new();
        let plan = &self.vae.plan;
        let get = |g: &BTreeMap<usize, Grid>, v: usize| g.get(&v).cloned().with_context(|| format!("value {v} is read before it is written"));
        for step in &s.steps {
            match step {
                Step::Conv { op, input, prologue, residual, stats: st, clamp, output } => {
                    let mut x = get(&grids, *input)?;
                    match prologue {
                        Pro::None => {}
                        Pro::Apply { op: n, stats: sv } => x = apply_grid(self.vae, *n, &x, &stats[sv]),
                        Pro::Upsample => x = host::upsample(&x),
                    }
                    let c = self.vae.conv_weights(*op).context("conv weights")?;
                    let mut y = self.vae.conv_step(*op, &x, c, stream);
                    if let Some(r) = residual {
                        let r = get(&grids, *r)?;
                        ensure!(r.data.len() == y.data.len(), "residual shape");
                        y.data.iter_mut().zip(&r.data).for_each(|(a, b)| *a += b);
                    }
                    if let Some(cl) = clamp {
                        let VaeOp::Clamp { lo, hi, .. } = plan.ops[*cl] else { unreachable!() };
                        y.data.iter_mut().for_each(|v| *v = v.clamp(lo, hi));
                    }
                    if let Some((n, sv)) = st {
                        let VaeOp::GroupNorm { groups, .. } = plan.ops[*n] else { unreachable!() };
                        stats.insert(*sv, host::group_stats(&y, groups));
                    }
                    grids.insert(*output, y);
                }
                Step::TimeUp { op, input, output } => {
                    let x = get(&grids, *input)?;
                    grids.insert(*output, self.vae.time_up_step(*op, &x, stream));
                }
                Step::Stats { op, input, output } => {
                    let VaeOp::GroupNorm { groups, .. } = plan.ops[*op] else { unreachable!() };
                    stats.insert(*output, host::group_stats(&get(&grids, *input)?, groups));
                }
                Step::Apply { op, input, stats: sv, output } => {
                    let y = apply_grid(self.vae, *op, &get(&grids, *input)?, &stats[sv]);
                    grids.insert(*output, y);
                }
                Step::Rms { op, input, output } => {
                    let VaeOp::RmsNorm { act, .. } = plan.ops[*op] else { unreachable!() };
                    let mut x = get(&grids, *input)?;
                    host::rms_norm(&mut x, self.vae.norm_weights(*op).unwrap().0, act);
                    grids.insert(*output, x);
                }
                Step::Upsample { input, output, .. } => {
                    let y = host::upsample(&get(&grids, *input)?);
                    grids.insert(*output, y);
                }
                Step::Attention { op, input, output } => {
                    let VaeOp::Attention { channels, .. } = plan.ops[*op] else { unreachable!() };
                    let y = host::attention(&get(&grids, *input)?, channels);
                    grids.insert(*output, y);
                }
                Step::Add { a, b, output, .. } => {
                    let mut x = get(&grids, *a)?;
                    let y = get(&grids, *b)?;
                    x.data.iter_mut().zip(&y.data).for_each(|(p, q)| *p += q);
                    grids.insert(*output, x);
                }
                Step::Clamp { op, input, output } => {
                    let VaeOp::Clamp { lo, hi, .. } = plan.ops[*op] else { unreachable!() };
                    let mut x = get(&grids, *input)?;
                    x.data.iter_mut().for_each(|v| *v = v.clamp(lo, hi));
                    grids.insert(*output, x);
                }
            }
        }
        get(&grids, s.output)
    }
}
