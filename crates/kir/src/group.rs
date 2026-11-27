use crate::expr::{Const, Expr};
use crate::layout::Dialect;
use crate::{Access, Kernel, Language, Param, Role, Scalar, Target};

#[derive(Clone, Debug, PartialEq)]
pub enum GroupStage {
    Load { dst: String, src: String, row: u32, group: u32, offset: u32, index: Option<String>, shift: Option<String>, by_row: bool, step: u32, limit: Option<u32> },
    Pick { dst: String, src: String, row: u32, group: u32, offset: u32 },
    Value { dst: String, expr: Expr },
    Sum { dst: String, expr: Expr },
    Scalar { dst: String, expr: Expr },
    Store { dst: String, value: Expr, row: u32, group: u32, offset: u32 },
    Planes { value: Expr, row: u32, group: u32, offset: u32 },
    Put { dst: String, value: Expr, row: u32, group: u32, offset: u32 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroupMap {
    pub width: u32,
    pub consts: Vec<Const>,
    pub stages: Vec<GroupStage>,
    pub scales: Vec<(String, f32)>,
}

const RESERVED: [&str; 8] = ["i", "row", "grp", "scratch", "WIDTH", "width", "hi", "lo"];

const PLANES: [&str; 2] = ["out_hi", "out_lo"];

fn address(row: u32, group: u32, offset: u32) -> String {
    stepped(row, group, offset, 1)
}

fn stepped(row: u32, group: u32, offset: u32, step: u32) -> String {
    let mut terms = Vec::new();
    if row != 0 {
        terms.push(format!("(size_t)row * {row}"));
    }
    if group != 0 {
        terms.push(format!("(size_t)grp * {group}"));
    }
    if offset != 0 {
        terms.push(offset.to_string());
    }
    terms.push(if step == 1 { "i".into() } else { format!("(size_t)i * {step}") });
    terms.join(" + ")
}

fn fixed(row: u32, group: u32, offset: u32) -> String {
    let mut terms = vec![format!("(size_t)row * {row}"), format!("(size_t)grp * {group}")];
    if offset != 0 {
        terms.push(offset.to_string());
    }
    terms.join(" + ")
}

impl GroupMap {
    pub fn has_sum(&self) -> bool {
        self.stages.iter().any(|s| matches!(s, GroupStage::Sum { .. }))
    }

    fn stored(&self, p: &Param) -> bool {
        self.scales.iter().any(|(n, _)| n == p.name)
    }

    fn element(&self, p: &Param, at: &str) -> String {
        let scale = self.scales.iter().find(|(n, _)| n == p.name).map_or(1.0, |(_, s)| *s);
        match p.ty {
            Scalar::Bf16 => format!("__uint_as_float(((unsigned){}[{at}]) << 16)", p.name),
            Scalar::U8 if scale == 1.0 => format!("kir_e4m3({}[{at}])", p.name),
            Scalar::U8 => format!("(kir_e4m3({}[{at}]) * {}f)", p.name, crate::lit(scale)),
            _ => format!("{}[{at}]", p.name),
        }
    }

    pub fn check(&self, t: &Target, params: &[Param]) -> Result<(), String> {
        if t.language != Language::Cuda {
            return Err("a group_map body has only a CUDA lowering".into());
        }
        if self.width == 0 {
            return Err("group_map needs a positive width".into());
        }
        for (i, p) in params.iter().enumerate() {
            let index = self.stages.iter().any(|s| matches!(s, GroupStage::Load { index: Some(n), .. } if n == p.name));
            let shift = self.stages.iter().any(|s| matches!(s, GroupStage::Load { shift: Some(n), .. } if n == p.name));
            let plane = PLANES.contains(&p.name);
            let ty_ok = if index {
                p.ty == Scalar::I32 && p.role == Role::Activation && p.access == Access::Read
            } else if plane {
                p.ty == Scalar::Bf16 && p.role == Role::Activation && p.access == Access::Write
            } else if shift {
                p.ty == Scalar::I32 && p.role == Role::Value
            } else if self.stored(p) {
                matches!(p.ty, Scalar::F32 | Scalar::Bf16 | Scalar::U8) && p.role == Role::Weight
            } else {
                p.ty == Scalar::F32
            };
            if params[..i].iter().any(|q| q.name == p.name) || RESERVED.contains(&p.name) || !ty_ok {
                return Err(format!("group_map param {:?} is not usable", p.name));
            }
        }
        let planes = self.stages.iter().any(|s| matches!(s, GroupStage::Planes { .. }));
        if planes != PLANES.iter().all(|n| params.iter().any(|p| p.name == *n)) || PLANES.iter().filter(|n| params.iter().any(|p| p.name == **n)).count() == 1 {
            return Err("group_map stores to planes through a planes stage with both out_hi and out_lo params".into());
        }
        if !self.stages.iter().any(|s| matches!(s, GroupStage::Store { .. } | GroupStage::Planes { .. } | GroupStage::Put { .. })) {
            return Err("group_map has no store".into());
        }
        self.body(params)?;
        Ok(())
    }

    fn body(&self, params: &[Param]) -> Result<String, String> {
        let mut elements: Vec<(String, Expr)> = Vec::new();
        let mut scalars: Vec<String> = Vec::new();
        let mut defined: Vec<String> = Vec::new();
        let mut out = String::new();
        let value_param = |n: &str| params.iter().find(|p| p.name == n && p.role == Role::Value).map(|p| Expr::Raw(p.name.to_string()));
        let ok = |n: &str, defined: &[String]| {
            !n.is_empty()
                && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && !n.starts_with(|c: char| c.is_ascii_digit())
                && !RESERVED.contains(&n)
                && !n.starts_with("ss_")
                && !params.iter().any(|p| p.name == n)
                && !self.consts.iter().any(|c| c.name == n)
                && !defined.iter().any(|d| d == n)
        };
        let readable = |src: &str| matches!(params.iter().find(|p| p.name == src), Some(p) if p.role != Role::Value && p.access == Access::Read && p.ty == Scalar::F32);
        let loop_head = "    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x)";
        for s in &self.stages {
            let element = |e: &Expr, elements: &[(String, Expr)], scalars: &[String]| -> Result<String, String> {
                let env = |n: &str| elements.iter().find(|(k, _)| k == n).map(|(_, x)| x.clone()).or_else(|| scalars.iter().find(|k| *k == n).map(|k| Expr::Raw(k.clone()))).or_else(|| value_param(n));
                e.clone().bind(&self.consts).subst(&env)?.lower(Dialect::Cuda, &[])
            };
            match s {
                GroupStage::Load { dst, src, row, group, offset, index, shift, by_row, step, limit } => {
                    if !ok(dst, &defined) {
                        return Err(format!("group_map name {dst:?} is not usable or is defined twice"));
                    }
                    let source = match params.iter().find(|p| p.name == src) {
                        Some(p) if p.role != Role::Value && p.access == Access::Read && (p.ty == Scalar::F32 || self.stored(p)) => p,
                        _ => return Err(format!("group_map load from {src:?}, which is not a read f32 or stored weight param")),
                    };
                    if *step == 0 || (*step != 1 && index.is_some()) || (limit.is_some() && index.is_none()) {
                        return Err(format!("group_map load {dst:?}: a step is positive and only on plain loads, a limit only on gathers"));
                    }
                    let at = match index {
                        None => match shift {
                            None => stepped(*row, *group, *offset, *step),
                            Some(s) => {
                                if !params.iter().any(|p| p.name == s && p.role == Role::Value) {
                                    return Err(format!("group_map shift {s:?} is not a value param"));
                                }
                                format!("{} + (size_t){s} * WIDTH", stepped(*row, *group, *offset, *step))
                            }
                        },
                        Some(ix) => {
                            if !params.iter().any(|p| p.name == ix) {
                                return Err(format!("group_map gather index {ix:?} is not a param"));
                            }
                            let at = if *by_row { format!("{ix}[row]") } else { format!("{ix}[(size_t)row * {row} + grp]") };
                            let at = match limit {
                                Some(l) => format!("({at} < {l} ? {at} : 0)"),
                                None => at,
                            };
                            if *by_row {
                                format!("(size_t){at} * {row} + (size_t)grp * {group} + {offset} + i")
                            } else {
                                format!("(size_t){at} * {group} + {offset} + i")
                            }
                        }
                    };
                    elements.push((dst.clone(), Expr::Raw(self.element(source, &at))));
                    defined.push(dst.clone());
                }
                GroupStage::Pick { dst, src, row, group, offset } => {
                    if !ok(dst, &defined) {
                        return Err(format!("group_map name {dst:?} is not usable or is defined twice"));
                    }
                    if !readable(src) {
                        return Err(format!("group_map pick from {src:?}, which is not a read f32 buffer param"));
                    }
                    out.push_str(&format!("    const float {dst} = {src}[{}];\n", fixed(*row, *group, *offset)));
                    scalars.push(dst.clone());
                    defined.push(dst.clone());
                }
                GroupStage::Value { dst, expr } => {
                    if !ok(dst, &defined) {
                        return Err(format!("group_map name {dst:?} is not usable or is defined twice"));
                    }
                    let text = element(expr, &elements, &scalars)?;
                    elements.push((dst.clone(), Expr::Raw(format!("({text})"))));
                    defined.push(dst.clone());
                }
                GroupStage::Sum { dst, expr } => {
                    if !ok(dst, &defined) {
                        return Err(format!("group_map name {dst:?} is not usable or is defined twice"));
                    }
                    let text = element(expr, &elements, &scalars)?;
                    out.push_str(&format!("    float ss_{dst} = 0.0f;\n{loop_head} ss_{dst} += {text};\n    const float {dst} = kir_block_sum(ss_{dst}, scratch);\n"));
                    scalars.push(dst.clone());
                    defined.push(dst.clone());
                }
                GroupStage::Scalar { dst, expr } => {
                    if !ok(dst, &defined) {
                        return Err(format!("group_map name {dst:?} is not usable or is defined twice"));
                    }
                    let text = element(expr, &[], &scalars).map_err(|e| format!("group_map scalar {dst:?}: {e}"))?;
                    out.push_str(&format!("    const float {dst} = {text};\n"));
                    scalars.push(dst.clone());
                    defined.push(dst.clone());
                }
                GroupStage::Put { dst, value, row, group, offset } => {
                    match params.iter().find(|p| p.name == dst) {
                        Some(p) if p.role == Role::Activation && p.access == Access::Write && p.ty == Scalar::F32 => {}
                        _ => return Err(format!("group_map put to {dst:?}, which is not a written f32 activation param")),
                    }
                    let text = element(value, &[], &scalars).map_err(|e| format!("group_map put {dst:?}: {e}"))?;
                    out.push_str(&format!("    if (threadIdx.x == 0) {dst}[{}] = {text};\n", fixed(*row, *group, *offset)));
                }
                GroupStage::Store { dst, value, row, group, offset } => {
                    match params.iter().find(|p| p.name == dst) {
                        Some(p) if p.role == Role::Activation && p.access == Access::Write => {}
                        _ => return Err(format!("group_map store to {dst:?}, which is not a written activation param")),
                    }
                    let text = element(value, &elements, &scalars)?;
                    out.push_str(&format!("{loop_head} {dst}[{}] = {text};\n", address(*row, *group, *offset)));
                }
                GroupStage::Planes { value, row, group, offset } => {
                    let text = element(value, &elements, &scalars)?;
                    let at = address(*row, *group, *offset);
                    out.push_str(&format!("{loop_head} {{\n        unsigned hi, lo;\n        split_hi_lo({text}, hi, lo);\n        out_hi[{at}] = (bf16_t)hi;\n        out_lo[{at}] = (bf16_t)lo;\n    }}\n"));
                }
            }
        }
        Ok(out)
    }

    pub fn emit(&self, k: &Kernel, head: &str) -> Result<String, String> {
        let scratch = if self.has_sum() { "    __shared__ float scratch[32];\n" } else { "" };
        Ok(format!("{head}    constexpr int WIDTH = {};\n{scratch}    const int grp = blockIdx.x;\n    const int row = blockIdx.y;\n{}}}\n\n", self.width, self.body(&k.params)?))
    }

    pub fn eval(&self, params: &[Param], grid: (usize, usize), bufs: &mut [&mut [f32]]) -> Result<(), String> {
        if params.len() != bufs.len() {
            return Err(format!("group_map eval takes one buffer per param, {} for {}", bufs.len(), params.len()));
        }
        if self.stages.iter().any(|s| matches!(s, GroupStage::Load { index: Some(_), .. } | GroupStage::Load { shift: Some(_), .. } | GroupStage::Planes { .. })) || !self.scales.is_empty() {
            return Err("group_map eval runs plain loads, picks, sums, values, scalars, stores and puts".into());
        }
        let at = |name: &str| params.iter().position(|p| p.name == name).ok_or_else(|| format!("group_map names unknown param {name:?}"));
        let width = self.width as usize;
        let consts = self.consts.clone();
        for row in 0..grid.1 {
            for grp in 0..grid.0 {
                let mut elements: Vec<(String, Vec<f64>)> = Vec::new();
                let mut scalars: Vec<(String, f64)> = Vec::new();
                let element = |e: &Expr, elements: &[(String, Vec<f64>)], scalars: &[(String, f64)], i: usize| -> Result<f64, String> {
                    e.clone().bind(&consts).eval(&|n| elements.iter().find(|(k, _)| k == n).map(|(_, v)| v[i]).or_else(|| scalars.iter().find(|(k, _)| k == n).map(|(_, v)| *v))).ok_or_else(|| format!("group_map expression {e} does not evaluate"))
                };
                for s in &self.stages {
                    match s {
                        GroupStage::Load { dst, src, row: r, group: g, offset: o, step, .. } => {
                            let b = &bufs[at(src)?];
                            let v = (0..width).map(|i| b.get(row * *r as usize + grp * *g as usize + *o as usize + i * *step as usize).copied().map(f64::from).ok_or_else(|| format!("group_map load {dst} reads past {src}"))).collect::<Result<Vec<_>, _>>()?;
                            elements.push((dst.clone(), v));
                        }
                        GroupStage::Pick { dst, src, row: r, group: g, offset: o } => {
                            let b = &bufs[at(src)?];
                            let v = *b.get(row * *r as usize + grp * *g as usize + *o as usize).ok_or_else(|| format!("group_map pick {dst} reads past {src}"))?;
                            scalars.push((dst.clone(), f64::from(v)));
                        }
                        GroupStage::Value { dst, expr } => {
                            let v = (0..width).map(|i| element(expr, &elements, &scalars, i)).collect::<Result<Vec<_>, _>>()?;
                            elements.push((dst.clone(), v));
                        }
                        GroupStage::Sum { dst, expr } => {
                            let v = (0..width).map(|i| element(expr, &elements, &scalars, i).map(|x| x as f32)).sum::<Result<f32, _>>()?;
                            scalars.push((dst.clone(), f64::from(v)));
                        }
                        GroupStage::Scalar { dst, expr } => {
                            let v = element(expr, &[], &scalars, 0)?;
                            scalars.push((dst.clone(), v));
                        }
                        GroupStage::Store { dst, value, row: r, group: g, offset: o } => {
                            let v = (0..width).map(|i| element(value, &elements, &scalars, i)).collect::<Result<Vec<_>, _>>()?;
                            let b = &mut bufs[at(dst)?];
                            for (i, x) in v.into_iter().enumerate() {
                                let p = row * *r as usize + grp * *g as usize + *o as usize + i;
                                *b.get_mut(p).ok_or_else(|| format!("group_map store writes past {dst}"))? = x as f32;
                            }
                        }
                        GroupStage::Put { dst, value, row: r, group: g, offset: o } => {
                            let v = element(value, &[], &scalars, 0)?;
                            let p = row * *r as usize + grp * *g as usize + *o as usize;
                            *bufs[at(dst)?].get_mut(p).ok_or_else(|| format!("group_map put writes past {dst}"))? = v as f32;
                        }
                        GroupStage::Planes { .. } => unreachable!("refused above"),
                    }
                }
            }
        }
        Ok(())
    }
}
