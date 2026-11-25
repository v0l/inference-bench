use crate::expr::{Const, Expr};
use crate::layout::Dialect;
use crate::{Access, Kernel, Language, Param, Role, Scalar, Target};

#[derive(Clone, Debug, PartialEq)]
pub enum GroupStage {
    Load { dst: String, src: String, row: u32, group: u32, offset: u32, index: Option<String>, shift: Option<String>, by_row: bool },
    Value { dst: String, expr: Expr },
    Sum { dst: String, expr: Expr },
    Scalar { dst: String, expr: Expr },
    Store { dst: String, value: Expr, row: u32, group: u32, offset: u32 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroupMap {
    pub width: u32,
    pub consts: Vec<Const>,
    pub stages: Vec<GroupStage>,
    pub scales: Vec<(String, f32)>,
}

const RESERVED: [&str; 6] = ["i", "row", "grp", "scratch", "WIDTH", "width"];

fn address(row: u32, group: u32, offset: u32) -> String {
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
    terms.push("i".into());
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
            let ty_ok = if index {
                p.ty == Scalar::I32 && p.role == Role::Activation && p.access == Access::Read
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
        if !self.stages.iter().any(|s| matches!(s, GroupStage::Store { .. })) {
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
        let loop_head = "    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x)";
        for s in &self.stages {
            let element = |e: &Expr, elements: &[(String, Expr)], scalars: &[String]| -> Result<String, String> {
                let env = |n: &str| elements.iter().find(|(k, _)| k == n).map(|(_, x)| x.clone()).or_else(|| scalars.iter().find(|k| *k == n).map(|k| Expr::Raw(k.clone()))).or_else(|| value_param(n));
                e.clone().bind(&self.consts).subst(&env)?.lower(Dialect::Cuda, &[])
            };
            match s {
                GroupStage::Load { dst, src, row, group, offset, index, shift, by_row } => {
                    if !ok(dst, &defined) {
                        return Err(format!("group_map name {dst:?} is not usable or is defined twice"));
                    }
                    let source = match params.iter().find(|p| p.name == src) {
                        Some(p) if p.role != Role::Value && p.access == Access::Read && (p.ty == Scalar::F32 || self.stored(p)) => p,
                        _ => return Err(format!("group_map load from {src:?}, which is not a read f32 or stored weight param")),
                    };
                    let at = match index {
                        None => match shift {
                            None => address(*row, *group, *offset),
                            Some(s) => {
                                if !params.iter().any(|p| p.name == s && p.role == Role::Value) {
                                    return Err(format!("group_map shift {s:?} is not a value param"));
                                }
                                format!("{} + (size_t){s} * WIDTH", address(*row, *group, *offset))
                            }
                        },
                        Some(ix) => {
                            if !params.iter().any(|p| p.name == ix) {
                                return Err(format!("group_map gather index {ix:?} is not a param"));
                            }
                            if *by_row {
                                format!("(size_t){ix}[row] * {row} + (size_t)grp * {group} + {offset} + i")
                            } else {
                                format!("(size_t){ix}[(size_t)row * {row} + grp] * {group} + {offset} + i")
                            }
                        }
                    };
                    elements.push((dst.clone(), Expr::Raw(self.element(source, &at))));
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
                GroupStage::Store { dst, value, row, group, offset } => {
                    match params.iter().find(|p| p.name == dst) {
                        Some(p) if p.role == Role::Activation && p.access == Access::Write => {}
                        _ => return Err(format!("group_map store to {dst:?}, which is not a written activation param")),
                    }
                    let text = element(value, &elements, &scalars)?;
                    out.push_str(&format!("{loop_head} {dst}[{}] = {text};\n", address(*row, *group, *offset)));
                }
            }
        }
        Ok(out)
    }

    pub fn emit(&self, k: &Kernel, head: &str) -> Result<String, String> {
        let scratch = if self.has_sum() { "    __shared__ float scratch[32];\n" } else { "" };
        Ok(format!("{head}    constexpr int WIDTH = {};\n{scratch}    const int grp = blockIdx.x;\n    const int row = blockIdx.y;\n{}}}\n\n", self.width, self.body(&k.params)?))
    }
}
