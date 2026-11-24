use crate::int::IntExpr;
use crate::recur::{LAUNCH, SEQUENCE};
use crate::{Access, Kernel, Language, Param, Role, Scalar, Target};

#[derive(Clone, Debug, PartialEq)]
pub struct TokenHash {
    pub context: u32,
    pub heads: u32,
    pub out_row: u32,
    pub layers: u32,
    pub blank: u64,
    pub per_head: Vec<(String, Vec<u64>)>,
    pub lets: Vec<(String, IntExpr)>,
    pub index: IntExpr,
}

fn history_name(n: &str, context: u32) -> Option<u32> {
    let k = n.strip_prefix('t')?.parse::<u32>().ok()?;
    (k <= context).then_some(k)
}

impl TokenHash {
    pub fn allowed() -> Vec<&'static str> {
        let mut v = vec!["tokens", "idx", "state"];
        v.extend(LAUNCH);
        v
    }

    fn known(&self, n: &str, lets: usize) -> bool {
        n == "head" || history_name(n, self.context).is_some() || self.per_head.iter().any(|(p, _)| p == n) || self.lets[..lets].iter().any(|(l, _)| l == n)
    }

    pub fn check(&self, t: &Target, threads: u32, params: &[Param]) -> Result<(), String> {
        if t.language != Language::Cuda {
            return Err("a token_hash body has only a CUDA lowering".into());
        }
        let want = [
            ("tokens", Scalar::I32, Role::Activation, Access::Read),
            ("idx", Scalar::I32, Role::Activation, Access::Write),
            ("state", Scalar::F32, Role::Activation, Access::Write),
        ];
        for &(name, ty, role, access) in want.iter().chain(SEQUENCE.iter()) {
            match params.iter().find(|p| p.name == name) {
                Some(p) if p.ty == ty && p.role == role && p.access == access => {}
                Some(p) => return Err(format!("token_hash param {name:?} must be {ty:?} {role:?} {access:?}, not {:?} {:?} {:?}", p.ty, p.role, p.access)),
                None => return Err(format!("token_hash needs a param named {name:?}")),
            }
        }
        if params.len() != want.len() + SEQUENCE.len() {
            return Err(format!("token_hash takes exactly the params {:?}", Self::allowed()));
        }
        if self.context == 0 || self.heads == 0 || self.layers == 0 || self.out_row < self.heads {
            return Err("token_hash needs a positive context, heads and layers and an out_row of at least heads".into());
        }
        if self.context > threads {
            return Err(format!("token_hash context {} must fit in {threads} threads", self.context));
        }
        if self.blank > i32::MAX as u64 {
            return Err(format!("token_hash blank {} is not a token id", self.blank));
        }
        for (n, table) in &self.per_head {
            if table.len() != self.heads as usize {
                return Err(format!("token_hash per head int {n:?} has {} values for {} heads", table.len(), self.heads));
            }
        }
        for (i, (n, e)) in self.lets.iter().enumerate() {
            if self.known(n, i) {
                return Err(format!("token_hash let {n:?} shadows another name"));
            }
            let mut names = Vec::new();
            e.names(&mut names);
            if let Some(bad) = names.iter().find(|m| !self.known(m, i)) {
                return Err(format!("token_hash let {n:?} names {bad:?}, which is not t0 to t{}, head, a per head int or an earlier let", self.context));
            }
        }
        let mut names = Vec::new();
        self.index.names(&mut names);
        if let Some(bad) = names.iter().find(|m| !self.known(m, self.lets.len())) {
            return Err(format!("token_hash index names {bad:?}, which is not t0 to t{}, head, a per head int or a let", self.context));
        }
        self.body()?;
        Ok(())
    }

    fn lower(&self, e: &IntExpr) -> Result<String, String> {
        e.lower_cuda(&|n| {
            if n == "head" {
                return Some("(unsigned long long)head".into());
            }
            if let Some(k) = history_name(n, self.context) {
                return Some(format!("(unsigned long long)h{k}"));
            }
            if self.per_head.iter().any(|(p, _)| p == n) {
                return Some(format!("{n}_tab[head]"));
            }
            self.lets.iter().any(|(l, _)| l == n).then(|| format!("l_{n}"))
        })
    }

    fn body(&self) -> Result<String, String> {
        let mut s = String::new();
        for k in 1..=self.context {
            s.push_str(&format!(
                "        {{\n            const int j = t - {k};\n            const long long v = j >= 0 ? (long long)tok[j] : (long long)(__float_as_int(st[CTX + j]) ^ BLANK);\n            h{k} = live ? v : (long long)BLANK;\n            live = live && v != (long long)BLANK;\n        }}\n"
            ));
        }
        for (n, e) in &self.lets {
            s.push_str(&format!("        const unsigned long long l_{n} = {};\n", self.lower(e)?));
        }
        s.push_str(&format!("        idx[((size_t)seq * rows_per_seq + t) * {} + head] = (int){};\n", self.out_row, self.lower(&self.index)?));
        Ok(s)
    }

    pub fn emit(&self, k: &Kernel, head: &str) -> Result<String, String> {
        let mut s = head.to_string();
        s.push_str(&format!("    constexpr int CTX = {};\n    constexpr int HEADS = {};\n    constexpr int BLANK = {};\n", self.context, self.heads, self.blank));
        for (n, table) in &self.per_head {
            let values: Vec<String> = table.iter().map(|v| format!("{v}ull")).collect();
            s.push_str(&format!("    const unsigned long long {n}_tab[HEADS] = {{{}}};\n", values.join(", ")));
        }
        let decl: Vec<String> = (1..=self.context).map(|k| format!("h{k}")).collect();
        s.push_str(&format!(
            "    const int seq = blockIdx.x;
    const int slot = slot_step ? slots[seq] : slot_base;
    float* st = state + ((size_t)slot * {layers} + layer) * CTX;
    const int* tok = tokens + (size_t)seq * rows_per_seq;
    for (int i = threadIdx.x; i < rows_per_seq * HEADS; i += blockDim.x) {{
        const int t = i / HEADS;
        const int head = i % HEADS;
        const long long h0 = (long long)tok[t];
        long long {decl};
        bool live = true;
{body}    }}
    __syncthreads();
    int keep = 0;
    if (threadIdx.x < CTX) {{
        const int j = rows_per_seq - CTX + (int)threadIdx.x;
        keep = j >= 0 ? (tok[j] ^ BLANK) : __float_as_int(st[CTX + j]);
    }}
    __syncthreads();
    if (threadIdx.x < CTX) st[threadIdx.x] = __int_as_float(keep);
}}

",
            layers = self.layers,
            decl = decl.join(", "),
            body = self.body()?,
        ));
        let _ = k;
        Ok(s)
    }

    pub fn eval(&self, tokens: &[i64], history: &mut [i64]) -> Result<Vec<i64>, String> {
        let ctx = self.context as usize;
        if history.len() != ctx {
            return Err(format!("token_hash history holds {ctx} tokens, not {}", history.len()));
        }
        let blank = self.blank as i64;
        let mut out = Vec::with_capacity(tokens.len() * self.heads as usize);
        for t in 0..tokens.len() {
            let at = |j: isize| if j >= 0 { tokens[j as usize] } else { history[(ctx as isize + j) as usize] };
            let mut h = vec![tokens[t]];
            let mut live = true;
            for k in 1..=ctx {
                let v = at(t as isize - k as isize);
                h.push(if live { v } else { blank });
                live = live && v != blank;
            }
            for head in 0..self.heads as usize {
                let mut lets: Vec<(String, u64)> = Vec::new();
                let env = |n: &str, lets: &[(String, u64)]| -> Option<u64> {
                    if n == "head" {
                        return Some(head as u64);
                    }
                    if let Some(k) = history_name(n, self.context) {
                        return Some(h[k as usize] as u64);
                    }
                    if let Some((_, table)) = self.per_head.iter().find(|(p, _)| p == n) {
                        return Some(table[head]);
                    }
                    lets.iter().find(|(l, _)| l == n).map(|(_, v)| *v)
                };
                for (n, e) in &self.lets {
                    let v = e.eval(&|m| env(m, &lets))?;
                    lets.push((n.clone(), v));
                }
                out.push(self.index.eval(&|m| env(m, &lets))? as u32 as i32 as i64);
            }
        }
        let all: Vec<i64> = history.iter().copied().chain(tokens.iter().copied()).collect();
        history.copy_from_slice(&all[all.len() - ctx..]);
        Ok(out)
    }
}
