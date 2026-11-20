use super::*;

pub const CHUNK_THREADS: u32 = 256;
const CHUNK_ROWS: u32 = 128;
const SUB_ROWS: u32 = 64;
const SLICE_ROWS: u32 = 32;
const RESERVED: [&str; 28] = ["tile", "m0", "n0", "warp", "lane", "WARPS", "C", "M", "N", "SUBS", "RK", "RV", "THREADS", "BP", "UP", "COLG", "NTO", "NTS", "MTS", "PL", "i", "e", "j", "o", "n", "s", "v", "m"];

#[derive(Clone, Debug, PartialEq)]
pub struct ChunkForm {
    pub rows: u32,
    pub subs: u32,
    pub slice: u32,
    pub min_rows: u32,
    pub threads: u32,
    pub state: &'static str,
    pub carry: &'static str,
    pub out: &'static str,
    pub state_smem: u32,
    pub out_smem: u32,
    pub state_floats: u32,
}

impl ChunkForm {
    pub fn tiles(&self, heads: usize, rows_per_seq: usize, seqs: usize) -> usize {
        heads * rows_per_seq.div_ceil(self.rows as usize) * seqs
    }
}

pub(crate) struct KeyLoad {
    pub src: String,
    pub addr: String,
}

pub(crate) struct ChunkText {
    pub update_key: KeyLoad,
    pub read_key: KeyLoad,
    pub before: String,
    pub decay: String,
    pub update: String,
    pub finish: String,
}

impl Recurrence {
    fn chunk_lines(&self, params: &[Param], upto_update: bool) -> Result<(String, String, String), String> {
        let mut env = Env { names: Vec::new(), values: params.iter().filter(|p| p.role == Role::Value).map(|p| p.name.to_string()).collect() };
        let lower = |e: &Expr, env: &Env, ctx: Ctx| -> Result<String, String> {
            e.clone().bind(&self.consts).subst(&|n| env.lookup(n, ctx)).map_err(|e| format!("{e} in this space"))?.lower(Dialect::Cuda, &[])
        };
        let mut text = String::new();
        for i in &self.inputs {
            env.names.push((i.name.clone(), i.space));
            let index = match i.space {
                Space::Key => continue,
                Space::Value => "col",
                Space::Scalar => "0",
            };
            text.push_str(&format!("const float r_{} = {}[{}];\n", i.name, i.src, address(i.row, i.head, i.per, i.offset, index)));
        }
        let (mut decay, mut update) = (String::new(), String::new());
        for st in &self.stages {
            match st {
                Stage::Scalar { dst, expr } => {
                    let e = lower(expr, &env, Ctx::Scalar)?;
                    env.names.push((dst.clone(), Space::Scalar));
                    text.push_str(&format!("const float r_{dst} = {e};\n"));
                }
                Stage::Value { dst, expr } => {
                    let e = lower(expr, &env, Ctx::Value)?;
                    env.names.push((dst.clone(), Space::Value));
                    text.push_str(&format!("const float r_{dst} = {e};\n"));
                }
                Stage::Decay { expr } => decay = lower(expr, &env, Ctx::Scalar)?,
                Stage::Update { value, .. } => {
                    update = lower(value, &env, Ctx::Value)?;
                    if upto_update {
                        break;
                    }
                }
                Stage::Read { dst, .. } => {
                    env.names.push((dst.clone(), Space::Value));
                    text.push_str(&format!("const float r_{dst} = ck_o;\n"));
                }
                Stage::Store { dst, value, row, head, offset } => {
                    let v = lower(value, &env, Ctx::Value)?;
                    text.push_str(&format!("{dst}[{}] = {v};\n", address(*row, *head, 1, *offset, "col")));
                }
                Stage::Key { .. } | Stage::KeySum { .. } => return Err("a chunked scan has no key stages".into()),
            }
        }
        Ok((text, decay, update))
    }

    pub(crate) fn chunk_text(&self, params: &[Param]) -> Result<ChunkText, String> {
        let at = |m: fn(&Stage) -> bool| -> Vec<usize> { self.stages.iter().enumerate().filter(|(_, s)| m(s)).map(|(i, _)| i).collect() };
        let decay = at(|s| matches!(s, Stage::Decay { .. }));
        let update = at(|s| matches!(s, Stage::Update { .. }));
        let read = at(|s| matches!(s, Stage::Read { .. }));
        let store = at(|s| matches!(s, Stage::Store { .. }));
        if decay.len() != 1 || update.len() != 1 || read.len() != 1 || store.len() != 1 || !(decay[0] < update[0] && update[0] < read[0] && read[0] < store[0]) {
            return Err("a chunked scan needs one decay, update, read and store, in that order".into());
        }
        if let Some(p) = params.iter().find(|p| p.name.starts_with("ck_") || RESERVED.contains(&p.name)) {
            return Err(format!("param {:?} is a chunked scan local", p.name));
        }
        let (Stage::Update { key: ukey, .. }, Stage::Read { key: rkey, .. }) = (&self.stages[update[0]], &self.stages[read[0]]) else {
            unreachable!("stage indices come from their kinds")
        };
        let key = |e: &Expr| -> Result<KeyLoad, String> {
            match e {
                Expr::Name(n) => match self.inputs.iter().find(|i| i.space == Space::Key && i.name == *n) {
                    Some(i) => Ok(KeyLoad { src: i.src.clone(), addr: address(i.row, i.head, i.per, i.offset, "d") }),
                    None => Err("a chunked scan reads and updates key inputs directly".into()),
                },
                _ => Err("a chunked scan reads and updates key inputs directly".into()),
            }
        };
        let (update_key, read_key) = (key(ukey)?, key(rkey)?);
        let (before, decay, update) = self.chunk_lines(params, true)?;
        let (finish, _, _) = self.chunk_lines(params, false)?;
        Ok(ChunkText { update_key, read_key, before, decay, update, finish })
    }

    fn chunk_smem(&self) -> (u32, u32) {
        let (c, m, n) = (CHUNK_ROWS, SUB_ROWS, SLICE_ROWS);
        let (bp, up) = (self.keys + 8, n + 8);
        let state = 4 * c + 4 * self.keys * n + 4 * self.values * up;
        let slices = (4 * n * bp + 4 * self.values * up).max(4 * self.values * up);
        (state, 4 * c + 4 * m * self.keys + 4 * m * n + slices)
    }

    pub fn chunk_form(&self, k: &Kernel) -> Option<ChunkForm> {
        let t = &k.target;
        let colg = CHUNK_THREADS / 32 / (SUB_ROWS / 16);
        if t.language != Language::Cuda || !t.has(crate::target::Feature::Mma16816) || !self.keys.is_multiple_of(16) || !self.values.is_multiple_of(8 * colg) || CHUNK_THREADS > t.max_threads {
            return None;
        }
        self.chunk_text(&k.params).ok()?;
        let (state_smem, out_smem) = self.chunk_smem();
        if state_smem.max(out_smem) > t.smem_limit {
            return None;
        }
        let name = |s: &str| crate::desc::intern(&format!("{}_chunk_{s}", k.name));
        Some(ChunkForm {
            rows: CHUNK_ROWS,
            subs: CHUNK_ROWS / SUB_ROWS,
            slice: SLICE_ROWS,
            min_rows: SUB_ROWS,
            threads: CHUNK_THREADS,
            state: name("state"),
            carry: name("carry"),
            out: name("out"),
            state_smem,
            out_smem,
            state_floats: self.keys * self.values,
        })
    }
}
