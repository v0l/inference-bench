use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Nest {
    Leaf(u64),
    Tuple(Vec<Nest>),
}

impl Nest {
    fn flatten(&self, out: &mut Vec<u64>) {
        match self {
            Nest::Leaf(v) => out.push(*v),
            Nest::Tuple(t) => t.iter().for_each(|n| n.flatten(out)),
        }
    }

    fn same_shape(&self, other: &Nest) -> bool {
        match (self, other) {
            (Nest::Leaf(_), Nest::Leaf(_)) => true,
            (Nest::Tuple(a), Nest::Tuple(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.same_shape(y)),
            _ => false,
        }
    }

    fn size(&self) -> u64 {
        match self {
            Nest::Leaf(v) => *v,
            Nest::Tuple(t) => t.iter().map(Nest::size).product(),
        }
    }
}

impl fmt::Display for Nest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Nest::Leaf(v) => write!(f, "{v}"),
            Nest::Tuple(t) => {
                f.write_str("(")?;
                for (i, n) in t.iter().enumerate() {
                    if i > 0 {
                        f.write_str(",")?;
                    }
                    write!(f, "{n}")?;
                }
                f.write_str(")")
            }
        }
    }
}

pub fn leaf(v: u64) -> Nest {
    Nest::Leaf(v)
}

pub fn tuple(v: Vec<Nest>) -> Nest {
    Nest::Tuple(v)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    Cuda,
    Wgsl,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub shape: Nest,
    pub stride: Nest,
}

impl Layout {
    pub fn new(shape: Nest, stride: Nest) -> Result<Layout, String> {
        if !shape.same_shape(&stride) {
            return Err(format!("shape {shape} and stride {stride} are not congruent"));
        }
        let mut dims = Vec::new();
        shape.flatten(&mut dims);
        if dims.contains(&0) {
            return Err(format!("shape {shape} has a zero mode"));
        }
        Ok(Layout { shape, stride })
    }

    pub fn flat(shape: &[u64], stride: &[u64]) -> Result<Layout, String> {
        if shape.len() != stride.len() {
            return Err("shape and stride differ in rank".into());
        }
        Layout::new(
            tuple(shape.iter().map(|v| leaf(*v)).collect()),
            tuple(stride.iter().map(|v| leaf(*v)).collect()),
        )
    }

    pub fn row_major(shape: &[u64]) -> Result<Layout, String> {
        let mut stride = vec![1u64; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        Layout::flat(shape, &stride)
    }

    pub fn col_major(shape: &[u64]) -> Result<Layout, String> {
        let mut stride = vec![1u64; shape.len()];
        for i in 1..shape.len() {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        Layout::flat(shape, &stride)
    }

    pub fn size(&self) -> u64 {
        self.shape.size()
    }

    pub fn leaves(&self) -> Vec<(u64, u64)> {
        let (mut s, mut d) = (Vec::new(), Vec::new());
        self.shape.flatten(&mut s);
        self.stride.flatten(&mut d);
        s.into_iter().zip(d).collect()
    }

    pub fn cosize(&self) -> u64 {
        1 + self.leaves().iter().map(|(s, d)| (s - 1) * d).sum::<u64>()
    }

    pub fn rank(&self) -> usize {
        match &self.shape {
            Nest::Leaf(_) => 1,
            Nest::Tuple(t) => t.len(),
        }
    }

    pub fn mode(&self, i: usize) -> Option<Layout> {
        match (&self.shape, &self.stride) {
            (Nest::Tuple(s), Nest::Tuple(d)) => Some(Layout { shape: s.get(i)?.clone(), stride: d.get(i)?.clone() }),
            (s @ Nest::Leaf(_), d) if i == 0 => Some(Layout { shape: s.clone(), stride: d.clone() }),
            _ => None,
        }
    }

    pub fn offset(&self, index: u64) -> u64 {
        let mut rest = index;
        let mut off = 0;
        for (s, d) in self.leaves() {
            off += (rest % s) * d;
            rest /= s;
        }
        off
    }

    pub fn coalesce(&self) -> Layout {
        let mut out: Vec<(u64, u64)> = Vec::new();
        for (s, d) in self.leaves() {
            if s == 1 {
                continue;
            }
            match out.last_mut() {
                Some((ps, pd)) if *pd * *ps == d => *ps *= s,
                _ => out.push((s, d)),
            }
        }
        if out.is_empty() {
            return Layout { shape: leaf(1), stride: leaf(0) };
        }
        if out.len() == 1 {
            return Layout { shape: leaf(out[0].0), stride: leaf(out[0].1) };
        }
        Layout {
            shape: tuple(out.iter().map(|(s, _)| leaf(*s)).collect()),
            stride: tuple(out.iter().map(|(_, d)| leaf(*d)).collect()),
        }
    }

    pub fn compose(&self, inner: &Layout) -> Result<Layout, String> {
        let outer = self.coalesce();
        let (os, od) = match (&outer.shape, &outer.stride) {
            (Nest::Leaf(s), Nest::Leaf(d)) => (*s, *d),
            _ => return Err("composition needs a layout that coalesces to one mode".into()),
        };
        let mut out = Vec::new();
        for (s, d) in inner.leaves() {
            if (s - 1) * d >= os {
                return Err(format!("inner layout reaches {} but the outer has {os} elements", (s - 1) * d));
            }
            out.push((s, d * od));
        }
        Layout::flat(&out.iter().map(|x| x.0).collect::<Vec<_>>(), &out.iter().map(|x| x.1).collect::<Vec<_>>())
    }

    pub fn expr(&self, coords: &[&str], dialect: Dialect) -> Result<String, String> {
        if coords.len() != self.rank() {
            return Err(format!("layout rank {} needs {} coordinate names", self.rank(), self.rank()));
        }
        let u = match dialect {
            Dialect::Cuda => "",
            Dialect::Wgsl => "u",
        };
        let mut terms = Vec::new();
        for (i, name) in coords.iter().enumerate() {
            let mode = self.mode(i).expect("rank checked");
            if let Ok(n) = name.parse::<u64>() {
                let (s, d) = mode.leaves().first().copied().unwrap_or((1, 0));
                if mode.leaves().len() != 1 || n >= s {
                    return Err(format!("constant coordinate {n} needs a single mode that holds it"));
                }
                if n * d != 0 {
                    terms.push(format!("{}{u}", n * d));
                }
                continue;
            }
            let mut div = 1u64;
            let leaves = mode.leaves();
            let count = leaves.len();
            for (j, (s, d)) in leaves.into_iter().enumerate() {
                if s > 1 && d != 0 {
                    let mut v = (*name).to_string();
                    if div > 1 {
                        v = format!("({v} / {div}{u})");
                    }
                    if j + 1 < count {
                        v = format!("({v} % {s}{u})");
                    }
                    terms.push(if d == 1 { v } else { format!("{v} * {d}{u}") });
                }
                div *= s;
            }
        }
        if terms.is_empty() {
            return Ok(format!("0{u}"));
        }
        Ok(terms.join(" + "))
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.shape, self.stride)
    }
}
