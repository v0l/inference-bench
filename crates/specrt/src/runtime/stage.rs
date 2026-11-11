use super::Runtime;
use crate::spec::Spec;
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageRange {
    pub device: usize,
    pub start: usize,
    pub end: usize,
}

impl StageRange {
    pub fn whole(spec: &Spec) -> Self {
        StageRange { device: 0, start: 0, end: spec.layers.count }
    }

    pub fn check(&self, spec: &Spec) -> Result<()> {
        ensure!(self.start < self.end && self.end <= spec.layers.count, "stage layers {}..{} outside the {} layers of {}", self.start, self.end, spec.layers.count, spec.name);
        if self.start != 0 || self.end != spec.layers.count {
            ensure!(spec.moe.is_none(), "pipeline stages do not support MoE models yet");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageRole {
    pub first: bool,
    pub last: bool,
}

impl StageRole {
    pub fn of(spec: &Spec, range: StageRange) -> Self {
        StageRole { first: range.start == 0, last: range.end == spec.layers.count }
    }

    pub fn whole(self) -> bool {
        self.first && self.last
    }
}

impl Runtime {
    pub fn role(&self) -> StageRole {
        self.role
    }

    pub fn hidden_row_bytes(&self) -> usize {
        4 * self.spec.dims.hidden
    }

    pub fn export_hidden(&self, rows: usize) -> Result<Vec<f32>> {
        let n = rows * self.spec.dims.hidden;
        Ok(self.stream.clone_dtoh(&self.pre.x.slice(..n))?)
    }

    pub fn import_hidden(&mut self, hidden: &[f32]) -> Result<()> {
        ensure!(!hidden.is_empty() && hidden.len().is_multiple_of(self.spec.dims.hidden), "hidden state of {} floats", hidden.len());
        ensure!(hidden.len() <= self.pre.x.len(), "hidden state of {} floats exceeds the {} the stage holds", hidden.len(), self.pre.x.len());
        self.stream.synchronize()?;
        self.stream.memcpy_htod(hidden, &mut self.pre.x.slice_mut(..hidden.len()))?;
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn device_ordinal(&self) -> usize {
        self.ctx.ordinal()
    }

    pub fn hand_off(&self, next: &mut Runtime, rows: usize) -> Result<()> {
        ensure!(self.spec.dims.hidden == next.spec.dims.hidden, "stages disagree on the hidden size");
        if self.device_ordinal() != next.device_ordinal() {
            let staged = self.export_hidden(rows)?;
            return next.import_hidden(&staged);
        }
        let n = rows * self.spec.dims.hidden;
        ensure!(n <= next.pre.x.len(), "hidden state of {n} floats exceeds the next stage");
        next.stream.synchronize()?;
        self.stream.synchronize()?;
        self.stream.memcpy_dtod(&self.pre.x.slice(..n), &mut next.pre.x.slice_mut(..n))?;
        self.stream.synchronize()?;
        Ok(())
    }
}
