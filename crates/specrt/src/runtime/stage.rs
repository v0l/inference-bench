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
            ensure!(spec.moe.as_ref().is_none_or(|m| m.dense_layers.is_empty() && m.sparse_step <= 1), "pipeline stages need every layer of a MoE model to be sparse");
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

    pub fn hidden_capacity_bytes(&self) -> usize {
        4 * self.pre.x.len()
    }

    pub fn export_hidden_device(&self, rows: usize, dst: u64) -> Result<()> {
        use cudarc::driver::DevicePtr;
        let bytes = rows * self.hidden_row_bytes();
        ensure!(rows > 0 && bytes <= self.hidden_capacity_bytes(), "hidden state of {rows} rows does not fit the stage");
        self.ctx.bind_to_thread()?;
        self.stream.synchronize()?;
        {
            let (src, _guard) = self.pre.x.device_ptr(&self.stream);
            unsafe { cudarc::driver::result::memcpy_dtod_async(dst, src, bytes, self.stream.cu_stream())? };
        }
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn import_hidden_device(&mut self, rows: usize, src: u64) -> Result<()> {
        use cudarc::driver::DevicePtrMut;
        let bytes = rows * self.hidden_row_bytes();
        ensure!(rows > 0 && bytes <= self.hidden_capacity_bytes(), "hidden state of {rows} rows does not fit the stage");
        self.ctx.bind_to_thread()?;
        self.stream.synchronize()?;
        {
            let (dst, _guard) = self.pre.x.device_ptr_mut(&self.stream);
            unsafe { cudarc::driver::result::memcpy_dtod_async(dst, src, bytes, self.stream.cu_stream())? };
        }
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn device_ordinal(&self) -> usize {
        self.ctx.ordinal()
    }

    pub fn hand_off(&self, next: &mut Runtime, rows: usize) -> Result<()> {
        ensure!(self.spec.dims.hidden == next.spec.dims.hidden, "stages disagree on the hidden size");
        if self.device_ordinal() != next.device_ordinal() {
            let n = rows * self.spec.dims.hidden;
            let link = crate::link::DeviceLink::new(&self.ctx, &next.ctx)?;
            return link.copy(&self.pre.x, &self.stream, &mut next.pre.x, &next.stream, n);
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
