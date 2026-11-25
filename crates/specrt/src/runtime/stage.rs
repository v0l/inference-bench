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

    fn hand_buf(&self) -> &cudarc::driver::CudaSlice<f32> {
        match (&self.gated, &self.spec.hyper) {
            (Some(g), _) => &g.streams,
            (None, Some(_)) => &self.pre.streams,
            _ => &self.pre.x,
        }
    }

    fn hand_width(&self) -> usize {
        self.hand_buf().len() / self.chunk
    }

    pub fn hidden_row_bytes(&self) -> usize {
        4 * self.hand_width()
    }

    pub fn export_hidden(&self, rows: usize) -> Result<Vec<f32>> {
        let n = rows * self.hand_width();
        Ok(self.stream.clone_dtoh(&self.hand_buf().slice(..n))?)
    }

    pub fn import_hidden(&mut self, hidden: &[f32]) -> Result<()> {
        let width = self.hand_width();
        ensure!(!hidden.is_empty() && hidden.len().is_multiple_of(width), "hidden state of {} floats", hidden.len());
        ensure!(hidden.len() <= width * self.chunk, "hidden state of {} floats exceeds the {} the stage holds", hidden.len(), width * self.chunk);
        self.stream.synchronize()?;
        let stream = self.stream.clone();
        let dst = match (&mut self.gated, &self.spec.hyper) {
            (Some(g), _) => &mut g.streams,
            (None, Some(_)) => &mut self.pre.streams,
            _ => &mut self.pre.x,
        };
        stream.memcpy_htod(hidden, &mut dst.slice_mut(..hidden.len()))?;
        stream.synchronize()?;
        Ok(())
    }

    fn hand_buf_mut(&mut self) -> &mut cudarc::driver::CudaSlice<f32> {
        match (&mut self.gated, &self.spec.hyper) {
            (Some(g), _) => &mut g.streams,
            (None, Some(_)) => &mut self.pre.streams,
            _ => &mut self.pre.x,
        }
    }

    pub fn hidden_capacity_bytes(&self) -> usize {
        4 * self.hand_buf().len()
    }

    pub fn export_hidden_device(&self, rows: usize, dst: u64) -> Result<()> {
        use cudarc::driver::DevicePtr;
        let bytes = rows * self.hidden_row_bytes();
        ensure!(rows > 0 && bytes <= self.hidden_capacity_bytes(), "hidden state of {rows} rows does not fit the stage");
        self.ctx.bind_to_thread()?;
        self.stream.synchronize()?;
        {
            let (src, _guard) = self.hand_buf().device_ptr(&self.stream);
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
            let stream = self.stream.clone();
            let buf = match (&mut self.gated, &self.spec.hyper) {
                (Some(g), _) => &mut g.streams,
                (None, Some(_)) => &mut self.pre.streams,
                _ => &mut self.pre.x,
            };
            let (dst, _guard) = buf.device_ptr_mut(&stream);
            unsafe { cudarc::driver::result::memcpy_dtod_async(dst, src, bytes, self.stream.cu_stream())? };
        }
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn device_ordinal(&self) -> usize {
        self.ctx.ordinal()
    }

    pub fn hand_off(&self, next: &mut Runtime, rows: usize) -> Result<()> {
        ensure!(self.hand_width() == next.hand_width(), "stages disagree on the hidden size");
        let n = rows * self.hand_width();
        if self.device_ordinal() != next.device_ordinal() {
            let link = crate::link::DeviceLink::new(&self.ctx, &next.ctx)?;
            let next_stream = next.stream.clone();
            let (src, dst) = (self.hand_buf(), next.hand_buf_mut());
            return link.copy(src, &self.stream, dst, &next_stream, n);
        }
        ensure!(n <= next.hand_buf().len(), "hidden state of {n} floats exceeds the next stage");
        next.stream.synchronize()?;
        self.stream.synchronize()?;
        let dst = next.hand_buf_mut();
        self.stream.memcpy_dtod(&self.hand_buf().slice(..n), &mut dst.slice_mut(..n))?;
        self.stream.synchronize()?;
        Ok(())
    }
}
