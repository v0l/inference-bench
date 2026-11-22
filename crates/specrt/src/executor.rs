use crate::plan::{Op, Plan};
use crate::spec::Block;
use anyhow::Result;

pub trait PlanBackend<R> {
    fn layers(&self) -> usize;
    fn block(&self, layer: usize) -> Block;
    fn sparse(&self, _layer: usize) -> bool {
        false
    }
    fn run_op(&self, op: &Op, layer: Option<usize>, run: &R) -> Result<()>;
    fn enter_layer(&self, _layer: usize) -> Result<()> {
        Ok(())
    }
    fn leave_layer(&self, _layer: usize) -> Result<()> {
        Ok(())
    }
}

pub fn run_ops<R, B: PlanBackend<R>>(be: &B, ops: &[Op], layer: Option<usize>, run: &R) -> Result<()> {
    for op in ops {
        be.run_op(op, layer, run)?;
    }
    Ok(())
}

pub fn run_plan<R, B: PlanBackend<R>>(be: &B, plan: &Plan, run: &R, after_layer: &mut dyn FnMut(usize) -> Result<()>) -> Result<()> {
    run_ops(be, &plan.head, None, run)?;
    for i in 0..be.layers() {
        be.enter_layer(i)?;
        run_ops(be, plan.layer_of(be.block(i), be.sparse(i)), Some(i), run)?;
        be.leave_layer(i)?;
        after_layer(i)?;
    }
    run_ops(be, &plan.tail, None, run)
}
