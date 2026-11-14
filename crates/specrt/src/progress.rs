use std::cell::RefCell;
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Kernels,
    Weights,
    KvCache,
}

impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Stage::Kernels => "compiling kernels",
            Stage::Weights => "reading weights",
            Stage::KvCache => "allocating KV cache",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    pub stage: Stage,
    pub done: u64,
    pub total: u64,
}

#[derive(Clone)]
pub struct Sink(Arc<dyn Fn(Progress) + Send + Sync>);

impl Sink {
    pub fn new(f: impl Fn(Progress) + Send + Sync + 'static) -> Self {
        Sink(Arc::new(f))
    }

    pub fn send(&self, p: Progress) {
        (self.0)(p)
    }
}

impl fmt::Debug for Sink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Sink")
    }
}

impl PartialEq for Sink {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Sink>> = const { RefCell::new(None) };
}

pub fn with_sink<T>(sink: Option<Sink>, f: impl FnOnce() -> T) -> T {
    let previous = CURRENT.with(|c| c.replace(sink));
    let out = f();
    CURRENT.with(|c| *c.borrow_mut() = previous);
    out
}

pub fn current() -> Option<Sink> {
    CURRENT.with(|c| c.borrow().clone())
}

pub fn report(stage: Stage, done: u64, total: u64) {
    if let Some(sink) = current() {
        sink.send(Progress { stage, done, total });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn reports_reach_the_installed_sink_and_nothing_else() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        report(Stage::Weights, 1, 2);
        with_sink(Some(Sink::new(move |p| s.lock().unwrap().push(p))), || {
            report(Stage::Weights, 1, 2);
            report(Stage::KvCache, 0, 0);
        });
        report(Stage::Weights, 2, 2);
        assert_eq!(
            *seen.lock().unwrap(),
            vec![Progress { stage: Stage::Weights, done: 1, total: 2 }, Progress { stage: Stage::KvCache, done: 0, total: 0 }]
        );
        assert!(current().is_none());
    }

    #[test]
    fn a_nested_sink_restores_the_outer_one() {
        let outer = Sink::new(|_| {});
        with_sink(Some(outer.clone()), || {
            with_sink(None, || assert!(current().is_none()));
            assert_eq!(current(), Some(outer.clone()));
        });
    }
}
