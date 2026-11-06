use inference_bench_core::engine::Wake;
use std::sync::mpsc::{Receiver, channel};

pub fn spawn<T: Send + 'static>(wake: &Wake, work: impl FnOnce() -> T + Send + 'static) -> Receiver<T> {
    let (tx, rx) = channel();
    let wake = wake.clone();
    std::thread::spawn(move || {
        let _ = tx.send(work());
        wake();
    });
    rx
}

pub struct Pending<T>(Option<Receiver<T>>);

impl<T> Default for Pending<T> {
    fn default() -> Self {
        Self(None)
    }
}

impl<T> Pending<T> {
    pub fn start(&mut self, rx: Receiver<T>) {
        self.0 = Some(rx);
    }

    pub fn busy(&self) -> bool {
        self.0.is_some()
    }

    pub fn poll(&mut self) -> Option<T> {
        let rx = self.0.as_ref()?;
        match rx.try_recv() {
            Ok(v) => {
                self.0 = None;
                Some(v)
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(_) => {
                self.0 = None;
                None
            }
        }
    }
}
