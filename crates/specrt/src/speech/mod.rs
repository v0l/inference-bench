pub mod compile;
pub mod cuda;
pub mod graph;
pub mod host;

pub use graph::{SpeechOp, SpeechPlan};
pub use host::HostSpeech;
