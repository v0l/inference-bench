use crate::block::{Node, Row};
use specrt::anatomy::quant_tag;
use specrt::plan::{Op, Phase, Plan};
use specrt::spec::Spec;
use specrt::structure::Structure;

const PREFILL_TOKENS: usize = 512;
const DECODE_ROWS: usize = 8;
const SMS: usize = 84;

fn node(op: &Op, spec: &Spec, layer: usize, phase: Phase, root: &str, index: Option<usize>) -> Node {
    let tokens = match phase {
        Phase::Prefill => PREFILL_TOKENS,
        Phase::Decode => DECODE_ROWS,
    };
    let kernels = op.kernels(spec, layer, tokens, SMS).join(", ");
    Node {
        title: op.label(),
        shape: format!("{}: {kernels}", op.flow()),
        dtype: op.quant(spec, layer).map(|q| quant_tag(q).to_string()),
        tensors: op.checkpoint_tensors(spec, root, index),
    }
}

fn column(ops: &[Op], spec: &Spec, layer: usize, phase: Phase, root: &str, index: Option<usize>) -> Vec<Row> {
    ops.iter().map(|op| Row::Boxes(vec![node(op, spec, layer, phase, root, index)])).collect()
}

pub fn rows_for(spec: &Spec, structure: &Structure, layer: usize, phase: Phase) -> Result<Vec<Row>, String> {
    spec.runnable().map_err(|e| format!("not runnable on specrt: {e:#}"))?;
    let root = structure.root().ok_or("not runnable on specrt: the checkpoint has no layer stack")?;
    let plan = Plan::build(spec, phase, true);
    let mut rows = column(&plan.head, spec, layer, phase, root, None);
    rows.extend(column(plan.layer(spec.block(layer)), spec, layer, phase, root, Some(layer)));
    rows.extend(column(&plan.tail, spec, layer, phase, root, None));
    Ok(rows)
}

pub fn title(spec: &Spec, layer: usize, phase: Phase) -> String {
    let phase = match phase {
        Phase::Prefill => format!("prefill at {PREFILL_TOKENS} tokens"),
        Phase::Decode => format!("decode at {DECODE_ROWS} rows"),
    };
    let kind = if spec.block(layer).is_linear() { "gated delta-net" } else { "attention" };
    format!("engine plan, layer {layer} ({kind}), {phase}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn fixtures() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures")
    }

    fn load(config: &str, header: &str) -> (Spec, Structure) {
        let json = std::fs::read_to_string(fixtures().join(config)).unwrap();
        let mut spec = Spec::describe_hf_config(&json).unwrap();
        let structure = Structure::from_json(&std::fs::read_to_string(fixtures().join("headers").join(header)).unwrap()).unwrap();
        spec.apply_structure(&structure);
        (spec, structure)
    }

    fn nodes(rows: &[Row]) -> Vec<&Node> {
        rows.iter()
            .flat_map(|r| match r {
                Row::Boxes(n) => n.iter().collect(),
                Row::Io(n) => vec![n],
                Row::Add => vec![],
            })
            .collect()
    }

    #[test]
    fn every_op_weight_tensor_exists_in_the_checkpoint_structure() {
        for (config, header, layers) in [
            ("qwen3-0.6b.config.json", "qwen3-0.6b.json", vec![0, 5]),
            ("qwen3.8-27b.config.json", "qwen3.8-27b-fp8.json", vec![0, 1, 3]),
            ("llama-3.2-1b.config.json", "llama-3.2-1b.json", vec![0, 2]),
        ] {
            let (spec, structure) = load(config, header);
            for layer in layers {
                if structure.layer(layer).is_none_or(|l| l.index != layer) {
                    continue;
                }
                for phase in [Phase::Prefill, Phase::Decode] {
                    let rows = rows_for(&spec, &structure, layer, phase).unwrap();
                    let drawn = nodes(&rows);
                    assert!(drawn.iter().any(|n| !n.tensors.is_empty()));
                    for n in drawn {
                        for t in &n.tensors {
                            assert!(structure.contains(t), "{header} layer {layer}: {} reads {t}, not in the checkpoint", n.title);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_engine_tab_shows_kernel_names_and_quant() {
        let (spec, structure) = load("qwen3.8-27b.config.json", "qwen3.8-27b-fp8.json");
        let rows = rows_for(&spec, &structure, 0, Phase::Prefill).unwrap();
        let drawn = nodes(&rows);
        let delta = drawn.iter().find(|n| n.title == "delta rule").unwrap();
        assert!(delta.shape.contains("gdn_chunk_scan"));
        let lin = drawn.iter().find(|n| n.title == "lin_in").unwrap();
        assert!(lin.shape.contains("lin_in_"));
        assert_eq!(lin.dtype.as_deref(), Some("fp8 block-128"));
    }

    #[test]
    fn a_model_specrt_cannot_run_shows_the_reason() {
        let (spec, structure) = load("qwen3.6-35b-a3b-fp8.config.json", "qwen3.6-35b-a3b-fp8.json");
        let err = rows_for(&spec, &structure, 0, Phase::Prefill).unwrap_err();
        assert!(err.starts_with("not runnable on specrt: "), "{err}");
        assert!(err.contains("mixture-of-experts"), "{err}");
    }
}
