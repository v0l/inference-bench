#!/usr/bin/env bash
set -euo pipefail
image=ghcr.io/local-inference-lab/vllm:release-10152026
snap=$(echo ~/.cache/huggingface/hub/models--Piecrust--GLM-5.3-Flash-FP4-Abliterated/snapshots/*/)
module=${1:-model.language_model.layers.3.mlp.experts.0.down_proj.weight_scale}
out=$(dirname "$0")/../tests/fixtures/csf
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
~/.venv-vllm/bin/python - "$snap" "$module" "$work/in.safetensors" <<'EOF'
import json, sys
from safetensors import safe_open
from safetensors.numpy import save_file
snap, module, dst = sys.argv[1:]
index = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
f = safe_open(snap + index[module + ".csf32_bases"], "np")
parts = {p: f.get_tensor(f"{module}.csf32_{p}") for p in ["bases", "codes", "page_bases", "offsets", "exceptions"]}
rows, packed = f.get_slice(module[: -len("_scale")]).get_shape()
save_file(parts, dst, metadata={"rows": str(rows), "columns": str(packed * 2 // 16)})
EOF
docker run --rm --network none --user "$(id -u):$(id -g)" -v "$work:/w" --entrypoint bash "$image" -c '
cd /opt/venv/lib/python3.12/site-packages/flashinfer/experimental/b12x/loader && /opt/venv/bin/python - <<EOF
import hashlib, importlib.util, sys
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file
spec = importlib.util.spec_from_file_location("csf", "_csf_blocks.py")
csf = importlib.util.module_from_spec(spec)
sys.modules["csf"] = csf
spec.loader.exec_module(csf)
meta = safe_open("/w/in.safetensors", "np").metadata()
plane = csf.CsfBlockPlane(int(meta["rows"]), int(meta["columns"]), "nvfp4", **load_file("/w/in.safetensors"))
plane.validate()
def logical(q):
    b, f, pos, v = csf.logical_csf_components(q)
    out = (b[:, None].astype(np.uint16) + f).astype(np.uint8)
    out.ravel()[pos] = v
    return out
print("sha256", hashlib.sha256(logical(plane).tobytes()).hexdigest())
for name, rr, cc in [("slice_head", (0, 160), (0, 10)), ("slice_tail", (plane.rows - 160, plane.rows), (plane.columns - 11, plane.columns))]:
    q = csf.slice_csf_blocks(plane, row_range=rr, column_range=cc)
    q.validate()
    parts = {k: getattr(q, k) for k in ["bases", "codes", "page_bases", "offsets", "exceptions"]}
    parts["expected"] = logical(q)
    save_file(parts, f"/w/{name}.safetensors", metadata={"rows": str(q.rows), "columns": str(q.columns)})
EOF'
cp "$work/slice_head.safetensors" "$work/slice_tail.safetensors" "$out/"
