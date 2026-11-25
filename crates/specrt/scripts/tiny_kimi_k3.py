import hashlib
import importlib
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).parent.parent
REPO = "moonshotai/Kimi-K3"
REVISION = "c7945ea0e737dcaddd22c68ddb7639b1407c081c"
REMOTE = {
    "modeling_kimi_linear.py": "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a",
    "configuration_kimi_k3.py": "735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae",
}
REAL = Path.home() / ".cache/specrt-popular/files/moonshotai__Kimi-K3/config.json"
NAME = "tiny-kimi-k3"
LINEAR_NAME = "tiny-kimi-linear"
SEED = 0
LAYERS = 6
PROMPT = 30
DECODE = 4
PREFIX = "language_model."
MAX_BYTES = 5 * 1024 * 1024

SIZES = dict(
    vocab_size=512,
    hidden_size=128,
    head_dim=72,
    intermediate_size=256,
    moe_intermediate_size=64,
    num_hidden_layers=LAYERS,
    num_attention_heads=4,
    num_key_value_heads=4,
    q_lora_rank=64,
    kv_lora_rank=64,
    qk_nope_head_dim=48,
    qk_rope_head_dim=16,
    v_head_dim=32,
    num_experts=8,
    num_experts_per_token=2,
    num_shared_experts=1,
    routed_scaling_factor=2.446,
    routed_expert_hidden_size=64,
    attn_res_block_size=2,
    first_k_dense_replace=1,
    max_position_embeddings=512,
    bos_token_id=None,
    eos_token_id=None,
    pad_token_id=None,
    linear_attn_config={
        "full_attn_layers": [4, 6],
        "kda_layers": [1, 2, 3, 5],
        "gate_lower_bound": -5.0,
        "head_dim": 64,
        "num_heads": 2,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
    },
)

F32_SUFFIXES = ("self_attn.A_log", "self_attn.dt_bias")


def fixture_dir(name):
    return ROOT / "tests/fixtures" / name


def remote_package():
    from huggingface_hub import hf_hub_download

    pkg = Path.home() / ".cache/specrt-ref/kimi_k3_remote"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for f, digest in REMOTE.items():
        path = Path(hf_hub_download(REPO, f, revision=REVISION))
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        assert got == digest, (f, got)
        shutil.copyfile(path, pkg / f)
    return pkg


def install_fla(ref):
    class ShortConvolution(nn.Conv1d):
        def __init__(self, hidden_size, kernel_size, activation=None, bias=False, **_):
            super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, padding=kernel_size - 1, bias=bias)
            self.activation = activation

        def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None, **_):
            assert cu_seqlens is None
            xt = x.transpose(1, 2)
            w = self.weight.squeeze(1)
            k = w.shape[-1]
            if cache is not None and x.shape[1] == 1:
                state = cache.clone()
                y = ref.causal_conv1d_update(xt, state, w, self.bias, self.activation)
                return y.transpose(1, 2), state
            full = xt if cache is None else torch.cat([cache, xt], dim=-1)
            y = ref.causal_conv1d_fn(full, w, self.bias, self.activation)[..., -x.shape[1]:]
            state = F.pad(full, (k, 0))[..., -k:].contiguous()
            return y.transpose(1, 2), state if output_final_state else None

    def gated_norm(hidden, eps=1e-6, activation="sigmoid", **_):
        assert activation == "sigmoid"
        return ref.KimiLinearRMSNormGated(hidden, eps=eps)

    def gate(g, a_log, dt_bias, lower_bound):
        heads = g.shape[-2]
        g = g.float() + dt_bias.float().view(heads, -1)
        a = a_log.float().view(heads, 1).exp()
        return lower_bound * torch.sigmoid(a * g) if lower_bound is not None else -a * F.softplus(g)

    def prepared(g, beta, A_log, dt_bias, lower_bound, use_gate_in_kernel, use_beta_sigmoid_in_kernel, cu_seqlens):
        assert use_gate_in_kernel and use_beta_sigmoid_in_kernel and cu_seqlens is None
        return gate(g, A_log, dt_bias, lower_bound), beta.float().sigmoid()

    def chunk_kda(q, k, v, g, beta, A_log=None, dt_bias=None, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False, safe_gate=False, lower_bound=None, transpose_state_layout=False, cu_seqlens=None, **_):
        g, beta = prepared(g, beta, A_log, dt_bias, lower_bound, use_gate_in_kernel, use_beta_sigmoid_in_kernel, cu_seqlens)
        return ref.chunk_kimi_delta_attention(q, k, v, g=g, beta=beta, initial_state=initial_state, output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)

    def fused_recurrent_kda(q, k, v, g, beta, A_log=None, dt_bias=None, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False, lower_bound=None, transpose_state_layout=False, cu_seqlens=None, **_):
        g, beta = prepared(g, beta, A_log, dt_bias, lower_bound, use_gate_in_kernel, use_beta_sigmoid_in_kernel, cu_seqlens)
        return ref.recurrent_kimi_delta_attention(q, k, v, g=g, beta=beta, initial_state=initial_state, output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)

    def unpadded(*_, **__):
        raise NotImplementedError("the reference runs unpadded batches only")

    names = ["fla", "fla.modules", "fla.ops", "fla.ops.kda", "fla.ops.utils", "fla.ops.utils.index", "fla.utils"]
    mods = {n: types.ModuleType(n) for n in names}
    mods["fla.modules"].ShortConvolution = ShortConvolution
    mods["fla.modules"].FusedRMSNormGated = gated_norm
    mods["fla.ops.kda"].chunk_kda = chunk_kda
    mods["fla.ops.kda"].fused_recurrent_kda = fused_recurrent_kda
    mods["fla.ops.utils.index"].prepare_cu_seqlens_from_mask = unpadded
    mods["fla.ops.utils.index"].prepare_lens_from_mask = unpadded
    mods["fla.utils"].tensor_cache = lambda f: f
    sys.modules.update(mods)


def causal_mask(config=None, input_embeds=None, attention_mask=None, cache_position=None, past_key_values=None, **_):
    q = input_embeds.shape[1]
    past = int(cache_position[0])
    keys = torch.arange(past + q)[None, :]
    rows = (past + torch.arange(q))[:, None]
    mask = torch.zeros(q, past + q).masked_fill(keys > rows, float("-inf"))
    return mask[None, None]


def remote():
    import transformers.utils.generic as generic
    from transformers.models.kimi_linear import modeling_kimi_linear as ref
    from transformers.utils.output_capturing import OutputRecorder

    generic.OutputRecorder = OutputRecorder
    install_fla(ref)
    pkg = remote_package()
    sys.path.insert(0, str(pkg.parent))
    mod = importlib.import_module(f"{pkg.name}.modeling_kimi_linear")
    mod.create_causal_mask = causal_mask
    return mod


def text_config(extra=None):
    real = json.loads(REAL.read_text())["text_config"]
    for k in ("auto_map", "quantization_config", "architectures", "_name_or_path", "id2label", "label2id"):
        real.pop(k, None)
    real.update(SIZES)
    real.update(extra or {})
    return real


def build(mod, cfg_dict):
    config_mod = sys.modules[mod.__name__.rsplit(".", 1)[0] + ".configuration_kimi_k3"]
    cfg = config_mod.KimiLinearConfig(**cfg_dict)
    torch.manual_seed(SEED)
    model = mod.KimiLinearForCausalLM(cfg).float().eval()
    model.config._attn_implementation = "eager"
    return model


def random_tensor(name, shape, g):
    randn = lambda: torch.randn(shape, generator=g)
    if name.endswith("e_score_correction_bias"):
        return 0.2 * randn()
    if name.endswith("A_log") or name.endswith("dt_bias") or name.endswith("_conv1d.weight"):
        return 0.5 * randn()
    if len(shape) == 1:
        return 1.0 + 0.1 * randn()
    if "embed_tokens" in name or name == "lm_head.weight":
        return randn()
    return randn() / shape[-1] ** 0.5


def stored_dtype(name):
    return torch.float32 if name.endswith(F32_SUFFIXES) else torch.bfloat16


def generate(model):
    from safetensors.torch import save_file
    import transformers

    g = torch.Generator().manual_seed(SEED)
    shapes = {n: tuple(p.shape) for n, p in model.state_dict().items()}
    tensors = {PREFIX + n: random_tensor(n, shapes[n], g).to(stored_dtype(n)).contiguous() for n in sorted(shapes)}
    out = fixture_dir(NAME)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    cfg = {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "model_type": "kimi_k3",
        "text_config": text_config(),
        "transformers_version": transformers.__version__,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def load(model, name, prefix):
    from safetensors.torch import load_file

    raw = {k[len(prefix):]: v.float() for k, v in load_file(fixture_dir(name) / "model.safetensors").items()}
    own = model.state_dict()
    raw = {(k if k in own else k.replace(".block_sparse_moe.", ".mlp.")): v for k, v in raw.items()}
    raw = {k: v.reshape(own[k].shape) for k, v in raw.items()}
    model.load_state_dict(raw, strict=True)


def prompt_tokens(vocab):
    g = torch.Generator().manual_seed(SEED + 3)
    return torch.randint(0, vocab, (PROMPT,), generator=g)


def run(model, layers):
    ids = prompt_tokens(model.config.vocab_size)
    trace = {}
    hooks = [model.model.layers[i].register_forward_hook(lambda _m, _a, o, i=i: trace.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0].detach().clone())) for i in layers]
    with torch.inference_mode():
        res = model(input_ids=ids[None], use_cache=True)
        for h in hooks:
            h.remove()
        prefill = res.logits[0]
        past = res.past_key_values
        greedy = [int(prefill[-1].argmax())]
        steps = []
        for _ in range(DECODE):
            res = model(input_ids=torch.tensor([[greedy[-1]]]), past_key_values=past, use_cache=True)
            past = res.past_key_values
            steps.append(res.logits[0, -1])
            greedy.append(int(res.logits[0, -1].argmax()))
        full = model(input_ids=torch.cat([ids, torch.tensor(greedy[:-1])])[None], use_cache=False).logits[0]
    hidden = torch.stack([trace[i] for i in layers]) if layers else None
    return ids, prefill, torch.stack(steps), greedy, hidden, full


def floats(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.float32))


def check(mod):
    cfg = json.loads((fixture_dir(LINEAR_NAME) / "config.json").read_text())
    model = build(mod, cfg)
    load(model, LINEAR_NAME, "")
    ids, prefill, decode, greedy, _, _ = run(model, [])
    ref = fixture_dir(LINEAR_NAME) / "reference"
    meta = json.loads((ref / f"{LINEAR_NAME}.ref.json").read_text())
    assert ids.tolist() == meta["tokens"]
    p = float((prefill.flatten() - floats(ref / f"{LINEAR_NAME}.ref.bin")).abs().max())
    d = float((decode.flatten() - floats(ref / f"{LINEAR_NAME}.decode.logits.bin")).abs().max())
    print(LINEAR_NAME, "remote code with the fla stubs against transformers: prefill max abs", p, "decode max abs", d, "greedy", greedy, meta["greedy"])
    assert p < 1e-4 and d < 1e-4 and greedy == meta["greedy"], (p, d)
    return {"prefill": p, "decode": d}


def references(model, stub):
    import transformers

    out = fixture_dir(NAME) / "reference"
    out.mkdir(parents=True, exist_ok=True)
    layers = list(range(LAYERS))
    ids, prefill, decode, greedy, hidden, full = run(model, layers)
    recompute = float((full[PROMPT:] - decode).abs().max())
    prefill_gap = float((full[:PROMPT] - prefill).abs().max())
    for k, t in {"prefill": prefill, "decode": decode, "hidden": hidden}.items():
        assert torch.isfinite(t).all(), k
    assert recompute < 1e-3 and prefill_gap < 1e-3, (recompute, prefill_gap)
    prefill.numpy().astype(np.float32).tofile(out / f"{NAME}.ref.bin")
    decode.numpy().astype(np.float32).tofile(out / f"{NAME}.decode.logits.bin")
    hidden.numpy().astype(np.float32).tofile(out / f"{NAME}.hidden.streams.bin")
    meta = {
        "tokens": ids.tolist(),
        "greedy": greedy,
        "vocab": model.config.vocab_size,
        "decode_steps": DECODE,
        "hidden_layers": layers,
        "hidden_size": model.config.hidden_size,
        "decode_vs_recompute": recompute,
        "remote": {"repo": REPO, "revision": REVISION, "files": REMOTE},
        "stub_vs_transformers_on_tiny_kimi_linear": stub,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    (out / f"{NAME}.ref.json").write_text(json.dumps(meta) + "\n")
    print(NAME, "prefill", tuple(prefill.shape), "greedy", greedy, "decode vs recompute", recompute)


def fixture_bytes(name):
    return sum(p.stat().st_size for p in fixture_dir(name).rglob("*") if p.is_file())


def main():
    torch.set_num_threads(4)
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    mod = remote()
    stub = check(mod)
    if command == "check":
        return
    model = build(mod, text_config())
    if command in ("all", "fixture"):
        generate(model)
    load(model, NAME, PREFIX)
    if command in ("all", "reference"):
        references(model, stub)
    size = fixture_bytes(NAME)
    print(NAME, size, "bytes")
    assert size < MAX_BYTES, size


if __name__ == "__main__":
    main()
