import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import reference as R

FIXTURE = Path(__file__).parent.parent / "tests/fixtures/tiny-qwen3.5-27b"
NVFP4_UNTIL = 3
LAYERS = 4
IMAGE, VIDEO, START, END = 1000, 1001, 998, 999
PROMPT = 48
GREEDY = 64
MTP_PREFIXES = [48, 56, 64, 72, 80, 88, 96, 104]
VISION_CASES = {"image": [(1, 8, 8)], "images2": [(1, 4, 12), (1, 8, 8)]}
CALIBRATION_TOKENS = 1024


def build_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    return Qwen3_5Config(
        text_config=dict(
            vocab_size=1024,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=LAYERS,
            layer_types=["linear_attention"] * (LAYERS - 1) + ["full_attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=256,
            linear_num_key_heads=2,
            linear_num_value_heads=6,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            rms_norm_eps=1e-6,
            rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10], "mrope_interleaved": True},
            attn_output_gate=True,
            tie_word_embeddings=False,
            max_position_embeddings=4096,
            eos_token_id=None,
            bos_token_id=None,
            pad_token_id=None,
        ),
        vision_config=dict(depth=2, hidden_size=128, intermediate_size=256, num_heads=4, out_hidden_size=256, num_position_embeddings=64, deepstack_visual_indexes=[]),
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        vision_start_token_id=START,
        vision_end_token_id=END,
    )


def random_weights(model):
    weights = R.tiny_weights(model)
    g = torch.Generator().manual_seed(R.TINY_SEED + 11)
    for name, p in model.state_dict().items():
        if not name.startswith("model.visual."):
            continue
        if name.endswith("norm.weight") or name.endswith("norm1.weight") or name.endswith("norm2.weight"):
            v = 1.0 + 0.1 * torch.randn(p.shape, generator=g)
        elif name.endswith(".bias"):
            v = 0.02 * torch.randn(p.shape, generator=g)
        elif name.endswith("pos_embed.weight"):
            v = 0.5 * torch.randn(p.shape, generator=g)
        else:
            fan_in = 1
            for d in p.shape[1:]:
                fan_in *= d
            v = torch.randn(p.shape, generator=g) / fan_in ** 0.5
        weights[name] = v.to(torch.bfloat16).contiguous()
    return {k: v for k, v in weights.items() if not k.endswith("inv_freq")}


def canonical(name):
    return "model." + name[len("model.language_model."):] if name.startswith("model.language_model.") else name


def text_model(config, state):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    model = Qwen3_5ForCausalLM(config.text_config).float().eval()
    model.load_state_dict({canonical(k): v.float() for k, v in state.items() if not k.startswith("model.visual.") and not k.startswith("mtp.")}, strict=True)
    return model


def calibrate(config, weights):
    model = text_model(config, {k: v for k, v in weights.items() if not k.startswith("mtp.")})
    amax = {}

    def hook(name):
        def f(_m, args):
            amax[name] = max(amax.get(name, 0.0), float(args[0].abs().max()))

        return f

    for name, module in model.named_modules():
        if name.endswith(("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")):
            module.register_forward_pre_hook(hook(name))
    with torch.inference_mode():
        model(R.tiny_tokens(CALIBRATION_TOKENS, 1)[None])
    return {"model.language_model." + k[len("model."):]: v for k, v in amax.items()}


def shard_of(name):
    parts = name.split(".")
    if parts[0] == "mtp":
        return "mtp.safetensors"
    if parts[:3] == ["model", "language_model", "layers"]:
        return f"layers-{parts[3]}.safetensors"
    return "outside.safetensors"


def write_checkpoint(config, weights, amax, dst):
    from safetensors.torch import save_file

    dst.mkdir(parents=True, exist_ok=True)
    for f in dst.glob("*.safetensors"):
        f.unlink()
    shards, targets = {}, {"group_0": [], "group_1": []}
    for name, w in sorted(weights.items()):
        group = shards.setdefault(shard_of(name), {})
        if name.startswith("mtp.") or name.startswith("model.visual.") or not R.is_fp8_projection(name):
            group[name] = w.contiguous()
            continue
        module = name[: -len(".weight")]
        layer = int(name.split(".")[3])
        if module.split(".")[-1] in ("gate_proj", "up_proj", "down_proj") and layer < NVFP4_UNTIL:
            mlp = module.rsplit(".", 1)[0]
            peers = [module] if module.endswith("down_proj") else [mlp + ".gate_proj", mlp + ".up_proj"]
            wgs = R.nvfp4_global(max(float(weights[p + ".weight"].float().abs().max()) for p in peers))
            igs = R.nvfp4_global(amax[peers[0]])
            packed, scale = R.quantize_nvfp4(w, wgs)
            group[module + ".weight_packed"] = packed.contiguous()
            group[module + ".weight_scale"] = scale.contiguous()
            group[module + ".weight_global_scale"] = wgs
            group[module + ".input_global_scale"] = igs
            targets["group_1"].append(module)
        else:
            q, s = R.quantize_fp8_channel(w)
            group[name] = q.contiguous()
            group[module + ".weight_scale"] = s.contiguous()
            targets["group_0"].append(module)
    weight_map = {}
    for shard, tensors in shards.items():
        save_file(tensors, dst / shard)
        weight_map.update({k: shard for k in tensors})
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    fp8_args = {"num_bits": 8, "type": "float", "symmetric": True, "dynamic": False, "strategy": "channel", "group_size": None}
    fp4_args = {"num_bits": 4, "type": "float", "symmetric": True, "dynamic": False, "strategy": "tensor_group", "group_size": 16, "scale_dtype": "torch.float8_e4m3fn"}
    out = config.to_dict()
    out["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    out["language_model_only"] = False
    out["dtype"] = "bfloat16"
    out["text_config"]["mtp_num_hidden_layers"] = 1
    out["text_config"]["mtp_use_dedicated_embeddings"] = False
    out["text_config"]["full_attention_interval"] = 4
    out["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {"format": "float-quantized", "weights": fp8_args, "input_activations": {**fp8_args, "dynamic": True, "strategy": "token"}, "targets": sorted(targets["group_0"])},
            "group_1": {"format": "nvfp4-pack-quantized", "weights": fp4_args, "input_activations": {**fp4_args, "dynamic": "local"}, "targets": sorted(targets["group_1"])},
        },
        "ignore": ["lm_head", "re:model.visual.*", "re:mtp.*"],
    }
    (dst / "config.json").write_text(json.dumps(out, indent=2))


def generate_fixture(dst):
    from safetensors.torch import save_file
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    torch.manual_seed(R.TINY_SEED)
    config = build_config()
    weights = random_weights(Qwen3_5ForConditionalGeneration(config))
    amax = calibrate(config, weights)
    weights.update(R.tiny_mtp_weights(config.text_config))
    write_checkpoint(config, weights, amax, dst)
    print("fixture", dst)


def load_reference_state(fixture):
    from safetensors.torch import load_file

    index = json.loads((fixture / "model.safetensors.index.json").read_text())
    raw = {}
    for shard in sorted(set(index["weight_map"].values())):
        raw.update(load_file(fixture / shard))
    mtp = {k: v for k, v in raw.items() if k.startswith("mtp.")}
    state, input_scales = R.dequantize_compressed({k: v for k, v in raw.items() if not k.startswith("mtp.")})
    return state, input_scales, mtp


def act_quantise(model, input_scales):
    scales = {canonical(k): v for k, v in input_scales.items()}
    handles = []
    for name, module in model.named_modules():
        key = canonical(name)
        if key in scales:
            gs = scales[key]
            handles.append(module.register_forward_pre_hook(lambda _m, args, gs=gs: (R.nvfp4_qdq(args[0], gs).to(args[0].dtype),)))
        elif isinstance(module, torch.nn.Linear) and key.startswith("model.layers.") and key.split(".")[-1] in R.FP8_PROJECTIONS:
            handles.append(module.register_forward_pre_hook(lambda _m, args: (R.quantize_activation_fp8_token(args[0]),)))
    return handles


def text_references(model, input_scales, out_dir, versions):
    ids = R.tiny_tokens(PROMPT, 3)
    with torch.inference_mode():
        greedy = model.generate(ids[None], max_new_tokens=GREEDY, do_sample=False)[0, PROMPT:]
        seq = torch.cat([ids, greedy])
        logits = model(seq[None]).logits[0]
        R.save(out_dir, "tiny27b-text", {"logits": logits}, {"prompt": ids.tolist(), "greedy": greedy.tolist(), "tokens": seq.tolist(), **versions})
        handles = act_quantise(model, input_scales)
        act_greedy = model.generate(ids[None], max_new_tokens=GREEDY, do_sample=False)[0, PROMPT:]
        act_logits = model(seq[None]).logits[0]
        for h in handles:
            h.remove()
    R.save(out_dir, "tiny27b-text-act", {"logits": act_logits}, {"prompt": ids.tolist(), "greedy": act_greedy.tolist(), "tokens": seq.tolist(), **versions})
    return seq


def mtp_references(model, mtp_state, seq, out_dir, versions):
    mtp = R.TinyMtp(model.config, mtp_state).eval()
    chains, first, d1, d2 = [], [], [], []
    with torch.inference_mode():
        for n in MTP_PREFIXES:
            ids = seq[:n]
            hidden = model.model(ids[None]).last_hidden_state
            t = int(model.lm_head(hidden)[0, -1].argmax())
            shifted = torch.cat([ids[1:], torch.tensor([t])])
            logits, mtp_hidden = mtp(model, shifted[None], hidden, torch.arange(n))
            a = int(logits[-1].argmax())
            chain_tokens = torch.cat([shifted, torch.tensor([a])])
            chain_hidden = torch.cat([hidden, mtp_hidden[None, -1:]], dim=1)
            chain, _ = mtp(model, chain_tokens[None], chain_hidden, torch.arange(n + 1))
            chains.append(chain[-1])
            first.append(t)
            d1.append(a)
            d2.append(int(chain[-1].argmax()))
    R.save(out_dir, "tiny27b-mtp", {"chain": torch.stack(chains)}, {"prefixes": MTP_PREFIXES, "first": first, "draft": d1, "chain_draft": d2, **versions})


def vision_references(config, state, out_dir, versions):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration(config).float().eval()
    model.load_state_dict(state, strict=True)
    vc = config.vision_config
    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    merge = vc.spatial_merge_size
    g = torch.Generator().manual_seed(R.TINY_SEED + 21)
    for case, grids in VISION_CASES.items():
        patches = [torch.rand((t * h * w, patch_dim), generator=g) * 2 - 1 for t, h, w in grids]
        pixels = torch.cat(patches)
        grid_thw = torch.tensor(grids)
        ids = [int(x) for x in torch.randint(0, 990, (5,), generator=g)]
        types = [0] * 5
        for (t, h, w), _ in zip(grids, patches):
            n = t * h * w // (merge * merge)
            ids += [START] + [IMAGE] * n + [END] + [int(x) for x in torch.randint(0, 990, (3,), generator=g)]
            types += [0] + [1] * n + [0] * 4
        ids_t = torch.tensor(ids)
        with torch.inference_mode():
            embeds = model.model.visual(pixels, grid_thw).pooler_output
            logits = model(input_ids=ids_t[None], pixel_values=pixels, image_grid_thw=grid_thw, mm_token_type_ids=torch.tensor(types)[None]).logits[0]
        R.save(out_dir, f"tiny27b-vision-{case}", {"patches": pixels, "embeds": embeds, "logits": logits}, {"grids": [list(x) for x in grids], "ids": ids, "patch_dim": patch_dim, **versions})
        print("vision", case, tuple(embeds.shape), tuple(logits.shape))


def references(fixture, out_dir):
    torch.set_num_threads(4)
    out_dir.mkdir(parents=True, exist_ok=True)
    versions = {"torch": torch.__version__, "transformers": __import__("transformers").__version__}
    config = build_config()
    state, input_scales, mtp_state = load_reference_state(fixture)
    model = text_model(config, state)
    seq = text_references(model, input_scales, out_dir, versions)
    mtp_references(model, mtp_state, seq, out_dir, versions)
    vision_references(config, state, out_dir, versions)


def main():
    command = sys.argv[1]
    if command == "fixture":
        generate_fixture(Path(sys.argv[2]) if len(sys.argv) > 2 else FIXTURE)
    elif command == "reference":
        references(FIXTURE, Path(sys.argv[2]) if len(sys.argv) > 2 else FIXTURE / "reference")
    elif command == "all":
        generate_fixture(FIXTURE)
        references(FIXTURE, FIXTURE / "reference")
    else:
        sys.exit("usage: tiny27b.py fixture|reference|all [dir]")


if __name__ == "__main__":
    main()
