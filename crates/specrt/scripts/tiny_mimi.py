import argparse
import json
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.models.pocket_tts_audio_tokenizer.modeling_pocket_tts_audio_tokenizer import (
    PocketTTSAudioTokenizerConfig,
    PocketTTSAudioTokenizerModel,
)

ROOT = Path(__file__).parent.parent
TINY = ROOT / "tests/fixtures/speech/tiny-mimi"
SEED = 0
FRAMES = 6

MIMI = dict(
    channels=1,
    sample_rate=24000,
    frame_rate=12.5,
    dimension=32,
    quantizer_dimension=8,
    quantizer_output_dimension=32,
    n_filters=4,
    n_residual_layers=1,
    ratios=[6, 5, 4],
    kernel_size=7,
    last_kernel_size=3,
    residual_kernel_size=3,
    dilation_base=2,
    compress=2,
    transformer_d_model=32,
    transformer_num_heads=2,
    transformer_num_layers=2,
    transformer_layer_scale=0.01,
    transformer_context=20,
    transformer_max_period=10000.0,
    transformer_dim_feedforward=64,
    gelu="erf",
)

CONFIG = dict(
    model_type="gradium_phonon_tts",
    architectures=["GradiumPhononTTSForConditionalGeneration"],
    mimi=MIMI,
)


def reference_config():
    m = MIMI
    return PocketTTSAudioTokenizerConfig(
        sampling_rate=m["sample_rate"],
        hidden_size=m["dimension"],
        num_filters=m["n_filters"],
        num_residual_layers=m["n_residual_layers"],
        upsampling_ratios=m["ratios"],
        kernel_size=m["kernel_size"],
        last_kernel_size=m["last_kernel_size"],
        residual_kernel_size=m["residual_kernel_size"],
        dilation_growth_rate=m["dilation_base"],
        compress=m["compress"],
        upsample_groups=m["dimension"],
        num_hidden_layers=m["transformer_num_layers"],
        intermediate_size=m["transformer_dim_feedforward"],
        num_attention_heads=m["transformer_num_heads"],
        num_key_value_heads=m["transformer_num_heads"],
        hidden_act="gelu",
        rope_parameters={"rope_type": "default", "rope_theta": m["transformer_max_period"]},
        sliding_window=m["transformer_context"],
        norm_eps=1e-5,
        latent_stride=int(m["sample_rate"] / (6 * 5 * 4) / m["frame_rate"]),
        latent_dim=m["quantizer_dimension"],
        encoder_config=None,
        attn_implementation="eager",
    )


def checkpoint_weights(g):
    m = MIMI
    d, heads = m["dimension"], m["transformer_num_heads"]
    ff = m["transformer_dim_feedforward"]
    r = lambda *s, std=0.3: (torch.randn(*s, generator=g) * std).float()
    w = {}
    w["mimi.quantizer.output_proj.weight"] = r(d, m["quantizer_dimension"], 1, std=0.4)
    w["mimi.upsample.convtr.convtr.weight"] = r(d, 1, 32, std=0.25)
    for l in range(m["transformer_num_layers"]):
        p = f"mimi.decoder_transformer.transformer.layers.{l}."
        w[p + "norm1.weight"] = 1.0 + r(d, std=0.1)
        w[p + "norm1.bias"] = r(d, std=0.1)
        w[p + "norm2.weight"] = 1.0 + r(d, std=0.1)
        w[p + "norm2.bias"] = r(d, std=0.1)
        w[p + "self_attn.in_proj.weight"] = r(3 * d, d, std=0.35)
        w[p + "self_attn.out_proj.weight"] = r(d, d, std=0.2)
        w[p + "linear1.weight"] = r(ff, d, std=0.2)
        w[p + "linear2.weight"] = r(d, ff, std=0.15)
        w[p + "layer_scale_1.scale"] = 0.3 + r(d, std=0.1)
        w[p + "layer_scale_2.scale"] = 0.3 + r(d, std=0.1)
    top = m["n_filters"] * 2 ** len(m["ratios"])
    w["mimi.decoder.model.0.conv.weight"] = r(top, d, m["kernel_size"], std=0.1)
    w["mimi.decoder.model.0.conv.bias"] = r(top, std=0.1)
    ch = top
    idx = 1
    for ratio in m["ratios"]:
        idx += 1
        w[f"mimi.decoder.model.{idx}.convtr.weight"] = r(ch, ch // 2, 2 * ratio, std=0.25)
        w[f"mimi.decoder.model.{idx}.convtr.bias"] = r(ch // 2, std=0.1)
        ch //= 2
        for _ in range(m["n_residual_layers"]):
            idx += 1
            hidden = ch // m["compress"]
            w[f"mimi.decoder.model.{idx}.block.1.conv.weight"] = r(hidden, ch, m["residual_kernel_size"], std=0.4)
            w[f"mimi.decoder.model.{idx}.block.1.conv.bias"] = r(hidden, std=0.1)
            w[f"mimi.decoder.model.{idx}.block.3.conv.weight"] = r(ch, hidden, 1, std=0.4)
            w[f"mimi.decoder.model.{idx}.block.3.conv.bias"] = r(ch, std=0.1)
        idx += 1
    idx += 1
    w[f"mimi.decoder.model.{idx}.conv.weight"] = r(1, ch, m["last_kernel_size"], std=0.4)
    w[f"mimi.decoder.model.{idx}.conv.bias"] = r(1, std=0.1)
    return w


def pairs_to_halves(rows, heads):
    n, cols = rows.shape
    half = n // heads // 2
    return rows.view(heads, half, 2, cols).transpose(1, 2).reshape(n, cols)


def reference_state(w):
    m = MIMI
    d, heads = m["dimension"], m["transformer_num_heads"]
    s = {"latent_proj.weight": w["mimi.quantizer.output_proj.weight"], "decoder.upsample.conv.weight": w["mimi.upsample.convtr.convtr.weight"]}
    for l in range(m["transformer_num_layers"]):
        p = f"mimi.decoder_transformer.transformer.layers.{l}."
        q = f"decoder.decoder_transformer.layers.{l}."
        qkv = w[p + "self_attn.in_proj.weight"]
        s[q + "self_attn.q_proj.weight"] = pairs_to_halves(qkv[:d], heads)
        s[q + "self_attn.k_proj.weight"] = pairs_to_halves(qkv[d : 2 * d], heads)
        s[q + "self_attn.v_proj.weight"] = qkv[2 * d :]
        s[q + "self_attn.o_proj.weight"] = w[p + "self_attn.out_proj.weight"]
        s[q + "mlp.fc1.weight"] = w[p + "linear1.weight"]
        s[q + "mlp.fc2.weight"] = w[p + "linear2.weight"]
        for a, b in [("norm1", "input_layernorm"), ("norm2", "post_attention_layernorm")]:
            s[q + b + ".weight"] = w[p + a + ".weight"]
            s[q + b + ".bias"] = w[p + a + ".bias"]
        s[q + "self_attn_layer_scale.scale"] = w[p + "layer_scale_1.scale"]
        s[q + "mlp_layer_scale.scale"] = w[p + "layer_scale_2.scale"]
    for k, v in w.items():
        if k.startswith("mimi.decoder.model."):
            rest = k[len("mimi.decoder.model.") :].replace(".convtr.", ".conv.")
            s["decoder.decoder_seanet.layers." + rest] = v
    return s


def write_bin(path, t):
    path.write_bytes(struct.pack(f"<{t.numel()}f", *t.detach().float().reshape(-1).tolist()))


def fixture():
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    w = checkpoint_weights(g)
    model = PocketTTSAudioTokenizerModel(reference_config()).eval()
    missing, unexpected = model.load_state_dict(reference_state(w), strict=False)
    missing = [k for k in missing if not k.endswith((".stride", ".kernel_size", ".padding_total"))]
    assert not missing and not unexpected, (missing, unexpected)
    latents = torch.randn(1, FRAMES, MIMI["quantizer_dimension"], generator=g)
    seen = {}
    model.decoder.decoder_transformer.register_forward_hook(lambda _m, _i, o: seen.__setitem__("transformer", o.last_hidden_state))
    model.decoder.upsample.register_forward_hook(lambda _m, _i, o: seen.__setitem__("upsample", o.transpose(1, 2)))
    with torch.no_grad():
        audio = model.decode(latents).audio_values
    TINY.mkdir(parents=True, exist_ok=True)
    (TINY / "reference").mkdir(exist_ok=True)
    (TINY / "config.json").write_text(json.dumps(CONFIG, indent=2, sort_keys=True) + "\n")
    save_file({k: v.contiguous() for k, v in sorted(w.items())}, str(TINY / "model.safetensors"))
    files = {}
    for key, t in [("latents", latents[0]), ("upsample", seen["upsample"][0]), ("transformer", seen["transformer"][0]), ("audio", audio[0, 0])]:
        name = f"{key}.bin"
        write_bin(TINY / "reference" / name, t)
        files[key] = {"file": name, "shape": list(t.shape)}
    meta = {
        "source": "transformers.models.pocket_tts_audio_tokenizer PocketTTSAudioTokenizerModel.decode (the Mimi decoder with latent_proj), f32, CPU, eager attention",
        "seed": SEED,
        "frames": FRAMES,
        "frame_samples": int(audio.shape[-1] // FRAMES),
        "files": files,
    }
    (TINY / "reference.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    total = sum(p.stat().st_size for p in TINY.rglob("*") if p.is_file())
    print(f"{TINY}: {total} bytes, audio {tuple(audio.shape)}, max |audio| {audio.abs().max():.3f}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="tiny Mimi decoder fixture").parse_args()
    fixture()
