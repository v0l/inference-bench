import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests/fixtures/haru"
TINY = FIXTURE / "tiny"
MAX_BYTES = 3 * 1024 * 1024
SEED = 0
CARD_TOKENS = 12
REFERENCE_FRAMES = 3
STEPS = 6
FRAME = 3200

CONFIG = {
    "model_type": "haru",
    "architectures": ["HaruForConditionalGeneration"],
    "dtype": "bfloat16",
    "sample_rate": 24000,
    "frame_samples": FRAME,
    "frame_rate": 7.5,
    "latent_dim": 64,
    "pad_token_id": 508,
    "eop_token_id": 509,
    "lm": {
        "model_type": "qwen3_5_text",
        "vocab_size": 512,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "hidden_act": "silu",
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-06,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "mamba_ssm_dtype": "float32",
        "partial_rotary_factor": 0.25,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [3, 3, 2],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "tie_word_embeddings": True,
        "initializer_range": 0.02,
        "eos_token_id": 510,
    },
    "codec": {
        "latent_dim": 64,
        "base_channels": 2,
        "ratios": [8, 5, 5, 4, 2, 2],
        "encoder_depths": [1, 1, 1, 1, 1, 1, 1],
        "decoder_depths": [1, 1, 1, 1, 1, 1, 1],
        "norm_eps": 1e-05,
        "latent_std": 0.5,
    },
    "head": {
        "hidden_size": 64,
        "latent_dim": 64,
        "num_layers": 2,
        "ffn_ratio": 3.0,
        "norm_eps": 1e-05,
        "interval_embed": 7.0,
        "num_steps": 1,
        "guidance_scale": 1.5,
        "guidance_range": [1.0, 2.0],
    },
}


def floats(t):
    return t.detach().float().contiguous().cpu().numpy().astype(np.float32)


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


class Reference:
    def __init__(self):
        self.out = TINY / "reference"
        self.out.mkdir(parents=True, exist_ok=True)
        for f in self.out.iterdir():
            f.unlink()
        self.meta = {"files": {}}

    def put(self, key, tensor):
        array = floats(torch.as_tensor(tensor))
        file = f"{key}.bin"
        array.tofile(self.out / file)
        self.meta["files"][key] = {"file": file, "shape": list(array.shape)}

    def finish(self, **meta):
        self.meta.update(meta)
        (TINY / "reference.json").write_text(json.dumps(self.meta, indent=1) + "\n")


def randomise(model, g):
    state = model.state_dict()
    for name, value in state.items():
        if name in ("latent_scale", "latent_bias"):
            continue
        fan_in = int(np.prod(value.shape[1:])) if value.dim() > 1 else 1
        scale = 0.02 if name == "lm.embed_tokens.weight" else 0.1 if value.dim() <= 1 else fan_in**-0.5
        noise = torch.randn(value.shape, generator=g) * scale
        if name.endswith("norm.weight"):
            noise += 1.0
        if name.endswith("gamma"):
            noise = 0.2 + 0.2 * noise
        if name.endswith("A_log"):
            noise = torch.log(torch.rand(value.shape, generator=g) * 4 + 1)
        state[name] = noise.to(torch.bfloat16).float()
    state["latent_scale"] = torch.tensor(0.2).to(torch.bfloat16).float()
    state["latent_bias"] = torch.tensor(0.05).to(torch.bfloat16).float()
    model.load_state_dict(state)


def tone(samples, g, base):
    t = torch.arange(samples, dtype=torch.float64) / 24000.0
    f = base * (1 + 0.3 * torch.sin(2 * torch.pi * 3.0 * t))
    wave = 0.3 * torch.sin(2 * torch.pi * torch.cumsum(f, 0) / 24000.0) + 0.02 * torch.randn(samples, generator=g, dtype=torch.float64)
    return wave.float()


def layer_outputs(model, embeddings, cache):
    seen = []
    hooks = [layer.register_forward_hook(lambda m, i, o: seen.append(o[0] if isinstance(o, tuple) else o)) for layer in model.lm.layers]
    try:
        hidden = model(embeddings, cache)
    finally:
        for h in hooks:
            h.remove()
    return hidden, torch.stack([embeddings] + seen)


def stream_logits(model, hidden):
    ids = torch.tensor([model.config.pad_token_id, model.config.eop_token_id])
    logits = model.logits(hidden)
    logits[..., ids] = model.logits_fp32(hidden, ids)
    return logits


def build(args):
    sys.path.insert(0, str(Path(args.haru).resolve()))
    from haru.config import HaruConfig
    from haru.models.haru import HaruModel
    from haru.models.vibevoice_codec import CodecStream
    from haru.prompt import silence_latent

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    g = torch.Generator().manual_seed(SEED)
    config = HaruConfig.from_dict(CONFIG)
    model = HaruModel(config).eval()
    randomise(model, g)
    TINY.mkdir(parents=True, exist_ok=True)
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}, str(TINY / "model.safetensors"), metadata={"format": "pt"})
    (TINY / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    ref = Reference()
    d, pad, eop = config.latent_dim, config.pad_token_id, config.eop_token_id

    with torch.no_grad():
        clip = tone(REFERENCE_FRAMES * FRAME - 1234, g, 180.0)
        ref_mean = model.codec.encode(clip)[0]
        ref_latents = model.codec.sample(ref_mean[None], g)[0]
        card = torch.randint(0, 500, (CARD_TOKENS,), generator=g)
        n_ref = ref_latents.shape[0]
        tokens = torch.cat([card, torch.full((n_ref - 1,), pad), torch.tensor([eop])])
        n = tokens.shape[0]
        model_latents = torch.zeros(n, d)
        model_latents[CARD_TOKENS:] = ref_latents
        user_latents = torch.zeros(n, d)
        model_mask = torch.arange(n) >= CARD_TOKENS
        user_mask = torch.zeros(n, dtype=torch.bool)

        embeds = model.embed(tokens[None], model_latents[None], user_latents[None], model_mask[None], user_mask[None])
        whole_cache = model.new_cache()
        prompt_hidden, prompt_layers = layer_outputs(model, embeds, whole_cache)
        no_audio = torch.zeros(n, dtype=torch.bool)
        text_embeds = model.embed(tokens[None], model_latents[None], user_latents[None], no_audio[None], no_audio[None])
        text_hidden, text_layers = layer_outputs(model, text_embeds, model.new_cache())

        user_audio = tone((STEPS) * FRAME, g, 220.0).reshape(STEPS, FRAME)
        head_noise = torch.randn(STEPS + 1, d, generator=g)
        cache = model.new_cache()
        hidden = model(embeds[:, : n - 3], cache)
        hidden = torch.cat([hidden, model(embeds[:, n - 3 :], cache)], dim=1)
        chunked = float((hidden - prompt_hidden).abs().max())
        last = hidden[:, -1]
        stream = CodecStream(model.codec)
        frames = {k: [] for k in ("hidden", "logits", "token", "latent", "audio", "user_mean", "user_latent", "embeds")}

        def emit(h, i):
            logits = stream_logits(model, h)
            token = logits.argmax(dim=-1)
            latent = model.unscale_latents(model.head.sample(h, head_noise[i : i + 1], config.head.guidance_scale))
            audio = stream.decode(latent[None])[0]
            frames["hidden"].append(h[0])
            frames["logits"].append(logits[0])
            frames["token"].append(token.float())
            frames["latent"].append(latent[0])
            frames["audio"].append(audio)
            return token, latent

        token, latent = emit(last, 0)
        user_noise = torch.randn(STEPS, d, generator=g)
        user_scale = torch.randn(STEPS, generator=g)
        for i in range(STEPS):
            mean = stream.encode(user_audio[i : i + 1])[0]
            user_latent = mean + user_scale[i] * (config.codec.latent_std / 0.8) * user_noise[i : i + 1]
            e = model.embed(token[None], latent[None], user_latent[None])
            h = model(e, cache)[:, -1]
            frames["user_mean"].append(mean[0])
            frames["user_latent"].append(user_latent[0])
            frames["embeds"].append(e[0, 0])
            token, latent = emit(h, i + 1)

        session_hidden = torch.stack(frames["hidden"])
        guided = {}
        for w in (1.0, 1.5, 2.0):
            guided[w] = model.unscale_latents(model.head.sample(session_hidden, head_noise, w))
        steps2 = model.unscale_latents(model.head.sample(session_hidden, head_noise, 1.5, steps=2))

        whole_audio = model.codec.decode(torch.stack(frames["latent"])[None])[0]
        whole_mean = model.codec.encode(user_audio.reshape(-1))[0]
        stream_audio = torch.cat(frames["audio"])
        stream_mean = torch.stack(frames["user_mean"])

        enc_stream = CodecStream(model.codec)
        dec_stream = CodecStream(model.codec)
        clip_frames = clip.new_zeros(REFERENCE_FRAMES * FRAME)
        clip_frames[: clip.shape[0]] = clip
        enc_frames = torch.cat([enc_stream.encode(clip_frames[i * FRAME : (i + 1) * FRAME][None])[0] for i in range(REFERENCE_FRAMES)])
        dec_frames = torch.cat([dec_stream.decode(ref_mean[i : i + 1][None])[0] for i in range(REFERENCE_FRAMES)])
        state_floats = {
            "encoder": sum(s[0].numel() for s in enc_stream.encoder_state),
            "decoder": sum(s[0].numel() for s in dec_stream.decoder_state),
        }

        silence = silence_latent(model.codec)
        whole_clip_audio = model.codec.decode(ref_mean[None])[0]

    ref.put("prompt.tokens", tokens)
    ref.put("prompt.model_latents", model_latents)
    ref.put("prompt.user_latents", user_latents)
    ref.put("prompt.model_mask", model_mask)
    ref.put("prompt.user_mask", user_mask)
    ref.put("prompt.embeds", embeds[0])
    ref.put("prompt.layers", prompt_layers[:, 0])
    ref.put("prompt.hidden", prompt_hidden[0])
    ref.put("prompt.logits", stream_logits(model, prompt_hidden[0]))
    ref.put("text.layers", text_layers[:, 0])
    ref.put("text.hidden", text_hidden[0])
    ref.put("session.user_audio", user_audio)
    ref.put("session.user_mean", stream_mean)
    ref.put("session.user_noise", user_noise)
    ref.put("session.user_scale", user_scale)
    ref.put("session.user_latent", torch.stack(frames["user_latent"]))
    ref.put("session.embeds", torch.stack(frames["embeds"]))
    ref.put("session.head_noise", head_noise)
    ref.put("session.hidden", session_hidden)
    ref.put("session.logits", torch.stack(frames["logits"]))
    ref.put("session.token", torch.cat(frames["token"]))
    ref.put("session.latent", torch.stack(frames["latent"]))
    ref.put("session.audio", stream_audio)
    for w, v in guided.items():
        ref.put(f"head.guidance{w}", v)
    ref.put("head.steps2", steps2)
    ref.put("codec.clip", clip)
    ref.put("codec.clip_mean", ref_mean)
    ref.put("codec.clip_mean_stream", enc_frames)
    ref.put("codec.clip_audio", whole_clip_audio)
    ref.put("codec.silence", silence)
    checks = {
        "chunked_prefill_max_diff": chunked,
        "stream_decode_vs_whole": rel(stream_audio, whole_audio),
        "stream_encode_vs_whole": rel(stream_mean, whole_mean),
        "clip_encode_stream_vs_whole": rel(enc_frames[: REFERENCE_FRAMES - 1], ref_mean[: REFERENCE_FRAMES - 1]),
        "clip_decode_stream_vs_whole": rel(dec_frames, whole_clip_audio),
    }
    ref.finish(
        haru_commit=args.commit,
        torch=torch.__version__,
        seed=SEED,
        dtype="float32 on bf16 rounded weights",
        card_tokens=CARD_TOKENS,
        reference_frames=REFERENCE_FRAMES,
        prompt_length=int(n),
        steps=STEPS,
        guidance=config.head.guidance_scale,
        codec_state_floats=state_floats,
        checks=checks,
        params=sum(v.numel() for v in model.state_dict().values()),
    )
    for k, v in checks.items():
        print(f"{k}: {v:.3e}")
        if v > 1e-4:
            sys.exit(f"{k} too large")
    size = sum(f.stat().st_size for f in FIXTURE.rglob("*") if f.is_file())
    print(f"{FIXTURE}: {size} bytes, {checks}")
    if size > MAX_BYTES:
        sys.exit(f"over {MAX_BYTES} bytes")


def header(args):
    entries = json.loads(Path(args.header).read_text())
    entries.pop("__metadata__", None)
    rows = [[k, v["dtype"], v["shape"]] for k, v in sorted(entries.items())]
    (FIXTURE / "haru-4b.header.json").write_text("[" + ",\n".join(json.dumps(r) for r in rows) + "]\n")
    print(f"{len(rows)} tensors")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("fixture")
    b.add_argument("--haru", required=True)
    b.add_argument("--commit", required=True)
    h = sub.add_parser("header")
    h.add_argument("header")
    args = parser.parse_args()
    FIXTURE.mkdir(parents=True, exist_ok=True)
    {"fixture": build, "header": header}[args.cmd](args)


if __name__ == "__main__":
    main()
