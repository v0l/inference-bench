import argparse
import json
import math
import struct
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers.models.gradium_phonon_tts import GradiumPhononTTSForConditionalGeneration
from transformers.models.pocket_tts_audio_tokenizer.modeling_pocket_tts_audio_tokenizer import PocketTTSAudioTokenizerModel

import tiny_mimi

ROOT = Path(__file__).parent.parent
HUB = ROOT / "tests/fixtures/speech/phonon-31m-en.hub-92741c5.config.json"
TEXT = "The weather tomorrow will be sunny with a light breeze, and the trains to the city leave every twenty minutes from the station."
FRAMES = 50
TEMPERATURE = 0.3
SEED = 1484


def write_bin(path, t):
    path.write_bytes(struct.pack(f"<{t.numel()}f", *t.detach().float().reshape(-1).tolist()))


def model_dir(checkpoint, tmp):
    d = Path(tmp)
    (d / "config.json").write_text(HUB.read_text())
    for f in ["model.safetensors", "tokenizer.json"]:
        (d / f).symlink_to(Path(checkpoint) / f)
    return d


def mimi_model(checkpoint):
    m = json.loads(HUB.read_text())["mimi"]
    w = {k: v.float() for k, v in load_file(str(Path(checkpoint) / "model.safetensors")).items() if k.startswith("mimi.")}
    model = PocketTTSAudioTokenizerModel(tiny_mimi.reference_config(m)).eval()
    missing, unexpected = model.load_state_dict(tiny_mimi.reference_state(w, m), strict=False)
    missing = [k for k in missing if not k.endswith((".stride", ".kernel_size", ".padding_total"))]
    assert not missing and not unexpected, (missing, unexpected)
    return model


def main():
    p = argparse.ArgumentParser(description="real Phonon reference: transformers, CPU f32, every step fed the previous reference latent and recorded noise")
    p.add_argument("checkpoint")
    p.add_argument("voice")
    p.add_argument("out")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(Path(args.checkpoint) / "tokenizer.json"))
    ids = tok.encode(TEXT, add_special_tokens=False).ids
    voice = load_file(args.voice)["speaker_wavs"].float()[0].T
    g = torch.Generator().manual_seed(SEED)
    noise = torch.randn(FRAMES, 32, generator=g) * math.sqrt(TEMPERATURE)
    with tempfile.TemporaryDirectory() as tmp:
        model = GradiumPhononTTSForConditionalGeneration.from_pretrained(str(model_dir(args.checkpoint, tmp)), dtype=torch.float32, attn_implementation="eager").eval()
        with torch.no_grad():
            prompt = torch.tensor([[5] * voice.shape[0] + ids + [4]])
            o = model(input_ids=prompt, audio_prompt_latents=voice[None], use_cache=True)
            cache = o.past_key_values
            prefill = o.last_hidden_state[0]
            hidden, eos, latents = [], [], []
            for f in range(FRAMES):
                cond = o.last_hidden_state[:, -1]
                hidden.append(cond[0])
                eos.append(o.logits[0, -1, 0])
                z = model.sample_latent(cond, noise[f : f + 1], 1)
                latents.append(z[0])
                o = model(input_ids=torch.tensor([[6]]), audio_latents=z[:, None], past_key_values=cache, use_cache=True)
                cache = o.past_key_values
            hidden, eos, latents = torch.stack(hidden), torch.stack(eos), torch.stack(latents)
            codec = model.denormalize_latents(latents)
            conditions = model.model.get_condition_embeds(torch.device("cpu"))
        audio = mimi_model(args.checkpoint).decode(codec[None]).audio_values[0, 0]
    files = {}
    for key, t in [("noise", noise), ("prefill.last", prefill[-1]), ("frames.hidden", hidden), ("frames.eos", eos), ("frames.latent", latents), ("frames.codec_latent", codec), ("audio", audio), ("conditions", conditions)]:
        name = f"{key}.bin"
        write_bin(out / name, t)
        files[key] = {"file": name, "shape": list(t.shape)}
    meta = {
        "source": "transformers GradiumPhononTTSForConditionalGeneration, gradium/phonon-31m-en model.safetensors (8c64b55e, identical at 92741c5b) with the 92741c5b config, f32, eager, CPU; each step fed the previous reference latent; audio from PocketTTSAudioTokenizerModel.decode of the codec latents",
        "text": TEXT,
        "tokens": ids,
        "prompt": prompt[0].tolist(),
        "voice": str(args.voice),
        "seed": SEED,
        "temperature": TEMPERATURE,
        "frames": FRAMES,
        "files": files,
    }
    (out / "reference.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"{out}: {len(ids)} tokens, {FRAMES} frames, eos logits max {eos.max():.2f}, max |audio| {audio.abs().max():.3f}")


if __name__ == "__main__":
    main()
