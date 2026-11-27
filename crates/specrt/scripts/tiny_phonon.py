import argparse
import json
import math
import struct
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders
from transformers.models.gradium_phonon_tts import GradiumPhononTTSForConditionalGeneration
from transformers.models.pocket_tts_audio_tokenizer.modeling_pocket_tts_audio_tokenizer import PocketTTSAudioTokenizerModel

import tiny_mimi

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests/fixtures/speech"
TINY = FIXTURE / "tiny-phonon"
MIMI = FIXTURE / "tiny-mimi"
SEED = 1
HUB = FIXTURE / "phonon-31m-en.hub-92741c5.config.json"
TEXT = "Hello there, how are you? I am fine."
VOICE_FRAMES = 5
FRAMES = 10
TEMPERATURE = 0.3
FRAMES_AFTER_EOS = 2

D, HEADS, LAYERS, FF = 32, 2, 2, 64
PROMPT_DIM, LATENT = 16, 8
FLOW_DIM, FLOW_DEPTH, FREQ = 48, 2, 16
SPEAKERS_VOCAB, SPEAKERS_DIM, DURATION_DIM = 4, 4, 8

CTL = ["\u0000<unk>", "\u0000<s>", "\u0000</s>", "\u0000<pad>", "<|audio_bos|>", "<|audio_prompt|>", "<|audio_latent|>"]
PIECES = ["\u2581", ".", ",", "?", "!", "...", "'", "\u2581I", "\u2581am", "\u2581are", "\u2581you", "\u2581how", "\u2581fine", "\u2581Hello", "\u2581there", "\u2581Ok", "\u2581one", "\u2581two"]
LETTERS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
VOCAB = len(CTL) + len(PIECES) + len(LETTERS)


def tokenizer():
    vocab = [(p, 0.0) for p in CTL] + [(p, -2.0) for p in PIECES] + [(p, -5.0) for p in LETTERS]
    tok = Tokenizer(models.Unigram(vocab, unk_id=0, byte_fallback=False))
    tok.normalizer = normalizers.Prepend("\u2581")
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="\u2581", prepend_scheme="always", split=True)
    tok.decoder = decoders.Sequence([decoders.Metaspace(replacement="\u2581", prepend_scheme="always", split=True), decoders.Fuse()])
    return tok


def config(eos_threshold):
    c = json.loads(HUB.read_text())
    mimi = json.loads((MIMI / "config.json").read_text())["mimi"]
    c.update(
        hidden_size=D, num_attention_heads=HEADS, num_key_value_heads=HEADS, head_dim=D // HEADS, intermediate_size=FF, num_hidden_layers=LAYERS,
        vocab_size=VOCAB, audio_prompt_dim=PROMPT_DIM, latent_dim=LATENT, num_speakers=2, num_speakers_vocab_size=SPEAKERS_VOCAB,
        num_speakers_embedding_dim=SPEAKERS_DIM, duration_delta=0.0013, duration_delta_embedding_dim=DURATION_DIM, eos_threshold=eos_threshold,
        max_position_embeddings=64, dtype="bfloat16",
    )
    c["flow_head_config"].update(hidden_size=FLOW_DIM, num_hidden_layers=FLOW_DEPTH, latent_dim=LATENT, condition_dim=D, frequency_embedding_size=FREQ)
    c["flow_lm"].update(d_model=D, num_heads=HEADS, num_layers=LAYERS, dim_feedforward=FF, n_bins=VOCAB - 1, lut_dim=D, flow_dim=FLOW_DIM, flow_depth=FLOW_DEPTH, ldim=LATENT)
    c["mimi"] = mimi
    c["conditioners"][0]["lut"].update(n_bins=SPEAKERS_VOCAB - 1, dim=SPEAKERS_DIM)
    c["conditioners"][1]["continuous"].update(dim=DURATION_DIM)
    c.pop("speaker_mimi")
    return c


def bf16(t):
    return t.to(torch.bfloat16)


def checkpoint(g):
    r = lambda *s, std=0.3: (torch.randn(*s, generator=g) * std).float()
    w = {}
    conds = "flow_lm.condition_provider.conditioners."
    w["flow_lm.bos_emb"] = r(LATENT, std=0.5)
    for n in ["duration_delta", "num_speakers", "speaker_wavs", "transcript_in_segment"]:
        w[conds + n + ".learnt_padding"] = r(1, 1, D, std=0.1)
    w[conds + "duration_delta.output_proj.weight"] = r(D, DURATION_DIM, std=0.3)
    w[conds + "num_speakers.embed.weight"] = r(SPEAKERS_VOCAB, SPEAKERS_DIM, std=0.6)
    w[conds + "num_speakers.output_proj.weight"] = r(D, SPEAKERS_DIM, std=0.4)
    w[conds + "speaker_wavs.output_proj.weight"] = r(D, PROMPT_DIM, std=0.3)
    w[conds + "transcript_in_segment.embed.weight"] = r(VOCAB, D, std=0.6)
    w["flow_lm.emb_mean"] = r(LATENT, std=0.5)
    w["flow_lm.emb_std"] = 1.0 + r(LATENT, std=0.2)
    for i, (o, k) in zip([0, 3, 6, 9], [(32, 2), (32, 32), (32, 32), (1, 32)]):
        w[f"flow_lm.flow.w_s_t.{i}.weight"] = r(o, k, std=0.1)
        w[f"flow_lm.flow.w_s_t.{i}.bias"] = r(o, std=0.1)
    f = "flow_lm.flow_net."
    w[f + "cond_embed.weight"] = r(FLOW_DIM, D, std=0.25)
    w[f + "cond_embed.bias"] = r(FLOW_DIM, std=0.1)
    w[f + "input_proj.weight"] = r(FLOW_DIM, LATENT, std=0.4)
    w[f + "input_proj.bias"] = r(FLOW_DIM, std=0.1)
    w[f + "final_layer.adaLN_modulation.1.weight"] = r(2 * FLOW_DIM, FLOW_DIM, std=0.2)
    w[f + "final_layer.adaLN_modulation.1.bias"] = r(2 * FLOW_DIM, std=0.1)
    w[f + "final_layer.linear.weight"] = r(LATENT, FLOW_DIM, std=0.25)
    w[f + "final_layer.linear.bias"] = r(LATENT, std=0.1)
    for b in range(FLOW_DEPTH):
        p = f"{f}res_blocks.{b}."
        w[p + "adaLN_modulation.1.weight"] = r(3 * FLOW_DIM, FLOW_DIM, std=0.2)
        w[p + "adaLN_modulation.1.bias"] = r(3 * FLOW_DIM, std=0.1)
        w[p + "in_ln.weight"] = 1.0 + r(FLOW_DIM, std=0.1)
        w[p + "in_ln.bias"] = r(FLOW_DIM, std=0.1)
        w[p + "mlp.0.weight"] = r(FLOW_DIM, FLOW_DIM, std=0.25)
        w[p + "mlp.0.bias"] = r(FLOW_DIM, std=0.1)
        w[p + "mlp.2.weight"] = r(FLOW_DIM, FLOW_DIM, std=0.25)
        w[p + "mlp.2.bias"] = r(FLOW_DIM, std=0.1)
    half = FREQ // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    for t in range(2):
        p = f"{f}time_embed.{t}."
        w[p + "freqs"] = freqs.clone()
        w[p + "mlp.0.weight"] = r(FLOW_DIM, FREQ, std=0.3)
        w[p + "mlp.0.bias"] = r(FLOW_DIM, std=0.1)
        w[p + "mlp.2.weight"] = r(FLOW_DIM, FLOW_DIM, std=0.25)
        w[p + "mlp.2.bias"] = r(FLOW_DIM, std=0.1)
        w[p + "mlp.3.alpha"] = 1.0 + r(FLOW_DIM, std=0.2)
    w["flow_lm.input_linear.weight"] = r(D, LATENT, std=0.4)
    w["flow_lm.out_eos.weight"] = r(1, D, std=0.3)
    w["flow_lm.out_eos.bias"] = r(1, std=0.1)
    w["flow_lm.out_norm.weight"] = 1.0 + r(D, std=0.1)
    w["flow_lm.out_norm.bias"] = r(D, std=0.1)
    for l in range(LAYERS):
        p = f"flow_lm.transformer.layers.{l}."
        w[p + "linear1.weight"] = r(FF, D, std=0.25)
        w[p + "linear2.weight"] = r(D, FF, std=0.2)
        w[p + "norm1.weight"] = 1.0 + r(D, std=0.1)
        w[p + "norm1.bias"] = r(D, std=0.1)
        w[p + "norm2.weight"] = 1.0 + r(D, std=0.1)
        w[p + "norm2.bias"] = r(D, std=0.1)
        w[p + "self_attn.in_proj.weight"] = r(3 * D, D, std=0.35)
        w[p + "self_attn.out_proj.weight"] = r(D, D, std=0.25)
    return {k: bf16(v) for k, v in w.items()}


def write_bin(path, t):
    path.write_bytes(struct.pack(f"<{t.numel()}f", *t.detach().float().reshape(-1).tolist()))


def mimi_model():
    model = PocketTTSAudioTokenizerModel(tiny_mimi.reference_config()).eval()
    missing, unexpected = model.load_state_dict(tiny_mimi.reference_state(load_file(str(MIMI / "model.safetensors"))), strict=False)
    missing = [k for k in missing if not k.endswith((".stride", ".kernel_size", ".padding_total"))]
    assert not missing and not unexpected, (missing, unexpected)
    return model


def frames(model, voice, ids, noise, steps=1):
    m = model.model
    prompt = torch.tensor([[5] * voice.shape[0] + ids + [4]])
    out = model(input_ids=prompt, audio_prompt_latents=voice[None], use_cache=True)
    cache = out.past_key_values
    prefill = out.last_hidden_state[0]
    hidden, eos, latents = [], [], []
    for f in range(noise.shape[0]):
        cond = out.last_hidden_state[:, -1]
        hidden.append(cond[0])
        eos.append(out.logits[0, -1, 0])
        z = model.sample_latent(cond, noise[f : f + 1], steps)
        latents.append(z[0])
        out = model(input_ids=torch.tensor([[6]]), audio_latents=z[:, None], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
    return prefill, torch.stack(hidden), torch.stack(eos), torch.stack(latents)


def eos_frame(eos):
    late = [k for k in range(2, FRAMES - FRAMES_AFTER_EOS) if eos[k] > eos[:k].max()]
    return late[0] if late else None


def fixture():
    seed = SEED
    while True:
        g = torch.Generator().manual_seed(seed)
        w = checkpoint(g)
        if write_and_check(seed, g, w):
            return
        seed += 1


def write_and_check(seed, g, w):
    torch.manual_seed(seed)
    tok = tokenizer()
    ids = tok.encode(TEXT).ids
    assert 0 not in ids, tok.encode(TEXT).tokens
    voice = torch.randn(1, PROMPT_DIM, VOICE_FRAMES, generator=g)
    unit = torch.randn(FRAMES, LATENT, generator=g)
    noise = unit * math.sqrt(TEMPERATURE)
    TINY.mkdir(parents=True, exist_ok=True)
    (TINY / "reference").mkdir(exist_ok=True)
    save_file({k: v.contiguous() for k, v in sorted(w.items())}, str(TINY / "model.safetensors"))
    save_file({"speaker_wavs": voice.contiguous()}, str(TINY / "voice.safetensors"))
    tok.save(str(TINY / "tokenizer.json"))
    (TINY / "config.json").write_text(json.dumps(config(0.0), indent=2, sort_keys=True) + "\n")
    model = GradiumPhononTTSForConditionalGeneration.from_pretrained(str(TINY), dtype=torch.float32, attn_implementation="eager").eval()
    with torch.no_grad():
        prefill, hidden, eos, latents = frames(model, voice[0].T, ids, noise)
        _, _, _, steps2 = frames(model, voice[0].T, ids, noise[:3], steps=2)
        k = eos_frame(eos)
        if k is None:
            return False
        threshold = float((eos[k] + eos[:k].max()) / 2)
        (TINY / "config.json").write_text(json.dumps(config(threshold), indent=2, sort_keys=True) + "\n")
        model.config.eos_threshold = threshold
        real_randn = torch.randn
        drawn = iter(unit)
        torch.randn = lambda *a, **kw: next(drawn)[None].clone()
        try:
            gen = model.generate(
                input_ids=torch.tensor([[5] * VOICE_FRAMES + ids + [4]]), audio_prompt_latents=voice[0].T[None], do_sample=True, temperature=TEMPERATURE,
                eos_threshold=threshold, frames_after_eos=FRAMES_AFTER_EOS, min_frames_before_eos=0, num_flow_steps=1, max_new_tokens=FRAMES, return_dict_in_generate=True, output_scores=True,
            )
        finally:
            torch.randn = real_randn
        count = int(gen.audio_latents_mask.sum())
        assert count == k + 1 + FRAMES_AFTER_EOS, (count, k)
        denorm = model.denormalize_latents(latents[:count])
        assert torch.allclose(gen.audio_latents[0, :count], denorm, atol=1e-6), (gen.audio_latents[0, :count] - denorm).abs().max()
        audio = mimi_model().decode(denorm[None]).audio_values[0, 0]
        conditions = model.model.get_condition_embeds(torch.device("cpu"))
    files = {}
    for key, t in [
        ("voice", voice[0].T), ("noise", noise), ("prefill.hidden", prefill), ("frames.hidden", hidden), ("frames.eos", eos), ("frames.latent", latents),
        ("frames.steps2", steps2), ("frames.codec_latent", denorm), ("audio", audio), ("conditions", conditions),
    ]:
        name = f"{key}.bin"
        write_bin(TINY / "reference" / name, t)
        files[key] = {"file": name, "shape": list(t.shape)}
    meta = {
        "source": "transformers.models.gradium_phonon_tts GradiumPhononTTSForConditionalGeneration (from_pretrained on this folder, f32, eager), generate with the recorded noise; audio from tests/fixtures/speech/tiny-mimi through PocketTTSAudioTokenizerModel.decode",
        "seed": seed,
        "text": TEXT,
        "tokens": ids,
        "prompt": [5] * VOICE_FRAMES + ids + [4],
        "temperature": TEMPERATURE,
        "frames_after_eos": FRAMES_AFTER_EOS,
        "eos_frame": k,
        "generated_frames": count,
        "eos_threshold": threshold,
        "files": files,
    }
    (TINY / "reference.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    total = sum(p.stat().st_size for p in TINY.rglob("*") if p.is_file())
    print(f"{TINY}: {total} bytes, seed {seed}, tokens {ids}, eos at frame {k}, {count} frames, max |audio| {audio.abs().max():.3f}")
    return True


if __name__ == "__main__":
    argparse.ArgumentParser(description="tiny Gradium Phonon TTS fixture").parse_args()
    fixture()
