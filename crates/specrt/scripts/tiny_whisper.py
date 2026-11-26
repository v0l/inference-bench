import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration
from transformers.generation import GenerationConfig

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests/fixtures/speech"
TINY = FIXTURE / "tiny-whisper"
MAX_BYTES = 2 * 1024 * 1024
SEED = 0
RATE = 16000
WINDOW_SECONDS = 1
STEPS = 12

TEXT = 512
EOT = TEXT
SOT = TEXT + 1
LANGS = ["en", "de", "fr", "es"]
LANG = {f"<|{l}|>": SOT + 1 + i for i, l in enumerate(LANGS)}
TRANSLATE = SOT + 1 + len(LANGS)
TRANSCRIBE = TRANSLATE + 1
START_LM = TRANSCRIBE + 1
PREV = START_LM + 1
NO_SPEECH = PREV + 1
NO_TIMESTAMPS = NO_SPEECH + 1
TIMESTAMPS = 64
VOCAB = NO_TIMESTAMPS + 1 + TIMESTAMPS

CONFIG = dict(
    vocab_size=VOCAB,
    num_mel_bins=80,
    d_model=64,
    encoder_layers=2,
    decoder_layers=2,
    encoder_attention_heads=2,
    decoder_attention_heads=2,
    encoder_ffn_dim=128,
    decoder_ffn_dim=128,
    max_source_positions=WINDOW_SECONDS * RATE // 160 // 2,
    max_target_positions=64,
    activation_function="gelu",
    scale_embedding=False,
    pad_token_id=EOT,
    bos_token_id=EOT,
    eos_token_id=EOT,
    decoder_start_token_id=SOT,
    begin_suppress_tokens=[220 % TEXT, EOT],
    suppress_tokens=[1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, TRANSLATE, START_LM, PREV, NO_SPEECH],
    init_std=0.2,
    tie_word_embeddings=True,
)

REAL = {
    "whisper-tiny.en": "openai/whisper-tiny.en",
    "whisper-base": "openai/whisper-base",
    "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
}


def generation_config():
    return {
        "decoder_start_token_id": SOT,
        "bos_token_id": EOT,
        "eos_token_id": EOT,
        "pad_token_id": EOT,
        "is_multilingual": True,
        "lang_to_id": LANG,
        "task_to_id": {"translate": TRANSLATE, "transcribe": TRANSCRIBE},
        "no_timestamps_token_id": NO_TIMESTAMPS,
        "prev_sot_token_id": PREV,
        "max_initial_timestamp_index": 50,
        "begin_suppress_tokens": CONFIG["begin_suppress_tokens"],
        "suppress_tokens": CONFIG["suppress_tokens"],
        "forced_decoder_ids": [[1, None], [2, TRANSCRIBE]],
        "max_length": CONFIG["max_target_positions"],
        "alignment_heads": [[1, 0], [1, 1]],
    }


def audio():
    rng = np.random.default_rng(SEED)
    t = np.arange(WINDOW_SECONDS * RATE) / RATE
    chirp = 0.3 * np.sin(2 * math.pi * (200 + 600 * t) * t)
    voiced = np.where((t > 0.1) & (t < 0.7), chirp, 0.0)
    full = (voiced + 0.01 * rng.standard_normal(t.shape)).astype(np.float32)
    short = full[: int(0.6 * RATE)].copy()
    return full, short


def save_ref(store, name, array):
    a = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    store[name] = a


def build():
    torch.manual_seed(SEED)
    cfg = WhisperConfig(**CONFIG)
    model = WhisperForConditionalGeneration(cfg).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=CONFIG["init_std"]) if p.dim() > 1 else torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        for ln in [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]:
            ln.weight.add_(1.0)
    model.generation_config = GenerationConfig(**generation_config())
    return model


def features(fe, pcm):
    return fe(pcm, sampling_rate=RATE, return_tensors="pt").input_features


def greedy(model, encoder_out, prefix, gen):
    tokens = list(prefix)
    rows = []
    with torch.no_grad():
        for step in range(STEPS):
            out = model(encoder_outputs=(encoder_out,), decoder_input_ids=torch.tensor([tokens]))
            logits = out.logits[0, -1].clone()
            rows.append(logits.numpy().copy())
            logits[gen["suppress_tokens"]] = -math.inf
            if step == 0:
                logits[gen["begin_suppress_tokens"]] = -math.inf
            nxt = int(logits.argmax())
            tokens.append(nxt)
            if nxt == EOT:
                break
    return tokens[len(prefix):], np.stack(rows)


def fixture():
    TINY.mkdir(parents=True, exist_ok=True)
    model = build()
    state = {k: v.contiguous() for k, v in model.state_dict().items() if k != "proj_out.weight"}
    save_file(state, TINY / "model.safetensors", metadata={"format": "pt"})
    config = model.config.to_dict()
    config["architectures"] = ["WhisperForConditionalGeneration"]
    config.pop("transformers_version", None)
    (TINY / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (TINY / "generation_config.json").write_text(json.dumps(generation_config(), indent=2, sort_keys=True) + "\n")
    fe = WhisperFeatureExtractor(feature_size=80, sampling_rate=RATE, hop_length=160, chunk_length=WINDOW_SECONDS, n_fft=400)
    (TINY / "preprocessor_config.json").write_text(fe.to_json_string())


def reference():
    model = WhisperForConditionalGeneration.from_pretrained(TINY, torch_dtype=torch.float32).eval()
    gen = generation_config()
    fe = WhisperFeatureExtractor.from_pretrained(TINY)
    full, short = audio()
    refs = {}
    save_ref(refs, "audio.full", full)
    save_ref(refs, "audio.short", short)
    save_ref(refs, "mel.filters", fe.mel_filters)
    mel = features(fe, full)
    mel_short = features(fe, short)
    save_ref(refs, "mel.full", mel[0])
    save_ref(refs, "mel.short", mel_short[0])
    with torch.no_grad():
        enc = model.model.encoder
        stem = torch.nn.functional.gelu(enc.conv2(torch.nn.functional.gelu(enc.conv1(mel))))
        save_ref(refs, "encoder.stem", stem[0].transpose(0, 1))
        enc_out = enc(mel).last_hidden_state
        save_ref(refs, "encoder.out", enc_out[0])
        lang = model(encoder_outputs=(enc_out,), decoder_input_ids=torch.tensor([[SOT]])).logits[0, -1]
        save_ref(refs, "language.logits", lang)
        prefix = [SOT, LANG["<|en|>"], TRANSCRIBE, NO_TIMESTAMPS]
        prompt = model(encoder_outputs=(enc_out,), decoder_input_ids=torch.tensor([prefix])).logits[0]
        save_ref(refs, "prefix.logits", prompt)
    tokens, rows = greedy(model, enc_out, prefix, gen)
    save_ref(refs, "greedy.logits", rows)
    lang_ids = [v for v in LANG.values()]
    detected = lang_ids[int(np.argmax(refs["language.logits"][lang_ids]))]
    with torch.no_grad():
        hf = model.generate(mel, language="en", task="transcribe", return_timestamps=False, max_new_tokens=STEPS)
    hf_tokens = [int(t) for t in hf[0]]
    meta = {
        "seed": SEED,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "window_samples": WINDOW_SECONDS * RATE,
        "prefix": prefix,
        "greedy": tokens,
        "generate": hf_tokens,
        "detected_language": detected,
        "files": {},
    }
    out = TINY / "reference"
    out.mkdir(exist_ok=True)
    for name, a in refs.items():
        (out / f"{name}.bin").write_bytes(a.astype("<f4").tobytes())
        meta["files"][name] = {"file": f"{name}.bin", "shape": list(a.shape)}
    (TINY / "reference.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return sorted([k, v["dtype"], v["shape"]] for k, v in h.items())


def fetch(repo, f):
    import urllib.request

    with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/{f}") as r:
        return r.read().decode()


def remote_header(repo):
    import urllib.request

    url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-7"})) as r:
        n = struct.unpack("<Q", r.read())[0]
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes=8-{7 + n}"})) as r:
        h = json.loads(r.read())
    h.pop("__metadata__", None)
    return sorted([k, v["dtype"], v["shape"]] for k, v in h.items())


def vibevoice():
    repo = "microsoft/VibeVoice-Realtime-0.5B"
    (FIXTURE / "vibevoice-realtime-0.5b.config.json").write_text(fetch(repo, "config.json"))
    rows = [r for r in remote_header(repo) if not r[0].startswith("model.acoustic_tokenizer.") or r[0].startswith("model.acoustic_tokenizer.decoder.")]
    (FIXTURE / "vibevoice-realtime-0.5b.header.json").write_text(json.dumps(rows) + "\n")


def real(hub, phonon):
    for name, repo in REAL.items():
        snap = next((Path(hub) / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*"), None)
        for f in ["config.json", "generation_config.json"]:
            text = (snap / f).read_text() if snap is not None and (snap / f).exists() else fetch(repo, f)
            (FIXTURE / f"{name}.{f}").write_text(text)
        if name == "whisper-tiny.en":
            (FIXTURE / f"{name}.header.json").write_text(json.dumps(header(snap / "model.safetensors")) + "\n")
    if phonon:
        p = Path(phonon)
        (FIXTURE / "phonon-31m-en.config.json").write_text((p / "config.json").read_text())
        (FIXTURE / "phonon-31m-en.header.json").write_text(json.dumps(header(p / "model.safetensors")) + "\n")
        voice = next((p / "voices").glob("*.safetensors"))
        (FIXTURE / "phonon-voice.header.json").write_text(json.dumps(header(voice)) + "\n")
        snap = next((Path(hub) / "models--gradium--phonon-31m-en" / "snapshots").glob("92741c5*"), None)
        text = (snap / "config.json").read_text() if snap is not None else fetch("gradium/phonon-31m-en", "config.json")
        (FIXTURE / "phonon-31m-en.hub-92741c5.config.json").write_text(text)
    vibevoice()


def size():
    total = sum(p.stat().st_size for p in FIXTURE.rglob("*") if p.is_file())
    print(f"{FIXTURE}: {total} bytes")
    if total > MAX_BYTES:
        raise SystemExit(f"over the {MAX_BYTES} byte cap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["all", "fixture", "reference", "real"])
    ap.add_argument("--hub", default=str(Path.home() / ".cache/huggingface/hub"))
    ap.add_argument("--phonon", help="a gradium/phonon-31m-en checkout at revision 8c64b55e with voices/")
    a = ap.parse_args()
    FIXTURE.mkdir(parents=True, exist_ok=True)
    if a.what in ("all", "fixture"):
        fixture()
    if a.what in ("all", "reference"):
        reference()
    if a.what in ("all", "real"):
        real(a.hub, a.phonon)
    size()


if __name__ == "__main__":
    main()
