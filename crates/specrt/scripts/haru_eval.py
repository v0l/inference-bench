import json
import sys
import wave
from pathlib import Path

import numpy as np


def read(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def write(path, audio):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def build_user(out_dir):
    out_dir = Path(out_dir)
    gap = np.zeros(int(3.0 * 24000), dtype=np.float32)
    parts = [np.zeros(int(1.0 * 24000), dtype=np.float32)]
    for name in ("u1", "u2", "u3"):
        parts += [read(out_dir / "user" / f"{name}.wav"), gap]
    clip = np.concatenate(parts)
    write(out_dir / "user.wav", clip)
    print(f"user clip {len(clip) / 24000:.1f} s")


def envelope(audio, window=2400):
    n = len(audio) // window
    return np.sqrt((audio[: n * window].reshape(n, window) ** 2).mean(axis=1) + 1e-12)


def compare(a_path, b_path, user_path):
    a, b, user = read(a_path), read(b_path), read(user_path)
    n = min(len(a), len(b), len(user))
    ea, eb, eu = envelope(a[:n]), envelope(b[:n]), envelope(user[:n])
    corr = float(np.corrcoef(ea, eb)[0, 1])
    db = lambda e: 20 * np.log10(e + 1e-9)
    voiced = lambda e: db(e) > -40
    va, vb, vu = voiced(ea), voiced(eb), voiced(eu)
    out = {
        "seconds": n / 24000,
        "rms_envelope_corr": corr,
        "rms_db": [float(db(np.sqrt((a[:n] ** 2).mean()))), float(db(np.sqrt((b[:n] ** 2).mean())))],
        "voiced_fraction": [float(va.mean()), float(vb.mean())],
        "overlap_with_user": [float((va & vu).mean()), float((vb & vu).mean())],
        "voiced_agreement": float((va == vb).mean()),
    }
    print(json.dumps(out, indent=2))


def whisper():
    import torch
    from transformers import pipeline

    return pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", torch_dtype=torch.float16, device="cuda")


def transcribe(paths):
    asr = whisper()
    for p in paths:
        text = asr(read(p), generate_kwargs={"language": "en"}, return_timestamps=True)["text"].strip()
        print(json.dumps({"wav": str(p), "text": text}))


def say_wer(report_path):
    import jiwer

    report = json.loads(Path(report_path).read_text())
    asr = whisper()
    norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
    refs, hyps = [], []
    for r in report:
        text = asr(read(r["wav"]), generate_kwargs={"language": "en"}, return_timestamps=True)["text"].strip()
        r["whisper"] = text
        r["wer"] = jiwer.wer(r["text"], text, reference_transform=norm, hypothesis_transform=norm)
        refs.append(r["text"])
        hyps.append(text)
        print(f"{r['index']:02d} wer {r['wer']:.2f} | {r['text']} | {text}")
    total = jiwer.wer(refs, hyps, reference_transform=norm, hypothesis_transform=norm)
    print(json.dumps({"sentences": len(refs), "wer": total, "median": float(np.median([r["wer"] for r in report]))}))
    Path(report_path).with_name("say_wer.json").write_text(json.dumps(report, indent=2))


def pairs(paths):
    import jiwer

    asr = whisper()
    norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
    for pair in paths:
        a, b = pair.split(",")
        ta, tb = (asr(read(p), generate_kwargs={"language": "en"}, return_timestamps=True)["text"].strip() for p in (a, b))
        va, vb = (float((20 * np.log10(envelope(read(p)) + 1e-9) > -40).mean()) for p in (a, b))
        wer = jiwer.wer(tb, ta, reference_transform=norm, hypothesis_transform=norm) if tb.strip() else float(bool(ta.strip()))
        print(json.dumps({"engine": a, "reference": b, "engine_text": ta, "reference_text": tb, "wer": wer, "voiced": [va, vb]}))


if __name__ == "__main__":
    commands = {
        "pairs": lambda: pairs(sys.argv[2:]),
        "build": lambda: build_user(sys.argv[2]),
        "compare": lambda: compare(sys.argv[2], sys.argv[3], sys.argv[4]),
        "transcribe": lambda: transcribe(sys.argv[2:]),
        "say-wer": lambda: say_wer(sys.argv[2]),
    }
    commands[sys.argv[1]]()
