import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import whisper_wer

ROOT = Path(__file__).parent.parent
SAY = ROOT / "tests/fixtures/haru/say_sentences.txt"
CLIPS = ROOT / "tests/fixtures/speech/librispeech-clips.json"
VOICES = {"Harper": "4SZHfMpw-p46Ywgs", "Hadley": "bXwrkHhURL1VFJdG"}
LIBRI = 20
WORDS = (8, 30)
RATE = 16000


def sentence_case(text):
    words = text.lower().split()
    words = ["I" if w == "i" else "I'" + w[2:] if w.startswith("i'") else w for w in words]
    s = " ".join(words)
    return s[:1].upper() + s[1:] + "."


def librispeech(manifest):
    fit = [(r["id"], sentence_case(r["text"])) for r in whisper_wer.load_manifest(manifest) if len(r["parts"]) == 1 and WORDS[0] <= len(r["text"].split()) <= WORDS[1]]
    return [fit[i * len(fit) // LIBRI] for i in range(LIBRI)]


def sentences(args):
    texts = [(f"say-{i:02d}", line.strip()) for i, line in enumerate(SAY.read_text().splitlines()) if line.strip()]
    texts += [(f"ls-{uid}", t) for uid, t in librispeech(args.manifest)]
    with open(args.out, "w") as f:
        for name, vid in VOICES.items():
            for sid, text in texts:
                f.write(json.dumps({"id": f"{name}-{sid}", "text": text, "voice": str(Path(args.voices) / f"{vid}.safetensors"), "voice_name": name}) + "\n")
    print(f"{args.out}: {len(texts)} sentences x {len(VOICES)} voices")


def candle(args):
    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    done = set()
    rows = out / "candle.jsonl"
    if rows.exists():
        done = {json.loads(l)["id"] for l in open(rows)}
    with open(rows, "a") as f:
        for r in whisper_wer.load_manifest(args.sentences):
            if r["id"] in done:
                continue
            start = time.time()
            text = subprocess.run(args.prefix.split() + [args.say, args.model, r["voice"], r["text"]], check=True, capture_output=True, text=True, env={**os.environ, "DEVICE": "cuda"}).stdout
            wav = out / "wav" / f"{r['id']}.wav"
            shutil.copy("/tmp/phonon-say.wav", wav)
            samples = (wav.stat().st_size - 44) // 2
            f.write(json.dumps({"id": r["id"], "wav": str(wav), "samples": samples, "audio_seconds": samples / 24000, "seconds": round(time.time() - start, 3), "log": text.strip().splitlines()[-1]}) + "\n")
            f.flush()
            print(r["id"], f"{samples / 24000:.2f}s", flush=True)


def to16k(args):
    out = Path(args.out)
    (out / "wav16").mkdir(parents=True, exist_ok=True)
    with open(out / "manifest16.jsonl", "w") as f:
        for r in whisper_wer.load_manifest(args.rows):
            dst = out / "wav16" / f"{r['id']}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", r["wav"], "-ar", str(RATE), "-ac", "1", "-c:a", "pcm_s16le", str(dst)], check=True)
            f.write(json.dumps({"id": r["id"], "wav": str(dst)}) + "\n")


def score(args):
    from transformers import WhisperTokenizer
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    tok = WhisperTokenizer.from_pretrained(args.whisper)
    norm = EnglishTextNormalizer(json.load(open(Path(args.whisper) / "normalizer.json")))
    truth = {r["id"]: r for r in whisper_wer.load_manifest(args.sentences)}
    runs = {}
    for spec in args.runs:
        name, rows, heard = spec.split("=", 1)[0], *spec.split("=", 1)[1].split(",")
        audio = {r["id"]: r for r in whisper_wer.load_manifest(rows)}
        text = {r["id"]: r["text"] if "text" in r else tok.decode(r["tokens"], skip_special_tokens=True) for r in whisper_wer.load_manifest(heard)}
        runs[name] = (audio, text)
    table = []
    for name, (audio, text) in runs.items():
        for group, pick in [("all", lambda i: True), ("Harper", lambda i: i.startswith("Harper")), ("Hadley", lambda i: i.startswith("Hadley")), ("say", lambda i: "-say-" in i), ("librispeech", lambda i: "-ls-" in i)]:
            ids = [i for i in truth if pick(i)]
            e = sum(whisper_wer.edits(norm(truth[i]["text"]).split(), norm(text[i]).split()) for i in ids)
            n = sum(len(norm(truth[i]["text"]).split()) for i in ids)
            exact = sum(norm(truth[i]["text"]) == norm(text[i]) for i in ids)
            secs = sum(audio[i]["audio_seconds"] for i in ids)
            table.append({"run": name, "group": group, "wer": 100.0 * e / max(n, 1), "exact": exact, "count": len(ids), "audio": secs})
    print("| run | set | WER % | exact transcripts | audio s |")
    print("|---|---|---|---|---|")
    for r in table:
        print(f"| {r['run']} | {r['group']} | {r['wer']:.2f} | {r['exact']}/{r['count']} | {r['audio']:.1f} |")
    if len(runs) == 2:
        (a, (aa, at)), (b, (ba, bt)) = runs.items()
        ratios = sorted(((aa[i]["audio_seconds"] / ba[i]["audio_seconds"], i) for i in truth))
        far = [(x, i) for x, i in ratios if abs(x - 1) > 0.10]
        print(f"\nduration {a} / {b}: min {ratios[0][0]:.3f} ({ratios[0][1]}), max {ratios[-1][0]:.3f} ({ratios[-1][1]}), {len(far)} of {len(ratios)} outside 10%")
        for x, i in far:
            print(f"  {i} {x:.3f} {aa[i]['audio_seconds']:.2f}s against {ba[i]['audio_seconds']:.2f}s")
        for i in truth:
            if norm(at[i]) != norm(bt[i]) or norm(at[i]) != norm(truth[i]["text"]):
                print(f"{i}\n  text: {truth[i]['text']}\n  {a}: {at[i].strip()}\n  {b}: {bt[i].strip()}")
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1))


def main():
    p = argparse.ArgumentParser(description="Phonon speech quality: sentences, candle runs, 16 kHz manifests, WER and durations")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sentences")
    s.add_argument("manifest")
    s.add_argument("voices")
    s.add_argument("out")
    c = sub.add_parser("candle")
    c.add_argument("say")
    c.add_argument("model")
    c.add_argument("sentences")
    c.add_argument("out")
    c.add_argument("--prefix", default="")
    r = sub.add_parser("to16k")
    r.add_argument("rows")
    r.add_argument("out")
    k = sub.add_parser("score")
    k.add_argument("whisper")
    k.add_argument("sentences")
    k.add_argument("runs", nargs="+", help="name=rows.jsonl,whisper.jsonl")
    k.add_argument("--json")
    args = p.parse_args()
    {"sentences": sentences, "candle": candle, "to16k": to16k, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
