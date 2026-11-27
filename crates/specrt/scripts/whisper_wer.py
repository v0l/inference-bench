import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RULE = "the first 200 utterances of test-clean in speaker, chapter, utterance order, plus 5 long clips: consecutive utterances of one chapter after the first 200, joined with 0.5 s of silence until the clip passes 30 s (at most 60 s), one per chapter, the first 5 chapters that allow it"
RATE = 16000
SHORT = 200
LONG = 5
LONG_MIN = 30.0
LONG_MAX = 60.0
GAP = 0.5


def utterances(root):
    out = []
    for trans in sorted(Path(root).glob("*/*/*.trans.txt"), key=lambda p: tuple(int(x) for x in p.stem.split(".")[0].split("-"))):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            out.append((uid, text, trans.parent / f"{uid}.flac"))
    out.sort(key=lambda u: tuple(int(x) for x in u[0].split("-")))
    return out


def decode_flac(path):
    import numpy as np

    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(RATE), "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16)


def write_wav(path, pcm):
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def clips(args):
    import numpy as np

    utts = utterances(args.root)
    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    rows = []
    for uid, text, flac in utts[:SHORT]:
        pcm = decode_flac(flac)
        write_wav(out / "wav" / f"{uid}.wav", pcm)
        rows.append({"id": uid, "text": text, "seconds": round(len(pcm) / RATE, 3), "parts": [uid]})
    by_chapter = {}
    for uid, text, flac in utts[SHORT:]:
        by_chapter.setdefault(uid.rsplit("-", 1)[0], []).append((uid, text, flac))
    made = 0
    for chapter, items in by_chapter.items():
        if made == LONG:
            break
        parts, texts, pcms, seconds = [], [], [], 0.0
        for uid, text, flac in items:
            pcm = decode_flac(flac)
            if pcms:
                pcms.append(np.zeros(int(GAP * RATE), dtype=np.int16))
                seconds += GAP
            pcms.append(pcm)
            parts.append(uid)
            texts.append(text)
            seconds += len(pcm) / RATE
            if seconds > LONG_MIN:
                break
        if seconds <= LONG_MIN or seconds > LONG_MAX:
            continue
        joined = np.concatenate(pcms)
        cid = f"long-{chapter}"
        write_wav(out / "wav" / f"{cid}.wav", joined)
        rows.append({"id": cid, "text": " ".join(texts), "seconds": round(len(joined) / RATE, 3), "parts": parts})
        made += 1
    assert made == LONG, f"only {made} long clips"
    with open(out / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({**r, "wav": str((out / "wav" / f"{r['id']}.wav").resolve())}) + "\n")
    if args.list:
        Path(args.list).write_text(json.dumps({"source": "LibriSpeech test-clean (openslr.org/12, CC BY 4.0)", "rule": RULE, "clips": [{k: r[k] for k in ("id", "seconds", "parts")} for r in rows]}, indent=1) + "\n")
    print(f"{len(rows)} clips, {sum(r['seconds'] for r in rows):.1f} s, long {[r['seconds'] for r in rows[SHORT:]]}")


def load_manifest(path):
    return [json.loads(line) for line in open(path)]


def reference(args):
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.set_num_threads(args.threads)
    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    lang = {"language": "en", "task": "transcribe"} if model.generation_config.is_multilingual else {}
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(line)["id"] for line in open(args.out)}
    with open(args.out, "a") as f:
        for r in load_manifest(args.manifest):
            if r["id"] in done:
                continue
            import wave

            with wave.open(r["wav"]) as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            start = time.time()
            long = len(pcm) > 30 * RATE
            if long:
                feats = proc.feature_extractor(pcm, sampling_rate=RATE, return_tensors="pt", padding="longest", truncation=False).input_features
                kw = {"return_timestamps": True, "condition_on_prev_tokens": False, "temperature": 0.0}
            else:
                feats = proc.feature_extractor(pcm, sampling_rate=RATE, return_tensors="pt").input_features
                kw = {"return_timestamps": False}
            with torch.no_grad():
                ids = model.generate(feats, **kw, **lang)
            tokens = [int(t) for t in ids[0]]
            f.write(json.dumps({"id": r["id"], "tokens": tokens, "seconds": round(time.time() - start, 3)}) + "\n")
            f.flush()
            print(r["id"], f"{time.time() - start:.1f}s", proc.tokenizer.decode(tokens, skip_special_tokens=True)[:80], flush=True)


def candle(args):
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(line)["id"] for line in open(args.out)}
    with open(args.out, "a") as f:
        for r in load_manifest(args.manifest):
            if r["id"] in done:
                continue
            start = time.time()
            out = subprocess.run(args.prefix.split() + [args.hear, r["wav"], args.models, args.model], check=True, capture_output=True, text=True).stdout.splitlines()
            out = [l for l in out if not l.startswith("gpu-run:")]
            text = "\n".join(out[2:]).strip()
            f.write(json.dumps({"id": r["id"], "text": text, "seconds": round(time.time() - start, 3)}) + "\n")
            f.flush()
            print(r["id"], f"{time.time() - start:.1f}s", text[:80], flush=True)


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def score(args):
    from transformers import WhisperTokenizer
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    tok = WhisperTokenizer.from_pretrained(args.model)
    norm = EnglishTextNormalizer(json.load(open(Path(args.model) / "normalizer.json")))
    manifest = load_manifest(args.manifest)

    def text_of(row):
        return row["text"] if "text" in row else tok.decode(row["tokens"], skip_special_tokens=True)

    def read(path):
        return {r["id"]: text_of(r) for r in map(json.loads, open(path))}

    runs = {"transformers": read(args.reference)}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        runs[name] = read(path)
    truth = {r["id"]: norm(r["text"]) for r in manifest}
    table = []
    for name, got in runs.items():
        missing = [r["id"] for r in manifest if r["id"] not in got]
        row = {"name": name, "missing": len(missing)}
        for part, ids in (("short", [r["id"] for r in manifest if len(r["parts"]) == 1]), ("long", [r["id"] for r in manifest if len(r["parts"]) > 1]), ("all", [r["id"] for r in manifest])):
            ids = [i for i in ids if i in got]
            e = sum(edits(truth[i].split(), norm(got[i]).split()) for i in ids)
            n = sum(len(truth[i].split()) for i in ids)
            row[part] = 100.0 * e / max(n, 1)
        if name != "transformers":
            base = runs["transformers"]
            same = [i for i in got if i in base]
            row["identical"] = sum(got[i].strip() == base[i].strip() for i in same)
            row["identical_norm"] = sum(norm(got[i]) == norm(base[i]) for i in same)
            row["compared"] = len(same)
            row["gap"] = row["all"] - table[0]["all"]
            row["differ"] = [i for i in same if got[i].strip() != base[i].strip()]
        table.append(row)
    print(f"| run | WER all % | WER 200 short % | WER 5 long % | gap to transformers | identical transcripts |")
    print("|---|---|---|---|---|---|")
    for r in table:
        same = f"{r['identical']}/{r['compared']} ({100.0 * r['identical'] / max(r['compared'], 1):.1f}%), normalised {r['identical_norm']}" if "identical" in r else ""
        gap = f"{r['gap']:+.2f}" if "gap" in r else ""
        miss = f" ({r['missing']} missing)" if r["missing"] else ""
        print(f"| {r['name']}{miss} | {r['all']:.2f} | {r['short']:.2f} | {r['long']:.2f} | {gap} | {same} |")
    for r in table[1:]:
        for i in r["differ"]:
            print(f"{r['name']} {i}\n  transformers: {runs['transformers'][i].strip()}\n  {r['name']}: {runs[r['name']][i].strip()}")
    if args.json:
        Path(args.json).write_text(json.dumps([{k: v for k, v in r.items()} for r in table], indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clips")
    c.add_argument("root")
    c.add_argument("out")
    c.add_argument("--list")
    r = sub.add_parser("reference")
    r.add_argument("model")
    r.add_argument("manifest")
    r.add_argument("out")
    r.add_argument("--threads", type=int, default=8)
    k = sub.add_parser("candle")
    k.add_argument("hear")
    k.add_argument("models")
    k.add_argument("model")
    k.add_argument("manifest")
    k.add_argument("out")
    k.add_argument("--prefix", default="")
    s = sub.add_parser("score")
    s.add_argument("model")
    s.add_argument("manifest")
    s.add_argument("reference")
    s.add_argument("runs", nargs="*")
    s.add_argument("--json")
    a = p.parse_args()
    {"clips": clips, "reference": reference, "candle": candle, "score": score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
