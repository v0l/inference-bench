import argparse
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

import whisper_wer


def post(url, token, body, content_type):
    req = urllib.request.Request(url, data=body, method="POST", headers={"Authorization": f"Bearer {token}", "Content-Type": content_type, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return r.read(), dict(r.headers)


def form(fields, wav):
    b = f"agentbench-{uuid.uuid4().hex}"
    out = b""
    for k, v in fields.items():
        out += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    out += f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode() + wav + f"\r\n--{b}--\r\n".encode()
    return out, f"multipart/form-data; boundary={b}"


def done_ids(path):
    return {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()


def hear(args):
    done = done_ids(args.out)
    with open(args.out, "a") as f:
        for r in whisper_wer.load_manifest(args.manifest):
            if r["id"] in done:
                continue
            body, ct = form({"model": args.model, "response_format": "json"}, Path(r["wav"]).read_bytes())
            start = time.time()
            got, _ = post(f"{args.url}/v1/audio/transcriptions", args.token, body, ct)
            v = json.loads(got)
            f.write(json.dumps({"id": r["id"], "text": v["text"], "no_speech_prob": v.get("no_speech_prob"), "avg_logprob": v.get("avg_logprob"), "seconds": round(time.time() - start, 3)}) + "\n")
            f.flush()
            print(r["id"], f"{time.time() - start:.2f}s", v["text"][:80], flush=True)


def wav_header(n, rate):
    import struct

    return b"RIFF" + struct.pack("<I", 36 + n * 2) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", n * 2)


def say(args):
    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    rows = out / f"{args.name}.jsonl"
    done = done_ids(rows)
    with open(rows, "a") as f:
        for r in whisper_wer.load_manifest(args.sentences):
            if r["id"] in done:
                continue
            body = {"model": "kokoro", "input": r["text"], "voice": r["voice_name"], "seed": args.seed}
            if args.stream:
                body["response_format"] = "pcm"
            start = time.time()
            got, headers = post(f"{args.url}/v1/audio/speech", args.token, json.dumps(body).encode(), "application/json")
            if args.stream:
                rate = int(headers.get("x-sample-rate") or headers.get("X-Sample-Rate"))
                wav = wav_header(len(got) // 2, rate) + got
            else:
                wav = got
            dst = out / "wav" / f"{r['id']}.wav"
            dst.write_bytes(wav)
            samples = (len(wav) - 44) // 2
            f.write(json.dumps({"id": r["id"], "wav": str(dst), "samples": samples, "audio_seconds": samples / 24000, "seconds": round(time.time() - start, 3)}) + "\n")
            f.flush()
            print(r["id"], f"{samples / 24000:.2f}s audio in {time.time() - start:.2f}s", flush=True)


def same(args):
    a, b = Path(args.a), Path(args.b)
    ids = [json.loads(l)["id"] for l in open(args.sentences)]
    equal = [i for i in ids if (a / f"{i}.wav").read_bytes()[44:] == (b / f"{i}.wav").read_bytes()[44:]]
    print(f"{len(equal)}/{len(ids)} sample identical")
    for i in ids:
        if i not in equal:
            print("differs", i)


def main():
    p = argparse.ArgumentParser(description="The c2 and f2 sets through agentbench's /v1/audio routes")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("hear", "say"):
        s = sub.add_parser(name)
        s.add_argument("url")
        s.add_argument("token")
    h = sub.choices["hear"]
    h.add_argument("model")
    h.add_argument("manifest")
    h.add_argument("out")
    s = sub.choices["say"]
    s.add_argument("sentences")
    s.add_argument("out")
    s.add_argument("--name", default="agentbench")
    s.add_argument("--seed", type=int, default=1484)
    s.add_argument("--stream", action="store_true")
    c = sub.add_parser("same")
    c.add_argument("sentences")
    c.add_argument("a")
    c.add_argument("b")
    args = p.parse_args()
    {"hear": hear, "say": say, "same": same}[args.cmd](args)


if __name__ == "__main__":
    main()
