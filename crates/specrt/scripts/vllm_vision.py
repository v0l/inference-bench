import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8001") + "/v1/chat/completions"
MODEL = "qwen3.8-27b"
NEW = 32
TOP = 20
FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures" / "vision"

MULTI = {
    "images3": (["house.png", "shapes.png", "lines.png"], "image/png", "image_url", "Describe each of these images in detail."),
    "video10": (["clip10.mp4"], "video/mp4", "video_url", "Describe this video in detail."),
    "video2f": (["clip2f.mp4"], "video/mp4", "video_url", "Describe this video in detail."),
    "video4f": (["clip4f.mp4"], "video/mp4", "video_url", "Describe this video in detail."),
}

CASES = {
    "image": ("house.png", "image/png", "image_url", "Describe this image in detail."),
    "video": ("clip.mp4", "video/mp4", "video_url", "Describe this video in detail."),
}


def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def token_id(key):
    return int(key.split(":", 1)[1])


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cases = {n: ([f], m, k, q) for n, (f, m, k, q) in CASES.items()} | MULTI
    for i in range(6):
        cases[f"set_img{i}"] = ([f"set/img{i}.png"], "image/png", "image_url", "Describe this image in detail.")
        cases[f"set_vid{i}"] = ([f"set/vid{i}.mp4"], "video/mp4", "video_url", "Describe this video in detail.")
    for name, (files, mime, kind, question) in cases.items():
        file = files[0]
        content = [{"type": kind, kind: {"url": f"data:{mime};base64," + base64.b64encode((FIXTURES / f).read_bytes()).decode()}} for f in files]
        content.append({"type": "text", "text": question})
        body = {"model": MODEL, "max_tokens": NEW, "temperature": 0, "logprobs": True, "top_logprobs": TOP,
                "return_tokens_as_token_ids": True, "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": content}]}
        r = post(body)
        choice = r["choices"][0]
        lp = choice["logprobs"]["content"]
        greedy = [token_id(t["token"]) for t in lp]
        rows = [{str(token_id(c["token"])): c["logprob"] for c in t["top_logprobs"]} for t in lp]
        record = {"question": question, "file": file, "kind": kind, "files": files, "prompt_ids": r["choices"][0].get("prompt_token_ids") or r.get("prompt_token_ids"),
                  "greedy": greedy, "logprobs": rows, "text": choice["message"]["content"], "usage": r["usage"]}
        (out / f"vllm-{name}.json").write_text(json.dumps(record))
        print(name, record["text"], r["usage"], "prompt ids" if record["prompt_ids"] else "no prompt ids", flush=True)


main()
