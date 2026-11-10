import base64
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

BASE = os.environ.get("VLLM_URL", "http://127.0.0.1:8001")
URL = BASE + "/v1/chat/completions"
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


def post(body, url=URL):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def token_id(key):
    return int(key.split(":", 1)[1])


TEXT_PROMPTS = [
    "Describe a red barn in a green field under a blue sky with a yellow sun, in detail.",
    "Describe a short abstract animation made of colour bars and moving shapes, in detail.",
    "Describe three simple drawings: a house, some coloured shapes and crossing lines, in detail.",
    "Describe a clock counting seconds on a test pattern video, in detail.",
    "Describe a rural scene with a barn, in detail.",
    "Describe a digital illustration of geometric shapes, in detail.",
]
PASSAGE = (
    "The harbour town sat at the mouth of a wide river, and every morning the fishing boats left before dawn while the bakers lit their ovens. "
    "A clockmaker on the main street repaired watches for sailors, and children gathered at his window to watch the tiny gears turn. "
    "In the market, traders argued over the price of salt, rope and dried fruit, while a street musician played a slow tune on a battered violin. "
)
LONG_REPEATS = (1, 3, 6, 10, 20)
RANDOM_PROMPTS = [(1, 160), (2, 240), (3, 320), (4, 240), (5, 480), (6, 160)]
PLP_CASES = ["image", "video", "images3", "video10", "video2f", "video4f"]


def text_record(ids, question):
    full = post({"model": MODEL, "prompt": ids, "max_tokens": NEW, "temperature": 0, "ignore_eos": True,
                 "return_tokens_as_token_ids": True, "logprobs": TOP}, BASE + "/v1/completions")
    lp = full["choices"][0]["logprobs"]
    return {"prompt_ids": ids, "greedy": [token_id(t) for t in lp["tokens"]],
            "logprobs": [{str(token_id(k)): v for k, v in top.items()} for top in lp["top_logprobs"]], "question": question}


def chat_prompt_ids(text):
    r = post({"model": MODEL, "max_tokens": NEW, "temperature": 0, "return_token_ids": True,
              "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": text}]})
    return r["choices"][0].get("prompt_token_ids") or r.get("prompt_token_ids")


def text_controls(out):
    sets = {
        "textctrl": [(chat_prompt_ids(q), q) for q in TEXT_PROMPTS],
        "longtext": [(chat_prompt_ids(q), q[:40]) for q in (PASSAGE * n + "\nDescribe the town in detail." for n in LONG_REPEATS)],
    }
    rand = []
    for seed, n in RANDOM_PROMPTS:
        random.seed(seed)
        rand.append(([random.randint(1000, 240000) for _ in range(n)], "random"))
    sets["randtext"] = rand
    for name, items in sets.items():
        records = [text_record(ids, q) for ids, q in items]
        (out / f"vllm-{name}.json").write_text(json.dumps(records))
        print(name, [len(r["prompt_ids"]) for r in records], flush=True)


def media_content(files, mime, kind, question):
    content = [{"type": kind, kind: {"url": f"data:{mime};base64," + base64.b64encode((FIXTURES / f).read_bytes()).decode()}} for f in files]
    content.append({"type": "text", "text": question})
    return content


def prompt_logprobs(out, cases):
    for name in PLP_CASES:
        files, mime, kind, question = cases[name]
        body = {"model": MODEL, "max_tokens": 1, "temperature": 0, "prompt_logprobs": TOP, "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": media_content(files, mime, kind, question)}]}
        r = post(body)
        rows = r.get("prompt_logprobs") or r["choices"][0].get("prompt_logprobs")
        ids = r.get("prompt_token_ids") or r["choices"][0].get("prompt_token_ids")
        (out / f"plp-{name}.json").write_text(json.dumps({"ids": ids, "prompt_logprobs": rows}))
        print("plp", name, len(ids), len(rows), flush=True)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cases = {n: ([f], m, k, q) for n, (f, m, k, q) in CASES.items()} | MULTI
    for i in range(6):
        cases[f"set_img{i}"] = ([f"set/img{i}.png"], "image/png", "image_url", "Describe this image in detail.")
        cases[f"set_vid{i}"] = ([f"set/vid{i}.mp4"], "video/mp4", "video_url", "Describe this video in detail.")
    for name, (files, mime, kind, question) in cases.items():
        file = files[0]
        content = media_content(files, mime, kind, question)
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
    text_controls(out)
    prompt_logprobs(out, cases)


main()
