#!/usr/bin/env python3
import json
import sys
import urllib.request
from tokenizers import Tokenizer

here = sys.argv[1]
tokenizer = Tokenizer.from_file(sys.argv[2])
url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8001/tokenize"
req = json.load(open(f"{here}/agent-request.json"))
cases = {"medium": None, "no_thinking": {"enable_thinking": False}, "low": {"reasoning_effort": "low"}}
prompts = {}
for name, kwargs in cases.items():
    body = {"model": req["model"], "messages": req["messages"], "tools": req["tools"]}
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    post = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    ids = json.load(urllib.request.urlopen(post))["tokens"]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    assert tokenizer.encode(text, add_special_tokens=False).ids == ids, name
    prompts[name] = {"tokens": ids, "text": text}
json.dump(prompts, open(f"{here}/agent-request.vllm.json", "w"), ensure_ascii=False, indent=1)
