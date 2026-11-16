import json
import sys
import urllib.request
from pathlib import Path

PORT = 18639
NEW = 48
TOP = 20

PROMPTS = [
    "The Treaty of Westphalia in 1648 ended the Thirty Years' War and is often described as the origin of the modern state system, because",
    "To configure a WireGuard tunnel between two Linux hosts, first generate a key pair on each machine with",
    "fn parse_header(input: &[u8]) -> Result<Header, Error> {\n    if input.len() < 8 {\n",
    "In a hash table with open addressing, deletion is harder than insertion because",
    "The recipe calls for 250 grams of flour, two eggs and a pinch of salt. First, sift the flour into a large bowl, then",
    "SELECT customer_id, COUNT(*) AS orders\nFROM orders\nWHERE created_at >= '2024-01-01'\n",
    "Once upon a time, in a village at the edge of a great forest, there lived an old clockmaker who",
    "The main difference between TCP and UDP is that",
]


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def token_id(key):
    return int(key.split(":", 1)[1])


def long_prompt(tok):
    text = " ".join(f"Entry {i}: the quick brown fox number {i} jumps over lazy dog {i * 7}." for i in range(160))
    return tok(text + " Summarise the entries above in one sentence:").input_ids


def main(model_dir, out_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    prompts = [tok(t).input_ids for t in PROMPTS] + [long_prompt(tok)]
    records = []
    for i, ids in enumerate(prompts):
        g = post({"model": "gemma27", "prompt": ids, "max_tokens": NEW, "temperature": 0, "ignore_eos": True, "return_tokens_as_token_ids": True, "logprobs": TOP})
        lp = g["choices"][0]["logprobs"]
        greedy = [token_id(t) for t in lp["tokens"]]
        rows = [{str(token_id(k)): v for k, v in top.items()} for top in lp["top_logprobs"]]
        records.append({"prompt": ids, "greedy": greedy, "logprobs": rows})
        print(f"prompt {i}: {len(ids)} tokens, greedy {tok.decode(greedy[:12])!r}", flush=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vllm-gemma3-27b.json").write_text(json.dumps({"model": model_dir, "new_tokens": NEW, "top": TOP, "records": records}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
