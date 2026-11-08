import json
import sys
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

URL = "http://127.0.0.1:8001/v1/completions"
MODEL = "qwen3.8-27b"
REPO = "orcarouter/Qwen3.8-27B-Uncensored-NVFP4"
NEW = 64
TOP = 20

EXTRA = [
    "The Treaty of Westphalia in 1648 ended the Thirty Years' War and is often described as the origin of the modern state system, because",
    "To configure a WireGuard tunnel between two Linux hosts, first generate a key pair on each machine with",
    "fn parse_header(input: &[u8]) -> Result<Header, Error> {\n    if input.len() < 8 {\n",
    "In a hash table with open addressing, deletion is harder than insertion because",
    "The recipe calls for 250 grams of flour, two eggs and a pinch of salt. First, sift the flour into a large bowl, then",
    "SELECT customer_id, COUNT(*) AS orders\nFROM orders\nWHERE created_at >= '2024-01-01'\n",
    "A capacitor charged to 12 volts through a 10 kilohm resistor reaches 63 percent of its final voltage after one time constant, which for a 100 microfarad capacitor is",
    "Once upon a time, in a village at the edge of a great forest, there lived an old clockmaker who",
    "The main difference between TCP and UDP is that",
    "Bitcoin's proof of work adjusts its difficulty every 2016 blocks so that",
    "import numpy as np\n\ndef softmax(x):\n    \"\"\"Numerically stable softmax over the last axis.\"\"\"\n",
    "The mitochondria produce most of the cell's ATP through oxidative phosphorylation, a process in which",
    "Dear Sir or Madam, I am writing to complain about the delivery of my order number 48213, which",
    "Translate to French: The weather is beautiful today and we are going to the beach.\nFrench:",
    "The proof that there are infinitely many primes goes back to Euclid. Suppose there were only finitely many primes p1, p2, ..., pn. Then",
]


def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def token_id(key):
    return int(key.split(":", 1)[1])


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    from reference import REAL_PROMPTS, snapshot_dir

    out = Path(sys.argv[1])
    tok = AutoTokenizer.from_pretrained(snapshot_dir(REPO))
    prompts = [tok(t).input_ids for t in REAL_PROMPTS + EXTRA]
    records = []
    for i, ids in enumerate(prompts):
        g = post({"model": MODEL, "prompt": ids, "max_tokens": NEW, "temperature": 0, "ignore_eos": True,
                  "return_tokens_as_token_ids": True, "logprobs": TOP})
        lp = g["choices"][0]["logprobs"]
        greedy = [token_id(t) for t in lp["tokens"]]
        rows = [{str(token_id(k)): v for k, v in top.items()} for top in lp["top_logprobs"]]
        records.append({"prompt": ids, "greedy": greedy, "logprobs": rows})
        print(f"prompt {i}: {len(ids)} tokens, greedy {tok.decode(greedy[:16])!r}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vllm-nvfp4.json").write_text(json.dumps({"model": REPO, "new_tokens": NEW, "top": TOP, "records": records}))


if __name__ == "__main__":
    main()
