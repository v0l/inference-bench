import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).parent.parent
OUT = ROOT / "tests/fixtures/prompt-ids/qwen3.json"
PROMPTS = ["a red fox, studio light", "", "café ☕ at dusk, 35mm"]
MAX_LENGTHS = [32, 16]


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(repo)
    cases = []
    for prompt in PROMPTS:
        for max_length in MAX_LENGTHS:
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tok(text, padding="max_length", truncation=True, max_length=max_length)
            cases.append({"prompt": prompt, "max_length": max_length, "text": text, "ids": inputs["input_ids"], "mask": inputs["attention_mask"]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"repo": repo, "cases": cases}, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
