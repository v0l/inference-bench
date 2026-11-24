import json
import sys
from pathlib import Path

from transformers.utils.chat_template_utils import render_jinja_template

hub = Path(__file__).resolve().parent.parent / "tests/fixtures/templates/hub"
cases = json.loads((hub / "cases.json").read_text())
only = sys.argv[1:]

for repo in sorted(p for p in hub.iterdir() if p.is_dir()):
    if only and repo.name not in only:
        continue
    config = json.loads((repo / "tokenizer_config.json").read_text())
    template = (repo / "chat_template.jinja").read_text()
    out = {}
    for name, case in cases.items():
        kwargs = dict(case["vars"])
        tools = kwargs.pop("tools", None)
        rendered, _ = render_jinja_template(
            conversations=[case["messages"]],
            tools=tools,
            chat_template=template,
            add_generation_prompt=True,
            bos_token=config.get("bos_token") or "",
            eos_token=config.get("eos_token") or "",
            **kwargs,
        )
        out[name] = rendered[0]
    (repo / "expected.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(repo.name, {k: len(v) for k, v in out.items()})
