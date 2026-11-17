import json
import shutil
import sys
import tempfile
from pathlib import Path

LLAMA = Path.home() / "git/llama.cpp"
sys.path.insert(0, str(LLAMA))
sys.path.insert(0, str(LLAMA / "gguf-py"))

import conversion.base as base
from convert_hf_to_gguf import ModelBase, get_model_class
import gguf
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

src = Path(sys.argv[1])
out = Path(sys.argv[2])
outtype = sys.argv[3] if len(sys.argv) > 3 else "bf16"

work = Path(tempfile.mkdtemp(prefix="qwen35-fixture-"))
for f in src.iterdir():
    (work / f.name).symlink_to(f.resolve())
cfg = json.loads((src / "config.json").read_text())
vocab_size = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]
cfg.setdefault("architectures", ["Qwen3_5ForCausalLM"])
(work / "config.json").unlink()
(work / "config.json").write_text(json.dumps(cfg))

byte_chars = list(pre_tokenizers.ByteLevel.alphabet())
vocab = {c: i for i, c in enumerate(byte_chars)}
for i in range(len(vocab), vocab_size - 3):
    vocab[f"tok{i}"] = i
specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
for s in specials:
    vocab[s] = len(vocab)
tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.add_special_tokens(specials)
tok.save(str(work / "tokenizer.json"))
(work / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|im_end|>", "bos_token": None, "pad_token": "<|endoftext|>"}))
(work / "vocab.json").write_text(json.dumps(vocab))
(work / "merges.txt").write_text("#version: 0.2\n")
if (work / "chat_template.jinja").exists():
    (work / "chat_template.jinja").unlink()
(work / "chat_template.jinja").write_text("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")

ftypes = {"bf16": gguf.LlamaFileType.MOSTLY_BF16, "f16": gguf.LlamaFileType.MOSTLY_F16, "f32": gguf.LlamaFileType.ALL_F32}
hparams = ModelBase.load_hparams(work, False)
cls_no_mtp = not (cfg.get("mtp_num_hidden_layers") or cfg.get("text_config", {}).get("mtp_num_hidden_layers"))
arch = hparams["architectures"][0] if hparams.get("architectures") else "Qwen3_5ForCausalLM"
cls = get_model_class(arch if arch in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration") else "Qwen3_5ForCausalLM")
cls.get_vocab_base_pre = lambda self, tokenizer: "qwen35"
cls.no_mtp = cls_no_mtp
model = cls(work, ftypes[outtype], out, is_big_endian=False, use_temp_file=False, eager=True)
model.write()
shutil.rmtree(work)
print(out)
