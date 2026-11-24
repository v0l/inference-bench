import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

LLAMA = Path.home() / "git/llama.cpp"
sys.path.insert(0, str(LLAMA))
sys.path.insert(0, str(LLAMA / "gguf-py"))

from convert_hf_to_gguf import ModelBase, get_model_class
import gguf
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
vocab_size = 1024

cfg = Qwen3MoeConfig(
    vocab_size=vocab_size,
    hidden_size=256,
    intermediate_size=512,
    moe_intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    num_experts=4,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    decoder_sparse_step=1,
    mlp_only_layers=[],
    max_position_embeddings=4096,
    rope_theta=1000000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
)
torch.manual_seed(1168)
model = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16)
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name:
            p.copy_(1.0 + 0.1 * torch.randn_like(p))
        else:
            p.copy_(0.06 * torch.randn_like(p))

work = Path(tempfile.mkdtemp(prefix="qwen3moe-fixture-"))
model.save_pretrained(work)
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
(work / "chat_template.jinja").write_text("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")

bf16 = out / "tiny-moe-BF16.gguf"
cls = get_model_class("Qwen3MoeForCausalLM")
cls.get_vocab_base_pre = lambda self, tokenizer: "qwen2"
conv = cls(work, gguf.LlamaFileType.MOSTLY_BF16, bf16, is_big_endian=False, use_temp_file=False, eager=True)
conv.write()
shutil.rmtree(work)
for q in ["Q4_K_M", "Q8_0"]:
    subprocess.run([str(LLAMA / "build/bin/llama-quantize"), str(bf16), str(out / f"tiny-moe-{q}.gguf"), q], check=True, capture_output=True)
bf16.unlink()
print(out)
