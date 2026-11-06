import argparse
import json
import resource
import sys
import threading
import time


def emit(**event):
    print(json.dumps(event), flush=True)


def fail(exc):
    emit(event="error", kind="failed", message=f"{type(exc).__name__}: {exc}")


class Worker:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.tok = None

    def load(self, name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
        self.model.eval()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def peak_memory(self):
        if self.device == "cuda":
            return int(self.torch.cuda.max_memory_allocated())
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def chat(self, msg):
        messages = msg["messages"]
        if msg.get("template", True) and getattr(self.tok, "chat_template", None):
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n\n".join(m["content"] for m in messages)
        temperature = msg.get("temperature", 0.0)
        sampling = dict(do_sample=temperature > 0, temperature=temperature or 1.0, top_p=msg.get("top_p", 1.0))
        self.generate(prompt, msg["max_tokens"], sampling, msg.get("seed"))

    def generate(self, prompt, max_tokens, sampling=None, seed=None):
        from transformers import TextIteratorStreamer

        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True)
        if seed:
            self.torch.manual_seed(seed)
        kwargs = dict(input_ids=ids, max_new_tokens=max_tokens, streamer=streamer, min_new_tokens=1)
        kwargs.update(sampling or dict(do_sample=False))
        started = time.perf_counter()
        thread = threading.Thread(target=lambda: self.model.generate(**kwargs))
        thread.start()
        first = None
        text = ""
        for piece in streamer:
            if piece and first is None:
                first = time.perf_counter()
            text += piece
            if piece:
                emit(event="text", text=piece)
        thread.join()
        self.sync()
        end = time.perf_counter()
        first = first or end
        completion = len(self.tok(text, add_special_tokens=False).input_ids)
        emit(
            event="done",
            generation=dict(
                text=text,
                prompt_tokens=ids.shape[1],
                completion_tokens=completion,
                ttft_secs=first - started,
                prefill_secs=first - started,
                decode_secs=end - first,
                peak_memory_bytes=self.peak_memory(),
            ),
        )

    def logits(self, prompt, path):
        import numpy as np

        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        with self.torch.inference_mode():
            out = self.model(ids).logits[0].float().cpu().numpy()
        out.astype(np.float32).tofile(path)
        emit(event="logits", tokens=ids[0].tolist(), vocab=int(out.shape[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    args = parser.parse_args()
    worker = Worker(args.device)
    for line in sys.stdin:
        msg = json.loads(line)
        op = msg["op"]
        try:
            if op == "load":
                worker.load(msg["model"])
                emit(event="loaded")
            elif op == "generate":
                worker.generate(msg["prompt"], msg["max_tokens"])
            elif op == "chat":
                worker.chat(msg)
            elif op == "logits":
                worker.logits(msg["prompt"], msg["path"])
            elif op == "quit":
                break
        except Exception as exc:
            fail(exc)


if __name__ == "__main__":
    main()
