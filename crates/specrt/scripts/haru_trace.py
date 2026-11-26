import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

LAYERS = (0, 8, 16, 24, 31)
TOP = 10
FRAME = 3200


def floats(t):
    return torch.as_tensor(t).detach().float().contiguous().cpu().numpy().astype(np.float32)


class Store:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.files = {}

    def put(self, key, tensor):
        array = floats(tensor)
        file = f"{key}.bin"
        array.tofile(self.out / file)
        self.files[key] = {"file": file, "shape": list(array.shape)}

    def finish(self, meta):
        (self.out / "index.json").write_text(json.dumps({"meta": meta, "files": self.files}, indent=1) + "\n")


def read_wav(path):
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    assert rate == 24000, f"{path} is {rate} Hz"
    return audio.mean(axis=1).astype(np.float32)


def write_wav(path, audio):
    import soundfile as sf

    sf.write(str(path), np.asarray(audio, dtype=np.float32), 24000, subtype="PCM_16")


def setup(args):
    sys.path.insert(0, str(Path(args.haru).resolve()))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    from haru.models import load_model

    dtype = torch.float32 if args.dtype == "f32" else torch.bfloat16
    return load_model(args.model, device="cuda", dtype=dtype)


class Teacher:
    def __init__(self, path, device):
        self.dir = Path(path)
        self.index = json.loads((self.dir / "index.json").read_text())
        self.device = device

    def get(self, key):
        f = self.index["files"][key]
        data = np.fromfile(self.dir / f["file"], dtype=np.float32).reshape(f["shape"])
        return torch.from_numpy(data).to(self.device)

    def prompt(self):
        from haru.prompt import Prompt

        return Prompt(
            tokens=self.get("prompt.tokens").long(),
            model_latents=self.get("prompt.model_latents"),
            user_latents=self.get("prompt.user_latents"),
            model_mask=self.get("prompt.model_mask").bool(),
            user_mask=self.get("prompt.user_mask").bool(),
        )


class Recorder:
    def __init__(self, model, say=None, teacher=None):
        import haru.session as session_mod

        self.model = model
        self.say = say
        self.teacher = teacher
        if teacher is not None:
            self.t = {k: teacher.get(k) for k in ("frame.token", "frame.latent", "frame.head_noise", "frame.user_scale", "frame.user_noise", "frame.top_ids")}
        self.reset()
        self.hooks = [model.lm.layers[i].register_forward_hook(self.layer_hook(i)) for i in LAYERS]
        self.orig = {
            "embed": model.embed,
            "sample": model.codec.sample,
            "head": model.head.sample,
            "token": session_mod.sample_token,
            "prompt": session_mod.build_prompt,
        }
        model.embed = self.embed
        model.codec.sample = self.codec_sample
        model.head.sample = self.head_sample
        session_mod.sample_token = self.sample_token
        session_mod.build_prompt = self.build_prompt

    def reset(self):
        self.layer = {}
        self.frames = {k: [] for k in ("embeds", "layers", "hidden", "top_ids", "top_logits", "teacher_logits", "pad_eop", "token", "head_noise", "latent", "user_mean", "user_scale", "user_noise", "user_latent")}
        self.prompt = None
        self.ref = None
        self.in_prompt = True

    def layer_hook(self, i):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            self.layer[i] = out[0, -1].detach().float().clone()

        return hook

    def embed(self, *a, **k):
        x = self.orig["embed"](*a, **k)
        self.last_embed = x[0, -1].detach().float().clone()
        return x

    def codec_sample(self, mean, generator=None):
        scale = torch.randn(mean.shape[0], 1, 1, device=mean.device, dtype=mean.dtype, generator=generator)
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        out = mean + scale * (self.model.codec.config.latent_std / 0.8) * noise
        if self.in_prompt:
            self.ref = (mean[0].float().clone(), scale.reshape(-1).float().clone(), noise[0].float().clone())
        else:
            f = self.frames
            if self.teacher is not None:
                i = len(f["user_mean"])
                scale = self.t["frame.user_scale"][i].reshape(1, 1, 1).to(mean.dtype)
                noise = self.t["frame.user_noise"][i].reshape(1, 1, -1).to(mean.dtype)
                out = mean + scale * (self.model.codec.config.latent_std / 0.8) * noise
            f["user_mean"].append(mean[0, 0].float().clone())
            f["user_scale"].append(scale.reshape(-1)[0].float().clone())
            f["user_noise"].append(noise[0, 0].float().clone())
            f["user_latent"].append(out[0, 0].float().clone())
        return out

    def build_prompt(self, *a, **k):
        p = self.orig["prompt"](*a, **k) if self.teacher is None else self.teacher.prompt()
        self.prompt = p
        return p

    def head_sample(self, hidden, noise, guidance=None, steps=None):
        i = len(self.frames["hidden"])
        if self.teacher is not None:
            noise = self.t["frame.head_noise"][i : i + 1].to(noise.dtype)
        out = self.orig["head"](hidden, noise, guidance, steps)
        f = self.frames
        f["hidden"].append(hidden[0].float().clone())
        f["head_noise"].append(noise[0].float().clone())
        f["latent"].append(self.model.unscale_latents(out)[0].float().clone())
        f["layers"].append(torch.stack([self.layer[i] for i in LAYERS]))
        if not self.in_prompt:
            f["embeds"].append(self.last_embed)
        self.in_prompt = False
        if self.teacher is not None:
            return self.model.scale_latents(self.t["frame.latent"][i : i + 1]).to(out.dtype)
        return out

    def sample_token(self, logits, temperature=0.7, top_k=25, top_p=1.0, generator=None):
        cfg = self.model.config
        if self.say is not None:
            allow = self.say.allow(cfg.pad_token_id, cfg.eop_token_id)
            if allow is not None:
                masked = torch.full_like(logits, float("-inf"))
                masked[:, allow] = logits[:, allow]
                logits = masked
        values, ids = torch.topk(logits[0], TOP)
        f = self.frames
        f["top_ids"].append(ids.float())
        f["top_logits"].append(values.float())
        f["pad_eop"].append(logits[0, [cfg.pad_token_id, cfg.eop_token_id]].float().clone())
        token = self.orig["token"](logits, temperature, top_k, top_p, generator)
        if self.teacher is not None:
            i = len(f["token"])
            f["teacher_logits"].append(logits[0, self.t["frame.top_ids"][i].long()].float().clone())
            token = self.t["frame.token"][i : i + 1].long()
        f["token"].append(token.float().reshape(-1)[0])
        if self.say is not None:
            self.say.accept(int(token))
        return token


class SayQueue:
    def __init__(self, relay=True):
        self.queue = []
        self.relay = relay
        self.forced = 0

    def push(self, tokens):
        self.queue.extend(tokens)

    def allow(self, pad, eop):
        if self.queue:
            return [pad, eop, self.queue[0]]
        return [pad, eop] if self.relay else None

    def accept(self, token):
        if self.queue and self.queue[0] == token:
            self.queue.pop(0)


def session_for(model, args, recorder, card, voice):
    from haru.audio import prepare_reference
    from haru.session import DuplexConfig, DuplexSession

    reference = prepare_reference(voice) if voice is not None else None
    config = DuplexConfig(
        temperature=0.0 if args.greedy else args.temperature,
        seed=args.seed,
        speak_first=args.speak_first,
    )
    recorder.reset()
    return DuplexSession(model, card, reference, config), reference


def inputs(args):
    card = Path(args.card).read_text(encoding="utf-8").strip() if args.card else ""
    voice = read_wav(args.voice) if args.voice else None
    if args.user == "silence":
        user = np.zeros(args.frames * FRAME, dtype=np.float32)
    else:
        user = read_wav(args.user)
    count = max(1, -(-len(user) // FRAME))
    user = np.pad(user, (0, count * FRAME - len(user)))
    return card, voice, user, count


def trace(args):
    model = setup(args)
    teacher = Teacher(args.teacher, model.device) if args.teacher else None
    rec = Recorder(model, teacher=teacher)
    card, voice, user, count = inputs(args)
    session, reference = session_for(model, args, rec, card, voice)
    started = time.perf_counter()
    frames = [session.start()]
    for i in range(count - 1):
        frames.append(session.step(user[i * FRAME : (i + 1) * FRAME]))
    seconds = time.perf_counter() - started
    st = Store(args.out)
    p = rec.prompt
    st.put("prompt.tokens", p.tokens.float())
    st.put("prompt.model_latents", p.model_latents)
    st.put("prompt.user_latents", p.user_latents)
    st.put("prompt.model_mask", p.model_mask.float())
    st.put("prompt.user_mask", p.user_mask.float())
    if rec.ref is not None:
        st.put("prompt.ref_mean", rec.ref[0])
        st.put("prompt.ref_scale", rec.ref[1])
        st.put("prompt.ref_noise", rec.ref[2])
        st.put("prompt.reference", torch.as_tensor(reference))
    st.put("frame.user_audio", torch.as_tensor(user[: (count - 1) * FRAME]).reshape(count - 1, FRAME) if count > 1 else torch.zeros(0, FRAME))
    for k, v in rec.frames.items():
        if v:
            st.put(f"frame.{k}", torch.stack(v))
    audio = np.stack([f.audio for f in frames])
    st.put("frame.audio", audio)
    write_wav(Path(args.out) / "reply.wav", audio.reshape(-1))
    text = "".join(f.text for f in frames).strip()
    meta = {
        "commit": args.commit,
        "dtype": args.dtype,
        "greedy": args.greedy,
        "temperature": 0.0 if args.greedy else args.temperature,
        "seed": args.seed,
        "speak_first": args.speak_first,
        "card": args.card,
        "voice": args.voice,
        "user": args.user,
        "frames": count,
        "layers": list(LAYERS),
        "top": TOP,
        "prompt_len": int(p.length),
        "text": text,
        "seconds": seconds,
        "teacher": args.teacher,
        "torch": torch.__version__,
    }
    st.finish(meta)
    print(json.dumps({k: meta[k] for k in ("dtype", "greedy", "seed", "frames", "prompt_len", "text", "seconds")}))


def say(args):
    model = setup(args)
    queue = SayQueue()
    rec = Recorder(model, say=queue)
    card, voice, _, _ = inputs(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(FRAME, dtype=np.float32)
    silent = {model.config.pad_token_id, model.config.eop_token_id}
    report = []
    lines = [l.strip() for l in Path(args.sentences).read_text().splitlines() if l.strip()]
    for i, line in enumerate(lines):
        args.seed = args.base_seed + i
        queue.queue = []
        session, _ = session_for(model, args, rec, card, voice)
        audio = [session.start().audio]
        for _ in range(args.lead_frames):
            audio.append(session.step(silence).audio)
        ids = model.tokenizer.encode(f" {line}", add_special_tokens=False).ids
        queue.push(ids)
        said, steps, after = [], 0, 0
        while steps < args.max_frames and after < args.tail_frames:
            fr = session.step(silence)
            if fr.token not in silent:
                said.append(fr.token)
            audio.append(fr.audio)
            steps += 1
            if not queue.queue:
                after += 1
        wav = out / f"{i:02d}.wav"
        write_wav(wav, np.concatenate(audio))
        report.append({"index": i, "text": line, "wav": str(wav), "queued": len(ids), "spoken_tokens": len(said), "unsaid": len(queue.queue), "said": model.tokenizer.decode(said), "frames": steps + args.lead_frames + 1})
        print(f"{i:02d}: {len(ids)} tokens queued, {len(queue.queue)} unsaid after {steps} frames", flush=True)
    (out / "say.json").write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("trace", "say"):
        p = sub.add_parser(name)
        p.add_argument("--haru", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--commit", default="7fe6b766")
        p.add_argument("--out", required=True)
        p.add_argument("--dtype", choices=("f32", "bf16"), default="f32")
        p.add_argument("--card")
        p.add_argument("--voice")
        p.add_argument("--seed", type=int, default=1489)
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--greedy", action="store_true")
        p.add_argument("--speak-first", action="store_true")
        p.add_argument("--user", default="silence")
        p.add_argument("--frames", type=int, default=149)
        p.add_argument("--teacher")
    s = sub.choices["say"]
    s.add_argument("--sentences", required=True)
    s.add_argument("--base-seed", type=int, default=0)
    s.add_argument("--lead-frames", type=int, default=8)
    s.add_argument("--tail-frames", type=int, default=12)
    s.add_argument("--max-frames", type=int, default=600)
    args = parser.parse_args()
    with torch.no_grad():
        {"trace": trace, "say": say}[args.cmd](args)


if __name__ == "__main__":
    main()
