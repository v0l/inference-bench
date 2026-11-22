#!/usr/bin/env python3
"""Host-only replay of recorded agent turns through a prompt-lookup drafter.

Usage: lookup_sim.py TOKENIZER_JSON [--turns 50] [--mtp 2.43] [--ngram 2,4] [--max-drafts 5]

Each turn is one assistant message from ~/.pi/agent/sessions (text, thinking text and
tool call arguments) with everything before it in the session as the context. The
output tokens are treated as the target's greedy tokens. Per step the lookup proposes
drafts from the context plus the output so far; it takes the step when it proposes more
than MTP's expected accepted length (mtp - 1), otherwise the step is an MTP step that
advances by the measured MTP mean.
"""
import argparse, glob, json, os, random
from tokenizers import Tokenizer


def render(m):
    c = m.get("content")
    if isinstance(c, str):
        return c
    out = []
    for b in c or []:
        t = b.get("type")
        if t == "text":
            out.append(b["text"])
        elif t == "toolCall":
            out.append(json.dumps(b.get("arguments"), ensure_ascii=False))
    return "\n".join(out)


def turns(root, want, seed):
    files = sorted(glob.glob(os.path.join(root, "**/*.jsonl"), recursive=True))
    random.Random(seed).shuffle(files)
    found = []
    for f in files:
        if len(found) >= want * 4:
            break
        hist = []
        picks = []
        for line in open(f):
            try:
                j = json.loads(line)
            except ValueError:
                continue
            m = j.get("message")
            if not m or m["role"] == "system":
                continue
            text = render(m)
            if m["role"] == "assistant" and len(text) > 200 and hist:
                picks.append((list(hist[-40:]), text))
            hist.append(text)
        random.Random(seed).shuffle(picks)
        found.extend(picks[:2])
    random.Random(seed).shuffle(found)
    return found[:want]


class Index:
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi
        self.toks = []
        self.last = {n: {} for n in range(lo, hi + 1)}

    def push(self, t):
        self.toks.append(t)
        i = len(self.toks)
        for n in range(self.lo, self.hi + 1):
            if i > n:
                self.last[n][tuple(self.toks[i - n - 1:i - 1])] = i - 1

    def propose(self, k):
        i = len(self.toks)
        for n in range(self.hi, self.lo - 1, -1):
            if i < n:
                continue
            pos = self.last[n].get(tuple(self.toks[i - n:i]))
            if pos is not None:
                return self.toks[pos:pos + k]
        return []


def replay(ctx, out, lo, hi, kmax, mtp):
    idx = Index(lo, hi)
    for t in ctx:
        idx.push(t)
    p = 0
    steps = lsteps = acc_total = 0
    credit = 0.0
    while p < len(out):
        d = idx.propose(kmax)
        if len(d) > mtp - 1:
            a = 0
            while a < len(d) and p + a < len(out) and d[a] == out[p + a]:
                a += 1
            adv = min(1 + a, len(out) - p)
            lsteps += 1
            acc_total += a
        else:
            credit += mtp
            adv = min(int(credit), len(out) - p)
            adv = max(adv, 1)
            credit -= adv
        for t in out[p:p + adv]:
            idx.push(t)
        p += adv
        steps += 1
    return len(out), steps, lsteps, acc_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("--turns", type=int, default=50)
    ap.add_argument("--mtp", type=float, default=2.43)
    ap.add_argument("--ngram", default="2,4")
    ap.add_argument("--max-drafts", type=int, default=5)
    ap.add_argument("--root", default=os.path.expanduser("~/.pi/agent/sessions"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    lo, hi = map(int, a.ngram.split(","))
    tok = Tokenizer.from_file(a.tokenizer)
    tot_tok = tot_steps = tot_l = tot_acc = 0
    n = 0
    for hist, text in turns(a.root, a.turns, a.seed):
        ctx = tok.encode("\n".join(hist), add_special_tokens=False).ids[-30000:]
        out = tok.encode(text, add_special_tokens=False).ids
        t, s, ls, ac = replay(ctx, out, lo, hi, a.max_drafts, a.mtp)
        tot_tok += t; tot_steps += s; tot_l += ls; tot_acc += ac
        n += 1
    tps = tot_tok / tot_steps
    print(f"turns {n} tokens {tot_tok} steps {tot_steps} lookup_steps {tot_l} "
          f"lookup_accepted_per_lookup_step {tot_acc / max(tot_l, 1):.2f}")
    print(f"MTP alone {a.mtp:.2f} tok/step, MTP+lookup {tps:.2f} tok/step, gain {tps - a.mtp:+.2f}")


if __name__ == "__main__":
    main()
