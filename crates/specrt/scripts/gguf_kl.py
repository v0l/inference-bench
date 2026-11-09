import sys
import numpy as np

V = 151936


def load(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, V).astype(np.float64)


def logsm(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


a, b = load(sys.argv[1]), load(sys.argv[2])
p, q = logsm(a), logsm(b)
kl = (np.exp(p) * (p - q)).sum(axis=1)
top = (a.argmax(1) == b.argmax(1)).sum()
print(f"mean KL {kl.mean():.3e} max KL {kl.max():.3e} top1 {top}/{len(kl)}")
print(" ".join(f"{k:.1e}" for k in kl))
