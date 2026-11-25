import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "tests/fixtures/conv-cases"
GOLDEN = np.uint64(0x9E3779B97F4A7C15)

CASES = [
    ("kl_post_quant", 4, 4, 1, 1, False, False, 1, 16, 16),
    ("kl_conv_in", 4, 32, 3, 1, False, False, 1, 16, 16),
    ("kl_resnet", 32, 32, 3, 1, False, False, 1, 16, 16),
    ("kl_up", 32, 32, 3, 1, True, False, 1, 16, 16),
    ("kl_narrow", 32, 16, 3, 1, False, False, 1, 32, 32),
    ("kl_shortcut", 32, 16, 1, 1, False, False, 1, 32, 32),
    ("kl_resnet16", 16, 16, 3, 1, False, False, 1, 32, 32),
    ("kl_out", 16, 3, 3, 1, False, False, 1, 32, 32),
    ("kl_qkv", 32, 96, 1, 1, False, False, 1, 16, 16),
    ("kl_proj", 32, 32, 1, 1, False, False, 1, 16, 16),
    ("wan_post_quant", 4, 4, 1, 1, False, False, 3, 8, 8),
    ("wan_conv_in", 4, 16, 3, 3, False, False, 3, 8, 8),
    ("wan_resnet", 16, 16, 3, 3, False, False, 3, 8, 8),
    ("wan_qkv", 16, 48, 1, 1, False, False, 3, 8, 8),
    ("wan_proj", 16, 16, 1, 1, False, False, 3, 8, 8),
    ("wan_time_up", 16, 32, 1, 3, False, True, 3, 8, 8),
    ("wan_up", 16, 8, 3, 1, True, False, 5, 8, 8),
    ("wan_widen", 8, 16, 3, 3, False, False, 5, 16, 16),
    ("wan_shortcut", 8, 16, 1, 1, False, False, 5, 16, 16),
    ("wan_up2", 16, 8, 3, 1, True, False, 5, 16, 16),
    ("wan_resnet8", 8, 8, 3, 3, False, False, 5, 32, 32),
    ("wan_out", 8, 3, 3, 3, False, False, 5, 32, 32),
    ("big_resnet", 512, 512, 3, 1, False, False, 1, 6, 10),
    ("big_conv_in", 32, 512, 3, 1, False, False, 1, 6, 10),
    ("big_shortcut", 256, 128, 1, 1, False, False, 1, 6, 10),
    ("big_out", 128, 3, 3, 1, False, False, 1, 12, 20),
    ("big_up", 512, 512, 3, 1, True, False, 1, 3, 5),
    ("big_qkv", 512, 1536, 1, 1, False, False, 1, 4, 4),
    ("big3d_in", 16, 384, 3, 3, False, False, 2, 4, 6),
    ("big3d_resnet", 384, 384, 3, 3, False, False, 2, 4, 6),
    ("big3d_time_up", 384, 768, 1, 3, False, True, 2, 4, 6),
    ("big3d_up", 384, 192, 3, 1, True, False, 1, 2, 3),
    ("big3d_out", 96, 3, 3, 3, False, False, 2, 4, 6),
]


def uniform(seed, n, scale):
    i = np.arange(1, n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + i * GOLDEN
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))
    return ((2.0 * u - 1.0) * scale).astype(np.float32)


def conv(x, w, b, k, kt, up):
    if up:
        x = x.repeat(2, axis=2).repeat(2, axis=3)
    c, t, h, wd = x.shape
    pad = k // 2
    xp = np.zeros((c, t + kt - 1, h + 2 * pad, wd + 2 * pad), dtype=np.float64)
    xp[:, kt - 1 :, pad : pad + h, pad : pad + wd] = x
    cout = w.shape[0]
    y = np.zeros((cout, t, h, wd), dtype=np.float64)
    for dt in range(kt):
        for dy in range(k):
            for dx in range(k):
                patch = xp[:, dt : dt + t, dy : dy + h, dx : dx + wd]
                y += np.einsum("oc,cthw->othw", w[:, :, dt, dy, dx].astype(np.float64), patch)
    return y + b.astype(np.float64)[:, None, None, None]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = {"generator": "splitmix64", "cases": []}
    for n, (name, cin, cout, k, kt, up, interleave, t, h, w) in enumerate(CASES):
        seed = 1000 + 17 * n
        x = uniform(seed, cin * t * h * w, 1.0).reshape(cin, t, h, w)
        fan = cin * k * k * kt
        wt = uniform(seed + 1, cout * fan, 1.0 / np.sqrt(fan)).reshape(cout, cin, kt, k, k)
        b = uniform(seed + 2, cout, 0.1)
        y = conv(x.astype(np.float64), wt, b, k, kt, up)
        if interleave:
            half = cout // 2
            y = np.stack((y[:half], y[half:]), axis=2).reshape(half, 2 * t, *y.shape[2:])
        y.astype(np.float32).tofile(OUT / f"{name}.bin")
        index["cases"].append(
            {"name": name, "seed": seed, "cin": cin, "cout": cout, "k": k, "kt": kt, "upsample": up, "interleave": interleave, "frames": t, "height": h, "width": w, "out": list(y.shape)}
        )
    (OUT / "cases.json").write_text(json.dumps(index, indent=1) + "\n")
    size = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"{len(CASES)} cases, {size} bytes")


if __name__ == "__main__":
    sys.exit(main())
