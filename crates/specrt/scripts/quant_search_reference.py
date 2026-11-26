import json
import sys

import numpy as np

NVFP4_STEPS = [0, -1, -2, -3, -4, 1]
FP8_FACTORS = [1.0, 0.96875, 0.9375, 0.90625, 0.875]
GROUP = 16
F = np.float32


def e4m3_table():
    out = []
    for code in range(0x7F):
        e, m = code >> 3, code & 7
        out.append(F(m) * F(2.0 ** -9) if e == 0 else F(1.0 + m / 8.0) * F(2.0 ** (e - 7)))
    return np.array(out, dtype=F)


E4M3 = e4m3_table()
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=F)


def e4m3_code(v):
    a = abs(F(v))
    d = np.abs(E4M3.astype(np.float64) - float(a))
    best = int(np.argmin(d))
    ties = np.flatnonzero(d == d[best])
    if len(ties) > 1:
        best = int(next(t for t in ties if t % 2 == 0))
    if a >= E4M3[0x7E]:
        best = 0x7E
    return best | (0x80 if np.signbit(F(v)) else 0)


def e4m3_value(code):
    v = E4M3[code & 0x7F]
    return -v if code & 0x80 else v


def e2m1_code(v):
    a = abs(F(v))
    for i, (lim, inclusive) in enumerate([(0.25, True), (0.75, False), (1.25, True), (1.75, False), (2.5, True), (3.5, False), (5.0, True)]):
        if (a <= lim) if inclusive else (a < lim):
            index = i
            break
    else:
        index = 7
    return index | (8 if np.signbit(F(v)) and index != 0 else 0)


def e2m1_value(n):
    v = E2M1[n & 7]
    return -v if n & 8 else v


def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def nvfp4_error(group, code, glob):
    s = F(e4m3_value(code)) / F(glob)
    err = 0.0
    for w in group:
        q = e2m1_code(F(w) / s) if s > 0 else 0
        err += (float(w) - float(F(e2m1_value(q)) * s)) ** 2
    return err


def nvfp4(w, cols):
    amax = F(np.max(np.abs(w)))
    glob = F(1.0) if amax == 0 else F(F(448.0 * 6.0) / amax)
    rows = w.size // cols
    codes, packed = [], []
    for r in range(rows):
        row = w[r * cols:(r + 1) * cols]
        nibbles = []
        for g in range(cols // GROUP):
            group = row[g * GROUP:(g + 1) * GROUP]
            gmax = F(np.max(np.abs(group)))
            base = e4m3_code(min(F(gmax / F(6.0) * glob), F(448.0)))
            best, best_err = base, nvfp4_error(group, base, glob)
            if gmax > 0:
                for step in NVFP4_STEPS[1:]:
                    c = base + step
                    if c < 1 or c > 0x7E:
                        continue
                    e = nvfp4_error(group, c, glob)
                    if e < best_err:
                        best, best_err = c, e
            codes.append(best)
            s = F(e4m3_value(best)) / glob
            nibbles.extend(e2m1_code(F(v) / s) if s > 0 else 0 for v in group)
        packed.extend(nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, cols, 2))
    return {"global": float(glob), "scales": codes, "packed": packed}


def fp8_error(row, s):
    return sum((float(v) - float(F(e4m3_value(e4m3_code(F(v) / s))) * s)) ** 2 for v in row)


def fp8_channel(w, cols):
    rows = w.size // cols
    scales, values = [], []
    for r in range(rows):
        row = w[r * cols:(r + 1) * cols]
        amax = F(np.max(np.abs(row)))
        if amax == 0:
            best = F(1.0)
        else:
            best, best_err = None, None
            for f in FP8_FACTORS:
                s = F(amax / F(448.0) * F(f))
                e = fp8_error(row, s)
                if best is None or e < best_err:
                    best, best_err = s, e
        scales.append(float(best))
        values.extend(e4m3_code(F(v) / best) for v in row)
    return {"scales": scales, "values": values}


def tensor(rows, cols, seed):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((rows, cols)).astype(F) * F(0.02)
    w[:, 5] *= F(9.0)
    w[3, 17] = F(0.4)
    w[6, :] = F(0.0)
    bits = (w.view(np.uint32) + np.uint32(0x7FFF) + ((w.view(np.uint32) >> 16) & 1)) >> 16
    return bits.astype(np.uint16).ravel()


def main(out):
    rows, cols = 8, 64
    raw = tensor(rows, cols, 1417)
    w = bf16_to_f32(raw)
    case = {"rows": rows, "cols": cols, "bf16": [int(v) for v in raw], "nvfp4_g16": nvfp4(w, cols), "fp8_channel": fp8_channel(w, cols)}
    with open(out, "w") as f:
        json.dump(case, f)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
