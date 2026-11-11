import sys
import torch

shapes = {"gate_up": (34816, 5120), "down": (5120, 17408), "lin_in": (16384, 5120)}
tokens = [int(t) for t in (sys.argv[1] if len(sys.argv) > 1 else "2048,16384").split(",")]
dev = "cuda"


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    return times[len(times) // 2]


for m in tokens:
    for name, (n, k) in shapes.items():
        x = torch.randn(m, k, device=dev).to(torch.float8_e4m3fn)
        w = torch.randn(n, k, device=dev).to(torch.float8_e4m3fn)
        sa = torch.rand(m, 1, device=dev)
        sb = torch.rand(1, n, device=dev)
        fp8 = timed(lambda: torch._scaled_mm(x, w.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
        xb, wb = x.to(torch.bfloat16), w.to(torch.bfloat16)
        bf = timed(lambda: xb @ wb.t())
        flop = 2 * m * n * k
        print(f"{name} m={m} n={n} k={k}: fp8 {fp8:.3f} ms {flop / fp8 / 1e9:.0f} TFLOP/s, bf16 {bf:.3f} ms {flop / bf / 1e9:.0f} TFLOP/s")
        del x, w, xb, wb
