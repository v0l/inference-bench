import sys
from cuda.bindings import nvrtc

src = open(sys.argv[1], "rb").read()
arch = sys.argv[2]
_, prog = nvrtc.nvrtcCreateProgram(src, b"m.cu", 0, [], [])
r = nvrtc.nvrtcCompileProgram(prog, 2, [f"--gpu-architecture={arch}".encode(), b"--std=c++17"])
_, n = nvrtc.nvrtcGetProgramLogSize(prog)
log = b" " * n
nvrtc.nvrtcGetProgramLog(prog, log)
sys.stderr.write("\n".join(l for l in log.decode(errors="replace").splitlines() if "error" in l) + "\n")
_, n = nvrtc.nvrtcGetPTXSize(prog)
buf = b" " * n
nvrtc.nvrtcGetPTX(prog, buf)
sys.stdout.buffer.write(buf.rstrip(b"\0"))
