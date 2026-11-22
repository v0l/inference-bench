#!/usr/bin/env bash
set -euo pipefail

if [ "$(uname)" != "Darwin" ]; then
  here=$(cd "$(dirname "$0")/../../.." && pwd)
  rsync -az --delete --exclude target --exclude .git "$here/" "${MAC:-mac}:git/ib-bench/"
  ssh "${MAC:-mac}" "SPECRT_REV=$(git -C "$here" rev-parse --short HEAD) bash ~/git/ib-bench/crates/specrt/scripts/mac_vs_llama.sh $*"
  exit
fi

source "$HOME/.cargo/env"
export PATH=/opt/homebrew/bin:$PATH
ROUNDS=${ROUNDS:-3}
RUNS=${RUNS:-3}
MODELS=${MODELS-"Qwen3-0.6B-BF16 Qwen3-0.6B-Q4_K_M Llama-3.2-1B-Instruct-Q4_K_M"}
LENS=${LENS:-128,1024,4096}
BIG=${BIG:-Qwen2.5-7B-Instruct-Q4_K_M}
OUT=${OUT:-$HOME/bench-out/$(date +%Y%m%d-%H%M)}
SRC=$HOME/git/ib-bench
LLAMA=$HOME/git/llama.cpp
mkdir -p "$OUT" "$HOME/models"

if [ ! -d "$LLAMA" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA"
fi
(cd "$LLAMA" && cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release >/dev/null &&
  cmake --build build -j 6 --target llama-bench llama-cli llama-completion >/dev/null)
(cd "$SRC" && cargo build --release -p inference-bench-specrt --features wgpu --bin bench)

{
  echo "date $(date -u +%FT%TZ)"
  echo "llama.cpp $(git -C "$LLAMA" rev-parse --short HEAD)"
  echo "specrt ${SPECRT_REV:-unknown}"
  sysctl -n machdep.cpu.brand_string
} > "$OUT/env.txt"

load_note() {
  echo "uptime: $(uptime)"
  if tail -n 40 /var/log/gcad-sim-worker.log 2>/dev/null | grep -qiE "claimed|running job|started job"; then
    echo "gcad sim: possible job in the last 40 log lines"
  else
    echo "gcad sim: no job in the last 40 log lines"
  fi
  echo "swap: $(sysctl -n vm.swapusage)"
}

cold() {
  if [ "${PURGE:-1}" = 1 ]; then
    sudo -n purge 2>/dev/null || true
  fi
}

run_specrt() {
  cold
  local m=$1 tag=$2 runs=$3 lens=$4
  /usr/bin/time -l "$SRC/target/release/bench" --wgpu --runs "$runs" "$HOME/models/$m.gguf" "$lens" 128 > "$OUT/$tag.specrt.txt" 2>&1
}

run_llama() {
  cold
  local m=$1 tag=$2 lens=$3
  /usr/bin/time -l "$LLAMA/build/bin/llama-bench" -m "$HOME/models/$m.gguf" -ngl 99 -p "$lens" -n 128 -r 3 -o json > "$OUT/$tag.llama.txt" 2>&1
  /usr/bin/time -l "$LLAMA/build/bin/llama-bench" -m "$HOME/models/$m.gguf" -ngl 99 -p 1 -n 0 -r 1 -o json > "$OUT/$tag.llama-load.txt" 2>&1
}

for round in $(seq 1 "$ROUNDS"); do
  load_note > "$OUT/round$round.load.txt"
  for m in $MODELS; do
    tag=round$round.$m
    if [ $((round % 2)) -eq 1 ]; then
      run_specrt "$m" "$tag" "$RUNS" $LENS
      run_llama "$m" "$tag" $LENS
    else
      run_llama "$m" "$tag" $LENS
      run_specrt "$m" "$tag" "$RUNS" $LENS
    fi
  done
done

if [ -n "$BIG" ] && [ -f "$HOME/models/$BIG.gguf" ]; then
  for round in $(seq 1 "$ROUNDS"); do
    load_note > "$OUT/big$round.load.txt"
    tag=big$round.$BIG
    if [ $((round % 2)) -eq 1 ]; then
      run_specrt "$BIG" "$tag" 1 $LENS
      run_llama "$BIG" "$tag" $LENS
    else
      run_llama "$BIG" "$tag" $LENS
      run_specrt "$BIG" "$tag" 1 $LENS
    fi
  done
fi

python3 - "$OUT" <<'PY'
import sys, glob, json, re, statistics, os
out = sys.argv[1]
rows = {}
for f in sorted(glob.glob(out + "/round*.llama.txt") + glob.glob(out + "/big*.llama.txt")):
    base = os.path.basename(f)[:-len(".llama.txt")]
    text = open(f).read()
    start = text.index("[")
    data = json.JSONDecoder().raw_decode(text[start:])[0]
    for d in data:
        key = (base.split(".", 1)[0], base.split(".", 1)[1], "llama", f"pp{d['n_prompt']}" if d["n_prompt"] else f"tg{d['n_gen']}")
        rows.setdefault(key, []).append(d["avg_ts"])
    peak = re.search(r"(\d+)\s+peak memory footprint", text)
    if peak:
        rows.setdefault((base.split(".", 1)[0], base.split(".", 1)[1], "llama", "peak_gb"), []).append(int(peak.group(1)) / 1e9)
for f in sorted(glob.glob(out + "/*.llama-load.txt")):
    base = os.path.basename(f)[:-len(".llama-load.txt")]
    real = re.search(r"([\d.]+)\s+real", open(f).read())
    if real:
        rows.setdefault((base.split(".", 1)[0], base.split(".", 1)[1], "llama", "load_s"), []).append(float(real[1]))
for f in sorted(glob.glob(out + "/*.specrt.txt")):
    base = os.path.basename(f)[:-len(".specrt.txt")]
    text = open(f).read()
    for m in re.finditer(r"prompt=(\d+) prefill=([\d.]+) tok/s decode=([\d.]+) tok/s", text):
        r, name = base.split(".", 1)
        rows.setdefault((r, name, "specrt", f"pp{m[1]}"), []).append(float(m[2]))
        rows.setdefault((r, name, "specrt", "tg128"), []).append(float(m[3]))
    load = re.search(r"load=([\d.]+)s", text)
    if load:
        r, name = base.split(".", 1)
        rows.setdefault((r, name, "specrt", "load_s"), []).append(float(load[1]))
    peak = re.search(r"(\d+)\s+peak memory footprint", text)
    if peak:
        r, name = base.split(".", 1)
        rows.setdefault((r, name, "specrt", "peak_gb"), []).append(int(peak[1]) / 1e9)
summary = {}
for (r, name, eng, metric), v in rows.items():
    summary.setdefault((name, eng, metric), []).append(statistics.median(v))
print(f"{'model':36} {'metric':8} {'specrt':>9} {'llama.cpp':>10} {'ratio':>7}")
names = sorted({k[0] for k in summary})
for name in names:
    for metric in ["pp128", "pp1024", "pp4096", "tg128", "load_s", "peak_gb"]:
        a = summary.get((name, "specrt", metric))
        b = summary.get((name, "llama", metric))
        if not a and not b:
            continue
        sa = statistics.median(a) if a else float("nan")
        sb = statistics.median(b) if b else float("nan")
        print(f"{name:36} {metric:8} {sa:9.2f} {sb:10.2f} {sa / sb:7.3f}")
PY
echo "raw output in $OUT"
