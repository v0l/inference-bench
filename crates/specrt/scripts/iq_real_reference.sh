#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
file=${1:?usage: iq_real_reference.sh <model.gguf> <out dir> [tokens, default 32]}
out=${2:?out dir}
n_gen=${3:-32}
llama=${LLAMA_DIR:-$HOME/git/llama.cpp}
export LD_LIBRARY_PATH=$llama/build/bin
mkdir -p "$out"
gcc -O2 "$here/gguf_reference.c" -I"$llama/include" -I"$llama/ggml/include" -L"$llama/build/bin" -lllama -lggml -lggml-base -lm -Wl,-rpath,"$llama/build/bin" -o "$out/gguf_reference"
n=0
while IFS= read -r text; do
    ids=$("$llama/build/bin/llama-tokenize" -m "$file" --ids --no-bos --log-disable -p "$text" | tr -d '[],')
    "$out/gguf_reference" "$file" "$out/p$n" 99 "$n_gen" $ids >/dev/null
    echo "$ids" > "$out/p$n.prompt.txt"
    n=$((n + 1))
done < "$root/tests/fixtures/iq_prompts.txt"
echo "wrote $n prompts to $out"
