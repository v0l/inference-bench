#!/usr/bin/env bash
set -euo pipefail
url=${1:-http://127.0.0.1:8013}
model=${2:-glm-5.3-flash}
out=${3:-docs/glm5-real/specrt}
mkdir -p "$out"
prompts=(
  "The capital of France is"
  "Write a short Python function that returns the n-th Fibonacci number.\n\ndef fib(n):"
  "Explain in two sentences why the sky is blue."
  "Q: A train leaves at 3 pm and travels 120 km at 60 km/h. When does it arrive?\nA:"
)
for i in "${!prompts[@]}"; do
  body=$(jq -n --arg m "$model" --arg p "$(printf '%b' "${prompts[$i]}")" '{model:$m,prompt:$p,max_tokens:64,temperature:0,logprobs:20,ignore_eos:true}')
  start=$(date +%s.%N)
  curl -s "$url/v1/completions" -H 'content-type: application/json' -d "$body" > "$out/p$i.json"
  end=$(date +%s.%N)
  printf 'p%d %.2fs %s\n' "$i" "$(echo "$end - $start" | bc)" "$(jq -c '[.usage.prompt_tokens,.usage.completion_tokens]' "$out/p$i.json")"
done
