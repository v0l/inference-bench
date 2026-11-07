#!/bin/bash
IB=~/git/ib272; BIN=$IB/target/release; LL=~/git/llama.cpp/build/bin; GG=~/git/specrt/out/gguf
OUT=/tmp/ib272out; mkdir -p $OUT; rm -f $OUT/*
export PATH=$HOME/.venv-vllm/bin:/usr/local/cuda/bin:$PATH
declare -A HF=([qwen3]=Qwen/Qwen3-0.6B [qwen25]=Qwen/Qwen2.5-0.5B)
declare -A GGUF=([qwen3]=Qwen3-0.6B-bf16.gguf [qwen25]=Qwen2.5-0.5B-bf16.gguf)
declare -A SPEC=([qwen3]=qwen3-0.6b [qwen25]=qwen2.5-0.5b)
drop() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }
for m in qwen3 qwen25; do
  id=${HF[$m]}; drop
  HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 nohup vllm serve $id --served-model-name $id --host 127.0.0.1 --port 18000 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.08 --max-num-seqs 16 --no-enable-prefix-caching > $OUT/$m-vllm.log 2>&1 &
  vpid=$!
  for _ in $(seq 150); do curl -sf localhost:18000/v1/models >/dev/null && break; sleep 2; done
  $BIN/inference-bench batch --backend openai --url http://127.0.0.1:18000 --label vllm --model $id --sizes 4,16 --json > /dev/null
  engines=(vllm llama base new)
  for r in 1 2 3 4 5 6; do
    for k in 0 1 2 3; do
      e=${engines[$(((r + k) % 4))]}
      drop
      case $e in
        vllm) $BIN/inference-bench batch --backend openai --url http://127.0.0.1:18000 --label vllm --model $id --sizes 4,16 --json > $OUT/$m-vllm-$r.json;;
        llama) $LL/llama-batched-bench -m $GG/${GGUF[$m]} -ngl 99 -c 8192 -npp 128 -ntg 128 -npl 4,16 --output-format jsonl 2>/dev/null | grep '^{' > $OUT/$m-llama-$r.jsonl;;
        base) (cd ~/git/ib272base/crates/specrt; for n in 4 16; do ~/git/ib272base/target/release/bench specs/${SPEC[$m]}.toml 128 128 $n; done) > $OUT/$m-base-$r.txt 2>/dev/null;;
        new) (cd $IB/crates/specrt; for n in 4 16; do $BIN/bench specs/${SPEC[$m]}.toml 128 128 $n; done) > $OUT/$m-new-$r.txt 2>/dev/null;;
      esac
    done
  done
  kill $vpid; wait $vpid 2>/dev/null; sleep 5
done
echo finished > $OUT/done
