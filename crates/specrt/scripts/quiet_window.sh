#!/usr/bin/env bash
set -uo pipefail
if [ $# -eq 0 ]; then
    echo "usage: quiet_window.sh <command> [args...]" >&2
    exit 2
fi
unit=${QUIET_WINDOW_UNIT:-local-llm.service}
port=${QUIET_WINDOW_PORT:-8001}
lock=${QUIET_WINDOW_LOCK:-/tmp/gpu27.lock}
wait_s=${QUIET_WINDOW_WAIT_S:-900}
max_s=${QUIET_WINDOW_MAX_S:-3000}
log() { echo "[quiet_window $(date +%T)] $*" >&2; }

exec 9>"$lock"
flock 9
export GPU27_LOCK_HELD=1

child=0
restarted=0
restart() {
    [ "$restarted" = 1 ] && return
    restarted=1
    if [ "$child" != 0 ] && kill -0 "$child" 2>/dev/null; then
        kill "$child" 2>/dev/null
        wait "$child" 2>/dev/null
    fi
    log "starting $unit"
    sudo systemctl start "$unit"
    local waited=0
    until curl -sf -m3 "localhost:$port/v1/models" >/dev/null; do
        if [ "$waited" -ge "$wait_s" ]; then
            log "$unit did not answer on $port within ${wait_s}s"
            return
        fi
        sleep 15
        waited=$((waited + 15))
    done
    local model
    model=$(curl -s -m5 "localhost:$port/v1/models" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    log "answering on $port after ${waited}s, chat request to $model:"
    curl -s -m120 "localhost:$port/v1/chat/completions" -H 'content-type: application/json' \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":8}" | head -c 300 >&2
    echo >&2
}
on_signal() {
    log "signal, restarting"
    restart
    exit 143
}
trap restart EXIT
trap on_signal INT TERM HUP

log "stopping $unit"
sudo systemctl stop "$unit"
sleep 5
log "gpu: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"

timeout --signal=TERM --kill-after=30 "$max_s" "$@" &
child=$!
wait "$child"
rc=$?
child=0
[ "$rc" = 124 ] && log "command cut at the ${max_s}s timeout"
log "command exited $rc"
exit "$rc"
