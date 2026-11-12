#!/usr/bin/env bash
if [ -n "${GPU27_LOCK_HELD:-}" ]; then
    exec "$@"
fi
exec flock "${QUIET_WINDOW_LOCK:-/tmp/gpu27.lock}" "$@"
