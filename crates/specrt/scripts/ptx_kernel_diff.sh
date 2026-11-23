#!/usr/bin/env bash
set -euo pipefail
old=$1
new=$2
split() {
    awk -v out="$2" '
        /^\.(visible|entry|func)|^\.visible \.entry|^\.entry/ { name=$0; sub(/.*(entry|func)[ \t]+/, "", name); sub(/\(.*/, "", name); file=out "/" name }
        file { print > file }
    ' "$1"
}
for f in "$old"/*.ptx; do
    b=$(basename "$f" .ptx)
    rm -rf "/tmp/ptxsplit/$b"
    mkdir -p "/tmp/ptxsplit/$b/old" "/tmp/ptxsplit/$b/new"
    split "$f" "/tmp/ptxsplit/$b/old"
    split "$new/$b.ptx" "/tmp/ptxsplit/$b/new"
    changed=$( (diff -rq "/tmp/ptxsplit/$b/old" "/tmp/ptxsplit/$b/new" || true) | sed -E 's#.*/(old|new)/([^ ]+).*#\2#' | sort -u | tr '\n' ' ')
    echo "$b: ${changed:-none}"
done
