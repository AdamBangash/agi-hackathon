#!/usr/bin/env bash
# Mirrors this project to the G1 (laptop is source of truth). Polls every 2s.
SRC="$(cd "$(dirname "$0")" && pwd)/"
DST=unitree@192.168.123.164:hackAGI/
ssh -o BatchMode=yes unitree@192.168.123.164 'mkdir -p ~/hackAGI'
last=""
while true; do
  cur=$(find "$SRC" -path "$SRC.git" -prune -o -printf '%P %T@ %s\n' | sort | md5sum)
  if [ "$cur" != "$last" ]; then
    rsync -az --delete --exclude .git --exclude __pycache__ "$SRC" "$DST" && echo "$(date +%T) synced"
    last=$cur
  fi
  [ "${1:-}" = "--once" ] && break
  sleep 2
done
