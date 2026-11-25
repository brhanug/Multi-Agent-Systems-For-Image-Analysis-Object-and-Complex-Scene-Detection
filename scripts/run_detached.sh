#!/bin/bash
# Start any experiment so that it keeps running after the Claude Code session, the terminal or the
# SSH connection ends. It is detached from the session (setsid + nohup), logs everything, and leaves
# its exit status behind, so the next session can see whether it finished and how.
#
#   bash scripts/run_detached.sh <name> <command...>
#   bash scripts/run_detached.sh kosmos_rerun .venv/bin/python3 scripts/signal_rebuild/run_kosmos_full_corpus.py
#
# Afterwards, from any session:
#   bash scripts/run_detached.sh --status            # every job: running, finished (exit 0) or failed
#   tail -f runs_detached/<name>/output.log
set -u
BASE=/data/brhanu/thesis_project
DIR=$BASE/runs_detached
mkdir -p "$DIR"

if [ "${1:-}" = "--status" ]; then
  for d in "$DIR"/*/; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    if [ -f "$d/exit_code" ]; then
      c=$(cat "$d/exit_code"); s=$([ "$c" = 0 ] && echo "finished" || echo "FAILED (exit $c)")
    elif [ -f "$d/pid" ] && kill -0 "$(cat "$d/pid")" 2>/dev/null; then s="running (pid $(cat "$d/pid"))"
    else s="stopped without an exit code (machine restart or killed)"; fi
    printf "%-28s %-40s started %s\n" "$n" "$s" "$(cat "$d/started" 2>/dev/null)"
  done
  exit 0
fi

[ $# -ge 2 ] || { echo "usage: $0 <name> <command...>   |   $0 --status"; exit 2; }
name=$1; shift
J=$DIR/$name
if [ -f "$J/pid" ] && kill -0 "$(cat "$J/pid")" 2>/dev/null; then
  echo "$name is already running (pid $(cat "$J/pid"))"; exit 1
fi
mkdir -p "$J"; rm -f "$J/exit_code"
printf '%q ' "$@" > "$J/command"; date '+%F %T' > "$J/started"
cd "$BASE"
setsid nohup bash -c '"$@" > "$0/output.log" 2>&1; echo $? > "$0/exit_code"; date "+%F %T" > "$0/ended"' "$J" "$@" \
  < /dev/null > /dev/null 2>&1 &
echo $! > "$J/pid"
echo "started $name (pid $!), detached from this session; log: runs_detached/$name/output.log"
