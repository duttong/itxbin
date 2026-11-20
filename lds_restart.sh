#!/bin/bash
# Restart logos_data_service.py (the service behind omi hats/logos_data.php) on omi, from catsdata.
#
#   lds_restart.sh            stop the running service, start a fresh one, wait until it answers a ping
#   lds_restart.sh status     only ping it (and show the process)
#   lds_restart.sh log [N]    the last N (default 40) lines of its log
#
# Run it after deploying itxbin: the service keeps the Python code it started with until it is restarted.
# Link it into ~/bin on catsdata:  ln -s /hats/gc/itxbin/lds_restart.sh ~/bin/lds_restart.sh

set -u

HOST=${LDS_HOST:-omi}
DIR=/nfs/hats/gc/itxbin
PY=/nfs/ccg/python/miniconda3/envs/prod6/bin/python3
SOCK=/dev/shm/logos_data.sock
LOG=/dev/shm/logos_data_service.log

cmd=${1:-restart}
case "$cmd" in
    restart|status|log) ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "usage: $(basename "$0") [restart|status|log [N]]" >&2; exit 2 ;;
esac

# Everything runs in one ssh session: a script on bash's stdin, so no command line on omi carries the
# service's name (a pkill -f on it would otherwise match, and kill, the remote shell itself).
ssh -T "$HOST" bash -s -- "$cmd" "${2:-40}" "$DIR" "$PY" "$SOCK" "$LOG" <<'REMOTE'
cmd=$1; n=$2; DIR=$3; PY=$4; SOCK=$5; LOG=$6
PAT='[p]ython3 logos_data_service.py'      # [p]: matches the service, not this pattern's own text

ping_svc() {      # prints the reply; fails if the service does not answer
    "$PY" - "$SOCK" <<'PY'
import json, socket, sys
s = socket.socket(socket.AF_UNIX)
s.settimeout(5)
try:
    s.connect(sys.argv[1])
    s.sendall(b'{"op": "ping"}\n')
    reply = s.makefile().readline()
    json.loads(reply)
except Exception as e:
    print(f"no answer on {sys.argv[1]}: {e}")
    sys.exit(1)
print(reply.strip())
PY
}

show() { pgrep -u "$USER" -af "$PAT" || echo "(no logos_data_service.py running as $USER on $(hostname -s))"; }

case "$cmd" in
log)
    tail -n "$n" "$LOG"; exit ;;
status)
    show; ping_svc; exit ;;
esac

old=$(pgrep -u "$USER" -f "$PAT" | tr '\n' ' ')
if [ -n "$old" ]; then
    echo "stopping pid $old"
    pkill -u "$USER" -f "$PAT"
    for i in $(seq 20); do pgrep -u "$USER" -f "$PAT" >/dev/null || break; sleep 0.5; done
    if pgrep -u "$USER" -f "$PAT" >/dev/null; then
        echo "still running after 10 s: kill -9"
        pkill -9 -u "$USER" -f "$PAT"; sleep 1
    fi
else
    echo "no service was running"
fi
if pgrep -af "$PAT" >/dev/null; then
    echo "another user's copy is still running; not starting a second one:"; pgrep -af "$PAT"; exit 1
fi

echo "=== restart $(date '+%F %T') by $USER ===" >> "$LOG"
cd "$DIR" || exit 1
setsid -f "$PY" logos_data_service.py >> "$LOG" 2>&1 < /dev/null

# The first answer comes once pandas, statsmodels and the instrument classes have loaded.
for i in $(seq 60); do
    sleep 1
    if reply=$(ping_svc 2>/dev/null); then
        echo "up after ${i} s: $reply"
        show
        exit 0
    fi
    pgrep -u "$USER" -f "$PAT" >/dev/null || break      # it died: no use waiting
done
echo "the service did not answer; the end of $LOG:"
tail -n 25 "$LOG"
exit 1
REMOTE
