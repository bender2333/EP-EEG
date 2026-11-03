#!/usr/bin/env bash
# M11 amendment 4: relaunch the C queue after a container restart. Idempotent: does nothing if the queue runs.
# Steps that already ended with rc=0 after the amendment-4 launch are skipped; EP runs resume from their checkpoints.
set -u
cd "$(dirname "$0")/.."
LOG=runs/068c27c_m11_queue.log
if pgrep -f "bash automation/m11_queue.sh" > /dev/null; then echo "queue already running"; exit 0; fi
since=$(grep -n "amendment 4; code b43bdaf" "$LOG" | head -1 | cut -d: -f1)
left=""
for s in Cep Ctime3 Clr3 Cclosed3 Cbptt3 Cep3; do
  tail -n +"$since" "$LOG" | grep -q "end $s rc=0" || left="$left $s"
done
left=${left# }
[ -z "$left" ] && { echo "nothing left"; exit 0; }
echo "[$(date +%H:%M:%S)] queue resumed by automation/m11_resume.sh after a container restart: $left" >> "$LOG"
M11_RUN_COMMIT=068c27c M11_STEPS="$left" setsid nohup bash automation/m11_queue.sh > /dev/null 2>&1 < /dev/null &
echo "relaunched: $left"
