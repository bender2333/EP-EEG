#!/bin/bash
# M9.5 Phase 2 fixes (exp2b): pooled + 9 LOSO folds, k=32 seed0 g0=0.05. Resumable: completed jobs (DONE) skipped,
# already-running jobs skipped, interrupted jobs resume from per-epoch state.pt. MAXJ concurrent jobs (~1.6 GB each),
# 1 torch thread each (measured faster per-throughput than 8 threads on this 8-core box).
cd "$(dirname "$0")/.."; PY=.venv/bin/python; MAXJ=${MAXJ:-4}; D=receipts/M9_5/exp2b/k32_seed0_g0.05
for j in pooled 1 2 3 4 5 6 7 8 9; do
  if [ "$j" = pooled ]; then tag=pooled; arg=""; pat="exp2b --k 32 --seeds 0 --g0 0.05$"; else tag=loso_s$j; arg="--holdout $j"; pat="exp2b --k 32 --seeds 0 --g0 0.05 --holdout $j$"; fi
  [ -f $D/$tag/DONE ] && continue
  pgrep -f -- "$pat" >/dev/null && continue
  while [ "$(pgrep -fc 'm95_scale exp2b')" -ge "$MAXJ" ]; do sleep 15; done
  M95_THREADS=1 OMP_NUM_THREADS=1 nohup $PY -u -m ep_eeg.m95_scale exp2b --k 32 --seeds 0 --g0 0.05 $arg >> receipts/M9_5/exp2b/logs/$tag.log 2>&1 &
  sleep 5
done
while [ "$(pgrep -fc 'm95_scale exp2b')" -gt 0 ]; do sleep 30; done; echo ALLDONE
