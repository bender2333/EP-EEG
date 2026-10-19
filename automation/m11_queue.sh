#!/usr/bin/env bash
# M11 run queue (reports/M11.md §10): A, B, D, E first, then C. Each step uses 4 worker processes, single-threaded.
# Resume after a container restart: rerun this script with the same M11_RUN_COMMIT; finished runs are skipped.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export M11_RUN_COMMIT=${M11_RUN_COMMIT:-$(git rev-parse --short HEAD)}
PY=.venv/bin/python; C=$M11_RUN_COMMIT
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] start $name" >> runs/${C}_m11_queue.log
         "$PY" -m "$@" >> runs/${C}_m11_${name}.log 2>&1; echo "[$(date +%H:%M:%S)] end $name rc=$?" >> runs/${C}_m11_queue.log; }
for s in ${M11_STEPS:-A B D1 D2 E1 E2}; do
  case $s in
    A)  step A  ep_eeg.m11_a --jobs 4 ;;
    B)  step B  ep_eeg.m11_b --jobs 4 ;;
    D1) step D1 ep_eeg.m11_d1 --jobs 4 --part small,asym,real ;;
    D2) step D2 ep_eeg.m11_d2 --jobs 4 ;;
    E1) step E1cal ep_eeg.m11_e --part calibrate && E=$("$PY" -c "import json,glob;print(json.load(open(sorted(glob.glob('receipts/M11/E/$C/e1_calibration.json'))[-1]))['pick'])") && step E1 ep_eeg.m11_e --part e1 --e "$E" --jobs 4 ;;
    E2) step E2 ep_eeg.m11_e --part e2 --jobs 4 ;;
  esac
done
echo "[$(date +%H:%M:%S)] queue done" >> runs/${C}_m11_queue.log
