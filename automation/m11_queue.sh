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
    # step C in the order of amendment M3-2/M3-3: data checks and settings, E3, closed form, BPTT, C0, EP grid
    Cint) step Cint ep_eeg.m11_c --phase integrity --jobs 4
          "$PY" -c "import json;r=json.load(open('receipts/M11/C/integrity.json'))['per_subject'];bad=[s for s,v in r.items() if not(v['labels_identical'] and v['min_corr']>=0.999)];print('integrity failures',bad);exit(1 if bad else 0)" \
            >> runs/${C}_m11_queue.log 2>&1 || { echo "[$(date +%H:%M:%S)] integrity failed: queue stopped" >> runs/${C}_m11_queue.log; exit 1; } ;;
    Csel)  step Csel  ep_eeg.m11_c --phase select --jobs 4 ;;
    Ctime) step Ctime ep_eeg.m11_c --phase timing --jobs 4 ;;
    Clr)   step Clr   ep_eeg.m11_c --phase lrsel --jobs 4 ;;
    E3)    step E3    ep_eeg.m11_c --phase e3 --jobs 4 ;;
    Cclosed) step Cclosed ep_eeg.m11_c --phase main --jobs 4 --arms var6,var_best,res,lin_res ;;
    Cbptt) step Cbptt ep_eeg.m11_c --phase main --jobs 4 --arms bptt_noproj_U,bptt_proj_U,bptt_noproj_2000,bptt_proj_2000,lin_bptt_2000 ;;
    C0)    mkdir -p receipts/M11/C0/runs
           G3_OUT=receipts/M11/C0 G3_RUN_DIR=receipts/M11/C0/runs step C0 ep_eeg.g3_toy --phase final --jobs 4 \
             --variant '[{"a_init": "orth", "opt": "adam", "proj": 0.95}, {"a_init": "orth", "opt": "adam", "proj": 0.99}]'
           G3_OUT=receipts/M11/C0 step C0refs ep_eeg.g3_toy --phase c0refs --jobs 4 ;;
    Cep)   step Cep   ep_eeg.m11_c --phase main --jobs 4 --arms ep_noproj,ep_proj ;;
  esac
done
echo "[$(date +%H:%M:%S)] queue done" >> runs/${C}_m11_queue.log
