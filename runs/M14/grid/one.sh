#!/usr/bin/env bash
# one grid job: arm subject eta (seed 0, U 3000, T 0.01); skips a finished receipt
cd /home/claude/ep-eeg
export PYTHONPATH=/home/claude/ep-eeg M14_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MNE_DATA=/workspace/mne_data
arm=$1; s=$2; eta=$3; seed=${4:-0}
f=receipts/M14/grid/${arm}_A0${s}_s${seed}_eta$(printf "%g" $eta)_T0.01.json
[ -f "$f" ] && exit 0
/home/user/.venv/bin/python -m ep_eeg.m14 --subject $s --arm $arm --seed $seed --U 3000 --eta $eta --temp 0.01 --out receipts/M14/grid > runs/M14/grid/${arm}_A0${s}_s${seed}_eta${eta}.log 2>&1
