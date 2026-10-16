#!/usr/bin/env bash
# M8 launcher. Run ONLY after M7 jobs (ep_eeg.run --stage M7) have finished: check `pgrep -af ep_eeg.run`.
# Cost estimate (8 CPU, measured smoke under load): ~484 warm steps/epoch at batch 32 per subject;
# 100 epochs ≈ several hours per subject; triple ≈ 3x single steps in one process.
set -euo pipefail; cd "$(dirname "$0")/.."
if pgrep -f "ep_eeg.run --stage M7" >/dev/null; then echo "M7 still running; abort"; exit 1; fi
export EPEEG_GATE=1  # user-approved E stage gate (full6)
OPTS='{"epochs":100,"split":"full6"}'
# a) single-subject nets (protocol split: train T runs0-3 timeline, val T runs4-5)
nohup .venv/bin/python -u -m ep_eeg.m8_train --arm ep_m8_single --subjects 1-9 --seeds 0 --jobs 4 --opts "$OPTS" --tag _full > runs/m8_single.log 2>&1
# b) shared net on A01|A02|A03 timeline; compare with single A01-A03 per-subject kappa
nohup .venv/bin/python -u -m ep_eeg.m8_train --arm ep_m8_triple --seeds 0 --jobs 1 --opts "$OPTS" --tag _full > runs/m8_triple.log 2>&1
