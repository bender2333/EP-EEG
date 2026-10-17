"""E-session loader. Stage-gate use ONLY (ep_eeg.gate --read-holdout)."""
from __future__ import annotations
import os
from moabb.datasets.bnci.base import BNCI_URL, _convert_mi, data_path
from .t_only_bnci import CHANNEL_NAMES, CHANNEL_TYPES, EVENTS


def load_e_session(subject: int, *, gate_token: str):
    if gate_token != "STAGE_GATE" or os.environ.get("EPEEG_GATE") != "1":
        raise PermissionError("E session is readable only through the stage gate")
    url = f"{BNCI_URL}001-2014/A{subject:02d}E.mat"
    fn = data_path(url)
    runs, ev = _convert_mi(fn[0], CHANNEL_NAMES, CHANNEL_TYPES, dataset_code="BNCI2014-001",
                           subject_id=subject, artifact_handling="ignore", artifact_interval=(2, 6))
    assert ev == EVENTS
    return runs
