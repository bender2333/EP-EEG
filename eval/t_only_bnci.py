"""T-only loader for BNCI2014-001.

MOABB 1.7.2's public ``BNCI2014_001.get_data`` loads both sessions before it
filters the requested session. The research protocol forbids that access
pattern, so this adapter requests and parses only the T file. The private
MOABB helpers are isolated here and require a pinned MOABB version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from moabb.datasets.bnci.base import BNCI_URL, _convert_mi, data_path


CHANNEL_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
    "EOG1", "EOG2", "EOG3",
]
CHANNEL_TYPES = ["eeg"] * 22 + ["eog"] * 3
EVENTS = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}


def load_t_session(
    subject: int,
    *,
    path: str | Path | None = None,
    force_update: bool = False,
    update_path: str | Path | None = None,
    verbose: bool | str | int | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Load exactly one subject's T session and no E session."""

    if isinstance(subject, bool) or not isinstance(subject, int) or not 1 <= subject <= 9:
        raise ValueError("subject must be an integer from 1 through 9")
    url = f"{BNCI_URL}001-2014/A{subject:02d}T.mat"
    filenames = data_path(
        url, path=path, force_update=force_update, update_path=update_path, verbose=verbose
    )
    if len(filenames) != 1:
        raise RuntimeError(f"T-only data path returned {len(filenames)} files")
    runs, event_id = _convert_mi(
        filenames[0], CHANNEL_NAMES, CHANNEL_TYPES, dataset_code="BNCI2014-001",
        subject_id=subject, artifact_handling="ignore", artifact_interval=(2, 6)
    )
    if event_id != EVENTS:
        raise RuntimeError(f"unexpected event mapping: {event_id!r}")
    return {"0train": {str(index): run for index, run in enumerate(runs)}}, event_id
