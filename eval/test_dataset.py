"""Boundary and leakage tests for eval/dataset.py."""

from __future__ import annotations

import unittest
from unittest.mock import patch
import numpy as np

from eval.dataset import load_subject_split


class DatasetLeakageTests(unittest.TestCase):
    @patch("eval.dataset.load_t_session")
    def test_split_boundary_and_shapes(self, mock_load) -> None:
        # Create synthetic MNE RawArray mock for 6 runs
        import mne
        n_times = 5000
        ch_names = [f"EEG{i}" for i in range(22)] + ["EOG1", "EOG2", "EOG3", "STI"]
        info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types=["eeg"]*22 + ["eog"]*3 + ["stim"])  # type: ignore[arg-type]

        runs = {}
        for r in range(6):
            data = np.zeros((26, n_times), dtype=np.float64)
            # Add 2 events per run
            data[25, 500] = 1   # Left hand (label 0)
            data[25, 2000] = 2  # Right hand (label 1)
            raw = mne.io.RawArray(data, info, verbose=False)
            runs[str(r)] = raw

        mock_load.return_value = ({"0train": runs}, {"left_hand": 1, "right_hand": 2})

        split = load_subject_split(1, track="track2")

        # 4 train runs * 2 trials = 8 trials * 11 windows = 88 subwindows
        self.assertEqual(split["X_train"].shape, (88, 22, 3))
        self.assertEqual(len(split["y_train_trial"]), 8)

        # 2 val runs * 2 trials = 4 trials * 11 windows = 44 subwindows
        self.assertEqual(split["X_val"].shape, (44, 22, 3))
        self.assertEqual(len(split["y_val_trial"]), 4)

        # Check normalization bounds
        self.assertTrue(np.all(split["X_train"] >= -1.0) and np.all(split["X_train"] <= 1.0))
        self.assertTrue(np.all(split["X_val"] >= -1.0) and np.all(split["X_val"] <= 1.0))


if __name__ == "__main__":
    unittest.main()
