"""Unit tests for eval/preprocessing.py and eval/metrics.py."""

from __future__ import annotations

import unittest
import numpy as np

from eval.preprocessing import (
    apply_eog_regression,
    bandpass_filter,
    compute_track1_features,
    compute_track2_features,
    extract_subwindows,
    fit_eog_regression,
    get_subwindow_starts,
    normalize_and_compress,
)
from eval.metrics import aggregate_trial_predictions, compute_metrics


class PreprocessingTests(unittest.TestCase):
    def test_subwindow_starts_exact_count_and_boundaries(self) -> None:
        starts = get_subwindow_starts(cue_start_s=0.5, cue_end_s=3.5, win_len_s=0.5, n_windows=11, fs=250.0)
        self.assertEqual(len(starts), 11)
        self.assertEqual(starts[0], 125)  # 0.5s * 250
        self.assertEqual(starts[-1] + 125, 875)  # 3.5s * 250

    def test_eog_regression_zeroes_correlated_artifact(self) -> None:
        rng = np.random.default_rng(42)
        n_times = 5000
        eog = rng.standard_normal((3, n_times))
        true_w = np.array([[0.5, -0.2, 0.1], [-0.3, 0.4, 0.0]])
        neural = rng.standard_normal((2, n_times)) * 0.1
        eeg = true_w @ eog + neural

        fitted_w = fit_eog_regression(eeg, eog)
        np.testing.assert_allclose(fitted_w, true_w, atol=1e-2)

        clean = apply_eog_regression(eeg, eog, fitted_w)
        # Cleaned signal variance should be close to pure neural signal variance
        np.testing.assert_allclose(np.var(clean, axis=1), np.var(neural, axis=1), atol=0.05)

    def test_bandpass_filter_shape_and_attenuation(self) -> None:
        fs = 250.0
        t = np.arange(2500) / fs
        # 50 Hz noise (out of band) + 10 Hz alpha wave (in band 4-35 Hz)
        sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)
        filtered = bandpass_filter(sig[None, :], fs=fs, low=4.0, high=35.0)
        self.assertEqual(filtered.shape, (1, 2500))
        # 50 Hz power should be heavily attenuated in interior (away from boundary pad transients)
        interior = slice(int(fs), -int(fs))
        self.assertLess(np.max(np.abs(filtered[0, interior] - np.sin(2 * np.pi * 10 * t[interior]))), 0.05)

    def test_track1_and_track2_output_shapes(self) -> None:
        dummy_subwins = np.random.randn(5, 11, 22, 125)
        t1 = compute_track1_features(dummy_subwins, fs=250.0)
        self.assertEqual(t1.shape, (5, 11, 22, 2))

        t2 = compute_track2_features(dummy_subwins)
        self.assertEqual(t2.shape, (5, 11, 22, 3))

    def test_normalize_and_compress_bounded_in_minus_one_to_one(self) -> None:
        train = np.random.randn(10, 11, 22, 3) * 5.0 + 2.0
        val = np.random.randn(4, 11, 22, 3) * 5.0 + 2.0
        train_norm, val_norm, stats = normalize_and_compress(train, val)

        self.assertTrue(np.all(train_norm > -1.0) and np.all(train_norm < 1.0))
        self.assertTrue(np.all(val_norm > -1.0) and np.all(val_norm < 1.0))
        self.assertIn("mean", stats)
        self.assertIn("std", stats)


class MetricsTests(unittest.TestCase):
    def test_aggregate_trial_predictions(self) -> None:
        # 2 trials, 3 subwindows each, 4 classes
        probs = np.array([
            [0.1, 0.7, 0.1, 0.1],
            [0.2, 0.6, 0.1, 0.1],
            [0.1, 0.8, 0.0, 0.1],  # Trial 0 -> mean favors class 1
            [0.8, 0.1, 0.1, 0.0],
            [0.7, 0.2, 0.0, 0.1],
            [0.6, 0.3, 0.0, 0.1],  # Trial 1 -> mean favors class 0
        ])
        tids = np.array([0, 0, 0, 1, 1, 1])
        preds = aggregate_trial_predictions(probs, tids)
        np.testing.assert_array_equal(preds, [1, 0])

    def test_compute_metrics(self) -> None:
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        m = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["kappa"], 1.0)


if __name__ == "__main__":
    unittest.main()
