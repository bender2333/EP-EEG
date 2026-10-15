"""Evaluation metrics and trial aggregation for BNCI2014-001 MI experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score


def aggregate_trial_predictions(subwindow_probs: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
    """Aggregate 11 subwindow posterior probabilities into a trial-level class prediction.

    Args:
        subwindow_probs: Shape (n_subwindows, n_classes)
        trial_ids: Shape (n_subwindows,)

    Returns:
        trial_preds: Shape (n_unique_trials,)
    """
    unique_trials = np.unique(trial_ids)
    preds = np.zeros(len(unique_trials), dtype=np.int64)
    for i, tid in enumerate(unique_trials):
        mask = (trial_ids == tid)
        mean_probs = np.mean(subwindow_probs[mask], axis=0)
        preds[i] = int(np.argmax(mean_probs))
    return preds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute trial-level Cohen's kappa and accuracy."""
    acc = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    return {"accuracy": acc, "kappa": kappa}
