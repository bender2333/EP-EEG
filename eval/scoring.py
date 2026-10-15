"""Frozen trial aggregation and scoring.

Trial decision: average the output activities (or class probabilities) of the
qualified sub-windows, argmax. A trial with no qualified sub-window gets the
fixed fallback class (training majority; ties -> lowest class index) and is
counted as rejected. kappa always covers every scheduled trial.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score


def fallback_class(y_train):
    return int(np.argmax(np.bincount(y_train, minlength=4)))


def aggregate(sub_scores, sub_ok, fallback):
    """sub_scores (n,11,4); sub_ok (n,11) bool -> (pred (n,), rejected (n,) bool)."""
    ok = sub_ok.astype(float)[..., None]
    cnt = ok.sum(1)
    mean = (sub_scores * ok).sum(1) / np.maximum(cnt, 1)
    pred = mean.argmax(1)
    rej = cnt[:, 0] == 0
    pred[rej] = fallback
    return pred, rej


def score(y, pred, sub_ok=None):
    out = {"kappa": float(cohen_kappa_score(y, pred)), "acc": float(accuracy_score(y, pred)), "n": int(len(y))}
    if sub_ok is not None:
        out["settle_rate"] = float(sub_ok.mean())
        out["coverage_8of11"] = float((sub_ok.sum(1) >= 8).mean())
        out["rejected_trials"] = int((sub_ok.sum(1) == 0).sum())
        out["rejected_subwindows"] = int((~sub_ok).sum())
    return out
