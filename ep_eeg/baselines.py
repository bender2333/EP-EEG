"""Baseline models for M0: LDA (Track 1/2), Shuffled labels, FBCSP, and EEGNet."""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
import torch.nn as nn
from mne.decoding import CSP
from braindecode.models import EEGNet

from eval.dataset import load_subject_raw_trials, load_subject_split
from eval.metrics import aggregate_trial_predictions, compute_metrics
from eval.preprocessing import bandpass_filter


def run_lda(
    subject_id: int,
    track: str = "track2",
    shuffle_labels: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Run LDA baseline on Track-1 or Track-2 subwindow features."""
    data = load_subject_split(subject_id, track=track)  # type: ignore[arg-type]
    X_train = data["X_train"].reshape(len(data["X_train"]), -1)
    X_val = data["X_val"].reshape(len(data["X_val"]), -1)

    y_train_trial = data["y_train_trial"]
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        y_train_trial = rng.permutation(y_train_trial)

    y_train = np.repeat(y_train_trial, 11)

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_train, y_train)

    val_probs = lda.predict_proba(X_val)
    trial_preds = aggregate_trial_predictions(val_probs, data["val_trial_ids"])

    metrics = compute_metrics(data["y_val_trial"], trial_preds)
    return {
        "accuracy": metrics["accuracy"],
        "kappa": metrics["kappa"],
        "trial_preds": trial_preds.tolist(),
        "y_true": data["y_val_trial"].tolist(),
        "settle_rate": 1.0,
    }


def run_fbcsp(
    subject_id: int,
    seed: int = 0,
    n_components: int = 4,
) -> dict[str, Any]:
    """Run Filter Bank Common Spatial Pattern (FBCSP) + shrinkage LDA baseline."""
    np.random.seed(seed)
    data = load_subject_raw_trials(subject_id)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
    tr_feats_list = []
    val_feats_list = []

    for low, high in bands:
        X_tr_band = bandpass_filter(X_train, fs=250.0, low=low, high=high)
        X_val_band = bandpass_filter(X_val, fs=250.0, low=low, high=high)
        csp = CSP(n_components=n_components, log=True, rank="full")
        tr_feats_list.append(csp.fit_transform(X_tr_band, y_train))
        val_feats_list.append(csp.transform(X_val_band))

    F_train = np.concatenate(tr_feats_list, axis=1)
    F_val = np.concatenate(val_feats_list, axis=1)

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(F_train, y_train)
    preds = lda.predict(F_val)

    metrics = compute_metrics(y_val, preds)
    return {
        "accuracy": metrics["accuracy"],
        "kappa": metrics["kappa"],
        "trial_preds": preds.tolist(),
        "y_true": y_val.tolist(),
        "settle_rate": 1.0,
    }


def run_eegnet(
    subject_id: int,
    seed: int = 0,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    """Run Braindecode EEGNet baseline on continuous trial epochs."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = load_subject_raw_trials(subject_id)
    X_train = torch.from_numpy(data["X_train"])
    y_train = torch.from_numpy(data["y_train"])
    X_val = torch.from_numpy(data["X_val"])
    y_val = data["y_val"]

    model = EEGNet(n_chans=22, n_outputs=4, n_times=750)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        preds = val_logits.argmax(dim=-1).numpy()

    metrics = compute_metrics(y_val, preds)
    return {
        "accuracy": metrics["accuracy"],
        "kappa": metrics["kappa"],
        "trial_preds": preds.tolist(),
        "y_true": y_val.tolist(),
        "settle_rate": 1.0,
    }
