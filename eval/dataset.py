"""Dataset loader and split manager for BNCI2014-001 T-only sessions.

Strictly follows the frozen protocol:
- Loads ONLY Session T (runs 0-5).
- Train split: runs 0, 1, 2, 3 (192 trials).
- Validation split: runs 4, 5 (96 trials).
- EOG regression weights and Z-score statistics fitted ONLY on train split.
"""

from __future__ import annotations

from typing import Any, Literal
import mne
import numpy as np

from eval.preprocessing import (
    apply_eog_regression,
    bandpass_filter,
    compute_track1_features,
    compute_track2_features,
    extract_subwindows,
    fit_eog_regression,
    normalize_and_compress,
)
from eval.t_only_bnci import load_t_session


def load_subject_split(
    subject_id: int,
    track: Literal["track1", "track2"] = "track2",
    train_runs: tuple[str, ...] = ("0", "1", "2", "3"),
    val_runs: tuple[str, ...] = ("4", "5"),
) -> dict[str, Any]:
    """Load, preprocess, and extract features for a single subject without data leakage.

    Returns:
        dict containing:
            X_train: (n_train_subwins, 22, feat_dim)
            y_train: (n_train_subwins,) - subwindow-level labels
            train_trial_ids: (n_train_subwins,)
            y_train_trial: (n_train_trials,)
            X_val: (n_val_subwins, 22, feat_dim)
            y_val: (n_val_subwins,)
            val_trial_ids: (n_val_subwins,)
            y_val_trial: (n_val_trials,)
            eog_weights: (22, 3)
            norm_stats: dict of mean, std
            subject: subject_id
            track: track name
    """
    sessions, event_mapping = load_t_session(subject_id)
    runs = sessions["0train"]

    # 1. Fit EOG regression strictly across training runs
    train_eeg_list = []
    train_eog_list = []
    for r_idx in train_runs:
        raw = runs[r_idx]
        data = raw.get_data()  # shape (26, n_times)
        train_eeg_list.append(data[:22, :])
        train_eog_list.append(data[22:25, :])

    train_eeg_concat = np.concatenate(train_eeg_list, axis=1)
    train_eog_concat = np.concatenate(train_eog_list, axis=1)
    eog_weights = fit_eog_regression(train_eeg_concat, train_eog_concat)

    # 2. Process runs: EOG clean -> Bandpass filter -> Extract trials
    def _process_run_group(run_indices: tuple[str, ...]):
        trial_subwindows_list = []
        trial_labels_list = []

        for r_idx in run_indices:
            raw = runs[r_idx]
            data = raw.get_data()
            eeg = data[:22, :]
            eog = data[22:25, :]

            # Clean EOG and bandpass filter continuous signal
            clean_eeg = apply_eog_regression(eeg, eog, eog_weights)
            filtered_eeg = bandpass_filter(clean_eeg, fs=raw.info["sfreq"], low=4.0, high=35.0)

            # Find trial events on stimulus channel
            events = mne.find_events(raw, verbose=False)
            for ev in events:
                cue_sample = int(ev[0])
                event_code = int(ev[2])
                label = event_code - 1  # 0-indexed: 0, 1, 2, 3

                subwins = extract_subwindows(filtered_eeg, cue_sample, fs=raw.info["sfreq"], n_windows=11)
                trial_subwindows_list.append(subwins)
                trial_labels_list.append(label)

        trials_subwins = np.stack(trial_subwindows_list, axis=0)  # (N_trials, 11, 22, win_samples)
        trial_labels = np.array(trial_labels_list, dtype=np.int64)  # (N_trials,)
        return trials_subwins, trial_labels

    train_raw_subwins, y_train_trial = _process_run_group(train_runs)
    val_raw_subwins, y_val_trial = _process_run_group(val_runs)

    # 3. Compute Track features
    if track == "track1":
        train_feats = compute_track1_features(train_raw_subwins, fs=250.0)
        val_feats = compute_track1_features(val_raw_subwins, fs=250.0)
    elif track == "track2":
        train_feats = compute_track2_features(train_raw_subwins)
        val_feats = compute_track2_features(val_raw_subwins)
    else:
        raise ValueError(f"Unknown track: {track}. Must be 'track1' or 'track2'.")

    # 4. Normalize and soft compress using strictly training stats
    train_norm, val_norm, norm_stats = normalize_and_compress(train_feats, val_feats)

    # Flatten trial x window into sample axis
    n_train_trials, n_wins, n_ch, feat_dim = train_norm.shape
    X_train = train_norm.reshape(n_train_trials * n_wins, n_ch, feat_dim)
    y_train = np.repeat(y_train_trial, n_wins)
    train_trial_ids = np.repeat(np.arange(n_train_trials), n_wins)

    n_val_trials = val_norm.shape[0]
    X_val = val_norm.reshape(n_val_trials * n_wins, n_ch, feat_dim)
    y_val = np.repeat(y_val_trial, n_wins)
    val_trial_ids = np.repeat(np.arange(n_val_trials), n_wins)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "train_trial_ids": train_trial_ids,
        "y_train_trial": y_train_trial,
        "X_val": X_val,
        "y_val": y_val,
        "val_trial_ids": val_trial_ids,
        "y_val_trial": y_val_trial,
        "eog_weights": eog_weights,
        "norm_stats": norm_stats,
        "subject": subject_id,
        "track": track,
    }


def load_subject_raw_trials(
    subject_id: int,
    train_runs: tuple[str, ...] = ("0", "1", "2", "3"),
    val_runs: tuple[str, ...] = ("4", "5"),
    cue_start_s: float = 0.5,
    cue_end_s: float = 3.5,
) -> dict[str, Any]:
    """Load continuous raw trial epochs (n_trials, 22, 750) for FBCSP and EEGNet."""
    sessions, _ = load_t_session(subject_id)
    runs = sessions["0train"]

    # 1. Fit EOG regression on training runs
    train_eeg_list, train_eog_list = [], []
    for r_idx in train_runs:
        data = runs[r_idx].get_data()
        train_eeg_list.append(data[:22, :])
        train_eog_list.append(data[22:25, :])
    eog_weights = fit_eog_regression(
        np.concatenate(train_eeg_list, axis=1),
        np.concatenate(train_eog_list, axis=1),
    )

    def _extract_epochs(run_indices: tuple[str, ...]):
        epochs_list, labels_list = [], []
        for r_idx in run_indices:
            raw = runs[r_idx]
            fs = raw.info["sfreq"]
            data = raw.get_data()
            clean_eeg = apply_eog_regression(data[:22, :], data[22:25, :], eog_weights)
            filtered = bandpass_filter(clean_eeg, fs=fs, low=4.0, high=35.0)

            start_off = int(round(cue_start_s * fs))
            epoch_len = int(round((cue_end_s - cue_start_s) * fs))
            events = mne.find_events(raw, verbose=False)
            for ev in events:
                cue = int(ev[0])
                label = int(ev[2]) - 1
                epochs_list.append(filtered[:, cue + start_off : cue + start_off + epoch_len])
                labels_list.append(label)
        return np.stack(epochs_list, axis=0).astype(np.float32), np.array(labels_list, dtype=np.int64)

    X_train, y_train = _extract_epochs(train_runs)
    X_val, y_val = _extract_epochs(val_runs)

    # Per-channel standardization using train split
    ch_mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    ch_std = np.std(X_train, axis=(0, 2), keepdims=True) + 1e-6
    X_train_norm = (X_train - ch_mean) / ch_std
    X_val_norm = (X_val - ch_mean) / ch_std

    return {
        "X_train": X_train_norm.astype(np.float32),
        "y_train": y_train,
        "X_val": X_val_norm.astype(np.float32),
        "y_val": y_val,
        "subject": subject_id,
    }

