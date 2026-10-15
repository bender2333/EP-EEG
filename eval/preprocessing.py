"""Preprocessing and feature extraction pipeline for BNCI2014-001 MI EEG.

Adheres strictly to ADR-0004 and ADR-0005:
- EOG regression fitted strictly on training runs (runs 0-3).
- 4-35 Hz Butterworth bandpass filtering (zero-phase).
- No cross-channel mixing (no CAR, no EA, no surface Laplacian).
- Subwindow extraction: 11 windows (500 ms length, 250 ms step) covering cue + 0.5s to 3.5s.
- Dual-track features:
  - Track-1: μ (8-13 Hz) and β (13-30 Hz) log-bandpower (22 x 2 = 44 dims).
  - Track-2: Hjorth triplet [log-Activity, Mobility, Complexity] (22 x 3 = 66 dims).
- Per-channel Z-score fitted strictly on training data, followed by tanh(x/3) soft compression.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, welch


def fit_eog_regression(eeg: np.ndarray, eog: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Fit EOG regression weights W on training data such that eeg_clean = eeg - W @ eog.

    Args:
        eeg: Shape (n_eeg_channels, n_times)
        eog: Shape (n_eog_channels, n_times)
        eps: Regularization factor for inversion stability.

    Returns:
        W: Shape (n_eeg_channels, n_eog_channels)
    """
    eog_c = eog - eog.mean(axis=1, keepdims=True)
    cov_eog = (eog_c @ eog_c.T) + eps * np.eye(eog.shape[0])
    weights = (eeg @ eog_c.T) @ np.linalg.inv(cov_eog)
    return weights.astype(np.float64)


def apply_eog_regression(eeg: np.ndarray, eog: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Subtract EOG artifacts using pre-fitted regression weights."""
    eog_c = eog - eog.mean(axis=1, keepdims=True)
    clean_eeg = eeg - weights @ eog_c
    return clean_eeg.astype(np.float64)


def bandpass_filter(
    data: np.ndarray,
    fs: float = 250.0,
    low: float = 4.0,
    high: float = 35.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter along the last axis."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    filtered = filtfilt(b, a, data, axis=-1)
    return filtered.astype(np.float64)


def get_subwindow_starts(
    cue_start_s: float = 0.5,
    cue_end_s: float = 3.5,
    win_len_s: float = 0.5,
    n_windows: int = 11,
    fs: float = 250.0,
) -> np.ndarray:
    """Compute sample index starts for exactly n_windows covering the cue interval."""
    first_start = int(round(cue_start_s * fs))
    last_start = int(round((cue_end_s - win_len_s) * fs))
    return np.round(np.linspace(first_start, last_start, n_windows)).astype(int)


def extract_subwindows(
    trial_data: np.ndarray,
    cue_sample: int,
    fs: float = 250.0,
    n_windows: int = 11,
) -> np.ndarray:
    """Extract 11 subwindows of 500 ms length (125 samples at 250Hz) for one trial.

    Args:
        trial_data: Shape (n_channels, total_run_times)
        cue_sample: Sample index of the cue onset.
        fs: Sampling rate.
        n_windows: Number of subwindows (default 11).

    Returns:
        Subwindows: Shape (n_windows, n_channels, win_samples)
    """
    win_samples = int(round(0.5 * fs))
    starts = get_subwindow_starts(cue_start_s=0.5, cue_end_s=3.5, win_len_s=0.5, n_windows=n_windows, fs=fs)
    windows = []
    for s in starts:
        idx = cue_sample + s
        windows.append(trial_data[:, idx : idx + win_samples])
    return np.stack(windows, axis=0)  # (11, n_channels, win_samples)


def compute_track1_features(subwindows: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """Compute Track-1 features: μ (8-13 Hz) and β (13-30 Hz) log bandpower.

    Args:
        subwindows: Shape (..., n_channels, win_samples)

    Returns:
        Features: Shape (..., n_channels, 2)
    """
    n_samples = subwindows.shape[-1]
    freqs, psd = welch(subwindows, fs=fs, nperseg=n_samples, axis=-1)
    mu_mask = (freqs >= 8.0) & (freqs <= 13.0)
    beta_mask = (freqs > 13.0) & (freqs <= 30.0)

    log_mu = np.log(np.mean(psd[..., mu_mask], axis=-1) + 1e-8)
    log_beta = np.log(np.mean(psd[..., beta_mask], axis=-1) + 1e-8)
    return np.stack([log_mu, log_beta], axis=-1).astype(np.float32)


def compute_track2_features(subwindows: np.ndarray) -> np.ndarray:
    """Compute Track-2 features: Hjorth triplet [log-Activity, Mobility, Complexity].

    Args:
        subwindows: Shape (..., n_channels, win_samples)

    Returns:
        Features: Shape (..., n_channels, 3)
    """
    dx = np.diff(subwindows, axis=-1)
    ddx = np.diff(dx, axis=-1)

    var_x = np.var(subwindows, axis=-1) + 1e-8
    var_dx = np.var(dx, axis=-1) + 1e-8
    var_ddx = np.var(ddx, axis=-1) + 1e-8

    activity = np.log(var_x)
    mobility = np.sqrt(var_dx / var_x)
    complexity = np.sqrt(var_ddx / var_dx) / (mobility + 1e-8)

    return np.stack([activity, mobility, complexity], axis=-1).astype(np.float32)


def normalize_and_compress(
    train_features: np.ndarray,
    val_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Per-channel Z-score standardization followed by tanh(x/3) soft compression.

    Mean and std are computed strictly over train_features across sample and subwindow axes.
    """
    # Features shape: (N, n_windows, n_channels, n_features) or (N_samples, n_channels, n_features)
    # Reduce over sample dimensions (axis 0 or (0, 1))
    reduce_axes = tuple(range(train_features.ndim - 2))
    mean = np.mean(train_features, axis=reduce_axes, keepdims=True)
    std = np.std(train_features, axis=reduce_axes, keepdims=True) + 1e-6

    train_norm = np.tanh((train_features - mean) / (std * 3.0)).astype(np.float32)
    val_norm = np.tanh((val_features - mean) / (std * 3.0)).astype(np.float32)

    stats = {"mean": mean, "std": std}
    return train_norm, val_norm, stats
