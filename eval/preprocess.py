"""Frozen preprocessing and dual-track feature encoding (ADR-0004, ADR-0005).

Order: per-run mean removal, EOG->EEG per-channel linear regression fitted on
T runs 0-3 only, 4-35 Hz zero-phase band-pass per run, per-channel Z-score with
T run 0-3 statistics, 11 sub-windows (cue+0.5..cue+3.5 s, 500 ms / 250 ms),
Track-1 (mu/beta log power) or Track-2 (Hjorth), per-feature standardisation
fitted on training sub-windows, tanh(x/3) soft compression.
No channel-mixing transform is applied to EEG.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 250.0
N_EEG = 22
TRAIN_RUNS = (0, 1, 2, 3)
VAL_RUNS = (4, 5)
WIN_START = int(2.5 * FS)  # trial start + 2.5 s == cue + 0.5 s
WIN_LEN = int(0.5 * FS)    # 125 samples
N_SUB = 11
SUB_STARTS = [int(round(k * 0.25 * FS)) for k in range(N_SUB)]
SEG_LEN = SUB_STARTS[-1] + WIN_LEN  # 750
CLASSES = ("left_hand", "right_hand", "feet", "tongue")


def run_arrays(raw):
    """Return (eeg[22,T], eog[3,T], onsets, labels 0..3) from an MNE raw run (volts->uV)."""
    import mne
    data = raw.get_data() * 1e6
    eeg, eog = data[:N_EEG], data[N_EEG:N_EEG + 3]
    events = mne.find_events(raw, verbose=False)
    # MOABB codes: 1 left_hand, 2 right_hand, 3 feet, 4 tongue; onset = trial start (cue at +2 s)
    onsets = events[:, 0].astype(int)
    labels = events[:, 2].astype(int) - 1
    return eeg, eog, np.asarray(onsets), np.asarray(labels)


@dataclass
class Preproc:
    eog_coef: np.ndarray  # (22, 4) incl intercept
    mean: np.ndarray
    std: np.ndarray


_SOS = butter(4, [4.0, 35.0], btype="band", fs=FS, output="sos")


def _clean(eeg, eog, coef):
    eeg = eeg - eeg.mean(1, keepdims=True)
    eog = eog - eog.mean(1, keepdims=True)
    X = np.vstack([eog, np.ones((1, eog.shape[1]))])
    out = eeg - coef @ X
    return sosfiltfilt(_SOS, out, axis=1)


def fit_preproc(train_runs):
    E = np.hstack([r[0] - r[0].mean(1, keepdims=True) for r in train_runs])
    O = np.hstack([r[1] - r[1].mean(1, keepdims=True) for r in train_runs])
    X = np.vstack([O, np.ones((1, O.shape[1]))])
    coef = np.linalg.lstsq(X.T, E.T, rcond=None)[0].T
    F = np.hstack([_clean(r[0], r[1], coef) for r in train_runs])
    return Preproc(coef, F.mean(1, keepdims=True), F.std(1, keepdims=True))


def epochs(runs, pp: Preproc):
    """Cleaned, z-scored trial segments (n, 22, 750) and labels."""
    xs, ys = [], []
    for eeg, eog, on, lab in runs:
        c = (_clean(eeg, eog, pp.eog_coef) - pp.mean) / pp.std
        for o, l in zip(on, lab):
            xs.append(c[:, o + WIN_START:o + WIN_START + SEG_LEN]); ys.append(l)
    return np.asarray(xs), np.asarray(ys)


def subwindows(seg):
    """(n,22,750) -> (n,11,22,125)."""
    return np.stack([seg[:, :, s:s + WIN_LEN] for s in SUB_STARTS], axis=1)


def track1(sw):
    w = np.hanning(WIN_LEN)
    spec = np.abs(np.fft.rfft(sw * w, axis=-1)) ** 2
    f = np.fft.rfftfreq(WIN_LEN, 1 / FS)
    mu = spec[..., (f >= 8) & (f < 13)].mean(-1)
    be = spec[..., (f >= 13) & (f < 30)].mean(-1)
    return np.concatenate([np.log(mu + 1e-12), np.log(be + 1e-12)], axis=-1)  # (...,44): [mu22, beta22]


def track2(sw):
    d1 = np.diff(sw, axis=-1); d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = sw.var(-1), d1.var(-1), d2.var(-1)
    mob = np.sqrt(v1 / v0); comp = np.sqrt(v2 / v1) / mob
    return np.concatenate([np.log(v0), mob, comp], axis=-1)  # (...,66): [act22, mob22, comp22]


RAW_T = 25  # M7: samples kept per lead per sub-window (125 -> 25 by 5-sample block mean)


def rawwin(sw, t_eff=RAW_T):
    """M7 raw-waveform encoding (no Hjorth): (...,22,125) -> (...,22*t_eff), lead-major.
    Block-mean downsampling along time only (125/t_eff samples per block)."""
    blk = sw.shape[-1] // t_eff
    x = sw[..., :blk * t_eff].reshape(*sw.shape[:-1], t_eff, blk).mean(-1)
    return x.reshape(*sw.shape[:-2], -1)


TRACKS = {"track1": track1, "track2": track2, "raw": rawwin}


class FeatureScaler:
    def fit(self, F):  # F (n,11,d)
        flat = F.reshape(-1, F.shape[-1])
        self.m, self.s = flat.mean(0), flat.std(0) + 1e-8
        return self

    def transform(self, F):
        return np.tanh(((F - self.m) / self.s) / 3.0)


def channel_of_feature(track: str):
    """Return the lead index for each feature column."""
    if track == "raw":
        return np.repeat(np.arange(N_EEG), RAW_T)
    k = 2 if track == "track1" else 3
    return np.tile(np.arange(N_EEG), k)
