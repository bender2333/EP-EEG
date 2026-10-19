"""M11 continuous-signal data layer (reports/M11.md §5.1): BCI IV 2a T session from the original .gdf.

Only A0xT.gdf is read (E files were never extracted). Runs are delimited by event 32766; calibration blocks
carry 276 (eyes open), 277 (eyes closed) or 1072 (eye movements); motor-imagery runs carry 768 (trial start)
followed by a cue 769-772 and, for expert-rejected trials, 1023. Split as the frozen protocol: MI runs 0-3
train, 4-5 validation. Preprocessing for C (not the frozen eval layer, which A and B keep using):
  causal 4th-order Butterworth 1-45 Hz per block (no look-ahead into the predicted samples);
  EOG regression [EOG, 1] -> EEG fitted on the calibration blocks only (A04: 1072 block only);
  250 -> 125 Hz by taking every second sample; per-lead z-score with MI runs 0-3 statistics;
  usable mask: drops the first 2 s of every block (filter transient) and, for 1023 trials, everything from
  that trial's start to the next trial's start (or the end of the run).
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
import mne
from scipy.signal import butter, sosfilt
from .m11_common import GDF_DIR, sha256

mne.set_log_level("ERROR")
FS, FS_C = 250, 125
CUES = {"769": 0, "770": 1, "771": 2, "772": 3}
CALIB = ("276", "277", "1072")


def read_blocks(subject):
    """{'calib': [(name, eeg, eog)], 'mi': [(eeg, eog, onsets, labels, rejected)]} in uV, 250 Hz, block-relative."""
    path = GDF_DIR / f"A{subject:02d}T.gdf"
    raw = mne.io.read_raw_gdf(path, preload=True, eog=["EOG-left", "EOG-central", "EOG-right"])
    X = raw.get_data() * 1e6; n_nan = int(np.isnan(X).sum())
    if n_nan:  # rare saturation gaps: linear interpolation per channel, counted in the receipt
        for c in range(X.shape[0]):
            bad = np.isnan(X[c]); X[c, bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), X[c, ~bad])
    ev, ed = mne.events_from_annotations(raw); inv = {v: k for k, v in ed.items()}; codes = [inv[e] for e in ev[:, 2]]
    starts = [int(s) for s, c in zip(ev[:, 0], codes) if c == "32766"] + [X.shape[1]]
    out = {"calib": [], "mi": [], "n_nan": n_nan, "sha256": sha256(path)}
    for a, b in zip(starts[:-1], starts[1:]):
        inside = [(int(s) - a, c) for s, c in zip(ev[:, 0], codes) if a <= s < b]
        eeg, eog = X[:22, a:b], X[22:25, a:b]
        cal = [c for _, c in inside if c in CALIB]
        if cal: out["calib"].append((cal[0], eeg, eog)); continue
        ons, labs, rej = [], [], []
        for i, (s, c) in enumerate(inside):
            if c != "768": continue
            nxt = [cc for _, cc in inside[i + 1:i + 4]]
            ons.append(s); labs.append(next(CUES[cc] for cc in nxt if cc in CUES)); rej.append("1023" in nxt)
        if ons: out["mi"].append((eeg, eog, np.array(ons), np.array(labs), np.array(rej)))
    assert len(out["mi"]) == 6, (subject, len(out["mi"]))
    return out


def integrity(subject):
    """Trial segments (trial start .. +6 s) of the .gdf MI runs vs the frozen layer's .mat runs."""
    from eval.t_only_bnci import load_t_session
    from eval import preprocess as P
    g = read_blocks(subject); sess, _ = load_t_session(subject, path="/workspace/mne_data"); cors, same = [], True
    for i in range(6):
        eeg_m, _, on_m, lab_m = P.run_arrays(sess["0train"][str(i)]); eeg_g, _, on_g, lab_g, _ = g["mi"][i]
        same &= bool(np.array_equal(lab_m, lab_g))
        for om, og in zip(on_m, on_g):
            a, b = eeg_m[:, om:om + 1500].ravel(), eeg_g[:, og:og + 1500].ravel(); cors.append(float(np.corrcoef(a, b)[0, 1]))
    return {"labels_identical": same, "min_corr": float(np.min(cors)), "mean_corr": float(np.mean(cors)), "n_trials": len(cors)}


@lru_cache(None)
def continuous(subject):
    """MI runs as [(z[22, T], usable[T])] at 125 Hz plus metadata."""
    g = read_blocks(subject); sos = butter(4, [1.0, 45.0], btype="band", fs=FS, output="sos")
    filt = lambda x: sosfilt(sos, x - x[:, :1], axis=1)
    cal_e = np.hstack([filt(e)[:, 2 * FS:] for _, e, _ in g["calib"]]); cal_o = np.hstack([filt(o)[:, 2 * FS:] for _, _, o in g["calib"]])
    Xo = np.vstack([cal_o, np.ones((1, cal_o.shape[1]))]); coef = np.linalg.lstsq(Xo.T, cal_e.T, rcond=None)[0].T  # (22, 4)
    runs = []
    for eeg, eog, ons, labs, rej in g["mi"]:
        fe, fo = filt(eeg), filt(eog); c = fe - coef @ np.vstack([fo, np.ones((1, fo.shape[1]))])
        ok = np.ones(c.shape[1], bool); ok[:2 * FS] = False
        for i in np.flatnonzero(rej): ok[ons[i]:(ons[i + 1] if i + 1 < len(ons) else c.shape[1])] = False
        runs.append((c[:, ::2], ok[::2], ons // 2, labs, rej))
    tr = np.hstack([r[0][:, r[1]] for r in runs[:4]]); mu, sd = tr.mean(1, keepdims=True), tr.std(1, keepdims=True)
    runs = [((z - mu) / sd, ok, o, l, rj) for z, ok, o, l, rj in runs]
    return {"runs": runs, "eog_coef": coef, "calib_blocks": [n for n, _, _ in g["calib"]], "n_nan": g["n_nan"],
            "sha256": g["sha256"], "n_rejected": int(sum(r[4].sum() for r in runs))}
