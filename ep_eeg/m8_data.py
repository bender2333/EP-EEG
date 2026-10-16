"""M8 data: continuous T-session timeline, 5-way per-step labels (4 MI + inter-trial rest).

IV-2a trial timing (per run, onset = trial start = fixation cross):
  t=0 fixation, t=2 s cue, MI until t=6 s, then a 1.5-2.5 s random break, next onset.
Per-step labelling (step = SEQ_C=25 raw samples = 100 ms, no downsampling, 22x25=550 drive dims):
  * MI class 0-3 : step fully inside [onset+2.5 s, onset+5.5 s)  (cue+0.5..3.5 s, same window as M0-M7)
  * rest (4)     : step fully inside [onset+6.5 s, next_onset)   (0.5 s margin after MI end; ends before
                   the next fixation cross; only gaps *between* two trials of the same run)
  * -1 (unlabelled, still fed to the network to keep time continuous): fixation/cue 0-2.5 s,
    5.5-6.5 s MI tail/margin, run lead-in and tail after the last trial.
Rejected-artifact trials are kept (artifact_handling="ignore", as M0-M7).
Preprocessing as M7: EOG regression + 4-35 Hz zero-phase band-pass + per-lead z-score (fit on the
training runs only), then per-feature standardisation + tanh(x/3) (fit on training steps).
Each run is one continuous segment; runs are separate recordings, so `segments` marks run boundaries
(warm reset there by default; see reports/M8.md 待决问题).
"""
from __future__ import annotations
import numpy as np
from eval import preprocess as P
from .data import _runs_T

FS = 250; C = P.SEQ_C
MI0, MI1 = int(2.5 * FS), int(5.5 * FS)
REST0 = int(6.5 * FS)
REST = 4


def step_labels(n_samples, onsets, labels):
    S = n_samples // C
    lab = -np.ones(S, int); ev = -np.ones(S, int)  # ev: trial/gap event id within run
    st = np.arange(S) * C; en = st + C
    for i, (o, l) in enumerate(zip(onsets, labels)):
        m = (st >= o + MI0) & (en <= o + MI1); lab[m] = l; ev[m] = 2 * i
        if i + 1 < len(onsets):
            m = (st >= o + REST0) & (en <= onsets[i + 1]); lab[m] = REST; ev[m] = 2 * i + 1
    return lab, ev


def run_steps(run, pp):
    eeg, eog, on, lab = run
    c = (P._clean(eeg, eog, pp.eog_coef) - pp.mean) / pp.std
    S = c.shape[1] // C
    X = c[:, :S * C].reshape(22, S, C).transpose(1, 0, 2).reshape(S, 22 * C)
    y, ev = step_labels(c.shape[1], on, lab)
    return X.astype(np.float32), y, ev


def timeline(runs, pp, sc=None):
    Xs, ys, evs, seg = [], [], [], []
    for r, run in enumerate(runs):
        X, y, ev = run_steps(run, pp); Xs.append(X); ys.append(y)
        evs.append(np.where(ev >= 0, ev + 10000 * r, -1)); seg.append(np.full(len(y), r))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    if sc is None: sc = P.FeatureScaler().fit(X[y >= 0][None])
    return {"x": sc.transform(X).astype(np.float32), "y": y, "event": np.concatenate(evs),
            "segment": np.concatenate(seg)}, sc


def load_subject_m8(subject, split="t04_val45"):
    """split 't04_val45': train timeline = T runs 0-3, val = T runs 4-5 (frozen protocol, search metric).
    split 'full6': train timeline = all 6 T runs (288 MI trials); no T validation -> eval only via a
    stage-gate on E (not implemented here; needs human sign-off, see reports/M8.md)."""
    runs = _runs_T(subject)
    tr = runs if split == "full6" else runs[:4]
    pp = P.fit_preproc(tr)
    train, sc = timeline(tr, pp)
    out = {"train": train}
    if split == "t04_val45": out["val"], _ = timeline(runs[4:], pp, sc)
    return out


def concat(parts, tags):
    """Head-to-tail concatenation (A01|A02|A03). Each subject keeps its own preprocessing fit;
    segment ids are made unique so warm resets at subject boundaries too."""
    out = {}
    for k in parts[0]:
        off = 0; acc = {"x": [], "y": [], "event": [], "segment": [], "subject": []}
        for p, t in zip(parts, tags):
            d = p[k]
            for f in ("x", "y"): acc[f].append(d[f])
            acc["event"].append(np.where(d["event"] >= 0, d["event"] + 10**6 * t, -1))
            acc["segment"].append(d["segment"] + off); off = acc["segment"][-1].max() + 1
            acc["subject"].append(np.full(len(d["y"]), t))
        out[k] = {f: np.concatenate(v) for f, v in acc.items()}
    return out
