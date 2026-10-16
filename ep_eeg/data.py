"""Subject data via the frozen eval layer, cached under cache/ (untracked)."""
from __future__ import annotations
import os, hashlib
from pathlib import Path
import numpy as np
import mne

mne.set_log_level("ERROR")
os.environ.setdefault("MNE_DATA", "/workspace/mne_data")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"

from eval.t_only_bnci import load_t_session
from eval import preprocess as P


def _runs_T(subject):
    sess, _ = load_t_session(subject, path="/workspace/mne_data")
    runs = [P.run_arrays(sess["0train"][str(i)]) for i in range(6)]
    return runs


def build(subject, runs_fit, runs_eval_sets):
    pp = P.fit_preproc(runs_fit)
    out = {}
    for name, runs in runs_eval_sets.items():
        seg, y = P.epochs(runs, pp)
        sw = P.subwindows(seg)
        out[name] = {"seg": seg.astype(np.float32), "y": y,
                     "track1": P.track1(sw), "track2": P.track2(sw)}
    for tr in ("track1", "track2"):
        sc = P.FeatureScaler().fit(out["train"][tr])
        for name in out:
            out[name][tr + "_x"] = sc.transform(out[name][tr]).astype(np.float32)
    return out


def add_raw(d):
    """M7: raw-waveform sub-window inputs from cached preprocessed segments (no Hjorth)."""
    if "raw_x" in d["train"]: return d
    F = {n: P.rawwin(P.subwindows(d[n]["seg"].astype(np.float64))) for n in d}
    sc = P.FeatureScaler().fit(F["train"])
    for n in d: d[n]["raw_x"] = sc.transform(F[n]).astype(np.float32)
    return d


def load_subject(subject: int):
    """T-only: train = runs 0-3, val = runs 4-5."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"A{subject:02d}T.npz"
    if f.exists():
        z = np.load(f)
    else:
        runs = _runs_T(subject)
        d = build(subject, runs[:4], {"train": runs[:4], "val": runs[4:]})
        flat = {f"{s}__{k}": v for s, dd in d.items() for k, v in dd.items()}
        np.savez(f, **flat); z = np.load(f)
    d = {"train": {}, "val": {}}
    for k in z.files:
        s, kk = k.split("__"); d[s][kk] = z[k]
    return add_raw(d)


def load_subject_gate(subject: int):
    """Stage gate only: train = T runs 0-3 (same fit), eval = E session."""
    from eval.holdout import load_e_session
    runsT = _runs_T(subject)
    runsE = [P.run_arrays(r) for r in load_e_session(subject, gate_token="STAGE_GATE")]
    return build(subject, runsT[:4], {"train": runsT[:4], "val": runsE})


def file_sha(subject, sess="T"):
    p = Path("/workspace/mne_data/MNE-bnci-data/~bci/database/001-2014") / f"A{subject:02d}{sess}.mat"
    return hashlib.sha256(p.read_bytes()).hexdigest()
