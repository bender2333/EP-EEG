"""M4 bad-lead robustness: zero 2-4 leads on post-standardisation Track features.

Lesion is applied to already-scaled track*_x (after FeatureScaler+tanh), so
filters do not invent transients. Same lead sets for all arms.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from eval.t_only_bnci import CHANNEL_NAMES
from eval.scoring import aggregate, fallback_class, score
from eval.preprocess import channel_of_feature
from .sandbox import train_sandbox
from . import baselines as B

LEADS = CHANNEL_NAMES[:22]
# preregistered lesion sets (same for every group)
LESION_SETS = {
    "none": [],
    "C3_C4": ["C3", "C4"],
    "C3_C4_Cz": ["C3", "C4", "Cz"],
    "C3_C4_Cz_FCz": ["C3", "C4", "Cz", "FCz"],
}


def zero_leads(X, track, lead_names):
    if not lead_names:
        return X
    ch = channel_of_feature(track)
    idx = [LEADS.index(n) for n in lead_names]
    mask = np.isin(ch, idx)
    Y = X.copy(); Y[..., mask] = 0.0
    return Y


def run_ep(train, val, seed, opts, lesion_names):
    sb, stats, _ = train_sandbox(train, seed, **opts)
    if "failed" in stats:
        n = len(val["y"])
        return np.zeros((n, 11, 4)), np.zeros((n, 11), bool), {"failed": stats["failed"]}
    track = sb.cfg.track
    Xv = zero_leads(val[track + "_x"], track, lesion_names)
    return (*sb.predict(Xv), {"params": sb.n_params})


def run_lda(train, val, seed, track, lesion_names):
    # train intact; lesion only at eval (robustness of trained decoder)
    Xtr = train[track + "_x"]; y = train["y"]
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    n, k, d = Xtr.shape
    clf = LDA(solver="lsqr", shrinkage="auto").fit(Xtr.reshape(-1, d), np.repeat(y, k))
    Xv = zero_leads(val[track + "_x"], track, lesion_names)
    p = clf.predict_proba(Xv.reshape(-1, d)).reshape(len(Xv), k, 4)
    return p, np.ones(p.shape[:2], bool), {}


def one(arm, subject, seed, opts, lesion_key):
    from .data import load_subject
    d = load_subject(subject)
    leads = LESION_SETS[lesion_key]
    if arm.startswith("ep_") or arm in ("ep_g1", "ep_g2", "ep_g3", "ep_sparse", "ep_fixed", "ep_track2"):
        o = dict(opts)
        if arm == "ep_g2": o["group"] = "g2"
        elif arm == "ep_g3": o["group"] = "g3"
        elif arm == "ep_sparse": o["group"] = "sparse"
        elif arm == "ep_fixed": o["train_internal"] = False
        scores, ok, extra = run_ep(d["train"], d["val"], seed, o, leads)
    elif arm == "lda_track2":
        scores, ok, extra = run_lda(d["train"], d["val"], seed, "track2", leads)
    else:
        raise ValueError(arm)
    pred, rej = aggregate(scores, ok, fallback_class(d["train"]["y"]))
    m = score(d["val"]["y"], pred, ok)
    m.update(subject=subject, seed=seed, lesion=lesion_key, leads=leads)
    return m


def main_cli():
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    from .data import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--opts", default='{"leak":1.0,"rowmass_project":0.9,"eta":0.5,"temperature":0.01}')
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--tag", default="m4")
    a = ap.parse_args()
    opts = json.loads(a.opts); opts.setdefault("track", "track2")
    seeds = [int(x) for x in a.seeds.split(",")]; subs = [int(x) for x in a.subjects.split(",")]
    t0 = time.time()
    tasks = [(a.arm, s, k, opts, lk) for lk in LESION_SETS for s in subs for k in seeds]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = [f.result() for f in [ex.submit(one, *t) for t in tasks]]
    curve = {}
    for lk in LESION_SETS:
        rows = [r for r in res if r["lesion"] == lk]
        kap = np.array([[r["kappa"] for r in rows if r["subject"] == s] for s in subs])
        curve[lk] = {
            "kappa_val": float(kap.mean()),
            "kappa_seed_sd": float(kap.mean(0).std(ddof=1)) if kap.shape[1] > 1 else 0.0,
            "per_subject_kappa": kap.mean(1).tolist(),
            "settle_rate": float(np.mean([r["settle_rate"] for r in rows])),
            "n_leads": len(LESION_SETS[lk]),
            "leads": LESION_SETS[lk],
        }
    out = {"arm": a.arm, "opts": opts, "curve": curve, "wall_s": time.time() - t0}
    dest = ROOT / "receipts" / "M4" / f"{a.arm}_{a.tag}"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(out, indent=1))
    (dest / "runs.json").write_text(json.dumps(res))
    print(json.dumps(curve, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main_cli()
