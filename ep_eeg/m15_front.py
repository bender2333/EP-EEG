"""M15 block 4: FBCSP, band log-variance LDA, and an EP linear head on those features.

Same T split as M14. The features are frozen; nothing here trains the M14 network.
"""
from __future__ import annotations
import argparse
import numpy as np

from . import m15_common as C

BANDS = [(lo, lo + 4) for lo in range(4, 36, 4)]


def logvar(seg):
    from scipy.signal import butter, sosfiltfilt
    cols = []
    x0 = seg.astype(np.float64)
    for lo, hi in BANDS:
        sos = butter(4, [lo, min(hi, 124.0)], btype="band", fs=250, output="sos")
        x = sosfiltfilt(sos, x0, axis=-1)
        cols.append(np.log(x.var(axis=-1) + 1e-12))
    return np.concatenate(cols, axis=1)


def one(subject, seed=0):
    from ep_eeg.baselines import fbcsp
    from ep_eeg.m14_probe import lda_scores
    data = C.load_t(subject)
    # load_t returns windowed arrays. FBCSP and log-var need the frozen segments.
    from ep_eeg.data import load_subject
    raw = load_subject(subject)
    p, _, _ = fbcsp(raw["train"], raw["val"], seed)
    fb = C.kappa_acc(raw["val"]["y"], p[:, 0].argmax(1))
    Ftr, Fte = logvar(raw["train"]["seg"]), logvar(raw["val"]["seg"])
    ytr, yte = np.asarray(raw["train"]["y"]), np.asarray(raw["val"]["y"])
    lda = lda_scores(Ftr, ytr, Fte, yte)
    bp_te, bp_tr = C.logreg_predict(Ftr, ytr, Fte)
    bp = {**C.kappa_acc(yte, bp_te), "train_acc": float((bp_tr == ytr).mean())}
    h = C.build_head(Ftr.shape[1], seed=18000 + subject + 10 * seed)
    info = C.train_head(h, Ftr, ytr, U=800, seed=seed, batch=32)
    pred, _ = C.head_activity(h, Fte)
    ep = {**C.kappa_acc(yte, pred.argmax(1)), "updates": info["updates"],
          "train_acc": info["train_acc"]}
    rec = {"stage": "M15", "block": "front", "split": "val", "subject": subject, "seed": seed,
           "dev": False, "fbcsp": fb, "logvar_lda": {"acc": lda["acc_val"], "kappa": lda["kappa_val"],
                                                     "train_acc": lda["acc_train"], "n_features": lda["n_features"]},
           "logvar_bp": bp, "logvar_ep": ep}
    C.write_json(C.OUT / "front" / f"A{subject:02d}_s{seed}.json", rec)
    return rec


def summarize(rows):
    rows = sorted(rows, key=lambda r: r["subject"])
    def col(key, field):
        return [r[key][field] for r in rows]
    fb, lda, ep = col("fbcsp", "acc"), col("logvar_lda", "acc"), col("logvar_ep", "acc")
    return {"fbcsp_acc": fb, "logvar_lda_acc": lda, "logvar_ep_acc": ep,
            "fbcsp_kappa": col("fbcsp", "kappa"), "logvar_lda_kappa": col("logvar_lda", "kappa"),
            "logvar_ep_kappa": col("logvar_ep", "kappa"),
            "ep_vs_lda": C.paired(ep, lda), "lda_vs_fbcsp": C.paired(lda, fb),
            "ep_vs_fbcsp": C.paired(ep, fb)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    seal = C.seal_or_stop()
    rows = []
    for s in a.subjects:
        path = C.OUT / "front" / f"A{s:02d}_s{a.seed}.json"
        if path.exists():
            import json
            rows.append(json.loads(path.read_text()))
            print("exists", path.name, flush=True)
            continue
        rec = one(s, a.seed)
        rows.append(rec)
        print(f"A{s:02d} fbcsp {rec['fbcsp']['acc']:.3f} lda {rec['logvar_lda']['acc']:.3f} "
              f"ep {rec['logvar_ep']['acc']:.3f}", flush=True)
    if len(rows) == 9:
        C.write_json(C.OUT / "front" / "summary.json",
                     {"stage": "M15", "block": "front", "e_seal": seal, **summarize(rows)})
        print("wrote summary", flush=True)


if __name__ == "__main__":
    main()
