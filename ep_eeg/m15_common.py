"""M15 shared pieces. T session only. Importing this module does not read session E."""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "M14_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ.setdefault("MNE_DATA", "/agent/repos/EP-EEG/mne_data")

import json
from pathlib import Path
import numpy as np

from . import m14 as M

ROOT = M.ROOT
OUT = ROOT / "receipts" / "M15"


def seal_or_stop():
    from ep_eeg.m13_common import e_seal
    seal = e_seal()
    if seal["matches"]:
        raise RuntimeError(f"E files present during an M15 T-only run: {seal['matches'][:3]}")
    return seal


def load_t(subject):
    seal_or_stop()
    return M.load(subject, "val")


def paired(a, b):
    """Per-subject paired comparison of two length-9 sequences. Two-sided Wilcoxon."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    wins, losses = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    ties = int(len(d) - wins - losses)
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(wilcoxon(d, alternative="two-sided").pvalue)
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(d.mean()),
            "wins": wins, "ties": ties, "losses": losses, "p": p}


def kappa_acc(y, pred):
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    return {"acc": float(accuracy_score(y, pred)), "kappa": float(cohen_kappa_score(y, pred)), "n": int(len(y))}


def logreg_predict(Ftr, ytr, Fte):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(solver="lbfgs", max_iter=800)
    clf.fit(Ftr, ytr)
    return clf.predict(Fte), clf.predict(Ftr)


def ce_loss(act, y, temp):
    """Mean cross-entropy of outlet activities, temperature-scaled, matching the M14 nudge."""
    z = np.asarray(act, float) / temp
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(y)), np.asarray(y)].mean())


def build_head(d, seed, eta=0.3, beta=0.04, temp=0.01, centered=True, nudge="cross_entropy"):
    """One-way linear sources -> 4 odd outlets. Equilibrium propagation trains only those synapses."""
    import warnings
    import cadence as cd
    from cadence.learning import Learner, LearnerConfig, evenodd_neuron_model
    rng = np.random.default_rng(seed)
    n = int(d) + 4
    out = np.arange(d, n)
    pre = np.repeat(np.arange(d), 4)
    post = np.tile(out, d)
    eff = rng.normal(0.0, 1.0 / np.sqrt(d), size=pre.size)
    conn = cd.Connectome.from_synapses(n, pre=pre, post=post)
    order = np.lexsort((pre, post))
    pre, post, eff = pre[order], post[order], eff[order]
    if not (np.array_equal(conn.pre, pre) and np.array_equal(conn.post, post)):
        raise RuntimeError("head synapse order does not match the connectome")
    model = evenodd_neuron_model(np.ones(n), np.zeros(n), linear=range(d), dt=0.5)
    lip = model.lipschitz_per_neuron(n)
    mass = float(np.bincount(post, np.abs(eff) * lip[pre], minlength=n).max())
    if mass > 0:
        eff = eff * 0.5 / mass
    brain = cd.NeuralGraph(conn, model, efficacy=eff, bias=np.zeros(n), dense_limit=64)
    lc = LearnerConfig(beta=beta, eta=eta, eta_bias=eta / 10, centered=centered, free_steps=2000,
                       nudged_steps=2000, tolerance=1e-4, nudge=nudge, temperature=temp, normalize=0.0,
                       normalize_floor=1e-3, momentum=0.0, decay=0.0, scale_cap=1e9, qualified=False,
                       damping=3, surprise=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = Learner(brain, list(out), config=lc, reciprocal=False)
    return {"lr": lr, "n": n, "d": int(d), "out": out, "ns": int(d)}


def _drive_head(h, X):
    D = np.zeros((len(X), h["n"]))
    D[:, :h["d"]] = X
    return D


def head_activity(h, X):
    eq = h["lr"].brain.equilibrate(_drive_head(h, X), budget=2000, tolerance=1e-5, damping=3)
    act = np.asarray(eq.state.activation)[:, h["out"]]
    ok = np.asarray(eq.converged) & np.all(np.isfinite(act), 1)
    return act, ok


def train_head(h, X, y, U, seed, batch=32):
    """Fit the linear head. Returns the last train accuracy and how many updates ran."""
    import warnings
    from cadence.learning import LearningPhaseError
    rng = np.random.default_rng(17000 + seed)
    y = np.asarray(y)
    nb = max(1, len(X) // batch)
    acc = None
    done = 0
    for u in range(U):
        if u % nb == 0:
            perm = rng.permutation(len(X))
        sl = perm[(u % nb) * batch:(u % nb + 1) * batch]
        if len(sl) == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                h["lr"].step(_drive_head(h, X[sl]), y[sl])
            done = u + 1
        except LearningPhaseError:
            continue
        if done % 100 == 0 or u + 1 == U:
            pred, _ = head_activity(h, X)
            acc = float((pred.argmax(1) == y).mean())
            if acc > 0.995 and done >= 100:
                break
    return {"updates": done, "train_acc": acc}


def write_json(path, obj):
    from ep_eeg.m13_common import write_receipt
    write_receipt(path, obj)
    return Path(path)


def load_trained(arm, subject, seed=0):
    from ep_eeg.m14_probe import selection, weight_file
    sel = selection()[arm]
    eta, U = float(sel["eta"]), int(sel["U"])
    g = M.build(arm, 14500 + 100 * subject + seed, eta=eta)
    path = weight_file(arm, subject, seed, eta, U)
    if not path.is_file():
        raise FileNotFoundError(path)
    M.load_weights(g, path)
    return g, {"arm": arm, "subject": subject, "seed": seed, "eta": eta, "U": U, "weights": str(path.relative_to(ROOT))}
