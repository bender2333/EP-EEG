"""M17 block 2: 32 frozen CSP features into a pool, a lake, and a lake with a skip.

Same features as M16. The linear head is not retrained. T session only.
"""
from __future__ import annotations
import argparse
import numpy as np

from . import m14 as M
from . import m15_common as C
from .m16_spatial import csp_bank

KINDS = ("pools", "lake", "lake_skip")
K = 2
U_FIXED = 600


def _add(E, a, b, w, plastic, path):
    a = np.asarray(a, int).ravel()
    b = np.asarray(b, int).ravel()
    Pm, Qm = np.meshgrid(a, b, indexing="ij")
    pre, post = Pm.ravel(), Qm.ravel()
    w = np.asarray(w, float).ravel()
    if w.size != pre.size:
        raise ValueError(f"{path} weight has {w.size} entries for {pre.size} edges")
    E["pre"] += [pre, post]
    E["post"] += [post, pre]
    E["eff"] += [w, w]
    E["pl"] += [np.full(pre.size, plastic)] * 2
    E["path"] += [np.full(2 * pre.size, M.PATHS[path])] * 2
    E["ff"] += [np.ones(pre.size, bool)] * 2


def build(kind, seed, d=32, k=K):
    import warnings
    import cadence as cd
    from cadence.learning import Learner, LearnerConfig, evenodd_neuron_model
    rng = np.random.default_rng(seed)
    n_col = 0 if kind == "pools" else d * k
    n = d + d + n_col + 4
    src = np.arange(d)
    pools = np.arange(d, 2 * d)
    cols = np.arange(2 * d, 2 * d + n_col)
    out = np.arange(n - 4, n)
    E = {"pre": [], "post": [], "eff": [], "pl": [], "path": [], "ff": []}
    # one-way drive, sources stay linear and have no incoming edge
    E["pre"].append(src.copy()); E["post"].append(pools.copy()); E["eff"].append(np.ones(d))
    E["pl"].append(np.zeros(d, bool)); E["path"].append(np.full(d, M.PATHS["pool"]))
    E["ff"].append(np.zeros(d, bool))
    if n_col:
        for i in range(d):
            _add(E, pools[i:i + 1], cols[i * k:(i + 1) * k], rng.normal(0, 1.0, k), True, "read")
        iu, ju = np.triu_indices(n_col, 1)
        w = rng.normal(0, 1.0 / np.sqrt(n_col), len(iu))
        E["pre"] += [cols[iu], cols[ju]]; E["post"] += [cols[ju], cols[iu]]; E["eff"] += [w, w]
        E["pl"] += [np.ones(len(iu), bool)] * 2
        E["path"] += [np.full(len(iu), M.PATHS["lat"])] * 2
        E["ff"] += [np.ones(len(iu), bool)] * 2
        _add(E, cols, out, rng.normal(0, 1.0 / np.sqrt(n_col), (n_col, 4)), True, "out")
    else:
        _add(E, pools, out, rng.normal(0, 1.0 / np.sqrt(d), (d, 4)), True, "out")
    if kind == "lake_skip":
        _add(E, pools, out, rng.normal(0, 1.0 / np.sqrt(d), (d, 4)), True, "skip")
    pre, post, eff, pl, path, ff = (np.concatenate(E[key]) for key in ("pre", "post", "eff", "pl", "path", "ff"))
    conn = cd.Connectome.from_synapses(n, pre=pre, post=post)
    order = np.lexsort((pre, post))
    pre, post, eff, pl, path, ff = (v[order] for v in (pre, post, eff, pl, path, ff))
    if not (np.array_equal(conn.pre, pre) and np.array_equal(conn.post, post)):
        raise RuntimeError("csp-net synapse order mismatch")
    odd = np.ones(n); even = np.zeros(n)
    odd[pools] = M.SQ_ODD; even[pools] = 1.0
    if n_col:
        even[cols] = 1.0
    model = evenodd_neuron_model(odd, even, linear=range(d), dt=0.5)
    lip = model.lipschitz_per_neuron(n)
    budgets = {"pool": 0.3, "read": 0.4, "lat": 0.15, "out": 0.4, "skip": 0.4}
    for name, b in budgets.items():
        m = path == M.PATHS[name]
        if not m.any():
            continue
        r = np.bincount(post, np.abs(eff) * lip[pre] * m, minlength=n).max()
        if r > 0:
            eff = np.where(m, eff * b / r, eff)
    free = np.bincount(post, np.abs(eff) * lip[pre] * ff, minlength=n)[d:].max()
    gscale = 1.0
    if free > M.RHO0_MAX:
        gscale = M.RHO0_MAX / float(free)
        eff = np.where(ff, eff * gscale, eff)
    brain = cd.NeuralGraph(conn, model, efficacy=eff, bias=np.zeros(n), dense_limit=64)
    lc = LearnerConfig(beta=0.04, eta=0.3, eta_bias=0.03, centered=True, free_steps=2000, nudged_steps=2000,
                       tolerance=1e-4, nudge="cross_entropy", temperature=0.01, normalize=0.0,
                       normalize_floor=1e-3, momentum=0.0, decay=0.0, scale_cap=1e9, qualified=False,
                       damping=3, surprise=False)
    pn = np.zeros(n, bool)
    pn[d:] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = Learner(brain, list(out), config=lc, plastic_synapses=pl, plastic_neurons=pn, reciprocal=True)
    return {"lr": lr, "n": n, "d": d, "out": out, "ns": d, "kind": kind, "gscale": float(gscale)}


def zscore(train, test):
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd


def run_one(kind, subject, seed, U):
    import time
    from ep_eeg.data import load_subject
    t0 = time.time()
    raw = load_subject(subject)
    ytr, yte = np.asarray(raw["train"]["y"]), np.asarray(raw["val"]["y"])
    Ftr, Fte = csp_bank(raw["train"]["seg"], ytr, raw["val"]["seg"])
    Ztr, Zte = zscore(Ftr, Fte)
    g = build(kind, 22000 + subject + 30 * seed)
    h = {"lr": g["lr"], "n": g["n"], "d": g["d"], "out": g["out"], "ns": g["ns"]}
    info = _train_fixed(h, Ztr, ytr, U, seed, label=f"{kind} A{subject:02d}")
    pred, ok = C.head_activity(h, Zte)
    score = C.kappa_acc(yte, pred.argmax(1))
    train_pred, train_ok = C.head_activity(h, Ztr)
    train_score = C.kappa_acc(ytr, train_pred.argmax(1))
    rec = {"stage": "M17", "block": "cspnet", "kind": kind, "subject": subject, "seed": seed, "U": U,
           "split": "val", "updates": info["updates"], "refused": info["refused"],
           "train_acc": train_score["acc"], "train_kappa": train_score["kappa"],
           "train_settle_rate": float(np.mean(train_ok)), "settle_rate": float(np.mean(ok)),
           "rejected_trials": int((~np.asarray(ok)).sum()), **score, "gscale": g["gscale"],
           "wall_s": time.time() - t0}
    return rec


def _train_fixed(h, X, y, U, seed, batch=32, label=""):
    """Exactly U attempts. No early stop, so the three architectures see the same update budget."""
    import time
    import warnings
    from cadence.learning import LearningPhaseError
    rng = np.random.default_rng(17000 + seed)
    t0 = time.time()
    y = np.asarray(y)
    nb = max(1, len(X) // batch)
    done = refused = 0
    perm = rng.permutation(len(X))
    for u in range(U):
        if u % nb == 0:
            perm = rng.permutation(len(X))
        sl = perm[(u % nb) * batch:(u % nb + 1) * batch]
        if len(sl) == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                h["lr"].step(C._drive_head(h, X[sl]), y[sl])
            done += 1
        except LearningPhaseError:
            refused += 1
        if label and ((u + 1) % 200 == 0 or u + 1 == U):
            print(f"[{label}] u={u + 1} refused={refused} t={time.time() - t0:.0f}", flush=True)
    return {"updates": done, "refused": refused}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--subjects", nargs="+", type=int, default=[1, 3, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--U", type=int, default=U_FIXED)
    a = ap.parse_args(argv)
    C.seal_or_stop()
    out = C.OUT.parent / "M17" / "cspnet"
    for kind in a.kinds:
        for s in a.subjects:
            path = out / f"{kind}_U{a.U}_A{s:02d}_s{a.seed}.json"
            if path.exists():
                print("exists", path.name, flush=True)
                continue
            rec = run_one(kind, s, a.seed, a.U)
            C.write_json(path, rec)
            print(f"{kind} A{s:02d} acc {rec['acc']:.3f} train {rec['train_acc']} updates {rec['updates']}", flush=True)


if __name__ == "__main__":
    main()
