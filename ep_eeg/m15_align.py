"""M15 block 2: cosine of the EP step and a finite-difference gradient, per pathway.

The finite-difference target is the free-phase cross-entropy at the training temperature.
The EP step is the centered contrast after tying, before it is multiplied by eta.
T session only. Does not update the checkpoint.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import numpy as np

from . import m14 as M
from . import m15_common as C

BETAS = (0.01, 0.04, 0.1)
N_SAMPLE = 16
EPS = 1e-3
BATCH = 32


def tied_step(lr, edges):
    """Contrast after the learner's rate, pair-mean and tie-mean. Not yet times eta."""
    delta = np.array(edges, dtype=float)
    if lr.synapse_rate is not None:
        delta *= lr.synapse_rate
    if len(lr._paired):
        mean = 0.5 * (delta[lr._paired] + delta[lr._paired_reverse])
        delta[lr._paired] = mean
    if len(lr._members):
        total = np.bincount(lr._member_groups, weights=delta[lr._members])
        delta[lr._members] = (total / np.maximum(lr._member_count, 1.0))[lr._member_groups]
    delta[~lr.plastic_synapses] = 0.0
    return delta


def ep_steps(g, drive, y, beta):
    lr = g["lr"]
    saved = lr.config
    lr.config = replace(saved, beta=float(beta))
    target = lr.targets(np.asarray(y))
    free = lr.brain.equilibrate(drive, budget=M.BUDGET, tolerance=M.TOL, damping=3)
    nudged = lr.brain.equilibrate(drive, state=free.state, budget=M.BUDGET, tolerance=M.TOL, damping=3,
                                   nudge=lr.nudge_for(target, beta))
    opposite = lr.brain.equilibrate(drive, state=free.state, budget=M.BUDGET, tolerance=M.TOL, damping=3,
                                     nudge=lr.nudge_for(target, -beta))
    edges, _ = lr.contrast(free.state, nudged.state, opposite.state)
    lr.config = saved
    return tied_step(lr, edges), {"free_ok": float(np.mean(free.converged)),
                                  "nudged_ok": float(np.mean(nudged.converged)),
                                  "opposite_ok": float(np.mean(opposite.converged))}


def loss_of(g, drive, y, eff):
    lr = g["lr"]
    bias = np.asarray(lr.brain.bias)
    lr.brain = lr.brain.with_parameters(efficacy=eff, bias=bias)
    eq = lr.brain.equilibrate(drive, budget=M.BUDGET, tolerance=M.TOL, damping=3)
    act = np.asarray(eq.state.activation)[:, g["out"]]
    return C.ce_loss(act, y, M.TEMP), float(np.mean(eq.converged))


def sample_groups(g, rng):
    """Plastic parameters: front tie-groups, and reciprocal pairs on the other pathways."""
    lr = g["lr"]
    groups = []
    front = np.array(sorted({int(v) for v in g["grp"] if v >= 0}))
    front = rng.permutation(front)
    for gid in front[:N_SAMPLE]:
        members = np.flatnonzero(g["grp"] == gid)
        if len(members):
            groups.append(("src", gid, members))
    rev = lr.reverse
    idx = np.arange(len(rev))
    for name, pid in M.PATHS.items():
        if name in ("src", "pool"):
            continue
        pair = np.flatnonzero((g["path"] == pid) & g["pl"] & (rev >= 0) & (idx < rev))
        if not len(pair):
            continue
        take = rng.choice(pair, size=min(N_SAMPLE, len(pair)), replace=False)
        for i in np.sort(take):
            groups.append((name, int(i), np.array([int(i), int(rev[i])])))
    return groups


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return None
    return float(a @ b / (na * nb))


def align_one(arm, subject, seed, trained):
    if trained:
        g, meta = C.load_trained(arm, subject, seed)
    else:
        sel_eta = 0.3 if arm != "A" else 1.0
        g = M.build(arm, 14500 + 100 * subject + seed, eta=sel_eta)
        meta = {"arm": arm, "subject": subject, "seed": seed, "eta": sel_eta, "U": 0, "weights": "init"}
    data = C.load_t(subject)
    X = data["Xtr"].reshape(-1, data["Xtr"].shape[-1])
    y = np.repeat(data["ytr"], M.N_SUB)
    rng = np.random.default_rng(15000 + subject + 100 * seed + (0 if trained else 7))
    take = rng.choice(len(X), size=BATCH, replace=False)
    drive = M.drive(g, X[take])
    yb = y[take]
    groups = sample_groups(g, np.random.default_rng(16000 + subject))
    eff0 = np.asarray(g["lr"].brain.efficacy).copy()
    bias0 = np.asarray(g["lr"].brain.bias).copy()
    fd = []
    for name, gid, members in groups:
        plus, minus = eff0.copy(), eff0.copy()
        plus[members] += EPS
        minus[members] -= EPS
        lp, okp = loss_of(g, drive, yb, plus)
        lm, okm = loss_of(g, drive, yb, minus)
        fd.append({"path": name, "id": int(gid), "grad": (lp - lm) / (2 * EPS), "ok": min(okp, okm)})
    g["lr"].brain = g["lr"].brain.with_parameters(efficacy=eff0, bias=bias0)
    by_beta = {}
    for beta in BETAS:
        delta, phase = ep_steps(g, drive, yb, beta)
        per = {}
        for name in sorted({row["path"] for row in fd}):
            ep = []
            grad = []
            for (pname, gid, members), row in zip(groups, fd):
                if pname != name:
                    continue
                ep.append(float(delta[members[0]]))
                grad.append(row["grad"])
            ep, grad = np.asarray(ep), np.asarray(grad)
            per[name] = {"n": int(len(ep)), "cosine": cosine(ep, grad),
                         "norm_ep": float(np.linalg.norm(ep)), "norm_fd": float(np.linalg.norm(grad)),
                         "norm_ratio": (None if np.linalg.norm(grad) < 1e-15 else float(np.linalg.norm(ep) / np.linalg.norm(grad)))}
        by_beta[str(beta)] = {"phase_ok": phase, "paths": per}
    g["lr"].brain = g["lr"].brain.with_parameters(efficacy=eff0, bias=bias0)
    return {**meta, "stage": "M15", "block": "align", "trained": bool(trained), "split": "val",
            "batch": BATCH, "eps": EPS, "betas": by_beta}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["C", "C_skip", "C_nolake", "A"])
    ap.add_argument("--subjects", nargs="+", type=int, default=[1, 3, 5, 8])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    C.seal_or_stop()
    for arm in a.arms:
        for s in a.subjects:
            for trained in (False, True):
                tag = "trained" if trained else "init"
                path = C.OUT / "align" / f"{arm}_A{s:02d}_s{a.seed}_{tag}.json"
                if path.exists():
                    print("exists", path.name, flush=True)
                    continue
                rec = align_one(arm, s, a.seed, trained)
                C.write_json(path, rec)
                line = rec["betas"]["0.04"]["paths"]
                print(f"{arm} A{s:02d} {tag} " + " ".join(
                    f"{k}={v['cosine'] if v['cosine'] is not None else float('nan'):.2f}" for k, v in line.items()), flush=True)


if __name__ == "__main__":
    main()
