"""M18 block 0: nudge size per layer and one-hop equilibrium gain. T session only.

The batch matches M17 (seed 17017 + subject) so the step-size receipts stay comparable.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import numpy as np

from . import m14 as M
from . import m15_common as C
from .m15_align import tied_step

BETA = 0.04
BATCH = 32
NAMES = ("source", "front", "pool", "column", "outlet")


def layer_index(g):
    lid = np.full(g["n"], -1, np.int8)
    lid[:g["ns"]] = 0
    front = np.asarray(g["entry"]).ravel()
    lid[front[front >= 0]] = 1
    if g["pools"] is not None:
        lid[np.asarray(g["pools"], int)] = 2
    if len(g["cols"]):
        lid[np.asarray(g["cols"], int)] = 3
    lid[np.asarray(g["out"], int)] = 4
    return lid


def _batch(data, subject):
    X = data["Xtr"].reshape(-1, data["Xtr"].shape[-1])
    y = np.repeat(data["ytr"], M.N_SUB)
    rng = np.random.default_rng(17017 + subject)
    take = rng.choice(len(X), size=BATCH, replace=False)
    return X[take], y[take]


def _med(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values))


def hop_gain(pre, post, weights, slope, pre_id, post_id, want_pre, want_post):
    """Median over receivers of slope(post) * sum |W| from the sending layer.

    This is a row-sum, so it is an upper-style gain when presynaptic perturbations share a sign.
    """
    m = (pre_id[pre] == want_pre) & (post_id[post] == want_post)
    if not np.any(m):
        return None
    acc = np.zeros(int(post_id.shape[0]))
    np.add.at(acc, post[m], np.abs(weights[m]) * slope[post[m]])
    receivers = np.flatnonzero(post_id == want_post)
    got = acc[receivers]
    got = got[got > 0]
    return _med(got)


def measure_graph(g, X, y):
    lr = g["lr"]
    lid = layer_index(g)
    drive = M.drive(g, X)
    saved = lr.config
    lr.config = replace(saved, beta=BETA)
    target = lr.targets(np.asarray(y))
    free = lr.brain.equilibrate(drive, budget=M.BUDGET, tolerance=M.TOL, damping=3)
    nudged = lr.brain.equilibrate(drive, state=free.state, budget=M.BUDGET, tolerance=M.TOL, damping=3,
                                   nudge=lr.nudge_for(target, BETA))
    opposite = lr.brain.equilibrate(drive, state=free.state, budget=M.BUDGET, tolerance=M.TOL, damping=3,
                                     nudge=lr.nudge_for(target, -BETA))
    hebb, _ = lr.contrast_rows(free.state, nudged.state, opposite.state)
    lr.config = saved
    s_free = np.asarray(free.state.activation)
    ds = np.abs(np.asarray(nudged.state.activation) - np.asarray(opposite.state.activation))
    slope = np.abs(g["model"].slope_at(np.asarray(free.state.v)))
    layers = {}
    for i, name in enumerate(NAMES):
        m = lid == i
        if not np.any(m):
            continue
        layers[name] = {
            "n": int(m.sum()),
            "abs_s": _med(np.mean(np.abs(s_free[:, m]), 0)),
            "abs_delta": _med(np.mean(ds[:, m], 0)),
            "abs_slope": _med(np.mean(slope[:, m], 0)),
        }
    conn = lr.brain.connectome
    w = np.asarray(lr.brain.weights, float)
    slope_m = np.mean(slope, 0)
    gains = {}
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 3), (3, 2), (2, 1), (2, 4), (4, 2)):
        if a >= len(NAMES) or b >= len(NAMES):
            continue
        key = f"{NAMES[a]}->{NAMES[b]}"
        gains[key] = hop_gain(conn.pre, conn.post, w, slope_m, lid, lid, a, b)
    # relative steps, same tying as M17, from this batch
    scaled = hebb * lr.synapse_rate if lr.synapse_rate is not None else hebb
    rev = np.asarray(lr.reverse)
    if len(lr._paired):
        scaled = scaled.copy()
        paired = np.asarray(lr._paired)
        scaled[:, paired] = 0.5 * (scaled[:, paired] + scaled[:, np.asarray(lr._paired_reverse)])
    eta = float(saved.eta)
    eff = np.abs(np.asarray(lr.brain.efficacy, float))
    rel = {}
    gids = [int(v) for v in np.unique(g["grp"]) if int(v) >= 0]
    if gids:
        rows, ws = [], []
        for gid in gids:
            members = np.flatnonzero(g["grp"] == gid)
            rows.append(scaled[:, members].mean(1))
            ws.append(eff[members[0]])
        step = np.abs(eta * np.mean(np.stack(rows, 1), 0))
        rel["src"] = _med(step / (np.asarray(ws) + 1e-8))
    idx = np.arange(len(rev))
    for name, pid in M.PATHS.items():
        if name in ("src", "pool"):
            continue
        pair = np.flatnonzero((g["path"] == pid) & g["pl"] & (rev >= 0) & (idx < rev))
        if not len(pair):
            continue
        step = np.abs(eta * scaled[:, pair].mean(0))
        rel[name] = _med(step / (eff[pair] + 1e-8))
    lip = g["model"].lipschitz_per_neuron(g["n"])
    is_free = np.ones(g["n"], bool)
    is_free[:g["ns"]] = False
    keep = is_free[conn.pre] & is_free[conn.post]
    incoming = np.bincount(conn.post[keep], weights=np.abs(w[keep]) * lip[conn.pre[keep]], minlength=g["n"])
    mass = {}
    for i, name in enumerate(NAMES):
        m = (lid == i) & is_free
        if np.any(m):
            mass[name] = {"max": float(incoming[m].max()), "slack_to_0.8": float(0.8 - incoming[m].max())}
    ratio = None
    if rel.get("src") and rel.get("out"):
        ratio = rel["out"] / rel["src"]
    if gains.get("outlet->column") and gains.get("column->pool") and gains.get("pool->front"):
        back = [gains["outlet->column"], gains["column->pool"], gains["pool->front"]]
    elif gains.get("outlet->pool") and gains.get("pool->front"):
        back = [gains["outlet->pool"], gains["pool->front"]]
    else:
        back = []
    product = float(np.prod(back)) if back else None
    return {
        "phase_ok": {"free": float(np.mean(free.converged)), "nudged": float(np.mean(nudged.converged)),
                     "opposite": float(np.mean(opposite.converged))},
        "layers": layers,
        "hop_gain": gains,
        "rel_median": rel,
        "rel_out_over_front": ratio,
        "backward_gain_product": product,
        "row_mass": mass,
        "tied_norm": float(np.linalg.norm(tied_step(lr, hebb.mean(0)))),
    }


def measure_one(arm, subject, trained):
    if trained:
        g, meta = C.load_trained(arm, subject, 0)
    else:
        g = M.build(arm, 14500 + 100 * subject, eta=0.3)
        meta = {"arm": arm, "subject": subject, "seed": 0, "U": 0, "weights": "init", "eta": 0.3}
    data = C.load_t(subject)
    X, y = _batch(data, subject)
    rec = measure_graph(g, X, y)
    front = rec["layers"].get("front", {}).get("abs_delta")
    outlet = rec["layers"].get("outlet", {}).get("abs_delta")
    src_s = rec["layers"].get("source", {}).get("abs_s")
    # outlet presynaptic activity is the column (or the pool, when there is no lake)
    pre_name = "column" if "column" in rec["layers"] else "pool"
    pre_s = rec["layers"].get(pre_name, {}).get("abs_s")
    atten = None if not front or not outlet else front / outlet
    pre_ratio = None if not src_s or not pre_s else src_s / pre_s
    if atten is not None and atten <= 1 / 20:
        cause = "nudge_attenuates"
    else:
        cause = "nudge_reaches"
    if pre_ratio is not None and pre_ratio <= 1 / 5:
        cause = cause + "+small_presynaptic"
    rec.update({**meta, "stage": "M18", "block": "mech", "trained": bool(trained), "split": "train",
                "beta": BETA, "delta_front_over_outlet": atten, "presynaptic_source_over_readout": pre_ratio,
                "presynaptic_readout_layer": pre_name, "attribution": cause})
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", type=int, default=[1, 3, 8])
    a = ap.parse_args(argv)
    C.seal_or_stop()
    out = C.OUT.parent / "M18" / "mech"
    jobs = [("C", s, True) for s in a.subjects] + [("C", 1, False), ("C_nolake", 1, True)]
    for arm, s, trained in jobs:
        tag = "trained" if trained else "init"
        path = out / f"{arm}_A{s:02d}_{tag}.json"
        if path.exists():
            print("exists", path.name, flush=True)
            continue
        rec = measure_one(arm, s, trained)
        C.write_json(path, rec)
        print(f"{arm} A{s:02d} {tag} dF/dO {rec['delta_front_over_outlet']} "
              f"pre {rec['presynaptic_source_over_readout']} R {rec['rel_out_over_front']} "
              f"gain {rec['backward_gain_product']} {rec['attribution']}", flush=True)


if __name__ == "__main__":
    main()
