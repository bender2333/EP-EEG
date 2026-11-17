"""M17 block 1: per-layer step size and coherence, then one front-end rate scale.

T session only. The rate multiplier is estimated on one training batch of A01, not on validation accuracy.
"""
from __future__ import annotations
import argparse
import numpy as np

from dataclasses import replace

from . import m14 as M
from . import m15_common as C
from .m15_align import tied_step

BATCH = 32
BETA = 0.04


def coherence(rows):
    """rows (batch, params). |mean| / mean(|.|). 1 = identical sign and no cancellation."""
    rows = np.asarray(rows, float)
    return np.abs(rows.mean(0)) / (np.abs(rows).mean(0) + 1e-15)


def _batch(data, subject):
    X = data["Xtr"].reshape(-1, data["Xtr"].shape[-1])
    y = np.repeat(data["ytr"], M.N_SUB)
    rng = np.random.default_rng(17017 + subject)
    take = rng.choice(len(X), size=BATCH, replace=False)
    return X[take], y[take]


def _pack(U, w, eta):
    rel = np.abs(eta * U.mean(0)) / (w + 1e-8)
    coh = coherence(U)
    return {"n": int(U.shape[1]), "rel_median": float(np.median(rel)), "rel_p90": float(np.quantile(rel, 0.9)),
            "coherence_median": float(np.median(coh)),
            "noise_median": float(np.median(U.std(0) / (np.abs(U.mean(0)) + 1e-12)))}


def layer_steps(g, X, y, beta=BETA):
    """One centered contrast. Per-example steps are tied the same way Learner.step ties them, before eta."""
    lr = g["lr"]
    drive = M.drive(g, X)
    saved = lr.config
    lr.config = replace(saved, beta=float(beta))
    target = lr.targets(np.asarray(y))
    free = lr.brain.equilibrate(drive, budget=M.BUDGET, tolerance=M.TOL, damping=3)
    nudged = lr.brain.equilibrate(drive, state=free.state, budget=M.BUDGET, tolerance=M.TOL, damping=3,
                                   nudge=lr.nudge_for(target, beta))
    opposite = lr.brain.equilibrate(drive, state=free.state, budget=M.BUDGET, tolerance=M.TOL, damping=3,
                                     nudge=lr.nudge_for(target, -beta))
    hebb, _ = lr.contrast_rows(free.state, nudged.state, opposite.state)
    lr.config = saved
    phase = {"free_ok": float(np.mean(free.converged)), "nudged_ok": float(np.mean(nudged.converged)),
             "opposite_ok": float(np.mean(opposite.converged))}
    # rate, then pair-mean, matching Learner.apply. Front groups are one-way, so the pair step is a no-op there.
    scaled = hebb * lr.synapse_rate if lr.synapse_rate is not None else hebb
    rev = np.asarray(lr.reverse)
    if len(getattr(lr, "_paired", ())):
        paired = np.asarray(lr._paired)
        scaled = scaled.copy()
        scaled[:, paired] = 0.5 * (scaled[:, paired] + scaled[:, np.asarray(lr._paired_reverse)])
    eff = np.asarray(lr.brain.efficacy, float)
    eta = float(saved.eta)
    layers = {}
    gids = np.array(sorted({int(v) for v in g["grp"] if int(v) >= 0}))
    if len(gids):
        rows, ws = [], []
        for gid in gids:
            members = np.flatnonzero(g["grp"] == gid)
            rows.append(scaled[:, members].mean(1))
            ws.append(abs(float(eff[members[0]])))
        layers["src"] = _pack(np.stack(rows, 1), np.asarray(ws), eta)
    idx = np.arange(len(rev))
    for name, pid in M.PATHS.items():
        if name in ("src", "pool"):
            continue
        pair = np.flatnonzero((g["path"] == pid) & g["pl"] & (rev >= 0) & (idx < rev))
        if not len(pair):
            continue
        # pair-mean already applied on `scaled`; both ends carry the same value
        u = scaled[:, pair]
        w = np.abs(eff[pair])
        layers[name] = _pack(u, w, eta)
    tied = tied_step(lr, hebb.mean(0))
    return {"phase_ok": phase, "tied_norm": float(np.linalg.norm(tied)), "layers": layers}


def rate_scale(layers):
    """Outlet-pathway relative step over the front relative step, clipped. Validation accuracy is not an input.

    The reference is the median of the other learnable pathways, and when "out" is present it is used
    directly, matching the preregistered sentence (出水口相对步长 / 前端相对步长).
    """
    front = layers["src"]["rel_median"]
    ref = layers["out"]["rel_median"] if "out" in layers else float(np.median(
        [layers[k]["rel_median"] for k in layers if k != "src"]))
    if front <= 0:
        return 100.0, None
    raw = float(ref / front)
    return float(np.clip(raw, 1.0, 100.0)), raw


def rule_flags(layers):
    """The preregistered reading. Both flags can be true; neither true is "in between"."""
    front = layers["src"]
    others = [layers[k]["rel_median"] for k in layers if k != "src"]
    ref = float(np.median(others))
    small = bool(front["rel_median"] < 0.2 * ref)
    coherent = bool(front["coherence_median"] >= 0.5)
    cancelling = bool(front["coherence_median"] < 0.2)
    if small and coherent and not cancelling:
        label = "small_coherent"
    elif cancelling and not (small and coherent):
        label = "cancelling"
    elif small and coherent and cancelling:
        label = "small_and_cancelling"
    else:
        label = "in_between"
    return {"small": small, "coherent": coherent, "cancelling": cancelling, "label": label,
            "front_over_other": float(front["rel_median"] / (ref + 1e-15))}


def measure_one(arm, subject, trained):
    if trained:
        g, meta = C.load_trained(arm, subject, 0)
    else:
        eta = 1.0 if arm == "A" else 0.3
        g = M.build(arm, 14500 + 100 * subject, eta=eta)
        meta = {"arm": arm, "subject": subject, "seed": 0, "U": 0, "weights": "init", "eta": eta}
    data = C.load_t(subject)
    X, y = _batch(data, subject)
    layers = layer_steps(g, X, y)
    rec = {**meta, "stage": "M17", "block": "move", "trained": bool(trained), "split": "train",
           "beta": BETA, "batch": BATCH, "layers": layers["layers"], "phase_ok": layers["phase_ok"],
           "tied_norm": layers["tied_norm"]}
    if "src" in rec["layers"] and "out" in rec["layers"]:
        rec["scale"], rec["scale_raw"] = rate_scale(rec["layers"])
        rec["rule"] = rule_flags(rec["layers"])
    return rec


def train_front_rate(subject, scale, seed, U):
    g = M.build("C", 14500 + 100 * subject + seed, eta=0.3)
    rate = np.array(g["lr"].synapse_rate, float)
    rate[g["path"] == M.PATHS["src"]] *= scale
    g["lr"].synapse_rate = rate
    data = C.load_t(subject)
    tr = M.train(g, data, U, seed, eval_every=min(250, U),
                 log=lambda m, s=subject: print(f"[front_rate A{s:02d}] {m}", flush=True))
    ev, _, _ = M.evaluate(g, data["Xte"], data["yte"], data["ytr"])
    ev.pop("pred", None)
    row = tr["curve"][-1]
    cos_csp, cos_matched = _csp_similarity(g, subject)
    return {"stage": "M17", "block": "front_rate", "subject": subject, "seed": seed, "U": U,
            "scale": scale, "split": "val", "test": ev, "front_rel": row["front_rel_change"],
            "cos_car": row["front_cos_car"], "cos_csp": cos_csp, "cos_csp_matched": cos_matched,
            "refused": tr["refused"], "certified_end": tr["certified_end"], "wall_s": tr["wall_s"]}


def _csp_similarity(g, subject):
    """Plain matrix cosine is the preregistered comparison. The matched-row value is exploratory."""
    from ep_eeg.data import load_subject
    from .m16_spatial import broadband_csp
    raw = load_subject(subject)
    Wc = broadband_csp(np.asarray(raw["train"]["seg"]), np.asarray(raw["train"]["y"]))
    Wm = M.front_matrix(g)

    def cos(a, b):
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    A = Wm / (np.linalg.norm(Wm, axis=1, keepdims=True) + 1e-12)
    B = Wc / (np.linalg.norm(Wc, axis=1, keepdims=True) + 1e-12)
    S = np.abs(A @ B.T)
    used, total = set(), 0.0
    for i in np.argsort(-S.max(1)):
        for j in np.argsort(-S[i]):
            j = int(j)
            if j not in used:
                used.add(j)
                total += float(S[i, j])
                break
    return cos(Wm, Wc), total / len(A)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--subjects", nargs="+", type=int, default=[1, 3, 5, 8])
    ap.add_argument("--arms", nargs="+", default=["C", "C_skip", "C_nolake"])
    ap.add_argument("--U", type=int, default=500)
    a = ap.parse_args(argv)
    C.seal_or_stop()
    out = C.OUT.parent / "M17" / "move"
    if a.measure:
        for arm in a.arms:
            for s in a.subjects:
                for trained in (False, True):
                    tag = "trained" if trained else "init"
                    path = out / f"{arm}_A{s:02d}_{tag}.json"
                    if path.exists():
                        print("exists", path.name, flush=True)
                        continue
                    rec = measure_one(arm, s, trained)
                    C.write_json(path, rec)
                    src = rec["layers"].get("src", {})
                    print(f"{arm} A{s:02d} {tag} front_rel {src.get('rel_median')} "
                          f"coh {src.get('coherence_median')} scale {rec.get('scale')}", flush=True)
    if a.train:
        ref = out / "C_A01_trained.json"
        if not ref.exists():
            C.write_json(ref, measure_one("C", 1, True))
        import json
        scale = json.loads(ref.read_text())["scale"]
        print("frozen scale", scale, flush=True)
        for s in a.subjects:
            path = out / f"front_rate_U{a.U}_A{s:02d}_s0.json"
            if path.exists():
                print("exists", path.name, flush=True)
                continue
            rec = train_front_rate(s, scale, 0, a.U)
            C.write_json(path, rec)
            print(f"front_rate A{s:02d} acc {rec['test']['acc']:.3f} Wrel {rec['front_rel']:.4f}", flush=True)


if __name__ == "__main__":
    main()
