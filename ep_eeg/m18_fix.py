"""M18 block 2 and 3: layer-wise step scales and a local class current.

T session only. The target relative step is the preregistered constant 3e-3.
"""
from __future__ import annotations
import argparse
import time
import numpy as np

from . import m14 as M
from . import m15_common as C
from .m15_align import tied_step
from .m18_synth import make_synthetic, principal_angle_deg, subspace_energy

TARGET_REL = 3e-3
BETA_LOCAL = 0.04
MOMENTUM = 0.9
MODES = ("baseline", "lars", "rms_mom", "local_pool", "local_column")


def pathway_scale(rel_median, target=TARGET_REL, lo=1.0, hi=100.0):
    if rel_median is None or not np.isfinite(rel_median) or rel_median <= 0:
        return float(hi)
    return float(np.clip(target / float(rel_median), lo, hi))


def class_prototypes(n_units, n_class=4, seed=18181):
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(n_units, n_class))
    R /= np.sqrt(np.mean(R ** 2, axis=0, keepdims=True))
    return R


def _phases(lr, drive, y, extra=None):
    from cadence.learning import LearningPhaseError
    beta = float(lr.config.beta)
    target = lr.targets(np.asarray(y))
    free = lr._qualified_phase(drive, None, lr.config.free_steps)
    if not np.all(free.qualified):
        raise LearningPhaseError("free", {"free": free})
    def nudged(sign):
        d = drive if extra is None else drive + extra(sign)
        phase = lr._qualified_phase(d, free.state, lr.config.nudged_steps, lr.nudge_for(target, sign * beta))
        if not np.all(phase.qualified):
            name = "nudged" if sign > 0 else "opposite"
            raise LearningPhaseError(name, {name: phase})
        return phase
    return free, nudged(+1.0), nudged(-1.0)


def _scales(g, delta, eta):
    eff = np.abs(np.asarray(g["lr"].brain.efficacy, float))
    out = {}
    for name, pid in M.PATHS.items():
        m = (g["path"] == pid) & g["pl"]
        if not np.any(m):
            continue
        rel = float(np.median(np.abs(eta * delta[m]) / (eff[m] + 1e-8)))
        out[name] = pathway_scale(rel)
    return out


def lars_step(g, X, y):
    lr = g["lr"]
    drive = M.drive(g, X)
    free, nudged, opposite = _phases(lr, drive, y)
    edges, neurons = lr.contrast(free.state, nudged.state, opposite.state)
    delta = tied_step(lr, edges)
    eta = float(lr.config.eta)
    base = np.array(lr.synapse_rate, float)
    rate = base.copy()
    scales = _scales(g, delta, eta)
    for name, pid in M.PATHS.items():
        if name not in scales:
            continue
        rate[(g["path"] == pid) & g["pl"]] *= scales[name]
    lr.synapse_rate = rate
    try:
        lr.apply(eta * np.asarray(edges, float), float(lr.config.eta_bias) * np.asarray(neurons, float))
    finally:
        lr.synapse_rate = base
    M.project(g)
    return scales


def rms_step(g, X, y):
    lr = g["lr"]
    drive = M.drive(g, X)
    free, nudged, opposite = _phases(lr, drive, y)
    edges, neurons = lr.contrast(free.state, nudged.state, opposite.state)
    delta = tied_step(lr, edges)
    eta = float(lr.config.eta)
    eff = np.abs(np.asarray(lr.brain.efficacy, float))
    new = np.zeros_like(delta)
    for name, pid in M.PATHS.items():
        m = (g["path"] == pid) & g["pl"]
        if not np.any(m):
            continue
        rms = float(np.sqrt(np.mean(delta[m] ** 2)))
        unit = delta[m] / (rms + 1e-12)
        rel_unit = np.abs(eta * unit) / (eff[m] + 1e-8)
        new[m] = unit * pathway_scale(float(np.median(rel_unit)))
    step = eta * new
    vel = MOMENTUM * g["velocity"] + (1.0 - MOMENTUM) * step
    g["velocity"] = vel
    efficacy = np.array(lr.brain.efficacy, float)
    keep = np.asarray(lr.plastic_synapses)
    efficacy = np.where(keep, efficacy + vel, efficacy)
    bias = np.array(lr.brain.bias, float)
    pn = np.asarray(lr.plastic_neurons)
    bias = np.where(pn, bias + float(lr.config.eta_bias) * np.asarray(neurons, float), bias)
    lr.brain = lr.brain.with_parameters(efficacy=efficacy, bias=bias)
    M.project(g)


def _extra(g, y):
    proto = g["proto"]
    idx = g["local_idx"]
    y = np.asarray(y)

    def extra(sign):
        drive = np.zeros((len(y), g["n"]))
        drive[:, idx] = sign * BETA_LOCAL * proto[:, y].T
        return drive
    return extra


def local_step(g, X, y):
    lr = g["lr"]
    drive = M.drive(g, X)
    free, nudged, opposite = _phases(lr, drive, y, extra=_extra(g, y))
    edges, neurons = lr.contrast(free.state, nudged.state, opposite.state)
    lr.apply(float(lr.config.eta) * np.asarray(edges, float), float(lr.config.eta_bias) * np.asarray(neurons, float))
    M.project(g)


def _prepare(mode, subject, seed):
    init = 19000 + seed if subject is None else 14500 + 100 * subject + seed
    g = M.build("C", init, eta=0.3)
    if mode == "rms_mom":
        g["velocity"] = np.zeros(g["lr"].brain.connectome.synapses)
    if mode == "local_pool":
        g["local_idx"] = np.asarray(g["pools"], int)
        g["proto"] = class_prototypes(len(g["local_idx"]))
    elif mode == "local_column":
        g["local_idx"] = np.asarray(g["cols"], int)
        g["proto"] = class_prototypes(len(g["local_idx"]))
    return g


def _xy(data):
    X = np.asarray(data["Xtr"])
    flat = X.reshape(-1, X.shape[-1])
    y = np.asarray(data["ytr"])
    if X.ndim == 3:
        y = np.repeat(y, X.shape[1])
    return flat, y


def _evaluate(g, data):
    from eval.scoring import aggregate, fallback_class, score
    X = np.asarray(data["Xte"])
    if X.ndim == 3:
        ev, _, _ = M.evaluate(g, X, data["yte"], data["ytr"])
        ev.pop("pred", None)
        return ev
    eq = g["lr"].brain.equilibrate(M.drive(g, X), budget=M.BUDGET, tolerance=M.EVAL_TOL, damping=3)
    act = np.asarray(eq.state.activation)[:, g["out"]]
    ok = np.asarray(eq.converged) & np.all(np.isfinite(act), axis=1)
    pred, _rej = aggregate(act.reshape(len(X), 1, 4), ok.reshape(len(X), 1), fallback_class(data["ytr"]))
    return score(data["yte"], pred, ok.reshape(len(X), 1))


def _pool_probe(g, data):
    from ep_eeg.m14_probe import lda_scores, settle_split
    Xtr = np.asarray(data["Xtr"])
    if Xtr.ndim == 3:
        tr, _ = settle_split(g, data["Xtr"], data["ytr"], data["ytr"])
        va, _ = settle_split(g, data["Xte"], data["yte"], data["ytr"])
        return lda_scores(tr["pools"], data["ytr"], va["pools"], data["yte"])
    def acts(X):
        eq = g["lr"].brain.equilibrate(M.drive(g, X), budget=M.BUDGET, tolerance=M.EVAL_TOL, damping=3)
        return np.asarray(eq.state.activation)[:, g["pools"]]
    return lda_scores(acts(data["Xtr"]), data["ytr"], acts(data["Xte"]), data["yte"])


def baseline_step(g, X, y):
    M.ep_step(g, X, y)
    M.project(g)


def _csp_overlap(W, W0, seg, y):
    from .m16_spatial import broadband_csp
    Wc = broadband_csp(np.asarray(seg), np.asarray(y))
    U = np.asarray(Wc[:4], float).T
    return {"csp_energy": subspace_energy(W, U), "csp_energy_init": subspace_energy(W0, U)}


def train_mode(mode, data, subject, seed, U, eval_every=50, log=print):
    from cadence.learning import LearningPhaseError
    g = _prepare(mode, subject, seed)
    Usub = data.get("U")
    energy0 = None if Usub is None else subspace_energy(M.front_matrix(g), Usub)
    X, y = _xy(data)
    batch = 128 if mode == "rms_mom" else 64
    rng = np.random.default_rng(14000 + seed + (0 if subject is None else 100 * subject))
    nb = max(1, len(X) // batch)
    refused = 0
    slow = []
    W0 = M.front_matrix(g)
    t0 = time.time()
    step_fn = {"baseline": lambda Xb, yb: baseline_step(g, Xb, yb),
               "lars": lambda Xb, yb: lars_step(g, Xb, yb),
               "rms_mom": lambda Xb, yb: rms_step(g, Xb, yb),
               "local_pool": lambda Xb, yb: local_step(g, Xb, yb),
               "local_column": lambda Xb, yb: local_step(g, Xb, yb)}[mode]
    for u in range(U):
        if u % nb == 0:
            perm = rng.permutation(len(X))
        sl = perm[(u % nb) * batch:(u % nb + 1) * batch]
        ts = time.time()
        try:
            step_fn(X[sl], y[sl])
        except LearningPhaseError:
            refused += 1
        slow.append(time.time() - ts)
        if u + 1 == 20:
            med = float(np.median(slow))
            if med > 2.0 or refused > 10:
                log(f"crash-stop u=20 median_step={med:.2f}s refused={refused}")
                break
        if (u + 1) % eval_every == 0 or u + 1 == U:
            ev = _evaluate(g, data)
            rel = M.front_stats(g, W0)["rel_change"]
            energy = None if Usub is None else subspace_energy(M.front_matrix(g), Usub)
            log(f"u={u + 1} acc={ev['acc']:.3f} k={ev['kappa']:.3f} settle={ev['settle_rate']:.3f} "
                f"rej={ev['rejected_trials']} Wrel={rel:.3f} E={energy} refused={refused} t={time.time() - t0:.0f}")
    ev = _evaluate(g, data)
    front = M.front_stats(g, W0)
    ran = 0 if not slow else int(len(slow))
    rec = {"stage": "M18", "mode": mode, "subject": subject, "seed": seed, "U": U, "updates": ran,
           "split": "val", "refused": refused, "wall_s": time.time() - t0,
           "median_step_s": float(np.median(slow)) if slow else None,
           "test": ev, "front_rel": front["rel_change"], "cos_car": front["cos_car"],
           "certified_end": bool(M.certificate(g).get("certified")), "energy0": energy0,
           "crashed": bool(ran < U)}
    if Usub is not None:
        W = M.front_matrix(g)
        rec["energy"] = subspace_energy(W, Usub)
        rec["angle_deg"] = principal_angle_deg(W, Usub)
    try:
        rec["pool_probe"] = _pool_probe(g, data)
    except Exception as exc:  # a probe failure should not erase the decision
        rec["pool_probe_error"] = str(exc)
        log(f"pool probe failed: {exc}")
    if subject is not None and "seg_tr" in data:
        rec.update(_csp_overlap(M.front_matrix(g), W0, data["seg_tr"], data["ytr"]))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--subjects", nargs="+", type=int, default=[1, 3, 8])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--U", type=int, default=250)
    a = ap.parse_args(argv)
    C.seal_or_stop()
    if a.synth:
        out = C.OUT.parent / "M18" / "synth"
        for seed in a.seeds:
            path = out / f"{a.mode}_s{seed}_U{a.U}.json"
            if path.exists():
                print("exists", path.name, flush=True)
                continue
            data = make_synthetic(seed)
            rec = train_mode(a.mode, data, None, seed, a.U, eval_every=50,
                             log=lambda m, s=seed: print(f"[synth {a.mode} s{s}] {m}", flush=True))
            C.write_json(path, rec)
            print(f"synth {a.mode} s{seed} acc {rec['test']['acc']:.3f} E {rec.get('energy')}", flush=True)
        return
    out = C.OUT.parent / "M18" / "real"
    for seed in a.seeds:
        for subject in a.subjects:
            path = out / f"{a.mode}_U{a.U}_A{subject:02d}_s{seed}.json"
            if path.exists():
                print("exists", path.name, flush=True)
                continue
            data = C.load_t(subject)
            rec = train_mode(a.mode, data, subject, seed, a.U, eval_every=min(250, a.U),
                             log=lambda m, s=subject: print(f"[{a.mode} A{s:02d}] {m}", flush=True))
            C.write_json(path, rec)
            print(f"{a.mode} A{subject:02d} acc {rec['test']['acc']:.3f} W {rec['front_rel']:.3f}", flush=True)


if __name__ == "__main__":
    main()
