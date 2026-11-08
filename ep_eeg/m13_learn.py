"""M13 step 4.5 (route 1): re-learn the v0 lake with a geometric field inside its energy (EP stays exact).

What changes against 4.3: the field couples the 22 OUTLETS (signed, trained to 0.5 tanh(x/3) of each lead's
value), not the 4 random-sign columns of a lead, and it has NO self term (the diagonal of the quasi-static
coupling is dropped), so anything it adds has to come from the other electrodes. The coupling is fixed
(non-plastic) symmetric outlet<->outlet synapses

    Q = lam * Goff / max_i sum_j Goff_ij,   G = (I - ell^2 L)^-1 on the scalp mesh (m13_dyn.mesh),
    Goff = G restricted to the 22 electrode nodes, diagonal removed,

so its largest row sum is lam. The rest of the lake (input, column, column<->outlet synapses and biases) is
then trained further by the same EP rule as E0, starting from the E0 checkpoint of (subject, seed).

Conditions (all trained the same number of updates from the same checkpoint, same batches):
  field   true electrode positions
  der     the same field with electrode positions deranged (C.derangement(k), --der k): same coupling strengths and
          degree, wrong geometry
  none    no field: plain further training (separates "more training" from "field")
Evaluation: m13_dyn's static arm (equilibrium per row) on the lake's own outlets, P3, same rows as E0.
All training and testing is within one subject (人 2026-11-07：先不做跨被试).

The gain projection keeps the certified row mass <= CAP counting the fixed field synapses, and scales only
the plastic free->free synapses (the field is never shrunk); a field whose row mass alone reaches CAP is refused.
Adding the field to a v0 lake already at CAP therefore first shrinks the plastic synapses on the outlet rows (and,
symmetrically, the column<->outlet synapses) at build time; field and der start from the same shrunk lake (same
row masses), none from v0 itself. The u = 0 evaluation of every condition is recorded.
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from . import m12_s1 as S                        # noqa: E402
from . import m13_common as C                    # noqa: E402
from . import m13_e0 as E                        # noqa: E402
from . import m13_dyn as M                       # noqa: E402
import cadence as cd                             # noqa: E402
from cadence.learning import Learner, LearnerConfig, LearningPhaseError, evenodd_neuron_model  # noqa: E402

OUT = C.ROOT / "receipts" / "M13" / "L"
ELL, LAM = 0.06, 0.5
EVAL_AT = (500, 1000, 2000, 3000)


def field_matrix(ell=ELL, lam=LAM, perm=None):
    xyz, L = M.mesh()
    G = np.linalg.inv(np.eye(len(xyz)) - ell ** 2 * L)[:C.N_LEADS, :C.N_LEADS]
    G = (G + G.T) / 2; np.fill_diagonal(G, 0.0)
    Q = lam * G / G.sum(1).max()
    if perm is not None:                      # lead c sits at electrode node perm[c]
        p = np.asarray(perm); Q = Q[np.ix_(p, p)]
    return Q


def build(subject, seed, cond, ell=ELL, lam=LAM, eta=E.ETA, der=0, trained=True, cap=S.CAP):
    """v0 lake of (subject, seed) from its E0 checkpoint (trained=False: the untrained E0 initial lake), plus the
    fixed field synapses for cond."""
    g0, init_sha = M.load_lake(subject, seed, trained=trained)
    c0 = g0["conn"]; eff0 = np.asarray(g0["lr"].brain.efficacy, float); bias0 = np.asarray(g0["lr"].brain.bias, float)
    ff0 = np.asarray(g0["ff"], bool); out = np.asarray(g0["out"]); n, ns = g0["n"], g0["ns"]
    pre, post, eff, ff, fx = [c0.pre], [c0.post], [eff0], [ff0], [np.zeros(len(eff0), bool)]
    Q = None
    if cond in ("field", "der"):
        Q = field_matrix(ell, lam, None if cond == "field" else C.derangement(der))
        i, j = np.nonzero(Q)
        pre.append(out[i]); post.append(out[j]); eff.append(Q[i, j]); ff.append(np.ones(len(i), bool)); fx.append(np.ones(len(i), bool))
    pre, post, eff, ff, fx = map(np.concatenate, (pre, post, eff, ff, fx))
    conn = cd.Connectome.from_synapses(n, pre=pre, post=post)
    order = np.lexsort((pre, post)); assert np.array_equal(conn.pre, pre[order]) and np.array_equal(conn.post, post[order])
    eff, ff, fx = eff[order], ff[order], fx[order]
    model = g0["lr"].brain.neuron_model; lip = model.lipschitz_per_neuron(n)
    brain = cd.NeuralGraph(conn, model, efficacy=eff, bias=bias0, dense_limit=4096)
    lc = LearnerConfig(beta=S.BETA, eta=eta, eta_bias=eta / 10, centered=True, free_steps=S.BUDGET, nudged_steps=S.BUDGET,
                       tolerance=S.TOL, nudge="quadratic", temperature=S.TEMP, normalize=0.0, normalize_floor=1e-3,
                       momentum=0.0, decay=0.0, scale_cap=1e9, qualified=True, damping=3, surprise=False)
    pn = np.zeros(n, bool); pn[ns:] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = Learner(brain, list(out), config=lc, plastic_synapses=~fx, plastic_neurons=pn, reciprocal=True)
    g = {"lr": lr, "conn": conn, "ff": ff, "fx": fx, "groups": None, "model": model, "lip": lip, "ns": ns, "n": n, "out": out,
         "cols": g0["cols"], "k": g0["k"], "Q": Q}
    if field_mass(g).max() >= cap:
        raise ValueError("field row mass alone reaches the cap")
    project(g, cap)                           # adding the field: plastic mass on outlet rows makes room for it
    if rho(g) >= 1.0 or not cd.certificate(g["lr"].brain, sources=range(ns)).certified:
        raise RuntimeError("lake with field not certified after projection")
    return g, init_sha


def _mass(g, mask, eff=None):
    eff = np.asarray(g["lr"].brain.efficacy) if eff is None else eff; c = g["conn"]
    m = np.bincount(c.post, np.abs(eff) * g["lip"][c.pre] * mask, minlength=g["n"]); m[:g["ns"]] = 0; return m


def field_mass(g): return _mass(g, g["fx"])
def rho(g): return float(_mass(g, g["ff"]).max())


def project(g, cap=S.CAP):
    """Scale plastic free->free synapses (symmetric, as S.project) so that plastic + fixed field mass <= cap."""
    lr, c = g["lr"], g["conn"]; pl = g["ff"] & ~g["fx"]; fm = field_mass(g)
    eff = np.asarray(lr.brain.efficacy, float).copy(); changed = False
    for _ in range(100):
        pm = _mass(g, pl, eff); over = pm + fm > cap * (1 - 1e-9)
        if not over.any(): break
        f = np.ones(g["n"]); f[over] = (cap - fm[over]) / pm[over] * (1 - 1e-6)
        eff = eff * np.where(pl, np.minimum(f[c.post], f[c.pre]), 1.0); changed = True
    if changed: S.set_params(lr, eff, np.asarray(lr.brain.bias, float))
    return changed


def train(g, Xtr, seed, U, log=print, eta=E.ETA, on_eval=None):
    """E0's training loop (same batches/masks/targets) with the field-aware projection; batches drawn from a
    seed stream distinct from E0's (seed + 1000) so the further training does not replay E0's first epochs."""
    pats = C.all_patterns(); stats = {"accepted": 0, "refused": 0, "projections": 0, "log": []}
    nb = len(Xtr) // E.BATCH; t0 = time.time(); loss = []
    for u in range(U):
        ep, bi = divmod(u, nb)
        rng = np.random.default_rng((4000 + seed + 1000) * 100003 + ep)
        perm = rng.permutation(len(Xtr)); Mk = E.masks_for(rng, len(Xtr), pats)
        idx = perm[bi * E.BATCH:(bi + 1) * E.BATCH]
        D = E.drive_rows(g, Xtr[idx], Mk[idx]); T = E.target(g, Xtr[idx])
        try:
            free, _ = S.ep_step(g, D, T, eta)
            s = np.asarray(free.activation)[:, g["out"]]; loss.append(float(0.5 * ((s - T[:, g["out"]]) ** 2).mean()))
            stats["accepted"] += 1
        except LearningPhaseError:
            stats["refused"] += 1
        if project(g): stats["projections"] += 1
        if rho(g) >= 1.0: raise RuntimeError("rho~ >= 1 after projection")
        if (u + 1) % 100 == 0:
            stats["log"].append({"u": u + 1, "loss": float(np.mean(loss)) if loss else None, "rho": rho(g),
                                 "field_mass_max": float(field_mass(g).max()), "t": time.time() - t0}); loss = []
            log(f"u={u + 1} loss={stats['log'][-1]['loss']} rho={rho(g):.3f} refused={stats['refused']}")
        if on_eval is not None and (u + 1) in EVAL_AT:
            on_eval(u + 1)
    stats["certified_end"] = bool(cd.certificate(g["lr"].brain, sources=range(g["ns"])).certified)
    stats["wall_s"] = time.time() - t0
    return stats


def evaluate(g, subject, split):
    """lake_own P3 on the static equilibrium per row (m13_dyn static arm), with the spatial diagnostic."""
    return M.run_job(subject, 0, "static", {}, split=split, lake_g=g)


def run(subject, seed, cond, split, U, ell=ELL, lam=LAM, der=0):
    t0 = time.time()
    data = C.load_real(subject, split="t04_val45" if split == "test" else "dev_r02_r3")
    seal = C.e_seal()
    if seal["matches"]: raise RuntimeError(f"E files present: {seal['matches']}")
    Xtr = C.rows(data["train"])
    g, init_sha = build(subject, seed, cond, ell, lam, der=der)
    evals = {0: evaluate(g, subject, split)["summary"]}
    def on_eval(u):
        evals[u] = evaluate(g, subject, split)["summary"]
        print(f"[{subject}/{seed}/{cond}] eval u={u} {evals[u]['lake_own_P3']:.4f}", flush=True)
    tr = train(g, Xtr, seed, U, log=lambda m: print(f"[{subject}/{seed}/{cond}] {m}", flush=True), on_eval=on_eval)
    final = evaluate(g, subject, split)
    return {"stage": "M13", "step": "4.5-route1", "cond": cond, "split": split, "subject": subject, "seed": seed, "U": U,
            "field": {"ell": ell, "lam": lam, "derangement": der if cond == "der" else None,
                      "row_mass_max": float(field_mass(g).max()), "Q_sha256": None if g["Q"] is None else C.sha(g["Q"])},
            "data": data["source"], "init_sha256": init_sha,
            "final_sha256": C.sha(np.asarray(g["lr"].brain.efficacy), np.asarray(g["lr"].brain.bias)),
            "train": tr, "evals": {str(k): v for k, v in evals.items()}, "summary": final["summary"],
            "patterns": final["patterns"], "e_seal": seal, "env": C.env_info(), "wall_s": time.time() - t0, "finished": C.now()}, g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", required=True, choices=["field", "der", "none"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"]); ap.add_argument("--U", type=int, default=1000)
    ap.add_argument("--ell", type=float, default=ELL); ap.add_argument("--lam", type=float, default=LAM); ap.add_argument("--der", type=int, default=0)
    a = ap.parse_args(argv)
    tag = "none" if a.cond == "none" else f"{a.cond}{a.der if a.cond == 'der' else ''}_ell{a.ell:g}_lam{a.lam:g}"
    path = OUT / a.split / f"A{a.subject:02d}_s{a.seed}_{tag}_U{a.U}.json"
    if path.exists(): print("exists", path); return
    rec, g = run(a.subject, a.seed, a.cond, a.split, a.U, a.ell, a.lam, a.der)
    W = OUT / "weights" / a.split; W.mkdir(parents=True, exist_ok=True)      # final lake, for later evaluations
    np.savez(W / (path.stem + ".npz"), eff=np.asarray(g["lr"].brain.efficacy, float), bias=np.asarray(g["lr"].brain.bias, float),
             pre=np.asarray(g["conn"].pre), post=np.asarray(g["conn"].post), fx=g["fx"])
    C.write_receipt(path, rec)
    print(path.name, {k: round(v, 4) for k, v in rec["summary"].items() if isinstance(v, float)}, f"wall {rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
