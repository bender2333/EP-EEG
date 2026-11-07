"""M13 §8: random hold-out patches in training, and test on leads / neighbourhoods never hidden in training.

Why (reports/M13.md §8): E0 trains and tests on the same 44 fixed hold-out patterns, and the hidden flags tell the
lake which one it is facing, so a per-pattern linear map from the visible leads solves the task without geometry
(ridge_raw 0.975 > spline 0.955). And a learnable W can absorb a fixed field on wrong positions. Here the lake can
only do well on the test leads if something carries information from the electrode geometry.

Lake: the untrained E0 initial lake of (subject, seed) (same init as E0), trained from scratch by E0's EP rule.
Conditions (m13_learn.build, trained=False): none, field (outlet field, true positions), der (same field, deranged
positions, C.derangement(k)); ell = 3 cm, lam = 0.5 (the §4 choice).

Folds (every other electrode column, fixed): A = FC3 FCz FC4 C3 Cz C4 CP3 CPz CP4 Pz, B = the other 12.
Seed 0 hides only A leads in training, seed 1 only B leads (H = the hide fold).
Training mask per row: a centre c uniform in H, size h uniform in {1, 2, 3}, the other h - 1 drawn without
replacement from the 4 nearest H leads of c; a draw equal to a tier-N test set is redrawn.
Test (eval runs, static equilibrium, lake's own outlet, fixed inverse): for every lead c, hide c with its 2 nearest
leads (any fold) and score R^2 on c only.
  tier U: c not in H (c was never hidden in training)
  tier N: c in H (c was hidden in training, but never together with this neighbourhood)
References on the same sets: spline (true / deranged positions), mean of visible leads, nearest visible lead,
ridge from visible leads fitted on the training rows (data-driven; uses c's training values).
All training and testing is within one subject.
"""
from __future__ import annotations

import argparse, sys, time, warnings

import numpy as np

warnings.filterwarnings("ignore")
from . import m12_s1 as S                        # noqa: E402
from . import m13_common as C                    # noqa: E402
from . import m13_e0 as E                        # noqa: E402
from . import m13_learn as Lr                    # noqa: E402
from cadence.learning import LearningPhaseError  # noqa: E402
import cadence as cd                             # noqa: E402

OUT = C.ROOT / "receipts" / "M13" / "R"
ELL, LAM = 0.03, 0.5
FOLD_A = [C.LEADS.index(x) for x in ("FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4", "Pz")]
FOLD_B = [c for c in range(C.N_LEADS) if c not in FOLD_A]
EVAL_AT = (1000, 2000)
DIST = np.linalg.norm(C.XYZ[:, None] - C.XYZ[None], axis=-1)


def hide_fold(seed): return FOLD_A if seed % 2 == 0 else FOLD_B


def test_sets():
    """centre c -> sorted hidden set (c and its 2 nearest leads)."""
    return {c: tuple(sorted([c, *[int(j) for j in np.argsort(DIST[c]) if j != c][:2]])) for c in range(C.N_LEADS)}


def tier_sets(seed):
    H = set(hide_fold(seed)); T = test_sets()
    return {"U": [c for c in T if c not in H], "N": [c for c in T if c in H]}


def sampler(seed):
    H = hide_fold(seed); T = test_sets()
    banned = {T[c] for c in H}                                      # tier-N test sets: never drawn
    near = {c: [h for h in sorted(H, key=lambda j: DIST[c, j]) if h != c][:4] for c in H}
    def draw(rng, n_rows):
        M = np.zeros((n_rows, C.N_LEADS), bool)
        for i in range(n_rows):
            while True:
                c = int(rng.choice(H)); h = int(rng.integers(1, 4))
                s = tuple(sorted([c, *[int(x) for x in rng.choice(near[c], h - 1, replace=False)]]))
                if s not in banned: break
            M[i, list(s)] = True
        assert not M[:, [c for c in range(C.N_LEADS) if c not in H]].any()
        return M
    return draw


def train(g, Xtr, seed, U, eta=E.ETA, log=print, on_eval=None):
    draw = sampler(seed); stats = {"accepted": 0, "refused": 0, "projections": 0, "log": []}
    nb = len(Xtr) // E.BATCH; t0 = time.time(); loss = []
    for u in range(U):
        ep, bi = divmod(u, nb)
        rng = np.random.default_rng((8000 + seed) * 100003 + ep)
        perm = rng.permutation(len(Xtr)); idx = perm[bi * E.BATCH:(bi + 1) * E.BATCH]
        Mk = draw(np.random.default_rng((9000 + seed) * 100003 + u), E.BATCH)
        D = E.drive_rows(g, Xtr[idx], Mk); T = E.target(g, Xtr[idx])
        try:
            free, _ = S.ep_step(g, D, T, eta)
            s = np.asarray(free.activation)[:, g["out"]]; loss.append(float(0.5 * ((s - T[:, g["out"]]) ** 2).mean()))
            stats["accepted"] += 1
        except LearningPhaseError:
            stats["refused"] += 1
        if Lr.project(g): stats["projections"] += 1
        if Lr.rho(g) >= 1.0: raise RuntimeError("rho~ >= 1 after projection")
        if (u + 1) % 200 == 0:
            stats["log"].append({"u": u + 1, "loss": float(np.mean(loss)) if loss else None, "rho": Lr.rho(g), "t": time.time() - t0}); loss = []
            log(f"u={u + 1} loss={stats['log'][-1]['loss']} rho={Lr.rho(g):.3f} refused={stats['refused']}")
        if on_eval is not None and (u + 1) in EVAL_AT: on_eval(u + 1)
    stats["certified_end"] = bool(cd.certificate(g["lr"].brain, sources=range(g["ns"])).certified)
    stats["wall_s"] = time.time() - t0
    return stats


def lake_own(g, X, hid, chunk=2400):
    M = np.zeros((len(X), C.N_LEADS), bool); M[:, list(hid)] = True; out, ok = [], []
    for i in range(0, len(X), chunk):
        eq = g["lr"].brain.equilibrate(E.drive_rows(g, X[i:i + chunk], M[i:i + chunk]), budget=S.BUDGET, tolerance=E.EVAL_TOL, damping=3)
        out.append(np.asarray(eq.state.activation)[:, g["out"]]); ok.append(np.asarray(eq.converged))
    return E.own_inverse(np.concatenate(out)), np.concatenate(ok)


def evaluate(g, Xte):
    T = test_sets(); res = {}
    for c, hid in T.items():
        p, ok = lake_own(g, Xte, hid)
        res[c] = {"lake_own": float(C.r2_cols(p[:, [c]], Xte[:, [c]])[0]), "qualified": float(ok.mean())}
    return res


def references(Xtr, Xte, Xrd):
    T = test_sets(); perms = [C.derangement(i) for i in range(3)]; res = {}
    for c, hid in T.items():
        vis = [j for j in range(C.N_LEADS) if j not in hid]; y = Xte[:, [c]]; k = list(hid).index(c)
        r = {"spline": float(C.r2_cols(Xte[:, vis] @ C.spline_matrix(hid)[[k]].T, y)[0]),
             "spline_shuf": float(np.mean([C.r2_cols(Xte[:, vis] @ C.spline_matrix(hid, xyz=C.XYZ[pi])[[k]].T, y)[0] for pi in perms])),
             "mean_visible": float(C.r2_cols(Xte[:, vis].mean(1, keepdims=True), y)[0]),
             "nearest": float(C.r2_cols(Xte[:, [min(vis, key=lambda j: DIST[c, j])]], y)[0])}
        p, _ = E.ridge_fit_predict(Xrd[:, vis], Xrd[:, [c]], Xte[:, vis]); r["ridge_raw"] = float(C.r2_cols(p, y)[0])
        res[c] = r
    return res


def summarise(per, seed, key="lake_own"):
    t = tier_sets(seed)
    return {f"{key}_{k}": float(np.mean([per[c][key] for c in cs])) for k, cs in t.items()}


def run(subject, seed, cond, split, U, ell=ELL, lam=LAM, der=None, refs=False):
    t0 = time.time()
    data = C.load_real(subject, split="t04_val45" if split == "test" else "dev_r02_r3")
    seal = C.e_seal()
    if seal["matches"]: raise RuntimeError(f"E files present: {seal['matches']}")
    Xtr = C.rows(data["train"]); Xte = C.rows(data["test"])
    der = seed if der is None else der
    g, init_sha = Lr.build(subject, seed, cond, ell, lam, der=der, trained=False)
    evals = {0: evaluate(g, Xte)}
    def on_eval(u):
        evals[u] = evaluate(g, Xte); print(f"[{subject}/{seed}/{cond}] eval u={u} {summarise(evals[u], seed)}", flush=True)
    tr = train(g, Xtr, seed, U, log=lambda m: print(f"[{subject}/{seed}/{cond}] {m}", flush=True), on_eval=on_eval)
    evals[U] = evaluate(g, Xte)
    rec = {"stage": "M13", "step": "8-random-patches", "cond": cond, "split": split, "subject": subject, "seed": seed, "U": U,
           "hide_fold": [C.LEADS[c] for c in hide_fold(seed)], "tiers": {k: [C.LEADS[c] for c in v] for k, v in tier_sets(seed).items()},
           "field": {"ell": ell, "lam": lam, "derangement": der if cond == "der" else None,
                     "row_mass_max": float(Lr.field_mass(g).max()), "Q_sha256": None if g["Q"] is None else C.sha(g["Q"])},
           "data": data["source"], "init_sha256": init_sha,
           "final_sha256": C.sha(np.asarray(g["lr"].brain.efficacy), np.asarray(g["lr"].brain.bias)), "train": tr,
           "evals": {str(u): {"per_lead": {C.LEADS[c]: v for c, v in e.items()}, "summary": summarise(e, seed)} for u, e in evals.items()},
           "summary": summarise(evals[U], seed), "e_seal": seal}
    if refs:
        R = references(Xtr, Xte, C.rows(data["train"], step=C.RIDGE_STEP))
        rec["refs"] = {C.LEADS[c]: v for c, v in R.items()}
        rec["refs_summary"] = {k: v for key in ("spline", "spline_shuf", "mean_visible", "nearest", "ridge_raw") for k, v in summarise(R, seed, key).items()}
    rec.update(env=C.env_info(), wall_s=time.time() - t0, finished=C.now())
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", required=True, choices=["field", "der", "none"]); ap.add_argument("--der", type=int, default=None)
    ap.add_argument("--split", default="test", choices=["dev", "test"]); ap.add_argument("--U", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true", help="timing only: tiny U, receipt under smoke/")
    a = ap.parse_args(argv)
    der = a.seed if a.der is None else a.der
    tag = "none" if a.cond == "none" else f"{a.cond}{der if a.cond == 'der' else ''}_ell{ELL:g}_lam{LAM:g}"
    path = OUT / ("smoke" if a.smoke else a.split) / f"A{a.subject:02d}_s{a.seed}_{tag}_U{a.U}.json"
    if path.exists(): print("exists", path); return
    rec = run(a.subject, a.seed, a.cond, a.split, a.U, der=der, refs=(a.cond == "none"))
    C.write_receipt(path, rec)
    print(path.name, rec["summary"], rec.get("refs_summary"), f"wall {rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
