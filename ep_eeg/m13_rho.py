"""M13 §9: the §8 random-patch retest with the gain cap rho~ raised (reports/M13.md §9).

Why: in §4-§8 the certified row mass rho~ sat at the cap 0.9 on every update (projection ~2 950 / 3 000 updates in §8),
so the outlet field and the plastic synapses compete for one fixed gain budget. One round with a higher cap, same
task as §8 (m13_rand: folds, random patches, tier U / N test sets, conditions none / field / der, ell 3 cm, lam 0.5,
U 3 000, from the untrained E0 initial lake, single subject).

New in the receipt (final lake only):
  gain  : lake_own times one scalar per test set, fitted by least squares on training rows (every 10th sample of
          runs 0-3, same hidden set), scored on the test rows: the readout-gain baseline
  power : log10(reconstructed / true) power of the scored lead in mu 8-13 and beta 13-30 Hz on the test trials,
          reconstructed every 2nd sample (125 Hz), zero-phase band-pass; 0 = exact (fields none and field only)
  rho   : rho~ trace and projection count (already in train)
Modes:
  --speed CAP   timing / convergence check for the cap choice: 200 updates of `none`, then one evaluation; records only
                seconds per update, seconds per evaluation and the converged fraction (no score is computed)
  (default)     one job; --cap selects the cap (0.9 = the §8 cap, rerun for none / field to get gain and power)
"""
from __future__ import annotations

import argparse, sys, time, warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt

warnings.filterwarnings("ignore")
from . import m12_s1 as S                        # noqa: E402
from . import m13_common as C                    # noqa: E402
from . import m13_e0 as E                        # noqa: E402
from . import m13_learn as Lr                    # noqa: E402
from . import m13_rand as R                      # noqa: E402

OUT = C.ROOT / "receipts" / "M13" / "P"
FS2 = 125.0
SOS2 = {b: butter(4, f, btype="band", fs=FS2, output="sos") for b, f in {"mu": (8, 13), "beta": (13, 30)}.items()}


def full_eval(g, Xte, Xg, seg_te, power):
    """per test set: raw lake_own R^2, gain-calibrated R^2 (scalar fitted on Xg rows), converged fraction;
    and, if power, the mu / beta log power ratio of the scored lead."""
    T = R.test_sets(); res = {}
    if power:
        n = seg_te.shape[0]; X2 = C.rows(seg_te, step=2); t2 = X2.shape[0] // n
    for c, hid in T.items():
        p, ok = R.lake_own(g, Xte, hid); pg, okg = R.lake_own(g, Xg, hid)
        a = float(pg[:, c] @ Xg[:, c] / max(pg[:, c] @ pg[:, c], 1e-12))
        r = {"lake_own": float(C.r2_cols(p[:, [c]], Xte[:, [c]])[0]),
             "gain": float(C.r2_cols(a * p[:, [c]], Xte[:, [c]])[0]), "scalar": a,
             "qualified": float(ok.mean()), "qualified_fit": float(okg.mean())}
        if power:
            q, okp = R.lake_own(g, X2, hid)
            pr = q[:, c].reshape(n, t2); y = X2[:, c].reshape(n, t2)
            for b, sos in SOS2.items():
                pb, yb = sosfiltfilt(sos, pr, axis=-1), sosfiltfilt(sos, y, axis=-1)
                r[f"logpow_{b}"] = float(np.log10(pb.var(-1).mean() / yb.var(-1).mean()))
            r["qualified_power"] = float(okp.mean())
        res[c] = r
    return res


def power_refs(seg_te):
    """the same log power ratios for the visible mean and the nearest visible lead (no lake)."""
    T = R.test_sets(); n = seg_te.shape[0]; X2 = C.rows(seg_te, step=2); t2 = X2.shape[0] // n; res = {}
    for c, hid in T.items():
        vis = [j for j in range(C.N_LEADS) if j not in hid]; y = X2[:, c].reshape(n, t2)
        preds = {"mean_visible": X2[:, vis].mean(1).reshape(n, t2), "nearest": X2[:, min(vis, key=lambda j: R.DIST[c, j])].reshape(n, t2)}
        r = {}
        for k, pr in preds.items():
            for b, sos in SOS2.items():
                r[f"{k}_logpow_{b}"] = float(np.log10(sosfiltfilt(sos, pr, axis=-1).var(-1).mean() / sosfiltfilt(sos, y, axis=-1).var(-1).mean()))
        res[c] = r
    return res


def run(subject, seed, cond, U, cap, der=None):
    t0 = time.time()
    data = C.load_real(subject, split="t04_val45")
    seal = C.e_seal()
    if seal["matches"]: raise RuntimeError(f"E files present: {seal['matches']}")
    Xtr = C.rows(data["train"]); Xte = C.rows(data["test"]); Xg = C.rows(data["train"], step=C.RIDGE_STEP)
    der = seed if der is None else der
    g, init_sha = Lr.build(subject, seed, cond, R.ELL, R.LAM, der=der, trained=False, cap=cap)
    evals = {0: R.evaluate(g, Xte)}
    def on_eval(u):
        evals[u] = R.evaluate(g, Xte); print(f"[{subject}/{seed}/{cond}/cap{cap}] eval u={u}", flush=True)
    tr = R.train(g, Xtr, seed, U, log=lambda m: print(f"[{subject}/{seed}/{cond}/cap{cap}] {m}", flush=True), on_eval=on_eval, cap=cap)
    fin = full_eval(g, Xte, Xg, data["test"], power=cond in ("none", "field"))
    evals[U] = {c: {"lake_own": v["lake_own"], "qualified": v["qualified"]} for c, v in fin.items()}
    summ = R.summarise(fin, seed)
    summ.update(R.summarise(fin, seed, "gain"))
    if cond in ("none", "field"):
        for b in ("mu", "beta"): summ.update(R.summarise(fin, seed, f"logpow_{b}"))
    rec = {"stage": "M13", "step": "9-rho-cap", "cap": cap, "cond": cond, "split": "test", "subject": subject, "seed": seed, "U": U,
           "hide_fold": [C.LEADS[c] for c in R.hide_fold(seed)], "tiers": {k: [C.LEADS[c] for c in v] for k, v in R.tier_sets(seed).items()},
           "field": {"ell": R.ELL, "lam": R.LAM, "derangement": der if cond == "der" else None,
                     "row_mass_max": float(Lr.field_mass(g).max()), "Q_sha256": None if g["Q"] is None else C.sha(g["Q"])},
           "data": data["source"], "init_sha256": init_sha,
           "final_sha256": C.sha(np.asarray(g["lr"].brain.efficacy), np.asarray(g["lr"].brain.bias)),
           "rho_end": Lr.rho(g), "train": tr,
           "evals": {str(u): {"per_lead": {C.LEADS[c]: v for c, v in e.items()}, "summary": R.summarise(e, seed)} for u, e in evals.items()},
           "final": {C.LEADS[c]: v for c, v in fin.items()}, "summary": summ, "e_seal": seal}
    if cond == "none":
        P = power_refs(data["test"]); rec["power_refs"] = {C.LEADS[c]: v for c, v in P.items()}
        rec["power_refs_summary"] = {k: v for key in P[0] for k, v in R.summarise(P, seed, key).items()}
    rec.update(env=C.env_info(), wall_s=time.time() - t0, finished=C.now())
    return rec


def speed(subject, seed, cap, n=200):
    data = C.load_real(subject, split="t04_val45")
    if C.e_seal()["matches"]: raise RuntimeError("E files present")
    Xtr = C.rows(data["train"]); Xte = C.rows(data["test"])
    g, _ = Lr.build(subject, seed, "none", trained=False, cap=cap)
    t0 = time.time(); tr = R.train(g, Xtr, seed, n, log=lambda m: None, cap=cap); tu = (time.time() - t0) / n
    T = R.test_sets(); t1 = time.time(); ok = [R.lake_own(g, Xte, hid)[1].mean() for hid in T.values()]; te = time.time() - t1
    rec = {"cap": cap, "subject": subject, "seed": seed, "updates": n, "s_per_update": tu, "s_per_eval": te,
           "converged_eval": float(np.mean(ok)), "converged_min": float(np.min(ok)), "refused": tr["refused"],
           "rho_end": Lr.rho(g), "projections": tr["projections"], "env": C.env_info(), "finished": C.now()}
    C.write_receipt(OUT / "speed" / f"A{subject:02d}_s{seed}_cap{cap:g}.json", rec)
    print(rec, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", default="none", choices=["field", "der", "none"]); ap.add_argument("--der", type=int, default=None)
    ap.add_argument("--U", type=int, default=3000); ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--speed", action="store_true")
    a = ap.parse_args(argv)
    if a.speed: return speed(a.subject, a.seed, a.cap)
    der = a.seed if a.der is None else a.der
    tag = "none" if a.cond == "none" else f"{a.cond}{der if a.cond == 'der' else ''}_ell{R.ELL:g}_lam{R.LAM:g}"
    path = OUT / "test" / f"A{a.subject:02d}_s{a.seed}_cap{a.cap:g}_{tag}_U{a.U}.json"
    if path.exists(): print("exists", path); return
    rec = run(a.subject, a.seed, a.cond, a.U, a.cap, der=der)
    C.write_receipt(path, rec)
    print(path.name, rec["summary"], f"rho_end {rec['rho_end']:.4f} wall {rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
