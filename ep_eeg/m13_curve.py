"""M13 §10.4 (data sufficiency): the §8 task (cap 0.9, U 3 000, random patches, tier U / N) trained on 25 % or 50 % of
one subject's training trials (a seeded random subset of runs 0-3; the test trials are unchanged). 100 % = the §8
receipts. Also the visible-lead ridge and the visible mean on the same subset, as references."""
from __future__ import annotations
import argparse, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from . import m13_common as C                    # noqa: E402
from . import m13_e0 as E                        # noqa: E402
from . import m13_learn as Lr                    # noqa: E402
from . import m13_rand as R                      # noqa: E402

OUT = C.ROOT / "receipts" / "M13" / "G" / "curve"; CAP = 0.9


def run(subject, seed, frac, U):
    t0 = time.time(); d = C.load_real(subject, split="t04_val45"); seal = C.e_seal()
    if seal["matches"]: raise RuntimeError(f"E files present: {seal['matches']}")
    n = len(d["train"]); keep = np.sort(np.random.default_rng(500 + 10 * subject + seed).permutation(n)[:int(round(frac * n))])
    tr = d["train"][keep]; Xtr = C.rows(tr); Xte = C.rows(d["test"]); Xrd = C.rows(tr, step=C.RIDGE_STEP)
    g, init_sha = Lr.build(subject, seed, "none", trained=False, cap=CAP)
    trn = R.train(g, Xtr, seed, U, log=lambda m: print(f"[{subject}/{seed}/{frac}] {m}", flush=True), cap=CAP)
    ev = R.evaluate(g, Xte); refs = {}
    for c, hid in R.test_sets().items():
        vis = [j for j in range(C.N_LEADS) if j not in hid]
        p, _ = E.ridge_fit_predict(Xrd[:, vis], Xrd[:, [c]], Xte[:, vis])
        refs[c] = {"ridge_vis": float(C.r2_cols(p, Xte[:, [c]])[0]),
                   "mean_visible": float(C.r2_cols(Xte[:, vis].mean(1, keepdims=True), Xte[:, [c]])[0])}
    summ = R.summarise(ev, seed)
    for k in ("ridge_vis", "mean_visible"): summ.update(R.summarise(refs, seed, k))
    return {"stage": "M13", "step": "10.4-data-curve", "subject": subject, "seed": seed, "frac": frac, "n_train_trials": int(len(keep)),
            "train_trials": keep.tolist(), "U": U, "cap": CAP, "init_sha256": init_sha,
            "final_sha256": C.sha(np.asarray(g["lr"].brain.efficacy), np.asarray(g["lr"].brain.bias)), "rho_end": Lr.rho(g),
            "train": trn, "per_lead": {C.LEADS[c]: dict(ev[c], **refs[c]) for c in ev}, "summary": summ, "e_seal": seal,
            "data": d["source"], "env": C.env_info(), "wall_s": time.time() - t0, "finished": C.now()}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--subject", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, required=True); ap.add_argument("--U", type=int, default=3000)
    a = ap.parse_args(argv); path = OUT / f"A{a.subject:02d}_s{a.seed}_frac{a.frac:g}_U{a.U}.json"
    if path.exists(): print("exists", path); return
    rec = run(a.subject, a.seed, a.frac, a.U); C.write_receipt(path, rec)
    print(path.name, {k: round(v, 3) for k, v in rec["summary"].items()}, f"wall {rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
