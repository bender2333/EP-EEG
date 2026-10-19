"""M12 S0 pre-check: regression (amendment 1, N12).

  --mode ref   run with PYTHONPATH=/workspace/cadence-c414c85/src (unmodified cadence c414c85, a git worktree):
               the M11 D1 leak=1.0 cells (3 seeds x 3 budgets x 2 amplitudes x 2 nudges = 36), recording the tight
               free state (1e-11), the reference IFT vector, the production EP contrast and M11's full beta x tol grid.
  --mode fork  with the fork (branch m12): (i) the sigmoid path must reproduce ref element-wise <= 1e-12;
               (ii) evenodd (a, b) = (1, 0), sources through the activation, must reproduce the fork's sigmoid path
               element-wise <= 1e-12 (tight state, EP contrast) and |cos(EP, IFT) difference| <= 1e-9.
The comparison with the M11 receipts (6499e56 / 8dc92d8) is descriptive only.
"""
from __future__ import annotations
import argparse, itertools, json, time, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from . import m12_common as C
from . import m11_d1 as D
from .ift import implicit_grad, ep_equivalent

CELLS = list(itertools.product((0, 1, 2), (0.3, 0.6, 0.9), (0.5, 4.0), ("ce", "quad")))


def evaluate(seed, budget, amp, nudge):
    lr, kind, conn = D.build(seed, 1.0, budget, nudge); X, y = D.batch(seed, amp, conn.n)
    tight = lr.brain.equilibrate(X, budget=50000, tolerance=1e-11, damping=3)
    g, gb, _, _ = implicit_grad(lr, tight.state, y); ce, cb = ep_equivalent(lr, g, gb)
    e, nb = D.ep_contrast(lr, X, y)
    idx, _ = D.physical(lr, kind)
    return {"v_tight": np.asarray(tight.state.v), "ift": np.r_[ce, cb], "ep": np.r_[np.asarray(e), np.asarray(nb)],
            "cos_ep_ift": D.cosine(np.asarray(e)[idx], ce[idx])}


def ref_cell(args):
    seed, budget, amp, nudge = args
    out = C.OUT / "S0" / "regress" / "ref"; nm = f"s{seed}_b{budget}_amp{amp}_{nudge}"
    prev = C.existing(out, nm)
    if prev: return prev
    t0 = time.time(); r = evaluate(seed, budget, amp, nudge)
    r["m11_cell"] = D.cell(seed, 1.0, budget, amp, nudge, write=False)
    r.update(seed=seed, budget=budget, amp=amp, nudge=nudge, wall_s=time.time() - t0)
    C.write_receipt(out, nm, r); return r


def fork_cell(args):
    seed, budget, amp, nudge = args
    out = C.OUT / "S0" / "regress" / "fork" / C.run_commit(); nm = f"s{seed}_b{budget}_amp{amp}_{nudge}"
    prev = C.existing(out, nm)
    if prev: return prev
    ref = C.existing(C.OUT / "S0" / "regress" / "ref", nm)
    if ref is None: raise RuntimeError("run --mode ref first")
    from cadence.learning import evenodd_neuron_model
    t0 = time.time(); sig = evaluate(seed, budget, amp, nudge)
    orig = D.learning_neuron_model
    D.learning_neuron_model = lambda leak=1.0: evenodd_neuron_model(1.0, 0.0, dt=0.5, gain=1.0, stimulus_amplitude=1.0)
    try:
        eo = evaluate(seed, budget, amp, nudge)
    finally:
        D.learning_neuron_model = orig
    d = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max())
    r = {"seed": seed, "budget": budget, "amp": amp, "nudge": nudge,
         "sigmoid_vs_ref": {k: d(sig[k], ref[k]) for k in ("v_tight", "ift", "ep")} | {"cos_ep_ift": abs(sig["cos_ep_ift"] - ref["cos_ep_ift"])},
         "evenodd_vs_sigmoid": {k: d(eo[k], sig[k]) for k in ("v_tight", "ift", "ep")} | {"cos_ep_ift": abs(eo["cos_ep_ift"] - sig["cos_ep_ift"])},
         "cos_ep_ift": {"ref": ref["cos_ep_ift"], "sigmoid": sig["cos_ep_ift"], "evenodd": eo["cos_ep_ift"]},
         "wall_s": time.time() - t0}
    r["pass"] = bool(max(r["sigmoid_vs_ref"][k] for k in ("v_tight", "ift", "ep")) <= 1e-12
                     and max(r["evenodd_vs_sigmoid"][k] for k in ("v_tight", "ift", "ep")) <= 1e-12
                     and r["evenodd_vs_sigmoid"]["cos_ep_ift"] <= 1e-9)
    C.write_receipt(out, nm, r); return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=("ref", "fork")); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(); C.e_seal(); C.start_clock(f"m12_s0_regress --mode {a.mode}")
    import cadence
    print("cadence from", cadence.__file__, C.env_info()["cadence_commit"], flush=True)
    if a.mode == "ref" and "cadence-c414c85" not in cadence.__file__: raise SystemExit("ref mode must import the c414c85 worktree")
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(ref_cell if a.mode == "ref" else fork_cell, CELLS))
    if a.mode == "fork":
        bad = [r for r in res if not r["pass"]]
        print(f"regression: {len(res) - len(bad)}/{len(res)} pass; worst sigmoid_vs_ref "
              f"{max(max(r['sigmoid_vs_ref'][k] for k in ('v_tight', 'ift', 'ep')) for r in res):.2e}, worst evenodd_vs_sigmoid "
              f"{max(max(r['evenodd_vs_sigmoid'][k] for k in ('v_tight', 'ift', 'ep')) for r in res):.2e}", flush=True)
        C.ledger("M12-S0", "precheck_regression", "NA", 3, time.time() - t0, "pass" if not bad else "fail",
                 f"N12 regression: {len(res) - len(bad)}/{len(res)} cells pass (fork sigmoid == c414c85, evenodd(1,0) == sigmoid, <= 1e-12)")
    else:
        C.ledger("M12-S0", "precheck_regression_ref", "NA", 3, time.time() - t0, "done", "N12 reference: M11 D1 leak=1.0 36 cells rerun on unmodified cadence c414c85 (worktree)")


if __name__ == "__main__":
    main()
