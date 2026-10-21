"""M11 amendment 3: does cadence 0.84.3 (muellerberndt/cadence 8dc92d8) reproduce the 0.83.0 (6499e56) receipts?

Deterministic re-runs compared value by value with the receipts already written under 0.83.0:
  D1  every small-graph cell (all beta x tol), the asymmetry sweep and the real sandbox
  B   random_lake and m6_trained (seed 1, retrained 100 epochs: weights compared) for A01-A02
  D2  recip_ep / recip_ift / recip_delta, A01 seed 3 (kappa, predictions, parameter change)
  C   the 0.83.0 smoke test of ep_proj (A01, seed 0, 3 EP updates at Adam 0.003): R2_15 and D_gen
Writes receipts/M11/cadence_repro_0843.json.
"""
from __future__ import annotations
import json, os, time, warnings
from pathlib import Path
import numpy as np
from .data import ROOT
from .m11_common import env_info

OLD = "068c27c"; R = ROOT / "receipts" / "M11"


def maxdiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")


def flat(d, prefix=""):
    """Numeric leaves of a nested dict/list, keyed by path."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items(): out.update(flat(v, f"{prefix}/{k}"))
    elif isinstance(d, list) and d and isinstance(d[0], (dict, list)):
        for i, v in enumerate(d): out.update(flat(v, f"{prefix}[{i}]"))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix] = float(d)
    elif isinstance(d, list):
        out[prefix] = d
    return out


def compare_dicts(new, old, skip=("wall_s", "env", "written", "wall", "train_wall_s", "eval_wall_s")):
    fn, fo = flat({k: v for k, v in new.items() if k not in skip}), flat({k: v for k, v in old.items() if k not in skip})
    keys = [k for k in fn if k in fo and not any(s in k for s in skip)]
    worst = max((maxdiff(fn[k], fo[k]) for k in keys), default=0.0)
    return {"n_compared": len(keys), "max_abs_diff": worst, "missing": len(set(fo) - set(fn))}


def main():
    t0 = time.time(); warnings.simplefilter("ignore"); out = {"env": env_info()}
    from . import m11_d1 as D1
    cells = {}
    for f in sorted((R / "D1" / OLD / "small").glob("*.json")):
        o = json.loads(f.read_text()); n = D1.cell(o["seed"], o["leak"], o["budget"], o["amp"], o["nudge"], write=False)
        cells[f.stem] = compare_dicts(n, o)
    out["D1_small"] = {"cells": len(cells), "max_abs_diff": max(c["max_abs_diff"] for c in cells.values())}
    old_asym = json.loads((R / "D1" / OLD / "asym.json").read_text())["cells"]
    new_asym = [D1.asym_cell(c["seed"], c["eps"], c["amp"], c["nudge"]) for c in old_asym]
    out["D1_asym"] = {"cells": len(new_asym), "max_abs_diff": max(compare_dicts(n, o)["max_abs_diff"] for n, o in zip(new_asym, old_asym))}
    old_real = json.loads((R / "D1" / OLD / "real_sandbox.json").read_text())["cells"]
    new_real = [D1.real_sandbox(c["beta"], c["tol"]) for c in old_real]
    out["D1_real"] = {"cells": len(new_real), "max_abs_diff": max(compare_dicts(n, o)["max_abs_diff"] for n, o in zip(new_real, old_real))}
    print("D1", out["D1_small"], out["D1_asym"], out["D1_real"], flush=True)

    from . import m11_b as B
    from .data import load_subject
    from .sandbox import Sandbox, SandboxConfig, train_sandbox
    b = {}
    for s in (1, 2):
        d = load_subject(s); sb = Sandbox(SandboxConfig(**B.M6), 0)
        S, _, ok = B.static_states(sb, d["val"]["track2_x"])
        old = json.loads((R / "B" / OLD / "random_lake" / f"A{s:02d}_s0.json").read_text())
        F = None
        b[f"random_lake_A{s:02d}"] = {"settle": float(ok.mean())}
        # rerun the full readout to compare kappa_cv / kappa_or and the honest predictions
        Str, _, oktr = B.static_states(sb, d["train"]["track2_x"])
        mtr, ctr = B.fmaps_static(Str, oktr); mva, cva = B.fmaps_static(S, ok, ref=mtr.mean(0))
        res = B.ceilings({"mean": (mtr, mva), "cat": (ctr, cva)}, d["train"]["y"], d["val"]["y"], 0)
        b[f"random_lake_A{s:02d}"].update(kappa_cv_diff=abs(res["kappa_cv"] - old["kappa_cv"]), kappa_or_diff=abs(res["kappa_or"] - old["kappa_or"]),
                                          pred_cv_same=bool(res["pred_cv"] == old["pred_cv"]))
        sb1, st, _ = train_sandbox(d["train"], 1, **B.M6)
        w_old = np.load(R / "B" / "weights" / "m6_trained" / f"A{s:02d}_s1.npz")
        b[f"m6_trained_A{s:02d}_s1"] = {"efficacy_max_abs_diff": maxdiff(np.asarray(sb1.learner.brain.efficacy), w_old["efficacy"]),
                                        "bias_max_abs_diff": maxdiff(np.asarray(sb1.learner.brain.bias), w_old["bias"]),
                                        "accept_rate": st["accepted_rows"] / max(1, st["attempted_rows"])}
        print("B", s, b, flush=True)
    out["B"] = b

    from .g1_readout import run
    d2 = {}
    for tr in ("ep", "ift", "delta"):
        cfg = dict(feat="lake", struct="recip", nudge="ce01", eta=0.05, rows="sub", trainer=tr, probe_every=20 if tr == "ep" else 0)
        n = run(1, cfg, 3, "final"); o = json.loads((R / "D2" / OLD / f"recip_{tr}" / "A01_s3.json").read_text())
        d2[tr] = {"kappa_diff": abs(n["kappa_ep"] - o["kappa_ep"]), "pred_same": n["pred"] == o["pred"],
                  "dtheta_max_abs_diff": maxdiff(n["dtheta"], o["dtheta"])}
        print("D2", tr, d2[tr], flush=True)
    out["D2"] = d2

    os.environ["M11_SMOKE"] = "1"
    from importlib import reload
    from . import m11_common, m11_c
    reload(m11_common); C = reload(m11_c)
    smoke = ROOT / "receipts" / "M11_smoke" / "C" / "runs" / "ep_proj" / "A01_s0.json"; old_text = smoke.read_text(); old = json.loads(old_text)
    (C.OUT).mkdir(parents=True, exist_ok=True)
    keep = C.OUT / "settings.json"; prev = keep.read_text() if keep.exists() else None
    keep.write_text(json.dumps({"gain": 3.0, "port": 5.0, "var_p": 6, "U": 3, "lr": 0.003}))
    n = C.run_arm("ep_proj", 1, 0)
    if prev is not None: keep.write_text(prev)
    nr = json.loads(smoke.read_text()); smoke.with_name("A01_s0_cadence0843.json").write_text(smoke.read_text()); smoke.write_text(old_text)
    out["C_ep_smoke"] = {"R2_15_diff": abs(nr["R2_15"] - old["R2_15"]), "D_gen_diff": abs(nr["D_gen"] - old["D_gen"]),
                         "old": [old["R2_15"], old["D_gen"]], "new": [nr["R2_15"], nr["D_gen"]]}
    print("C", out["C_ep_smoke"], flush=True)
    out["wall_s"] = time.time() - t0
    (R / "cadence_repro_0843.json").write_text(json.dumps(out, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
