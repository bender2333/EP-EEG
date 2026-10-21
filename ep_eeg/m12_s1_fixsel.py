"""M12 S1 fix round: pick the setting per task from the dev-seed-101 selection receipts (S1.6 rule + amendment 8 ties),
write receipts/M12/S1/fix_selected.json and the official after-fix job file. Missing or crashed settings score -inf."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from . import m12_s1 as S
from .m12_s1_fix import GRID

def score(r, task):
    if r is None or r.get("accept_rate") is None or r["accept_rate"] < 0.95 or r["eval_unqualified"] > 0.01 * r["eval_rows"]: return -np.inf
    v = r["kappa_ridge"] if task == "T2" else min(r["r2_ridge"], r["r2_out"])
    return v if np.isfinite(v) else -np.inf

def main(commit):
    base = S.S1 / commit / "fixsel"; out = {}
    for task, H in (("T2", None), ("T1", 8), ("T1", 32)):
        rows = []
        for gi, (beta, eta, nz, ab) in enumerate(GRID):
            p = base / f"g{gi}" / (f"{task}_ep" + (f"_H{H}" if H else "") + "_s101.json")
            r = json.loads(p.read_text()) if p.exists() else None
            rows.append({"g": gi, "beta": beta, "eta": eta, "normalize": nz, "ab": ab, "score": score(r, task), "present": r is not None})
        best = max(r["score"] for r in rows)
        cand = [r for r in rows if r["score"] >= best - 0.005] if np.isfinite(best) else rows
        cand.sort(key=lambda r: (r["eta"], r["beta"], r["normalize"] != 0.0, r["ab"] != "N2"))
        key = task + (f"_H{H}" if H else ""); out[key] = {"chosen": cand[0], "best_score": best, "all": rows}
    (S.S1 / "fix_selected.json").write_text(json.dumps(out, indent=1, default=float))
    jobs = []
    for key, v in out.items():
        c = v["chosen"]; st = json.dumps({"beta": c["beta"], "eta": c["eta"], "normalize": c["normalize"], "ab": c["ab"], "grid_index": c["g"]}).replace(" ", "")
        if key == "T2":
            jobs += [f"official T2 {a} {s} 0 {st}" for a in ("ep", "shuffled", "random") for s in (0, 1, 2)]
        else:
            H = int(key.split("_H")[1]); jobs += [f"official T1 {a} {s} {H} {st}" for a in ("ep", "random", "randodd") for s in (0, 1, 2)]
    Path("/workspace/m12_logs/S1/fixrun_jobs.txt").write_text("\n".join(jobs) + "\n")
    print(json.dumps({k: {"chosen": v["chosen"], "best": v["best_score"]} for k, v in out.items()}, indent=1, default=float))

if __name__ == "__main__":
    main(sys.argv[1])
