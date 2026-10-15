"""Research-loop runner: T session only. No holdout option by design."""
from __future__ import annotations
import argparse, json, subprocess, time, platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from eval.scoring import aggregate, fallback_class, score
from .data import load_subject, ROOT


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()


def one(arm, subject, seed, opts):
    from .arms import get
    d = load_subject(subject)
    t0 = time.time()
    scores, ok, extra = get(arm)(d["train"], d["val"], seed, **opts)
    wall = time.time() - t0
    pred, rej = aggregate(scores, ok, fallback_class(d["train"]["y"]))
    m = score(d["val"]["y"], pred, ok)
    m.update(subject=subject, seed=seed, wall_s=wall, extra=extra,
             pred=pred.tolist(), y=d["val"]["y"].tolist(), rejected=rej.tolist())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--subjects", default="1-9"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--opts", default="{}"); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    lo, _, hi = a.subjects.partition("-")
    subs = list(range(int(lo), int(hi or lo) + 1)); seeds = [int(s) for s in a.seeds.split(",")]
    opts = json.loads(a.opts); commit = git_commit(); t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(one, a.arm, s, k, opts) for s in subs for k in seeds]
        res = [f.result() for f in futs]
    kap = np.array([[r["kappa"] for r in res if r["subject"] == s] for s in subs])
    summ = {"commit": commit, "stage": a.stage, "arm": a.arm, "opts": opts, "subjects": subs, "seeds": seeds,
            "kappa_val": float(kap.mean()), "kappa_seed_sd": float(kap.mean(0).std(ddof=1)) if len(seeds) > 1 else 0.0,
            "acc_val": float(np.mean([r["acc"] for r in res])),
            "settle_rate": float(np.mean([r["settle_rate"] for r in res])),
            "per_subject_kappa": kap.mean(1).tolist(), "wall_s": time.time() - t0,
            "python": platform.python_version()}
    out = ROOT / "receipts" / a.stage / a.arm / f"{commit}{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_text(json.dumps(res))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items()}, indent=1))


if __name__ == "__main__":
    main()
