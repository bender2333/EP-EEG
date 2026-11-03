"""M13 E0 gate verifier: recompute the pre-registered E0 verdict from the receipts alone.

python -m ep_eeg.m13_verify [--dir receipts/M13/E0/official] [--json out.json]
Fails (verdict "inconsistent") on a missing subject or seed, a dev receipt in the input, a config
that is not the pre-registered one, a non-finite score, an E-file match, or a summary that does not
match the per-pattern scores. The gate itself (reports/M13.md E0.6):
  E0-a  lake > random_lake on P3 in >= 7/9 subjects and one-sided Wilcoxon p < 0.05
  E0-b  lake > spline      on P3 in >= 7/9 subjects and one-sided Wilcoxon p < 0.05
  E0-c  spline > spline_shuf on P3 in >= 8/9 subjects (validity: the task carries spatial information)
  Q     every lake and random-lake receipt qualifies >= 98% of its evaluation rows and ends certified
Subject score = mean over seeds {0,1,2} (lakes); spline arms are deterministic.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

from . import m13_common as C

SUBJECTS, SEEDS = range(1, 10), (0, 1, 2)
PREREG = {"k": 4, "U": 6000, "batch": 64, "eta": 0.05, "beta": 0.04, "cap": 0.9, "rho0": 0.6, "t_scale": 0.5}


def wilcoxon_greater(d):
    from scipy.stats import wilcoxon
    d = np.asarray(d, float)
    if np.all(d == 0): return 1.0
    return float(wilcoxon(d, alternative="greater").pvalue)


def load(dirpath):
    recs = {}
    for f in sorted(Path(dirpath).glob("A*_s*.json")):
        r = json.loads(f.read_text()); recs[(r["subject"], r["seed"])] = r
    return recs


def check(recs):
    issues = []
    for s in SUBJECTS:
        for sd in SEEDS:
            if (s, sd) not in recs: issues.append(f"missing A{s:02d} seed {sd}")
    pats = [(f, list(t)) for f, t in C.all_patterns()]
    for (s, sd), r in recs.items():
        tag = f"A{s:02d}/s{sd}"
        if r.get("dev"): issues.append(f"{tag}: dev receipt in official input")
        for k, v in PREREG.items():
            if r["config"].get(k) != v: issues.append(f"{tag}: config {k}={r['config'].get(k)} != {v}")
        if r["e_seal"]["matches"]: issues.append(f"{tag}: E files present")
        got = [(p["family"], p["hidden"]) for p in r["patterns"]]
        if got != pats: issues.append(f"{tag}: pattern list differs from the pre-registered one")
        for p in r["patterns"]:
            for arm in ("lake", "random_lake", "spline", "spline_shuf"):
                if not np.all(np.isfinite(p[arm])): issues.append(f"{tag}: non-finite {arm}")
            for q in ("qualified_lake", "qualified_random"):
                if p[q] < 0.98: issues.append(f"{tag}: {q} {p[q]:.4f} < 0.98 on {p['hidden']}")
        if not r["train"].get("certified_end", False): issues.append(f"{tag}: not certified at the end")
        if r["train"]["accepted"] + r["train"]["refused"] != r["config"]["U"]: issues.append(f"{tag}: update count")
        for arm in ("lake", "random_lake", "spline", "spline_shuf"):
            for fam in ("P1", "P3"):
                v = float(np.mean([np.mean(p[arm]) for p in r["patterns"] if p["family"] == fam]))
                if abs(v - r["summary"][f"{arm}_{fam}"]) > 1e-9: issues.append(f"{tag}: summary {arm}_{fam} mismatch")
    return issues


def scores(recs, fam="P3"):
    out = {}
    for arm in ("lake", "random_lake", "spline", "spline_shuf", "ridge_raw", "mean_visible", "lake_own"):
        per = []
        for s in SUBJECTS:
            v = [np.mean([np.mean(p[arm]) for p in recs[(s, sd)]["patterns"] if p["family"] == fam]) for sd in SEEDS if (s, sd) in recs]
            per.append(float(np.mean(v)) if v else float("nan"))
        out[arm] = per
    return out


def verdict(recs):
    issues = check(recs)
    if issues: return {"verdict": "inconsistent", "issues": issues}
    sc = scores(recs, "P3")
    def cmp(a, b, need):
        d = np.array(sc[a]) - np.array(sc[b]); n = int((d > 0).sum()); p = wilcoxon_greater(d)
        return {"wins": n, "need": need, "p": p, "mean_diff": float(d.mean()), "pass": bool(n >= need and (p < 0.05 if need == 7 else True))}
    g = {"E0-a lake>random_lake": cmp("lake", "random_lake", 7), "E0-b lake>spline": cmp("lake", "spline", 7),
         "E0-c spline>spline_shuf (validity)": cmp("spline", "spline_shuf", 8)}
    seed_sd = [float(np.std([np.mean([np.mean(p["lake"]) for p in recs[(s, sd)]["patterns"] if p["family"] == "P3"]) for sd in SEEDS]))
               for s in SUBJECTS]
    ok = all(v["pass"] for v in g.values())
    return {"verdict": "pass" if ok else "fail", "criteria": g, "P3": sc, "P1": scores(recs, "P1"), "lake_seed_sd_P3": seed_sd, "issues": []}


def tamper_selftest(recs):
    """Each tampered copy must be flagged (inconsistent or fail)."""
    import copy
    flagged = {}
    r = copy.deepcopy(recs); k = next(iter(r)); r[k]["patterns"][0]["lake"] = [float(x) + 0.5 for x in r[k]["patterns"][0]["lake"]]
    flagged["summary_not_updated"] = verdict(r)["verdict"] == "inconsistent"
    r = copy.deepcopy(recs); r.pop(next(iter(r))); flagged["missing_seed"] = verdict(r)["verdict"] == "inconsistent"
    r = copy.deepcopy(recs); r[next(iter(r))]["dev"] = True; flagged["dev_receipt"] = verdict(r)["verdict"] == "inconsistent"
    r = copy.deepcopy(recs); r[next(iter(r))]["config"]["U"] = 100; flagged["config"] = verdict(r)["verdict"] == "inconsistent"
    r = copy.deepcopy(recs)
    for rr in r.values():   # make the lake identical to the random lake: E0-a must not pass
        for p in rr["patterns"]: p["lake"] = p["random_lake"]
        for fam in ("P1", "P3"): rr["summary"][f"lake_{fam}"] = rr["summary"][f"random_lake_{fam}"]
    flagged["lake_equals_random"] = not verdict(r).get("criteria", {}).get("E0-a lake>random_lake", {}).get("pass", True)
    return flagged


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=str(C.ROOT / "receipts/M13/E0/official"))
    ap.add_argument("--json", default=None); a = ap.parse_args(argv)
    recs = load(a.dir); v = verdict(recs)
    if v["verdict"] != "inconsistent": v["tamper"] = tamper_selftest(recs)
    txt = json.dumps(v, indent=1, ensure_ascii=False); print(txt)
    if a.json: Path(a.json).write_text(txt)
    sys.exit(0 if v["verdict"] in ("pass", "fail") else 1)


if __name__ == "__main__":
    main()
