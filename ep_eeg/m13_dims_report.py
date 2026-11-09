"""M13 修改6: aggregate the dimension receipts (descriptive; reports/M13.md §5).

  python -m ep_eeg.m13_dims_report [--lakes v0 C L_field L_none]
Per subject = mean of seeds 0,1; per dimension: 9-subject mean, and lake vs reference wins / one-sided Wilcoxon p.
"""
from __future__ import annotations
import argparse, glob, json, sys
import numpy as np
from scipy.stats import wilcoxon
from . import m13_common as C

D = C.ROOT / "receipts" / "M13" / "D"
SUBJ = range(1, 10)
DIMS = ["D0", "D1_local", "D2_theta", "D2_mu", "D2_beta", "D4_logpow_mu", "D4_logpow_beta"]
MI = ("C3", "Cz", "C4")


def audit():
    return {s: json.load(open(D / "audit" / f"A{s:02d}.json"))["summary"] for s in SUBJ}


def lake(kind):
    out = {}
    for s in SUBJ:
        fs = sorted(glob.glob(str(D / "lake" / f"A{s:02d}_s[01]_{kind}.json")))
        if not fs: continue
        rs = [json.load(open(f))["summary"] for f in fs]
        out[s] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    return out


def cmp(a, b):
    s = [k for k in SUBJ if k in a and k in b]
    d = np.array([a[k] - b[k] for k in s])
    p = float(wilcoxon(d, alternative="greater").pvalue) if len(d) and np.any(d != 0) else 1.0
    return {"n": len(s), "wins": int((d > 0).sum()), "p": round(p, 4), "mean_diff": round(float(d.mean()), 4)}


def run(kinds):
    A = audit(); res = {"refs": {}, "lakes": {}}
    def col(src, key): return {s: src[s][key] for s in src if key in src[s]}
    for ref in ("spline", "mean_visible", "nearest"):
        res["refs"][ref] = {d: round(float(np.mean(list(col(A, f"{ref}_{d}").values()))), 4) for d in DIMS}
    res["refs"]["D3_true"] = round(float(np.mean(list(col(A, "D3_true").values()))), 4)
    for r in ("visible_only", "meanvis", "spline"):
        res["refs"][f"D3_{r}"] = round(float(np.mean([np.mean([A[s][f"D3_{r}_{m}"] for m in MI]) for s in SUBJ])), 4)
    for r in ("true", "meanvis", "spline"):
        res["refs"][f"D3b_{r}"] = round(float(np.mean([np.mean([A[s][f"D3b_{r}_{m}"] for m in MI]) for s in SUBJ])), 4)
    for k in kinds:
        L = lake(k)
        if not L: continue
        row = {"n_subj": len(L)}
        for d in DIMS:
            lv = col(L, f"lake_{d}")
            row[d] = {"mean": round(float(np.mean(list(lv.values()))), 4),
                      "vs_spline": cmp(lv, col(A, f"spline_{d}")),
                      "vs_mean_visible": cmp(lv, col(A, f"mean_visible_{d}"))}
        for pre, ref in (("D3", "spline"), ("D3", "meanvis"), ("D3b", "spline"), ("D3b", "meanvis")):
            lv = {s: float(np.mean([L[s][f"{pre}_lake_{m}"] for m in MI])) for s in L}
            rv = {s: float(np.mean([A[s][f"{pre}_{ref}_{m}"] for m in MI])) for s in SUBJ}
            row.setdefault(pre, {"mean": round(float(np.mean(list(lv.values()))), 4)})[f"vs_{ref}"] = cmp(lv, rv)
        res["lakes"][k] = row
    for a_, b_ in (("C", "v0"), ("L_field", "v0"), ("L_field", "C"), ("L_field", "L_none"), ("T_none", "v0"), ("T_none", "L_none"),
                   ("T_osc", "T_none"), ("T_osc", "v0")):
        if a_ in res["lakes"] and b_ in res["lakes"]:
            La, Lb = lake(a_), lake(b_)
            res.setdefault("pairs", {})[f"{a_}>{b_}"] = {d: cmp(col(La, f"lake_{d}"), col(Lb, f"lake_{d}")) for d in DIMS}
    C.write_receipt(D / "summary.json", dict(res, written=C.now(), commit=C.git("rev-parse", "HEAD")))
    print(json.dumps(res, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--lakes", nargs="*", default=["v0", "C"])
    run(ap.parse_args(argv).lakes)


if __name__ == "__main__":
    sys.exit(main())
