"""M13 §9 verdicts (reports/M13.md §9.2): gain cap rho~ raised to 0.99 on the §8 random-patch task.
  python -m ep_eeg.m13_rho_report   -> receipts/M13/P/summary.json
Per subject = mean of seeds 0, 1 (as §8). References (mean of visible leads, nearest, spline, ridge) are the §8 ones
(same data, same test sets; they do not depend on the lake)."""
from __future__ import annotations
import glob, json, sys
import numpy as np
from . import m13_common as C
from .m13_abc import compare, SUBJ

P = C.ROOT / "receipts" / "M13" / "P"
R8 = C.ROOT / "receipts" / "M13" / "R"
TAIL = "ell0.03_lam0.5_U3000"


def load(pattern, root=P):
    out = {}
    for f in glob.glob(str(root / "test" / pattern)):
        r = json.load(open(f)); out[(r["subject"], r["seed"])] = r
    return out


def col(recs, get):
    v = {}
    for s in SUBJ:
        xs = [get(recs[(s, z)]) for z in (0, 1) if (s, z) in recs]
        v[s] = float(np.mean(xs)) if len(xs) == 2 else None
    return v


def main():
    A = {}
    for cap in ("0.99", "0.9"):
        A[cap] = {"none": load(f"A0?_s?_cap{cap}_none_U3000.json"), "field": load(f"A0?_s?_cap{cap}_field_{TAIL}.json")}
    ders = [load(f"A0?_s?_cap0.99_der{k}_{TAIL}.json") for k in range(3)]
    n8 = load("A0?_s?_none_U3000.json", R8); f8 = load(f"A0?_s?_field_{TAIL}.json", R8)
    ders8 = [load(f"A0?_s?_der{k}_{TAIL}.json", R8) for k in range(3)]
    def dmean(ds, get):
        cs = [col(d, get) for d in ds]
        return {s: (float(np.mean([c[s] for c in cs])) if all(c[s] is not None for c in cs) else None) for s in SUBJ}
    t = {}
    for tier in ("U", "N"):
        for key in ("lake_own", "gain"):
            g = lambda r, k=f"{key}_{tier}": r["summary"][k]
            for cap in ("0.99", "0.9"):
                for arm in ("none", "field"): t[f"{arm}_{key}_{tier}_cap{cap}"] = col(A[cap][arm], g)
            t[f"der_{key}_{tier}_cap0.99"] = dmean(ders, g)
        for b in ("mu", "beta"):
            g = lambda r, k=f"logpow_{b}_{tier}": r["summary"][k]
            for cap in ("0.99", "0.9"):
                for arm in ("none", "field"): t[f"{arm}_logpow_{b}_{tier}_cap{cap}"] = col(A[cap][arm], g)
            for ref in ("mean_visible", "nearest"):
                t[f"{ref}_logpow_{b}_{tier}"] = col(A["0.99"]["none"], lambda r, k=f"{ref}_logpow_{b}_{tier}": r["power_refs_summary"][k])
        for ref in ("spline", "mean_visible", "nearest", "ridge_raw"):
            t[f"{ref}_{tier}"] = col(n8, lambda r, k=f"{ref}_{tier}": r["refs_summary"][k])
        t[f"none_s8_{tier}"] = col(n8, lambda r, k=f"lake_own_{tier}": r["summary"][k])
        t[f"field_s8_{tier}"] = col(f8, lambda r, k=f"lake_own_{tier}": r["summary"][k])
        t[f"der_s8_{tier}"] = dmean(ders8, lambda r, k=f"lake_own_{tier}": r["summary"][k])
    for cap in ("0.99", "0.9"):
        for arm in ("none", "field"):
            t[f"{arm}_rho_end_cap{cap}"] = col(A[cap][arm], lambda r: r["rho_end"])
            t[f"{arm}_proj_frac_cap{cap}"] = col(A[cap][arm], lambda r: r["train"]["projections"] / r["U"])
    # reproduction check: cap 0.9 reruns vs §8 receipts (lake_own per lead)
    rep = []
    for arm, old in (("none", n8), ("field", f8)):
        for k, r in A["0.9"][arm].items():
            if k in old: rep.append(max(abs(r["final"][c]["lake_own"] - old[k]["evals"]["3000"]["per_lead"][c]["lake_own"]) for c in C.LEADS))
    cmp = lambda a, b, lab: compare(t[a], t[b], lab)
    res = {"L1_field_gt_mean_visible": cmp("field_lake_own_U_cap0.99", "mean_visible_U", "L1 tier U cap .99: field > mean_visible"),
           "L2_field_gt_der": cmp("field_lake_own_U_cap0.99", "der_lake_own_U_cap0.99", "L2 tier U cap .99: field > der"),
           "gain_L1": cmp("field_gain_U_cap0.99", "mean_visible_U", "gain, tier U cap .99: field > mean_visible"),
           "gain_L2": cmp("field_gain_U_cap0.99", "der_gain_U_cap0.99", "gain, tier U cap .99: field > der"),
           "U_field_gt_none": cmp("field_lake_own_U_cap0.99", "none_lake_own_U_cap0.99", "tier U cap .99: field > none"),
           "U_der_gt_none": cmp("der_lake_own_U_cap0.99", "none_lake_own_U_cap0.99", "tier U cap .99: der > none"),
           "N_field_gt_der": cmp("field_lake_own_N_cap0.99", "der_lake_own_N_cap0.99", "tier N cap .99: field > der"),
           "N_field_gt_none": cmp("field_lake_own_N_cap0.99", "none_lake_own_N_cap0.99", "tier N cap .99: field > none"),
           "none_U_99_gt_90": cmp("none_lake_own_U_cap0.99", "none_lake_own_U_cap0.9", "tier U none: cap .99 > .9"),
           "field_U_99_gt_90": cmp("field_lake_own_U_cap0.99", "field_lake_own_U_cap0.9", "tier U field: cap .99 > .9"),
           "none_N_99_gt_90": cmp("none_lake_own_N_cap0.99", "none_lake_own_N_cap0.9", "tier N none: cap .99 > .9"),
           "field_N_99_gt_90": cmp("field_lake_own_N_cap0.99", "field_lake_own_N_cap0.9", "tier N field: cap .99 > .9"),
           "U_field_gt_nearest": cmp("field_lake_own_U_cap0.99", "nearest_U", "tier U cap .99: field > nearest"),
           "U_gain_field_gt_nearest": cmp("field_gain_U_cap0.99", "nearest_U", "gain tier U cap .99: field > nearest"),
           "reproduction_max_abs_diff_cap0.9_vs_s8": float(max(rep)) if rep else None, "reproduction_n": len(rep)}
    res["means"] = {k: round(float(np.mean([v[s] for s in SUBJ])), 4) for k, v in t.items() if all(v[s] is not None for s in SUBJ)}
    res["tables"] = t
    for k, v in res.items():
        if k != "tables": print(k, json.dumps(v))
    C.write_receipt(P / "summary.json", dict(res, written=C.now(), commit=C.git("rev-parse", "HEAD")))


if __name__ == "__main__":
    sys.exit(main())
