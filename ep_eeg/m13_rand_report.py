"""M13 §8 verdicts: random-patch training, test on never-hidden leads (tier U) and new neighbourhoods (tier N).
  python -m ep_eeg.m13_rand_report   -> receipts/M13/R/summary.json
Per subject = mean of seeds 0, 1 (the two seeds swap the hide fold, so each lead is scored once per tier)."""
from __future__ import annotations
import glob, json, sys
import numpy as np
from . import m13_common as C
from .m13_abc import compare, SUBJ

R = C.ROOT / "receipts" / "M13" / "R"
TAIL = "ell0.03_lam0.5_U3000"


def load(pattern):
    out = {}
    for f in glob.glob(str(R / "test" / pattern)):
        r = json.load(open(f)); out[(r["subject"], r["seed"])] = r
    return out


def col(recs, key, where="summary", u=None):
    v = {}
    for s in SUBJ:
        xs = []
        for z in (0, 1):
            if (s, z) not in recs: continue
            r = recs[(s, z)]
            src = r["evals"][str(u)]["summary"] if u is not None else r[where]
            xs.append(src[key])
        v[s] = float(np.mean(xs)) if len(xs) == 2 else None
    return v


def main():
    none = load("A0?_s?_none_U3000.json"); fld = load(f"A0?_s?_field_{TAIL}.json")
    ders = [load(f"A0?_s?_der{k}_{TAIL}.json") for k in range(3)]
    def dmean(key, u=None):
        cs = [col(d, key, u=u) for d in ders]
        return {s: (float(np.mean([c[s] for c in cs])) if all(c[s] is not None for c in cs) else None) for s in SUBJ}
    t = {}
    for tier in ("U", "N"):
        k = f"lake_own_{tier}"
        t[f"none_{tier}"] = col(none, k); t[f"field_{tier}"] = col(fld, k); t[f"der_{tier}"] = dmean(k)
        for u in (0, 1000, 2000):
            t[f"none_{tier}_u{u}"] = col(none, k, u=u); t[f"field_{tier}_u{u}"] = col(fld, k, u=u); t[f"der_{tier}_u{u}"] = dmean(k, u)
        for ref in ("spline", "spline_shuf", "mean_visible", "nearest", "ridge_raw"):
            t[f"{ref}_{tier}"] = col(none, f"{ref}_{tier}", where="refs_summary")
    res = {"main_U_field_gt_der": compare(t["field_U"], t["der_U"], "tier U: field > der"),
           "U_field_gt_none": compare(t["field_U"], t["none_U"], "tier U: field > none"),
           "N_field_gt_der": compare(t["field_N"], t["der_N"], "tier N: field > der"),
           "N_field_gt_none": compare(t["field_N"], t["none_N"], "tier N: field > none"),
           "U_der_gt_none": compare(t["der_U"], t["none_U"], "tier U: der > none"),
           "U0_field_gt_der": compare(t["field_U_u0"], t["der_U_u0"], "tier U at u=0: field > der")}
    for tier in ("U", "N"):
        for arm in ("none", "field", "der"):
            for ref in ("spline", "mean_visible", "nearest", "ridge_raw"):
                res[f"{tier}_{arm}_gt_{ref}"] = compare(t[f"{arm}_{tier}"], t[f"{ref}_{tier}"], f"tier {tier}: {arm} > {ref}")
    res["means"] = {k: round(float(np.mean([v[s] for s in SUBJ])), 4) for k, v in t.items() if all(v[s] is not None for s in SUBJ)}
    res["tables"] = t
    for k, v in res.items():
        if k not in ("tables",): print(k, json.dumps(v))
    C.write_receipt(R / "summary.json", dict(res, written=C.now(), commit=C.git("rev-parse", "HEAD")))


if __name__ == "__main__":
    sys.exit(main())
