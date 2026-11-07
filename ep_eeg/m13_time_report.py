"""M13 4.5 route 2 (reports/M13.md §6): selection on the dev split, verdicts on the test split.

  python -m ep_eeg.m13_time_report --select    writes receipts/M13/T/selected.json (dev split only)
  python -m ep_eeg.m13_time_report --report    test verdicts -> receipts/M13/T/summary.json
"""
from __future__ import annotations

import argparse, glob, json, sys

import numpy as np

from . import m13_common as C
from .m13_abc import compare, e0, subj_mean, SUBJ, R as R13, load as load_abc

R = C.ROOT / "receipts" / "M13" / "T"
ETAS, US = (0.02, 0.05, 0.2), (500, 1000, 2000)
FIELD = "ell0.03_lam0.5"


def load(split, pattern):
    out = {}
    for f in glob.glob(str(R / split / pattern)):
        r = json.load(open(f)); out[(r["subject"], r["seed"])] = r
    return out


def select():
    rows = []
    for eta in ETAS:
        recs = load("dev", f"A0?_s0_none_bptt_eta{eta:g}_U2000.json")
        for u in (0,) + US:
            v = [recs[(s, 0)]["evals"][str(u)]["dyn1_lake_own_P3"] for s in SUBJ if (s, 0) in recs]
            rows.append({"eta": eta, "u": u, "n": len(v), "dev_dyn1": float(np.mean(v)) if len(v) == 9 else None})
            print(f"eta {eta:<5} u {u:>4}  dyn1 {rows[-1]['dev_dyn1']}")
    val = [r for r in rows if r["u"] > 0 and r["dev_dyn1"] is not None]
    best = max(r["dev_dyn1"] for r in val)
    pick = min((r for r in val if r["dev_dyn1"] >= best - 0.002), key=lambda r: (r["u"], r["eta"]))
    sel = {"step": "T", "rule": "max dev mean dyn1 lake_own (none, bptt); within 0.002 take smaller u, then smaller eta",
           "grid": rows, "selected": {"eta": pick["eta"], "U": pick["u"]}}
    C.write_receipt(R / "selected.json", dict(sel, written=C.now(), commit=C.git("rev-parse", "HEAD")))
    print("selected", sel["selected"])


def col(recs, key, u=None):
    """per-subject seed mean of summary[key] (or evals[u][key])."""
    v = {}
    for s in SUBJ:
        xs = [(recs[(s, z)]["summary"] if u is None else recs[(s, z)]["evals"][str(u)]).get(key) for z in (0, 1) if (s, z) in recs]
        xs = [x for x in xs if x is not None]
        v[s] = float(np.mean(xs)) if xs else None
    return v


def report():
    s = json.load(open(R / "selected.json"))["selected"]; tail = f"bptt_eta{s['eta']:g}_U{s['U']}"
    none = load("test", f"A0?_s?_none_{tail}.json"); fld = load("test", f"A0?_s?_field_{FIELD}_{tail}.json")
    ders = [load("test", f"A0?_s?_der{k}_{FIELD}_{tail}.json") for k in range(3)]
    def dmean(key, u=None):
        cs = [col(d, key, u) for d in ders]
        return {k: (float(np.mean([c[k] for c in cs])) if all(c[k] is not None for c in cs) else None) for k in SUBJ}
    K = "dyn1_lake_own_P3"
    t = {"none": col(none, K), "none_u0": col(none, K, 0), "field": col(fld, K), "field_u0": col(fld, K, 0), "der": dmean(K),
         "none_static": col(none, "static_lake_own_P3"), "field_static": col(fld, "static_lake_own_P3"),
         "none_cos": col(none, "dyn1_cos_spline_P3"), "field_cos": col(fld, "dyn1_cos_spline_P3"), "der_cos": dmean("dyn1_cos_spline_P3")}
    base = e0(); ref = {k: subj_mean(base, k) for k in ("lake_own_P3", "spline_P3")}
    cs = json.load(open(R13 / "C" / "selected.json"))["selected"]
    ctag = "_".join(f"{k}{v}" for k, v in sorted(cs.items()))
    ck = subj_mean(load_abc("C", "test", f"A0?_s?_field_{ctag}.json"), "lake_own_P3")
    r1 = subj_mean(load_abc("L", "test", "A0?_s?_none_U500.json"), "lake_own_P3")
    best = "field" if np.mean([t["field"][k] for k in SUBJ]) > np.mean([t["none"][k] for k in SUBJ]) else "none"
    res = {"selected": s,
           "main_none_learnt_gt_u0": compare(t["none"], t["none_u0"], "none in-time learnt > same lake before (dyn1)"),
           "space_field_gt_der": compare(t["field"], t["der"], "field > der (dyn1)"),
           "space_cos_field_gt_der": compare(t["field_cos"], t["der_cos"], "cos_spline field > der (dyn1)"),
           "best": best,
           "keep_best_gt_C": compare(t[best], ck, f"{best} (dyn1) > C kept (static)"),
           "best_gt_v0_static": compare(t[best], ref["lake_own_P3"], f"{best} (dyn1) > v0 static"),
           "best_gt_route1_none_static": compare(t[best], r1, f"{best} (dyn1) > route-1 none static"),
           "field_gt_none": compare(t["field"], t["none"], "field > none (dyn1)"),
           "static_eval_none_gt_v0": compare(t["none_static"], ref["lake_own_P3"], "in-time learnt none, static eval > v0"),
           "vs_spline_wins": sum(1 for k in SUBJ if t[best][k] is not None and t[best][k] > ref["spline_P3"][k]),
           "means": {k: round(float(np.mean([v[k] for k in SUBJ])), 4) for k, v in
                     dict(t, C=ck, v0=ref["lake_own_P3"], route1_none=r1, spline=ref["spline_P3"]).items()},
           "tables": dict(t, C=ck, v0=ref["lake_own_P3"], route1_none=r1, spline=ref["spline_P3"])}
    for k, v in res.items():
        if k != "tables": print(k, json.dumps(v))
    C.write_receipt(R / "summary.json", dict(res, written=C.now(), commit=C.git("rev-parse", "HEAD")))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--select", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.select: select()
    if a.report: report()


if __name__ == "__main__":
    sys.exit(main())
