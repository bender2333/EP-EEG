"""M13 §4.5 route 1: dev selection and test verdicts (reports/M13.md §4).

  python -m ep_eeg.m13_learn_report --select    writes receipts/M13/L/selected.json (dev split only)
  python -m ep_eeg.m13_learn_report --report    test-split tables and verdicts -> receipts/M13/L/summary.json
"""
from __future__ import annotations

import argparse, glob, json, sys

import numpy as np

from . import m13_common as C
from .m13_abc import compare, e0, subj_mean, SUBJ, R as R13

R = R13 / "L"
ELLS, LAMS, US = (0.03, 0.06, 0.10), (0.25, 0.5), (0, 500, 1000, 2000, 3000)


def load(split, pattern):
    out = {}
    for f in glob.glob(str(R / split / pattern)):
        r = json.load(open(f)); out[(r["subject"], r["seed"])] = r
    return out


def at(recs, u, key="lake_own_P3"):
    return {k: r["evals"][str(u)][key] for k, r in recs.items()}


def select():
    rows = []
    none = load("dev", "A0?_s0_none_U3000.json")
    for ell in ELLS:
        for lam in LAMS:
            f = load("dev", f"A0?_s0_field_ell{ell:g}_lam{lam:g}_U3000.json")
            d = load("dev", f"A0?_s0_der0_ell{ell:g}_lam{lam:g}_U3000.json")
            ok = len(f) == 9 and len(d) == 9 and all(r["train"]["certified_end"] for r in list(f.values()) + list(d.values()))
            for u in US:
                if not ok: rows.append({"ell": ell, "lam": lam, "u": u, "valid": False}); continue
                fv, dv = at(f, u), at(d, u)
                diff = [fv[k] - dv[k] for k in fv]
                rows.append({"ell": ell, "lam": lam, "u": u, "valid": True, "field_minus_der": float(np.mean(diff)),
                             "wins": int(sum(x > 0 for x in diff)), "field": float(np.mean(list(fv.values()))),
                             "der": float(np.mean(list(dv.values()))),
                             "none": float(np.mean(list(at(none, u).values()))) if len(none) == 9 else None,
                             "cos_field": float(np.mean(list(at(f, u, "cos_spline_P3").values()))),
                             "cos_der": float(np.mean(list(at(d, u, "cos_spline_P3").values()))),
                             "cos_none": float(np.mean(list(at(none, u, "cos_spline_P3").values()))) if len(none) == 9 else None})
    val = [r for r in rows if r["valid"] and r["u"] > 0]
    best = max(r["field_minus_der"] for r in val)
    near = [r for r in val if r["field_minus_der"] >= best - 0.002]
    pick = min(near, key=lambda r: (r["lam"], r["u"]))
    sel = {"step": "L", "rule": "max dev mean(field - der) over (ell, lam, u>0); within 0.002 take smaller lam, then smaller u",
           "grid": rows, "selected": {"ell": pick["ell"], "lam": pick["lam"], "U": pick["u"]}}
    C.write_receipt(R / "selected.json", dict(sel, written=C.now(), commit=C.git("rev-parse", "HEAD")))
    for r in rows:
        if r["valid"]:
            print(f"ell {r['ell']:.2f} lam {r['lam']:.2f} u {r['u']:4d}  field {r['field']:.4f} der {r['der']:.4f} none {r['none']:.4f}"
                  f"  f-d {r['field_minus_der']:+.4f} ({r['wins']}/9)  cos f/d/n {r['cos_field']:.3f}/{r['cos_der']:.3f}/{r['cos_none']:.3f}")
    print("selected", sel["selected"])


def table(recs):
    return {k: subj_mean(recs, f"{k}_P3") for k in ("lake_own", "cos_spline", "rho_dist")}


def report():
    s = json.load(open(R / "selected.json"))["selected"]; U = s["U"]; tag = f"ell{s['ell']:g}_lam{s['lam']:g}_U{U}"
    fld = table(load("test", f"A0?_s?_field_{tag}.json"))
    nonr = table(load("test", f"A0?_s?_none_U{U}.json"))
    der = {}
    for k in range(3):
        for key, r in load("test", f"A0?_s?_der{k}_{tag}.json").items(): der.setdefault(key, []).append(r)
    der_rec = {key: {"summary": {f: float(np.mean([r["summary"][f] for r in rs])) for f in ("lake_own_P3", "cos_spline_P3", "rho_dist_P3")}}
               for key, rs in der.items() if len(rs) == 3}
    dr = table(der_rec)
    base = e0(); ref = {k: subj_mean(base, k) for k in ("lake_own_P3", "spline_P3")}
    ckept = {}
    cs = json.load(open(R13 / "C" / "selected.json"))["selected"]
    ctag = "_".join(f"{k}{v}" for k, v in sorted(cs.items()))
    from .m13_abc import load as load_abc
    ckept = table(load_abc("C", "test", f"A0?_s?_field_{ctag}.json"))
    res = {"selected": s,
           "main_field_gt_der": compare(fld["lake_own"], dr["lake_own"], "field > der (lake_own)"),
           "space_cos_field_gt_none": compare(fld["cos_spline"], nonr["cos_spline"], "cos_spline field > none"),
           "space_cos_field_gt_der": compare(fld["cos_spline"], dr["cos_spline"], "cos_spline field > der"),
           "keep_field_gt_C": compare(fld["lake_own"], ckept["lake_own"], "field > C kept"),
           "field_gt_v0": compare(fld["lake_own"], ref["lake_own_P3"], "field > v0"),
           "field_gt_none": compare(fld["lake_own"], nonr["lake_own"], "field > none"),
           "none_gt_v0": compare(nonr["lake_own"], ref["lake_own_P3"], "none > v0"),
           "vs_spline_wins": sum(1 for k in SUBJ if fld["lake_own"][k] is not None and fld["lake_own"][k] > ref["spline_P3"][k]),
           "tables": {"field": fld, "der": dr, "none": nonr, "C": ckept, "v0": ref["lake_own_P3"], "spline": ref["spline_P3"]}}
    for k, v in res.items():
        if k != "tables": print(k, json.dumps(v))
    for name, t in res["tables"].items():
        vals = t["lake_own"] if isinstance(t, dict) and "lake_own" in t else t
        print(name, "lake_own mean", round(float(np.mean([vals[k] for k in SUBJ])), 4),
              *(["cos", round(float(np.mean([t["cos_spline"][k] for k in SUBJ])), 4), "rho", round(float(np.mean([t["rho_dist"][k] for k in SUBJ])), 4)]
                if isinstance(t, dict) and "cos_spline" in t else []))
    C.write_receipt(R / "summary.json", dict(res, written=C.now(), commit=C.git("rev-parse", "HEAD")))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--select", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.select: select()
    if a.report: report()


if __name__ == "__main__":
    sys.exit(main())
