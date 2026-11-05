"""M13 A/B/C: selection on the dev split, verdicts on the test split, spatial-relation tables (reports/M13.md §3).

  python -m ep_eeg.m13_abc --select B|C1|C2     writes receipts/M13/<B|C>/selected.json (dev split only)
  python -m ep_eeg.m13_abc --report             prints the step tables and verdicts from the test receipts
"""
from __future__ import annotations

import argparse, glob, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from . import m13_common as C

R = C.ROOT / "receipts" / "M13"
SUBJ = range(1, 10)


def load(step, split, pattern):
    out = {}
    for f in glob.glob(str(R / step / split / pattern)):
        r = json.load(open(f)); out[(r["subject"], r["seed"])] = r
    return out


def e0():
    out = {}
    for f in glob.glob(str(R / "E0" / "official" / "A*_s[01].json")):
        r = json.load(open(f)); out[(r["subject"], r["seed"])] = r
    return out


def subj_mean(recs, key, seeds=(0, 1)):
    """per-subject mean over available seeds of summary[key]; None where nothing."""
    v = {}
    for s in SUBJ:
        xs = [recs[(s, z)]["summary"].get(key) for z in seeds if (s, z) in recs]
        xs = [x for x in xs if x is not None]
        v[s] = float(np.mean(xs)) if xs else None
    return v


def compare(new, old, label):
    s = [k for k in SUBJ if new.get(k) is not None and old.get(k) is not None]
    d = np.array([new[k] - old[k] for k in s])
    wins = int((d > 0).sum())
    p = float(wilcoxon(d, alternative="greater").pvalue) if len(d) and np.any(d != 0) else 1.0
    keep = len(s) == 9 and wins >= 7 and p < 0.05
    return {"label": label, "n": len(s), "wins": wins, "p": p, "mean_diff": float(d.mean()) if len(d) else None,
            "pass": keep}


def dev_mean(step, arm_glob):
    recs = load(step, "dev", arm_glob)
    vals = [r["summary"]["lake_own_P3"] for r in recs.values()]
    fin = all(r["summary"]["finite_P3"] == 1.0 for r in recs.values())
    conv = all(all(x is None or x <= 1e-6 for x in r.get("max_residual", [None])) for r in recs.values())
    return (float(np.mean(vals)) if len(vals) == 9 else None), fin and conv, len(vals)


def select(which):
    if which == "B":
        rows = []
        for z in (0.05, 0.2, 0.5, 1.0):
            m, ok, n = dev_mean("B", f"A0?_s0_inert_zeta{z}.json"); rows.append({"zeta": z, "dev_lake_own_P3": m, "valid": ok, "n": n})
        val = [r for r in rows if r["valid"] and r["dev_lake_own_P3"] is not None]
        best = max(r["dev_lake_own_P3"] for r in val)
        pick = max((r for r in val if r["dev_lake_own_P3"] >= best - 0.005), key=lambda r: r["zeta"])
        sel = {"step": "B", "rule": "max dev mean; within 0.005 take larger zeta", "grid": rows, "selected": {"zeta": pick["zeta"]}}
    elif which == "C1":
        rows = []
        for att in ("out", "cols"):
            for ell in (0.02, 0.04, 0.08):
                for a in (0.3, 0.6, 1.0):
                    m, ok, n = dev_mean("C", f"A0?_s0_field_alpha{a}_attach{att}_ell{ell}.json")
                    rows.append({"attach": att, "ell": ell, "alpha": a, "dev_lake_own_P3": m, "valid": ok, "n": n})
        val = [r for r in rows if r["valid"] and r["dev_lake_own_P3"] is not None]
        best = max(r["dev_lake_own_P3"] for r in val)
        near = [r for r in val if r["dev_lake_own_P3"] >= best - 0.005]
        pick = min(near, key=lambda r: (r["alpha"], -r["dev_lake_own_P3"]))
        sel = {"step": "C1", "rule": "quasi-static; max dev mean; within 0.005 take smaller alpha", "grid": rows,
               "selected": {k: pick[k] for k in ("attach", "ell", "alpha")}}
    elif which == "C2":
        c1 = json.load(open(R / "C" / "selected_C1.json"))["selected"]
        base, ok0, _ = dev_mean("C", f"A0?_s0_field_alpha{c1['alpha']}_attach{c1['attach']}_ell{c1['ell']}.json")
        rows = [{"c": None, "dev_lake_own_P3": base, "valid": ok0}]
        for c in (0.5, 2.0):
            cfg = dict(c1, c=c)
            tag = "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))
            m, ok, n = dev_mean("C", f"A0?_s0_field_{tag}.json"); rows.append({"c": c, "dev_lake_own_P3": m, "valid": ok, "n": n})
        val = [r for r in rows[1:] if r["valid"] and r["dev_lake_own_P3"] is not None]
        pick = max(val, key=lambda r: r["dev_lake_own_P3"]) if val else None
        c = pick["c"] if pick and pick["dev_lake_own_P3"] > base + 0.005 else None
        sel = {"step": "C2", "rule": "wave replaces quasi-static only if dev mean > quasi-static + 0.005", "grid": rows,
               "selected": dict(c1, **({"c": c} if c else {}))}
    else:
        raise ValueError(which)
    name = {"B": "selected.json", "C1": "selected_C1.json", "C2": "selected.json"}[which]
    (R / which[0]).mkdir(parents=True, exist_ok=True)
    C.write_receipt(R / which[0] / name, dict(sel, written=C.now(), commit=C.git("rev-parse", "HEAD")))
    print(json.dumps(sel, indent=1))


def table(arms):
    """arms: list of (name, {(subj, seed): rec}) -> per-subject lake_own / cos_spline / rho_dist means."""
    out = {}
    for name, recs in arms:
        out[name] = {k: subj_mean(recs, f"{k}_P3") for k in ("lake_own", "cos_spline", "rho_dist")}
    return out


def fmt(x): return "—" if x is None else f"{x:.3f}"


def report():
    base = e0()
    ref = {k: subj_mean(base, k) for k in ("lake_own_P3", "spline_P3", "mean_visible_P3", "random_lake_own_P3")}
    A_static = load("A", "test", "A0?_s?_static.json"); A_power = load("A", "test", "A0?_s?_power.json")
    lines = []
    def emit(s=""): lines.append(s); print(s)
    emit("E0 baseline (subject mean over seeds 0,1)")
    for s in SUBJ:
        emit(f"  A{s:02d} lake_own {fmt(ref['lake_own_P3'][s])} spline {fmt(ref['spline_P3'][s])} "
             f"mean_vis {fmt(ref['mean_visible_P3'][s])} random {fmt(ref['random_lake_own_P3'][s])}")
    st = table([("static", A_static), ("power", A_power)])
    emit("static re-run vs E0 (should be equal): max |diff| = " + fmt(max(abs(st["static"]["lake_own"][s] - ref["lake_own_P3"][s])
                                                                      for s in SUBJ if st["static"]["lake_own"][s] is not None and ref["lake_own_P3"][s] is not None)))
    res = {"A": compare(st["static"]["lake_own"], st["power"]["lake_own"], "A: static(raw) > power")}
    emit(json.dumps(res["A"]))
    prev = st["static"]; prev_name = "static"
    for step, arm, extra in (("B", "inert", "dyn1"), ("C", "field", None)):
        sel_f = R / step / "selected.json"
        if not sel_f.exists(): continue
        cfg = json.load(open(sel_f))["selected"]
        tag = "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))
        main = load(step, "test", f"A0?_s?_{arm}_{tag}.json")
        tb = table([(arm, main)])[arm]
        res[step] = {"cfg": cfg, "keep": compare(tb["lake_own"], prev["lake_own"], f"{step}: {arm} > {prev_name}"),
                     "space": compare(tb["cos_spline"], prev["cos_spline"], f"{step}: cos_spline {arm} > {prev_name}"),
                     "vs_spline_wins": sum(1 for s in SUBJ if tb["lake_own"][s] is not None and tb["lake_own"][s] > ref["spline_P3"][s])}
        if extra:
            ex = table([(extra, load(step, "test", f"A0?_s?_{extra}.json"))])[extra]
            res[step]["vs_" + extra] = compare(tb["lake_own"], ex["lake_own"], f"{step}: {arm} > {extra}")
            res[step][extra] = ex
        if step == "C":
            der = {}
            for k in range(3):
                for key, r in load(step, "test", f"A0?_s?_{arm}_{tag}_der{k}.json").items():
                    der.setdefault(key, []).append(r)
            der_rec = {key: {"summary": {f: float(np.mean([r["summary"][f] for r in rs])) for f in ("lake_own_P3", "cos_spline_P3", "rho_dist_P3")}}
                       for key, rs in der.items() if len(rs) == 3}
            dt = table([("der", der_rec)])["der"]
            res[step]["true_vs_derange"] = compare(tb["lake_own"], dt["lake_own"], "C: true nodes > deranged")
            res[step]["derange"] = dt
            rnd = table([("rand", load(step, "test", f"A0?_s0_{arm}_{tag}_rand.json"))])["rand"]
            res[step]["random"] = rnd
        res[step]["table"] = tb
        emit(json.dumps({k: v for k, v in res[step].items() if k not in ("table", "dyn1", "derange", "random")}))
        if res[step]["keep"]["pass"]:
            prev, prev_name = tb, arm
    res["static"] = st["static"]; res["power"] = st["power"]; res["e0"] = ref
    C.write_receipt(R / "abc_summary.json", dict(res, written=C.now(), commit=C.git("rev-parse", "HEAD")))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--select"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.select: select(a.select)
    if a.report: report()


if __name__ == "__main__":
    sys.exit(main())
