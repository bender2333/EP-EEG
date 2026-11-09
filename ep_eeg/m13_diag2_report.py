"""M13 §10.4 data curve + §10.5 MI probe verdicts.  python -m ep_eeg.m13_diag2_report -> receipts/M13/G/summary2.json
Per subject = mean of seeds 0, 1 (as §8), except ERD/lr probes which are also split by tier (seed 0 = N, seed 1 = U for C3/C4)."""
from __future__ import annotations
import glob, json, sys
import numpy as np
from . import m13_common as C
from .m13_abc import compare, SUBJ

G = C.ROOT / "receipts" / "M13" / "G"
R8 = C.ROOT / "receipts" / "M13" / "R"


def load(path_glob):
    out = {}
    for f in glob.glob(str(path_glob)):
        r = json.load(open(f)); out[(r["subject"], r["seed"], r.get("frac", 1.0))] = r
    return out


def per_subj(recs, get, frac=None, seeds=(0, 1)):
    v = {}
    for s in SUBJ:
        xs = [get(recs[k]) for k in recs if k[0] == s and k[1] in seeds and (frac is None or k[2] == frac)]
        v[s] = float(np.mean(xs)) if len(xs) == len(seeds) else None
    return v


def mean(v): return round(float(np.mean([x for x in v.values() if x is not None])), 4)


def main():
    res = {}
    # ---- 10.4 data curve
    cur = load(G / "curve" / "A0?_s?_frac*_U3000.json")
    full = load(R8 / "test" / "A0?_s?_none_U3000.json")
    curve = {}
    for tier in ("U", "N"):
        for f in (0.25, 0.5):
            for p in ("lake_own", "ridge_vis", "mean_visible"):
                curve[f"{p}_{tier}@{f}"] = per_subj(cur, lambda r, k=f"{p}_{tier}": r["summary"][k], f)
        curve[f"lake_own_{tier}@1.0"] = per_subj(full, lambda r, k=f"lake_own_{tier}": r["summary"][k])
    res["curve_means"] = {k: mean(v) for k, v in curve.items()}
    for tier in ("U", "N"):
        res[f"curve_100_gt_50_{tier}"] = compare(curve[f"lake_own_{tier}@1.0"], curve[f"lake_own_{tier}@0.5"], f"{tier}: lake_own 100% > 50%")
        res[f"curve_100_gt_25_{tier}"] = compare(curve[f"lake_own_{tier}@1.0"], curve[f"lake_own_{tier}@0.25"], f"{tier}: lake_own 100% > 25%")
        res[f"curve_gain_25_to_100_{tier}"] = round(res["curve_means"][f"lake_own_{tier}@1.0"] - res["curve_means"][f"lake_own_{tier}@0.25"], 4)
    res["curve_rho_end"] = {f: mean(per_subj(cur, lambda r: r["rho_end"], f)) for f in (0.25, 0.5)}
    # ---- 10.5 probe
    pr = {(k[0], k[1]): v for k, v in load(G / "probe" / "A0?_s?.json").items()}
    pr3 = {(s, z, 1.0): r for (s, z), r in pr.items()}
    names = list(next(iter(pr.values()))["probe"].keys())
    probe = {}
    for n in names:
        for m in ("acc4", "acc_lr"):
            probe[f"{n}.{m}"] = per_subj(pr3, lambda r, n=n, m=m: r["probe"][n][m])
    res["probe_means"] = {k: mean(v) for k, v in probe.items()}
    res["probe_names"] = names
    for m in ("acc4", "acc_lr"):
        res[f"probe_trained_gt_untrained_{m}"] = compare(probe[f"trained.{m}"], probe[f"untrained.{m}"], f"probe {m}: trained lake state > untrained")
        res[f"probe_trained_gt_leads_{m}"] = compare(probe[f"trained.{m}"], probe[f"leads.{m}"], f"probe {m}: trained lake state > leads")
    # ERD lateralisation: truth vs reconstructions, by tier (seed 0 = N, seed 1 = U for C3/C4)
    erd_keys = sorted(next(iter(pr.values()))["erd"].keys())
    srcs = sorted({k.split(".")[0] for k in erd_keys})
    erd = {}
    for tier, z in (("N", 0), ("U", 1)):
        for k in erd_keys:
            erd[f"{k}@{tier}"] = per_subj(pr3, lambda r, k=k: r["erd"][k], seeds=(z,))
    res["erd_means"] = {k: mean(v) for k, v in erd.items()}
    # does lake_own carry ERD: same sign as truth and |x| >= 50% |truth|, count subjects (>= 7/9)
    carry = {}
    for tier in ("N", "U"):
        for src in srcs:
            if src == "truth": continue
            for ch in ("C3", "C4"):
                for b in ("mu", "beta"):
                    t = erd[f"truth.{ch}_{b}@{tier}"]; x = erd[f"{src}.{ch}_{b}@{tier}"]
                    carry[f"{src}.{ch}_{b}@{tier}"] = int(sum(np.sign(x[s]) == np.sign(t[s]) and abs(x[s]) >= 0.5 * abs(t[s]) for s in SUBJ))
    res["erd_carry_count"] = carry
    # truth itself: how many subjects show contralateral ERD (C3 left-right > 0, C4 < 0)
    res["truth_contralateral"] = {f"{ch}_{b}@{tier}": int(sum((erd[f'truth.{ch}_{b}@{tier}'][s] > 0) == (ch == "C3") for s in SUBJ))
                                  for tier in ("N", "U") for ch in ("C3", "C4") for b in ("mu", "beta")}
    lr = {}
    for tier, z in (("N", 0), ("U", 1)):
        for src in next(iter(pr.values()))["lr_from_C3C4"].keys():
            lr[f"{src}@{tier}"] = per_subj(pr3, lambda r, src=src: r["lr_from_C3C4"][src], seeds=(z,))
    res["lr_from_C3C4_means"] = {k: mean(v) for k, v in lr.items()}
    for tier in ("N", "U"):
        res[f"lr_lake_gt_mean_{tier}"] = compare(lr[f"lake_own@{tier}"], lr[f"mean_visible@{tier}"], f"lr from C3C4 {tier}: lake_own > mean_visible")
    res["tables"] = {"curve": curve, "probe": probe, "erd": erd, "lr": lr}
    for k, v in res.items():
        if k != "tables": print(k, json.dumps(v))
    C.write_receipt(G / "summary2.json", dict(res, written=C.now(), commit=C.git("rev-parse", "HEAD")))


if __name__ == "__main__":
    sys.exit(main())
