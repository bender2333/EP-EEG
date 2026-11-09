"""M13 §10 verdicts (reports/M13.md §10.3).  python -m ep_eeg.m13_diag_report -> receipts/M13/G/summary.json
Per subject = mean of seeds 0, 1 (as §8)."""
from __future__ import annotations
import glob, json, sys
import numpy as np
from . import m13_common as C
from .m13_abc import compare, SUBJ
from .m13_diag import PRED

G = C.ROOT / "receipts" / "M13" / "G"
R8 = C.ROOT / "receipts" / "M13" / "R"
MET = ("r2", "phase_mu", "phase_beta", "logpow_theta", "logpow_mu", "logpow_beta", "shape", "mu_peak_err", "track_mu", "track_beta", "corr_mv")
EXTRA = ("truth_corr_mv", "lake_var_by_mv", "truth_var_by_mv", "scalar", "qualified")


def load(pattern, root=G):
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
    A = {k: load(f"A0?_s?_{k}_U3000.json") for k in ("none_raw", "field_raw", "none_lap")}
    t = {}
    for arm, recs in A.items():
        for tier in ("U", "N"):
            for p in PRED:
                for m in MET: t[f"{arm}.{p}.{m}_{tier}"] = col(recs, lambda r, k=f"{p}.{m}_{tier}": r["summary"][k])
            for m in EXTRA: t[f"{arm}.{m}_{tier}"] = col(recs, lambda r, k=f"{m}_{tier}": r["summary"][k])
        t[f"{arm}.rho_end"] = col(recs, lambda r: r["rho_end"])
    # reproduction: raw lakes vs the §8 receipts (same final weights)
    rep = {}
    for cond in ("none", "field"):
        old = load(f"A0?_s?_{cond}{'' if cond == 'none' else '_ell0.03_lam0.5'}_U3000.json", R8)
        new = A[f"{cond}_raw"]
        rep[cond] = {"n": len([k for k in new if k in old]),
                     "same_final_sha": sum(new[k]["final_sha256"] == old[k]["final_sha256"] for k in new if k in old)}
    cmp = lambda a, b, lab: compare(t[a], t[b], lab)
    d = lambda a, b: {s: (t[a][s] - t[b][s]) if t[a][s] is not None and t[b][s] is not None else None for s in SUBJ}
    res = {"reproduction_vs_s8": rep}
    # H1 (existing receipts)
    s8 = json.load(open(R8 / "summary.json")); s9 = json.load(open(C.ROOT / "receipts" / "M13" / "P" / "summary.json"))
    sc = json.load(open(R8 / "check" / "scale.json"))
    res["H1"] = {"s8_U_field_gt_der": s8["main_U_field_gt_der"], "s9_L2_field_gt_der": s9["L2_field_gt_der"],
                 "s9_gain_field_gt_der": s9["gain_L2"], "s8_check3_scale": {k: float(np.mean([v[k] for v in sc["per_subject"].values()])) for k in ("lake_own_P3", "scaled_P3", "mean_scalar")}}
    for m in ("logpow_mu", "logpow_beta", "shape", "track_mu", "track_beta", "phase_mu"):
        res["H1"][f"field_gt_none_{m}_U"] = cmp(f"field_raw.lake_own.{m}_U", f"none_raw.lake_own.{m}_U", f"raw U field > none: {m}")
    # H2
    res["H2a_lake_more_like_mean_than_truth"] = cmp("none_raw.lake_own.corr_mv_U", "none_raw.truth_corr_mv_U", "raw U: corr(lake, mean) > corr(truth, mean)")
    res["H2b_lap_lake_gt_mean"] = cmp("none_lap.lake_own.r2_U", "none_lap.mean_visible.r2_U", "lap U: lake > mean_visible")
    res["H2b_lap_lake_gt_nearest"] = cmp("none_lap.lake_own.r2_U", "none_lap.nearest.r2_U", "lap U: lake > nearest")
    res["H2_raw_lake_gt_mean"] = cmp("none_raw.lake_own.r2_U", "none_raw.mean_visible.r2_U", "raw U: lake > mean_visible")
    res["H2_lap_ridge_lake_gt_ridge_vis"] = cmp("none_lap.lake_ridge.r2_U", "none_lap.ridge_vis.r2_U", "lap U: lake_ridge > ridge_vis")
    # H3
    for arm in ("none_raw", "field_raw"):
        dd = d(f"{arm}.lake_own.logpow_theta_U", f"{arm}.lake_own.logpow_beta_U")
        res[f"H3_flat_{arm}"] = {"n_abs_theta_minus_beta_lt_0.1": int(sum(abs(v) < 0.1 for v in dd.values() if v is not None)),
                                 "mean_theta_minus_beta": float(np.mean([v for v in dd.values() if v is not None]))}
        res[f"H3_readout_r2_{arm}"] = cmp(f"{arm}.lake_ridge.r2_U", f"{arm}.lake_own.r2_U", f"{arm} U: lake_ridge > lake_own (R2)")
        dr = d(f"{arm}.lake_ridge.r2_U", f"{arm}.lake_own.r2_U")
        res[f"H3_readout_r2_{arm}"]["n_gain_ge_0.1"] = int(sum(v >= 0.1 for v in dr.values() if v is not None))
        for b in ("mu", "beta"):
            res[f"H3_readout_pow_{b}_{arm}"] = cmp(f"{arm}.lake_ridge.logpow_{b}_U", f"{arm}.lake_own.logpow_{b}_U", f"{arm} U: lake_ridge power {b} > lake_own")
    # priors side vs waveform side, raw none, tier U
    for ref in ("mean_visible", "nearest", "spline"):
        for m in ("shape", "track_mu", "track_beta", "phase_mu", "phase_beta", "r2"):
            res[f"P_lake_gt_{ref}_{m}"] = cmp(f"none_raw.lake_own.{m}_U", f"none_raw.{ref}.{m}_U", f"raw U lake > {ref}: {m}")
    res["means"] = {k: round(float(np.mean([v[s] for s in SUBJ])), 4) for k, v in t.items() if all(v[s] is not None for s in SUBJ)}
    res["tables"] = t
    for k, v in res.items():
        if k not in ("tables", "means"): print(k, json.dumps(v))
    C.write_receipt(G / "summary.json", dict(res, written=C.now(), commit=C.git("rev-parse", "HEAD")))


if __name__ == "__main__":
    sys.exit(main())
