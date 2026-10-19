"""M11 verifier (reports/M11.md §9): recompute every M11 decision from the per-run receipts.

Fails (exit 1) on a missing subject or seed, or on a non-finite value that its receipt does not flag as
diverged/crashed (amendment M1-6). Kappas are recomputed from the stored predictions and labels where the
receipt keeps them. Writes reports/M11_verify.json and prints one block per step.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score
from .data import ROOT
from .stats_util import paired_wilcoxon, bayes_hier, rank_biserial

R = ROOT / "receipts" / "M11"
SUBJ = range(1, 10)
PUBLIC = {"mdm": 0.488, "tslr": 0.626, "eegnet_moabb": 0.473, "eegnet_bd": 0.473}   # §3.2, MOABB within-session
M0_PUBLIC = {"fbcsp": (0.483, 0.554), "eegnet": (0.401, 0.473)}                     # (M0 kappa_val, public)
FBCSP = 0.483
problems = []


def load(pattern):
    out = []
    for f in sorted(glob.glob(str(R / pattern))):
        try: out.append(json.load(open(f)))
        except Exception as e: problems.append(f"unreadable {f}: {e}")
    return out


def by(recs, *keys):
    d = {}
    for r in recs: d.setdefault(tuple(r[k] for k in keys), []).append(r)
    return d


def complete(recs, label, seeds=(0, 1, 2), subjects=SUBJ, subj_key="subject"):
    have = {(r[subj_key], r["seed"]) for r in recs}; miss = [(s, r) for s in subjects for r in seeds if (s, r) not in have]
    if miss: problems.append(f"{label}: missing {len(miss)} (subject, seed) e.g. {miss[:4]}")
    return not miss


def kappa(r, pred="pred", lab="label"):
    k = float(cohen_kappa_score(r[lab], r[pred]))
    if "kappa_val" in r and abs(k - r["kappa_val"]) > 1e-9: problems.append(f"kappa mismatch {r.get('arm')} {r.get('subject')} {r.get('seed')}")
    return k


def subj_mean(recs, f):
    g = by(recs, "subject"); return np.array([np.mean([f(r) for r in g[(s,)]]) for s in SUBJ])


def subj_seed(recs, f, seeds):
    g = {(r["subject"], r["seed"]): r for r in recs}; return np.array([[f(g[(s, k)]) for k in seeds] for s in SUBJ])


def stats(a, b, d_sr=None):
    out = {"wilcoxon": paired_wilcoxon(a, b), "rank_biserial": rank_biserial(a, b),
           "bayes": bayes_hier(d_sr if d_sr is not None else np.asarray(a) - np.asarray(b))}
    return out


def fbcsp_subject():
    rs = json.load(open(ROOT / "receipts/M0/fbcsp/db12387/runs.json"))
    return np.array([np.mean([float(cohen_kappa_score(r["y"], r["pred"])) for r in rs if r["subject"] == s]) for s in SUBJ])


# ---------------- A ----------------
def step_a(commit):
    out = {}
    for arm in ("mdm", "tslr", "eegnet_moabb", "eegnet_bd", "mdm_moabbpp", "tslr_moabbpp"):
        recs = load(f"A/{commit}/{arm}/*.json")
        if not recs: continue
        complete(recs, f"A/{arm}"); ks = subj_mean(recs, kappa)
        o = {"kappa_mean": float(ks.mean()), "per_subject": ks.round(4).tolist(), "settle_rate": 1.0, "rejected": 0,
             "seed_sd_mean": float(np.mean([np.std([kappa(r) for r in g]) for g in by(recs, "subject").values()]))}
        if arm in PUBLIC:
            gap = PUBLIC[arm] - o["kappa_mean"]; o["public"] = PUBLIC[arm]; o["gap_to_public"] = float(gap)
            o["flag"] = "欠训" if gap > 0.1 else ("高于公开数字" if gap < -0.1 else "一致（差距 ≤ 0.1）")
        out[arm] = o
    for arm, (k, pub) in M0_PUBLIC.items():
        gap = pub - k; out["M0_" + arm] = {"kappa_mean": k, "public": pub, "gap_to_public": gap,
                                           "flag": "欠训" if gap > 0.1 else ("高于公开数字" if gap < -0.1 else "一致（差距 ≤ 0.1）")}
    return out


# ---------------- B ----------------
LINEAR_CAND = ("random_lake", "m6_trained", "m64_random", "m64_trained")


def step_b(commit):
    out = {}; fb = fbcsp_subject(); out["fbcsp_per_subject"] = fb.round(4).tolist(); out["fbcsp_mean"] = float(fb.mean())
    for arm in ("random_lake", "m6_trained", "m64_random", "m64_trained", "m9_reservoir", "m9_bptt", "input_track2", "input_rawseg",
                "m6_trained_stored", "m64_trained_stored"):
        recs = load(f"B/{commit}/{arm}/*.json")
        if not recs: continue
        seeds = (0,) if arm.endswith("_stored") else (0, 1, 2)
        complete(recs, f"B/{arm}", seeds=seeds)
        for r in recs:  # the honest kappa is recomputed from its stored predictions
            k = float(cohen_kappa_score(r["label"], r["pred_cv"]))
            if abs(k - r["kappa_cv"]) > 1e-9: problems.append(f"B kappa_cv mismatch {arm} {r['subject']} {r['seed']}")
        kcv, kor = subj_mean(recs, lambda r: r["kappa_cv"]), subj_mean(recs, lambda r: r["kappa_or"])
        lin = [r["diag"].get("linear_r2") for r in recs if r["diag"].get("linear_r2") is not None]
        o = {"kappa_cv": float(kcv.mean()), "kappa_or": float(kor.mean()), "kappa_cv_per_subject": kcv.round(4).tolist(),
             "kappa_or_per_subject": kor.round(4).tolist(), "settle_rate": float(np.mean([r["settle_rate"] for r in recs])),
             "rejected": int(sum(r["rejected"] for r in recs)), "linear_r2_min": float(min(lin)) if lin else None,
             "cert_all": all(r["diag"].get("cert", {}).get("certified", True) for r in recs),
             "weights": sorted({r["diag"].get("weights", "-") for r in recs}), "B1_below_fbcsp": bool(kor.mean() < FBCSP),
             "n_subjects_or_below_fbcsp": int((kor < fb).sum())}
        if arm.endswith(("_trained",)) and any("repro_max_abs_diff" in r["diag"] for r in recs):
            o["seed0_repro_max_abs_diff"] = [r["diag"]["repro_max_abs_diff"] for r in recs if "repro_max_abs_diff" in r["diag"]]
        if not arm.endswith("_stored"):
            o["stats_cv_vs_fbcsp"] = stats(kcv, fb, subj_seed(recs, lambda r: r["kappa_cv"], seeds) - fb[:, None])
        if arm == "input_track2":
            o["nonlinear_or"] = float(subj_mean(recs, lambda r: r["nonlinear"]["kappa_or"]).mean())
            o["nonlinear_cv"] = float(subj_mean(recs, lambda r: r["nonlinear"]["kappa_cv"]).mean())
        out[arm] = o
    subs = [a for a in ("random_lake", "m6_trained", "m64_random", "m64_trained", "m9_reservoir", "m9_bptt") if a in out]
    linear = [a for a in LINEAR_CAND if a in out and out[a]["linear_r2_min"] is not None and out[a]["linear_r2_min"] > 0.99]
    v = {"all_B1": all(out[a]["B1_below_fbcsp"] for a in subs) if subs else None, "linear_regime": linear,
         "nonlinear": [a for a in subs if a not in linear]}
    if "input_track2" in out and linear:
        slack = max(out[a]["kappa_or"] - out["input_track2"]["kappa_or"] for a in linear)
        v["linear_bound"] = out["input_track2"]["kappa_or"] + slack  # M1-1, exactly as pre-registered
        v["linear_slack"] = slack; v["linear_extension_holds"] = bool(v["linear_bound"] < FBCSP)
    out["verdict"] = v
    return out


# ---------------- D1 ----------------
def step_d1(commit):
    valid = load(f"D1/*/valid/*.json"); small = load(f"D1/{commit}/small/*.json"); out = {"n_valid_cells": len(valid), "n_cells": len(small)}
    vbad = {(r["seed"], r["leak"], r["budget"], r["amp"], r["nudge"]) for r in valid if min(c["cos"] for c in r["ift_vs_fd"].values()) < 0.99}
    out["invalid_cells"] = sorted(vbad)
    cells = by(small, "leak", "budget", "amp", "nudge"); table = []
    for (leak, budget, amp, nudge), rs in sorted(cells.items()):
        if len(rs) < 3: problems.append(f"D1 cell {(leak, budget, amp, nudge)} has {len(rs)} seeds")
        row = {"leak": leak, "budget": budget, "amp": amp, "nudge": nudge, "sat": float(np.mean([r["sat_frac"] for r in rs]))}
        for g in rs[0]["grid"]:
            key = f"b{g['beta']}_t{g['tol']}"; vals = [next(x for x in r["grid"] if x["beta"] == g["beta"] and x["tol"] == g["tol"]) for r in rs]
            ok = [x for x in vals if x["refused"] is None]
            row[key] = {"refused": len(vals) - len(ok),
                        "cos_all_min": float(min(x["ep_vs_ift"]["all"]["cos"] for x in ok)) if ok else None,
                        "cos_deep_min": float(min(x["ep_vs_ift"]["deep"]["cos"] for x in ok)) if ok else None,
                        "rel_err_mean": float(np.mean([x["ep_vs_ift"]["all"]["rel_err"] for x in ok])) if ok else None}
        table.append(row)
    out["table"] = table
    prod = [r for r in table if r["leak"] == 1.0 and not any((s, r["leak"], r["budget"], r["amp"], r["nudge"]) in vbad for s in (0, 1, 2))]
    passed = [r for r in prod if r["b0.04_t0.003"]["refused"] == 0 and r["b0.04_t0.003"]["cos_all_min"] >= 0.9 and r["b0.04_t0.003"]["cos_deep_min"] >= 0.9]
    out["verdict"] = {"production_cells": len(prod), "passed": len(passed),
                      "failed_cells": [(r["budget"], r["amp"], r["nudge"]) for r in prod if r not in passed],
                      "D1": "成立" if prod and len(passed) == len(prod) else ("部分成立" if passed else "不成立")}
    asym = load(f"D1/{commit}/asym.json"); real = load(f"D1/{commit}/real_sandbox.json")
    if asym:
        a = by(asym[0]["cells"], "eps"); out["asym"] = {str(e[0]): {"cos_directed": float(np.mean([c["ep_vs_directed"]["all"]["cos"] for c in cs if c["refused"] is None])),
                                                                   "cos_pair_sym": float(np.mean([c["ep_vs_pair_symmetrised"]["all"]["cos"] for c in cs if c["refused"] is None]))}
                                                      for e, cs in sorted(a.items())}
    if real: out["real_sandbox"] = [{"beta": c["beta"], "tol": c["tol"], "refused": c["refused"], "cos": c.get("ep_vs_ift", {}).get("all", {}).get("cos")} for c in real[0]["cells"]]
    return out


# ---------------- D2 ----------------
def physical_cos(a, b):
    a, b = np.asarray(a), np.asarray(b); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def step_d2(commit):
    out = {}; recs = {}
    for st in ("recip", "lateral", "recip_lat"):
        for tr in ("ep", "ift", "delta"):
            rs = load(f"D2/{commit}/{st}_{tr}/*.json")
            if rs: complete(rs, f"D2/{st}_{tr}", seeds=(3, 4, 5)); recs[(st, tr)] = {(r["subject"], r["seed"]): r for r in rs}
    for st in ("recip", "lateral", "recip_lat"):
        if (st, "ep") not in recs: continue
        E = recs[(st, "ep")]; o = {}
        for tr in ("ep", "ift", "delta"):
            if (st, tr) not in recs: continue
            X = recs[(st, tr)]; k = np.array([[X[(s, sd)]["kappa_ep"] for sd in (3, 4, 5)] for s in SUBJ])
            o[tr] = {"kappa": float(k.mean()), "gap_to_ridge": float(np.mean([[X[(s, sd)]["gap"] for sd in (3, 4, 5)] for s in SUBJ])),
                     "accept_min": float(min(X[k_]["accept_rate"] for k_ in X)), "cert_fail": sum(X[k_]["cert_fail"] is not None for k_ in X),
                     "rejected": int(sum(X[k_]["rejected"] for k_ in X)), "settle": float(np.mean([X[k_]["settle_val"] for k_ in X]))}
            if tr != "ep":
                d = np.array([[E[(s, sd)]["kappa_ep"] - X[(s, sd)]["kappa_ep"] for sd in (3, 4, 5)] for s in SUBJ])
                o[f"ep_minus_{tr}"] = {"mean": float(d.mean()), "abs_mean": float(abs(d.mean())), "bayes": bayes_hier(d),
                                       "dtheta_cos": float(np.mean([physical_cos(E[k_]["dtheta"], X[k_]["dtheta"]) for k_ in E]))}
        if "ift" in o and "delta" in o:
            Ii = recs[(st, "ift")]; D = recs[(st, "delta")]
            dk = np.array([[D[(s, sd)]["kappa_ep"] - Ii[(s, sd)]["kappa_ep"] for sd in (3, 4, 5)] for s in SUBJ])
            o["delta_minus_ift"] = {"mean": float(dk.mean()), "dtheta_cos": float(np.mean([physical_cos(D[k_]["dtheta"], Ii[k_]["dtheta"]) for k_ in D]))}
            ep_dist = (abs(o["ep_minus_ift"]["mean"]), 1 - o["ep_minus_ift"]["dtheta_cos"]); de_dist = (abs(dk.mean()), 1 - o["delta_minus_ift"]["dtheta_cos"])
            o["discriminative"] = bool(de_dist[0] > ep_dist[0] or de_dist[1] > ep_dist[1])
            b = o["ep_minus_ift"]["bayes"]["ci95"]
            o["D2b"] = bool(o["ep_minus_ift"]["abs_mean"] <= 0.03 and -0.03 <= b[0] and b[1] <= 0.03)
        if st == "recip": o["D2a"] = bool(o["ep"]["gap_to_ridge"] <= 0.03)
        out[st] = o
    return out


# ---------------- E ----------------
def step_e(commit):
    out = {}
    for mix in ("narrow", "broad"):
        rs = load(f"E/{commit}/e1_{mix}/*.json")
        if not rs: continue
        complete(rs, f"E1/{mix}"); g = lambda k: float(subj_mean(rs, lambda r: r[k]).mean())
        out[f"e1_{mix}"] = {k: g(k) for k in ("truth_oracle", "fbcsp", "tslr", "mdm", "lda_track2", "input_track2_cv", "input_track2_or", "random_lake_cv", "random_lake_or")}
        out[f"e1_{mix}"]["e"] = rs[0]["e"]
    if "e1_narrow" in out and "e1_broad" in out:
        n, b = out["e1_narrow"], out["e1_broad"]
        out["E1_predictions"] = {"P1_narrow_track2_close_to_fbcsp": abs(n["input_track2_or"] - n["fbcsp"]) < 0.1,
                                 "P2_broad_track2_below_fbcsp_by_0.1": b["input_track2_or"] < b["fbcsp"] - 0.1,
                                 "P3_lake_equals_track2": all(abs(x["random_lake_or"] - x["input_track2_or"]) <= 0.03 for x in (n, b))}
    e2 = {}
    for tr in ("ep", "ift", "ep_asym"):
        rs = load(f"E/{commit}/e2_{tr}/*.json")
        if rs: complete(rs, f"E2/{tr}"); e2[tr] = {k: float(subj_mean(rs, lambda r: r[k]).mean()) for k in ("agreement", "kappa_vs_teacher", "planted_auc", "lateral_learned_mean")}
    if e2: out["e2"] = e2
    if {"ep", "ift", "ep_asym"} <= set(e2):
        out["E2_predictions"] = {"ep_agreement_ge_0.9_ift": e2["ep"]["agreement"] >= 0.9 * e2["ift"]["agreement"],
                                 "ep_auc_ge_ift_minus_0.05": e2["ep"]["planted_auc"] >= e2["ift"]["planted_auc"] - 0.05,
                                 "asym_fails_by_0.05": e2["ep_asym"]["agreement"] < e2["ep"]["agreement"] - 0.05}
    return out


# ---------------- C ----------------
def c_arm(arm):
    rs = load(f"C/runs/{arm}/*.json")
    for r in rs:
        bad = [k for k in ("R2_15", "D_gen") if not np.isfinite(r.get(k, np.nan))]
        if bad and not any(f.startswith(("diverged", "crashed")) for f in r.get("flags", [])): problems.append(f"C {arm} {r['subject']} {r['seed']}: unflagged non-finite {bad}")
    return {(r["subject"], r["seed"]): r for r in rs}


def wins(ep, comps, seeds=(0, 1, 2)):
    """Amendment M1-8: per subject, EP must beat each comparator on R2_15 and D_gen by max(seed SDs)."""
    res = []
    for s in SUBJ:
        e_r = np.array([ep[(s, k)]["R2_15"] for k in seeds]); e_d = np.array([ep[(s, k)]["D_gen"] for k in seeds]); ok = True
        for c in comps:
            c_r = np.array([c[(s, k)]["R2_15"] for k in seeds]); c_d = np.array([c[(s, k)]["D_gen"] for k in seeds])
            with np.errstate(invalid="ignore"):
                m_r = max(np.nan_to_num(e_r.std()), np.nan_to_num(c_r.std())); m_d = max(np.nan_to_num(e_d.std()), np.nan_to_num(c_d.std()))
                ok &= bool(e_r.mean() > c_r.mean() + m_r and e_d.mean() < c_d.mean() - m_d)
        res.append(ok)
    return res


def step_c():
    arms = ("var6", "var_best", "res", "lin_res", "ep_noproj", "ep_proj", "bptt_noproj_U", "bptt_proj_U", "bptt_noproj_2000", "bptt_proj_2000", "lin_bptt_2000")
    A = {a: c_arm(a) for a in arms}; A = {a: v for a, v in A.items() if v}
    for a, v in A.items():
        miss = [(s, k) for s in SUBJ for k in (0, 1, 2) if (s, k) not in v]
        if miss: problems.append(f"C/{a}: missing {miss[:4]}")
    out = {a: {"R2_15": float(np.mean([r["R2_15"] for r in v.values()])) if all(np.isfinite(r["R2_15"]) for r in v.values()) else "has -inf",
               "D_gen": float(np.median([r["D_gen"] for r in v.values()])), "flags": sum(bool(r.get("flags")) for r in v.values()),
               "accept_mean": float(np.mean([r.get("train", {}).get("accept_rate", 1.0) for r in v.values()]))} for a, v in A.items()}
    if {"var6", "res"} <= set(A):
        for name, group in (("EP", ("ep_noproj", "ep_proj")), ("substrate_BPTT2000", ("bptt_noproj_2000", "bptt_proj_2000"))):
            w = {a: wins(A[a], [A["var6"], A["res"]]) for a in group if a in A}
            out[f"verdict_{name}"] = {a: {"wins": int(sum(x)), "per_subject": x} for a, x in w.items()}
            out[f"verdict_{name}"]["holds"] = any(sum(x) >= 7 for x in w.values()) if w else None
        st = {}
        for a in ("ep_noproj", "ep_proj"):
            if a not in A: continue
            rs = list(A[a].values()); acc = [r.get("train", {}).get("accept_rate", 0) for r in rs]
            coll = [bool(r.get("train", {}).get("curve")) and (r["train"]["curve"][-1]["mon_r2"] < max(c["mon_r2"] for c in r["train"]["curve"]) - 0.05) for r in rs]
            st[a] = {"accept_ge_0.98": int(sum(x >= 0.98 for x in acc)), "collapsed": int(sum(coll)), "rho_max": float(max(r.get("train", {}).get("rho_max", 0) for r in rs)), "n": len(rs)}
        if {"ep_noproj", "ep_proj"} <= set(st):
            p, n = st["ep_proj"], st["ep_noproj"]
            st["projection_solves"] = bool(p["accept_ge_0.98"] >= 25 and p["n"] - p["collapsed"] >= 25 and (n["n"] - n["accept_ge_0.98"] + n["collapsed"]) >= 3)
            st["noproj_already_stable"] = bool((n["n"] - n["accept_ge_0.98"] + n["collapsed"]) < 3)
        out["stability"] = st
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--commit", required=True, help="queue commit of steps A, B, D, E")
    a = ap.parse_args(); res = {"A": step_a(a.commit), "B": step_b(a.commit), "D1": step_d1(a.commit), "D2": step_d2(a.commit),
                                "E": step_e(a.commit), "C": step_c(), "problems": problems}
    (ROOT / "reports" / "M11_verify.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    for k, v in res.items():
        if k != "problems": print(f"== {k}\n" + json.dumps(v.get("verdict", v) if isinstance(v, dict) else v, default=str)[:1500])
    print("problems:", problems or "none"); sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
