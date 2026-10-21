"""M12 verifier: recompute every S0 judgement from the receipts (reports/M12.md S0.3, S0.6 and amendment 1).

Cosines are recomputed from the raw gradient vectors, never copied. A missing seed, a missing receipt or a non-finite
value is a failure. U and the G0 table are computed here only. Writes receipts/M12/S0/verify_<commit>.json and
prints the G0 table as markdown.
"""
from __future__ import annotations
import argparse, itertools, json, math
from pathlib import Path
import numpy as np
from . import m12_common as C

W_S3, F, U_MIN, R_RUNS, N_PROT = 6 * 7 * 24 * 3600, 0.6, max(2700, 30 * 324), 27, 2
NEURONS, BUDGETS, AMPS, NUDGES, SEEDS = ("N1", "N2"), (0.3, 0.6, 0.9), (0.5, 4.0), ("ce", "quad"), (0, 1, 2)
PROD = (0.04, 3e-3)


def cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()): return float("nan")
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def load_dir(p):
    return {f.stem: json.loads(f.read_text()) for f in sorted(Path(p).glob("*.json"))} if Path(p).exists() else {}


def latest(base):
    dirs = [d for d in Path(base).iterdir() if d.is_dir()] if Path(base).exists() else []
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def cells(variant):
    if variant == "std": return [("std", nn, b, a, u) for nn, b, a, u in itertools.product(NEURONS, BUDGETS, AMPS, NUDGES)]
    if variant == "hetero": return [("hetero", "N1N2", b, a, u) for b, a, u in itertools.product(BUDGETS, AMPS, NUDGES)]
    return [("tied", "N2", b, a, u) for b, a, u in itertools.product(BUDGETS, AMPS, NUDGES)]


def judge_d1(d1, variant, backends=("cpu64", "torch64")):
    valid = load_dir(d1 / "valid"); out = {"cells": [], "invalid": 0, "missing": 0, "unsatisfied": 0}
    eps = {bk: load_dir(d1 / "ep" / bk) for bk in ("cpu64", "torch64", "torch32")}
    f32_min = math.inf
    for v, nn, b, a, u in cells(variant):
        row = {"cell": f"{v} {nn} b{b} amp{a} {u}", "valid": True, "min_cos": {}, "min_cos_deep": {}}
        for s in SEEDS:
            nm = f"{v}_{nn}_s{s}_b{b}_amp{a}_{u}"; r = valid.get(nm)
            if r is None: row["valid"] = None; out["missing"] += 1; continue
            cls = np.asarray(r["classes"]); ift = np.asarray(r["g_ift"]); fd = np.asarray(r["g_fd"]); it = np.asarray(r["g_ift_torch64"])
            ok = r["tight_converged"] and r["tight_converged_torch64"] and cos(ift, fd) >= 0.99999 and cos(it, ift) >= 0.99999
            for c in np.unique(cls):
                m = cls == c; ok = ok and cos(ift[m], fd[m]) >= 0.99999 and cos(it[m], ift[m]) >= 0.99999
            if not ok: row["valid"] = False
            for bk in ("cpu64", "torch64", "torch32"):
                e = eps[bk].get(nm)
                if e is None:
                    row["min_cos"][bk] = float("nan"); out["missing"] += bk in backends; continue
                g = next((g for g in e["grid"] if (g["beta"], g["tol"]) == PROD), None)
                if g is None or g.get("refused") or not g.get("finite", False):
                    c_all = c_deep = float("-inf")   # refusal or non-finite at the production setting: not satisfied (B3)
                else:
                    ep = np.asarray(g["g_ep"]); c_all = cos(ep, ift); c_deep = cos(ep[cls == "deep"], ift[cls == "deep"])
                    if variant == "tied" and bk in backends:
                        conv = g.get("conversion")
                        if not conv or conv["rel_err"] > 1e-12 or conv["within_group_ptp"] > 1e-12 or conv["group_sizes"] != [2, 3]:
                            c_all = float("-inf")
                row["min_cos"][bk] = min(row["min_cos"].get(bk, math.inf), c_all)
                row["min_cos_deep"][bk] = min(row["min_cos_deep"].get(bk, math.inf), c_deep)
        if row["valid"] is False: out["invalid"] += 1
        sat = all(row["min_cos"].get(bk, -1) >= 0.999 and row["min_cos_deep"].get(bk, -1) >= 0.999 for bk in backends)
        row["satisfied"] = bool(sat and row["valid"]); out["unsatisfied"] += (row["valid"] is True and not sat)
        f32_min = min(f32_min, row["min_cos"].get("torch32", -1), row["min_cos_deep"].get("torch32", -1))
        out["cells"].append(row)
    n = len(out["cells"])
    out["verdict"] = ("过" if all(r["satisfied"] for r in out["cells"]) and out["missing"] == 0 else
                      "不过" if out["unsatisfied"] > 0 else "检验力不足")
    out["n_configs"] = n; out["n_seed_cells"] = n * 3
    out["torch32_min_cos"] = f32_min; out["torch32_eligible"] = bool(f32_min >= 0.999 and out["missing"] == 0)
    return out


def judge_a2(a2dir):
    rs = list(load_dir(a2dir).values())
    cert = [r for r in rs if r["kind"] == "cert"]; multi = [r for r in rs if r["kind"] == "multi"]
    n_cert = sum(r["certificate"]["certified"] for r in cert)
    fp = sum(r["false_positive"] or (r["certificate"]["certified"] and not all(r["finite"])) for r in rs)
    rec_bad = sum(not r["recompute_ok"] for r in rs); bound_bad = sum(not r["visited_bound_ok"] for r in rs if all(r["finite"]))
    det = sum(r["detector_flagged"] for r in multi) / max(1, len(multi))
    mut_a = sum(r["mutant_alpha_flagged"] for r in rs); mut_b = sum(r["mutant_beta_flagged"] for r in rs)
    nonfinite = sum(not all(r["finite"]) for r in rs)
    groups = {g: sum(r["groups"] == g for r in cert) for g in (1, 2, 3, 4)}
    ok = (len(rs) == 1400 and n_cert >= 1000 and fp == 0 and rec_bad == 0 and bound_bad == 0 and mut_a >= 1 and mut_b >= 1)
    valid = len(multi) >= 200 and det >= 0.9
    return {"configs": len(rs), "certified": n_cert, "false_positives": fp, "recompute_mismatch": rec_bad, "visited_bound_violations": bound_bad,
            "multi_fp_controls": len(multi), "detector_rate": det, "mutant_alpha_flagged": mut_a, "mutant_beta_flagged": mut_b,
            "nonfinite": nonfinite, "near_one": sum(r["near_one"] for r in rs), "rate_violations": sum(r["rate_violation"] for r in cert),
            "max_start_diff_cert": max((r["start_diff"] for r in cert), default=None), "group_counts": groups,
            "verdict": "无效" if not valid else ("过" if ok else "不过")}


def judge_a3(a3dir, eligible):
    rs = load_dir(a3dir); out = {"configs": {}, "U": None}
    for bk in ("cpu64", "torch64", "torch32"):
        for th, jobs in ((8, 1), (2, 4)):
            names = [f"h512_{bk}_t{th}_j{jobs}_{i}" for i in range(jobs)]
            if not all(nm in rs for nm in names): out["configs"][f"{bk} {th}x{jobs}"] = {"missing": True}; continue
            rows = [x for nm in names for x in rs[nm]["rows"] if not x["warmup"]]
            t_u = float(np.median([x["t_u"] for x in rows])); t_free = float(np.median([x["t_free"] for x in rows]))
            if not all(np.isfinite([x["t_u"] for x in rows])) or len(rows) != 20 * jobs: out["configs"][f"{bk} {th}x{jobs}"] = {"missing": True}; continue
            bk_ift = bk if bk != "torch32" else "torch64"
            ift_names = [f"h512_{bk_ift}_t{th}_j{jobs}_{i}" for i in range(jobs)]
            ift_rows = [x for nm in ift_names if nm in rs for x in rs[nm]["ift_rows"] if not x["warmup"]]
            t_ift = float(np.median([x["t_ift"] for x in ift_rows])) if ift_rows else float("nan")
            c_ift = t_ift / t_u
            U = math.floor(F * W_S3 * jobs / (t_u * R_RUNS * N_PROT * (3 + c_ift + 2)))
            # evaluation cost (N8c): free phases on all train + val windows of 5 lakes x 27 runs, both protocols
            eval_s = t_free * (math.ceil(288 * 9 / 64) + math.ceil((8 * 288 + 96) * 9 / 64)) * 27 * 5 / jobs
            out["configs"][f"{bk} {th}x{jobs}"] = {"t_u": t_u, "t_free": t_free, "t_ift": t_ift, "c_ift": c_ift, "P": jobs, "U": U,
                                                   "eval_s": eval_s, "eval_frac_of_reserve": eval_s / ((1 - F) * W_S3),
                                                   "eligible": bk in eligible, "loadavg_max": max(x["loadavg1"] for x in rows)}
    el = {k: v for k, v in out["configs"].items() if v.get("eligible")}
    if el:
        best = max(el, key=lambda k: el[k]["U"]); out["U"] = el[best]["U"]; out["U_from"] = best
        out["cpu_verdict"] = "过" if el[best]["U"] >= U_MIN and el[best]["eval_frac_of_reserve"] <= 1.0 else "不过"
    else:
        out["cpu_verdict"] = "不过"
    out["U_min"] = U_MIN; out["gpu"] = "未测（本机无 GPU）"
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--commit", default=None); ap.add_argument("--stage", default="S0")
    ap.add_argument("--tag", default="official"); ap.add_argument("--eta", type=float, default=0.05); a = ap.parse_args()
    if a.stage == "S1":
        from .m12_s1_verify import main_s1, main_s1_fix
        return main_s1_fix(a.commit) if a.tag == "fix" else main_s1(a.commit, a.tag, a.eta)
    base = C.OUT / "S0"; key = a.commit
    pick = lambda sub: (base / sub / key) if key else latest(base / sub)
    unit = load_dir(pick("unit")).get("unit"); reg = load_dir(pick("regress/fork") if not key else base / "regress" / "fork" / key)
    d1 = pick("D1")
    t = {}
    t["unit"] = "过" if unit and unit["pass"] else ("不过" if unit else "缺失")
    t["regress"] = "过" if len(reg) == 36 and all(r["pass"] for r in reg.values()) else ("缺失" if len(reg) < 36 else "不过")
    j = {v: judge_d1(d1, v) for v in ("std", "hetero", "tied")} if d1 else {}
    t["hetero"] = j["hetero"]["verdict"] if j else "缺失"; t["tied"] = j["tied"]["verdict"] if j else "缺失"; t["A1"] = j["std"]["verdict"] if j else "缺失"
    a2 = {bk: judge_a2(pick("A2") / bk) for bk in ("cpu64", "torch64")} if pick("A2") else {}
    t["A2"] = ("过" if a2 and all(x["verdict"] == "过" for x in a2.values()) else
               "无效" if a2 and any(x["verdict"] == "无效" for x in a2.values()) else "不过" if a2 else "缺失")
    eligible = [bk for bk in ("cpu64", "torch64") if t["A1"] == "过" and t["A2"] == "过"]
    if j and j["std"]["torch32_eligible"] and t["A2"] == "过": eligible.append("torch32")
    a3 = judge_a3(pick("A3"), eligible) if pick("A3") else {"cpu_verdict": "缺失"}
    t["A3_cpu"] = a3["cpu_verdict"]; t["A3_gpu"] = "未测"
    passed = [t[k] == "过" for k in ("unit", "regress", "hetero", "tied", "A1", "A2", "A3_cpu")]
    t["G0"] = "不过" if any(t[k] == "不过" for k in ("unit", "regress", "hetero", "tied", "A1", "A2", "A3_cpu")) else ("待人" if all(passed) else "待人（判据未全出）")
    res = {"table": t, "d1": j, "a2": a2, "a3": a3}
    out = base / f"verify_{C.run_commit()}.json"; out.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    print("| 项 | 判定 |\n|---|---|")
    for k, v in t.items(): print(f"| {k} | {v} |")
    print(json.dumps({"U": a3.get("U"), "U_from": a3.get("U_from"), "A2": {k: {kk: vv for kk, vv in v.items() if kk != 'group_counts'} for k, v in a2.items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
