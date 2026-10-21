"""M12 S1 verifier (reports/M12.md S1.7, S1.9, amendment 7 B2/B3/B4/N6): recompute every S1 verdict from receipts,
regenerating the data from the seeds; tamper self-test (i)-(x). Usage: python -m ep_eeg.m12_verify --stage S1 --commit <c> [--tag official]."""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np
from . import m12_common as C
from . import m12_s1 as S

ARMS_T1 = ("ep", "random", "randodd"); ARMS_T2 = ("ep", "random", "shuffled", "ift"); SEEDS = (0, 1, 2)
FORBIDDEN = ("calib", "affine", "intercept", "output_scale_fit")


def load(dirpath):
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(dirpath).glob("*.json"))}


def _r2(yh, y): y = np.asarray(y, float); yh = np.asarray(yh, float); return float(1 - ((yh - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _kappa(y, p):
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(np.asarray(y), np.asarray(p)))


def r2_true(seed):
    from sklearn.linear_model import RidgeCV
    d = S.t1_data(seed); f = lambda x: np.c_[x, x ** 2, x[:, :1] * x[:, 1:]]
    return _r2(RidgeCV(alphas=S.ALPHAS, cv=5).fit(f(d["x_tr"]), d["y_tr"]).predict(f(d["x_te"])), d["y_te"])


def verify(recs, minit=None, eta=S.ETA, U=S.U_FULL, data_cache=None):
    issues, failed = [], {}
    dc = data_cache if data_cache is not None else {}
    def data(task, seed):
        k = (task, seed)
        if k not in dc: dc[k] = S.t1_data(seed) if task == "T1" else S.t2_data(seed)
        return dc[k]
    get = {}
    for name, r in recs.items():
        if r.get("dev"): issues.append(f"{name}: dev=true receipt in the verdict input")
        if r.get("e_seal", {}).get("matches"): issues.append(f"{name}: E seal matches")
        for k in r:
            if any(f in k.lower() for f in FORBIDDEN): issues.append(f"{name}: output calibration field {k}")
        task, arm, seed, H = r["task"], r["arm"], r["seed"], r.get("H")
        get[(task, arm, H, seed)] = r
        trained = arm in ("ep", "shuffled", "ift")
        cfg = r["config"]
        if trained and (abs(cfg["eta"] - eta) > 1e-15 or abs(cfg["eta_bias"] - eta / 10) > 1e-15): issues.append(f"{name}: eta {cfg['eta']}")
        if r["updates_attempted"] != (U if trained else 0): issues.append(f"{name}: updates attempted {r['updates_attempted']}")
        acc = np.asarray(r["accept_by_batch"])
        if len(acc) != r["updates_attempted"] or int(acc.sum()) != r["updates_accepted"]: issues.append(f"{name}: per-batch acceptance inconsistent")
        if any(l["rho_after"] > S.CAP + 1e-12 for l in r["logs"]) or r["rho_final"] > S.CAP + 1e-12: issues.append(f"{name}: rho~ above cap")
        if r["cert_false_count"] != 0 or not r["cert_final"]["certified"]: issues.append(f"{name}: certificate false")
        d = data(task, seed)
        if r["data_sha256"] != d["sha"]: issues.append(f"{name}: data SHA differs from regeneration")
        if not np.array_equal(np.asarray(r["y_te"], float), np.asarray(d["y_te"], float)): issues.append(f"{name}: test labels differ from regeneration")
        fail = (trained and (r["accept_rate"] is None or r["accept_rate"] < 0.95)) or r["eval_unqualified"] > 0.01 * r["eval_rows"]
        if task == "T1":
            if r.get("yhat_const") != S.K_T1: issues.append(f"{name}: yhat constant {r.get('yhat_const')}")
            if C.sha(d["x_te"]) != r["x_te_sha"]: issues.append(f"{name}: x_te SHA")
            for key, val in (("r2_out", _r2(S.K_T1 * np.asarray(r["s_out_te"]), d["y_te"])), ("r2_ridge", _r2(r["pred_ridge_te"], d["y_te"]))):
                if not np.isfinite(r[key]) or abs(val - r[key]) > 1e-9: issues.append(f"{name}: {key} recomputed {val} vs {r[key]}")
                if not np.isfinite(val): fail = True
        else:
            k = _kappa(d["y_te"], r["pred_ridge_te"])
            if abs(k - r["kappa_ridge"]) > 1e-9: issues.append(f"{name}: kappa recomputed {k} vs {r['kappa_ridge']}")
            ko = _kappa(d["y_te"], r["pred_oracle_te"])
            if abs(ko - r["kappa_oracle"]) > 1e-9: issues.append(f"{name}: kappa_oracle recomputed")
            m = S.match(np.asarray(r["W_filters"]), d["uA"], d["uB"])[0]
            if abs(m - r["m"]) > 1e-9: issues.append(f"{name}: m recomputed {m} vs {r['m']}")
            mc = S.match(np.asarray(r["csp_filters"]), d["uA"], d["uB"])[0]
            if abs(mc - r["m_csp"]) > 1e-9: issues.append(f"{name}: m_csp recomputed")
            if minit is not None:
                exp = minit["T2"][str(seed)]["init_seed"]
                if r["init_seed"] != exp: issues.append(f"{name}: init seed {r['init_seed']} != m_init record {exp}")
        failed[(task, arm, H, seed)] = bool(fail)
    # presence and shared initial state
    for H in (8, 32):
        for s in SEEDS:
            for a in ARMS_T1:
                if ("T1", a, H, s) not in get: issues.append(f"missing T1 {a} H{H} s{s}")
            if ("T1", "ep", H, s) in get and ("T1", "random", H, s) in get and get[("T1", "ep", H, s)]["init_sha256"] != get[("T1", "random", H, s)]["init_sha256"]:
                issues.append(f"T1 H{H} s{s}: random init SHA != EP init SHA")
    for s in SEEDS:
        for a in ARMS_T2:
            if ("T2", a, None, s) not in get: issues.append(f"missing T2 {a} s{s}")
        shas = {get[("T2", a, None, s)]["init_sha256"] for a in ARMS_T2 if ("T2", a, None, s) in get}
        if len(shas) > 1: issues.append(f"T2 s{s}: arms do not share the initial state")
    # verdicts (amendment 7 B2: least favourable value for a failed or missing seed)
    def val(task, arm, H, s, key, worst):
        r = get.get((task, arm, H, s))
        if r is None or failed.get((task, arm, H, s)): return worst
        v = r[key]; return v if np.isfinite(v) else worst
    t = {"T1": {}, "T2": {}, "T3": {}}
    for H in (8, 32):
        i_ = min(val("T1", "ep", H, s, "r2_ridge", -np.inf) for s in SEEDS)
        ii = min(val("T1", "ep", H, s, "r2_out", -np.inf) for s in SEEDS)
        iii = max(val("T1", "random", H, s, "r2_out", np.inf) for s in SEEDS)
        desc = {f"s{s}": {a: {k: get[("T1", a, H, s)][k] for k in ("r2_ridge", "r2_out", "r2_t", "corr2_out_y")} for a in ARMS_T1 if ("T1", a, H, s) in get} for s in SEEDS}
        t["T1"][f"H{H}"] = {"i_min_r2_ridge_ep": i_, "ii_min_r2_out_ep": ii, "iii_max_r2_out_random": iii,
                            "i": i_ >= 0.9, "ii": ii >= 0.9, "iii": iii <= 0.3, "pass": bool(i_ >= 0.9 and ii >= 0.9 and iii <= 0.3), "per_seed": desc}
    t["T1"]["r2_true"] = {f"s{s}": r2_true(s) for s in SEEDS}
    t1_valid = all(v >= 0.999 for v in t["T1"]["r2_true"].values())
    t["T1"]["valid"] = t1_valid
    t["T1"]["verdict"] = "无效" if not t1_valid else ("过" if all(t["T1"][f"H{H}"]["pass"] for H in (8, 32)) else "不过")
    per = {}
    for s in SEEDS:
        ko = get[("T2", "ep", None, s)]["kappa_oracle"] if ("T2", "ep", None, s) in get else np.nan
        k = val("T2", "ep", None, s, "kappa_ridge", -1.0); m = val("T2", "ep", None, s, "m", 0.0)
        per[f"s{s}"] = {"kappa_ep": k, "kappa_oracle": ko, "ratio": k / ko if ko and np.isfinite(ko) else -np.inf, "m_ep": m,
                        "m_random": val("T2", "random", None, s, "m", 1.0), "m_shuffled": val("T2", "shuffled", None, s, "m", 1.0),
                        "m_csp": get[("T2", "ep", None, s)]["m_csp"] if ("T2", "ep", None, s) in get else np.nan,
                        "kappa_random": val("T2", "random", None, s, "kappa_ridge", -1.0), "kappa_shuffled": val("T2", "shuffled", None, s, "kappa_ridge", -1.0),
                        "kappa_ift": val("T2", "ift", None, s, "kappa_ridge", -1.0), "m_ift": val("T2", "ift", None, s, "m", 0.0)}
    a_ok = all(p["ratio"] >= 0.9 for p in per.values()); m_min = min(p["m_ep"] for p in per.values())
    valid = (all(p["kappa_oracle"] >= 0.8 for p in per.values()) and all(p["m_csp"] >= 0.95 for p in per.values())
             and max(p["m_shuffled"] for p in per.values()) < 0.8)
    t["T2"] = {"per_seed": per, "kappa_ratio_min": min(p["ratio"] for p in per.values()), "m_min": m_min, "a": a_ok, "b": m_min >= 0.8,
               "valid": bool(valid), "verdict": "无效" if not valid else ("过" if a_ok and m_min >= 0.8 else "不过")}
    any_fail = any(failed.get(("T2", a, None, s), ("T2", a, None, s) not in get) for a in ("ep", "ift") for s in SEEDS)
    ke = np.mean([per[f"s{s}"]["kappa_ep"] for s in SEEDS]); ki = np.mean([per[f"s{s}"]["kappa_ift"] for s in SEEDS])
    t["T3"] = {"mean_kappa_ep": ke, "mean_kappa_ift": ki, "abs_diff": abs(ke - ki), "any_failed_seed": any_fail,
               "applied_updates": {f"s{s}": {a: get[("T2", a, None, s)]["updates_accepted"] for a in ("ep", "ift") if ("T2", a, None, s) in get} for s in SEEDS},
               "verdict": "不过" if any_fail or abs(ke - ki) > 0.03 else "过"}
    t["failed_seeds"] = [f"{k[0]}_{k[1]}_H{k[2]}_s{k[3]}" for k, v in failed.items() if v]
    t["G1"] = ("不一致" if issues else "无效" if "无效" in (t["T1"]["verdict"], t["T2"]["verdict"]) else
               "过" if t["T1"]["verdict"] == "过" and t["T2"]["verdict"] == "过" else "不过")
    return t, issues


def tamper_selftest(recs, minit, eta=S.ETA, U=S.U_FULL):
    dc = {}; base, base_issues = verify(recs, minit, eta, U, dc); out = {}
    def first(task, arm, H=None):
        return next(k for k, r in recs.items() if r["task"] == task and r["arm"] == arm and r.get("H") == H and r["seed"] == 0)
    def run(label, mut, check=lambda t, iss: len(iss) > len(base_issues)):
        R = copy.deepcopy(recs); mut(R); t, iss = verify(R, minit, eta, U, dc); out[label] = bool(check(t, iss))
    k1 = first("T1", "ep", 8); k2 = first("T2", "ep"); kr = first("T1", "random", 8)
    run("i_labels_permuted", lambda R: R[k2].__setitem__("y_te", list(np.random.default_rng(0).permutation(R[k2]["y_te"]))))
    run("ii_filters_random", lambda R: R[k2].__setitem__("W_filters", np.random.default_rng(1).normal(size=(4, 6)).tolist()))
    run("iii_calibrated_output", lambda R: R[k1].__setitem__("s_out_te", (S.C_T1 * np.asarray(R[k1]["y_te"])).tolist()))
    run("iv_dev_receipt", lambda R: R.__setitem__("devcopy", {**copy.deepcopy(R[k1]), "dev": True}))
    run("v_seed_deleted", lambda R: R.pop(k2))
    run("vi_eta", lambda R: R[k1]["config"].__setitem__("eta", 0.1))
    run("vi_U", lambda R: R[k1].__setitem__("updates_attempted", S.U_FULL - 1))
    run("vii_calibration_field", lambda R: R[k1].__setitem__("output_calibration", [1.0, 0.0]))
    run("viii_random_init_sha", lambda R: R[kr].__setitem__("init_sha256", "0" * 64))
    def rho_bad(R):
        if R[k1]["logs"]: R[k1]["logs"][0]["rho_after"] = 0.95
        else: R[k1]["rho_final"] = 0.95
    run("ix_rho_095", rho_bad)
    run("x_random_seed_failed", lambda R: R[kr].__setitem__("eval_unqualified", R[kr]["eval_rows"]),
        check=lambda t, iss: t["T1"]["H8"]["iii_max_r2_out_random"] == np.inf and t["T1"]["H8"]["iii"] is False)
    out["pass"] = all(out.values())
    return out


def main_s1(commit, tag="official", eta=S.ETA, U=S.U_FULL):
    base = S.S1 / commit; recs = load(base / tag)
    mp = S.S1 / "minit.json"; minit = json.loads(mp.read_text()) if mp.exists() else None
    t, issues = verify(recs, minit, eta, U)
    st = tamper_selftest(recs, minit, eta, U) if recs else {"pass": False}
    if not st["pass"]: issues.append("verifier tamper self-test failed")
    entry = {p.stem: json.loads(p.read_text())["result"]["pass"] for p in sorted((base / "entry").glob("*.json"))} if (base / "entry").exists() else {}
    res = {"table": t, "issues": issues, "tamper_selftest": st, "entry": entry, "tag": tag, "eta": eta, "U": U}
    out = S.S1 / f"verify_{commit}_{tag}.json"
    out.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else (None if o is None else float(o))))
    print(json.dumps({"G1": t["G1"], "T1": t["T1"]["verdict"], "T2": t["T2"]["verdict"], "T3": t["T3"]["verdict"], "issues": issues[:20],
                      "tamper": st, "entry": entry}, indent=1, default=float))
    return res
