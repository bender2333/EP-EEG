"""M12 S1 entry checks (reports/M12.md S1.12, amendment 7 B5 / B4(c)). Dev seeds >= 100, (a, b) = (1, 0.6) on every
free neuron; no judged quantity (R2, kappa, m) is computed. Receipts dev=true under receipts/M12/S1/<commit>/entry/."""
from __future__ import annotations
import json, sys, time
import numpy as np
from . import m12_s1 as S
from . import m12_common as C
from .m12_s0_d1 import coords, to_phys, cosine

DEVS = (100, 101, 102)


def rel(a, b): return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def check_projection():
    out = {}
    for name, mk in (("T1_H32", lambda: S.build_t1(32, 100, hid_ab=S.DEV, out_ab=S.DEV)), ("T2", lambda: S.build_t2(100, ab=S.DEV, out_ab=S.DEV))):
        g = mk(); lr = g["lr"]; e0 = np.asarray(lr.brain.efficacy).copy(); b0 = np.asarray(lr.brain.bias).copy()
        changed_at_06 = S.project(g); same06 = np.array_equal(np.asarray(lr.brain.efficacy), e0)
        big = np.where(g["ff"], e0 * 1.3 / S.rho(g, e0), e0); S.set_params(lr, big, b0)
        r_big = S.rho(g); S.project(g); e1 = np.asarray(lr.brain.efficacy)
        cert = S.certificate(g); rev = lr.reverse; p = rev >= 0
        inp = ~g["ff"]
        tie_diff = 0.0
        if g["groups"] is not None:
            for k in np.unique(g["groups"][g["groups"] >= 0]):
                v = e1[g["groups"] == k]; tie_diff = max(tie_diff, float(v.max() - v.min()))
        res = {"rho_before": r_big, "rho_after": S.rho(g), "certified": bool(cert["certified"]),
               "cadence_weighted_row_mass_free": cert.get("weighted_row_mass_free"), "inputs_unchanged": bool(np.array_equal(e1[inp], big[inp])),
               "tie_max_diff": tie_diff, "pair_max_diff": float(np.max(np.abs(e1[p] - e1[rev[p]]))) if p.any() else 0.0,
               "unchanged_at_0.6": bool(same06 and not changed_at_06)}
        res["pass"] = bool(res["certified"] and res["rho_after"] <= 0.9 + 1e-12 and res["inputs_unchanged"] and res["tie_max_diff"] <= 1e-12
                           and res["pair_max_diff"] <= 1e-12 and res["unchanged_at_0.6"]
                           and abs(res["cadence_weighted_row_mass_free"] - res["rho_after"]) <= 1e-12)
        out[name] = res
    out["pass"] = all(v["pass"] for v in out.values())
    return out


def fd_grad(g, D, t, params, ns, h=1e-4):
    lr = g["lr"]; e0 = np.asarray(lr.brain.efficacy).copy(); b0 = np.asarray(lr.brain.bias).copy(); o = g["out"][0]
    def loss(e, b):
        S.set_params(lr, e, b); eq = lr.brain.equilibrate(D, budget=S.BUDGET, tolerance=1e-11, damping=3)
        if not np.all(eq.converged): raise ArithmeticError("FD tight phase did not converge")
        s = np.asarray(eq.state.activation)[:, o]; return 0.5 * ((s - t) ** 2).mean()
    rev = lr.reverse; gr = []
    for pidx in params:
        idx = list(pidx) + [int(rev[i]) for i in pidx if rev[i] >= 0]
        e = e0.copy(); e[idx] += h; lp = loss(e, b0); e = e0.copy(); e[idx] -= h; lm = loss(e, b0); gr.append((lp - lm) / (2 * h))
    for j in range(ns, g["n"]):
        b = b0.copy(); b[j] += h; lp = loss(e0, b); b = b0.copy(); b[j] -= h; lm = loss(e0, b); gr.append((lp - lm) / (2 * h))
    S.set_params(lr, e0, b0)
    return np.array(gr)


def check_t1_continuous():
    out = {}
    for H in (8, 32):
        for seed in DEVS:
            d = S.t1_data(seed); idx = np.arange(64); x = d["x_tr"][idx]; t = d["t_tr"][idx]
            gc = S.build_t1(H, seed, hid_ab=S.DEV, out_ab=S.DEV, backend="cpu64")
            kind = np.where(gc["ff"], "lateral", "deep"); groups = -np.ones(gc["conn"].synapses, np.int64)
            params, pcls = coords(gc["lr"], kind, groups, gc["ns"], gc["n"])
            D = S.drive(gc, x)
            ce, cb, loss, eq = S.ift_grad(gc, D, S.gs_quad(gc, t), tol=1e-11)
            ift = to_phys(ce, cb, params, gc["ns"])
            fd = -fd_grad(gc, D, t, params, gc["ns"])
            gt = S.build_t1(H, seed, hid_ab=S.DEV, out_ab=S.DEV, backend="torch64"); lr = gt["lr"]
            free = lr.free(D); T = S.target_array(gt, t)
            pl = lr.nudged(D, free, T, 1.0); mi = lr.nudged(D, free, T, -1.0); e_, b_ = lr.contrast(free, pl, mi)
            ep = to_phys(np.asarray(e_), np.asarray(b_), params, gt["ns"])
            out[f"H{H}_s{seed}"] = {"cos_ift_fd": cosine(ift, fd), "cos_ep_ift": cosine(ep, ift), "scale_ep_ift": float(np.linalg.norm(ep) / np.linalg.norm(ift)),
                                    "n_params": len(ift)}
    v = list(out.values())
    out["pass"] = bool(min(r["cos_ift_fd"] for r in v) >= 0.99999 and min(r["cos_ep_ift"] for r in v) >= 0.999)
    return out


def check_single_step():
    out = {}
    for name in ("T1_H8", "T2"):
        mk = (lambda: S.build_t1(8, 100, hid_ab=S.DEV, out_ab=S.DEV)) if name == "T1_H8" else (lambda: S.build_t2(100, ab=S.DEV, out_ab=S.DEV))
        if name == "T1_H8":
            d = S.t1_data(100); x = d["x_tr"][:64]; y = np.zeros(64, np.int64)
        else:
            d = S.t2_data(100); x = d["in_tr"][:64]; y = d["y_tr"][:64]
        ga, gb = mk(), mk(); D = S.drive(ga, x)
        e0 = np.asarray(ga["lr"].brain.efficacy).copy(); b0 = np.asarray(ga["lr"].brain.bias).copy()
        ga["lr"].step(D, y)
        S.ep_step(gb, D, gb["lr"].targets(y))
        da = np.r_[np.asarray(ga["lr"].brain.efficacy) - e0, np.asarray(ga["lr"].brain.bias) - b0]
        db = np.r_[np.asarray(gb["lr"].brain.efficacy) - e0, np.asarray(gb["lr"].brain.bias) - b0]
        out[name] = {"rel_err": rel(db, da), "max_abs_delta": float(np.abs(da).max()), "pass": bool(rel(db, da) <= 1e-12 and np.abs(da).max() > 0)}
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def check_generators():
    out = {}
    for seed in (0, 1, 2) + DEVS:
        a, b = S.t2_data(seed), S.t2_data(seed); t1a, t1b = S.t1_data(seed), S.t1_data(seed)
        r = {"cond": a["cond"], "cos_ab_true": a["cos_ab_true"], "draws": a["draws"],
             "balanced": bool(all((a["y_" + s] == 0).sum() * 2 == len(a["y_" + s]) for s in ("tr", "dv", "te"))),
             "t2_sha_reproducible": a["sha"] == b["sha"], "t1_sha_reproducible": t1a["sha"] == t1b["sha"], "t2_sha": a["sha"], "t1_sha": t1a["sha"]}
        r["pass"] = bool(r["cond"] <= 10 and r["cos_ab_true"] <= 0.5 and r["balanced"] and r["t2_sha_reproducible"] and r["t1_sha_reproducible"])
        out[f"s{seed}"] = r
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def check_t3_scale():
    d = S.t2_data(101); x = d["in_tr"][:64]; y = d["y_tr"][:64]
    ge, gi = S.build_t2(101, ab=S.DEV, out_ab=S.DEV), S.build_t2(101, ab=S.DEV, out_ab=S.DEV); D = S.drive(ge, x)
    e0 = np.asarray(ge["lr"].brain.efficacy).copy(); b0 = np.asarray(ge["lr"].brain.bias).copy()
    S.ep_step(ge, D, ge["lr"].targets(y)); ok, eq = S.ift_step(gi, D, S.gs_ce(gi, y))
    de = np.r_[np.asarray(ge["lr"].brain.efficacy) - e0, np.asarray(ge["lr"].brain.bias) - b0]
    di = np.r_[np.asarray(gi["lr"].brain.efficacy) - e0, np.asarray(gi["lr"].brain.bias) - b0]
    ns_syn = len(e0); tie = np.r_[ge["groups"] >= 0, np.zeros(len(b0), bool)]
    res = {"ift_converged": bool(ok)}
    for nm, m in (("all", np.ones(len(de), bool)), ("entry_tied", tie)):
        res[nm] = {"ratio": float(np.linalg.norm(de[m]) / np.linalg.norm(di[m])), "cos": cosine(de[m], di[m])}
    res["pass"] = bool(ok and all(0.98 <= res[k]["ratio"] <= 1.02 and res[k]["cos"] >= 0.999 for k in ("all", "entry_tied")))
    return res


def main():
    C.e_seal(); import torch; torch.set_num_threads(S.THREADS)
    S.start_clock("entry checks")
    out_dir = S.S1 / C.run_commit() / "entry"
    which = sys.argv[1:] or ["projection", "single_step", "generators", "t3_scale", "t1_continuous"]
    fns = {"projection": check_projection, "single_step": check_single_step, "generators": check_generators,
           "t3_scale": check_t3_scale, "t1_continuous": check_t1_continuous}
    for w in which:
        t0 = time.time(); r = fns[w]()
        C.write_receipt(out_dir, w, {"check": w, "result": r, "neuron": S.DEV, "wall_s": time.time() - t0}, dev=True)
        C.ledger("M12-S1", f"entry_{w}", "NA", 1, time.time() - t0, "pass" if r["pass"] else "fail", f"S1 entry check {w} (dev, (1,0.6))")
        print(w, "PASS" if r["pass"] else "FAIL", json.dumps(r, default=float)[:1500], flush=True)


if __name__ == "__main__":
    main()
