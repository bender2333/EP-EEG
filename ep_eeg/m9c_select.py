"""M9c final experiment, operating-point selection (closed form, random lake only; rule pre-registered in M9.md).

For sigma_in x m (m in 1, 4, 8, 16), leak 1.0, M6 config, cert_free_only (ADR-0012, proposed), seed 0:
R^2 of a linear fit X -> column activations, share |a| > 0.95 (columns, motor), random lake + ridge kappa,
refined certificate, free-phase convergence. Rule: smallest m with R^2 < 0.9 and saturation < 5%.
"""
from __future__ import annotations
import json, subprocess
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import cohen_kappa_score
from .data import load_subject, ROOT
from .sandbox import Sandbox, SandboxConfig
from .m9c_linearity import r2

MULTS = (1, 4, 8, 16)
BASE = dict(leak=1.0, rowmass_project=0.9, eta=0.5, temperature=0.01, cert_free_only=True)


def one(s):
    d = load_subject(s); out = {}
    for m in MULTS:
        sb = Sandbox(SandboxConfig(sigma_in=0.1 * m, **BASE), 0)
        def st(X):
            n, k, f = X.shape; eq = sb.free_states(X); a = np.asarray(eq.state.activation)
            return a[:, sb.col0:sb.out0].reshape(n, k, -1), a, np.asarray(eq.converged)
        Str, atr, c1 = st(d["train"]["track2_x"]); Sva, ava, c2 = st(d["val"]["track2_x"])
        F, Fv = Str.mean(1), Sva.mean(1); mu, sd = F.mean(0), F.std(0) + 1e-9
        kap = cohen_kappa_score(d["val"]["y"], RidgeClassifierCV(alphas=np.logspace(-3, 3, 13))
                                .fit((F - mu) / sd, d["train"]["y"]).predict((Fv - mu) / sd))
        A = np.concatenate([atr, ava]); col = A[:, sb.col0:sb.out0]; mot = A[:, sb.outputs]
        X = np.concatenate([d["train"]["track2_x"], d["val"]["track2_x"]]).reshape(-1, sb.d)
        c = sb.cert()
        out[m] = {"ridge_kappa": float(kap), "r2_col_X": r2(col, X), "r2_col_inputact": r2(col, A[:, :sb.d]),
                  "sat_col": float((np.abs(col) > 0.95).mean()), "sat_motor": float((np.abs(mot) > 0.95).mean()),
                  "col_v_std": float(np.asarray(sb.free_states(d["val"]["track2_x"]).state.v)[:, sb.col0:sb.out0].std()),
                  "certified_refined": c["certified"], "rho_free": c["row_mass"], "certified_full": c.get("certified_full"),
                  "converged": float(min(c1.mean(), c2.mean()))}
    return s, out


def main():
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    with ProcessPoolExecutor(4) as ex: res = dict(ex.map(one, range(1, 10)))
    out = ROOT / "receipts" / "M9c" / "select" / commit; out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_text(json.dumps(res, indent=1))
    print(f"{'m':>3s} {'ridge k':>8s} {'R2(col|X)':>10s} {'R2(col|in)':>10s} {'sat col':>8s} {'sat mot':>8s} {'v std':>6s} cert conv")
    chosen = None
    for m in MULTS:
        v = {k: np.mean([res[s][m][k] for s in res]) for k in ("ridge_kappa", "r2_col_X", "r2_col_inputact", "sat_col", "sat_motor", "col_v_std", "converged")}
        cert = all(res[s][m]["certified_refined"] for s in res)
        print(f"{m:3d} {v['ridge_kappa']:8.3f} {v['r2_col_X']:10.4f} {v['r2_col_inputact']:10.4f} {v['sat_col']:8.4f} {v['sat_motor']:8.4f} {v['col_v_std']:6.3f} {cert} {v['converged']:.3f}"
              f"   per-subj k {[round(res[s][m]['ridge_kappa'], 2) for s in res]}")
        if chosen is None and m > 1 and cert and v["r2_col_X"] < 0.9 and max(v["sat_col"], v["sat_motor"]) < 0.05: chosen = m
    print("selected m:", chosen); print(out / "runs.json")


if __name__ == "__main__":
    main()
