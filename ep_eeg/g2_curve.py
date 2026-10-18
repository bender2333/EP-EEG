"""G2 (reports/M10-gates.md, pre-registered): learning curve on the linear lake.

N in {48, 96, 144, 192} training trials, class-stratified from T runs 0-3 (subsample seed r; the lake
init seed is also r, so the random lake is the trained lake's starting point); validation fixed (runs 4-5).
Trained lake: M6 config, full EP training, updates fixed at ~6600 (epochs = 100 * 192 / N).
Both lakes read out by the same closed-form ridge (trial mean of sub-window column states, standardised).
Delta(N) = kappa(trained + ridge) - kappa(random + ridge).
"""
from __future__ import annotations
import argparse, json, subprocess, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import cohen_kappa_score
from eval.scoring import aggregate, fallback_class
from .data import load_subject, ROOT
from .sandbox import Sandbox, SandboxConfig, train_sandbox

M6 = dict(leak=1.0, rowmass_project=0.9, eta=0.5, temperature=0.01)
NS = (48, 96, 144, 192)


def subsample(y, N, r):
    rng = np.random.default_rng(r)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), N // 4, replace=False) for c in range(4)]))


def col_mean(sb, X):
    n, k, _ = X.shape; eq = sb.free_states(X); a = np.asarray(eq.state.activation)
    ok = (np.asarray(eq.converged) & np.isfinite(a).all(1)).reshape(n, k)
    S = a[:, sb.col0:sb.out0].reshape(n, k, -1); w = ok[..., None].astype(float)
    return (S * w).sum(1) / np.maximum(w.sum(1), 1), float(ok.mean())


def ridge_kappa(sb, tr, va):
    F, s1 = col_mean(sb, tr["track2_x"]); Fv, s2 = col_mean(sb, va["track2_x"]); mu, sd = F.mean(0), F.std(0) + 1e-9
    r = RidgeClassifierCV(alphas=np.logspace(-3, 3, 13)).fit((F - mu) / sd, tr["y"])
    return float(cohen_kappa_score(va["y"], r.predict((Fv - mu) / sd))), min(s1, s2)


def one(subject, N, r):
    t0 = time.time(); d = load_subject(subject); idx = subsample(d["train"]["y"], N, r)
    tr = {"track2_x": d["train"]["track2_x"][idx], "y": d["train"]["y"][idx]}; va = d["val"]
    epochs = int(round(100 * 192 / N))
    k_rand, s_rand = ridge_kappa(Sandbox(SandboxConfig(**M6, epochs=epochs), r), tr, va)
    sb, st, _ = train_sandbox(tr, r, **M6, epochs=epochs)
    failed = st.get("failed")
    k_tr, s_tr = ridge_kappa(sb, tr, va) if not failed else (float("nan"), 0.0)
    act, ok = sb.predict(va["track2_x"]); k_ep = float(cohen_kappa_score(va["y"], aggregate(act, ok, fallback_class(tr["y"]))[0]))
    return {"subject": subject, "N": N, "r": r, "epochs": epochs, "updates": st["attempted_batches"],
            "accept_rate": st["accepted_rows"] / max(1, st["attempted_rows"]), "failed": failed,
            "cert_all": all(e["certified"] for e in st["epochs"]), "n_projections": getattr(sb, "n_projections", 0),
            "kappa_random_ridge": k_rand, "kappa_trained_ridge": k_tr, "delta": k_tr - k_rand, "kappa_trained_ep": k_ep,
            "settle_val": min(s_rand, s_tr), "wall_s": time.time() - t0}


def _job(a): return one(*a)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4); a = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    out = ROOT / "receipts" / "G2" / commit; out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, N, r) for N in NS[::-1] for r in (0, 1, 2) for s in range(1, 10)]
    res = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for x in ex.map(_job, jobs):
            res.append(x); print(f"A{x['subject']:02d} N={x['N']} r={x['r']} rand {x['kappa_random_ridge']:.3f} "
                                 f"trained {x['kappa_trained_ridge']:.3f} {x['wall_s']:.0f}s", flush=True)
            (out / "runs.json").write_text(json.dumps(res, indent=1, default=float))
    D = {N: np.array([[np.mean([x["delta"] for x in res if x["subject"] == s and x["N"] == N]) for s in range(1, 10)]][0]) for N in NS}
    for N in NS:
        kr = np.mean([x["kappa_random_ridge"] for x in res if x["N"] == N]); kt = np.mean([x["kappa_trained_ridge"] for x in res if x["N"] == N])
        print(f"N={N:3d} random+ridge {kr:.3f} trained+ridge {kt:.3f} Delta {D[N].mean():+.3f}")
    slopes = [np.polyfit(NS, [D[N][i] for N in NS], 1)[0] for i in range(9)]
    widen = D[192].mean() - D[48].mean(); pos = sum(s > 0 for s in slopes)
    print(f"Delta(192)-Delta(48) = {widen:+.3f}; positive slope in {pos}/9 subjects")
    print("verdict:", "SAMPLE-LIMITED (G2 fails)" if (widen >= 0.03 and pos >= 7) else "not sample-limited (G2 passes)")


if __name__ == "__main__":
    main()
