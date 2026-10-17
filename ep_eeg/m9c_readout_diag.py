"""M9c readout diagnostics and topology reconciliation (Track-2, M6 config, seed 0, weights saved).

Per subject, two EP nets are trained: "full" (M6) and "readout" (strictly frozen lake, EP trains col->motor).
  1. EP readout as a linear map: score = rho_bar(col) @ W[col->motor] + b_motor on the SAME trial-averaged
     column activations ridge uses, vs ADR-0008 (mean of motor activations) and vs ridge on those states.
  2. Motor saturation: share of motor activations >= 0.98 or <= -0.08 (rho in (-0.1, 1)), share with
     slope rho'(v) below 2% of its maximum (the latch: no slope for the nudge).
  3. scale_cap: share of col->motor synapses with |efficacy| >= 0.99 * scale_cap.
  4. ADR-0006 reconciliation: input sensitivity (main quantity) for the trained EP net, for the random
     lake + ridge (control: does the sign pattern need EP at all?) and for the trained net after a symmetric
     permutation of its col<->col reciprocal pairs; plus the two-segment product
     sum_k W[act_lead -> col_k] W[col_k -> class] as a decomposition only (not a criterion).
Weights (efficacy, bias, dense W) are saved per subject/net under receipts/M9c/readout_diag/<commit>/.
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
from .topology import PREDICTIONS, feat_index, EPS

M6 = dict(leak=1.0, rowmass_project=0.9, eta=0.5, temperature=0.01, epochs=100)
K = lambda y, p: float(cohen_kappa_score(y, p))


def col_states(sb, X):
    n, k, _ = X.shape; eq = sb.free_states(X)
    a = np.asarray(eq.state.activation); v = np.asarray(eq.state.v)
    ok = (np.asarray(eq.converged) & np.isfinite(a).all(1)).reshape(n, k)
    return (a[:, sb.col0:sb.out0].reshape(n, k, -1), a[:, sb.outputs].reshape(n, k, -1),
            v[:, sb.outputs].reshape(n, k, -1), ok)


def trial_mean(S, ok):
    w = ok[..., None].astype(float); return (S * w).sum(1) / np.maximum(w.sum(1), 1)


def readout_matrix(sb):
    W = np.asarray(sb.learner.brain.dense())  # W[pre, post]
    return W[sb.col0:sb.out0][:, sb.outputs], np.asarray(sb.learner.brain.bias)[sb.outputs]


def rho_slope(v):  # learning_neuron_model: rho = tanh(v/2) (v>0), 0.1 tanh(v/2) (v<0)
    s = 0.5 / np.cosh(v / 2) ** 2; return np.where(v > 0, s, 0.1 * s)


def sens(score_fn, X, y, fi, cls, eps=EPS):
    """Mean d score[cls] / d feature over class trials (all sub-windows), score_fn(X) -> (n, k, 4)."""
    m = y == cls; Xc = X[m]; Xp = Xc.copy(); Xp[..., fi] += eps
    return float(((score_fn(Xp)[..., cls] - score_fn(Xc)[..., cls]) / eps).mean())


def permute_pairs(sb, rng):
    """Symmetric control: shuffle efficacies among col<->col reciprocal pairs, keep each pair equal."""
    eff = np.asarray(sb.learner.brain.efficacy).copy(); pre, post = sb.conn.pre, sb.conn.post
    col = np.flatnonzero((sb.kind == "col") & (pre < post))
    rev = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(pre, post))}
    vals = eff[col][rng.permutation(len(col))]
    for i, v in zip(col, vals):
        eff[i] = v; eff[rev[(int(post[i]), int(pre[i]))]] = v
    sb.learner.brain.efficacy = eff
    return sb


def diag_net(sb, d, tag, out):
    tr, va = d["train"], d["val"]; Xtr, Xva = tr["track2_x"], va["track2_x"]; fb = fallback_class(tr["y"])
    Str, Mtr, Vtr, oktr = col_states(sb, Xtr); Sva, Mva, Vva, okva = col_states(sb, Xva)
    pred_ep, _ = aggregate(Mva, okva, fb)
    W, b = readout_matrix(sb); Fva, Ftr = trial_mean(Sva, okva), trial_mean(Str, oktr)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 13)).fit((Ftr - mu) / sd, tr["y"])
    M = np.concatenate([Mtr.reshape(-1, 4), Mva.reshape(-1, 4)]); V = np.concatenate([Vtr.reshape(-1, 4), Vva.reshape(-1, 4)])
    out_syn = (sb.kind == "out"); eff = np.abs(np.asarray(sb.learner.brain.efficacy))
    r = {"kappa_ep_adr0008": K(va["y"], pred_ep),
         "kappa_ep_linear_on_mean_states": K(va["y"], (Fva @ W + b).argmax(1)),
         "kappa_ridge_on_same_states": K(va["y"], ridge.predict((Fva - mu) / sd)),
         "kappa_train_ep_adr0008": K(tr["y"], aggregate(Mtr, oktr, fb)[0]),
         "motor_sat_hi": float((M >= 0.98).mean()), "motor_sat_lo": float((M <= -0.08).mean()),
         "motor_flat_slope": float((rho_slope(V) < 0.02 * 0.25).mean()),
         "motor_act_mean": M.mean(0).round(4).tolist(), "motor_act_std": M.std(0).round(4).tolist(),
         "scale_cap": sb.cfg.scale_cap, "out_at_cap": float((eff[out_syn] >= 0.99 * sb.cfg.scale_cap).mean()),
         "out_eff_max": float(eff[out_syn].max()), "W_out_absmean": float(np.abs(W).mean()),
         "cert": sb.cert()}
    # ADR-0006 reconciliation
    ep_score = lambda X: col_states(sb, X)[1]
    r["sens_ep"] = {nm: sens(ep_score, Xva, va["y"], feat_index("track2", ld, kd), c) for nm, ld, kd, c, _ in PREDICTIONS}
    Wfull = np.asarray(sb.learner.brain.dense())
    cols = np.arange(sb.col0, sb.out0)
    r["two_segment_product"] = {  # decomposition only (ADR-0006): sum_k W[feat -> col_k] * W[col_k -> class]
        nm: float(Wfull[feat_index("track2", ld, kd), cols] @ Wfull[cols, sb.outputs[c]])
        for nm, ld, kd, c, _ in PREDICTIONS}
    np.savez_compressed(out / f"{tag}.npz", efficacy=np.asarray(sb.learner.brain.efficacy),
                        bias=np.asarray(sb.learner.brain.bias), dense=Wfull.astype(np.float32),
                        pre=sb.conn.pre, post=sb.conn.post, kind=sb.kind)
    return r


def one(subject, seed, out_dir, epochs=100):
    from pathlib import Path
    out = Path(out_dir); t0 = time.time(); d = load_subject(subject); va = d["val"]
    res = {"subject": subject, "seed": seed}
    base = {**M6, "epochs": epochs}
    for tag, opts in (("full", base), ("readout", {**base, "train_internal": False, "project_plastic_only": True})):
        sb, stats, _ = train_sandbox(d["train"], seed, **opts)
        res[tag] = diag_net(sb, d, f"A{subject:02d}_{tag}", out)
        res[tag]["n_projections"] = getattr(sb, "n_projections", 0)
        if tag == "full":  # symmetric reciprocal-pair permutation of the trained lake
            permute_pairs(sb, np.random.default_rng(seed + 7))
            sc = lambda X: col_states(sb, X)[1]
            res["full_perm"] = {"cert": sb.cert(), "sens": {nm: sens(sc, va["track2_x"], va["y"], feat_index("track2", ld, kd), c)
                                                          for nm, ld, kd, c, _ in PREDICTIONS}}
    # random lake + ridge sensitivity (no EP at all)
    sb0 = Sandbox(SandboxConfig(**M6), seed)
    Str, _, _, oktr = col_states(sb0, d["train"]["track2_x"]); F = trial_mean(Str, oktr)
    mu, sd = F.mean(0), F.std(0) + 1e-9
    rg = RidgeClassifierCV(alphas=np.logspace(-3, 3, 13)).fit((F - mu) / sd, d["train"]["y"])
    rs = lambda X: (((col_states(sb0, X)[0] - mu) / sd) @ rg.coef_.T + rg.intercept_)
    res["ridge_random_lake"] = {"sens": {nm: sens(rs, va["track2_x"], va["y"], feat_index("track2", ld, kd), c)
                                         for nm, ld, kd, c, _ in PREDICTIONS}}
    res["wall_s"] = time.time() - t0
    print(f"A{subject:02d} done {res['wall_s']:.0f}s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default="1-9"); ap.add_argument("--jobs", type=int, default=3); ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args(); lo, _, hi = a.subjects.partition("-"); subs = list(range(int(lo), int(hi or lo) + 1))
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    out = ROOT / "receipts" / "M9c" / "readout_diag" / commit; out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, subs, [0] * len(subs), [str(out)] * len(subs), [a.epochs] * len(subs)))
    (out / "runs.json").write_text(json.dumps(res, default=float, indent=1)); print(out / "runs.json")


if __name__ == "__main__":
    main()
