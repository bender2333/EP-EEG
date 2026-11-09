"""M13 §10: diagnosis before closing (reports/M13.md §10). Why do rhythm / power-spectrum priors not help to
reconstruct a lead the lake has never seen hidden?

Same task and lakes as §8 (m13_rand: folds, random patches, tier U / N test sets, cap 0.9, U 3 000, from the untrained
E0 initial lake, single subject). One job = one (subject, seed, cond, data):
  cond  none | field (true positions, ell 3 cm, lam 0.5)
  data  raw  : the §8 data (the receipt's final_sha256 must equal the §8 one: exact reproduction)
        lap  : volume conduction reduced: Hjorth surface Laplacian on the 22-lead montage (each lead minus the mean of
               its 4 nearest leads), then z-scored per lead with the training-trial statistics; the lake is trained
               and tested on these signals; every reference is recomputed in the same space
Test signals: every 2nd sample of the test trials (125 Hz; the data are 4-35 Hz band-passed, so no aliasing), static
equilibrium per sample. Fits (scalar gain, ridge readouts) use every 10th sample of the training trials, same hidden set.

Predictors of the scored lead c (its test set hides c and its 2 nearest leads):
  lake_own     outlet of c through the fixed inverse (the score used so far)
  lake_gain    lake_own times one least-squares scalar
  lake_ridge   ridge from the whole lake state at equilibrium (88 columns + 22 outlets) to c: is the information in the
               lake but not in its own outlet
  mean_visible, nearest, spline, ridge_vis (ridge from the visible leads)
  surrogate    the true signal of c with its Fourier phases randomised per trial: exact spectrum, no phase
Two families of metrics (the human's split):
  waveform / phase  r2; phase agreement in mu and beta (mean cos of the Hilbert phase difference; 1 = in phase, 0 = none)
  priors side       log10 power ratio (theta 4-8, mu 8-13, beta 13-30 Hz; 0 = exact); spectral shape (Pearson r of
                    the log Welch spectrum 4-35 Hz, per trial); mu peak frequency error (Hz, trial-averaged spectrum);
                    trial-to-trial tracking of log mu and log beta power (Pearson r across the 96 test trials)
H2 extras: corr(prediction, visible mean) per trial (median), for lake_own and for the truth; share of lake_own's
variance explained by the visible mean (R^2 of a scalar fit on the test rows).
H3 extras: calibration slope of lake_own (the scalar), equilibrium converged fraction.
"""
from __future__ import annotations

import argparse, sys, time, warnings

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

warnings.filterwarnings("ignore")
from . import m12_s1 as S                        # noqa: E402
from . import m13_common as C                    # noqa: E402
from . import m13_e0 as E                        # noqa: E402
from . import m13_learn as Lr                    # noqa: E402
from . import m13_rand as R                      # noqa: E402

OUT = C.ROOT / "receipts" / "M13" / "G"
WDIR = C.ROOT / "runs" / "M13G" / "w"
FS2 = 125.0; STEP2 = 2; CAP = 0.9
BANDS = {"theta": (4, 8), "mu": (8, 13), "beta": (13, 30)}
SOS = {b: butter(4, f, btype="band", fs=FS2, output="sos") for b, f in BANDS.items()}
NB4 = {c: [int(j) for j in np.argsort(R.DIST[c]) if j != c][:4] for c in range(C.N_LEADS)}
PRED = ("lake_own", "lake_gain", "lake_ridge", "mean_visible", "nearest", "spline", "ridge_vis", "surrogate")


def laplacian(tr, te):
    """Hjorth Laplacian per lead, then per-lead z-score with the training-trial statistics."""
    L = np.eye(C.N_LEADS)
    for c, nb in NB4.items(): L[c, nb] -= 0.25
    a, b = np.einsum("ij,njt->nit", L, tr), np.einsum("ij,njt->nit", L, te)
    mu = a.mean((0, 2), keepdims=True); sd = a.std((0, 2), keepdims=True)
    return (a - mu) / sd, (b - mu) / sd


def states(g, X, hid, chunk=2400):
    M = np.zeros((len(X), C.N_LEADS), bool); M[:, list(hid)] = True; A, ok = [], []
    for i in range(0, len(X), chunk):
        eq = g["lr"].brain.equilibrate(E.drive_rows(g, X[i:i + chunk], M[i:i + chunk]), budget=S.BUDGET, tolerance=E.EVAL_TOL, damping=3)
        A.append(np.asarray(eq.state.activation)); ok.append(np.asarray(eq.converged))
    A = np.concatenate(A); return A[:, np.r_[g["cols"], g["out"]]], E.own_inverse(A[:, g["out"]]), np.concatenate(ok)


def surrogate(y, rng):
    F = np.fft.rfft(y, axis=-1); ph = rng.uniform(0, 2 * np.pi, F.shape); ph[:, 0] = 0
    if y.shape[-1] % 2 == 0: ph[:, -1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=y.shape[-1], axis=-1)


def metrics(p, y, mv):
    """p, y, mv: (trials, T) prediction, truth, visible mean of the scored lead."""
    r = {"r2": float(C.r2_cols(p.reshape(-1, 1), y.reshape(-1, 1))[0])}
    pb, yb = {}, {}
    for b, sos in SOS.items():
        pb[b], yb[b] = sosfiltfilt(sos, p, axis=-1), sosfiltfilt(sos, y, axis=-1)
        r[f"logpow_{b}"] = float(np.log10(pb[b].var(-1).mean() / yb[b].var(-1).mean()))
        lp, ly = np.log10(pb[b].var(-1) + 1e-12), np.log10(yb[b].var(-1))
        r[f"track_{b}"] = float(np.corrcoef(lp, ly)[0, 1]) if lp.std() > 1e-9 else 0.0
    for b in ("mu", "beta"):
        r[f"phase_{b}"] = float(np.cos(np.angle(hilbert(pb[b], axis=-1)) - np.angle(hilbert(yb[b], axis=-1))).mean())
    f, Pp = welch(p, fs=FS2, nperseg=125, axis=-1); _, Py = welch(y, fs=FS2, nperseg=125, axis=-1)
    k = (f >= 4) & (f <= 35); lp, ly = np.log10(Pp[:, k] + 1e-12), np.log10(Py[:, k])
    r["shape"] = float(np.mean([np.corrcoef(a, b)[0, 1] if a.std() > 1e-9 else 0.0 for a, b in zip(lp, ly)]))
    m = (f >= 8) & (f <= 13); fm = f[m]
    r["mu_peak_err"] = float(abs(fm[np.argmax(Pp[:, m].mean(0))] - fm[np.argmax(Py[:, m].mean(0))]))
    r["corr_mv"] = float(np.median([np.corrcoef(a, b)[0, 1] if a.std() > 1e-9 else 0.0 for a, b in zip(p, mv)]))
    return r


def run(subject, seed, cond, data, U):
    t0 = time.time()
    d = C.load_real(subject, split="t04_val45")
    seal = C.e_seal()
    if seal["matches"]: raise RuntimeError(f"E files present: {seal['matches']}")
    tr, te = d["train"], d["test"]
    if data == "lap": tr, te = laplacian(tr, te)
    Xtr = C.rows(tr); Xg = C.rows(tr, step=C.RIDGE_STEP); X2 = C.rows(te, step=STEP2)
    n = te.shape[0]; T2 = X2.shape[0] // n
    g, init_sha = Lr.build(subject, seed, cond, R.ELL, R.LAM, der=seed, trained=False, cap=CAP)
    trn = R.train(g, Xtr, seed, U, log=lambda m: print(f"[{subject}/{seed}/{cond}/{data}] {m}", flush=True), cap=CAP)
    eff, bias = np.asarray(g["lr"].brain.efficacy, float), np.asarray(g["lr"].brain.bias, float)
    WDIR.mkdir(parents=True, exist_ok=True); np.savez(WDIR / f"A{subject:02d}_s{seed}_{cond}_{data}.npz", eff=eff, bias=bias)
    rng = np.random.default_rng(1000 * subject + seed); per = {}
    for c, hid in R.test_sets().items():
        vis = [j for j in range(C.N_LEADS) if j not in hid]; k = list(hid).index(c)
        Sg, Pg, okg = states(g, Xg, hid); St, Pt, okt = states(g, X2, hid)
        a = float(Pg[:, c] @ Xg[:, c] / max(Pg[:, c] @ Pg[:, c], 1e-12))
        pr_lr, _ = E.ridge_fit_predict(Sg, Xg[:, [c]], St)
        pr_rv, _ = E.ridge_fit_predict(Xg[:, vis], Xg[:, [c]], X2[:, vis])
        y = X2[:, c].reshape(n, T2); mv = X2[:, vis].mean(1).reshape(n, T2)
        P = {"lake_own": Pt[:, c], "lake_gain": a * Pt[:, c], "lake_ridge": pr_lr[:, 0], "mean_visible": X2[:, vis].mean(1),
             "nearest": X2[:, min(vis, key=lambda j: R.DIST[c, j])], "spline": X2[:, vis] @ C.spline_matrix(hid)[k],
             "ridge_vis": pr_rv[:, 0]}
        res = {kk: metrics(v.reshape(n, T2), y, mv) for kk, v in P.items()}
        res["surrogate"] = metrics(surrogate(y, rng), y, mv)
        res["truth_corr_mv"] = float(np.median([np.corrcoef(u, v)[0, 1] for u, v in zip(y, mv)]))
        lo = Pt[:, c]; b_ = float(mv.ravel() @ lo / (mv.ravel() @ mv.ravel()))
        res["lake_var_by_mv"] = float(C.r2_cols((b_ * mv.ravel())[:, None], lo[:, None])[0])
        yy = X2[:, c]; b2 = float(mv.ravel() @ yy / (mv.ravel() @ mv.ravel()))
        res["truth_var_by_mv"] = float(C.r2_cols((b2 * mv.ravel())[:, None], yy[:, None])[0])
        res.update(scalar=a, qualified=float(okt.mean()), qualified_fit=float(okg.mean()))
        per[c] = res
        print(f"[{subject}/{seed}/{cond}/{data}] {C.LEADS[c]} own {res['lake_own']['r2']:.3f} ridge {res['lake_ridge']['r2']:.3f}", flush=True)
    flat = {}
    for c, r in per.items():
        f = {f"{kk}.{m}": v for kk in PRED for m, v in r[kk].items()}
        f.update({kk: r[kk] for kk in ("truth_corr_mv", "lake_var_by_mv", "truth_var_by_mv", "scalar", "qualified", "qualified_fit")})
        flat[c] = f
    summ = {}
    for key in flat[0]: summ.update(R.summarise(flat, seed, key))
    rec = {"stage": "M13", "step": "10-diagnosis", "cond": cond, "data": data, "cap": CAP, "split": "test", "subject": subject,
           "seed": seed, "U": U, "hide_fold": [C.LEADS[c] for c in R.hide_fold(seed)],
           "tiers": {kk: [C.LEADS[c] for c in v] for kk, v in R.tier_sets(seed).items()},
           "data_source": d["source"], "init_sha256": init_sha, "final_sha256": C.sha(eff, bias), "rho_end": Lr.rho(g),
           "train": trn, "per_lead": {C.LEADS[c]: v for c, v in per.items()}, "summary": summ, "e_seal": seal}
    rec.update(env=C.env_info(), wall_s=time.time() - t0, finished=C.now())
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", default="none", choices=["none", "field"]); ap.add_argument("--data", default="raw", choices=["raw", "lap"])
    ap.add_argument("--U", type=int, default=3000)
    a = ap.parse_args(argv)
    path = OUT / "test" / f"A{a.subject:02d}_s{a.seed}_{a.cond}_{a.data}_U{a.U}.json"
    if a.U != 3000: path = OUT / "smoke" / path.name
    if path.exists(): print("exists", path); return
    rec = run(a.subject, a.seed, a.cond, a.data, a.U)
    C.write_receipt(path, rec)
    s = rec["summary"]
    print(path.name, {k: round(s[k], 3) for k in ("lake_own.r2_U", "lake_ridge.r2_U", "mean_visible.r2_U")}, f"wall {rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
