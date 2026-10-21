"""M12 S1 (reports/M12.md S1 + amendments 5, 6, 7): toy learnability on the even/odd neuron under the certificate.

  T1  x ~ N(0, I2), y = x1 x2; quadratic nudge to t = clip(0.1 y, -0.7, 1.2) on one N1 output; hidden N2, H in {8, 32}.
      Judged: (i) trained lake + ridge R2 >= 0.9, (ii) EP output R2 (yhat = 10 s_out, nothing fitted) >= 0.9,
      (iii) random-lake output R2 <= 0.3 (3-seed min / min / max), per H.
  T2  6 sources (A, B, 4 noise), 6 channels, 64-sample windows; class 0: var A 0.5, class 1: var B 0.5; 6x64 linear
      sources, K = 4 entry filters tied across time, 4 pools, 2 N2 outputs (CE, T = 0.1).
      Judged: kappa_ridge(EP) >= 0.9 kappa_oracle per seed, m (distinct-filter match) >= 0.8 (3-seed min).
  T3  same T2 architecture trained by the implicit gradient (IFT), |mean kappa_EP - mean kappa_IFT| <= 0.03.
Fixed: torch f64, beta 0.04, tol 3e-3, qualified, damping 3, eta 0.05, eta_bias 0.005, U = 19 200, batch 64,
init rho~ 0.6, symmetric weighted row-mass projection to rho~ <= 0.9 after every update.
Subcommands: entry (S1.12), minit (B3b), smoke (N3/N4), run (one official/fix job), timing helpers.
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np

THREADS = int(os.environ.get("M12_THREADS", "2"))
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, str(THREADS))

from . import m12_common as C
import cadence as cd
from cadence.learning import Learner, LearnerConfig, LearningPhaseError, evenodd_neuron_model, time_shared_entry, tie_rates
from .ift import ep_equivalent

NEUR = {"N1": (1.0, 0.25), "N2": (1.0, 1.0)}
DEV = (1.0, 0.6)
ETA, ETA_B, U_FULL, BATCH = 0.05, 0.005, 19200, 64
BETA, TOL, TEMP, CAP, RHO0 = 0.04, 3e-3, 0.1, 0.9, 0.6
C_T1, K_T1 = 0.1, 10.0
EVAL_TOL, IFT_TOL, BUDGET = 1e-6, 1e-9, 50000
N1_TR, N1_DV, N1_TE = 4096, 1024, 4096
N2_TR, N2_DV, N2_TE = 2048, 512, 2048
CH, POS, KF, NOUT2 = 6, 64, 4, 2
ALPHAS = np.logspace(-6, 4, 21)
S1 = C.OUT / "S1"
CLOCK = S1 / "clock.json"
CLOCK_LIMIT_S = 7 * 24 * 3600
CKPT_EVERY, LOG_EVERY = 500, 100


# ----------------------------------------------------------------------------- clock (amendment 7 N2)
DRY = bool(os.environ.get("M12_DRYRUN"))   # verifier plumbing test only: no clock, no ledger


def start_clock(what):
    if DRY: return 0.0
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    if not CLOCK.exists():
        CLOCK.write_text(json.dumps({"started": time.time(), "started_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                     "first_run": what, "limit_s": CLOCK_LIMIT_S}))
    return time.time() - json.loads(CLOCK.read_text())["started"]


def clock_left():
    if not CLOCK.exists(): return CLOCK_LIMIT_S
    return CLOCK_LIMIT_S - (time.time() - json.loads(CLOCK.read_text())["started"])


# ----------------------------------------------------------------------------- generators
def t1_data(seed):
    rng = np.random.default_rng(1000 + seed)
    d = {"x_tr": rng.normal(size=(N1_TR, 2)), "x_dv": rng.normal(size=(N1_DV, 2)), "x_te": rng.normal(size=(N1_TE, 2))}
    for s in ("tr", "dv", "te"):
        x = d["x_" + s]; d["y_" + s] = x[:, 0] * x[:, 1]; d["t_" + s] = np.clip(C_T1 * d["y_" + s], -0.7, 1.2)
    d["sha"] = C.sha(d["x_tr"], d["x_dv"], d["x_te"])
    return d


VAR_MEAN = np.array([0.75, 0.75, 1, 1, 1, 1.0])


def mixing(rng):
    draws = 0
    while True:
        draws += 1
        M = rng.normal(size=(CH, CH)); M /= np.linalg.norm(M, axis=0, keepdims=True)
        if np.linalg.cond(M) > 10: continue
        Mi = np.linalg.inv(M); Dt = np.sqrt((M ** 2) @ VAR_MEAN)
        ua, ub = Mi[0] * Dt, Mi[1] * Dt
        cab = abs(ua @ ub) / np.linalg.norm(ua) / np.linalg.norm(ub)
        if cab > 0.5: continue
        return M, Mi, draws, float(cab), float(np.linalg.cond(M))


def t2_windows(rng, M, n):
    y = rng.permutation(np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)])
    var = np.ones((n, CH)); var[y == 0, 0] = 0.5; var[y == 1, 1] = 0.5
    S = rng.normal(size=(n, CH, POS)) * np.sqrt(var)[:, :, None]
    X = np.einsum("ij,njt->nit", M, S)
    logv = np.log(S[:, :2].var(axis=2))
    return X, y, logv


def t2_data(seed):
    rng = np.random.default_rng(1000 + seed)
    M, Mi, draws, cab, cond = mixing(rng)
    d = {"M": M, "draws": draws, "cos_ab_true": cab, "cond": cond}
    for s, n in (("tr", N2_TR), ("dv", N2_DV), ("te", N2_TE)):
        X, y, lv = t2_windows(rng, M, n); d["X_" + s], d["y_" + s], d["logv_" + s] = X, y, lv
    mu = d["X_tr"].mean(axis=(0, 2)); sd = d["X_tr"].std(axis=(0, 2)); d["mu"], d["D"] = mu, sd
    for s in ("tr", "dv", "te"):
        Z = (d["X_" + s] - mu[None, :, None]) / sd[None, :, None]
        d["Z_" + s] = Z; d["in_" + s] = Z.transpose(0, 2, 1).reshape(len(Z), POS * CH)   # source (p, c) = p*6 + c
    ua, ub = Mi[0] * sd, Mi[1] * sd; d["uA"], d["uB"] = ua / np.linalg.norm(ua), ub / np.linalg.norm(ub)
    d["sha"] = C.sha(d["X_tr"], d["y_tr"], d["X_dv"], d["y_dv"], d["X_te"], d["y_te"])
    return d


def match(W, ua, ub):
    """m = max_{k != k'} min(|cos(w_k, uA)|, |cos(w_k', uB)|) (amendment 7 B3a)."""
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    ca, cb = np.abs(Wn @ ua), np.abs(Wn @ ub); best = -1.0
    for k in range(len(W)):
        for k2 in range(len(W)):
            if k != k2: best = max(best, min(ca[k], cb[k2]))
    return float(best), ca.tolist(), cb.tolist()


def csp_filters(d):
    Z, y = d["Z_tr"], d["y_tr"]
    covs = [np.mean([z @ z.T / POS for z in Z[y == c]], axis=0) for c in (0, 1)]
    from scipy.linalg import eigh
    w, V = eigh(covs[0], covs[0] + covs[1])
    F = V[:, [0, 1, -2, -1]].T
    return F, w


# ----------------------------------------------------------------------------- graphs
def _recip(pre, post, eff, ff, a_idx, b_idx, w):
    P, Q = np.meshgrid(a_idx, b_idx, indexing="ij")
    pre += [P.ravel(), Q.ravel()]; post += [Q.ravel(), P.ravel()]; eff += [w, w]; ff += [np.ones(P.size, bool)] * 2


def finish(n, ns, pre, post, eff, ff, groups, a, b, out, nudge, eta, backend="torch64", rho0=RHO0, rescale=True, lip_for_scale=None):
    pre, post, eff, ff = map(np.concatenate, (pre, post, eff, ff))
    groups = None if groups is None else np.concatenate(groups)
    conn = cd.Connectome.from_synapses(n, pre=pre, post=post)
    order = np.lexsort((pre, post)); assert np.array_equal(conn.pre, pre[order]) and np.array_equal(conn.post, post[order])
    eff, ff = eff[order], ff[order]
    if groups is not None: groups = groups[order]
    model = evenodd_neuron_model(a, b, linear=range(ns), dt=0.5); lip = model.lipschitz_per_neuron(n)
    if rescale:
        L = lip if lip_for_scale is None else lip_for_scale
        r = np.bincount(conn.post, np.abs(eff) * L[conn.pre] * ff, minlength=n)[ns:].max()
        eff = np.where(ff, eff * rho0 / r, eff)
    kw = {"cpu64": {}, "torch64": {"backend": "torch", "device": "cpu", "precision": "float64"}}[backend]
    brain = cd.NeuralGraph(conn, model, efficacy=eff, bias=np.zeros(n), dense_limit=4096, **kw)
    lc = LearnerConfig(beta=BETA, eta=eta, eta_bias=eta / 10, centered=True, free_steps=BUDGET, nudged_steps=BUDGET, tolerance=TOL,
                       nudge=nudge, temperature=TEMP, normalize=0.0, normalize_floor=1e-3, momentum=0.0, decay=0.0,
                       scale_cap=1e9, qualified=True, damping=3, surprise=False)
    pn = np.zeros(n, bool); pn[ns:] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = Learner(brain, list(out), config=lc, plastic_synapses=np.ones(len(eff), bool), plastic_neurons=pn, reciprocal=True,
                     tie_groups=groups, synapse_rate=None if groups is None else tie_rates(groups))
    return {"lr": lr, "conn": conn, "ff": ff, "groups": groups, "model": model, "lip": lip, "ns": ns, "n": n, "out": np.asarray(out)}


def build_t1(H, init_seed, hid_ab=NEUR["N2"], out_ab=NEUR["N1"], backend="torch64", eta=ETA, odd=False):
    rng = np.random.default_rng(init_seed)
    ns = 2; hid = np.arange(ns, ns + H); o = ns + H; n = o + 1
    pre, post, eff, ff = [], [], [], []
    P, Q = np.meshgrid(np.arange(ns), hid, indexing="ij")
    pre.append(P.ravel()); post.append(Q.ravel()); eff.append(rng.normal(0, 1, P.size)); ff.append(np.zeros(P.size, bool))
    iu, ju = np.triu_indices(H, 1); w = rng.normal(0, 1 / np.sqrt(H), len(iu))
    pre += [hid[iu], hid[ju]]; post += [hid[ju], hid[iu]]; eff += [w, w]; ff += [np.ones(len(iu), bool)] * 2
    _recip(pre, post, eff, ff, hid, np.array([o]), rng.normal(0, 1 / np.sqrt(H), H))
    a = np.ones(n); b = np.zeros(n)
    a[hid], b[hid] = hid_ab; a[o], b[o] = out_ab
    lip_scale = None
    if odd:   # amendment 7 N11: same init (scaled with the N2 Lipschitz constants), only b changed, no rescale
        lip_scale = evenodd_neuron_model(a, b, linear=range(ns), dt=0.5).lipschitz_per_neuron(n)
        b[hid] = 0.0
    return finish(n, ns, pre, post, eff, ff, None, a, b, [o], "quadratic", eta, backend, lip_for_scale=lip_scale)


def build_t2(init_seed, ab=NEUR["N2"], out_ab=NEUR["N2"], backend="torch64", eta=ETA):
    rng = np.random.default_rng(init_seed)
    ns = CH * POS
    p0, q0, tie, entry = time_shared_entry(CH, POS, KF)
    shared = rng.normal(0, 1 / np.sqrt(CH), KF * CH)
    pre, post, eff, ff, grp = [p0], [q0], [shared[tie]], [np.zeros(len(p0), bool)], [tie]
    ent = np.arange(ns, ns + KF * POS); pool = np.arange(ns + KF * POS, ns + KF * POS + KF); out = np.arange(pool[-1] + 1, pool[-1] + 1 + NOUT2)
    n = out[-1] + 1
    def rec(a_, b_, w):
        k0 = len(pre); _recip(pre, post, eff, ff, a_, b_, w); grp.extend([-np.ones(len(pre[i]), np.int64) for i in range(k0, len(pre))])
    iu, ju = np.triu_indices(len(ent), 1); w = rng.normal(0, 1 / np.sqrt(len(ent)), len(iu))
    pre += [ent[iu], ent[ju]]; post += [ent[ju], ent[iu]]; eff += [w, w]; ff += [np.ones(len(iu), bool)] * 2; grp += [-np.ones(len(iu), np.int64)] * 2
    for k in range(KF): rec(entry[k], pool[k:k + 1], np.full(POS, 1.0 / POS))
    iu, ju = np.triu_indices(KF, 1); w = rng.normal(0, 1 / np.sqrt(KF), len(iu))
    pre += [pool[iu], pool[ju]]; post += [pool[ju], pool[iu]]; eff += [w, w]; ff += [np.ones(len(iu), bool)] * 2; grp += [-np.ones(len(iu), np.int64)] * 2
    rec(pool, out, rng.normal(0, 1 / np.sqrt(KF), KF * NOUT2))
    a = np.ones(n); b = np.zeros(n); a[ns:], b[ns:] = ab; a[out], b[out] = out_ab
    g = finish(n, ns, pre, post, eff, ff, grp, a, b, list(out), "cross_entropy", eta, backend)
    g["entry"] = entry; g["pool"] = pool
    return g


def filters(g):
    """Tied entry filters W[k, c] (efficacy of group k*6 + c)."""
    eff = np.asarray(g["lr"].brain.efficacy); grp = g["groups"]; W = np.zeros((KF, CH))
    for k in range(KF):
        for c in range(CH):
            vals = eff[grp == k * CH + c]; W[k, c] = vals[0]
    return W


# ----------------------------------------------------------------------------- certificate, projection
def rho(g, eff=None):
    eff = np.asarray(g["lr"].brain.efficacy) if eff is None else eff
    c = g["conn"]; return float(np.bincount(c.post, np.abs(eff) * g["lip"][c.pre] * g["ff"], minlength=g["n"])[g["ns"]:].max())


def set_params(lr, eff, bias):
    kernel = lr.brain._torch
    if kernel is None:
        lr.brain = lr.brain.with_parameters(efficacy=eff, bias=bias)
    else:
        torch = kernel.torch
        lr.brain = lr.brain._with_device_parameters(torch.from_numpy(np.ascontiguousarray(eff)).to(kernel.device, kernel.param_dtype),
                                                    torch.from_numpy(np.ascontiguousarray(bias)).to(kernel.device, kernel.param_dtype))


def project(g, cap=CAP):
    """Symmetric per-neuron weighted row-mass projection on free->free edges only (sandbox.project_rowmass, weighted)."""
    lr, c, ff, L, n, ns = g["lr"], g["conn"], g["ff"], g["lip"], g["n"], g["ns"]
    eff = np.asarray(lr.brain.efficacy, float).copy(); changed = False
    for _ in range(100):
        inc = np.bincount(c.post, np.abs(eff) * L[c.pre] * ff, minlength=n); inc[:ns] = 0
        over = inc > cap * (1 - 1e-9)
        if not over.any(): break
        f = np.ones(n); f[over] = cap / inc[over] * (1 - 1e-6)
        fac = np.where(ff, np.minimum(f[c.post], f[c.pre]), 1.0)
        eff = eff * fac; changed = True
    if changed: set_params(lr, eff, np.asarray(lr.brain.bias, float))
    return changed


def certificate(g):
    cert = cd.certificate(g["lr"].brain, sources=range(g["ns"]))
    return cert.to_dict()


# ----------------------------------------------------------------------------- steps
def target_array(g, t):
    T = np.zeros((len(t), g["n"])); T[:, g["out"][0]] = t; return T


def drive(g, xin):
    D = np.zeros((len(xin), g["n"])); D[:, :g["ns"]] = xin; return D


def ep_step(g, D, tgt, eta=ETA):
    lr = g["lr"]
    free = lr.free(D)
    plus = lr.nudged(D, free, tgt, 1.0); minus = lr.nudged(D, free, tgt, -1.0)
    ce, cb = lr.contrast(free, plus, minus)
    lr.apply(eta * np.asarray(ce), (eta / 10) * np.asarray(cb))
    return free, [int(getattr(s, "steps", -1) or -1) for s in (free, plus, minus)]


def ift_grad(g, D, gs_fn, tol=IFT_TOL):
    """Free-block adjoint (sources have no incoming synapses). Returns (c, c_b, loss, eq) in EP-equivalent coordinates."""
    lr, ns, c = g["lr"], g["ns"], g["conn"]
    eq = lr.brain.equilibrate(D, budget=BUDGET, tolerance=tol, damping=3)
    if not np.all(eq.converged): return None, None, None, eq
    s, v = np.asarray(eq.state.activation), np.asarray(eq.state.v)
    sp = g["model"].slope_at(v)[:, ns:]; gs, loss = gs_fn(s)
    W = np.asarray(lr.brain.dense())[ns:, ns:]
    JT = np.eye(g["n"] - ns)[None] - sp[:, :, None] * W[None]
    lam = np.linalg.solve(JT, (sp * gs[:, ns:])[..., None])[..., 0]
    lamf = np.pad(lam, ((0, 0), (ns, 0)))
    ge = (s[:, c.pre] * lamf[:, c.post]).mean(0)
    ce, cb = ep_equivalent(lr, ge, lamf.mean(0))
    return ce, cb, loss, eq


def gs_ce(g, y):
    def f(s):
        out = g["out"]; z = s[:, out] / TEMP; z = z - z.max(1, keepdims=True); lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        rows = np.arange(len(y)); loss = float(-(TEMP * lp[rows, y]).mean())
        G = np.zeros_like(s); p = np.exp(lp); p[rows, y] -= 1.0; G[:, out] = p
        return G, loss
    return f


def gs_quad(g, t):
    def f(s):
        o = g["out"][0]; G = np.zeros_like(s); G[:, o] = s[:, o] - t
        return G, float(0.5 * ((s[:, o] - t) ** 2).mean())
    return f


def ift_step(g, D, gs_fn, eta=ETA):
    ce, cb, loss, eq = ift_grad(g, D, gs_fn)
    if ce is None: return False, eq
    g["lr"].apply(eta * ce, (eta / 10) * cb)
    return True, eq


# ----------------------------------------------------------------------------- evaluation
def settle(g, xin, tol=EVAL_TOL, chunk=512):
    S, res, ok = [], [], []
    for i in range(0, len(xin), chunk):
        eq = g["lr"].brain.equilibrate(drive(g, xin[i:i + chunk]), budget=BUDGET, tolerance=tol, damping=3)
        S.append(np.asarray(eq.state.activation)); res.append(np.asarray(eq.residual)); ok.append(np.asarray(eq.converged))
    return np.concatenate(S), np.concatenate(res), np.concatenate(ok)


def features(g, S):
    keep = np.ones(g["n"], bool); keep[:g["ns"]] = False; keep[g["out"]] = False
    return S[:, keep]


def standardize(Ftr, *rest):
    mu, sd = Ftr.mean(0), Ftr.std(0); sd = np.where(sd > 1e-12, sd, 1.0)
    return [(F - mu) / sd for F in (Ftr, *rest)]


def r2(yh, y): return float(1 - ((yh - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def eval_t1(g, d):
    from sklearn.linear_model import RidgeCV
    Str, rtr, oktr = settle(g, d["x_tr"]); Ste, rte, okte = settle(g, d["x_te"])
    Ftr, Fte = standardize(features(g, Str), features(g, Ste))
    rg = RidgeCV(alphas=ALPHAS, cv=5).fit(Ftr, d["y_tr"]); pr = rg.predict(Fte)
    s_out = Ste[:, g["out"][0]]; yh = K_T1 * s_out
    return {"r2_ridge": r2(pr, d["y_te"]), "r2_out": r2(yh, d["y_te"]), "r2_t": r2(s_out, d["t_te"]),
            "corr2_out_y": float(np.corrcoef(s_out, d["y_te"])[0, 1] ** 2) if s_out.std() > 0 else 0.0,
            "ridge_alpha": float(rg.alpha_), "pred_ridge_te": pr, "s_out_te": s_out, "yhat_const": K_T1, "y_te": d["y_te"],
            "x_te_sha": C.sha(d["x_te"]), "eval_unqualified": int((~oktr).sum() + (~okte).sum()),
            "eval_rows": int(len(oktr) + len(okte)), "eval_max_residual": float(max(rtr.max(), rte.max()))}


def eval_t2(g, d):
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import cohen_kappa_score
    Str, rtr, oktr = settle(g, d["in_tr"]); Ste, rte, okte = settle(g, d["in_te"])
    Ftr, Fte = standardize(features(g, Str), features(g, Ste))
    rc = RidgeClassifierCV(alphas=ALPHAS, cv=5).fit(Ftr, d["y_tr"]); pr = rc.predict(Fte)
    W = filters(g); m, ca, cb = match(W, d["uA"], d["uB"])
    outp = Ste[:, g["out"]].argmax(1)
    return {"kappa_ridge": float(cohen_kappa_score(d["y_te"], pr)), "acc_ridge": float((pr == d["y_te"]).mean()),
            "pred_ridge_te": pr, "y_te": d["y_te"], "ridge_alpha": float(rc.alpha_), "W_filters": W, "m": m, "cosA": ca, "cosB": cb,
            "kappa_out": float(cohen_kappa_score(d["y_te"], outp)), "eval_unqualified": int((~oktr).sum() + (~okte).sum()),
            "eval_rows": int(len(oktr) + len(okte)), "eval_max_residual": float(max(rtr.max(), rte.max()))}


def oracle_t2(d):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import cohen_kappa_score
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(d["logv_tr"], d["y_tr"]); p = lda.predict(d["logv_te"])
    F, w = csp_filters(d); mc, ca, cb = match(F, d["uA"], d["uB"])
    return {"kappa_oracle": float(cohen_kappa_score(d["y_te"], p)), "pred_oracle_te": p, "csp_filters": F, "csp_eigs": w, "m_csp": mc}


# ----------------------------------------------------------------------------- B3b initial m
def t2_init_seed(seed, record=None):
    d = t2_data(seed); tries = []
    j = 0
    while True:
        s = seed + 10 * j; g = build_t2(s, backend="cpu64"); m = match(filters(g), d["uA"], d["uB"])[0]
        tries.append({"init_seed": s, "m_init": m})
        if m < 0.8: return s, tries
        j += 1


# ----------------------------------------------------------------------------- training job
def run_job(task, arm, seed, H=None, eta=ETA, beta=BETA, ab=None, out_ab=None, U=U_FULL, tag="official", dev=False, out_dir=None, ckpt_dir=None):
    C.e_seal(); import torch; torch.set_num_threads(THREADS)
    t_start = time.time(); clk = start_clock(f"{task}/{arm}/s{seed}")
    out_dir = Path(out_dir); name = f"{task}_{arm}" + (f"_H{H}" if H else "") + f"_s{seed}"
    if C.existing(out_dir, name): print("exists", name); return
    ckpt = Path(ckpt_dir) / f"{name}.npz"; ckpt.parent.mkdir(parents=True, exist_ok=True)
    reseed = None
    if task == "T1":
        d = t1_data(seed); init = seed
        g = build_t1(H, init, hid_ab=ab or NEUR["N2"], out_ab=out_ab or NEUR["N1"], eta=eta, odd=(arm == "randodd"))
        xin, ylab = d["x_tr"], None
    else:
        d = t2_data(seed); init, reseed = t2_init_seed(seed)
        g = build_t2(init, ab=ab or NEUR["N2"], out_ab=out_ab or NEUR["N2"], eta=eta)
        xin = d["in_tr"]; ylab = d["y_tr"].copy()
        if arm == "shuffled": ylab = np.random.default_rng(2000 + seed).permutation(ylab)
    lr = g["lr"]
    if beta != BETA: lr.config = type(lr.config)(**{**lr.config.__dict__, "beta": beta})
    init_sha = C.sha(np.asarray(lr.brain.efficacy), np.asarray(lr.brain.bias)); cert0 = certificate(g); rho_init = rho(g)
    trained = arm in ("ep", "shuffled", "ift")
    Uj = U if trained else 0
    nb = len(xin) // BATCH
    logs, acc, start_u, nproj, cert_false, refused_by = [], np.zeros(Uj, np.int8), 0, 0, 0, {}
    if trained and ckpt.exists():
        z = np.load(ckpt, allow_pickle=False)
        set_params(lr, z["eff"], z["bias"]); start_u = int(z["u"]); acc = z["acc"].copy(); nproj = int(z["nproj"]); cert_false = int(z["cert_false"])
        logs = json.loads(str(z["logs"])); print("resumed", name, start_u, flush=True)
    loss_win = []
    for u in range(start_u, Uj):
        ep_i, b_i = divmod(u, nb)
        perm = np.random.default_rng((3000 + seed) * 100003 + ep_i).permutation(len(xin))
        idx = perm[b_i * BATCH:(b_i + 1) * BATCH]; D = drive(g, xin[idx])
        steps = None
        try:
            if task == "T1":
                tgt = target_array(g, d["t_tr"][idx])
                free, steps = ep_step(g, D, tgt, eta)
                s = np.asarray(free.activation)[:, g["out"][0]]; loss_win.append(float(0.5 * ((s - d["t_tr"][idx]) ** 2).mean()))
                ok = True
            elif arm == "ift":
                ok, eq = ift_step(g, D, gs_ce(g, ylab[idx]), eta); steps = [int(eq.steps)]
                if ok: loss_win.append(float(gs_ce(g, ylab[idx])(np.asarray(eq.state.activation))[1]))
            else:
                tgt = lr.targets(ylab[idx]); free, steps = ep_step(g, D, tgt, eta)
                loss_win.append(float(gs_ce(g, ylab[idx])(np.asarray(free.activation))[1])); ok = True
        except LearningPhaseError as e:
            ok = False; refused_by[e.args[0] if e.args else "?"] = refused_by.get(e.args[0] if e.args else "?", 0) + 1
        acc[u] = 1 if ok else 0
        r_before = rho(g)
        if project(g): nproj += 1
        r_after = rho(g)
        if r_after >= 1.0: raise RuntimeError(f"implementation error: rho~ {r_after} >= 1 after projection (amendment 7 N11)")
        if not np.isfinite(np.asarray(lr.brain.efficacy)).all(): raise FloatingPointError("non-finite efficacy")
        if (u + 1) % LOG_EVERY == 0:
            cc = certificate(g); cert_false += 0 if cc["certified"] else 1
            logs.append({"u": u + 1, "loss_mean": float(np.mean(loss_win)) if loss_win else None, "accepted": int(acc[:u + 1].sum()),
                         "refused": int(u + 1 - acc[:u + 1].sum()), "rho_before": r_before, "rho_after": r_after, "projections": nproj,
                         "certified": bool(cc["certified"]), "steps": steps, "t": time.time() - t_start})
            loss_win = []
        if (u + 1) % CKPT_EVERY == 0 or u + 1 == Uj:
            tmp = ckpt.with_suffix(".tmp.npz")
            np.savez(tmp, eff=np.asarray(lr.brain.efficacy), bias=np.asarray(lr.brain.bias), u=u + 1, acc=acc, nproj=nproj,
                     cert_false=cert_false, logs=json.dumps(logs)); tmp.replace(ckpt)
            if (u + 1) % 2000 == 0: print(name, u + 1, logs[-1] if logs else "", flush=True)
    final_sha = C.sha(np.asarray(lr.brain.efficacy), np.asarray(lr.brain.bias)); cert1 = certificate(g)
    ev = eval_t1(g, d) if task == "T1" else eval_t2(g, d)
    rec = {"task": task, "arm": arm, "seed": seed, "H": H, "K": KF if task == "T2" else None, "tag": tag,
           "data_seed": 1000 + seed, "data_sha256": d["sha"], "init_seed": init, "init_reseed": reseed,
           "config": {"backend": "torch64", "threads": THREADS, "beta": lr.config.beta, "tolerance": TOL, "damping": 3, "qualified": True,
                      "eta": eta, "eta_bias": eta / 10, "normalize": 0.0, "U": Uj, "batch": BATCH, "temperature": TEMP if task == "T2" else None,
                      "nudge": lr.config.nudge, "dt": 0.5, "rho_init": RHO0, "cap": CAP, "eval_tol": EVAL_TOL, "ift_tol": IFT_TOL if arm == "ift" else None,
                      "a": np.asarray(g["model"].odd if hasattr(g["model"], "odd") else []).tolist(),
                      "b": np.asarray(g["model"].even if hasattr(g["model"], "even") else []).tolist(),
                      "target": "clip(0.1*x1*x2,-0.7,1.2)" if task == "T1" else "onehot"},
           "updates_attempted": int(Uj), "updates_accepted": int(acc.sum()), "accept_rate": float(acc.mean()) if Uj else None,
           "accept_by_batch": acc.tolist(), "refused_by_phase": refused_by, "projections": nproj, "cert_false_count": cert_false,
           "init_sha256": init_sha, "final_sha256": final_sha, "rho_init": rho_init, "rho_final": rho(g), "cert_init": cert0, "cert_final": cert1,
           "L_vector": g["lip"], "logs": logs, "checkpoint": str(ckpt), "wall_s": time.time() - t_start, "clock_at_start_s": clk, **ev}
    if task == "T2":
        rec.update({"M": d["M"], "D": d["D"], "mu": d["mu"], "uA": d["uA"], "uB": d["uB"], "mixing_draws": d["draws"],
                    "cos_ab_true": d["cos_ab_true"], "cond_M": d["cond"]})
        rec.update(oracle_t2(d))
    C.write_receipt(out_dir, name, rec, dev=dev)
    if not DRY: C.ledger("M12-S1", f"{task}_{arm}" + (f"_H{H}" if H else "") + f"_s{seed}_{tag}", 1.0 - rec["eval_unqualified"] / rec["eval_rows"], 1,
             rec["wall_s"], "done", f"S1 {tag} {task} {arm} seed {seed} U={Uj} eta={eta} (verdicts only in m12_verify --stage S1)")
    print("done", name, round(rec["wall_s"]), flush=True)


def minit():
    """B3b: initial T2 filter match per official seed, before any training; reseed init by +10 while m_init >= 0.8."""
    C.e_seal(); start_clock("minit")
    rec = {"T2": {}}
    for seed in (0, 1, 2):
        init, tries = t2_init_seed(seed); rec["T2"][str(seed)] = {"init_seed": init, "tries": tries, "replacements": len(tries) - 1}
    rec.update({"env": C.env_info(), "e_seal": C.e_seal(), "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rule": "m = max_{k!=k'} min(|cos(w_k,uA)|,|cos(w_k',uB)|) on the initial tied entry filters; init seed seed+10j until m_init < 0.8"})
    (S1 / "minit.json").write_text(json.dumps(rec, indent=1, default=float)); print(json.dumps(rec["T2"], default=float))


def smoke():
    """N3/N4: (a, b) = (1, 0.6), seed 100, 200 updates per (task, trainer), timing only; no judged quantity computed."""
    C.e_seal(); import torch; torch.set_num_threads(THREADS); start_clock("smoke")
    res = {}
    jobs = [("T1_H8_ep", 8), ("T1_H32_ep", 32), ("T2_ep", None), ("T2_ift", None)]
    for name, H in jobs:
        if H:
            d = t1_data(100); g = build_t1(H, 100, hid_ab=DEV, out_ab=DEV); xin = d["x_tr"]
        else:
            d = t2_data(100); g = build_t2(100, ab=DEV, out_ab=DEV); xin = d["in_tr"]
        acc, ts = 0, []
        for u in range(200):
            idx = np.random.default_rng(u).permutation(len(xin))[:BATCH]; D = drive(g, xin[idx]); t0 = time.perf_counter()
            try:
                if H: ep_step(g, D, target_array(g, d["t_tr"][idx]))
                elif name == "T2_ep": ep_step(g, D, g["lr"].targets(d["y_tr"][idx]))
                else:
                    ok, _ = ift_step(g, D, gs_ce(g, d["y_tr"][idx]))
                    if not ok: raise LearningPhaseError("ift")
                acc += 1
            except LearningPhaseError:
                pass
            project(g); ts.append(time.perf_counter() - t0)
        res[name] = {"t_u_median": float(np.median(ts)), "t_u_mean": float(np.mean(ts)), "accepted": acc, "updates": 200,
                     "rho_final": rho(g), "certified": certificate(g)["certified"], "finite": bool(np.isfinite(np.asarray(g["lr"].brain.efficacy)).all())}
        print(name, res[name], flush=True)
    m = {k: v["t_u_mean"] for k, v in res.items()}
    first = 6 * U_FULL * (m["T1_H8_ep"] + m["T1_H32_ep"]) / 2 + 6 * U_FULL * m["T2_ep"] + 3 * U_FULL * m["T2_ift"]
    fix = 48 * U_FULL * (m["T1_H8_ep"] + m["T1_H32_ep"] + m["T2_ep"]) + first
    res["extrapolation"] = {"first_round_job_hours": first / 3600, "first_round_wall_hours_4jobs": first / 3600 / 4,
                            "worst_fix_round_wall_hours_4jobs": fix / 3600 / 4, "limit_hours": CLOCK_LIMIT_S / 3600,
                            "over_0.6_week": bool((first + fix) / 3600 / 4 > 0.6 * 168)}
    C.write_receipt(S1 / C.run_commit() / "dev", "smoke", {"smoke": res, "neuron": DEV, "seed": 100}, dev=True)
    C.ledger("M12-S1", "smoke", "NA", 1, 0, "done", f"S1 dev smoke (1,0.6) seed 100, 200 updates each; timing only")
    print(json.dumps(res["extrapolation"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--task"); ap.add_argument("--arm"); ap.add_argument("--seed", type=int); ap.add_argument("--H", type=int)
    ap.add_argument("--eta", type=float, default=ETA); ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--tag", default="official"); ap.add_argument("--out", default=None); ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    if a.cmd == "minit": return minit()
    if a.cmd == "smoke": return smoke()
    if a.cmd == "run":
        out = a.out or str(S1 / C.run_commit() / a.tag); ck = a.ckpt or str(Path("/workspace/m12_ckpt/S1") / a.tag)
        run_job(a.task, a.arm, a.seed, H=a.H, eta=a.eta, beta=a.beta, tag=a.tag, out_dir=out, ckpt_dir=ck)


if __name__ == "__main__":
    main()
