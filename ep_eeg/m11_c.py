"""M11 step C (reports/M11.md §5 and amendment 1): is the certified network, driven by EEG, closer to EEG's
own dynamics than a parameter-matched linear dynamical system?

Data: ep_eeg.gdf_data.continuous (T session .gdf, causal 1-45 Hz, calibration-block EOG regression, 125 Hz,
z with MI runs 0-3 statistics, 1023 trials and block starts masked). MI runs 0-3 train, 4-5 validation.
Network (M9 structure): PartitionedTemporalPatchNet, 22 leads x k=2 microcolumns (H=44); mask A full,
mask B = own lead + constant port, mask C full 22x44 (one-step prediction of all leads). Input encoding
u = gain * tanh(z / 3) (+ constant port). h[t] = A tanh(h[t-1]) + B u[t]; prediction of z[t+1] = C tanh(h[t]).
Arms:
  var6            VAR(6) OLS with intercept (2926 parameters)              closed form
  var_best        VAR(p), p pooled-selected on runs 0-2 -> run 3          closed form (reference)
  res             untrained net (orthogonal A, radius 0.9) + ridge C       closed form
  lin_res         same masks, identity activation, u = z (+ port), ridge C (reference, IIR linear)
  ep_noproj/proj  EP gradient (observe, rate 0) -> Adam, from res (C = ridge), proj: rho(A) <= 0.95
  bptt_*_U        same net / init / crops / optimiser by BPTT, U updates (reference)
  bptt_*_2000     BPTT, 2000 updates (substrate verdict, amendment M1-7)
  lin_bptt_2000   identity-activation net by BPTT, 2000 updates (reference)
Metrics: closed-loop R^2 at h in {1, 5, 25, 125} (criterion R2_{1,5}); generation with innovations
N(0, Sigma_train) from 4 fixed seeded starts x 60 s (2 s discarded): LSD (2-40 Hz, dB), lagged-correlation
distance LCD (tau 0-32), each normalised by matched 4 x 60 s real train windows vs all usable val data
(10 draws): D_gen = (LSD / LSD_rr + LCD / LCD_rr) / 2; mu / beta peaks and 1/f slope as decomposition.
"""
from __future__ import annotations
import argparse, itertools, json, os, time, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.signal import welch
from .data import ROOT
from .m11_common import write_receipt, git, M11_OUT

OUT = M11_OUT / "C"
N_CH, K_COL, H, BURN, CROP, FS = 22, 2, 44, 25, 125, 125
HORIZONS, LAGS = (1, 5, 25, 125), (0, 1, 2, 4, 8, 16, 32)
GAINS, PORTS, P_GRID = (1.0, 3.0, 10.0), (0.0, 1.0, 5.0), (1, 2, 4, 6, 10, 16, 25)
OPT = dict(beta=0.05, batch=16, budget_s=7200.0, crash_s=14400.0, proj=0.95)
GEN_LEN, GEN_DROP, N_GEN = 7750, 250, 4
ALPHAS = np.logspace(-3, 3, 13)


# ---------------- data helpers ----------------
def data(subject):
    from .gdf_data import continuous
    return continuous(subject)


def encode(z, gain, port):
    """z (22, T) -> u (T, 22 [+1])."""
    u = gain * np.tanh(z.T / 3.0)
    return np.concatenate([u, np.full((len(u), 1), port)], 1) if port else u


def masks(port):
    B = np.zeros((H, N_CH + int(port > 0)), bool)
    for c in range(N_CH): B[c * K_COL:(c + 1) * K_COL, c] = True
    if port: B[:, N_CH] = True
    return {"A": np.ones((H, H), bool), "B": B, "C": np.ones((N_CH, H), bool)}


def new_net(seed, port, radius=0.9):
    from cadence.experimental import PartitionedTemporalPatchNet
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PartitionedTemporalPatchNet(N_CH + int(port > 0), H, N_CH, masks=masks(port), seed=seed, initial_radius=radius)


def params(net):
    P = net.parameters(); return {k: np.array(P[k], float, copy=True) for k in ("A", "B", "C")}


def states(P, U, act=np.tanh, h0=None):
    """Teacher-forced hidden states h[t] (T, H) over one run, h[t] = A act(h[t-1]) + B u[t]."""
    A, B = P["A"], P["B"]; h = np.zeros(H) if h0 is None else h0; out = np.empty((len(U), H)); Bu = U @ B.T
    for t in range(len(U)):
        h = A @ act(h) + Bu[t]; out[t] = h
    return out


def usable_pairs(ok, lo=0, hi=1):
    """Indices t with ok[t - lo .. t + hi] all True."""
    T = len(ok); c = np.r_[0, np.cumsum(~ok)]; t = np.arange(lo, T - hi)
    return t[(c[t + hi + 1] - c[t - lo]) == 0]


def ridge(X, Y):
    from sklearn.linear_model import RidgeCV
    r = RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(X, Y); return r.coef_, float(r.alpha_)


# ---------------- model classes (common interface: one-step prediction, closed-loop) ----------------
class VAR:
    def __init__(self, p): self.p = p

    def fit(self, runs):
        X, Y = [], []
        for z, ok, *_ in runs:
            t = usable_pairs(ok, self.p - 1, 1)
            X.append(np.hstack([z[:, t - k].T for k in range(self.p)] + [np.ones((len(t), 1))])); Y.append(z[:, t + 1].T)
        X, Y = np.vstack(X), np.vstack(Y); self.W = np.linalg.lstsq(X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ Y, rcond=None)[0]
        self.n_params = int(self.W.size); return self

    def init(self, z, ok, t):  # state at origins t: the last p samples
        return np.stack([z[:, t - k].T for k in range(self.p)], 1)   # (O, p, 22)

    def predict(self, st): return np.concatenate([st.reshape(len(st), -1), np.ones((len(st), 1))], 1) @ self.W

    def advance(self, st, x): return np.concatenate([x[:, None], st[:, :-1]], 1)

    def one_step_resid(self, runs):
        R = []
        for z, ok, *_ in runs:
            t = usable_pairs(ok, self.p - 1, 1); R.append(z[:, t + 1].T - self.predict(self.init(z, ok, t)))
        return np.vstack(R)


class Net:
    """M9-structure recurrent net with parameters P = {A, B, C}; act tanh (or identity for lin_*)."""
    def __init__(self, P, gain, port, linear=False):
        self.P, self.gain, self.port, self.linear = P, gain, port, linear
        self.act = (lambda x: x) if linear else np.tanh
        self.n_params = int(sum(np.count_nonzero(m) for m in masks(port).values()))

    def enc(self, z): return np.concatenate([z.T, np.full((z.shape[1], 1), self.port)], 1) if (self.linear and self.port) else (z.T if self.linear else encode(z, self.gain, self.port))

    def run_states(self, z): return states(self.P, self.enc(z), self.act)

    def fit_readout(self, runs):
        X, Y = [], []
        for z, ok, *_ in runs:
            S = self.act(self.run_states(z)); t = usable_pairs(ok, 0, 1); X.append(S[t]); Y.append(z[:, t + 1].T)
        self.P["C"], self.alpha = ridge(np.vstack(X), np.vstack(Y)); return self

    def init(self, z, ok, t, S=None):
        S = self.run_states(z) if S is None else S; return S[t]

    def predict(self, h): return self.act(h) @ self.P["C"].T

    def advance(self, h, x):
        u = self.enc(x.T)
        return self.act(h) @ self.P["A"].T + u @ self.P["B"].T

    def one_step_resid(self, runs):
        R = []
        for z, ok, *_ in runs:
            S = self.run_states(z); t = usable_pairs(ok, 0, 1); R.append(z[:, t + 1].T - self.predict(S[t]))
        return np.vstack(R)


# ---------------- metrics ----------------
def closed_loop(model, z, ok, origins, horizon=max(HORIZONS)):
    st = model.init(z, ok, origins); out = {}
    for k in range(1, horizon + 1):
        x = model.predict(st)
        if k in HORIZONS: out[k] = x
        st = model.advance(st, x)
    return out


def r2_sums(model, val_runs, step=25):
    """Per horizon and lead: SSE and SST over all origins (every `step` samples, window fully usable)."""
    acc = {h: {"sse": np.zeros(N_CH), "sum": np.zeros(N_CH), "sq": np.zeros(N_CH), "n": 0} for h in HORIZONS}
    for z, ok, *_ in val_runs:
        o = usable_pairs(ok, max(P_GRID), max(HORIZONS)); o = o[::step]
        with np.errstate(all="ignore"): pr = closed_loop(model, z, ok, o)
        for h in HORIZONS:
            y = z[:, o + h].T; a = acc[h]; a["sse"] += ((pr[h] - y) ** 2).sum(0); a["sum"] += y.sum(0); a["sq"] += (y ** 2).sum(0); a["n"] += len(o)
    res = {}
    for h, a in acc.items():
        sst = a["sq"] - a["sum"] ** 2 / a["n"]; r2c = 1 - a["sse"] / sst
        res[h] = {"r2": float(np.mean(r2c)) if np.all(np.isfinite(r2c)) else -np.inf, "sse": a["sse"].tolist(), "sst": sst.tolist(), "n": a["n"]}
    return res


def stretches(ok, min_len):
    c = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0])); return [(a, b) for a, b in zip(c[::2], c[1::2]) if b - a >= min_len]


def seg_stats(segs):
    """Welch PSD (2 s windows, 50% overlap) and lagged correlation matrices, length-weighted over segments."""
    Ps, Rs, w = [], [], []
    for X in segs:  # X (T, 22)
        f, P = welch(X, fs=FS, nperseg=250, noverlap=125, axis=0); Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
        Rs.append(np.stack([Xs[:len(Xs) - t].T @ Xs[t:] / (len(Xs) - t) for t in LAGS])); Ps.append(P); w.append(len(X))
    w = np.array(w, float) / sum(w)
    return f, sum(wi * P for wi, P in zip(w, Ps)), sum(wi * R for wi, R in zip(w, Rs))


def lsd(f, Pg, Pr):
    m = (f >= 2) & (f <= 40); d = 10 * np.log10(Pg[m] + 1e-30) - 10 * np.log10(Pr[m] + 1e-30)
    return float(np.mean(np.sqrt((d ** 2).mean(0))))


def lcd(Rg, Rr): return float(np.mean([np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(Rg, Rr)]))


def spectral_shape(f, P):
    """1/f slope on 2-7 U 30-40 Hz (log-log), mu (8-13) / beta (13-30) peak frequency and height above it, lead means."""
    lf, lp = np.log10(f[1:]), np.log10(P[1:] + 1e-30); ff = f[1:]; m = ((ff >= 2) & (ff <= 7)) | ((ff >= 30) & (ff <= 40))
    out = {"slope": [], "mu_f": [], "mu_h": [], "beta_f": [], "beta_h": []}
    for c in range(P.shape[1]):
        k, b = np.polyfit(lf[m], lp[m, c], 1); r = lp[:, c] - (k * lf + b); out["slope"].append(-k)
        for name, lo, hi in (("mu", 8, 13), ("beta", 13, 30)):
            sel = (ff >= lo) & (ff < hi); i = np.argmax(r[sel]); out[name + "_f"].append(float(ff[sel][i])); out[name + "_h"].append(float(r[sel][i]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def real_val_stats(val_runs):
    return seg_stats([z[:, a:b].T for z, ok, *_ in val_runs for a, b in stretches(ok, 500)])


def floor(train_runs, val_ref, subject, draws=10):
    """Matched real-vs-real baseline: 4 x 60 s random usable train windows vs all usable val data."""
    f, Pr, Rr = val_ref; rng = np.random.default_rng(10_000 + subject); L = GEN_LEN - GEN_DROP; ls, lc = [], []
    pool = [(z, a, b) for z, ok, *_ in train_runs for a, b in stretches(ok, L)]
    for _ in range(draws):
        segs = []
        for _ in range(N_GEN):
            z, a, b = pool[rng.integers(len(pool))]; s = rng.integers(a, b - L + 1); segs.append(z[:, s:s + L].T)
        _, Pg, Rg = seg_stats(segs); ls.append(lsd(f, Pg, Pr)); lc.append(lcd(Rg, Rr))
    return float(np.mean(ls)), float(np.mean(lc))


def generate(model, val_runs, Sigma, subject, seed):
    """Closed loop with innovations N(0, Sigma) from N_GEN seeded val origins (same origins for every model)."""
    rng_o = np.random.default_rng(20_000 + subject); rng_e = np.random.default_rng(30_000 + subject * 10 + seed)
    pool = [(i, t) for i, (z, ok, *_) in enumerate(val_runs) for t in usable_pairs(ok, max(P_GRID), 1)[::25]]
    picks = [pool[j] for j in rng_o.choice(len(pool), N_GEN, replace=False)]
    Lc = np.linalg.cholesky(Sigma + 1e-9 * np.eye(N_CH)); segs = []
    for i, t in picks:
        z, ok = val_runs[i][0], val_runs[i][1]; st = model.init(z, ok, np.array([t])); xs = np.empty((GEN_LEN, N_CH))
        with np.errstate(all="ignore"):
            for k in range(GEN_LEN):
                x = model.predict(st) + rng_e.normal(size=(1, N_CH)) @ Lc.T; xs[k] = x[0]; st = model.advance(st, x)
        if not np.all(np.isfinite(xs)): return None
        segs.append(xs[GEN_DROP:])
    return segs


def evaluate(model, runs, subject, seed, val_ref=None, fl=None):
    tr, va = runs[:4], runs[4:]; t0 = time.time()
    r2 = r2_sums(model, va); r15 = float(np.mean([r2[1]["r2"], r2[5]["r2"]]))
    Sigma = np.cov(model.one_step_resid(tr).T)
    val_ref = val_ref or real_val_stats(va); fl = fl or floor(tr, val_ref, subject)
    segs = generate(model, va, Sigma, subject, seed)
    out = {"r2": {str(h): v for h, v in r2.items()}, "R2_15": r15, "R2_all": float(np.mean([r2[h]["r2"] for h in HORIZONS])),
           "floor": {"lsd_rr": fl[0], "lcd_rr": fl[1]}, "flags": []}
    if not np.isfinite(r15): out["flags"].append("diverged_r2")
    if segs is None:
        out.update(D_gen=np.inf, flags=out["flags"] + ["diverged_generation"]); return out
    f, Pg, Rg = seg_stats(segs); _, Pr, Rr = val_ref; L1, L2 = lsd(f, Pg, Pr), lcd(Rg, Rr)
    out.update(LSD=L1, LCD=L2, D_gen=0.5 * (L1 / fl[0] + L2 / fl[1]), shape_gen=spectral_shape(f, Pg), shape_real=spectral_shape(f, Pr),
               psd_gen=Pg.mean(1).tolist(), psd_real=Pr.mean(1).tolist(), freqs=f.tolist(), eval_wall_s=time.time() - t0)
    return out


# ---------------- training ----------------
def crops(runs, rng, n, n_tr=4):
    pool = [(i, usable_pairs(ok, 0, CROP)) for i, (z, ok, *_) in enumerate(runs[:n_tr])]
    w = np.array([len(p) for _, p in pool], float); w /= w.sum(); out_u, out_y = [], []
    for _ in range(n):
        i = rng.choice(len(pool), p=w); s = pool[i][1][rng.integers(len(pool[i][1]))]; z = runs[i][0]
        out_u.append(z[:, s:s + CROP]); out_y.append(z[:, s + 1:s + CROP + 1].T)
    return out_u, np.stack(out_y)


def monitor_window(runs):
    z, ok = runs[4][0], runs[4][1]; a, b = stretches(ok, 60 * FS)[0]; return z[:, a:a + 60 * FS]


def mon_r2(model, zw):
    S = model.run_states(zw); y = model.predict(S[BURN:-1]); t = zw[:, BURN + 1:].T
    return float(np.mean(1 - ((y - t) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)))


def project(A, cap):
    r = float(np.abs(np.linalg.eigvals(A)).max()); return (A * (cap / r), True) if r > cap else (A, False)


def train_ep(net, model, runs, seed, U, proj, gain, port, deadline, lr):
    from .m9_temporal import Adam
    rng = np.random.default_rng(seed + 100); adam = Adam(lr); zw = monitor_window(runs); t0 = time.time()
    st = {"attempted": 0, "updated": 0, "reasons": {}, "projections": 0, "rho_max": 0.0, "curve": [], "stopped": "U"}
    every = max(1, U // 8)
    for it in range(U):
        if time.time() - t0 > deadline: st["stopped"] = "budget"; break
        zs, tgt = crops(runs, rng, OPT["batch"]); u = np.stack([encode(z, gain, port) for z in zs])
        free = net.imagine(u, state=np.zeros((len(u), H))).output; tgt = tgt.copy(); tgt[:, :BURN] = free[:, :BURN]
        net.reset(); r = net.observe(u, tgt, beta=OPT["beta"], rate=0.0)
        if r.updated:
            adam.step(net, r.delta)
            if proj:
                P = net.parameters(); A, did = project(np.asarray(P["A"]), OPT["proj"])
                if did: P["A"] = A; net.set_parameters(P); st["projections"] += 1
        st["attempted"] += 1; st["updated"] += int(r.updated); st["reasons"][r.reason] = st["reasons"].get(r.reason, 0) + 1
        st["rho_max"] = max(st["rho_max"], float(np.abs(np.linalg.eigvals(net.parameters()["A"])).max()))
        if (it + 1) % every == 0:
            model.P = params(net); st["curve"].append({"update": it + 1, "mon_r2": mon_r2(model, zw), "wall": time.time() - t0})
    model.P = params(net); st["train_wall_s"] = time.time() - t0
    st["accept_rate"] = st["updated"] / max(1, st["attempted"]); return st


def train_bptt(P0, runs, seed, U, proj, gain, port, linear, deadline, lr, n_tr=4, monitor=True):
    import torch
    torch.manual_seed(seed); torch.set_num_threads(1); m = masks(port)
    A = torch.tensor(P0["A"], requires_grad=True); B = torch.tensor(P0["B"], requires_grad=True); C = torch.tensor(P0["C"], requires_grad=True)
    mA, mB = torch.tensor(m["A"], dtype=torch.float64), torch.tensor(m["B"], dtype=torch.float64)
    act = (lambda x: x) if linear else torch.tanh; opt = torch.optim.Adam([A, B, C], lr=lr); rng = np.random.default_rng(seed + 100)
    model = Net({k: v.copy() for k, v in P0.items()}, gain, port, linear); zw = monitor_window(runs) if monitor else None; t0 = time.time()
    st = {"attempted": 0, "updated": 0, "projections": 0, "rho_max": 0.0, "curve": [], "stopped": "U"}; every = max(1, U // 8)
    for it in range(U):
        if time.time() - t0 > deadline: st["stopped"] = "budget"; break
        zs, tgt = crops(runs, rng, OPT["batch"], n_tr); u = torch.tensor(np.stack([model.enc(z) for z in zs])); tg = torch.tensor(tgt)
        Am, Bm = A * mA, B * mB; h = torch.zeros(len(u), H, dtype=torch.float64); ys = []
        for t in range(CROP):
            h = act(h) @ Am.T + u[:, t] @ Bm.T; ys.append(act(h) @ C.T)
        y = torch.stack(ys, 1); loss = 0.5 * ((y[:, BURN:] - tg[:, BURN:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            if proj:
                An, did = project((A * mA).numpy(), OPT["proj"])
                if did: A.copy_(torch.tensor(An)); st["projections"] += 1
            st["rho_max"] = max(st["rho_max"], float(np.abs(np.linalg.eigvals((A * mA).numpy())).max()))
        st["attempted"] += 1; st["updated"] += 1
        if monitor and (it + 1) % every == 0:
            model.P = {"A": (A * mA).detach().numpy().copy(), "B": (B * mB).detach().numpy().copy(), "C": C.detach().numpy().copy()}
            st["curve"].append({"update": it + 1, "mon_r2": mon_r2(model, zw), "wall": time.time() - t0})
    model.P = {"A": (A * mA).detach().numpy().copy(), "B": (B * mB).detach().numpy().copy(), "C": C.detach().numpy().copy()}
    st["train_wall_s"] = time.time() - t0; st["accept_rate"] = 1.0; return model, st


# ---------------- selection phases ----------------
def settings():
    p = OUT / "settings.json"; return json.loads(p.read_text()) if p.exists() else {}


def encode_score(subject, seed, gain, port):
    runs = data(subject)["runs"]; inner_tr, inner_va = runs[:3], [runs[3]]
    m = Net(params(new_net(seed, port)), gain, port).fit_readout(inner_tr); z, ok = inner_va[0][0], inner_va[0][1]
    S = m.run_states(z); t = usable_pairs(ok, 0, 1); y = m.predict(S[t]); tt = z[:, t + 1].T
    return float(np.mean(1 - ((y - tt) ** 2).sum(0) / ((tt - tt.mean(0)) ** 2).sum(0)))


def varp_score(subject, p):
    runs = data(subject)["runs"]; m = VAR(p).fit(runs[:3]); z, ok = runs[3][0], runs[3][1]; t = usable_pairs(ok, p - 1, 1)
    y = m.predict(m.init(z, ok, t)); tt = z[:, t + 1].T
    return float(np.mean(1 - ((y - tt) ** 2).sum(0) / ((tt - tt.mean(0)) ** 2).sum(0)))


LRS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)


def lr_score(subject, lr):
    """Amendment M2-1: BPTT from C_res on runs 0-2 crops, min(U, 200) updates, h=1 R^2 on run 3 (seed 0)."""
    s = settings(); gain, port = s["gain"], s["port"]; runs = data(subject)["runs"]
    base = Net(params(new_net(0, port)), gain, port).fit_readout(runs[:3])
    m, st = train_bptt(base.P, runs, 0, min(s["U"], 200), False, gain, port, False, 1e9, lr, n_tr=3, monitor=False)
    z, ok = runs[3][0], runs[3][1]; S = m.run_states(z); t = usable_pairs(ok, 0, 1); y = m.predict(S[t]); tt = z[:, t + 1].T
    with np.errstate(all="ignore"): r = float(np.mean(1 - ((y - tt) ** 2).sum(0) / ((tt - tt.mean(0)) ** 2).sum(0)))
    return r if np.isfinite(r) else -np.inf


def timing_job(subject):
    s = settings(); gain, port = s["gain"], s["port"]; runs = data(subject)["runs"]
    net = new_net(0, port); m = Net(params(net), gain, port).fit_readout(runs[:4]); P = net.parameters(); P["C"] = m.P["C"]; net.set_parameters(P)
    from .m9_temporal import Adam
    rng = np.random.default_rng(100); adam = Adam(0.003); ts = []  # timing only
    for _ in range(20):
        t0 = time.time(); zs, tgt = crops(runs, rng, OPT["batch"]); u = np.stack([encode(z, gain, port) for z in zs])
        free = net.imagine(u, state=np.zeros((len(u), H))).output; tgt[:, :BURN] = free[:, :BURN]
        net.reset(); r = net.observe(u, tgt, beta=OPT["beta"], rate=0.0)
        if r.updated: adam.step(net, r.delta)
        ts.append(time.time() - t0)
    return {"subject": subject, "per_update_s": ts}


# ---------------- main runs ----------------
def run_arm(arm, subject, seed):
    t0 = time.time(); s = settings(); gain, port, U = s.get("gain", 3.0), s.get("port", 5.0), s.get("U")
    budget, crash = s.get("budget_s", OPT["budget_s"]), s.get("crash_s", OPT["crash_s"])
    D = data(subject); runs = D["runs"]; info = {}; model = None
    try:
        if arm in ("var6", "var_best"):
            model = VAR(6 if arm == "var6" else s["var_p"]).fit(runs[:4])
        elif arm in ("res", "lin_res"):
            model = Net(params(new_net(seed, port)), gain, port, linear=arm == "lin_res").fit_readout(runs[:4])
        elif arm.startswith("ep_"):
            net = new_net(seed, port); model = Net(params(net), gain, port).fit_readout(runs[:4])
            P = net.parameters(); P["C"] = model.P["C"]; net.set_parameters(P)
            info["train"] = train_ep(net, model, runs, seed, U, arm == "ep_proj", gain, port, budget, s["lr"])
        else:  # bptt_{noproj,proj}_{U,2000}, lin_bptt_2000
            parts = arm.split("_"); linear = arm.startswith("lin_"); n_upd = U if parts[-1] == "U" else int(parts[-1])
            base = Net(params(new_net(seed, port)), gain, port, linear=linear).fit_readout(runs[:4])
            model, info["train"] = train_bptt(base.P, runs, seed, n_upd, "proj" in parts, gain, port, linear,
                                               budget if parts[-1] == "U" else 4 * OPT["budget_s"], s["lr"])
        if time.time() - t0 > crash and not arm.endswith("2000"): raise TimeoutError(f"over {crash:.0f} s")
        res = evaluate(model, runs, subject, seed)
    except Exception as e:  # amendment M1-6: a crash is a loss, flagged
        res = {"R2_15": -np.inf, "D_gen": np.inf, "flags": [f"crashed: {type(e).__name__}: {e}"]}
    A = model.P.get("A") if isinstance(model, Net) else None
    rec = {"stage": "M11", "step": "C", "arm": arm, "subject": subject, "seed": seed, **res, **info,
           "n_params": getattr(model, "n_params", None),
           "A_radius_final": float(np.abs(np.linalg.eigvals(A)).max()) if A is not None else None,
           "settings": s, "gdf_sha256": D["sha256"], "n_rejected_trials": D["n_rejected"], "calib_blocks": D["calib_blocks"],
           "settle_rate": info.get("train", {}).get("accept_rate", 1.0), "rejected": info.get("train", {}).get("attempted", 0) - info.get("train", {}).get("updated", 0),
           "wall_s": time.time() - t0}
    write_receipt(OUT / "runs" / arm, f"A{subject:02d}_s{seed}", rec, subject=subject, kind="gdf")
    return {"arm": arm, "subject": subject, "seed": seed, "R2_15": rec["R2_15"], "D_gen": rec["D_gen"], "flags": rec["flags"], "wall_s": rec["wall_s"]}


# ---------------- E3 (metric positive control, closed form) ----------------
def e3_job(subject, seed):
    s = settings(); gain, port = s["gain"], s["port"]; runs = data(subject)["runs"]; t0 = time.time()
    rng = np.random.default_rng(40_000 + subject * 10 + seed); out = {}
    teachers = {"a_var6": VAR(6).fit(runs[:4]), "b_res": Net(params(new_net(0, port)), gain, port).fit_readout(runs[:4])}
    for name, T in teachers.items():
        Sigma = np.cov(T.one_step_resid(runs[:4]).T); Lc = np.linalg.cholesky(Sigma + 1e-9 * np.eye(N_CH)); syn = []
        for z, ok, *rest in runs:  # same lengths and masks; warm start from the real run's first samples
            st = T.init(z, ok, np.array([max(P_GRID)])); xs = np.empty_like(z); xs[:, :max(P_GRID) + 1] = z[:, :max(P_GRID) + 1]
            for k in range(max(P_GRID) + 1, z.shape[1]):
                x = T.predict(st) + rng.normal(size=(1, N_CH)) @ Lc.T; xs[:, k] = x[0]; st = T.advance(st, x)
            syn.append((xs, ok, *rest))
        mu = np.hstack([r[0][:, r[1]] for r in syn[:4]]).mean(1, keepdims=True); sd = np.hstack([r[0][:, r[1]] for r in syn[:4]]).std(1, keepdims=True)
        syn = [((x - mu) / sd, ok, *r) for x, ok, *r in syn]
        fits = {"var6": VAR(6).fit(syn[:4]), "res": Net(params(new_net(seed + 1, port)), gain, port).fit_readout(syn[:4])}
        vr = real_val_stats(syn[4:]); fl = floor(syn[:4], vr, subject)
        out[name] = {k: {kk: v for kk, v in evaluate(m, syn, subject, seed, vr, fl).items() if kk in ("R2_15", "D_gen", "LSD", "LCD", "flags")}
                     for k, m in fits.items()}
    rec = {"stage": "M11", "step": "E3", "arm": "e3", "subject": subject, "seed": seed, **out, "settings": s, "wall_s": time.time() - t0,
           "settle_rate": 1.0, "rejected": 0}
    write_receipt(OUT / "e3", f"A{subject:02d}_s{seed}", rec, subject=subject, kind="gdf"); return rec


def _enc(a): return encode_score(*a)
def _varp(a): return varp_score(*a)
def _arm(a): return run_arm(*a)
def _lr(a): return lr_score(*a)
def _e3(a): return e3_job(*a)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True, choices=("integrity", "select", "timing", "lrsel", "main", "e3"))
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--arms", default=None); ap.add_argument("--subjects", default="1-9")
    ap.add_argument("--seeds", default="0,1,2"); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    lo, hi = map(int, a.subjects.split("-")); subs = range(lo, hi + 1); seeds = list(map(int, a.seeds.split(",")))
    with ProcessPoolExecutor(a.jobs) as ex:
        if a.phase == "integrity":
            from .gdf_data import integrity
            res = dict(zip(subs, ex.map(integrity, subs))); write_receipt(OUT, "integrity", {"per_subject": res}); print(json.dumps(res, indent=1))
        elif a.phase == "select":
            grid = list(itertools.product(GAINS, PORTS)); jobs = [(s, r, g, p) for g, p in grid for s in subs for r in (0, 1, 2)]
            sc = list(ex.map(_enc, jobs)); table = {f"{g}_{p}": float(np.mean([v for (s, r, gg, pp), v in zip(jobs, sc) if (gg, pp) == (g, p)])) for g, p in grid}
            best = max(grid, key=lambda gp: table[f"{gp[0]}_{gp[1]}"])
            vj = [(s, p) for p in P_GRID for s in subs]; vs = list(ex.map(_varp, vj))
            vt = {str(p): float(np.mean([v for (s, pp), v in zip(vj, vs) if pp == p])) for p in P_GRID}; vp = max(P_GRID, key=lambda p: vt[str(p)])
            st = {**settings(), "gain": best[0], "port": best[1], "var_p": vp}
            (OUT / "settings.json").write_text(json.dumps(st, indent=1))
            write_receipt(OUT, "select", {"encode_r2_h1_run3": table, "var_r2_h1_run3": vt, "chosen": st}); print(table, vt, st)
        elif a.phase == "timing":
            res = list(ex.map(timing_job, (1, 2, 3, 4))); tbar = float(np.mean([np.mean(r["per_update_s"]) for r in res]))
            # amendment M3-4 (human decision): if U < 200 at 7200 s, the EP run budget is 14,400 s (crash 28,800 s)
            budget, crash = OPT["budget_s"], OPT["crash_s"]; U0 = int(min(400, np.floor(budget / tbar)))
            if U0 < 200: budget, crash = 2 * budget, 2 * crash
            U = int(min(400, np.floor(budget / tbar))); st = {**settings(), "U": U, "t_bar_s": tbar, "U_at_7200s": U0, "budget_s": budget, "crash_s": crash,
                                                               "low_power": U < 200}
            (OUT / "settings.json").write_text(json.dumps(st, indent=1)); write_receipt(OUT, "timing", {"per_process": res, "t_bar_s": tbar, "U": U, "U_at_7200s": U0,
                                                                                                    "budget_s": budget, "crash_s": crash})
            print("t_bar", tbar, "U", U, "budget", budget)
        elif a.phase == "lrsel":
            jobs = [(s_, lr) for lr in LRS for s_ in subs]; sc = list(ex.map(_lr, jobs))
            table = {str(lr): float(np.mean([v for (s_, l), v in zip(jobs, sc) if l == lr])) for lr in LRS}
            best = max(LRS, key=lambda lr: table[str(lr)]); st = {**settings(), "lr": best}
            (OUT / "settings.json").write_text(json.dumps(st, indent=1))
            write_receipt(OUT, "lrsel", {"r2_h1_run3": table, "per_job": [list(j) + [v] for j, v in zip(jobs, sc)], "chosen_lr": best}); print(table, best)
        elif a.phase == "main":
            jobs = [(arm, s, r) for arm in a.arms.split(",") for r in seeds for s in subs]
            done = {(p.parent.name, p.stem) for p in (OUT / "runs").glob("*/*.json")} if (OUT / "runs").exists() else set()
            jobs = [j for j in jobs if (j[0], f"A{j[1]:02d}_s{j[2]}") not in done]
            for r in ex.map(_arm, jobs):
                print(f"{r['arm']} A{r['subject']:02d} s{r['seed']} R2_15 {r['R2_15']:.4f} D_gen {r['D_gen']:.3f} {r['flags']} {r['wall_s']:.0f}s", flush=True)
        else:
            for r in ex.map(_e3, [(s, r) for s in subs for r in seeds]):
                print(r["subject"], r["seed"], {k: {kk: round(vv["R2_15"], 4) for kk, vv in v.items()} for k, v in r.items() if k in ("a_var6", "b_res")}, flush=True)


if __name__ == "__main__":
    main()
