"""M13 step 4.5 (route 2): the lake learns IN TIME (reports/M13.md §6).

The lake runs as m13_dyn's ``dyn1`` arm (cadence.experimental.InertialLake, first order, tau 2 ms, dt 1 ms,
4 substeps per 4 ms sample, every trial starts from the static equilibrium of its first sample), and is trained
on that same trajectory, so the learning sees the lag that the static-trained v0 lake does not know about.
Lakes are built exactly as in route 1 (m13_learn.build: E0 checkpoint, plus the fixed outlet field for
field/der), so field / der / none are the same three conditions.

Discretisation (the eval's own, InertialLake.run): at sample t the source rows are set to the drive, the outlets are
read (loss on the read), then 4 substeps  v <- v - a r(v)  on the free neurons, a = dt/tau = 0.5,
r(v) = v - s(v) W - b (W = brain.dense(), [pre, post]; the field is part of W).

Loss per window: mean over windows and loss samples (samples >= BURN) of 1/2 sum_outlets (s - t)^2, t = E0's target
0.5 tanh(x/3) for every lead, input = E0's drive (hidden leads: value 0, flag 1), one P3/E0 mask per window.

Two update rules, both reported in EP coordinates (ep_eeg.ift.ep_equivalent, what E0's centred EP contrast estimates):
  bptt   exact gradient of the window loss through the substeps (adjoint backwards in time; not local, not EP)
  tep    local, causal "EP in time": two nudged trajectories (outlets pulled toward / pushed from the target with
         strength beta during each substep) run forward next to the free one from the same start; the contrast
         -(1/2beta) sum_steps a (s+_i s+_j - s-_i s-_j), normalised like the loss. In the static limit (input held)
         it is the centred EP contrast of the static lake.
Which rule trains route 2 is decided by the gradient check (§6.1, ``--check``), before any training run.
All training and testing within one subject.
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from . import m12_s1 as S                        # noqa: E402
from . import m13_common as C                    # noqa: E402
from . import m13_e0 as E                        # noqa: E402
from . import m13_dyn as M                       # noqa: E402
from . import m13_learn as Lm                    # noqa: E402
from .ift import ep_equivalent                   # noqa: E402
import cadence as cd                             # noqa: E402

OUT = C.ROOT / "receipts" / "M13" / "T"
A = M.DT / M.TAU_FAST          # 0.5
SUB = M.SUB                    # 4
WIN, BURN, BATCH = 100, 25, 16 # 400 ms windows, loss from 100 ms on, 16 windows per update
BETA = 0.04                    # E0's beta
EVAL_AT = (500, 1000, 2000)


# ----------------------------------------------------------------------------- the lake as arrays
class Arr:
    def __init__(self, g):
        b = g["lr"].brain
        self.W = np.asarray(b.dense(), float); self.b = np.asarray(b.bias, float); self.model = b.neuron_model
        self.n = g["n"]; self.ns = g["ns"]; self.out = np.asarray(g["out"])
        self.free = np.arange(self.ns, self.n)

    def act(self, v): return self.model.activation(v)
    def slope(self, v): return self.model.slope_at(v)

    def rest(self, d0, iters=4000, tol=1e-10):
        v = np.zeros((len(d0), self.n)); v[:, :self.ns] = d0
        for _ in range(iters):
            r = v - self.act(v) @ self.W - self.b; r[:, :self.ns] = 0.0; v -= 0.5 * r
            if np.abs(r).max() < tol: break
        return v


def windows(rng, seg, n):
    """n random (trial, start) windows of WIN samples from (trials, 22, 750) z-scored segments, with one E0 mask each."""
    tr = rng.integers(len(seg), size=n); st = rng.integers(seg.shape[2] - WIN + 1, size=n)
    X = np.stack([seg[i, :, s:s + WIN] for i, s in zip(tr, st)]).transpose(2, 0, 1)    # (WIN, n, 22)
    pats = C.all_patterns(); Mk = E.masks_for(rng, n, pats)                                 # (n, 22)
    D = np.zeros((WIN, n, 2 * C.N_LEADS))
    D[:, :, :C.N_LEADS] = np.where(Mk[None], 0.0, np.tanh(X / 3.0)); D[:, :, C.N_LEADS:] = Mk[None]
    return D, E.T_SCALE * np.tanh(X / 3.0)                                                 # drive, outlet targets


def forward(L, D, tgt=None, beta=0.0, keep=False, v0=None, shift=0):
    """Returns (reads (T,B,22), list of pre-substep states if keep, loss, contrast accumulators if beta != 0)."""
    T, B, _ = D.shape
    v = L.rest(D[0]) if v0 is None else v0.copy()
    reads, hist = [], []; loss = 0.0; nl = 0
    cW = np.zeros_like(L.W) if beta else None; cb = np.zeros(L.n) if beta else None
    for t in range(T):
        v[:, :L.ns] = D[t]
        s = L.act(v); y = s[:, L.out]; reads.append(y)
        if tgt is not None and t >= BURN:
            loss += 0.5 * ((y - tgt[t]) ** 2).sum(1).mean(); nl += 1
        for _ in range(SUB):
            if keep: hist.append(v.copy())
            s = L.act(v); r = v - s @ L.W - L.b
            if beta:
                tn = min(t + shift, T - 1)
                r[:, L.out] += beta * (s[:, L.out] - tgt[tn])
                if tn >= BURN:
                    cW += A * (s.T @ s) / B; cb += A * s.mean(0)
            r[:, :L.ns] = 0.0; v = v - A * r
    return np.stack(reads), hist, (loss / max(nl, 1)), (cW, cb)


def bptt(L, D, tgt):
    """Exact gradient of the window loss: dL/dW (dense, [pre,post]) and dL/db. Initial rest state: no gradient."""
    T, B, _ = D.shape
    reads, hist, loss, _ = forward(L, D, tgt, keep=True)
    nl = T - BURN; gW = np.zeros_like(L.W); gb = np.zeros(L.n)
    mu = np.zeros((B, L.n)); Wf = L.W.copy(); Wf[:, :L.ns] = 0.0          # only free posts move
    for t in range(T - 1, -1, -1):
        for k in range(SUB - 1, -1, -1):
            v = hist[t * SUB + k]; s = L.act(v); sp = L.slope(v)
            mf = mu.copy(); mf[:, :L.ns] = 0.0
            gW += A * (s.T @ mf); gb += A * mf.sum(0)
            mu = (1 - A) * mf + A * sp * (mf @ Wf.T)
            mu[:, :L.ns] = 0.0
        if t >= BURN:                                                   # read at t: before the substeps of t
            v = hist[t * SUB]; sp = L.slope(v)
            mu[:, L.out] += (reads[t] - tgt[t]) * sp[:, L.out] / (B * nl)
        mu[:, :L.ns] = 0.0
    return gW, gb, loss


def tep(L, D, tgt, beta=BETA, shift=0):
    """Local causal EP in time: centred contrast of two nudged forward trajectories, as an estimate of dL/dW."""
    T, B, _ = D.shape
    v0 = L.rest(D[0]); nl = T - BURN
    _, _, _, (pW, pb) = forward(L, D, tgt, beta=beta, v0=v0, shift=shift)
    _, _, _, (mW, mb) = forward(L, D, tgt, beta=-beta, v0=v0, shift=shift)
    # energy -1/2 sum_ij s_i W_ij s_j - b.s: dE/dW_ij = -1/2 s_i s_j per directed synapse (its twin gets the
    # other half; ep_equivalent adds the two); contrast estimates d(loss)/dtheta
    gW = -(pW - mW) / (4 * beta) / (nl * SUB * A)
    gb = -(pb - mb) / (2 * beta) / (nl * SUB * A)
    return gW, gb


def to_ep(g, gW, gb):
    c = g["conn"]; return ep_equivalent(g["lr"], gW[c.pre, c.post], gb)


def plastic_cos(g, a, b):
    m = (~np.asarray(g["fx"], bool)) & np.asarray(g["ff"] | True, bool)
    m &= np.asarray(g["conn"].post) >= g["ns"]
    return float(a[m] @ b[m] / (np.linalg.norm(a[m]) * np.linalg.norm(b[m]) + 1e-30))


# ----------------------------------------------------------------------------- gradient check (§6.1)
def check(subject, seed, n_batches=8, beta=BETA):
    """cos(tep, bptt) on real training windows at the route-1 start lake; also the static-limit and a finite
    difference sanity check of bptt itself."""
    data = C.load_real(subject, split="t04_val45"); seal = C.e_seal()
    if seal["matches"]: raise RuntimeError("E files present")
    g, _ = Lm.build(subject, seed, "none"); L = Arr(g)
    rng = np.random.default_rng(777 + subject)
    rows = []
    for k in range(n_batches):
        D, tg = windows(rng, data["train"], BATCH)
        gW, gb, loss = bptt(L, D, tg); tW, tb = tep(L, D, tg, beta)
        ce_b, cb_b = to_ep(g, gW, gb); ce_t, cb_t = to_ep(g, tW, tb)
        ce_1, _ = to_ep(g, *tep(L, D, tg, beta, shift=1))
        rows.append({"loss": float(loss), "cos_w": plastic_cos(g, ce_b, ce_t), "cos_w_shift1": plastic_cos(g, ce_b, ce_1),
                     "cos_b": float(cb_b[g["ns"]:] @ cb_t[g["ns"]:] / (np.linalg.norm(cb_b[g["ns"]:]) * np.linalg.norm(cb_t[g["ns"]:]) + 1e-30)),
                     "norm_ratio": float(np.linalg.norm(ce_t) / (np.linalg.norm(ce_b) + 1e-30))})
    # static limit: input held constant (one sample repeated), bptt and tep vs the static IFT gradient
    D, tg = windows(rng, data["train"], BATCH); Dh = np.repeat(D[:1], WIN, 0); th = np.repeat(tg[:1], WIN, 0)
    gW, gb, _ = bptt(L, Dh, th); tW, tb = tep(L, Dh, th, beta)
    Dst = np.zeros((BATCH, g["n"])); Dst[:, :g["ns"]] = Dh[0]
    Tst = np.zeros((BATCH, g["n"])); Tst[:, g["out"]] = th[0]
    ce_i, cb_i, _, _ = S.ift_grad(g, Dst, S.gs_quad(dict(g, out=[g["out"]]), th[0]))
    ce_b, _ = to_ep(g, gW, gb); ce_t, _ = to_ep(g, tW, tb)
    static = {"cos_bptt_ift": plastic_cos(g, ce_b, ce_i), "cos_tep_ift": plastic_cos(g, ce_t, ce_i),
              "ratio_bptt_ift": float(np.linalg.norm(ce_b) / np.linalg.norm(ce_i)),
              "ratio_tep_ift": float(np.linalg.norm(ce_t) / np.linalg.norm(ce_i))}
    # finite difference of bptt on 6 random plastic synapses (with twin, as ep coords do)
    D, tg = windows(rng, data["train"], 4); gW, gb, l0 = bptt(L, D, tg)
    c = g["conn"]; rev = g["lr"].reverse; pl = np.nonzero((~g["fx"]) & (c.post >= g["ns"]) & (c.pre >= g["ns"]))[0]
    fd = []
    for e in rng.choice(pl, 6, replace=False):
        i, j = c.pre[e], c.post[e]; h = 1e-5
        Wp = L.W.copy(); Wp[i, j] += h; Wp[j, i] += h
        Lp = Arr(g); Lp.W = Wp; lp = forward(Lp, D, tg)[2]
        Wm = L.W.copy(); Wm[i, j] -= h; Wm[j, i] -= h
        Lm_ = Arr(g); Lm_.W = Wm; lm = forward(Lm_, D, tg)[2]
        fd.append({"fd": (lp - lm) / (2 * h), "bptt": float(gW[i, j] + gW[j, i])})
    return {"stage": "M13", "step": "4.5-route2-check", "subject": subject, "seed": seed, "beta": beta,
            "batches": rows, "median_cos_w": float(np.median([r["cos_w"] for r in rows])),
            "median_cos_w_shift1": float(np.median([r["cos_w_shift1"] for r in rows])),
            "static_limit": static, "finite_diff": fd, "e_seal": seal, "env": C.env_info(), "finished": C.now()}


# ----------------------------------------------------------------------------- training
def train(g, seg, seed, U, rule, eta, log=print, on_eval=None):
    stats = {"rule": rule, "eta": eta, "log": [], "projections": 0}
    t0 = time.time(); loss = []
    for u in range(U):
        rng = np.random.default_rng((6000 + seed) * 100003 + u)
        D, tg = windows(rng, seg, BATCH); L = Arr(g)
        if rule == "bptt":
            gW, gb, l = bptt(L, D, tg)
        else:
            gW, gb = tep(L, D, tg); l = None
        ce, cb = to_ep(g, gW, gb)
        g["lr"].apply(eta * ce, (eta / 10) * cb)
        if Lm.project(g): stats["projections"] += 1
        if Lm.rho(g) >= 1.0: raise RuntimeError("rho~ >= 1 after projection")
        if l is not None: loss.append(l)
        if (u + 1) % 100 == 0:
            stats["log"].append({"u": u + 1, "loss": float(np.mean(loss)) if loss else None, "rho": Lm.rho(g), "t": time.time() - t0}); loss = []
            log(f"u={u + 1} loss={stats['log'][-1]['loss']} rho={Lm.rho(g):.3f}")
        if on_eval is not None and (u + 1) in EVAL_AT: on_eval(u + 1)
    stats["certified_end"] = bool(cd.certificate(g["lr"].brain, sources=range(g["ns"])).certified)
    stats["wall_s"] = time.time() - t0
    return stats


def evaluate(g, subject, split, static=True):
    r = {"dyn1": M.run_job(subject, 0, "dyn1", {}, split=split, lake_g=g)}
    if static: r["static"] = M.run_job(subject, 0, "static", {}, split=split, lake_g=g)
    return r


def summ(ev):
    out = {f"dyn1_{k}": v for k, v in ev["dyn1"]["summary"].items()}
    if "static" in ev: out.update({f"static_{k}": v for k, v in ev["static"]["summary"].items()})
    return out


def run(subject, seed, cond, split, U, rule, eta, ell=0.03, lam=0.5, der=0):
    t0 = time.time()
    data = C.load_real(subject, split="t04_val45" if split == "test" else "dev_r02_r3")
    seal = C.e_seal()
    if seal["matches"]: raise RuntimeError(f"E files present: {seal['matches']}")
    g, init_sha = Lm.build(subject, seed, cond, ell, lam, der=der)
    evals = {0: summ(evaluate(g, subject, split))}
    def on_eval(u):
        evals[u] = summ(evaluate(g, subject, split, static=False))
        print(f"[{subject}/{seed}/{cond}] eval u={u} dyn1 {evals[u]['dyn1_lake_own_P3']:.4f}", flush=True)
    tr = train(g, data["train"], seed, U, rule, eta, log=lambda m: print(f"[{subject}/{seed}/{cond}] {m}", flush=True), on_eval=on_eval)
    final = evaluate(g, subject, split)
    return {"stage": "M13", "step": "4.5-route2", "cond": cond, "split": split, "subject": subject, "seed": seed, "U": U,
            "rule": rule, "eta": eta, "window": {"WIN": WIN, "BURN": BURN, "BATCH": BATCH, "a": A, "sub": SUB},
            "field": {"ell": ell, "lam": lam, "derangement": der if cond == "der" else None},
            "data": data["source"], "init_sha256": init_sha,
            "final_sha256": C.sha(np.asarray(g["lr"].brain.efficacy), np.asarray(g["lr"].brain.bias)),
            "train": tr, "evals": {str(k): v for k, v in evals.items()}, "summary": summ(final),
            "patterns_dyn1": final["dyn1"]["patterns"], "e_seal": seal, "env": C.env_info(), "wall_s": time.time() - t0,
            "finished": C.now()}, g


def tag(cond, rule, eta, U, ell, lam, der):
    f = "none" if cond == "none" else f"{cond}{der if cond == 'der' else ''}_ell{ell:g}_lam{lam:g}"
    return f"{f}_{rule}_eta{eta:g}_U{U}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cond", choices=["field", "der", "none"]); ap.add_argument("--der", type=int, default=0)
    ap.add_argument("--split", default="dev", choices=["dev", "test"]); ap.add_argument("--U", type=int, default=2000)
    ap.add_argument("--rule", default="bptt", choices=["bptt", "tep"]); ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--ell", type=float, default=0.03); ap.add_argument("--lam", type=float, default=0.5)
    a = ap.parse_args(argv)
    if a.check:
        rec = check(a.subject, a.seed); path = OUT / "check" / f"A{a.subject:02d}_s{a.seed}.json"
        C.write_receipt(path, rec)
        print(path.name, rec["median_cos_w"], json.dumps(rec["static_limit"]), json.dumps(rec["finite_diff"][:2]), flush=True)
        return
    t = tag(a.cond, a.rule, a.eta, a.U, a.ell, a.lam, a.der)
    path = OUT / a.split / f"A{a.subject:02d}_s{a.seed}_{t}.json"
    if path.exists(): print(path.name, "exists"); return
    rec, g = run(a.subject, a.seed, a.cond, a.split, a.U, a.rule, a.eta, a.ell, a.lam, a.der)
    C.write_receipt(path, rec)
    wdir = OUT / "weights" / a.split; wdir.mkdir(parents=True, exist_ok=True)
    np.savez(wdir / f"{path.stem}.npz", eff=np.asarray(g["lr"].brain.efficacy, float), bias=np.asarray(g["lr"].brain.bias, float),
             pre=g["conn"].pre, post=g["conn"].post, fx=g["fx"])
    print(path.name, {k: round(v, 4) for k, v in rec["summary"].items() if isinstance(v, float) and "lake_own" in k},
          f"wall {rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
