"""M14: the lake judged on motor-imagery decoding, end to end from a raw time window (reports/M14.md).

One EP network per sub-window row, everything trained by the same local EP rule (cadence Learner):

  sources   22 x P linear neurons: one sub-window of the frozen-layer EEG (no channel mixing, no band
            power), resampled 250 -> 62.5 Hz, P = 32 samples (512 ms). Source (p, c) = p * 22 + c.
  front     22 spatial-filter neurons per position. Source -> filter synapses are one-way and tied across
            time (cadence.learning.time_shared_entry): one learnable 22 x 22 matrix W, the same at every
            position. Init: W = g (I - 11^T / 22) (average reference, "car") or W = g I ("identity").
  lake      22 x K microcolumns (ADR-0002): the K columns of virtual channel k are reciprocal with that
            channel's P filter neurons (a learnable temporal receptive field each); columns all-to-all
            reciprocal; columns reciprocal with the 4 outlets. Columns carry an even component
            (s = tanh(v/2) + tanh(v/2)^2), so a column can square by itself.
  outlets   4, odd only (s = tanh(v/2)): a monotone readout in every arm, so any quadratic capacity sits
            in the lake or in the explicit square of arm C. Cross-entropy nudge, T = 0.1.

Arms
  A        car front -> lake -> outlets
  B        identity front -> lake -> outlets
  C        car front with even-only filter neurons (s = tanh(v/2)^2, a parameter-free square), one pool
           neuron per virtual channel holding the time mean (fixed reciprocal weights, not plastic),
           columns read the pools -> lake -> outlets
  nolake   car front -> outlets directly (outlets reciprocal with every filter neuron)
  C_nolake C's front (square + time mean) -> outlets directly (exploratory reference)
  randlake A with the lake frozen at its initial weights (filter and column<->outlet still learn)

Trial decision: mean outlet activity over the 11 sub-windows (frozen aggregation, eval/scoring.py).
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
from pathlib import Path

THREADS = int(os.environ.get("M14_THREADS", "1"))
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, str(THREADS))
import numpy as np  # noqa: E402  (after the thread caps)

import cadence as cd  # noqa: E402
from cadence.learning import (Learner, LearnerConfig, LearningPhaseError, evenodd_neuron_model,  # noqa: E402
                              time_shared_entry, tie_rates)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "receipts" / "M14"
NL = 22
P = 32                       # samples per sub-window at 62.5 Hz
N_SUB = 11
SUB_STARTS = [int(round(k * 15.625)) for k in range(N_SUB)]   # 0 .. 156, last window ends at 188
K = 4                        # columns per virtual channel
BETA, TOL, TEMP, CAP, RHO0 = 0.04, 3e-3, 0.01, 0.9, 0.6   # TEMP 0.1 (M12) left outlet differences ~1e-3 unread
RHO0_MAX = 0.8               # init row-mass ceiling after the per-pathway budgets
BUDGET, EVAL_TOL = 50000, 1e-5
BATCH = 64
DENSE_LIMIT = int(os.environ.get("M14_DENSE_LIMIT", "64"))   # small: scipy CSR transport (the net is ~2% dense)
SQ_ODD = 0.1                 # arm C filter neurons: s = 0.1 tanh(v/2) + tanh(v/2)^2
ARMS = ("A", "B", "C", "nolake", "C_nolake", "randlake")


# ----------------------------------------------------------------------------- data
def windows(seg):
    """(n, 22, 750) frozen-layer segments at 250 Hz -> (n, 11, P*22) rows, position-major (p * 22 + c)."""
    from scipy.signal import resample_poly
    x = resample_poly(seg.astype(np.float64), 1, 4, axis=-1)          # (n, 22, 188), anti-aliased
    W = np.stack([x[:, :, s:s + P] for s in SUB_STARTS], axis=1)       # (n, 11, 22, P)
    return np.ascontiguousarray(W.transpose(0, 1, 3, 2).reshape(len(seg), N_SUB, P * NL))


def load(subject, split):
    """split 'val': T runs 0-3 train / T runs 4-5 test (research loop).
    split 'gate': T runs 0-3 train / E session test (stage gate only, EPEEG_GATE=1)."""
    from ep_eeg.data import load_subject, load_subject_gate
    d = load_subject(subject) if split == "val" else load_subject_gate(subject)
    return {"Xtr": windows(d["train"]["seg"]), "ytr": np.asarray(d["train"]["y"]),
            "Xte": windows(d["val"]["seg"]), "yte": np.asarray(d["val"]["y"]),
            "seg_tr": d["train"]["seg"], "seg_te": d["val"]["seg"]}


# ----------------------------------------------------------------------------- network
PATHS = {"src": 0, "pool": 1, "read": 2, "lat": 3, "out": 4}
# Init budget per pathway: after drawing, each pathway is scaled so that the largest weighted row mass it brings
# to any neuron (either end of its reciprocal pairs) equals this number. One global scale over all free->free
# edges (the M12/M13 rule) let the 87 lateral column edges take most of every column's budget, and the signal
# fell ~30x per hop (filter 0.2 -> columns 0.007 -> outlets 0.0002 at init on A01).
BUDGET_INIT = {"pool": 0.3, "read": 0.5, "lat": 0.15, "out": 0.5}
G_FRONT = 2.0                # front-end gain: filter potentials ~ N(0, 1) on z-scored EEG


def _recip(E, a_idx, b_idx, w, plastic=True, path="lat"):
    Pm, Qm = np.meshgrid(a_idx, b_idx, indexing="ij")
    w = np.asarray(w, float).ravel()
    E["pre"] += [Pm.ravel(), Qm.ravel()]; E["post"] += [Qm.ravel(), Pm.ravel()]; E["eff"] += [w, w]
    E["ff"] += [np.ones(Pm.size, bool)] * 2; E["grp"] += [-np.ones(Pm.size, np.int64)] * 2
    E["pl"] += [np.full(Pm.size, plastic)] * 2; E["path"] += [np.full(2 * Pm.size, PATHS[path])]


def build(arm, init_seed, eta=0.05, g_front=G_FRONT, k=K, p=P, temp=None):
    temp = TEMP if temp is None else temp
    assert arm in ARMS, arm
    rng = np.random.default_rng(init_seed)
    ns = p * NL
    p0, q0, tie, entry = time_shared_entry(NL, p, NL)            # entry[kf][pos] -> neuron id (ns + kf*p + pos)
    front = "identity" if arm == "B" else "car"
    W0 = g_front * (np.eye(NL) if front == "identity" else np.eye(NL) - 1.0 / NL)
    E = {"pre": [p0], "post": [q0], "eff": [W0.ravel()[tie]], "ff": [np.zeros(len(p0), bool)],
         "grp": [tie.astype(np.int64)], "pl": [np.ones(len(p0), bool)], "path": [np.zeros(len(p0), int)]}
    nxt = ns + NL * p
    square = arm in ("C", "C_nolake")
    lake = arm not in ("nolake", "C_nolake")
    pools = None
    if square:
        pools = np.arange(nxt, nxt + NL); nxt += NL
        for kf in range(NL):                                         # fixed time mean: not plastic
            _recip(E, entry[kf], pools[kf:kf + 1], np.full(p, 1.0 / p), plastic=False, path="pool")
    reads = [pools[kf:kf + 1] for kf in range(NL)] if square else [entry[kf] for kf in range(NL)]
    fan = 1 if square else p
    cols = np.array([], int)
    if lake:
        nc = NL * k; cols = np.arange(nxt, nxt + nc); nxt += nc
        frozen = arm == "randlake"
        for kf in range(NL):
            _recip(E, reads[kf], cols[kf * k:(kf + 1) * k], rng.normal(0, 1 / np.sqrt(fan), (fan, k)), plastic=not frozen, path="read")
        iu, ju = np.triu_indices(nc, 1); w = rng.normal(0, 1 / np.sqrt(nc), len(iu))
        E["pre"] += [cols[iu], cols[ju]]; E["post"] += [cols[ju], cols[iu]]; E["eff"] += [w, w]
        E["ff"] += [np.ones(len(iu), bool)] * 2; E["grp"] += [-np.ones(len(iu), np.int64)] * 2
        E["pl"] += [np.full(len(iu), not frozen)] * 2; E["path"] += [np.full(2 * len(iu), PATHS["lat"])]
    out = np.arange(nxt, nxt + 4); nxt += 4; n = nxt
    if lake:
        _recip(E, cols, out, rng.normal(0, 1 / np.sqrt(len(cols)), (len(cols), 4)), path="out")
    else:
        src = np.concatenate(reads)
        _recip(E, src, out, rng.normal(0, 1 / np.sqrt(len(src)), (len(src), 4)), path="out")
    pre, post, eff, ff, grp, pl, path = (np.concatenate(E[key]) for key in ("pre", "post", "eff", "ff", "grp", "pl", "path"))
    conn = cd.Connectome.from_synapses(n, pre=pre, post=post)
    order = np.lexsort((pre, post)); assert np.array_equal(conn.pre, pre[order]) and np.array_equal(conn.post, post[order])
    eff, ff, grp, pl, path = eff[order], ff[order], grp[order], pl[order], path[order]
    odd = np.ones(n); even = np.zeros(n)
    ent = np.arange(ns, ns + NL * p)
    if square: odd[ent], even[ent] = SQ_ODD, 1.0     # cadence needs odd > 0: a near-pure square
    even[cols] = 1.0
    model = evenodd_neuron_model(odd, even, linear=range(ns), dt=0.5)
    lip = model.lipschitz_per_neuron(n)
    for name, b in BUDGET_INIT.items():
        m = path == PATHS[name]
        if m.any():
            r = np.bincount(conn.post, np.abs(eff) * lip[conn.pre] * m, minlength=n).max()
            eff = np.where(m, eff * b / r, eff)
    r = np.bincount(conn.post, np.abs(eff) * lip[conn.pre] * ff, minlength=n)[ns:].max()
    if r > RHO0_MAX: eff = np.where(ff, eff * RHO0_MAX / r, eff)
    brain = cd.NeuralGraph(conn, model, efficacy=eff, bias=np.zeros(n), dense_limit=DENSE_LIMIT)
    lc = LearnerConfig(beta=BETA, eta=eta, eta_bias=eta / 10, centered=True, free_steps=BUDGET, nudged_steps=BUDGET,
                       tolerance=TOL, nudge="cross_entropy", temperature=temp, normalize=0.0, normalize_floor=1e-3,
                       momentum=0.0, decay=0.0, scale_cap=1e9, qualified=True, damping=3, surprise=False)
    pn = np.zeros(n, bool); pn[ns:] = True
    if arm == "randlake": pn[cols] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = Learner(brain, list(out), config=lc, plastic_synapses=pl, plastic_neurons=pn, reciprocal=True,
                     tie_groups=grp, synapse_rate=tie_rates(grp))
    return {"lr": lr, "conn": conn, "model": model, "ff": ff, "pl": pl, "grp": grp, "lip": lip, "ns": ns, "n": n, "out": out,
            "entry": entry, "cols": cols, "pools": pools, "arm": arm, "front": front, "p": p, "tie": tie, "eta": eta,
            "path": path, "n_params": int(len(np.unique(grp[(grp >= 0) & pl])) + ((grp < 0) & pl).sum() // 2 + pn.sum())}


def front_matrix(g):
    """Current spatial filter W[k, c] (one tie group per entry)."""
    eff = np.asarray(g["lr"].brain.efficacy); grp = g["grp"]; Wm = np.zeros((NL, NL))
    first = {}
    for i, gid in enumerate(grp):
        if gid >= 0 and gid not in first: first[int(gid)] = i
    for gid, i in first.items(): Wm[gid // NL, gid % NL] = eff[i]
    return Wm


def rho(g):
    eff = np.asarray(g["lr"].brain.efficacy); c = g["conn"]
    return float(np.bincount(c.post, np.abs(eff) * g["lip"][c.pre] * g["ff"], minlength=g["n"])[g["ns"]:].max())


def project(g, cap=CAP):
    """Symmetric per-neuron weighted row-mass projection on plastic free->free edges (fixed edges keep their value)."""
    lr, c, ff, L, n, ns = g["lr"], g["conn"], g["ff"], g["lip"], g["n"], g["ns"]
    eff = np.asarray(lr.brain.efficacy, float).copy(); changed = False; scal = ff & g["pl"]
    for _ in range(100):
        inc = np.bincount(c.post, np.abs(eff) * L[c.pre] * ff, minlength=n); inc[:ns] = 0
        over = inc > cap * (1 - 1e-9)
        if not over.any(): break
        f = np.ones(n); f[over] = cap / inc[over] * (1 - 1e-6)
        eff = eff * np.where(scal, np.minimum(f[c.post], f[c.pre]), 1.0); changed = True
    if changed: lr.brain = lr.brain.with_parameters(efficacy=eff, bias=np.asarray(lr.brain.bias, float))
    return changed


def certificate(g):
    return cd.certificate(g["lr"].brain, sources=range(g["ns"])).to_dict()


def drive(g, X):
    D = np.zeros((len(X), g["n"])); D[:, :g["ns"]] = X; return D


def ep_step(g, X, y):
    """One EP update (free, +beta, -beta, centred contrast); raises LearningPhaseError if a phase is refused."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st, rep = g["lr"].step(drive(g, X), np.asarray(y))
    return st, rep


def settle(g, X, chunk=512, tol=EVAL_TOL):
    A, OK = [], []
    for i in range(0, len(X), chunk):
        eq = g["lr"].brain.equilibrate(drive(g, X[i:i + chunk]), budget=BUDGET, tolerance=tol, damping=3)
        a = np.asarray(eq.state.activation); A.append(a); OK.append(np.asarray(eq.converged) & np.all(np.isfinite(a), 1))
    return np.concatenate(A), np.concatenate(OK)


def evaluate(g, X, y, ytr):
    """X (n, 11, d): outlet activity per sub-window -> frozen aggregation -> kappa / acc."""
    from eval.scoring import aggregate, fallback_class, score
    n = len(X); A, ok = settle(g, X.reshape(n * N_SUB, -1))
    act = A[:, g["out"]].reshape(n, N_SUB, 4); ok = ok.reshape(n, N_SUB)
    pred, rej = aggregate(act, ok, fallback_class(ytr))
    s = score(y, pred, ok); s["pred"] = pred
    return s, A, ok


# ----------------------------------------------------------------------------- one job
def front_stats(g, W0):
    Wm = front_matrix(g); car = np.eye(NL) - 1.0 / NL
    def cos(a, b): return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    off = Wm[~np.eye(NL, dtype=bool)]
    return {"rel_change": float(np.linalg.norm(Wm - W0) / np.linalg.norm(W0)),
            "cos_car": cos(Wm, car), "cos_identity": cos(Wm, np.eye(NL)),
            "diag_mean": float(np.diag(Wm).mean()), "offdiag_mean": float(off.mean()), "offdiag_sd": float(off.std()),
            "row_sum_mean": float(Wm.sum(1).mean()), "row_sum_abs_mean": float(np.abs(Wm.sum(1)).mean())}


def save_weights(g, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, eff=np.asarray(g["lr"].brain.efficacy, float), bias=np.asarray(g["lr"].brain.bias, float))
    tmp.replace(path)


def load_weights(g, path):
    z = np.load(path); g["lr"].brain = g["lr"].brain.with_parameters(efficacy=z["eff"], bias=z["bias"]); return g


def train(g, data, U, seed, eval_every=250, log=print, wdir=None):
    from cadence.learning import LearningPhaseError as LPE
    X = data["Xtr"].reshape(-1, data["Xtr"].shape[-1]); y = np.repeat(data["ytr"], N_SUB)
    rng = np.random.default_rng(14000 + seed)
    W0 = front_matrix(g); nb = len(X) // BATCH
    st = {"accepted": 0, "refused": 0, "projections": 0, "curve": [], "rho": []}
    t0 = time.time(); perm = None
    for u in range(U):
        ep, bi = divmod(u, nb)
        if bi == 0: perm = rng.permutation(len(X))
        idx = perm[bi * BATCH:(bi + 1) * BATCH]
        try:
            ep_step(g, X[idx], y[idx]); st["accepted"] += 1
        except LPE:
            st["refused"] += 1
        if project(g): st["projections"] += 1
        if (u + 1) % eval_every == 0 or u + 1 == U:
            s, _, _ = evaluate(g, data["Xte"], data["yte"], data["ytr"])
            str_, _, _ = evaluate(g, data["Xtr"], data["ytr"], data["ytr"]) if (u + 1) % (4 * eval_every) == 0 or u + 1 == U else (None, 0, 0)
            row = {"u": u + 1, "acc": s["acc"], "kappa": s["kappa"], "settle_rate": s["settle_rate"],
                   "rejected_trials": s["rejected_trials"], "rho": rho(g), "t": time.time() - t0,
                   "train_acc": None if str_ is None else str_["acc"], "refused": st["refused"], **{"front_" + k: v for k, v in front_stats(g, W0).items()}}
            st["curve"].append(row)
            if wdir is not None: save_weights(g, Path(wdir) / f"u{u + 1}.npz")
            log(f"u={u + 1} acc={s['acc']:.3f} k={s['kappa']:.3f} train={row['train_acc']} rho={row['rho']:.3f} "
                f"refused={st['refused']} front_rel={row['front_rel_change']:.3f} cos_car={row['front_cos_car']:.3f} t={row['t']:.0f}")
    st["wall_s"] = time.time() - t0; st["certified_end"] = bool(certificate(g).get("certified"))
    return st


def run_job(subject, arm, seed, U, eta, split="val", out_dir=None, eval_every=250, temp=None):
    temp = TEMP if temp is None else temp
    import hashlib, subprocess
    t_start = time.time()
    if split == "val":
        from ep_eeg.m13_common import e_seal
        seal = e_seal()
        if seal["matches"]: raise RuntimeError(f"E files present during a research run: {seal['matches'][:3]}")
    else:
        seal = {"gate": True}
        if os.environ.get("EPEEG_GATE") != "1": raise PermissionError("gate split needs EPEEG_GATE=1")
    data = load(subject, split)
    init = 14500 + 100 * subject + seed
    g = build(arm, init, eta=eta, temp=temp); cert0 = certificate(g); W0 = front_matrix(g)
    wdir = OUT / "weights" / (f"{arm}_A{subject:02d}_s{seed}_eta{eta:g}_T{temp:g}" + os.environ.get("M14_WTAG", ""))  # exploratory runs keep their own weights
    tr = train(g, data, U, seed, eval_every=eval_every, log=lambda m: print(f"[{arm} A{subject:02d} s{seed}] {m}", flush=True), wdir=wdir)
    s, A, ok = evaluate(g, data["Xte"], data["yte"], data["ytr"])
    pred = s.pop("pred")
    git = lambda *a, cwd=ROOT: subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True).stdout.strip()
    import cadence
    rec = {"stage": "M14", "split": split, "subject": subject, "arm": arm, "seed": seed, "U": U, "eta": eta, "init_seed": init,
           "config": {"P": P, "fs": 62.5, "n_sub": N_SUB, "K": K, "beta": BETA, "tol": TOL, "eval_tol": EVAL_TOL, "temp": temp, "budget_init": BUDGET_INIT, "g_front": G_FRONT,
                      "cap": CAP, "rho0": RHO0, "batch": BATCH, "sq_odd": SQ_ODD, "front": g["front"], "n_params": g["n_params"],
                      "neurons": g["n"], "synapses": int(len(g["conn"].pre))},
           "cert_init": cert0, "train": tr, "test": s, "pred": pred, "y": data["yte"],
           "front_end": front_stats(g, W0), "front_W": front_matrix(g),
           "data_sha256": hashlib.sha256(np.ascontiguousarray(data["Xtr"]).tobytes() + np.ascontiguousarray(data["Xte"]).tobytes()).hexdigest(),
           "e_seal": seal, "env": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                                   "cadence_version": cadence.__version__,
                                   "cadence_commit": git("rev-parse", "HEAD", cwd=Path(cadence.__file__).resolve().parents[2])},
           "wall_s": time.time() - t_start, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(out_dir or (OUT / split)) / f"{arm}_A{subject:02d}_s{seed}_eta{eta:g}_T{temp:g}.json"
    from ep_eeg.m13_common import write_receipt
    write_receipt(out, rec)
    print(f"[{arm} A{subject:02d} s{seed}] DONE acc={s['acc']:.3f} kappa={s['kappa']:.3f} wall={rec['wall_s']:.0f}s -> {out}", flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True); ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--U", type=int, required=True)
    ap.add_argument("--eta", type=float, default=0.05); ap.add_argument("--split", default="val", choices=("val", "gate"))
    ap.add_argument("--out", default=None); ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--temp", type=float, default=TEMP)
    a = ap.parse_args(argv)
    run_job(a.subject, a.arm, a.seed, a.U, a.eta, a.split, a.out, a.eval_every, a.temp)


if __name__ == "__main__":
    main()
