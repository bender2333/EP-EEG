"""M13 E0 entry ticket: held-out electrode reconstruction (reports/M13.md section E0).

The v0 lake is the M12 lake as it stood at F1 (cadence fork m12: linear sources, smooth even/odd
neurons, weighted cascade certificate, symmetric row-mass projection to rho~ <= 0.9), laid out as
ADR-0002 microcolumns: each lead's two sources (value, hidden flag) feed only that lead's k columns,
columns are all-to-all reciprocal, 22 outlets (one per lead) are reciprocal with every column.
Trained by EP with a quadratic nudge to every lead's value under random hold-out masks.

Arms (one receipt per subject x seed, all arms inside it so they share rows and masks):
  lake_own         GATE (amendment 1): the EP-trained lake's own outlets at the hidden leads, inverse of the
                   fixed target map 3 artanh(s / 0.5), nothing fitted
  random_lake_own  GATE control: the same initial network's own outlets, no update (U = 0)
  spline           GATE control: spherical spline on the true positions (geometry only, nothing fitted)
  spline_shuf      validity: spherical spline on deranged positions (3 derangements, averaged)
  lake, random_lake  (descriptive since amendment 1) column states -> ridge readout fitted per pattern
  ridge_raw        (descriptive) ridge from the visible leads' values: the linear data-driven reference
  mean_visible     (descriptive) mean of the visible leads
Shuffled positions for the v0 lake are not a separate run: v0 reads no position, so a lake trained on
deranged leads is the same network relabelled (proved in test_m13.py, bit for bit).
"""
from __future__ import annotations
import argparse, os, sys, time, warnings
from pathlib import Path
import numpy as np

os.environ.setdefault("M12_THREADS", os.environ.get("M13_THREADS", "1"))
from . import m12_s1 as S                       # noqa: E402  (M12 toolchain: build, project, EP step, settle)
from . import m13_common as C                   # noqa: E402
from cadence.learning import LearningPhaseError  # noqa: E402

K = 4                      # columns per lead (ADR-0002)
N_SRC = 2 * C.N_LEADS      # [value_0..21 | hidden_0..21]
T_SCALE = 0.5              # outlet target t = 0.5 tanh(x/3), inside N1's (-0.75, 1.25)
U_E0, BATCH = 6000, 64     # fixed number of update attempts (pre-registered in M13 E0.4)
ETA, BETA = S.ETA, S.BETA  # 0.05, 0.04 (M12 first-round values)
EVAL_TOL = 1e-5
ALPHAS = np.logspace(-3, 4, 15)
OUT = C.ROOT / "receipts" / "M13" / "E0"


# ----------------------------------------------------------------------------- lake
def build_lake(init_seed, k=K, perm=None):
    """perm: optional lead relabelling (lead c is wired as lead perm[c]); used only by the equivariance test."""
    rng = np.random.default_rng(init_seed)
    L = C.N_LEADS; nc = L * k; ns = N_SRC
    cols = np.arange(ns, ns + nc); outs = np.arange(ns + nc, ns + nc + L); n = ns + nc + L
    lead_slot = np.arange(L) if perm is None else np.asarray(perm)
    pre, post, eff, ff = [], [], [], []
    w_in = rng.normal(0, 1, size=(L, 2, k))
    for c in range(L):
        q = cols[lead_slot[c] * k:(lead_slot[c] + 1) * k]
        for j, src in enumerate((c, L + c)):
            pre.append(np.full(k, src)); post.append(q); eff.append(w_in[lead_slot[c], j]); ff.append(np.zeros(k, bool))
    iu, ju = np.triu_indices(nc, 1); w = rng.normal(0, 1 / np.sqrt(nc), len(iu))
    pre += [cols[iu], cols[ju]]; post += [cols[ju], cols[iu]]; eff += [w, w]; ff += [np.ones(len(iu), bool)] * 2
    wo = rng.normal(0, 1 / np.sqrt(nc), size=(nc, L))            # column slot x outlet slot
    out_of_lead = outs[lead_slot]                                   # outlet neuron serving lead c
    P, Q = np.meshgrid(cols, outs, indexing="ij")
    # outlet slot j belongs to lead lead_slot^-1[j]; draw by slot so the relabelled net is the same net
    pre += [P.ravel(), Q.ravel()]; post += [Q.ravel(), P.ravel()]; eff += [wo.ravel(), wo.ravel()]; ff += [np.ones(P.size, bool)] * 2
    a = np.ones(n); b = np.zeros(n)
    a[cols], b[cols] = S.NEUR["N2"]; a[outs], b[outs] = S.NEUR["N1"]
    g = S.finish(n, ns, pre, post, eff, ff, None, a, b, out_of_lead, "quadratic", ETA, backend="cpu64")
    g.update(cols=cols, k=k, lead_slot=lead_slot)
    return g


def drive_rows(g, X, hidden_mask):
    """X (rows,22) z-scored values; hidden_mask (rows,22) bool. Hidden leads send value 0 and flag 1."""
    D = np.zeros((len(X), g["n"]))
    D[:, :C.N_LEADS] = np.where(hidden_mask, 0.0, np.tanh(X / 3.0))
    D[:, C.N_LEADS:N_SRC] = hidden_mask
    return D


def target(g, X):
    T = np.zeros((len(X), g["n"])); T[:, g["out"]] = T_SCALE * np.tanh(X / 3.0); return T


def masks_for(rng, n_rows, pats):
    idx = rng.integers(len(pats), size=n_rows); M = np.zeros((n_rows, C.N_LEADS), bool)
    for i, p in enumerate(idx): M[i, list(pats[p][1])] = True
    return M


def train_lake(g, Xtr, seed, U=U_E0, log=print):
    pats = C.all_patterns(); stats = {"accepted": 0, "refused": 0, "projections": 0, "log": []}
    nb = len(Xtr) // BATCH; t0 = time.time(); loss = []
    for u in range(U):
        ep, bi = divmod(u, nb)
        rng = np.random.default_rng((4000 + seed) * 100003 + ep)
        perm = rng.permutation(len(Xtr)); M = masks_for(rng, len(Xtr), pats)
        idx = perm[bi * BATCH:(bi + 1) * BATCH]
        D = drive_rows(g, Xtr[idx], M[idx]); T = target(g, Xtr[idx])
        try:
            free, steps = S.ep_step(g, D, T, ETA)
            s = np.asarray(free.activation)[:, g["out"]]; loss.append(float(0.5 * ((s - T[:, g["out"]]) ** 2).mean()))
            stats["accepted"] += 1
        except LearningPhaseError:
            stats["refused"] += 1
        if S.project(g): stats["projections"] += 1
        if S.rho(g) >= 1.0: raise RuntimeError("rho~ >= 1 after projection")
        if (u + 1) % 200 == 0:
            cc = S.certificate(g)
            stats["log"].append({"u": u + 1, "loss": float(np.mean(loss)) if loss else None, "rho": S.rho(g),
                                 "certified": bool(cc["certified"]), "t": time.time() - t0}); loss = []
            log(f"u={u + 1} loss={stats['log'][-1]['loss']} rho={stats['log'][-1]['rho']:.3f} refused={stats['refused']}")
    stats["wall_s"] = time.time() - t0; stats["certified_end"] = bool(S.certificate(g)["certified"])
    return stats


def lake_states(g, X, hid, chunk=2048):
    M = np.zeros((len(X), C.N_LEADS), bool); M[:, list(hid)] = True
    S_, R_, OK = [], [], []
    for i in range(0, len(X), chunk):
        eq = g["lr"].brain.equilibrate(drive_rows(g, X[i:i + chunk], M[i:i + chunk]), budget=S.BUDGET, tolerance=EVAL_TOL, damping=3)
        S_.append(np.asarray(eq.state.activation)); OK.append(np.asarray(eq.converged))
    A = np.concatenate(S_); return A[:, g["cols"]], A[:, g["out"]], np.concatenate(OK)


# ----------------------------------------------------------------------------- readouts
def ridge_fit_predict(Ftr, Ytr, Fte):
    from sklearn.linear_model import RidgeCV
    mu, sd = Ftr.mean(0), Ftr.std(0); sd = np.where(sd > 1e-12, sd, 1.0)
    m = RidgeCV(alphas=ALPHAS).fit((Ftr - mu) / sd, Ytr)
    return m.predict((Fte - mu) / sd).reshape(len(Fte), -1), float(m.alpha_)


def own_inverse(s):
    return 3.0 * np.arctanh(np.clip(s / T_SCALE, -0.999999, 0.999999))


# ----------------------------------------------------------------------------- one job
def run_job(subject, seed, dev=False, U=U_E0, out_dir=None, quick_patterns=None):
    t_start = time.time()
    data = C.synth_subject(subject) if dev else C.load_real(subject)
    Xtr = C.rows(data["train"]); Xrd = C.rows(data["train"], step=C.RIDGE_STEP); Xte = C.rows(data["test"])
    seal = C.e_seal()
    if seal["matches"]: raise RuntimeError(f"E files present: {seal['matches']}")
    init = 5000 + 100 * subject + seed
    g_rand = build_lake(init); g = build_lake(init)
    init_sha = C.sha(np.asarray(g["lr"].brain.efficacy))
    assert init_sha == C.sha(np.asarray(g_rand["lr"].brain.efficacy))
    cert0 = S.certificate(g)
    tr = train_lake(g, Xtr, seed, U=U, log=lambda m: print(f"[{subject}/{seed}] {m}", flush=True))
    pats = C.all_patterns() if quick_patterns is None else C.all_patterns()[:quick_patterns]
    perms = [C.derangement(i) for i in range(3)]
    res = []
    for fam, hid in pats:
        hid = list(hid); vis = [c for c in range(C.N_LEADS) if c not in hid]
        y = Xte[:, hid]; row = {"family": fam, "hidden": hid}
        Ftr_l, _, ok_tr = lake_states(g, Xrd, hid); Fte_l, Ote_l, ok_te = lake_states(g, Xte, hid)
        Ftr_r, _, okr_tr = lake_states(g_rand, Xrd, hid); Fte_r, Ote_r, okr_te = lake_states(g_rand, Xte, hid)
        p, a1 = ridge_fit_predict(Ftr_l, Xrd[:, hid], Fte_l); row["lake"] = C.r2_cols(p, y)
        p, a2 = ridge_fit_predict(Ftr_r, Xrd[:, hid], Fte_r); row["random_lake"] = C.r2_cols(p, y)
        row["lake_own"] = C.r2_cols(own_inverse(Ote_l[:, hid]), y)
        row["random_lake_own"] = C.r2_cols(own_inverse(Ote_r[:, hid]), y)
        row["spline"] = C.r2_cols(Xte[:, vis] @ C.spline_matrix(hid).T, y)
        row["spline_shuf"] = np.mean([C.r2_cols(Xte[:, vis] @ C.spline_matrix(hid, xyz=C.XYZ[pi]).T, y) for pi in perms], 0)
        p, a3 = ridge_fit_predict(Xrd[:, vis], Xrd[:, hid], Xte[:, vis]); row["ridge_raw"] = C.r2_cols(p, y)
        row["mean_visible"] = C.r2_cols(np.repeat(Xte[:, vis].mean(1, keepdims=True), len(hid), 1), y)
        row.update(alpha_lake=a1, alpha_random=a2, alpha_raw=a3,
                   qualified_lake=float(np.r_[ok_tr, ok_te].mean()), qualified_random=float(np.r_[okr_tr, okr_te].mean()),
                   unqualified_rows_lake=int((~np.r_[ok_tr, ok_te]).sum()), unqualified_rows_random=int((~np.r_[okr_tr, okr_te]).sum()))
        res.append(row)
        print(f"[{subject}/{seed}] {fam}{hid} lake={row['lake'].mean():.3f} rand={row['random_lake'].mean():.3f} "
              f"spline={row['spline'].mean():.3f} shuf={row['spline_shuf'].mean():.3f} raw={row['ridge_raw'].mean():.3f}", flush=True)
    summ = {}
    for arm in ("lake", "random_lake", "spline", "spline_shuf", "ridge_raw", "mean_visible", "lake_own", "random_lake_own"):
        for fam in ("P1", "P3"):
            summ[f"{arm}_{fam}"] = float(np.mean([r[arm].mean() for r in res if r["family"] == fam]))
    rec = {"stage": "M13", "step": "E0", "subject": subject, "seed": seed, "dev": dev, "data": data["source"],
           "data_sha256": C.sha(data["train"], data["test"]), "rows": {"train": len(Xtr), "ridge": len(Xrd), "test": len(Xte)},
           "init_seed": init, "init_sha256": init_sha, "final_sha256": C.sha(np.asarray(g["lr"].brain.efficacy), np.asarray(g["lr"].brain.bias)),
           "config": {"k": K, "U": U, "batch": BATCH, "eta": ETA, "beta": BETA, "tol": S.TOL, "eval_tol": EVAL_TOL, "cap": S.CAP,
                      "rho0": S.RHO0, "t_scale": T_SCALE, "neurons": {"cols": S.NEUR["N2"], "out": S.NEUR["N1"]}, "alphas": ALPHAS},
           "cert_init": cert0, "train": tr, "patterns": res, "summary": summ, "e_seal": seal, "env": C.env_info(),
           "wall_s": time.time() - t_start, "finished": C.now()}
    out_dir = Path(out_dir or (OUT / ("dev" if dev else "official")))
    C.write_receipt(out_dir / f"{'S' if dev else 'A'}{subject:02d}_s{seed}.json", rec)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dev", action="store_true"); ap.add_argument("--U", type=int, default=U_E0)
    ap.add_argument("--quick-patterns", type=int, default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if not a.dev and (a.U != U_E0 or a.quick_patterns is not None):
        sys.exit("official runs use the pre-registered U and all patterns")
    rec = run_job(a.subject, a.seed, dev=a.dev, U=a.U, out_dir=a.out, quick_patterns=a.quick_patterns)
    print({k: round(v, 4) for k, v in rec["summary"].items()}, f"wall {rec['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
