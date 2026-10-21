"""M11 step B (reports/M11.md §4): representation-ceiling audit of each substrate with an oracle readout.

Per substrate / subject / seed: extract states on T runs 0-3 (train) and 4-5 (val), then run the readout
grid {F_mean, F_cat (+ F_pow for temporal substrates, M1-2)} x {ridge (15 alpha), shrinkage LDA (auto + 6 fixed), L2 logistic (15 C)}:
  kappa_cv  configuration chosen by 5-fold stratified CV on the 192 training trials (ties: first in grid
            order), refit on all training trials, scored once on val (honest);
  kappa_or  max over the whole grid of val kappa (uses val labels; deliberately optimistic upper bound).
input_track2 additionally gets a non-linear reference (RBF-SVM, kNN), same two numbers, not a criterion.
Trained static lakes: seed 0 = stored weights (receipts/M9c/readout_diag), seeds 1-2 retrained by the
same code (train_sandbox, M6 schedule); seed 0 is also retrained to check reproduction (max |d eff|).
"""
from __future__ import annotations
import argparse, itertools, json, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.metrics import cohen_kappa_score
from .data import load_subject, ROOT
from .m11_common import write_receipt, git, M11_OUT, run_commit, existing

OUT = M11_OUT / "B"
M6 = dict(leak=1.0, rowmass_project=0.9, eta=0.5, temperature=0.01, epochs=100)
M64 = {**M6, "sigma_in": 0.1 * 64, "cert_free_only": True, "scale_cap": 192.0}
STORED = {"m6_trained": ROOT / "receipts/M9c/readout_diag/bc4234e",
          "m64_trained": ROOT / "receipts/M9c/readout_diag/8a9f8c4_sigma_in6.4_cert_free_onlyTrue_scale_cap192.0"}
STATIC = {"random_lake": M6, "m6_trained": M6, "m64_random": M64, "m64_trained": M64}
TEMPORAL = {"m9_reservoir": dict(gain=3.0, bias=5.0), "m9_bptt": dict(gain=10.0, bias=1.0)}
SUBSTRATES = (*STATIC, *TEMPORAL, "input_track2", "input_rawseg")
BURN, K = 25, lambda y, p: float(cohen_kappa_score(y, p))


# ---------- readout grid ----------
def grid():
    from sklearn.linear_model import RidgeClassifier, LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    g = [("ridge", a, lambda a=a: RidgeClassifier(alpha=a)) for a in np.logspace(-3, 4, 15)]
    g += [("slda", s, lambda s=s: LDA(solver="lsqr", shrinkage=s)) for s in ["auto", 0.01, 0.1, 0.3, 0.5, 0.7, 0.9]]
    g += [("logreg", c, lambda c=c: LogisticRegression(C=c, max_iter=2000)) for c in np.logspace(-4, 3, 15)]
    return g


def nl_grid(d):
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    g = [("rbf_svm", (c, gm), lambda c=c, gm=gm: SVC(C=c, gamma=gm / d)) for c in np.logspace(-2, 3, 6) for gm in (0.3, 1, 3)]
    return g + [("knn", k, lambda k=k: KNeighborsClassifier(k)) for k in (5, 15, 45)]


def ceilings(F, ytr, yva, seed, nonlinear=False):
    """F: {fmap: (Ftr, Fva)} standardised inside. Returns kappa_cv, kappa_or and the full val table."""
    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(ytr)), ytr))
    rows = []
    for fmap, (Ftr, Fva) in F.items():
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9; A, B = (Ftr - mu) / sd, (Fva - mu) / sd
        for name, hp, make in (nl_grid(A.shape[1]) if nonlinear else grid()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv = float(np.mean([K(ytr[te], make().fit(A[tr], ytr[tr]).predict(A[te])) for tr, te in folds]))
                pv = make().fit(A, ytr).predict(B)
            rows.append({"fmap": fmap, "clf": name, "hp": hp if not isinstance(hp, float) else float(hp), "cv": cv,
                         "val": K(yva, pv), "pred": pv.tolist()})
    best_cv = max(range(len(rows)), key=lambda i: (rows[i]["cv"], -i))
    best_or = max(range(len(rows)), key=lambda i: (rows[i]["val"], -i))
    return {"kappa_cv": rows[best_cv]["val"], "cv_choice": {k: rows[best_cv][k] for k in ("fmap", "clf", "hp", "cv")},
            "pred_cv": rows[best_cv]["pred"], "kappa_or": rows[best_or]["val"],
            "or_choice": {k: rows[best_or][k] for k in ("fmap", "clf", "hp", "cv")},
            "table": [{k: v for k, v in r.items() if k != "pred"} for r in rows]}


# ---------- static lakes ----------
def static_sandbox(sub, subject, seed, d, stats_out, stored=False):
    from .sandbox import Sandbox, SandboxConfig, train_sandbox
    cfg = STATIC[sub]
    if sub.endswith("_random"): return Sandbox(SandboxConfig(**cfg), seed)
    if sub == "random_lake": return Sandbox(SandboxConfig(**cfg), seed)
    wdir = OUT / "weights" / sub; wdir.mkdir(parents=True, exist_ok=True); wp = wdir / f"A{subject:02d}_s{seed}.npz"
    if wp.exists():
        z = np.load(wp); sb = Sandbox(SandboxConfig(**cfg), seed)
        sb.learner.brain.efficacy = z["efficacy"]; sb.learner.brain.bias = z["bias"]
        stats_out.update(json.loads(str(z["stats"])))
    else:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); sb, st, _ = train_sandbox(d["train"], seed, **cfg)
        tst = {"retrained": True, "train_wall_s": time.time() - t0, "failed": st.get("failed"),
               "learn_accept_rate": st["accepted_rows"] / max(1, st["attempted_rows"]), "refused_batches": st["refused_batches"],
               "attempted_batches": st["attempted_batches"], "cert_all_epochs": all(e["certified"] for e in st["epochs"]),
               "n_projections": getattr(sb, "n_projections", 0)}
        np.savez_compressed(wp, efficacy=np.asarray(sb.learner.brain.efficacy), bias=np.asarray(sb.learner.brain.bias),
                            stats=json.dumps(tst))
        stats_out.update(tst)
    stats_out["weights"] = "retrained"
    if seed == 0:  # amendment M1-4: stored weights if the retrain reproduces them to 1e-6, else the retrain
        z = np.load(STORED[sub] / f"A{subject:02d}_full.npz")
        assert np.array_equal(z["pre"], sb.conn.pre) and np.array_equal(z["post"], sb.conn.post)
        diff = float(np.abs(np.asarray(sb.learner.brain.efficacy) - z["efficacy"]).max()); stats_out["repro_max_abs_diff"] = diff
        if diff <= 1e-6 or stored:
            sb.learner.brain.efficacy = z["efficacy"]; sb.learner.brain.bias = z["bias"]; stats_out["weights"] = "stored"
    return sb


def static_states(sb, X):
    n, k, dd = X.shape; eq = sb.free_states(X)
    a = np.asarray(eq.state.activation); v = np.asarray(eq.state.v)
    ok = np.asarray(eq.converged) & np.isfinite(a).all(1)
    return a[:, sb.col0:sb.out0].reshape(n, k, -1), v[:, sb.col0:sb.out0].reshape(n, k, -1), ok.reshape(n, k)


def fmaps_static(S, ok, ref=None):
    w = ok[..., None].astype(float); mean = (S * w).sum(1) / np.maximum(w.sum(1), 1)
    fill = ref if ref is not None else mean.mean(0)
    cat = np.where(ok[..., None], S, fill[None, None]).reshape(len(S), -1)
    return mean, cat


def linear_r2(Xtr, Str, Xva, Sva):
    """Share of variance of the column states linear (affine) in the input features, fit on train rows."""
    A = np.c_[Xtr.reshape(-1, Xtr.shape[-1]), np.ones(Xtr.shape[0] * Xtr.shape[1])]
    W = np.linalg.lstsq(A, Str.reshape(-1, Str.shape[-1]), rcond=None)[0]
    P = np.c_[Xva.reshape(-1, Xva.shape[-1]), np.ones(Xva.shape[0] * Xva.shape[1])] @ W; T = Sva.reshape(-1, Sva.shape[-1])
    return float(1 - ((P - T) ** 2).sum() / ((T - T.mean(0)) ** 2).sum())


# ---------- temporal (M9) ----------
def temporal_states(sub, subject, seed, d, stats_out):
    from .m9_temporal import inputs, masks, N_CH
    o = TEMPORAL[sub]; Utr = inputs(d["train"]["seg"], 2, o["bias"], o["gain"]); Uva = inputs(d["val"]["seg"], 2, o["bias"], o["gain"])
    if sub == "m9_reservoir":
        from cadence.experimental import PartitionedTemporalPatchNet
        net = PartitionedTemporalPatchNet(Utr.shape[2], N_CH * 2, 4, masks=masks(2, True), seed=seed, initial_radius=0.9)
        hs = lambda U: np.tanh(net.imagine(U, state=np.zeros((len(U), net.hidden))).hidden)
        stats_out["A_radius"] = float(np.abs(np.linalg.eigvals(net.parameters()["A"])).max())
        return hs(Utr), hs(Uva)
    import torch
    from .m9_bptt import one as bptt_one
    nets = []; t0 = time.time()
    r = bptt_one(subject, seed, {"k": 2, "crop": 125, "epochs": 100, "batch": 16, "lr": 0.01, "radius": 0.9, "ds": 2,
                                 "bias": o["bias"], "gain": o["gain"]}, net_out=nets)
    net = nets[0]; stats_out.update(bptt_kappa_val=r["kappa_val"], train_wall_s=time.time() - t0,
                                    A_radius=float(np.abs(np.linalg.eigvals((net.A * net.m["A"]).detach().numpy())).max()))
    def hs(U):
        U = torch.tensor(U); A, B = net.A * net.m["A"], net.B * net.m["B"]; h = torch.zeros(len(U), net.H, dtype=U.dtype); out = []
        with torch.no_grad():
            for t in range(U.shape[1]):
                h = torch.tanh(h) @ A.T + U[:, t] @ B.T; out.append(torch.tanh(h))
        return torch.stack(out, 1).numpy()
    return hs(Utr), hs(Uva)


def fmaps_temporal(S):
    """F_mean, F_cat (10 bins) and F_pow (log-variance per unit, amendment M1-2) over t in [BURN, 375)."""
    S = S[:, BURN:]; bins = np.array_split(np.arange(S.shape[1]), 10)
    return S.mean(1), np.concatenate([S[:, b].mean(1) for b in bins], 1), np.log(S.var(1) + 1e-12)


def one(sub, subject, seed):
    prev = existing(OUT / run_commit() / sub, f"A{subject:02d}_s{seed}")
    if prev: return {"arm": sub, "subject": subject, "seed": seed, "kappa_cv": prev["kappa_cv"], "kappa_or": prev["kappa_or"],
                     "wall_s": prev["wall_s"], "nl_or": prev.get("nonlinear", {}).get("kappa_or")}
    t0 = time.time(); d = load_subject(subject); ytr, yva = d["train"]["y"], d["val"]["y"]; info = {}
    if sub.endswith("_stored"):  # sensitivity (M1-4): seed 0 with the stored weights even when the retrain differs
        sub = sub[:-7]; info["sensitivity"] = "stored_weights"
    if sub in STATIC:
        sb = static_sandbox(sub, subject, seed, d, info, stored=info.get("sensitivity") == "stored_weights")
        Str, Vtr, oktr = static_states(sb, d["train"]["track2_x"]); Sva, Vva, okva = static_states(sb, d["val"]["track2_x"])
        mtr, ctr = fmaps_static(Str, oktr); mva, cva = fmaps_static(Sva, okva, ref=mtr.mean(0))
        c = sb.cert()
        info.update(cert=c, linear_r2=linear_r2(d["train"]["track2_x"], Str, d["val"]["track2_x"], Sva),
                    sat_frac=float((np.abs(np.r_[Vtr.ravel(), Vva.ravel()]) > 2).mean()),
                    settle_rate=float(np.r_[oktr.ravel(), okva.ravel()].mean()),
                    rejected=int((~oktr).sum() + (~okva).sum()))
        F = {"mean": (mtr, mva), "cat": (ctr, cva)}
    elif sub in TEMPORAL:
        Htr, Hva = temporal_states(sub, subject, seed, d, info)
        F = dict(zip(("mean", "cat", "pow"), zip(fmaps_temporal(Htr), fmaps_temporal(Hva))))
        info.update(settle_rate=1.0, rejected=0, linear_r2=None)  # explicit recursion, no equilibrium solve
    elif sub == "input_track2":
        Xtr, Xva = d["train"]["track2_x"], d["val"]["track2_x"]; ok1 = np.ones(Xtr.shape[:2], bool)
        F = dict(zip(("mean", "cat"), zip(fmaps_static(Xtr, ok1), fmaps_static(Xva, np.ones(Xva.shape[:2], bool)))))
        info.update(settle_rate=1.0, rejected=0, linear_r2=1.0)
    else:  # input_rawseg: the M9 encoding of the raw segment (gain 3, port dropped), same two maps
        from .m9_temporal import inputs
        Utr, Uva = inputs(d["train"]["seg"], 2, 0.0, 3.0), inputs(d["val"]["seg"], 2, 0.0, 3.0)
        F = dict(zip(("mean", "cat", "pow"), zip(fmaps_temporal(Utr), fmaps_temporal(Uva))))
        info.update(settle_rate=1.0, rejected=0, linear_r2=1.0)
    res = ceilings(F, ytr, yva, seed)
    if sub == "input_track2": res["nonlinear"] = ceilings(F, ytr, yva, seed, nonlinear=True)
    arm = sub + ("_stored" if info.get("sensitivity") else "")
    rec = {"stage": "M11", "step": "B", "arm": arm, "subject": subject, "seed": seed, **res, "label": yva, "diag": info,
           "n_train": int(len(ytr)), "n_val": int(len(yva)), "settle_rate": info["settle_rate"], "rejected": info["rejected"],
           "wall_s": time.time() - t0}
    write_receipt(OUT / run_commit() / arm, f"A{subject:02d}_s{seed}", rec, subject=subject)
    return {"arm": arm, "subject": subject, "seed": seed, "kappa_cv": res["kappa_cv"], "kappa_or": res["kappa_or"],
            "wall_s": rec["wall_s"], "nl_or": res.get("nonlinear", {}).get("kappa_or")}


def _job(a): return one(*a)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subs", default=",".join(SUBSTRATES)); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seeds", default="0,1,2"); a = ap.parse_args()
    jobs = [(s, subj, r) for s in a.subs.split(",") for r in map(int, a.seeds.split(",")) for subj in range(1, 10)]
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(_job, jobs):
            print(f"{r['arm']} A{r['subject']:02d} s{r['seed']} cv {r['kappa_cv']:.3f} or {r['kappa_or']:.3f}"
                  + (f" nl_or {r['nl_or']:.3f}" if r["nl_or"] is not None else "") + f" {r['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
