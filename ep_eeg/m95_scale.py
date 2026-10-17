"""M9.5 Scaling Probe (exploratory, branch `scale`). Validation only: T runs 0-3 train, runs 4-5 val; E never read.

Input: identical to M9 (`m9_temporal.inputs`): frozen eval-layer segment, 125 Hz, gain*tanh(x/3), gain=3,
plus a constant port. Recurrence as M9: h[t] = A tanh(h[t-1]) + B u[t], decision = mean_{t>=25} readout.
Per-unit statistics are kept equal to the M9 k=2 reservoir at every width: lead weight ~ N(0,1/44),
constant-port weight ~ N(0,1/44) * 5 (=> per-unit offset sd ~0.75, the "+-0.7 operating point").
A = diag(alpha) + W_rec/sqrt(H), alpha~U(0.5,0.98), W_rec~N(0,1); scaled down so ||A||_2 <= 0.98.
Reservoir parameters depend on the seed only (shared across subjects).

exp1: frozen (A,B), features = mean_{t>=25} tanh(h), standardised, RidgeClassifier, lambda in
      {0.1,1,10,100,1000} by 5-fold stratified CV on the training split only; val kappa per subject.
exp2: pooled 9-subject train splits, shared (A,B) + subject readouts (C_s, b_s) initialised from exp1 ridge,
      BPTT 100 epochs, mean-output MSE on +-1 targets, Adam(lr 1e-3, wd 1e-3), ||A||_2 clip 0.98.
Checkpoints: exp1 one json per (seed,k,subject); exp2 state after every epoch (resume on restart).
"""
from __future__ import annotations
import argparse, json, time, platform, subprocess
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon
from .data import load_subject, ROOT
from .m9_temporal import N_CH, N_CLS, BURN, inputs, kappa_acc

GAIN, PORT, DS = 3.0, 5.0, 2
LAMS = [0.1, 1, 10, 100, 1000]
OUT = ROOT / "receipts" / "M9_5"


def reservoir(k, seed):
    g = np.random.default_rng(1000 + seed); H = N_CH * k
    alpha = g.uniform(0.5, 0.98, H)
    A = np.diag(alpha) + g.normal(size=(H, H)) / np.sqrt(H)
    n0 = np.linalg.norm(A, 2); A = A * min(1.0, 0.98 / n0)
    B = np.zeros((H, N_CH + 1)); s = 1 / np.sqrt(44)
    for c in range(N_CH): B[c * k:(c + 1) * k, c] = g.normal(size=k) * s
    B[:, N_CH] = g.normal(size=H) * s  # port value PORT multiplies this
    mask = (B != 0).astype(float)
    return A, B, mask, {"norm_before_clip": float(n0), "clip_scale": float(min(1.0, 0.98 / n0)),
                        "eff_alpha_range": [float(alpha.min() * min(1, .98 / n0)), float(alpha.max() * min(1, .98 / n0))]}


def reservoir_b(k, seed, g0):
    """Corrected Exp1 (exp1b): diagonal alpha~U(0.5,0.98) kept intact; only the coupling g*W_rec/sqrt(H) is scaled.
    g starts at g0 and is halved (coupling part only) until spectral radius <= 0.98. g0=0 -> pure diagonal."""
    gr = np.random.default_rng(1000 + seed); H = N_CH * k
    alpha = gr.uniform(0.5, 0.98, H); W = gr.normal(size=(H, H)) / np.sqrt(H)
    B = np.zeros((H, N_CH + 1)); s = 1 / np.sqrt(44)
    for c in range(N_CH): B[c * k:(c + 1) * k, c] = gr.normal(size=k) * s
    B[:, N_CH] = gr.normal(size=H) * s
    g = g0
    while True:
        A = np.diag(alpha) + g * W; ev = np.abs(np.linalg.eigvals(A))
        if ev.max() <= 0.98 or g == 0: break
        g /= 2
    d = np.diag(A)
    info = {"g0": g0, "g": g, "rho": float(ev.max()), "norm2": float(np.linalg.norm(A, 2)),
            "diag_min": float(d.min()), "diag_q": np.percentile(d, [5, 50, 95]).tolist(), "diag_max": float(d.max()),
            "eig_abs_q": np.percentile(ev, [0, 5, 50, 95, 100]).tolist(), "offdiag_norm2": float(np.linalg.norm(g * W, 2))}
    return A, B, (B != 0).astype(float), info


def exp1b(seeds, ks, subs, g0):
    torch.set_num_threads(8); data = {s: load_subject(s) for s in subs}; tag = f"g{g0:g}"
    for seed in seeds:
        for k in ks:
            A, B, m, info = reservoir_b(k, seed, g0); At, Bt = torch.tensor(A), torch.tensor(B)
            print(f"[exp1b {tag} seed{seed} k{k}] {info}", flush=True)
            for s in subs:
                f = OUT / "exp1b" / tag / f"seed{seed}_k{k}_s{s}.json"
                if f.exists(): continue
                t0 = time.time(); d = data[s]
                tf = time.time(); F = feats(At, Bt, U(d["train"]["seg"])); Fv = feats(At, Bt, U(d["val"]["seg"]))
                tfeat = time.time() - tf; nsteps = (len(F) + len(Fv)) * 375
                ytr, yva = d["train"]["y"].astype(int), d["val"]["y"].astype(int)
                lam, cv, p, C, b = ridge(F, ytr, Fv); kv, av = kappa_acc(yva, p)
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps({"seed": seed, "k": k, "H": N_CH * k, "subject": s, "kappa_val": kv, "acc_val": av,
                    "lambda": lam, "cv_kappa": cv, "pred_counts": np.bincount(p, minlength=4).tolist(), "reservoir": info,
                    "feat_wall_s": tfeat, "us_per_trial_step": 1e6 * tfeat / nsteps, "wall_s": time.time() - t0}))
                print(f"[exp1b {tag} seed{seed} k{k} s{s}] k_va={kv:.3f} lam={lam} {time.time()-t0:.1f}s", flush=True)
    rows = {}
    for seed in seeds:
        for k in ks:
            r = [json.loads((OUT / "exp1b" / tag / f"seed{seed}_k{k}_s{s}.json").read_text()) for s in subs]
            rows[f"seed{seed}_k{k}"] = {"mean": float(np.mean([x["kappa_val"] for x in r])), "per": {x["subject"]: x["kappa_val"] for x in r},
                "lam": {x["subject"]: x["lambda"] for x in r}, "reservoir": r[0]["reservoir"], "wall": sum(x["wall_s"] for x in r),
                "us_per_trial_step": float(np.mean([x["us_per_trial_step"] for x in r]))}
    (OUT / "exp1b" / tag / "summary.json").write_text(json.dumps(rows, indent=1))


def U(seg):
    return torch.tensor(inputs(seg, DS, PORT, GAIN))  # last column = constant port (PORT, not scaled by gain)


def feats(A, B, u, bs=96):
    out = []
    with torch.no_grad():
        for i in range(0, len(u), bs):
            x = u[i:i + bs]; h = torch.zeros(len(x), A.shape[0], dtype=x.dtype); acc = 0
            for t in range(x.shape[1]):
                h = torch.tanh(h) @ A.T + x[:, t] @ B.T
                if t >= BURN: acc = acc + torch.tanh(h)
            out.append(acc / (x.shape[1] - BURN))
    return torch.cat(out).numpy()


def ridge(F, y, Fv):
    mu, sd = F.mean(0), F.std(0) + 1e-8; Z, Zv = (F - mu) / sd, (Fv - mu) / sd
    cv = {}
    for lam in LAMS:
        ks = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, y):
            ks.append(kappa_acc(y[te], RidgeClassifier(alpha=lam).fit(Z[tr], y[tr]).predict(Z[te]))[0])
        cv[lam] = float(np.mean(ks))
    lam = max(cv, key=cv.get); r = RidgeClassifier(alpha=lam).fit(Z, y)
    p = r.predict(Zv); C = r.coef_ / sd; b = r.intercept_ - C @ mu  # scores = C f + b
    return lam, cv, p, C, b


def exp1(seeds, ks, subs):
    torch.set_num_threads(8)
    data = {s: load_subject(s) for s in subs}
    for seed in seeds:
        for k in ks:
            A, B, m, info = reservoir(k, seed); At, Bt = torch.tensor(A), torch.tensor(B)
            for s in subs:
                f = OUT / "exp1" / f"seed{seed}_k{k}_s{s}.json"
                if f.exists(): continue
                t0 = time.time(); d = data[s]
                F, Fv = feats(At, Bt, U(d["train"]["seg"])), feats(At, Bt, U(d["val"]["seg"]))
                ytr, yva = d["train"]["y"].astype(int), d["val"]["y"].astype(int)
                lam, cv, p, C, b = ridge(F, ytr, Fv); kv, av = kappa_acc(yva, p)
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps({"seed": seed, "k": k, "H": N_CH * k, "subject": s, "kappa_val": kv, "acc_val": av,
                    "lambda": lam, "cv_kappa": cv, "pred_counts": np.bincount(p, minlength=4).tolist(),
                    "feat_sd": float(F.std()), "reservoir": info, "C": C.tolist(), "b": b.tolist(), "wall_s": time.time() - t0}))
                print(f"[exp1 seed{seed} k{k} s{s}] k_va={kv:.3f} lam={lam} {time.time()-t0:.0f}s", flush=True)


class Pooled(torch.nn.Module):
    def __init__(self, A, B, mask, Cs, bs, mus, sds):
        super().__init__(); self.mu = torch.tensor(np.stack(mus)); self.sd = torch.tensor(np.stack(sds))
        self.A = torch.nn.Parameter(torch.tensor(A)); self.B = torch.nn.Parameter(torch.tensor(B))
        self.mask = torch.tensor(mask); self.C = torch.nn.Parameter(torch.tensor(np.stack(Cs)))
        self.b = torch.nn.Parameter(torch.tensor(np.stack(bs)))

    def forward(self, u, sid):  # mean_{t>=BURN} scores, (n,4)
        B = self.B * self.mask; h = torch.zeros(len(u), self.A.shape[0], dtype=u.dtype); acc = 0
        for t in range(u.shape[1]):
            h = torch.tanh(h) @ self.A.T + u[:, t] @ B.T
            if t >= BURN: acc = acc + torch.tanh(h)
        f = (acc / (u.shape[1] - BURN) - self.mu[sid]) / self.sd[sid]  # frozen exp1 standardisation
        return torch.einsum("nh,nch->nc", f, self.C[sid]) + self.b[sid]


def exp2(k, seed, subs, epochs=100, crop=125, batch=16, lr_ab=1e-3):
    torch.set_num_threads(8); ck = OUT / "exp2" / (f"k{k}_seed{seed}" + ("" if lr_ab == 1e-3 else f"_lrAB{lr_ab:g}")); ck.mkdir(parents=True, exist_ok=True)
    A, B, mask, info = reservoir(k, seed)
    r1 = [json.loads((OUT / "exp1" / f"seed{seed}_k{k}_s{s}.json").read_text()) for s in subs]
    data = {s: load_subject(s) for s in subs}
    # v1: readout acts on features standardised with frozen exp1 (train-split) mean/sd; C_s,b_s = exp1 ridge
    # (same lambda) in that space. v0 folded 1/sd into C_s -> |C| huge, loss exploded (receipt k32_seed0_v0_folded).
    Cs, bs, mus, sds = [], [], [], []
    for r, s in zip(r1, subs):
        F = feats(torch.tensor(A), torch.tensor(B), U(data[s]["train"]["seg"])); mu, sd = F.mean(0), F.std(0) + 1e-8
        rc = RidgeClassifier(alpha=r["lambda"]).fit((F - mu) / sd, data[s]["train"]["y"].astype(int))
        Cs.append(rc.coef_); bs.append(rc.intercept_); mus.append(mu); sds.append(sd)
    net = Pooled(A, B, mask, Cs, bs, mus, sds)
    opt = torch.optim.Adam([{"params": [net.A, net.B], "lr": lr_ab}, {"params": [net.C, net.b], "lr": 1e-3}], lr=1e-3, weight_decay=1e-3)
    Utr = torch.cat([U(data[s]["train"]["seg"]) for s in subs]); ytr = np.concatenate([data[s]["train"]["y"].astype(int) for s in subs])
    str_ = np.concatenate([[i] * len(data[s]["train"]["y"]) for i, s in enumerate(subs)])
    Uva = {s: U(data[s]["val"]["seg"]) for s in subs}; yva = {s: data[s]["val"]["y"].astype(int) for s in subs}
    Y = -torch.ones(len(ytr), N_CLS, dtype=Utr.dtype); Y[np.arange(len(ytr)), ytr] = 1.0
    rng = np.random.default_rng(seed); state = {"epoch": 0, "curve": [], "wall": 0.0}
    if (ck / "state.pt").exists():
        z = torch.load(ck / "state.pt", weights_only=False); net.load_state_dict(z["net"]); opt.load_state_dict(z["opt"])
        rng.bit_generator.state = z["rng"]; state = z["state"]; print(f"[exp2] resumed at epoch {state['epoch']}", flush=True)

    def evaluate():
        out = {}
        with torch.no_grad():
            for i, s in enumerate(subs):
                p = torch.cat([net(Uva[s][j:j + 96], torch.full((len(Uva[s][j:j + 96]),), i)) for j in range(0, len(Uva[s]), 96)]).argmax(1).numpy()
                kv, av = kappa_acc(yva[s], p); out[s] = {"kappa_val": kv, "acc_val": av, "pred_counts": np.bincount(p, minlength=4).tolist()}
        return out

    def specs():
        a = net.A.detach().numpy(); return float(np.abs(np.linalg.eigvals(a)).max()), float(np.linalg.norm(a, 2))
    if state["epoch"] == 0 and not state["curve"]:
        ev = evaluate(); rho, nrm = specs()
        state["curve"].append({"epoch": 0, "per_subject": ev, "kappa_mean": float(np.mean([v["kappa_val"] for v in ev.values()])), "rho": rho, "norm2": nrm, "loss": None})
        print(f"[exp2] epoch 0 k_mean={state['curve'][-1]['kappa_mean']:.3f} rho={rho:.3f}", flush=True)
    T = Utr.shape[1]
    while state["epoch"] < epochs:
        te = time.time(); order = rng.permutation(len(Utr)); L = []
        for i in range(0, len(order), batch):
            j = order[i:i + batch]; st = rng.integers(0, T - crop + 1, len(j))
            u = torch.stack([Utr[a, b:b + crop] for a, b in zip(j, st)])
            loss = 0.5 * ((net(u, torch.tensor(str_[j])) - Y[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); L.append(loss.item())
            with torch.no_grad():
                n2 = torch.linalg.matrix_norm(net.A, 2)
                if n2 > 0.98: net.A.mul_(0.98 / n2)
        state["epoch"] += 1; rho, nrm = specs(); rec = {"epoch": state["epoch"], "loss": float(np.mean(L)), "rho": rho, "norm2": nrm, "epoch_wall": time.time() - te}
        if state["epoch"] % 5 == 0 or state["epoch"] == epochs:
            ev = evaluate(); rec.update(per_subject=ev, kappa_mean=float(np.mean([v["kappa_val"] for v in ev.values()])))
        state["curve"].append(rec); state["wall"] += time.time() - te
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "rng": rng.bit_generator.state, "state": state}, ck / "state.tmp")
        (ck / "state.tmp").replace(ck / "state.pt")
        print(f"[exp2] {json.dumps({kk: v for kk, v in rec.items() if kk != 'per_subject'})}", flush=True)
    c0, c1 = state["curve"][0], state["curve"][-1]
    k0 = [c0["per_subject"][s]["kappa_val"] for s in subs]; k1 = [c1["per_subject"][s]["kappa_val"] for s in subs]
    d = np.array(k1) - np.array(k0)
    w2 = wilcoxon(k1, k0).pvalue if np.any(d) else 1.0; w1 = wilcoxon(k1, k0, alternative="greater").pvalue if np.any(d) else 1.0
    summ = {"lr_AB": lr_ab, "k": k, "H": N_CH * k, "seed": seed, "kappa_before": float(np.mean(k0)), "kappa_after": float(np.mean(k1)),
            "per_subject_before": dict(zip(subs, k0)), "per_subject_after": dict(zip(subs, k1)),
            "wilcoxon_p_two_sided": float(w2), "wilcoxon_p_greater": float(w1), "reservoir": info, "wall_s": state["wall"],
            "best_epoch_mean": max((c for c in state["curve"] if "kappa_mean" in c), key=lambda c: c["kappa_mean"])["epoch"]}
    (ck / "summary.json").write_text(json.dumps(summ, indent=1)); (ck / "curve.json").write_text(json.dumps(state["curve"]))
    print(json.dumps(summ, indent=1), flush=True)


# ---------------------------------------------------------------- exp2b: Phase 2 fixes (corrected-alpha reservoir)
# (a) full 375-step segments for training (same as ridge init / eval), (b) A frozen, train only B (+C_s,b_s),
# (c) leave-one-subject-out: shared B learned on 8 subjects, held-out subject gets only a ridge readout on its own
# train split with frozen learned B. float32 throughout (same width / data / epochs; precision only).
DT = torch.float32


class PooledB(torch.nn.Module):
    def __init__(self, A, B, mask, Cs, bs, mus, sds):
        super().__init__()
        self.register_buffer("AT", torch.tensor(A.T, dtype=DT).contiguous())  # frozen A
        self.B = torch.nn.Parameter(torch.tensor(B, dtype=DT)); self.register_buffer("mask", torch.tensor(mask, dtype=DT))
        self.register_buffer("mu", torch.tensor(np.stack(mus), dtype=DT)); self.register_buffer("sd", torch.tensor(np.stack(sds), dtype=DT))
        self.C = torch.nn.Parameter(torch.tensor(np.stack(Cs), dtype=DT)); self.b = torch.nn.Parameter(torch.tensor(np.stack(bs), dtype=DT))

    def feats(self, u):  # raw time-averaged tanh(h), t>=BURN
        BT = (self.B * self.mask).T; h = torch.zeros(len(u), self.AT.shape[0], dtype=DT); acc = 0
        for t in range(u.shape[1]):
            h = torch.tanh(h) @ self.AT + u[:, t] @ BT
            if t >= BURN: acc = acc + torch.tanh(h)
        return acc / (u.shape[1] - BURN)

    def forward(self, u, sid):
        f = (self.feats(u) - self.mu[sid]) / self.sd[sid]
        return torch.einsum("nh,nch->nc", f, self.C[sid]) + self.b[sid]


def _feats_np(net, u, bs=96):
    with torch.no_grad():
        return torch.cat([net.feats(u[i:i + bs]) for i in range(0, len(u), bs)]).double().numpy()


def exp2b(k, seed, subs, holdout=None, g0=0.05, epochs=100, batch=16, lr=1e-3, wd=1e-3):
    """holdout=None: pooled over all subs (fix a+b). holdout=s: LOSO fold (fix c), trained on subs minus s."""
    torch.set_num_threads(int(__import__("os").environ.get("M95_THREADS", "8")))
    tag = "pooled" if holdout is None else f"loso_s{holdout}"
    ck = OUT / "exp2b" / f"k{k}_seed{seed}_g{g0:g}" / tag; ck.mkdir(parents=True, exist_ok=True)
    A, B, mask, info = reservoir_b(k, seed, g0)
    tr_subs = [s for s in subs if s != holdout]
    data = {s: load_subject(s) for s in subs}
    Ua = {s: {sp: U(data[s][sp]["seg"]).to(DT) for sp in ("train", "val")} for s in subs}
    ya = {s: {sp: data[s][sp]["y"].astype(int) for sp in ("train", "val")} for s in subs}
    # ridge init in the exact float32 feature space used for training (full 375 steps), lambda by 5-fold CV on train
    tmp = PooledB(A, B, mask, [np.zeros((N_CLS, A.shape[0]))], [np.zeros(N_CLS)], [np.zeros(A.shape[0])], [np.ones(A.shape[0])])
    Cs, bs_, mus, sds, init = [], [], [], [], {}
    for s in tr_subs:
        F, Fv = _feats_np(tmp, Ua[s]["train"]), _feats_np(tmp, Ua[s]["val"])
        lam, cv, p, _, _ = ridge(F, ya[s]["train"], Fv); mu, sd = F.mean(0), F.std(0) + 1e-8
        rc = RidgeClassifier(alpha=lam).fit((F - mu) / sd, ya[s]["train"])
        Cs.append(rc.coef_); bs_.append(rc.intercept_); mus.append(mu); sds.append(sd)
        init[s] = {"lambda": lam, "kappa_val": kappa_acc(ya[s]["val"], p)[0]}
    net = PooledB(A, B, mask, Cs, bs_, mus, sds)
    opt = torch.optim.Adam([net.B, net.C, net.b], lr=lr, weight_decay=wd)
    Utr = torch.cat([Ua[s]["train"] for s in tr_subs]); ytr = np.concatenate([ya[s]["train"] for s in tr_subs])
    sid = np.concatenate([[i] * len(ya[s]["train"]) for i, s in enumerate(tr_subs)])
    Y = -torch.ones(len(ytr), N_CLS, dtype=DT); Y[np.arange(len(ytr)), ytr] = 1.0
    rng = np.random.default_rng(seed); state = {"epoch": 0, "curve": [], "wall": 0.0}
    if (ck / "state.pt").exists():
        z = torch.load(ck / "state.pt", weights_only=False); net.load_state_dict(z["net"]); opt.load_state_dict(z["opt"])
        rng.bit_generator.state = z["rng"]; state = z["state"]; print(f"[exp2b {tag}] resumed at epoch {state['epoch']}", flush=True)

    def evaluate():
        out = {}
        with torch.no_grad():
            for i, s in enumerate(tr_subs):
                u = Ua[s]["val"]; p = torch.cat([net(u[j:j + 96], torch.full((len(u[j:j + 96]),), i)) for j in range(0, len(u), 96)]).argmax(1).numpy()
                kv, av = kappa_acc(ya[s]["val"], p); out[s] = {"kappa_val": kv, "acc_val": av, "pred_counts": np.bincount(p, minlength=4).tolist()}
        if holdout is not None:  # held-out subject: ridge readout only, on its own train split, frozen learned B
            s = holdout; F, Fv = _feats_np(net, Ua[s]["train"]), _feats_np(net, Ua[s]["val"])
            lam, cv, p, _, _ = ridge(F, ya[s]["train"], Fv); kv, av = kappa_acc(ya[s]["val"], p)
            out[f"heldout_{s}"] = {"kappa_val": kv, "acc_val": av, "lambda": lam, "pred_counts": np.bincount(p, minlength=4).tolist()}
        return out

    def rec_eval(rec):
        ev = evaluate(); rec["per_subject"] = ev
        rec["kappa_mean_train_subjects"] = float(np.mean([ev[s]["kappa_val"] for s in tr_subs]))
        if holdout is not None: rec["kappa_heldout"] = ev[f"heldout_{holdout}"]["kappa_val"]
        rec["B_norm"] = float((net.B * net.mask).norm()); return rec

    if not state["curve"]:
        state["curve"].append(rec_eval({"epoch": 0, "loss": None}))
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "rng": rng.bit_generator.state, "state": state}, ck / "state.pt")
        print(f"[exp2b {tag}] {json.dumps({kk: v for kk, v in state['curve'][-1].items() if kk != 'per_subject'})}", flush=True)
    while state["epoch"] < epochs:
        te = time.time(); order = rng.permutation(len(Utr)); L = []
        for i in range(0, len(order), batch):
            j = order[i:i + batch]
            loss = 0.5 * ((net(Utr[j], torch.tensor(sid[j])) - Y[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); L.append(loss.item())
        state["epoch"] += 1; ttrain = time.time() - te
        rec = rec_eval({"epoch": state["epoch"], "loss": float(np.mean(L)), "loss_max_batch": float(np.max(L)), "train_wall": ttrain})
        rec["epoch_wall"] = time.time() - te; state["curve"].append(rec); state["wall"] += rec["epoch_wall"]
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "rng": rng.bit_generator.state, "state": state}, ck / "state.tmp")
        (ck / "state.tmp").replace(ck / "state.pt")
        print(f"[exp2b {tag}] {json.dumps({kk: v for kk, v in rec.items() if kk != 'per_subject'})}", flush=True)
    c0, c1 = state["curve"][0], state["curve"][-1]
    summ = {"k": k, "H": N_CH * k, "seed": seed, "g0": g0, "holdout": holdout, "train_subjects": tr_subs, "reservoir": info,
            "ridge_init": init, "wall_s": state["wall"], "epochs": state["epoch"],
            "per_subject_ep0": {s: c0["per_subject"][s]["kappa_val"] for s in tr_subs},
            "per_subject_final": {s: c1["per_subject"][s]["kappa_val"] for s in tr_subs},
            "pred_counts_final": {s: c1["per_subject"][s]["pred_counts"] for s in tr_subs}}
    if holdout is not None:
        summ["heldout_ep0"] = c0["per_subject"][f"heldout_{holdout}"]; summ["heldout_final"] = c1["per_subject"][f"heldout_{holdout}"]
    (ck / "summary.json").write_text(json.dumps(summ, indent=1)); (ck / "curve.json").write_text(json.dumps(state["curve"]))
    (ck / "DONE").write_text("ok"); print(f"[exp2b {tag}] DONE", flush=True)


def summarize1(seeds, ks, subs):
    rows = {}
    for seed in seeds:
        for k in ks:
            r = [json.loads((OUT / "exp1" / f"seed{seed}_k{k}_s{s}.json").read_text()) for s in subs]
            rows[f"seed{seed}_k{k}"] = {"mean": float(np.mean([x["kappa_val"] for x in r])), "per": {x["subject"]: x["kappa_val"] for x in r},
                                        "lam": {x["subject"]: x["lambda"] for x in r}, "wall": sum(x["wall_s"] for x in r)}
    (OUT / "exp1" / "summary.json").write_text(json.dumps(rows, indent=1)); print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--seeds", default="0")
    ap.add_argument("--ks", default="2,8,32,128"); ap.add_argument("--k", type=int, default=32); ap.add_argument("--lr_ab", type=float, default=1e-3); ap.add_argument("--g0", type=float, default=0.05); ap.add_argument("--holdout", type=int, default=None)
    a = ap.parse_args(); subs = list(range(1, 10)); seeds = [int(x) for x in a.seeds.split(",")]; ks = [int(x) for x in a.ks.split(",")]
    if a.cmd == "exp1": exp1(seeds, ks, subs); summarize1(seeds, ks, subs)
    elif a.cmd == "exp1b": exp1b(seeds, ks, subs, a.g0)
    elif a.cmd == "exp2": exp2(a.k, seeds[0], subs, lr_ab=a.lr_ab)
    elif a.cmd == "exp2b": exp2b(a.k, seeds[0], subs, holdout=a.holdout, g0=a.g0)
