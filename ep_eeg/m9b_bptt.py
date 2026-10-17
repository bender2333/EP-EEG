"""M9b route B (BPTT feasibility before any EP compute): fast oscillators -> rectifier -> slow leaky integrators -> readout.

Fast block (as M9): f[t] = A tanh(f[t-1]) + B u[t]; lead c drives only its k microcolumns (mask B), A dense.
Each fast unit j owns one slow unit s_j; the readout C sees only the slow block.
  variant "square" : s[t] = lam*s[t-1] + (1-lam)*tanh(f[t])**2          (even map, no bias port needed)
  variant "rectify": s[t] = lam*s[t-1] + (1-lam)*|tanh(f[t])|
  variant "tpn"    : s[t] = lam*tanh(s[t-1]) + w*tanh(f[t-1]) + v*bias   (TemporalPatchNet-expressible block
                     form: slow units are ordinary hidden units with a diagonal self-weight; rectification
                     only through the fast units' bias operating point)
  y[t] = C g(s[t]), g = identity (square/rectify) or tanh (tpn). lam = sigmoid(a) per slow unit, learnable,
  initialised to `lam` (0.98 ~ 0.4 s at 125 Hz).
Loss: the per-step quadratic loss (EP-native). Decision (frozen, as M9): mean y over t >= BURN, argmax.
Baseline on the SAME untrained structure: ridge (RidgeClassifierCV, standardised) on mean_{t>=BURN} g(s).
rho_max>0: after every optimizer step the fast recurrence is rescaled, A <- A*rho_max/rho(A) when
rho(A) > rho_max (prevents the stiffness / blow-up seen in M9; the same projection can wrap an EP update).
Monitoring: spectral radius of the fast recurrence A (and of the full block matrix for "tpn") per eval.
"""
from __future__ import annotations
import argparse, json, platform, subprocess, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch
from sklearn.linear_model import RidgeClassifierCV
from .data import load_subject, ROOT
from .m9_temporal import N_CH, N_CLS, BURN, masks, inputs, kappa_acc

DEFAULTS = dict(variant="square", k=2, crop=125, epochs=100, batch=16, lr=0.01, radius=0.9, ds=2, bias=0.0,
                gain=3.0, lam=0.98, center=False, shuffle_labels=False, rho_max=0.0)


class RouteB(torch.nn.Module):
    def __init__(self, o, seed):
        super().__init__(); g = np.random.default_rng(seed); H = N_CH * o["k"]; nb = int(o["bias"] > 0)
        self.v, self.H, self.nb = o["variant"], H, nb
        q, _ = np.linalg.qr(g.normal(size=(H, H)))
        self.mB = torch.tensor(masks(o["k"], False)["B"], dtype=torch.float64)
        self.A = torch.nn.Parameter(torch.tensor(o["radius"] * q))
        self.B = torch.nn.Parameter(torch.tensor(g.normal(size=(H, N_CH)) / np.sqrt(H)) * self.mB)
        self.Bb = torch.nn.Parameter(torch.tensor(g.normal(size=(H,)) / np.sqrt(H)))  # bias port -> fast
        self.a = torch.nn.Parameter(torch.full((H,), float(np.log(o["lam"] / (1 - o["lam"]))), dtype=torch.float64))
        self.w = torch.nn.Parameter(torch.tensor(g.normal(size=(H,)) / np.sqrt(H)))     # tpn: fast_j -> slow_j
        self.vb = torch.nn.Parameter(torch.tensor(g.normal(size=(H,)) / np.sqrt(H)))    # tpn: bias port -> slow
        self.C = torch.nn.Parameter(torch.tensor(g.normal(size=(N_CLS, H)) / np.sqrt(H)))

    def slow(self, u):  # u (b,T,22[+1]) -> readout features g(s) (b,T,H)
        A, B = self.A, self.B * self.mB; lam = torch.sigmoid(self.a)
        b = u[..., N_CH:N_CH + 1] if self.nb else torch.zeros_like(u[..., :1])
        f = torch.zeros(u.shape[0], self.H, dtype=u.dtype); s = torch.zeros_like(f); out = []
        for t in range(u.shape[1]):
            if self.v == "tpn":
                s = lam * torch.tanh(s) + self.w * torch.tanh(f) + self.vb * b[:, t]
            f = torch.tanh(f) @ A.T + u[:, t, :N_CH] @ B.T + b[:, t] * self.Bb
            if self.v == "square": s = lam * s + (1 - lam) * torch.tanh(f) ** 2
            elif self.v == "rectify": s = lam * s + (1 - lam) * torch.tanh(f).abs()
            out.append(torch.tanh(s) if self.v == "tpn" else s)
        return torch.stack(out, 1)

    def forward(self, u):
        return self.slow(u) @ self.C.T

    def radii(self):
        with torch.no_grad():
            ra = float(np.abs(np.linalg.eigvals(self.A.numpy())).max())
            lam = torch.sigmoid(self.a).numpy()
        return {"rho_A_fast": ra, "lam_min": float(lam.min()), "lam_max": float(lam.max()),
                "rho_block": max(ra, float(lam.max())) if self.v == "tpn" else None}


def ridge_baseline(net, Utr, ytr, Uva, yva):
    with torch.no_grad():
        F0, F1 = (net.slow(U)[:, BURN:].mean(1).numpy() for U in (Utr, Uva))
    mu, sd = F0.mean(0), F0.std(0) + 1e-9
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 13)).fit((F0 - mu) / sd, ytr)
    return kappa_acc(yva, clf.predict((F1 - mu) / sd))[0]


def one(subject, seed, opts):
    o = {**DEFAULTS, **opts}; t0 = time.time(); torch.set_num_threads(2)
    d = load_subject(subject)
    Utr, ytr = torch.tensor(inputs(d["train"]["seg"], o["ds"], o["bias"], o["gain"])), d["train"]["y"].astype(int)
    Uva, yva = torch.tensor(inputs(d["val"]["seg"], o["ds"], o["bias"], o["gain"])), d["val"]["y"].astype(int)
    rng = np.random.default_rng(seed); ylearn = rng.permutation(ytr) if o["shuffle_labels"] else ytr
    torch.manual_seed(seed); net = RouteB(o, seed)
    k_ridge = ridge_baseline(net, Utr, ylearn, Uva, yva)
    print(f"[B {o['variant']} s{subject}] ridge baseline (untrained) k_va={k_ridge:.3f} {net.radii()}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=o["lr"]); T, L = Utr.shape[1], o["crop"]; curve = []
    pred = lambda U: net(U)[:, BURN:].mean(1).argmax(1).numpy()
    for ep in range(o["epochs"]):
        order = rng.permutation(len(Utr))
        for i in range(0, len(order), o["batch"]):
            j = order[i:i + o["batch"]]; s = rng.integers(0, T - L + 1, len(j))
            u = torch.stack([Utr[a, b:b + L] for a, b in zip(j, s)])
            tgt = torch.full((len(j), L, N_CLS), -0.25 if o["center"] else 0.0, dtype=u.dtype)
            tgt[np.arange(len(j)), :, ylearn[j]] += 1.0
            loss = 0.5 * ((net(u) - tgt) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if o["rho_max"] > 0:
                with torch.no_grad():
                    r = float(np.abs(np.linalg.eigvals(net.A.numpy())).max())
                    if r > o["rho_max"]: net.A.mul_(o["rho_max"] / r)
        if (ep + 1) % 10 == 0 or ep == o["epochs"] - 1:
            with torch.no_grad():
                curve.append({"epoch": ep + 1, "kappa_train": kappa_acc(ytr, pred(Utr))[0],
                              "kappa_val": kappa_acc(yva, pred(Uva))[0], **net.radii()})
            print(f"[B {o['variant']} s{subject}] {curve[-1]}", flush=True)
    with torch.no_grad(): p = pred(Uva)
    k, a = kappa_acc(yva, p)
    return {"subject": subject, "seed": seed, "kappa_val": k, "acc_val": a, "kappa_ridge_untrained": k_ridge,
            "settle_rate": 1.0, "curve": curve, "pred": p.tolist(), "y": yva.tolist(), "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="m9b_square"); ap.add_argument("--subjects", default="1-9")
    ap.add_argument("--seeds", default="0"); ap.add_argument("--opts", default="{}")
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--tag", default="")
    a = ap.parse_args(); opts = json.loads(a.opts)
    lo, _, hi = a.subjects.partition("-"); subs = list(range(int(lo), int(hi or lo) + 1))
    seeds = [int(s) for s in a.seeds.split(",")]
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    jobs = [(s, k) for s in subs for k in seeds]; t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, *zip(*[(s, k, dict(opts)) for s, k in jobs])))
    mean = lambda key, s=None: float(np.mean([r[key] for r in res if s is None or r["subject"] == s]))
    summ = {"commit": commit, "stage": "M9b", "arm": a.arm, "opts": {**DEFAULTS, **opts}, "subjects": subs,
            "seeds": seeds, "kappa_val": mean("kappa_val"), "acc_val": mean("acc_val"),
            "kappa_ridge_untrained": mean("kappa_ridge_untrained"), "settle_rate": 1.0,
            "kappa_seed_sd": float(np.std([np.mean([r["kappa_val"] for r in res if r["seed"] == k]) for k in seeds])),
            "per_subject_kappa": {s: mean("kappa_val", s) for s in subs},
            "per_subject_ridge": {s: mean("kappa_ridge_untrained", s) for s in subs},
            "wall_s": time.time() - t0, "python": platform.python_version()}
    out = ROOT / "receipts" / "M9b" / a.arm / f"{commit}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_text(json.dumps(res)); (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1)); print(out / "summary.json")


if __name__ == "__main__":
    main()
