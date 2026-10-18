"""G3 (reports/M10-gates.md, pre-registered): can a cadence temporal module learn "square + ~1 s integration"
from a waveform by EP?

Synthetic, 125 Hz, 4 s sequences: x = a(t) sin(2 pi 10 t / 125 + phi) + 0.05 eps; a(t) in [0.1, 1] is a
0.5 Hz low-passed Gaussian process. Target y(t) = 2 * causal 1 s (125-sample) mean of x^2 (~ a^2), defined
for t >= 124. Tasks: "full" (input x, target y), "int" (input x^2: integration only), "sq" (input x, target
2 * 0.1 s mean of x^2: squaring only). Input ports: [signal, constant 1] (the constant breaks the odd
symmetry, M9). Net: TemporalPatchNet, h[t] = A tanh(h[t-1]) + B u[t], y[t] = C tanh(h[t]), H = 16.
EP training: random 250-step crops, batch 16, 2000 updates; per-step quadratic target, with steps lacking a
full history window in the crop (first W) set to the free output (no teaching there). Variants: A init
{orth, diag} x optimiser {EP increment (rate), EP gradient + Adam}. Metric: R^2 of y over t >= W on
held-out sequences from a zero state. References (not criteria): linear ridge on 125 lags of the input,
untrained net + ridge readout, same structure trained by BPTT (torch).
"""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.signal import butter, filtfilt
from cadence.temporal import TemporalPatchNet
from .data import ROOT

FS, T, W, L = 125, 500, 125, 250
SPLITS = {"train": (64, 1), "select": (16, 2), "test": (32, 3)}
VARIANTS = [dict(a_init=a, opt=o) for a in ("orth", "diag") for o in ("sgd", "adam")]
OPTS = dict(H=16, beta=0.05, rate=1.0, lr=0.003, radius=0.9, batch=16, updates=2000)


def make(task, n, seed):
    rng = np.random.default_rng(seed); b, a = butter(2, 0.5 / (FS / 2))
    z = filtfilt(b, a, rng.normal(size=(n, T + 200)), axis=1)[:, 100:-100]; z /= z.std()
    amp = 0.1 + 0.9 / (1 + np.exp(-2 * z))
    t = np.arange(T) / FS; x = amp * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi, (n, 1))) + 0.05 * rng.normal(size=(n, T))
    win = 12 if task == "sq" else W
    c = np.cumsum(np.pad(x ** 2, ((0, 0), (1, 0))), 1); y = np.full((n, T), np.nan)
    y[:, win - 1:] = 2 * (c[:, win:] - c[:, :-win]) / win
    sig = x ** 2 if task == "int" else x
    U = np.stack([sig, np.ones_like(sig)], 2)
    return U, y[..., None], win


def r2(yhat, y, win):
    a, b = yhat[:, win:].ravel(), y[:, win:].ravel(); return float(1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum())


def predict(net, U): return net.imagine(U, state=np.zeros((len(U), net.hidden))).output


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8): self.lr, self.b1, self.b2, self.eps, self.t, self.m, self.v = lr, b1, b2, eps, 0, {}, {}

    def step(self, net, grad):
        self.t += 1; P = net.parameters(); new = {}
        for k, g in grad.items():
            self.m[k] = self.b1 * self.m.get(k, 0) + (1 - self.b1) * g; self.v[k] = self.b2 * self.v.get(k, 0) + (1 - self.b2) * g * g
            mh, vh = self.m[k] / (1 - self.b1 ** self.t), self.v[k] / (1 - self.b2 ** self.t)
            new[k] = P[k] - self.lr * mh / (np.sqrt(vh) + self.eps)
        net.set_parameters({**P, **new})


def ep_run(task, variant, seed, eval_split="select", o=OPTS):
    t0 = time.time(); Utr, ytr, win = make(task, SPLITS["train"][0], SPLITS["train"][1])
    Uev, yev, _ = make(task, SPLITS[eval_split][0], SPLITS[eval_split][1])
    net = TemporalPatchNet(2, o["H"], 1, seed=seed, initial_radius=o["radius"])
    if variant["a_init"] == "diag":
        P = net.parameters(); P["A"] = o["radius"] * np.eye(o["H"]); net.set_parameters(P)
    adam = Adam(o["lr"]) if variant["opt"] == "adam" else None; rng = np.random.default_rng(seed + 100)
    st = {"attempted": 0, "updated": 0, "reasons": {}, "curve": []}
    for it in range(o["updates"]):
        j = rng.integers(0, len(Utr), o["batch"]); s0 = rng.integers(0, T - L + 1, o["batch"])
        u = np.stack([Utr[a, b:b + L] for a, b in zip(j, s0)]); tgt = np.stack([ytr[a, b:b + L] for a, b in zip(j, s0)])
        free = net.imagine(u, state=np.zeros((len(u), net.hidden))).output
        tgt = np.where(np.isnan(tgt), free, tgt); tgt[:, :win] = free[:, :win]
        net.reset(); r = net.observe(u, tgt, beta=o["beta"], rate=0.0 if adam else o["rate"])
        if adam and r.updated: adam.step(net, r.delta)
        st["attempted"] += 1; st["updated"] += int(r.updated); st["reasons"][r.reason] = st["reasons"].get(r.reason, 0) + 1
        if (it + 1) % 250 == 0:
            st["curve"].append({"update": it + 1, "r2_train": r2(predict(net, Utr[:16]), ytr[:16], win),
                                "r2_eval": r2(predict(net, Uev), yev, win), "wall": time.time() - t0})
            print(f"[{task} {variant} s{seed}] {it + 1} r2_eval={st['curve'][-1]['r2_eval']:.3f} upd {st['updated']}/{st['attempted']} {time.time() - t0:.0f}s", flush=True)
    rad = float(np.abs(np.linalg.eigvals(net.parameters()["A"])).max())
    out = {"task": task, **variant, "seed": seed, "split": eval_split, "r2": r2(predict(net, Uev), yev, win),
           "settle_rate": st["updated"] / max(1, st["attempted"]), "A_radius": rad, "train": st, "wall_s": time.time() - t0}
    if os.environ.get("G3_RUN_DIR"):  # per-run receipt, so a container restart loses at most the runs in flight
        Path(os.environ["G3_RUN_DIR"], f"{task}_{variant['a_init']}_{variant['opt']}_s{seed}_{eval_split}.json").write_text(json.dumps(out, default=float))
    return out


def references(task, seed=0, o=OPTS):
    from sklearn.linear_model import RidgeCV
    Utr, ytr, win = make(task, *SPLITS["train"]); Ute, yte, _ = make(task, *SPLITS["test"])
    def lags(U):
        s = U[..., 0]; P = np.stack([np.pad(s, ((0, 0), (k, 0)))[:, :T] for k in range(W)], 2); return P[:, win:].reshape(-1, W)
    lin = RidgeCV(alphas=np.logspace(-4, 3, 15)).fit(lags(Utr), ytr[:, win:].ravel())
    out = {"task": task, "linear_lag_ridge": float(1 - ((lin.predict(lags(Ute)) - yte[:, win:].ravel()) ** 2).sum() / ((yte[:, win:] - yte[:, win:].mean()) ** 2).sum())}
    net = TemporalPatchNet(2, o["H"], 1, seed=seed, initial_radius=o["radius"])
    hid = lambda U: np.tanh(net.imagine(U, state=np.zeros((len(U), o["H"]))).hidden)[:, win:].reshape(-1, o["H"])
    res = RidgeCV(alphas=np.logspace(-4, 3, 15)).fit(hid(Utr), ytr[:, win:].ravel())
    out["reservoir_ridge"] = float(1 - ((res.predict(hid(Ute)) - yte[:, win:].ravel()) ** 2).sum() / ((yte[:, win:] - yte[:, win:].mean()) ** 2).sum())
    out["bptt"] = bptt(task, seed, o)
    return out


def bptt(task, seed, o):
    import torch
    torch.manual_seed(seed); Utr, ytr, win = make(task, *SPLITS["train"]); Ute, yte, _ = make(task, *SPLITS["test"])
    H = o["H"]; q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(H, H)))
    A = torch.tensor(o["radius"] * q, requires_grad=True); B = torch.tensor(np.random.default_rng(seed + 1).normal(size=(H, 2)) / np.sqrt(H), requires_grad=True)
    C = torch.tensor(np.random.default_rng(seed + 2).normal(size=(1, H)) / np.sqrt(H), requires_grad=True)
    opt = torch.optim.Adam([A, B, C], lr=0.01); rng = np.random.default_rng(seed + 100)
    def fwd(u):
        h = torch.zeros(u.shape[0], H, dtype=torch.float64); ys = []
        for t in range(u.shape[1]):
            h = torch.tanh(h) @ A.T + u[:, t] @ B.T; ys.append(torch.tanh(h) @ C.T)
        return torch.stack(ys, 1)
    for it in range(o["updates"]):
        j = rng.integers(0, len(Utr), o["batch"]); s0 = rng.integers(0, T - L + 1, o["batch"])
        u = torch.tensor(np.stack([Utr[a, b:b + L] for a, b in zip(j, s0)])); tg = np.stack([ytr[a, b:b + L] for a, b in zip(j, s0)])
        m = torch.tensor(~np.isnan(tg)); m[:, :win] = False; tg = torch.tensor(np.nan_to_num(tg))
        loss = (((fwd(u) - tg) ** 2) * m).sum() / m.sum(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): yh = fwd(torch.tensor(Ute)).numpy()
    return r2(yh, yte, win)


def _ep(a): return ep_run(*a)
def _ref(t): return references(t)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", choices=("select", "final", "refs"), required=True)
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--variant", default=None); ap.add_argument("--updates", type=int, default=None); ap.add_argument("--seeds", default=None, help="json list of seed lists, one per variant")
    a = ap.parse_args(); o = {**OPTS, **({"updates": a.updates} if a.updates else {})}
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    out = ROOT / "receipts" / "G3" / commit; out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        if a.phase == "refs":
            res = list(ex.map(_ref, ("full", "int", "sq")))
        elif a.phase == "select":
            res = list(ex.map(_ep, [(t, v, 0, "select") for t in ("full", "int", "sq") for v in VARIANTS]))
        else:  # reports/M10-gates.md deviation: list of variants, each on the test split with seeds 0-2
            vs = json.loads(a.variant); vs = vs if isinstance(vs, list) else [vs]
            seeds = json.loads(a.seeds) if a.seeds else [[0, 1, 2]] * len(vs)
            res = list(ex.map(_ep, [("full", v, s, "test") for v, ss in zip(vs, seeds) for s in ss]))
    (out / f"{a.phase}.json").write_text(json.dumps(res, indent=1, default=float)); print(out / f"{a.phase}.json")
    for r in res: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("train",)})


if __name__ == "__main__":
    main()
