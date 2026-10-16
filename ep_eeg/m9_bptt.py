"""M9 control: the same masked recurrence as m9_temporal, trained by BPTT (torch, Adam) instead of EP.

Identical input, crops, one-hot quadratic target on every step, masks, init (orthogonal A * radius,
B,C ~ N(0,1/H)) and decision rule (zero boundary, mean output over t >= BURN). Separates
"this model class cannot learn from raw waveforms" from "EP learning is the bottleneck".
"""
from __future__ import annotations
import argparse, json, platform, subprocess, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch
from .data import load_subject, ROOT
from .m9_temporal import N_CH, N_CLS, BURN, masks, inputs, kappa_acc

DEFAULTS = dict(k=2, crop=125, epochs=20, batch=16, lr=3e-3, radius=0.9, ds=2, shuffle_labels=False)


class Rec(torch.nn.Module):
    def __init__(self, k, radius, seed):
        super().__init__(); H = N_CH * k; g = np.random.default_rng(seed); m = masks(k)
        q, _ = np.linalg.qr(g.normal(size=(H, H)))
        self.m = {n: torch.tensor(v, dtype=torch.float64) for n, v in m.items()}
        self.A = torch.nn.Parameter(torch.tensor(radius * q))
        self.B = torch.nn.Parameter(torch.tensor(g.normal(size=(H, N_CH)) / np.sqrt(H)) * self.m["B"])
        self.C = torch.nn.Parameter(torch.tensor(g.normal(size=(N_CLS, H)) / np.sqrt(H)))
        self.H = H

    def forward(self, u):  # u (b,T,22) -> y (b,T,4)
        A, B = self.A * self.m["A"], self.B * self.m["B"]; h = torch.zeros(u.shape[0], self.H, dtype=u.dtype)
        ys = []
        for t in range(u.shape[1]):
            h = torch.tanh(h) @ A.T + u[:, t] @ B.T; ys.append(torch.tanh(h) @ self.C.T)
        return torch.stack(ys, 1)


def one(subject, seed, opts):
    o = {**DEFAULTS, **opts}; t0 = time.time(); torch.set_num_threads(2)
    d = load_subject(subject)
    Utr, ytr = torch.tensor(inputs(d["train"]["seg"], o["ds"])), d["train"]["y"].astype(int)
    Uva, yva = torch.tensor(inputs(d["val"]["seg"], o["ds"])), d["val"]["y"].astype(int)
    rng = np.random.default_rng(seed); ylearn = rng.permutation(ytr) if o["shuffle_labels"] else ytr
    net = Rec(o["k"], o["radius"], seed); opt = torch.optim.Adam(net.parameters(), lr=o["lr"])
    T, L = Utr.shape[1], o["crop"]; curve = []
    pred = lambda U: net(U)[:, BURN:].mean(1).argmax(1).numpy()
    for ep in range(o["epochs"]):
        order = rng.permutation(len(Utr))
        for i in range(0, len(order), o["batch"]):
            j = order[i:i + o["batch"]]; s = rng.integers(0, T - L + 1, len(j))
            u = torch.stack([Utr[a, b:b + L] for a, b in zip(j, s)])
            tgt = torch.zeros(len(j), L, N_CLS, dtype=u.dtype); tgt[np.arange(len(j)), :, ylearn[j]] = 1.0
            loss = 0.5 * ((net(u) - tgt) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 5 == 0 or ep == o["epochs"] - 1:
            with torch.no_grad():
                curve.append({"epoch": ep + 1, "kappa_train": kappa_acc(ytr, pred(Utr))[0],
                              "kappa_val": kappa_acc(yva, pred(Uva))[0]})
            print(f"[bptt s{subject} seed{seed}] {curve[-1]}", flush=True)
    with torch.no_grad(): p = pred(Uva)
    k, a = kappa_acc(yva, p)
    return {"subject": subject, "seed": seed, "kappa_val": k, "acc_val": a, "settle_rate": 1.0,
            "curve": curve, "pred": p.tolist(), "y": yva.tolist(), "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="bptt_tpn"); ap.add_argument("--subjects", default="1")
    ap.add_argument("--seeds", default="0"); ap.add_argument("--opts", default="{}")
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--tag", default="")
    a = ap.parse_args(); opts = json.loads(a.opts)
    lo, _, hi = a.subjects.partition("-"); subs = list(range(int(lo), int(hi or lo) + 1))
    seeds = [int(s) for s in a.seeds.split(",")]
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    jobs = [(s, k) for s in subs for k in seeds]; t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, *zip(*[(s, k, dict(opts)) for s, k in jobs])))
    by_seed = [np.mean([r["kappa_val"] for r in res if r["seed"] == k]) for k in seeds]
    summ = {"commit": commit, "stage": "M9", "arm": a.arm, "opts": {**DEFAULTS, **opts}, "subjects": subs,
            "seeds": seeds, "kappa_val": float(np.mean([r["kappa_val"] for r in res])),
            "kappa_seed_sd": float(np.std(by_seed)), "acc_val": float(np.mean([r["acc_val"] for r in res])),
            "settle_rate": 1.0,
            "per_subject_kappa": {s: float(np.mean([r["kappa_val"] for r in res if r["subject"] == s])) for s in subs},
            "wall_s": time.time() - t0, "python": platform.python_version()}
    out = ROOT / "receipts" / "M9" / a.arm / f"{commit}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_text(json.dumps(res)); (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1)); print(out / "summary.json")


if __name__ == "__main__":
    main()
