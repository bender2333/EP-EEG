"""M1 gradient-direction check: centred EP contrast vs finite-difference gradient
of the free-equilibrium loss T*CE(softmax(s_out/T), y), on a small sandbox (k=1)."""
from __future__ import annotations
import json, sys, numpy as np
from .data import load_subject
from .sandbox import Sandbox, SandboxConfig


def loss(sb, D, y, T, tol):
    eq = sb.learner.brain.equilibrate(D, budget=4096, tolerance=tol, damping=3)
    assert np.all(eq.converged)
    s = np.asarray(eq.state.activation)[:, sb.outputs] / T
    s = s - s.max(1, keepdims=True); lp = s - np.log(np.exp(s).sum(1, keepdims=True))
    return float(-(T * lp[np.arange(len(y)), y]).mean())


def main(opts, seed=0, n_check=60, beta=None):
    cfg = SandboxConfig(**{**opts, "k": 1, "sigma_col": 0.05, "sigma_out": 0.1, "tolerance": 1e-9, "free_steps": 4096, "nudged_steps": 4096})
    if beta: cfg.beta = beta
    sb = Sandbox(cfg, seed); d = load_subject(1)
    X = d["train"]["track2_x"][:16, 5]; y = d["train"]["y"][:16]; D = sb.drive(X)
    L = sb.learner; T = cfg.temperature
    free = L.free(D); tgt = L.targets(y)
    plus = L.nudged(D, free, tgt, sign=1.0); minus = L.nudged(D, free, tgt, sign=-1.0)
    edges, _ = L.contrast(free, plus, minus)
    rev = L.reverse; rng = np.random.default_rng(seed)
    cand = [i for i in range(len(edges)) if rev[i] == -1 or rev[i] > i]
    pick = rng.choice(cand, size=min(n_check, len(cand)), replace=False)
    br = L.brain; eff0 = np.asarray(br.efficacy).copy(); h = 1e-4
    ep, fd = [], []
    for i in pick:
        idx = [i] + ([rev[i]] if rev[i] >= 0 else [])
        e = eff0.copy(); e[idx] += h; br.efficacy = e; lp = loss(sb, D, y, T, 1e-11)
        e = eff0.copy(); e[idx] -= h; br.efficacy = e; lm = loss(sb, D, y, T, 1e-11)
        br.efficacy = eff0
        fd.append(-(lp - lm) / (2 * h)); ep.append(edges[idx].sum())
    ep, fd = np.array(ep), np.array(fd)
    cos = float(ep @ fd / (np.linalg.norm(ep) * np.linalg.norm(fd)))
    return {"beta": cfg.beta, "cosine": cos, "sign_agree": float(np.mean(np.sign(ep) == np.sign(fd))),
            "scale_ratio": float(np.linalg.norm(ep) / np.linalg.norm(fd)), "n": len(pick)}


if __name__ == "__main__":
    opts = json.loads(sys.argv[1])
    for b in (0.04, 0.01, 0.0025):
        print(json.dumps(main(opts, beta=b)))
