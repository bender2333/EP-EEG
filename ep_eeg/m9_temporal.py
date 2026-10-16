"""M9 (ADR-0010, proposed): path EP on raw waveforms with PartitionedTemporalPatchNet.

h[t] = A tanh(h[t-1]) + B u[t],  y[t] = C tanh(h[t]); centred +-beta EP over the whole path.
Topology (ADR-0002 anchoring kept): lead c drives only its k microcolumns (mask B); microcolumns are
all-to-all recurrent (mask A, no spatial prior); every microcolumn reaches the 4 outlets (mask C).
Optional constant input port (opts["bias"]>0, mask: all microcolumns) breaks the odd symmetry
y(-u)=-y(u) of the bias-free recurrence; without it the time-averaged output cannot carry power.
Input: frozen eval-layer segment (EOG regression -> 4-35 Hz -> per-lead z, fit on T runs 0-3; cue+0.5..3.5 s),
no Hjorth, 250 -> 125 Hz by taking every 2nd sample, tanh(x/3). drive per step = 22 raw samples.
Training: each epoch one random `crop`-step window per training trial, zero initial boundary,
one-hot class target on every step (quadratic loss, the only TemporalPatchNet loss).
Decision (frozen before results): causal free path over the whole 375-step validation segment from a
zero boundary; outlet outputs averaged over steps t >= BURN, argmax.
settle_rate = accepted observe() calls / attempted (phase failures and asymmetric contrasts are refusals).
"""
from __future__ import annotations
import argparse, json, platform, subprocess, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.metrics import cohen_kappa_score
from cadence.experimental import PartitionedTemporalPatchNet
from .data import load_subject, ROOT

N_CH, N_CLS, BURN = 22, 4, 25
DEFAULTS = dict(k=2, crop=125, epochs=20, batch=16, beta=0.05, rate=1.0, radius=0.9, ds=2,
                shuffle_labels=False, eval_every=5, bias=0.0, gain=1.0)


def masks(k, bias=False):
    H = N_CH * k; B = np.zeros((H, N_CH + int(bias)), bool)
    for c in range(N_CH): B[c * k:(c + 1) * k, c] = True
    if bias: B[:, N_CH] = True
    return {"A": np.ones((H, H), bool), "B": B, "C": np.ones((N_CLS, H), bool)}


def inputs(seg, ds, bias=0.0, gain=1.0):
    """(n,22,750) z-scored segment -> (n, 750/ds, 22[+1]) compressed drive (+ constant port)."""
    u = gain * np.tanh(seg[:, :, ::ds] / 3.0).transpose(0, 2, 1).astype(float)
    return np.concatenate([u, np.full(u.shape[:2] + (1,), bias)], 2) if bias else u


def predict(net, U):
    out = net.imagine(U, state=np.zeros((len(U), net.hidden))).output
    return out[:, BURN:].mean(1)


def kappa_acc(y, p):
    return float(cohen_kappa_score(y, p)), float((y == p).mean())


def one(subject, seed, opts):
    o = {**DEFAULTS, **opts}; t0 = time.time()
    d = load_subject(subject)
    Utr, ytr = inputs(d["train"]["seg"], o["ds"], o["bias"], o["gain"]), d["train"]["y"].astype(int)
    Uva, yva = inputs(d["val"]["seg"], o["ds"], o["bias"], o["gain"]), d["val"]["y"].astype(int)
    rng = np.random.default_rng(seed)
    ylearn = rng.permutation(ytr) if o["shuffle_labels"] else ytr
    net = PartitionedTemporalPatchNet(Utr.shape[2], N_CH * o["k"], N_CLS, masks=masks(o["k"], o["bias"] > 0), seed=seed,
                                      initial_radius=o["radius"])
    T, L = Utr.shape[1], o["crop"]
    stats = {"attempted": 0, "updated": 0, "reasons": {}, "halvings": 0, "curve": []}
    for ep in range(o["epochs"]):
        te = time.time(); order = rng.permutation(len(Utr))
        for i in range(0, len(order), o["batch"]):
            j = order[i:i + o["batch"]]; s = rng.integers(0, T - L + 1, len(j))
            u = np.stack([Utr[a, b:b + L] for a, b in zip(j, s)])
            tgt = np.zeros((len(j), L, N_CLS)); tgt[np.arange(len(j)), :, ylearn[j]] = 1.0
            net.reset()
            r = net.observe(u, tgt, beta=o["beta"], rate=o["rate"])
            stats["attempted"] += 1; stats["updated"] += int(r.updated)
            stats["reasons"][r.reason] = stats["reasons"].get(r.reason, 0) + 1
            stats["halvings"] += r.contrast_halvings
        if (ep + 1) % o["eval_every"] == 0 or ep == o["epochs"] - 1:
            ktr, _ = kappa_acc(ytr, predict(net, Utr).argmax(1))
            kva, _ = kappa_acc(yva, predict(net, Uva).argmax(1))
            stats["curve"].append({"epoch": ep + 1, "kappa_train": ktr, "kappa_val": kva,
                                   "epoch_wall": time.time() - te})
            print(f"[s{subject} seed{seed}] ep {ep + 1} k_tr={ktr:.3f} k_va={kva:.3f} "
                  f"upd={stats['updated']}/{stats['attempted']} {time.time() - te:.0f}s", flush=True)
    sc = predict(net, Uva); pred = sc.argmax(1)
    hid = np.tanh(net.imagine(Uva[:8], state=np.zeros((8, net.hidden))).hidden)
    k, a = kappa_acc(yva, pred)
    return {"subject": subject, "seed": seed, "kappa_val": k, "acc_val": a,
            "settle_rate": stats["updated"] / max(stats["attempted"], 1), "train": stats,
            "hidden_activity_std": float(hid.std()), "pred": pred.tolist(), "y": yva.tolist(),
            "A_radius": float(np.abs(np.linalg.eigvals(net.parameters()["A"])).max()),
            "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ep_tpn"); ap.add_argument("--subjects", default="1")
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
            "settle_rate": float(np.mean([r["settle_rate"] for r in res])),
            "per_subject_kappa": {s: float(np.mean([r["kappa_val"] for r in res if r["subject"] == s])) for s in subs},
            "wall_s": time.time() - t0, "python": platform.python_version()}
    out = ROOT / "receipts" / "M9" / a.arm / f"{commit}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_text(json.dumps(res)); (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1)); print(out / "summary.json")


if __name__ == "__main__":
    main()
