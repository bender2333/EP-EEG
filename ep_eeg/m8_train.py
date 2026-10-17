"""M8 trainer/runner: continuous-timeline EP (rawseq 22x25 drive per 100 ms step, warm chained),
5 outlets (4 MI + rest). Arms: ep_m8_single (per subject), ep_m8_triple (A01|A02|A03 shared net).

Training: the training timeline is cut into `batch` contiguous parallel streams (equal length, aligned to
segment boundaries where possible); each epoch walks every stream step by step. Every step is a
Learner.step warm-started from the previous step's free state; unlabelled steps get weight 0 (state still
advances, no pull). Warm resets to None at run/subject boundaries and after a refused step.
Scoring: per event (MI trial or rest gap) mean outlet activity over qualified steps -> argmax.
  kappa5 : 5-way over all MI trials + rest gaps.  kappa4_mi : MI trials only, argmax over outlets 0-3
  (comparable to LDA 0.389 on the same trials).
"""
from __future__ import annotations
import argparse, json, time, warnings, platform, subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score
from cadence.learning import LearningPhaseError
from .sandbox import Sandbox, SandboxConfig
from .m8_data import load_subject_m8, concat, REST
from .data import ROOT

M1_OPTS = dict(sigma_in=0.012, leak=1.0, rowmass_project=0.9, eta=0.5, temperature=0.01)
BASE = dict(track="rawseq", use_warm=True, n_classes=5, epochs=100, **M1_OPTS)


def streams(seg, B):
    n = len(seg); L = n // B
    return [np.arange(i * L, (i + 1) * L) for i in range(B)]


def fit(sb, D, log):
    cfg = sb.cfg; X, y, seg = D["x"], D["y"], D["segment"]
    S = streams(seg, cfg.batch); L = len(S[0]); idx = np.stack(S)  # (B,L)
    st = {"steps": 0, "refused": 0, "epochs": []}
    for ep in range(cfg.epochs):
        t0 = time.time(); warm = None
        for t in range(L):
            j = idx[:, t]
            if t > 0 and np.any(seg[j] != seg[idx[:, t - 1]]): warm = None
            yt = y[j]; w = (yt >= 0).astype(float); lab = np.where(yt >= 0, yt, 0)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if w.any():
                        s, _ = sb.learner.step(sb.drive(X[j]), lab, warm=warm, weight=w); warm = s.free
                        if cfg.rowmass_project > 0: sb.project_rowmass()
                    else:
                        warm = sb.learner.brain.equilibrate(sb.drive(X[j]), budget=cfg.free_steps,
                                                            tolerance=cfg.tolerance, damping=cfg.damping, state=warm).state
                st["steps"] += 1
            except LearningPhaseError:
                st["refused"] += 1; warm = None
        c = sb.cert(); c.update(epoch=ep, wall=time.time() - t0); st["epochs"].append(c)
        log(f"epoch {ep} cert={c['certified']} refused={st['refused']}/{st['steps']} wall={c['wall']:.1f}")
        if not c["certified"]: st["failed"] = f"certificate_epoch_{ep}"; break
    return st


def infer(sb, D):
    """Run the timeline as one stream per segment-chunk (batched by segment), warm chained."""
    X, seg = D["x"], D["segment"]
    act = np.zeros((len(X), sb.n_classes)); ok = np.zeros(len(X), bool)
    segs = np.unique(seg); L = max((seg == s).sum() for s in segs)
    rows = [np.where(seg == s)[0] for s in segs]; state = None
    for t in range(L):
        live = [r[min(t, len(r) - 1)] for r in rows]
        eq = sb.learner.brain.equilibrate(sb.drive(X[live]), budget=sb.cfg.free_steps,
                                          tolerance=sb.cfg.tolerance, damping=sb.cfg.damping, state=state)
        a = np.asarray(eq.state.activation)[:, sb.outputs]
        for i, r in enumerate(rows):
            if t < len(r): act[r[t]] = a[i]; ok[r[t]] = bool(eq.converged[i]) and np.isfinite(a[i]).all()
        state = eq.state if np.isfinite(np.asarray(eq.state.v)).all() else None
    return act, ok


def score(D, act, ok):
    ev = D["event"]; m = ev >= 0; ids = np.unique(ev[m])
    yt, p5, p4, mi = [], [], [], []
    for e in ids:
        r = np.where(ev == e)[0]; q = r[ok[r]] if ok[r].any() else r
        a = act[q].mean(0); yt.append(D["y"][r[0]]); p5.append(a.argmax()); p4.append(a[:4].argmax())
    yt, p5, p4 = map(np.array, (yt, p5, p4)); mi = yt != REST
    return {"kappa5": float(cohen_kappa_score(yt, p5)), "kappa4_mi": float(cohen_kappa_score(yt[mi], p4[mi])),
            "acc5": float((yt == p5).mean()), "n_mi": int(mi.sum()), "n_rest": int((~mi).sum()),
            "settle_rate": float(ok.mean())}


def one(arm, subject_tag, seed, opts):
    t0 = time.time()
    if arm == "ep_m8_single":
        d = load_subject_m8(subject_tag, opts.pop("split", "t04_val45")); subs = [subject_tag]
    else:
        subs = [1, 2, 3]; sp = opts.pop("split", "t04_val45")
        d = concat([load_subject_m8(s, sp) for s in subs], subs)
    sb = Sandbox(SandboxConfig(**{**BASE, **opts}), seed); logs = []
    st = fit(sb, d["train"], lambda m: (logs.append(m), print(f"[{arm} {subs} s{seed}] {m}", flush=True))); res = {"arm": arm, "subjects": subs, "seed": seed,
                                                  "train": st, "log": logs, "train_wall": time.time() - t0}
    if "val" in d:
        act, ok = infer(sb, d["val"]); res["val"] = score(d["val"], act, ok)
        if len(subs) > 1:
            res["val_per_subject"] = {s: score({k: v[d["val"]["subject"] == s] for k, v in d["val"].items()},
                                               act[d["val"]["subject"] == s], ok[d["val"]["subject"] == s]) for s in subs}
    res["wall_s"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["ep_m8_single", "ep_m8_triple"])
    ap.add_argument("--subjects", default="1-9"); ap.add_argument("--seeds", default="0")
    ap.add_argument("--opts", default="{}"); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(); opts = json.loads(a.opts)
    lo, _, hi = a.subjects.partition("-"); subs = list(range(int(lo), int(hi or lo) + 1))
    seeds = [int(s) for s in a.seeds.split(",")]
    jobs = [(s, k) for s in subs for k in seeds] if a.arm == "ep_m8_single" else [(0, k) for k in seeds]
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, *zip(*[(a.arm, s, k, dict(opts)) for s, k in jobs])))
    summ = {"commit": commit, "stage": "M8", "arm": a.arm, "opts": {**BASE, **opts}, "wall_s": time.time() - t0,
            "python": platform.python_version()}
    if all("val" in r for r in res):
        for key in ("kappa5", "kappa4_mi", "settle_rate"): summ[key] = float(np.mean([r["val"][key] for r in res]))
    out = ROOT / "receipts" / "M8" / a.arm / f"{commit}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_text(json.dumps(res, default=str)); (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
