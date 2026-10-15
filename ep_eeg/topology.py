"""M3 preregistered input-sensitivity tests (ADR-0006).

At a free equilibrium, perturb a lead's power-like feature (Track-2 Activity
or Track-1 log mu-power) by a small epsilon and measure the change in the
target outlet activation, averaged over validation trials of that class.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from eval.t_only_bnci import CHANNEL_NAMES
from .sandbox import SandboxConfig, train_sandbox

LEADS = CHANNEL_NAMES[:22]
# class indices: 0 left_hand, 1 right_hand, 2 feet, 3 tongue
PREDICTIONS = [
    # (name, lead, feature_kind, class_idx, expected_sign)
    ("C3_to_right", "C3", "activity", 1, -1),
    ("C4_to_left", "C4", "activity", 0, -1),
    ("Cz_to_feet", "Cz", "activity", 2, -1),
    ("C3_to_feet_weak", "C3", "activity", 2, +1),  # weak / exploratory ERS
    ("C4_to_feet_weak", "C4", "activity", 2, +1),
]
EPS = 0.05  # on soft-compressed feature scale


def feat_index(track: str, lead: str, kind: str) -> int:
    li = LEADS.index(lead)
    if track == "track2":
        # [act22, mob22, comp22]
        return {"activity": li, "mobility": 22 + li, "complexity": 44 + li}[kind]
    # track1: [mu22, beta22]
    return {"mu": li, "beta": 22 + li, "activity": li}[kind]


def sensitivity(sb, X, y, lead, kind, class_idx, eps=EPS):
    """Mean d(outlet[class])/d(feat) over trials with y==class_idx, qualified windows."""
    track = sb.cfg.track
    fi = feat_index(track, lead, kind)
    mask = y == class_idx
    if not mask.any():
        return {"mean": float("nan"), "n_trials": 0, "n_windows": 0}
    Xc = X[mask].copy()
    eq0 = sb.free_states(Xc)
    act0 = np.asarray(eq0.state.activation)[:, sb.outputs]
    ok0 = np.asarray(eq0.converged) & np.all(np.isfinite(act0), 1)
    Xp = Xc.copy()
    Xp[..., fi] = Xp[..., fi] + eps
    eq1 = sb.free_states(Xp)
    act1 = np.asarray(eq1.state.activation)[:, sb.outputs]
    ok1 = np.asarray(eq1.converged) & np.all(np.isfinite(act1), 1)
    ok = ok0 & ok1
    if not ok.any():
        return {"mean": float("nan"), "n_trials": int(mask.sum()), "n_windows": 0}
    d = (act1[ok, class_idx] - act0[ok, class_idx]) / eps
    # also report difference vs mean of other outlets
    other = [i for i in range(4) if i != class_idx]
    d_diff = ((act1[ok][:, class_idx] - act1[ok][:, other].mean(1))
              - (act0[ok][:, class_idx] - act0[ok][:, other].mean(1))) / eps
    return {
        "mean": float(d.mean()),
        "mean_vs_others": float(d_diff.mean()),
        "std": float(d.std()),
        "n_trials": int(mask.sum()),
        "n_windows": int(ok.sum()),
        "frac_neg": float((d < 0).mean()),
        "frac_pos": float((d > 0).mean()),
    }


def run_subject(train, val, seed, opts, predictions=PREDICTIONS):
    sb, stats, _ = train_sandbox(train, seed, **opts)
    if "failed" in stats:
        return {"failed": stats["failed"], "preds": {}}
    X = val[sb.cfg.track + "_x"]; y = val["y"]
    out = {}
    for name, lead, kind, cls, expected in predictions:
        s = sensitivity(sb, X, y, lead, kind, cls)
        s["expected_sign"] = expected
        s["holds"] = (np.sign(s["mean"]) == expected) if np.isfinite(s["mean"]) else False
        out[name] = s
    # untrained control: fresh sandbox, no fit
    ctrl = SandboxConfig(**{**opts})
    from .sandbox import Sandbox
    sb0 = Sandbox(ctrl, seed + 999)
    ctrl_out = {}
    for name, lead, kind, cls, expected in predictions:
        s = sensitivity(sb0, X, y, lead, kind, cls)
        s["expected_sign"] = expected
        ctrl_out[name] = s
    return {"failed": None, "preds": out, "untrained": ctrl_out,
            "cert_init": stats.get("cert_init"), "params": sb.n_params}


def main_cli():
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    from .data import load_subject, ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--opts", default='{"leak":1.0,"rowmass_project":0.9,"eta":0.5,"temperature":0.01}')
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--tag", default="m3")
    ap.add_argument("--group", default="g1")
    a = ap.parse_args()
    opts = json.loads(a.opts); opts["group"] = a.group; opts.setdefault("track", "track2")
    seeds = [int(x) for x in a.seeds.split(",")]; subs = [int(x) for x in a.subjects.split(",")]

    def one(sub, seed):
        d = load_subject(sub)
        r = run_subject(d["train"], d["val"], seed, opts)
        r.update(subject=sub, seed=seed)
        return r

    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        res = [f.result() for f in [ex.submit(one, s, k) for s in subs for k in seeds]]
    # aggregate: for each prediction, fraction of subject-seed where holds, mean sensitivity
    agg = {}
    for name, *_ in PREDICTIONS:
        vals = [r["preds"][name]["mean"] for r in res if r.get("preds") and name in r["preds"]
                and np.isfinite(r["preds"][name]["mean"])]
        holds = [r["preds"][name]["holds"] for r in res if r.get("preds") and name in r["preds"]]
        ctrl = [r["untrained"][name]["mean"] for r in res if r.get("untrained") and name in r["untrained"]
                and np.isfinite(r["untrained"][name]["mean"])]
        # per-subject mean then sign
        by_sub = {}
        for r in res:
            if not r.get("preds") or name not in r["preds"]: continue
            by_sub.setdefault(r["subject"], []).append(r["preds"][name]["mean"])
        sub_means = {s: float(np.nanmean(v)) for s, v in by_sub.items()}
        expected = next(p[4] for p in PREDICTIONS if p[0] == name)
        sub_holds = {s: (np.sign(m) == expected) for s, m in sub_means.items() if np.isfinite(m)}
        agg[name] = {
            "expected_sign": expected,
            "mean_sens": float(np.nanmean(vals)) if vals else None,
            "frac_holds_runs": float(np.mean(holds)) if holds else None,
            "n_subjects_hold": int(sum(sub_holds.values())),
            "n_subjects": len(sub_holds),
            "per_subject_mean": sub_means,
            "untrained_mean": float(np.nanmean(ctrl)) if ctrl else None,
            "verdict": (
                "成立" if sum(sub_holds.values()) >= 6 and abs(float(np.nanmean(vals) or 0)) > abs(float(np.nanmean(ctrl) or 0)) * 0.5
                else ("不成立" if sum(sub_holds.values()) <= 3 else "证据不足")
            ) if sub_holds else "证据不足",
        }
    out = {"opts": opts, "seeds": seeds, "subjects": subs, "agg": agg, "runs": res, "wall_s": time.time() - t0}
    dest = ROOT / "receipts" / "M3" / a.tag
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
    (dest / "runs.json").write_text(json.dumps(out["runs"]))
    print(json.dumps(out["agg"], indent=1, ensure_ascii=False))
    print("wrote", dest)


if __name__ == "__main__":
    main_cli()
