"""M15 block 1: odd decision vs LDA vs backprop logistic vs an EP linear head.

Frozen states come from the M14 seed-0 checkpoints. The EP head is a new one-way
map from those states; it does not change the checkpoint. T session only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from . import m14 as M
from . import m15_common as C
from .m14_probe import layer_index, lda_scores, settle_split, trial_mean

ARMS = ("C", "C_nolake", "C_skip", "A")
HEAD_U = {"outlets": 400, "pools": 600, "columns": 800}
LAYER_SEED = {"outlets": 1, "pools": 2, "columns": 3}


def geometry(act, ok, y):
    tm = trial_mean(act, ok)
    y = np.asarray(y)
    correct = tm[np.arange(len(y)), y]
    rival = tm.copy()
    rival[np.arange(len(y)), y] = -np.inf
    margin = correct - rival.max(1)
    flat = act.reshape(-1, act.shape[-1])
    std = flat.std(0)
    return {"margin_mean": float(margin.mean()), "margin_median": float(np.median(margin)),
            "sat_frac": float((np.abs(act) > 0.8).mean()), "dead_outlets": int((std < 1e-3).sum()),
            "outlet_std": [float(v) for v in std], "correct_mean": float(correct.mean()),
            "rival_mean": float(np.mean(rival.max(1)))}


def zscore(Ftr, Fte):
    mu, sd = Ftr.mean(0), Ftr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (Ftr - mu) / sd, (Fte - mu) / sd


def ep_on_trials(Ftr, ytr, Fte, yte, seed, U):
    h = C.build_head(Ftr.shape[1], seed=17000 + seed)
    info = C.train_head(h, Ftr, ytr, U, seed)
    pred, _ = C.head_activity(h, Fte)
    pred_tr, _ = C.head_activity(h, Ftr)
    return {**C.kappa_acc(yte, pred.argmax(1)), "train_acc": float((pred_tr.argmax(1) == ytr).mean()),
            "updates": info["updates"]}


def one(arm, subject, seed=0):
    g, meta = C.load_trained(arm, subject, seed)
    data = C.load_t(subject)
    tr_feats, tr_sc = settle_split(g, data["Xtr"], data["ytr"], data["ytr"])
    va_feats, va_sc = settle_split(g, data["Xte"], data["yte"], data["ytr"])
    # rebuild outlet activities for geometry: settle_split already reduced them.
    # geometry needs the sub-window tensor, so settle once more for outlets only.
    n = len(data["Xte"])
    A, ok = M.settle(g, data["Xte"].reshape(n * M.N_SUB, -1))
    ok = ok.reshape(n, M.N_SUB)
    out_act = A[:, g["out"]].reshape(n, M.N_SUB, len(g["out"]))
    layers = {}
    ytr, yte = data["ytr"], data["yte"]
    for name in tr_feats:
        lda = lda_scores(tr_feats[name], ytr, va_feats[name], yte)
        bp_te, bp_tr = C.logreg_predict(tr_feats[name], ytr, va_feats[name])
        bp = {**C.kappa_acc(yte, bp_te), "train_acc": float((bp_tr == ytr).mean())}
        ep = ep_on_trials(tr_feats[name], ytr, va_feats[name], yte,
                          1000 * subject + 50 * seed + LAYER_SEED[name], HEAD_U[name])
        # Exploratory, added after the A01 smoke: outlet activities are ~1e-3, so a
        # default logistic penalty and an EP head scaled for O(1) inputs both stall.
        # Z-scoring uses training statistics only.
        Ztr, Zte = zscore(tr_feats[name], va_feats[name])
        zp_te, zp_tr = C.logreg_predict(Ztr, ytr, Zte)
        bp_z = {**C.kappa_acc(yte, zp_te), "train_acc": float((zp_tr == ytr).mean())}
        ep_z = ep_on_trials(Ztr, ytr, Zte, yte, 5000 + 1000 * subject + LAYER_SEED[name], HEAD_U[name])
        layers[name] = {"lda": lda, "bp": bp, "ep": ep, "bp_z": bp_z, "ep_z": ep_z,
                        "feature_std": [float(v) for v in tr_feats[name].std(0)],
                        "n_features": int(tr_feats[name].shape[1]),
                        "scale_note": "bp_z and ep_z are exploratory z-scored features"}
    decision = {"acc": va_sc["acc"], "kappa": va_sc["kappa"], "train_acc": tr_sc["acc"],
                "train_kappa": tr_sc["kappa"], "settle_rate": va_sc["settle_rate"],
                "rejected_trials": va_sc["rejected_trials"]}
    rec = {**meta, "stage": "M15", "block": "readout", "split": "val", "dev": False,
           "decision": decision, "geometry": geometry(out_act, ok, yte), "layers": layers}
    path = C.OUT / "readout" / f"{arm}_A{subject:02d}_s{seed}.json"
    C.write_json(path, rec)
    return rec


def _col(rows, arm, layer, kind, field):
    return [r["layers"][layer][kind][field] for r in rows if r["arm"] == arm and layer in r["layers"]]


def summarize(rows):
    """rows: one receipt per (arm, subject), subjects 1..9 in order."""
    out = {}
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if len(sub) < 9:
            continue
        sub = sorted(sub, key=lambda r: r["subject"])
        block = {"decision_acc": [r["decision"]["acc"] for r in sub],
                 "decision_kappa": [r["decision"]["kappa"] for r in sub],
                 "geometry_margin": float(np.mean([r["geometry"]["margin_mean"] for r in sub])),
                 "geometry_sat": float(np.mean([r["geometry"]["sat_frac"] for r in sub])),
                 "geometry_dead": float(np.mean([r["geometry"]["dead_outlets"] for r in sub]))}
        for layer in ("outlets", "pools", "columns"):
            if layer not in sub[0]["layers"]:
                continue
            lda = _col(sub, arm, layer, "lda", "acc_val")
            bp = _col(sub, arm, layer, "bp", "acc")
            ep = _col(sub, arm, layer, "ep", "acc")
            bp_z = _col(sub, arm, layer, "bp_z", "acc")
            ep_z = _col(sub, arm, layer, "ep_z", "acc")
            dec = block["decision_acc"]
            block[layer] = {
                "lda_acc": lda, "bp_acc": bp, "ep_acc": ep, "bp_z_acc": bp_z, "ep_z_acc": ep_z,
                "lda_kappa": _col(sub, arm, layer, "lda", "kappa_val"),
                "bp_kappa": _col(sub, arm, layer, "bp", "kappa"),
                "ep_kappa": _col(sub, arm, layer, "ep", "kappa"),
                "bp_z_kappa": _col(sub, arm, layer, "bp_z", "kappa"),
                "ep_z_kappa": _col(sub, arm, layer, "ep_z", "kappa"),
                "lda_vs_bp": C.paired(lda, bp), "bp_vs_decision": C.paired(bp, dec),
                "lda_vs_decision": C.paired(lda, dec), "ep_vs_bp": C.paired(ep, bp),
                "bp_z_vs_lda": C.paired(bp_z, lda), "ep_z_vs_bp_z": C.paired(ep_z, bp_z)}
        out[arm] = block
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    seal = C.seal_or_stop()
    done = []
    for arm in a.arms:
        for s in a.subjects:
            path = C.OUT / "readout" / f"{arm}_A{s:02d}_s{a.seed}.json"
            if path.exists():
                import json
                done.append(json.loads(path.read_text()))
                print("exists", path.name, flush=True)
                continue
            rec = one(arm, s, a.seed)
            done.append(rec)
            d = rec["decision"]
            print(f"{arm} A{s:02d} decision {d['acc']:.3f} " +
                  " ".join(f"{k}:lda {v['lda']['acc_val']:.3f} bp {v['bp']['acc']:.3f} ep {v['ep']['acc']:.3f}"
                           for k, v in rec["layers"].items()), flush=True)
    summary = summarize(done)
    C.write_json(C.OUT / "readout" / "summary.json", {"stage": "M15", "block": "readout", "e_seal": seal, "arms": summary})
    print("wrote", C.OUT / "readout" / "summary.json", flush=True)


if __name__ == "__main__":
    main()
