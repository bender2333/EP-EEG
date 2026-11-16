"""Layer-wise linear probe of M14 free-phase equilibrium. T session only.

Forward diagnostic: settle() on T training runs 0–3 and T validation runs 4–5, read the
equilibrium activation of each layer, average the qualified sub-windows the same way
``eval.scoring.aggregate`` averages the outlets, and fit a shrinkage LDA
(``LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")``) on the training trials.

Layers, when the arm has them: pools (22), columns (88), outlets (4). The arm's own outlet
decision (frozen aggregation, not the LDA) is reported as layer ``readout``.

Weights: ``receipts/M14/weights/<arm>_A<ss>_s<seed>_eta<eta>_T<temp>/u<U>.npz`` if that file
exists, otherwise retrain with the selected (eta, U) and seed. Retraining does not write a
grid receipt and does not evaluate between updates; evaluation does not move parameters, so
the saved ``u<U>.npz`` matches a normal ``m14`` run at that update.

``--dev`` uses a synthetic segment, seed subject 0, and never calls the data loader.
"""
from __future__ import annotations
import argparse, csv, json, os, time, warnings
from pathlib import Path
import numpy as np

from . import m14 as M

PROBE_ARMS = ("C", "C_nolake", "A")
# Frozen T-validation selection (reports/M14.md §2, receipts/M14/selected.json).
# C_skip is not in that file: it is matched to C so the skip is the only change.
FROZEN_SELECTION = {
    "A": {"eta": 1.0, "U": 3000},
    "B": {"eta": 1.0, "U": 1750},
    "C": {"eta": 0.3, "U": 3000},
    "nolake": {"eta": 0.3, "U": 500},
    "C_nolake": {"eta": 0.3, "U": 2000},
    "randlake": {"eta": 1.0, "U": 500},
    "C_skip": {"eta": 0.3, "U": 3000},
}
LAYER_ZH = {"pools": "池", "columns": "微柱", "outlets": "出水口", "readout": "出水口判决"}
TSV_FIELDS = ("arm", "subject", "layer", "n_features", "acc_val", "kappa_val", "acc_train", "kappa_train",
              "settle_rate_val", "settle_rate_train", "rejected_val", "rejected_train", "eta", "U", "seed",
              "weight_source", "dev")


def selection():
    """Preregistered (eta, U) from selected.json when it is present, else the frozen copy."""
    sel = {k: dict(v) for k, v in FROZEN_SELECTION.items()}
    path = M.OUT / "selected.json"
    if path.is_file():
        disk = json.loads(path.read_text())
        for arm, c in disk.items():
            if arm in M.PREREG_ARMS:
                sel[arm] = {"eta": float(c["eta"]), "U": int(c["U"])}
    return sel


def weight_file(arm, subject, seed, eta, U, temp=None):
    temp = M.TEMP if temp is None else temp
    wdir = M.OUT / "weights" / (f"{arm}_A{subject:02d}_s{seed}_eta{eta:g}_T{temp:g}" + os.environ.get("M14_WTAG", ""))
    return wdir / f"u{U}.npz"


def e_seal_or_stop():
    from ep_eeg.m13_common import e_seal
    seal = e_seal()
    if seal["matches"]:
        raise RuntimeError(f"E files present during a T-only probe: {seal['matches'][:3]}")
    return seal


def load_t(subject):
    """T runs 0–3 / 4–5. The only session split this module requests."""
    e_seal_or_stop()
    return M.load(subject, "val")


def synth_bundle(n_train, n_val, seed, T=750):
    """DEV ONLY. Class-conditional rhythms on a few leads, then the same window encoder as M14."""
    def make(n, s):
        rng = np.random.default_rng(s)
        y = np.tile(np.arange(4), int(np.ceil(n / 4)))[:n]
        seg = rng.normal(scale=0.4, size=(n, M.NL, T))
        t = np.linspace(0.0, 1.0, T, endpoint=False)
        for i, c in enumerate(y):
            f = 8.0 + 3.0 * int(c)
            seg[i, int(c)] += np.sin(2 * np.pi * f * 4.0 * t)
            seg[i, 10 + int(c)] += 0.5 * np.cos(2 * np.pi * (f / 2.0) * 4.0 * t)
        return seg, y.astype(int)
    tr, ytr = make(n_train, 10_000 + seed)
    te, yte = make(n_val, 20_000 + seed)
    return {"Xtr": M.windows(tr), "ytr": ytr, "Xte": M.windows(te), "yte": yte,
            "source": f"synthetic dev seed {seed}"}


def trial_mean(act, ok):
    """(n, n_sub, d), (n, n_sub) -> (n, d). Qualified-sub-window mean, same reduction as eval.scoring.aggregate."""
    w = np.asarray(ok, float)[..., None]
    return (act * w).sum(1) / np.maximum(w.sum(1), 1.0)


def layer_index(g):
    layers = []
    if g["pools"] is not None:
        layers.append(("pools", np.asarray(g["pools"], int)))
    if len(g["cols"]):
        layers.append(("columns", np.asarray(g["cols"], int)))
    layers.append(("outlets", np.asarray(g["out"], int)))
    return layers


def settle_split(g, X, y, ytr):
    from eval.scoring import aggregate, fallback_class, score
    n = len(X)
    A, ok = M.settle(g, X.reshape(n * M.N_SUB, -1))
    ok = ok.reshape(n, M.N_SUB)
    feats = {}
    for name, idx in layer_index(g):
        feats[name] = trial_mean(A[:, idx].reshape(n, M.N_SUB, idx.size), ok)
    pred, _rej = aggregate(A[:, g["out"]].reshape(n, M.N_SUB, 4), ok, fallback_class(ytr))
    return feats, score(y, pred, ok)


def lda_scores(Ftr, ytr, Fte, yte):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Ftr, ytr)
        pred_te, pred_tr = clf.predict(Fte), clf.predict(Ftr)
    return {"n_features": int(Ftr.shape[1]),
            "acc_val": float(accuracy_score(yte, pred_te)), "kappa_val": float(cohen_kappa_score(yte, pred_te)),
            "acc_train": float(accuracy_score(ytr, pred_tr)), "kappa_train": float(cohen_kappa_score(ytr, pred_tr))}


def ensure_graph(arm, subject, seed, eta, U, data, dev):
    """Load the checkpoint at U, or retrain on ``data`` and (unless dev) save it."""
    init = 14500 + 100 * subject + seed
    g = M.build(arm, init, eta=eta)
    path = weight_file(arm, subject, seed, eta, U)
    if path.is_file() and not dev:
        M.load_weights(g, path)
        return g, "npz"
    if U > 0:
        M.train(g, data, U, seed, eval_every=max(int(U), 1), log=lambda m: None, eval_during=False)
    if not dev:
        M.save_weights(g, path)
        return g, "retrain"
    return g, "dev-retrain" if U else "dev-init"


def probe_one(arm, subject, data, seed, eta, U, dev):
    g, source = ensure_graph(arm, subject, seed, eta, U, data, dev)
    tr, tr_sc = settle_split(g, data["Xtr"], data["ytr"], data["ytr"])
    va, va_sc = settle_split(g, data["Xte"], data["yte"], data["ytr"])
    base = {"arm": arm, "subject": subject, "eta": eta, "U": U, "seed": seed, "weight_source": source, "dev": bool(dev)}
    rows = []
    for name in tr:
        rows.append({**base, **lda_scores(tr[name], data["ytr"], va[name], data["yte"]),
                     "layer": name, "settle_rate_val": va_sc["settle_rate"], "settle_rate_train": tr_sc["settle_rate"],
                     "rejected_val": va_sc["rejected_trials"], "rejected_train": tr_sc["rejected_trials"]})
    rows.append({**base, "layer": "readout", "n_features": 4,
                 "acc_val": va_sc["acc"], "kappa_val": va_sc["kappa"], "acc_train": tr_sc["acc"], "kappa_train": tr_sc["kappa"],
                 "settle_rate_val": va_sc["settle_rate"], "settle_rate_train": tr_sc["settle_rate"],
                 "rejected_val": va_sc["rejected_trials"], "rejected_train": tr_sc["rejected_trials"]})
    return rows


def with_means(rows):
    """Append one ``subject=mean`` row per (arm, layer). Rejected counts are sums; other metrics are means."""
    groups = {}
    for r in rows:
        groups.setdefault((r["arm"], r["layer"]), []).append(r)
    out = list(rows)
    rate = ("acc_val", "kappa_val", "acc_train", "kappa_train", "settle_rate_val", "settle_rate_train")
    for (arm, layer), rs in groups.items():
        m = {k: float(np.mean([r[k] for r in rs])) for k in rate}
        m["rejected_val"] = int(sum(r["rejected_val"] for r in rs))
        m["rejected_train"] = int(sum(r["rejected_train"] for r in rs))
        out.append({**rs[0], **m, "subject": "mean", "n_subjects": len(rs)})
    return out


def write_tsv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TSV_FIELDS, extrasaction="ignore", delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TSV_FIELDS})
    return path


def write_md(path, rows, meta):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    means = [r for r in rows if r["subject"] == "mean"]
    lines = ["# 分层线性探针", ""]
    if meta.get("dev"):
        lines += ["本表是合成数据上的开发运行（`dev=true`），不进任何判定，不能拿来读 M14 的损失发生在哪一层。", ""]
    lines += [f"数据：{meta.get('source', '')}。种子 {meta['seed']}。收缩 LDA 在训练试次上拟合，下表是验证集。",
              "合格率是自由相子窗收敛比例，拒判是没有合格子窗的试次（均值行的拒判是合计）。",
              "`readout` 是出水口自身的冻结聚合，不是 LDA。", ""]
    lines += ["| 臂 | 层 | 特征数 | 验证 acc | 验证 κ | 训练 κ | 合格率 | 拒判 | 权重 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in means:
        lines.append(f"| {r['arm']} | {LAYER_ZH.get(r['layer'], r['layer'])} | {r['n_features']} | {r['acc_val']:.3f} | "
                     f"{r['kappa_val']:.3f} | {r['kappa_train']:.3f} | {r['settle_rate_val']:.3f} | {r['rejected_val']} | {r['weight_source']} |")
    lines += ["", "每人的数在同目录的 TSV 里。读法写在 `reports/M14_diag.md`，不在这里改。", ""]
    path.write_text("\n".join(lines))
    return path


def run(arms, subjects, seed, eta, U, dev, n_train, n_val, out_dir):
    t0 = time.time()
    seal = e_seal_or_stop()
    sel = selection()
    out_dir = Path(out_dir)
    rows, sources = [], []
    for arm in arms:
        if arm not in sel:
            raise KeyError(arm)
        use_eta = sel[arm]["eta"] if eta is None else eta
        use_U = (1 if dev else sel[arm]["U"]) if U is None else U
        for subject in subjects:
            data = synth_bundle(n_train, n_val, seed + subject) if dev else load_t(subject)
            got = probe_one(arm, subject, data, seed, use_eta, use_U, dev)
            rows.extend(got)
            sources.append({"arm": arm, "subject": subject, "weight_source": got[0]["weight_source"], "eta": use_eta, "U": use_U})
            print(f"{arm} A{subject:02d} " + " ".join(f"{r['layer']}={r['kappa_val']:.3f}" for r in got), flush=True)
    rows = with_means(rows)
    meta = {"stage": "M14", "step": "diag-probe", "dev": bool(dev), "split": "val", "seed": seed,
            "arms": list(arms), "subjects": list(subjects), "sources": sources, "e_seal": seal,
            "source": "synthetic dev" if dev else "T runs 0-3 / 4-5",
            "wall_s": None}
    write_tsv(out_dir / "probe.tsv", rows)
    write_md(out_dir / "probe.md", rows, meta)
    meta["wall_s"] = time.time() - t0
    meta["rows"] = rows
    from ep_eeg.m13_common import write_receipt
    write_receipt(out_dir / "probe.json", meta)
    print(f"wrote {out_dir / 'probe.tsv'}  wall {meta['wall_s']:.1f}s", flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="M14 layer-wise LDA probe (T session only)")
    ap.add_argument("--arms", nargs="+", default=["C", "C_nolake"], choices=M.ARMS)
    ap.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eta", type=float, default=None, help="override every arm; default is the selected eta")
    ap.add_argument("--U", type=int, default=None, help="override every arm; default is the selected U (1 if --dev)")
    ap.add_argument("--dev", action="store_true", help="synthetic subject 0, dev=true, no session loader")
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    subjects = [0] if a.dev else a.subjects
    out = a.out or str(M.OUT / "diag" / ("dev" if a.dev else "probe"))
    run(a.arms, subjects, a.seed, a.eta, a.U, a.dev, a.n_train, a.n_val, out)


if __name__ == "__main__":
    main()
