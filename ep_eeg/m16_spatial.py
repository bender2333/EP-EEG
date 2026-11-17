"""M16 block 2: CSP features (cheap) and a CSP-initialized spatial front-end. T session only."""
from __future__ import annotations
import argparse
import numpy as np

from . import m14 as M
from . import m15_common as C
from .m14_probe import lda_scores

BANDS = [(lo, lo + 4) for lo in range(4, 36, 4)]


def _bandpass(seg, lo, hi):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [lo, min(hi, 124.0)], btype="band", fs=250, output="sos")
    return sosfiltfilt(sos, seg.astype(np.float64), axis=-1)


def csp_bank(train_seg, y, test_seg):
    """8 bands x 4 CSP components, fit on the training trials only. Returns (n, 32) features."""
    from mne.decoding import CSP
    cols_tr, cols_te = [], []
    for lo, hi in BANDS:
        xtr, xte = _bandpass(train_seg, lo, hi), _bandpass(test_seg, lo, hi)
        csp = CSP(n_components=4, log=True).fit(xtr, y)
        cols_tr.append(csp.transform(xtr))
        cols_te.append(csp.transform(xte))
    return np.hstack(cols_tr), np.hstack(cols_te)


def broadband_csp(seg, y):
    """22 spatial filters on 8-30 Hz, shape (22, 22): row is a filter, column a lead."""
    from mne.decoding import CSP
    x = _bandpass(seg, 8.0, 30.0)
    n = min(22, x.shape[0] - 1)
    csp = CSP(n_components=n, log=False).fit(x, y)
    W = np.asarray(csp.filters_, float)
    if W.shape != (22, 22):
        full = np.eye(22)
        full[:W.shape[0]] = W
        W = full
    return W


def set_front(g, W):
    """Write W[k, c] onto the tied front-end synapses. Does not touch other pathways."""
    eff = np.array(g["lr"].brain.efficacy, float)
    for i, gid in enumerate(g["grp"]):
        if gid >= 0:
            eff[i] = W[int(gid) // M.NL, int(gid) % M.NL]
    bias = np.array(g["lr"].brain.bias, float)
    g["lr"].brain = g["lr"].brain.with_parameters(efficacy=eff, bias=bias)
    return g


def match_norm(W, target):
    n = np.linalg.norm(W)
    return W * (np.linalg.norm(target) / n) if n > 0 else W


def freeze_front(g):
    g["lr"].plastic_synapses[np.asarray(g["grp"]) >= 0] = False
    g["pl"][np.asarray(g["grp"]) >= 0] = False
    return g


def features_one(subject, seed=0):
    from ep_eeg.data import load_subject
    raw = load_subject(subject)
    ytr = np.asarray(raw["train"]["y"])
    yte = np.asarray(raw["val"]["y"])
    Ftr, Fte = csp_bank(raw["train"]["seg"], ytr, raw["val"]["seg"])
    lda = lda_scores(Ftr, ytr, Fte, yte)
    mu, sd = Ftr.mean(0), Ftr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Ztr, Zte = (Ftr - mu) / sd, (Fte - mu) / sd
    h = C.build_head(Ztr.shape[1], seed=21000 + subject + 10 * seed)
    info = C.train_head(h, Ztr, ytr, U=800, seed=seed, batch=32)
    pred, _ = C.head_activity(h, Zte)
    ep = {**C.kappa_acc(yte, pred.argmax(1)), "train_acc": info["train_acc"], "updates": info["updates"]}
    rec = {"stage": "M16", "block": "spatial_features", "subject": subject, "seed": seed, "split": "val",
           "exploratory": False, "lda_acc": lda["acc_val"], "lda_kappa": lda["kappa_val"],
           "ep_acc": ep["acc"], "ep_kappa": ep["kappa"], "ep_train_acc": ep["train_acc"]}
    C.write_json(C.OUT.parent / "M16" / "spatial" / f"csp32_A{subject:02d}_s{seed}.json", rec)
    print(f"A{subject:02d} csp32 lda {lda['acc_val']:.3f} ep {ep['acc']:.3f}", flush=True)
    return rec


def front_one(kind, subject, seed, U):
    data_raw_y = None
    from ep_eeg.data import load_subject
    raw = load_subject(subject)
    g = M.build("C", 14500 + 100 * subject + seed, eta=0.3)
    W_car = M.front_matrix(g)
    if kind == "rand_front":
        rng = np.random.default_rng(14600 + subject + 20 * seed)
        W = match_norm(rng.normal(size=(M.NL, M.NL)), W_car)
    else:
        W = match_norm(broadband_csp(raw["train"]["seg"], np.asarray(raw["train"]["y"])), W_car)
    set_front(g, W)
    if kind == "csp_frozen":
        freeze_front(g)
    if not M.certificate(g).get("certified"):
        rec = {"stage": "M16", "block": "spatial_net", "kind": kind, "subject": subject, "seed": seed,
               "U": U, "crash": "cert"}
        C.write_json(C.OUT.parent / "M16" / "spatial" / f"{kind}_U{U}_A{subject:02d}_s{seed}.json", rec)
        return rec
    data = C.load_t(subject)
    tr = M.train(g, data, U, seed, eval_every=min(250, U),
                 log=lambda m, k=kind, s=subject: print(f"[{k} A{s:02d}] {m}", flush=True))
    ev, _, _ = M.evaluate(g, data["Xte"], data["yte"], data["ytr"])
    ev.pop("pred", None)
    Wend = M.front_matrix(g)
    def _cos(a, b):
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    rec = {"stage": "M16", "block": "spatial_net", "kind": kind, "subject": subject, "seed": seed,
           "U": U, "split": "val", "crash": None, "test": ev,
           "front_rel": float(np.linalg.norm(Wend - W) / (np.linalg.norm(W) + 1e-12)),
           "cos_init": _cos(Wend, W), "cos_car": _cos(Wend, W_car),
           "train_refused": tr["refused"], "certified_end": tr["certified_end"]}
    C.write_json(C.OUT.parent / "M16" / "spatial" / f"{kind}_U{U}_A{subject:02d}_s{seed}.json", rec)
    print(f"{kind} A{subject:02d} acc {ev['acc']:.3f} rel {rec['front_rel']:.3f} cos_init {rec['cos_init']:.3f}", flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--nets", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=["csp_train", "csp_frozen", "rand_front"])
    ap.add_argument("--subjects", nargs="+", type=int, default=[1, 3, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--U", type=int, default=1500)
    a = ap.parse_args(argv)
    C.seal_or_stop()
    if a.features:
        subs = a.subjects if a.subjects != [1, 3, 8] else list(range(1, 10))
        # explicit --subjects still wins; default for features is all 9 when flag set alone
        if a.subjects == [1, 3, 8] and "--subjects" not in __import__("sys").argv:
            subs = list(range(1, 10))
        for s in subs:
            path = C.OUT.parent / "M16" / "spatial" / f"csp32_A{s:02d}_s{a.seed}.json"
            if path.exists():
                print("exists", path.name, flush=True)
                continue
            features_one(s, a.seed)
    if a.nets:
        for kind in a.kinds:
            for s in a.subjects:
                path = C.OUT.parent / "M16" / "spatial" / f"{kind}_U{a.U}_A{s:02d}_s{a.seed}.json"
                if path.exists():
                    print("exists", path.name, flush=True)
                    continue
                front_one(kind, s, a.seed, a.U)


if __name__ == "__main__":
    main()
