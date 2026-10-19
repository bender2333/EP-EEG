"""M11 step A (reports/M11.md §3): baseline calibration under the frozen protocol.

Arms (T runs 0-3 train, runs 4-5 val; frozen eval-layer segment cue+0.5..3.5 s, EOG-regressed, 4-35 Hz, z):
  mdm            pyRiemann Covariances(oas) -> MDM(riemann)
  tslr           Covariances(oas) -> TangentSpace(riemann) -> LogisticRegression(C=1.0)   (MOABB TSLR.yml)
  eegnet_moabb   braindecode EEGNet with MOABB's Keras_EEGNet_8_2 pipeline: 128 Hz, StandardScaler_Epoch
                 (as MOABB's code does it: per trial, each time sample standardised across channels),
                 F1 8 / D 2 / F2 16 / kernel 64 / dropout 0.5 / final max-norm 0.5; Adam 1e-3, batch 64,
                 <= 300 epochs, 20% stratified inner hold-out for val_loss, EarlyStopping 75 (last weights),
                 ReduceLROnPlateau 75 x 0.5
  eegnet_bd      braindecode EEGNet defaults, 250 Hz; braindecode BCIC IV 2a trialwise example training:
                 AdamW lr 0.0625*0.01, wd 0, batch 64, CosineAnnealingLR(T_max=n_epochs-1), n_epochs 100,
                 EarlyStopping patience 10 with load_best on a 20% stratified inner hold-out
Diagnostic (not a criterion): mdm_moabbpp / tslr_moabbpp on MOABB's default MI preprocessing
(8-32 Hz 4th-order Butterworth forward-backward, cue+0..4 s, no EOG regression), T session only.
Non-EP baselines have no equilibrium solve: settle 1.0, 0 rejections.
"""
from __future__ import annotations
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score
from .data import load_subject, ROOT
from .m11_common import write_receipt, git, M11_OUT

OUT = M11_OUT / "A"
ARMS = ("mdm", "tslr", "eegnet_moabb", "eegnet_bd", "mdm_moabbpp", "tslr_moabbpp")


def riemann(Xtr, ytr, Xva, kind):
    from pyriemann.estimation import Covariances
    from pyriemann.classification import MDM
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    clf = (make_pipeline(Covariances("oas"), MDM(metric="riemann")) if kind == "mdm" else
           make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), LogisticRegression(C=1.0, max_iter=1000)))
    clf.fit(Xtr.astype(np.float64), ytr); return clf.predict(Xva.astype(np.float64))


def moabb_pp(subject):
    """MOABB MotorImagery defaults on the T session only: 8-32 Hz IIR (4th-order Butterworth, zero phase), cue+0..4 s."""
    from scipy.signal import butter, sosfiltfilt
    from eval.t_only_bnci import load_t_session
    from eval import preprocess as P
    sess, _ = load_t_session(subject, path="/workspace/mne_data"); sos = butter(4, [8, 32], btype="band", fs=250, output="sos")
    out = {}
    for part, runs in (("train", range(4)), ("val", (4, 5))):
        xs, ys = [], []
        for i in runs:
            eeg, eog, on, lab = P.run_arrays(sess["0train"][str(i)]); f = sosfiltfilt(sos, eeg, axis=1)
            for o, l in zip(on, lab): xs.append(f[:, o + 500:o + 1501]); ys.append(l)
        out[part] = (np.asarray(xs), np.asarray(ys))
    return out


def inner_split(y, seed, frac=0.2):
    """Stratified inner hold-out (seed-determined) from the training trials: (fit mask, hold-out mask)."""
    m = np.zeros(len(y), bool); rng = np.random.default_rng(seed)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c); m[rng.choice(idx, int(round(frac * len(idx))), replace=False)] = True
    return ~m, m


def eegnet(Xtr, ytr, Xva, seed, kind):
    import torch
    from braindecode.models import EEGNet
    from scipy.signal import resample_poly
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(1)
    if kind == "eegnet_moabb":
        prep = lambda X: (lambda Z: (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-12))(resample_poly(X.astype(np.float64), 64, 125, axis=-1))
        model = EEGNet(n_chans=22, n_outputs=4, n_times=prep(Xtr[:1]).shape[-1], F1=8, D=2, F2=16, kernel_length=64,
                       drop_prob=0.5, final_layer_with_constraint=True, norm_rate=0.5)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3); epochs, patience = 300, 75
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=75); restore = False
    else:
        prep = lambda X: X.astype(np.float64)
        model = EEGNet(n_chans=22, n_outputs=4, n_times=Xtr.shape[-1])
        epochs, patience = 100, 10
        opt = torch.optim.AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - 1); restore = True
    import os
    epochs = int(os.environ.get("M11_SMOKE_EPOCHS", epochs))  # code smoke tests only
    tr, ho = inner_split(ytr, seed)
    A = torch.tensor(prep(Xtr), dtype=torch.float32); Y = torch.tensor(ytr, dtype=torch.long)
    V = torch.tensor(prep(Xva), dtype=torch.float32)
    lossf = torch.nn.CrossEntropyLoss(); rng = np.random.default_rng(seed); best, best_state, wait, hist = np.inf, None, 0, []
    itr = np.flatnonzero(tr); iho = np.flatnonzero(ho)
    for ep in range(epochs):
        model.train(); perm = rng.permutation(itr)
        for b in range(0, len(perm), 64):
            j = perm[b:b + 64]; opt.zero_grad(); l = lossf(model(A[j]), Y[j]); l.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = float(lossf(model(A[iho]), Y[iho]))
        hist.append(vl)
        if kind == "eegnet_moabb": sched.step(vl)
        else: sched.step()
        if vl < best - 1e-12: best, wait = vl, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    if restore and best_state is not None: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): p = model(V).argmax(1).numpy()
    return p, {"epochs_run": ep + 1, "val_loss_curve": hist, "best_val_loss": best, "restored_best": restore}


def one(arm, subject, seed):
    t0 = time.time(); info = {}
    if arm.endswith("_moabbpp"):
        d = moabb_pp(subject); (Xtr, ytr), (Xva, yva) = d["train"], d["val"]
    else:
        d = load_subject(subject); Xtr, ytr, Xva, yva = d["train"]["seg"], d["train"]["y"], d["val"]["seg"], d["val"]["y"]
    if arm.startswith(("mdm", "tslr")): p = riemann(Xtr, ytr, Xva, arm.split("_")[0])
    else: p, info = eegnet(Xtr, ytr, Xva, seed, arm)
    k, a = float(cohen_kappa_score(yva, p)), float(accuracy_score(yva, p))
    rec = {"stage": "M11", "step": "A", "arm": arm, "subject": subject, "seed": seed, "kappa_val": k, "acc_val": a,
           "settle_rate": 1.0, "rejected": 0, "n_train": int(len(ytr)), "n_val": int(len(yva)),
           "pred": p, "label": yva, "train_info": info, "wall_s": time.time() - t0}
    write_receipt(OUT / git("rev-parse", "--short", "HEAD") / arm, f"A{subject:02d}_s{seed}", rec, subject=subject)
    return {k_: v for k_, v in rec.items() if k_ not in ("pred", "label", "train_info")}


def _job(a): return one(*a)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", default=",".join(ARMS)); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--subjects", default="1-9"); a = ap.parse_args()
    lo, hi = map(int, a.subjects.split("-")); arms = a.arms.split(",")
    jobs = [(arm, s, r) for arm in arms for r in (0, 1, 2) for s in range(lo, hi + 1)]
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(_job, jobs):
            print(f"{r['arm']} A{r['subject']:02d} s{r['seed']} kappa {r['kappa_val']:.3f} {r['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
