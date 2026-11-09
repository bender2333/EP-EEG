"""M13 §10.5 (objective alignment): does the trained lake's state carry motor-imagery structure at all?
Lakes: the §10 raw `none` lakes (weights in runs/M13G/w/, the §8 lakes), and the same subject's untrained E0 initial
lake (control: what a random lake passes through). Single subject; LDA trained on runs 0-3, tested on runs 4-5.
  probe  nothing hidden; per trial, log-variance in mu 8-13 and beta 13-30 Hz of every lake unit at equilibrium
         (88 columns + 22 outlets = 220 features), 4-class and left-vs-right shrinkage LDA; the same on the 22 leads
         themselves (44 features)
  erd    hide the test set of C3 (C3 + 2 nearest) and of C4; the left-minus-right difference of mean log mu / beta
         power on C3 and on C4 (contralateral ERD: positive on C3, negative on C4), truth vs lake_own vs visible
         mean vs nearest vs spline; and left-vs-right LDA trained on the true C3, C4 log-variance (runs 0-3) and
         tested on runs 4-5 with C3, C4 replaced by each reconstruction
C3, C4 are in fold A: hidden in training for seed 0 (tier N), never hidden for seed 1 (tier U).
Rows every 2nd sample (125 Hz)."""
from __future__ import annotations
import argparse, sys, time, warnings
import numpy as np
from scipy.signal import sosfiltfilt
warnings.filterwarnings("ignore")
from . import m12_s1 as S                        # noqa: E402
from . import m13_common as C                    # noqa: E402
from . import m13_learn as Lr                    # noqa: E402
from . import m13_rand as R                      # noqa: E402
from .m13_diag import SOS, states, WDIR, CAP     # noqa: E402

OUT = C.ROOT / "receipts" / "M13" / "G" / "probe"
C3, C4 = C.LEADS.index("C3"), C.LEADS.index("C4")


def lda():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def logvar(Z):            # Z (n, T, f) -> (n, 2f)
    return np.concatenate([np.log(sosfiltfilt(SOS[b], Z, axis=1).var(1) + 1e-12) for b in ("mu", "beta")], axis=1)


def scores(Ftr, ytr, Fte, yte):
    out = {"acc4": float(lda().fit(Ftr, ytr).score(Fte, yte))}
    mtr, mte = np.isin(ytr, (0, 1)), np.isin(yte, (0, 1))
    out["acc_lr"] = float(lda().fit(Ftr[mtr], ytr[mtr]).score(Fte[mte], yte[mte])); return out


def run(subject, seed):
    t0 = time.time(); d = C.load_real(subject, split="t04_val45"); seal = C.e_seal()
    if seal["matches"]: raise RuntimeError(f"E files present: {seal['matches']}")
    from ep_eeg.data import load_subject
    yy = load_subject(subject); ytr, yte = np.asarray(yy["train"]["y"]), np.asarray(yy["val"]["y"])
    tr, te = d["train"], d["test"]; ntr, nte = len(tr), len(te)
    Xtr, Xte = C.rows(tr, step=2), C.rows(te, step=2); T2 = Xtr.shape[0] // ntr
    lakes = {}
    g0, _ = Lr.build(subject, seed, "none", trained=False, cap=CAP); lakes["untrained"] = g0
    g1, _ = Lr.build(subject, seed, "none", trained=False, cap=CAP)
    w = np.load(WDIR / f"A{subject:02d}_s{seed}_none_raw.npz"); S.set_params(g1["lr"], w["eff"], w["bias"]); lakes["trained"] = g1
    rec = {"probe": {"leads": scores(logvar(Xtr.reshape(ntr, T2, -1)), ytr, logvar(Xte.reshape(nte, T2, -1)), yte)}}
    for name, g in lakes.items():
        Str, _, ok1 = states(g, Xtr, ()); Ste, _, ok2 = states(g, Xte, ())
        rec["probe"][name] = dict(scores(logvar(Str.reshape(ntr, T2, -1)), ytr, logvar(Ste.reshape(nte, T2, -1)), yte),
                                  qualified=float(min(ok1.mean(), ok2.mean())))
    T = R.test_sets(); recon = {}
    for c in (C3, C4):
        hid = T[c]; vis = [j for j in range(C.N_LEADS) if j not in hid]; k = list(hid).index(c)
        _, P, _ = states(g1, Xte, hid)
        recon[c] = {"truth": Xte[:, c], "lake_own": P[:, c], "mean_visible": Xte[:, vis].mean(1),
                    "nearest": Xte[:, min(vis, key=lambda j: R.DIST[c, j])], "spline": Xte[:, vis] @ C.spline_matrix(hid)[k]}
    L, Rr = yte == 0, yte == 1; erd = {}
    for name in recon[C3]:
        for c, lab in ((C3, "C3"), (C4, "C4")):
            z = recon[c][name].reshape(nte, T2)
            for b in ("mu", "beta"):
                lp = np.log(sosfiltfilt(SOS[b], z, axis=-1).var(-1) + 1e-12)
                erd[f"{name}.{lab}_{b}"] = float(lp[L].mean() - lp[Rr].mean())
    Ftrue_tr = logvar(np.stack([Xtr[:, C3], Xtr[:, C4]], 1).reshape(ntr, T2, 2))
    mtr, mte = np.isin(ytr, (0, 1)), np.isin(yte, (0, 1))
    clf = lda().fit(Ftrue_tr[mtr], ytr[mtr]); lr_acc = {}
    for name in recon[C3]:
        F = logvar(np.stack([recon[C3][name], recon[C4][name]], 1).reshape(nte, T2, 2))
        lr_acc[name] = float(clf.score(F[mte], yte[mte]))
    rec.update(erd=erd, lr_from_C3C4=lr_acc)
    return dict(rec, stage="M13", step="10.5-probe", subject=subject, seed=seed, tier_C3C4="N" if seed == 0 else "U",
                weights_sha256=C.sha(w["eff"], w["bias"]), e_seal=seal, data=d["source"], env=C.env_info(),
                wall_s=time.time() - t0, finished=C.now())


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--subject", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv); path = OUT / f"A{a.subject:02d}_s{a.seed}.json"
    if path.exists(): print("exists", path); return
    rec = run(a.subject, a.seed); C.write_receipt(path, rec)
    print(path.name, rec["probe"], rec["lr_from_C3C4"], f"wall {rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
