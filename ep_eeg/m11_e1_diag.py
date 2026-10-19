"""M11 E1 diagnostic (exploratory, not preregistered): why FBCSP falls below Track-2 on the synthetic data.

Same synthetic subjects (e from the E1 calibration, seed 0) and the same frozen-layer segments as E1:
  fbcsp            the M0/E1 pipeline (MNE CSP, 9 bands, MI top-8, LDA), as in E1
  fbcsp_lw         the same with Ledoit-Wolf regularised CSP covariances
  lead_logvar_sLDA per-lead log variance in 8-12 and 18-22 Hz (44 features) -> shrinkage LDA, no spatial filter
Writes receipts/M11/E/<run commit>/e1_diag.json.
"""
from __future__ import annotations
import json, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import cohen_kappa_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from .m11_e import synth_subject, to_d, OUT
from .m11_common import write_receipt, run_commit


def bandlv(seg, lo, hi):
    sos = butter(4, [lo, hi], btype="band", fs=250, output="sos"); return np.log(sosfiltfilt(sos, seg.astype(np.float64), axis=-1).var(-1))


def fbcsp_reg(tr, va, seed, reg):
    from mne.decoding import CSP
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    bands = [(lo, lo + 4) for lo in range(4, 36, 4)]; F, Fv = [], []
    for lo, hi in bands:
        sos = butter(4, [lo, hi], btype="band", fs=250, output="sos")
        x, xv = sosfiltfilt(sos, tr["seg"].astype(np.float64), axis=-1), sosfiltfilt(sos, va["seg"].astype(np.float64), axis=-1)
        c = CSP(n_components=4, log=True, reg=reg).fit(x, tr["y"]); F.append(c.transform(x)); Fv.append(c.transform(xv))
    F, Fv = np.hstack(F), np.hstack(Fv)
    sel = SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=seed), k=8).fit(F, tr["y"])
    return LDA().fit(sel.transform(F), tr["y"]).predict(sel.transform(Fv))


def one(args):
    subject, mix, e = args; warnings.simplefilter("ignore"); t0 = time.time()
    runs, _ = synth_subject(subject, e, mix); d = to_d(runs); tr, va = d["train"], d["val"]; K = lambda p: float(cohen_kappa_score(va["y"], p))
    F = np.hstack([bandlv(tr["seg"], 8, 12), bandlv(tr["seg"], 18, 22)]); Fv = np.hstack([bandlv(va["seg"], 8, 12), bandlv(va["seg"], 18, 22)])
    return {"subject": subject, "mix": mix, "fbcsp": K(fbcsp_reg(tr, va, 0, None)), "fbcsp_lw": K(fbcsp_reg(tr, va, 0, "ledoit_wolf")),
            "lead_logvar_sLDA": K(LDA(solver="lsqr", shrinkage="auto").fit(F, tr["y"]).predict(Fv)), "wall_s": time.time() - t0}


def main():
    e = json.loads((OUT / run_commit() / "e1_calibration.json").read_text())["pick"]
    with ProcessPoolExecutor(2) as ex: res = list(ex.map(one, [(s, m, e) for m in ("narrow", "broad") for s in range(1, 10)]))
    summ = {m: {k: float(np.mean([r[k] for r in res if r["mix"] == m])) for k in ("fbcsp", "fbcsp_lw", "lead_logvar_sLDA")} for m in ("narrow", "broad")}
    write_receipt(OUT / run_commit(), "e1_diag", {"exploratory": True, "e": e, "seed": 0, "per_subject": res, "mean": summ}); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
