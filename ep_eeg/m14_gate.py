"""M14 stage gate: evaluate the frozen candidates once on the E session (EPEEG_GATE=1).

For every arm in receipts/M14/selected.json and seeds 0-2, rebuild the network, load the weights saved at U_sel by
the T-validation run (trained on T runs 0-3 only), check that the gate loader rebuilds the identical training
windows, and score the E session. One receipt per (arm, subject, seed) under receipts/M14/gate/.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np

from . import m14 as M  # sets the thread caps before numpy work

ROOT = M.ROOT; R = M.OUT


def gate_one(arm, subject, seed, sel, cache):
    eta, U, temp = sel["eta"], sel["U"], M.TEMP
    wpath = R / "weights" / f"{arm}_A{subject:02d}_s{seed}_eta{eta:g}_T{temp:g}" / f"u{U}.npz"
    if not wpath.exists(): raise FileNotFoundError(wpath)
    if subject not in cache:
        dv = M.load(subject, "val"); dg = M.load(subject, "gate")
        same = np.array_equal(dv["Xtr"], dg["Xtr"]) and np.array_equal(dv["ytr"], dg["ytr"])
        cache[subject] = (dg, bool(same), float(np.abs(dv["Xtr"] - dg["Xtr"]).max()))
    dg, same, maxdiff = cache[subject]
    g = M.build(arm, 14500 + 100 * subject + seed, eta=eta, temp=temp); M.load_weights(g, wpath)
    s, _, ok = M.evaluate(g, dg["Xte"], dg["yte"], dg["ytr"]); pred = s.pop("pred")
    rec = {"stage": "M14", "split": "gate_E", "arm": arm, "subject": subject, "seed": seed, "eta": eta, "U": U, "temp": temp,
           "weights": str(wpath.relative_to(ROOT)), "weights_sha256": hashlib.sha256(wpath.read_bytes()).hexdigest(),
           "train_windows_identical": same, "train_windows_maxdiff": maxdiff,
           "test": s, "pred": pred, "y": dg["yte"], "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    from ep_eeg.m13_common import write_receipt
    write_receipt(R / "gate" / f"{arm}_A{subject:02d}_s{seed}.json", rec)
    return rec


def main():
    if os.environ.get("EPEEG_GATE") != "1": raise PermissionError("set EPEEG_GATE=1 (stage gate only)")
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", nargs="*", default=None); ap.add_argument("--subjects", nargs="*", type=int, default=list(range(1, 10)))
    a = ap.parse_args()
    sel = json.loads((R / "selected.json").read_text())
    cache = {}
    for subject in a.subjects:
        for arm in (a.arms or list(sel)):
            for seed in (0, 1, 2):
                f = R / "gate" / f"{arm}_A{subject:02d}_s{seed}.json"
                if f.exists(): continue
                r = gate_one(arm, subject, seed, sel[arm], cache)
                print(f"{arm} A{subject:02d} s{seed} E acc={r['test']['acc']:.3f} kappa={r['test']['kappa']:.3f} "
                      f"settle={r['test']['settle_rate']:.3f} same_train={r['train_windows_identical']}", flush=True)
        cache.pop(subject, None)


if __name__ == "__main__":
    main()
