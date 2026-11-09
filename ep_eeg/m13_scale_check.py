"""M13 §8.0 check 3: how much of v0's lake_own error is scale. Per P3 pattern and hidden lead, fit one scalar a
(least squares, on the ridge training rows) to the lake's own reconstruction and score a*yhat on the eval rows.
Seed 0, 9 subjects; writes receipts/M13/R/check/scale.json."""
import json, sys
import numpy as np
from . import m13_common as C, m13_e0 as E, m13_dyn as M

def main():
    out = {}
    for s in range(1, 10):
        g, _ = M.load_lake(s, 0)
        d = C.load_real(s); Xrd = C.rows(d["train"], step=C.RIDGE_STEP); Xte = C.rows(d["test"])
        raw, sc, slope = [], [], []
        for hid in C.patterns()["P3"]:
            _, Otr, _ = E.lake_states(g, Xrd, hid); _, Ote, _ = E.lake_states(g, Xte, hid)
            ptr, pte = E.own_inverse(Otr[:, list(hid)]), E.own_inverse(Ote[:, list(hid)])
            a = (ptr * Xrd[:, list(hid)]).sum(0) / (ptr ** 2).sum(0)
            raw.append(C.r2_cols(pte, Xte[:, list(hid)]).mean()); sc.append(C.r2_cols(pte * a, Xte[:, list(hid)]).mean()); slope.append(a.mean())
        out[s] = {"lake_own_P3": float(np.mean(raw)), "scaled_P3": float(np.mean(sc)), "mean_scalar": float(np.mean(slope))}
        print(s, out[s], flush=True)
    C.write_receipt(C.ROOT / "receipts/M13/R/check/scale.json", {"per_subject": out, "written": C.now(), "commit": C.git("rev-parse", "HEAD")})

if __name__ == "__main__":
    sys.exit(main())
