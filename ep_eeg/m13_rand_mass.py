"""M13 §8, post hoc (after the §8 verdicts were read): where the outlet field's input to a scored lead comes from in the
§8 test sets. For each test set (c + its 2 nearest leads hidden), field mass into c from the hidden neighbours vs from
visible leads, true positions vs derangements 0-2.  python -m ep_eeg.m13_rand_mass -> receipts/M13/R/check/field_mass.json"""
from __future__ import annotations
import sys
import numpy as np
from . import m13_common as C, m13_learn as L, m13_rand as R


def main():
    T = R.test_sets(); out = {}
    for name, perm in [("true", None)] + [(f"der{k}", C.derangement(k)) for k in range(3)]:
        Q = L.field_matrix(perm=perm); h, v = [], []
        for c, hid in T.items():
            h.append(Q[c, [j for j in hid if j != c]].sum()); v.append(Q[c, [j for j in range(C.N_LEADS) if j not in hid]].sum())
        out[name] = {"from_hidden": float(np.mean(h)), "from_visible": float(np.mean(v))}
        print(name, out[name])
    C.write_receipt(C.ROOT / "receipts" / "M13" / "R" / "check" / "field_mass.json", dict(out, written=C.now(), commit=C.git("rev-parse", "HEAD")))


if __name__ == "__main__":
    sys.exit(main())
