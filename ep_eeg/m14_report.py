"""M14 tables: grid selection on T validation (--select), seeds at the selection (--seeds), E gate (--gate).

Selection rule (reports/M14.md 1.3): per arm, the (eta, U) with the highest 9-subject mean T-validation accuracy
on seed 0. Writes receipts/M14/selected.json.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "receipts" / "M14"
ARMS = ("A", "B", "C", "nolake", "C_nolake", "randlake")


def load(d, pattern="*.json"):
    out = {}
    for f in glob.glob(str(R / d / pattern)):
        r = json.loads(Path(f).read_text())
        out[(r["arm"], r["subject"], r["seed"], float(r["eta"]))] = r
    return out


def curve_table(recs, seed=0):
    """{(arm, eta): {U: [acc per subject 1..9]}}"""
    tab = {}
    for (arm, s, sd, eta), r in recs.items():
        if sd != seed: continue
        for row in r["train"]["curve"]:
            tab.setdefault((arm, eta), {}).setdefault(row["u"], {})[s] = row
    return tab


def select(recs):
    tab = curve_table(recs); sel = {}
    for arm in ARMS:
        best = None
        for (a, eta), byU in tab.items():
            if a != arm: continue
            for U, rows in byU.items():
                if len(rows) < 9: continue
                m = float(np.mean([rows[s]["acc"] for s in range(1, 10)]))
                if best is None or m > best[0]: best = (m, eta, U)
        if best: sel[arm] = {"eta": best[1], "U": int(best[2]), "val_acc_mean_seed0": best[0]}
    return sel, tab


def wilcoxon(a, b):
    from scipy.stats import wilcoxon as W
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0): return 1.0
    return float(W(a, b, alternative="greater").pvalue)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--select", action="store_true"); a = ap.parse_args()
    recs = load("grid")
    sel, tab = select(recs)
    print(f"{len(recs)} grid receipts")
    for (arm, eta) in sorted(tab):
        byU = tab[(arm, eta)]
        line = " ".join(f"{U}:{np.mean([r['acc'] for r in rows.values()]):.3f}({len(rows)})" for U, rows in sorted(byU.items()))
        print(f"{arm:9s} eta {eta:g}  {line}")
    print(json.dumps(sel, indent=1))
    if a.select:
        (R / "selected.json").write_text(json.dumps(sel, indent=1)); print("wrote", R / "selected.json")


if __name__ == "__main__":
    main()
