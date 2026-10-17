"""Paired per-subject comparison of two receipts: wins/losses/ties and two-sided sign test.

usage: python -m ep_eeg.paired A/runs.json B/runs.json [label]
Per-subject kappa is the seed mean. Ties are excluded from the sign test (reported separately).
With 9 subjects and 1 seed, p < 0.05 needs >= 8/9 wins.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest


def per_subject(path):
    acc = defaultdict(list)
    for r in json.load(open(path)):
        acc[int(r["subject"])].append(float(r.get("kappa", r.get("kappa_val"))))
    return {s: float(np.mean(v)) for s, v in sorted(acc.items())}


def compare(a, b):
    A, B = per_subject(a), per_subject(b); subs = sorted(set(A) & set(B))
    d = np.array([A[s] - B[s] for s in subs]); w, l = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0
    return {"n": len(subs), "mean_a": float(np.mean([A[s] for s in subs])), "mean_b": float(np.mean([B[s] for s in subs])),
            "mean_diff": float(d.mean()), "wins": w, "losses": l, "ties": len(subs) - w - l, "sign_p": float(p),
            "per_subject_diff": {s: round(float(x), 4) for s, x in zip(subs, d)}}


if __name__ == "__main__":
    r = compare(sys.argv[1], sys.argv[2]); lab = sys.argv[3] if len(sys.argv) > 3 else ""
    print(f"{lab:38s} {r['mean_a']:.3f} vs {r['mean_b']:.3f}  diff {r['mean_diff']:+.3f}  "
          f"W/L/T {r['wins']}/{r['losses']}/{r['ties']}  sign p={r['sign_p']:.3f}")
