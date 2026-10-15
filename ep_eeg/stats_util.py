"""Paired subject-level Wilcoxon helpers for stage reports."""
from __future__ import annotations
import numpy as np

def paired_wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    try:
        from scipy.stats import wilcoxon
        if np.allclose(d, 0):
            return {"diff_mean": 0.0, "stat": None, "pvalue": 1.0, "n": len(d)}
        # zero_method='wilcox' default; exclude zeros
        r = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        return {"diff_mean": float(d.mean()), "diff_sd": float(d.std(ddof=1)),
                "stat": float(r.statistic), "pvalue": float(r.pvalue), "n": int(len(d))}
    except Exception as e:
        return {"diff_mean": float(d.mean()), "error": str(e), "n": int(len(d))}
