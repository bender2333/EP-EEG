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


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation of a - b (zeros dropped), in [-1, 1]."""
    from scipy.stats import rankdata
    d = np.asarray(a, float) - np.asarray(b, float); d = d[d != 0]
    if not len(d): return 0.0
    r = rankdata(np.abs(d)); return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def bayes_hier(d, tau_scale=0.1, grid=4000):
    """M11 §7: normal hierarchical model on per-subject paired differences ("eight schools", BDA ch. 5).

    d: (subjects, seeds) differences (or (subjects,) when deterministic). Per-subject mean dbar_s with
    standard error sigma_s from the pooled within-subject (seed) variance, floor 1e-3;
    dbar_s ~ N(mu + u_s, sigma_s^2), u_s ~ N(0, tau^2), flat prior on mu, tau ~ HalfCauchy(0, tau_scale).
    p(tau | data) on a grid, then mu | tau, data normal; returns posterior mean, 95% interval, P(mu > 0).
    """
    from scipy.stats import norm
    d = np.asarray(d, float); d = d[:, None] if d.ndim == 1 else d
    J, R = d.shape; y = d.mean(1)
    within = d.var(1, ddof=1).mean() if R > 1 else 0.0
    s2 = np.full(J, max(within / R, 1e-6))
    taus = np.linspace(0, max(1.0, 10 * y.std()), grid)[1:]
    logp = []
    for t in taus:
        v = s2 + t * t; w = 1 / v; mh = (w * y).sum() / w.sum(); vm = 1 / w.sum()
        logp.append(np.log(t_prior := 1 / (1 + (t / tau_scale) ** 2)) + 0.5 * np.log(vm)
                    - 0.5 * np.log(v).sum() - 0.5 * ((y - mh) ** 2 * w).sum())
    logp = np.array(logp); p = np.exp(logp - logp.max()); p /= p.sum()
    mh = np.array([(y / (s2 + t * t)).sum() / (1 / (s2 + t * t)).sum() for t in taus])
    vm = np.array([1 / (1 / (s2 + t * t)).sum() for t in taus])
    xs = np.linspace(y.min() - 1, y.max() + 1, 8001)
    dens = (p[:, None] * norm.pdf(xs[None], mh[:, None], np.sqrt(vm)[:, None])).sum(0); dens /= dens.sum()
    cdf = np.cumsum(dens)
    return {"mu_mean": float((dens * xs).sum()), "ci95": [float(xs[np.searchsorted(cdf, 0.025)]), float(xs[np.searchsorted(cdf, 0.975)])],
            "p_gt0": float(dens[xs > 0].sum()), "tau_mean": float((p * taus).sum()), "J": J, "R": R}
