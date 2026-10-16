"""M7 acceptance probe: internal-state power spectra of the warm-chained raw-sequence EP sandbox.

Training uses 30 non-overlapping 100 ms raw chunks (state sampled at 10 Hz -> Nyquist 5 Hz, cannot
resolve mu/beta). For the PSD probe the *trained* network is driven with the same 22x25 raw-chunk
input format but sliding by 1 sample (hop 4 ms), warm-chained, so column activations x(t) form a
250 Hz series over cue+0.5..3.5 s (726 samples). Welch PSD -> mu (8-13) / beta (13-30) band power,
per class, for C3 / C4 columns (ERD check). Same analysis on the input EEG for reference.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import welch
from eval import preprocess as P
from .data import load_subject, ROOT
from .sandbox import train_sandbox, LEADS

OPTS = dict(track="rawseq", use_warm=True, epochs=100, sigma_in=0.012, leak=1.0,
            rowmass_project=0.9, eta=0.5, temperature=0.01)
FS = 250.0; C = P.SEQ_C
CLS = ["left", "right", "feet", "tongue"]


def sliding(seg, sc):
    n, L, T = seg.shape
    X = np.stack([seg[:, :, s:s + C].reshape(n, L * C) for s in range(T - C + 1)], 1)
    return sc.transform(X)


def run_states(sb, X):
    n, S, d = X.shape; st = None; cols = np.empty((n, S, sb.nc), np.float32); conv = []
    for t in range(S):
        eq = sb.learner.brain.equilibrate(sb.drive(X[:, t]), budget=sb.cfg.free_steps,
                                          tolerance=sb.cfg.tolerance, damping=sb.cfg.damping, state=st)
        a = np.asarray(eq.state.activation); cols[:, t] = a[:, sb.col0:sb.out0]
        conv.append(float(np.mean(eq.converged))); st = eq.state
    return cols, float(np.mean(conv))


def bands(f, p):
    return p[..., (f >= 8) & (f < 13)].mean(-1), p[..., (f >= 13) & (f < 30)].mean(-1)


def peak(f, p):
    """Band mean PSD / flank mean PSD (4-8 & 30-40 Hz); >1 suggests a band peak."""
    fl = p[(f >= 4) & (f < 8)].mean() * 0.5 + p[(f >= 30) & (f < 40)].mean() * 0.5
    mu, be = bands(f, p)
    return float(mu / fl), float(be / fl)


def analyse(subject, seed=0, out=None):
    out = Path(out or ROOT / "receipts" / "M7" / "psd"); out.mkdir(parents=True, exist_ok=True)
    d = load_subject(subject); t0 = time.time()
    sb, stats, logs = train_sandbox(d["train"], seed, **OPTS); tw = time.time() - t0
    sc = P.FeatureScaler().fit(P.rawseq(d["train"]["seg"].astype(np.float64)))
    seg = d["val"]["seg"].astype(np.float64); y = d["val"]["y"]
    cols, conv = run_states(sb, sliding(seg, sc))
    k = sb.cfg.k; ci, c4 = LEADS.index("C3"), LEADS.index("C4")
    lead_x = cols.reshape(len(y), cols.shape[1], 22, k).mean(-1)
    f, p_x = welch(lead_x, fs=FS, nperseg=128, axis=1); p_x = np.moveaxis(p_x, 1, -1)
    _, p_in = welch(seg[:, :, C // 2:C // 2 + lead_x.shape[1]], fs=FS, nperseg=128, axis=-1)
    res = {"subject": subject, "seed": seed, "train_wall_s": tw, "probe_wall_s": time.time() - t0 - tw,
           "probe_converged": conv, "cert": sb.cert(), "fs_state": FS, "n_state_samples": int(lead_x.shape[1]),
           "x_std_C3": float(lead_x[:, :, ci].std())}
    for name, p in (("state", p_x), ("input", p_in)):
        r = {}
        for lead, li in (("C3", ci), ("C4", c4)):
            mu, be = bands(f, p[:, li])
            r[lead] = {"peak_ratio_mu_beta": peak(f, p[:, li].mean(0)),
                       "mu_by_class": [float(mu[y == c].mean()) for c in range(4)],
                       "beta_by_class": [float(be[y == c].mean()) for c in range(4)]}
        r["erd_C3_right<left"] = {b: bool(r["C3"][b + "_by_class"][1] < r["C3"][b + "_by_class"][0]) for b in ("mu", "beta")}
        r["erd_C4_left<right"] = {b: bool(r["C4"][b + "_by_class"][0] < r["C4"][b + "_by_class"][1]) for b in ("mu", "beta")}
        r["mean_psd_C3_by_class"] = [p[y == c, ci].mean(0).tolist() for c in range(4)]
        r["mean_psd_C4_by_class"] = [p[y == c, c4].mean(0).tolist() for c in range(4)]
        res[name] = r
    res["freqs"] = f.tolist()
    (out / f"A{subject:02d}_s{seed}.json").write_text(json.dumps(res))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for i, name in enumerate(("state", "input")):
        for j, lead in enumerate(("C3", "C4")):
            a = ax[i, j]
            for c in range(4): a.semilogy(f, res[name][f"mean_psd_{lead}_by_class"][c], label=CLS[c])
            a.axvspan(8, 13, alpha=.1, color="g"); a.axvspan(13, 30, alpha=.07, color="b")
            a.set_xlim(1, 45); a.set_title(f"A{subject:02d} {name} {lead}"); a.set_xlabel("Hz")
    ax[0, 0].legend(); fig.tight_layout(); fig.savefig(out / f"A{subject:02d}_s{seed}.png", dpi=90)
    brief = {k: v for k, v in res.items() if k not in ("freqs", "state", "input")}
    for n in ("state", "input"):
        brief[n] = {kk: vv for kk, vv in res[n].items() if not kk.startswith("mean_psd")}
    print(json.dumps(brief, indent=1))
    return res


if __name__ == "__main__":
    analyse(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0)
