"""M13 修改 6: is the spline's 0.95 real, and how do the lakes compare with it on other dimensions (reports/M13.md §5).

Everything within one subject (T runs 0-3 train / runs 4-5 test; nothing across subjects).

  --audit            spline audit (no lake): reproduce E0's spline R^2; geometry check (deranged positions);
                     how much is the common component (visible mean, spline constant term); local R^2 after
                     removing the visible mean; band R^2 (theta/mu/beta, full 250 Hz rate); spline on microvolts
                     instead of per-channel z-scores; nearest-visible-lead copy
  --lake KIND        the same dimensions for a lake (v0 | C | L:<receipt stem>) plus the MI-decoding probe
  --seed S

Reconstructions are made at the full sample rate (every sample, static equilibrium per sample for the lakes),
so band-limited scores are not aliased. Dimensions (pre-registered in reports/M13.md §5 before any lake number):
  D0  R^2 on E0's rows (every 5th sample), as before
  D1  local R^2: visible mean removed from prediction and truth (the visible mean is known to every method)
  D2  band R^2: theta 4-8, mu 8-13, beta 13-30 Hz (zero-phase band-pass of prediction and truth)
  D3  MI decoding: hide the P3 patterns centred on C3, Cz, C4 (one at a time); 4-class LDA on log-variance in
      mu and beta of all 22 leads, trained on true runs 0-3, tested on runs 4-5 with the hidden leads replaced
      by the reconstruction; references: true leads (ceiling), LDA retrained on the 19 visible leads (no
      reconstruction), hidden leads = visible mean
  D3b left vs right hand from the hidden leads alone (log-variance mu, beta of the 3 hidden leads; LDA trained on
      the true hidden leads of runs 0-3, tested on runs 4-5 as reconstructed): does the reconstruction carry the
      lateralised class information of those leads; references: true leads, visible mean
  D4  spectral fidelity: log10 of (reconstructed / true) power in mu and beta on the hidden leads (0 = exact)
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

warnings.filterwarnings("ignore")
from . import m13_common as C                    # noqa: E402
from . import m13_e0 as E                        # noqa: E402
from . import m13_dyn as M                       # noqa: E402

OUT = C.ROOT / "receipts" / "M13" / "D"
FS = 250.0
BANDS = {"theta": (4, 8), "mu": (8, 13), "beta": (13, 30)}
SOS = {b: butter(4, f, btype="band", fs=FS, output="sos") for b, f in BANDS.items()}
P3 = [h for f, h in C.all_patterns() if f == "P3"]
CENTRE = {name: C.LEADS.index(name) for name in ("C3", "Cz", "C4")}


def mi_patterns():
    """P3 patterns centred on C3, Cz, C4: the lead and its two nearest (C.patterns' rule)."""
    D = np.linalg.norm(C.XYZ[:, None] - C.XYZ[None], axis=-1); out = {}
    for name, c in CENTRE.items():
        nn = [j for j in np.argsort(D[c]) if j != c][:2]; out[name] = tuple(sorted([c, *map(int, nn)]))
    return out


def bandpass(x, b): return sosfiltfilt(SOS[b], x, axis=-1)


# ----------------------------------------------------------------------------- reconstructions (trials, 22, T)
def spline_recon(seg, hid, xyz=C.XYZ):
    vis = [c for c in range(C.N_LEADS) if c not in hid]
    out = seg.copy(); out[:, hid] = np.einsum("hv,nvt->nht", C.spline_matrix(hid, xyz=xyz), seg[:, vis]); return out


def meanvis_recon(seg, hid):
    vis = [c for c in range(C.N_LEADS) if c not in hid]
    out = seg.copy(); out[:, hid] = seg[:, vis].mean(1, keepdims=True); return out


def nearest_recon(seg, hid):
    vis = [c for c in range(C.N_LEADS) if c not in hid]
    D = np.linalg.norm(C.XYZ[:, None] - C.XYZ[None], axis=-1)
    out = seg.copy()
    for h in hid: out[:, h] = seg[:, vis[int(np.argmin(D[h, vis]))]]
    return out


def lake_recon(lake, g, seg, pats, every=1, in_time=False):
    """lake_own reconstruction at every `every`-th sample, static equilibrium per sample (or, in_time: the lake run in
    time, InertialLake.run, as M13 §6/§7 lakes are trained and scored). Returns {hid: (n,22,T')}."""
    nt, T = seg.shape[0], seg.shape[2]; res = {}
    for i in range(0, len(pats), 11):
        chunk = pats[i:i + 11]
        D = np.empty((T, nt * len(chunk), g["ns"]))
        for j, hid in enumerate(chunk):
            Mk = np.zeros(C.N_LEADS, bool); Mk[list(hid)] = True
            val = np.where(Mk[None, :, None], 0.0, np.tanh(seg / 3.0))
            D[:, j * nt:(j + 1) * nt, :C.N_LEADS] = val.transpose(2, 0, 1); D[:, j * nt:(j + 1) * nt, C.N_LEADS:] = Mk[None, None, :]
        with np.errstate(all="ignore"):
            if in_time:
                out = lake.run(D, M.SUB, g["out"], every=every)
                if not np.isfinite(out).all(): raise RuntimeError("non-finite in-time run")
            else:
                out, resid = lake.track_static(D, g["out"], every=every)          # (T', B, 22)
                if float(np.nan_to_num(resid, nan=np.inf).max()) > 1e-6: raise RuntimeError("static equilibrium not reached")
        for j, hid in enumerate(chunk):
            s = out[:, j * nt:(j + 1) * nt][:, :, list(hid)].transpose(1, 2, 0)    # (n, h, T')
            r = seg[:, :, ::every].copy(); r[:, list(hid)] = E.own_inverse(s); res[tuple(hid)] = r
    return res


# ----------------------------------------------------------------------------- scores
def r2(pred, y):
    """pred, y (n, h, T): R^2 per hidden lead over all samples (SST around the mean), averaged over leads."""
    p = pred.transpose(0, 2, 1).reshape(-1, pred.shape[1]); t = y.transpose(0, 2, 1).reshape(-1, y.shape[1])
    return float(C.r2_cols(p, t).mean())


def dims(recon, seg, hid):
    """D0-D2, D4 for one pattern; recon and seg are full-rate (n,22,750)."""
    hid = list(hid); vis = [c for c in range(C.N_LEADS) if c not in hid]
    out = {"D0": r2(recon[:, hid, ::C.STEP], seg[:, hid, ::C.STEP])}
    mv = seg[:, vis].mean(1, keepdims=True)
    out["D1_local"] = r2(recon[:, hid] - mv, seg[:, hid] - mv)
    for b in BANDS:
        pb, yb = bandpass(recon[:, hid], b), bandpass(seg[:, hid], b)
        out[f"D2_{b}"] = r2(pb, yb)
        if b in ("mu", "beta"):
            out[f"D4_logpow_{b}"] = float(np.mean(np.log10(pb.var(-1).mean(0) / yb.var(-1).mean(0))))
    return out


def logvar_feats(seg):
    return np.concatenate([np.log(bandpass(seg, b).var(-1)) for b in ("mu", "beta")], axis=1)   # (n, 44)


def lda():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def decode_refs(tr, ytr, te, yte):
    """D3 references that do not depend on the reconstruction method."""
    clf = lda().fit(logvar_feats(tr), ytr); out = {"true": float(clf.score(logvar_feats(te), yte))}
    for name, hid in mi_patterns().items():
        vis = [c for c in range(C.N_LEADS) if c not in hid]
        cv = lda().fit(logvar_feats(tr[:, vis]), ytr)
        out[f"visible_only_{name}"] = float(cv.score(logvar_feats(te[:, vis]), yte))
        out[f"meanvis_{name}"] = float(clf.score(logvar_feats(meanvis_recon(te, hid)), yte))
    return out, clf


def decode(clf, recon_by_name, yte):
    return {name: float(clf.score(logvar_feats(r), yte)) for name, r in recon_by_name.items()}


LR = (0, 1)                                       # left hand, right hand


def lr_probe(tr, ytr, te, yte, recon_by_name=None):
    """D3b: left vs right hand from the HIDDEN leads alone (log-variance mu, beta of the 3 leads), LDA trained on the
    true hidden leads of runs 0-3, tested on runs 4-5 with those leads as reconstructed. recon_by_name None ->
    the references (true leads, visible mean)."""
    mtr, mte = np.isin(ytr, LR), np.isin(yte, LR); out = {}
    for name, hid in mi_patterns().items():
        hid = list(hid)
        clf = lda().fit(logvar_feats(tr[mtr][:, hid]), ytr[mtr])
        if recon_by_name is None:
            out[f"true_{name}"] = float(clf.score(logvar_feats(te[mte][:, hid]), yte[mte]))
            out[f"meanvis_{name}"] = float(clf.score(logvar_feats(meanvis_recon(te[mte], hid)[:, hid]), yte[mte]))
        else:
            out[name] = float(clf.score(logvar_feats(recon_by_name[name][mte][:, hid]), yte[mte]))
    return out


# ----------------------------------------------------------------------------- jobs
def data(subject):
    d = C.load_real(subject)
    from ep_eeg.data import load_subject
    y = load_subject(subject)
    seal = C.e_seal()
    if seal["matches"]: raise RuntimeError(f"E files present: {seal['matches']}")
    return d["train"], np.asarray(y["train"]["y"]), d["test"], np.asarray(y["val"]["y"]), seal, d["source"]


def microvolt_std(subject):
    """Per-channel std (uV) the frozen layer divided by (fit on T runs 0-3): undoes the z-score for the uV check."""
    from ep_eeg.data import _runs_T
    from eval import preprocess as P
    return P.fit_preproc(_runs_T(subject)[:4]).std[:, 0]


def audit(subject):
    tr, ytr, te, yte, seal, src = data(subject)
    std = microvolt_std(subject)
    rows = []
    for hid in P3:
        hid = list(hid)
        r = {"hidden": hid, "spline": dims(spline_recon(te, hid), te, hid)}
        r["spline_shuf"] = float(np.mean([dims(spline_recon(te, hid, xyz=C.XYZ[C.derangement(k)]), te, hid)["D0"] for k in range(3)]))
        r["mean_visible"] = dims(meanvis_recon(te, hid), te, hid)
        r["nearest"] = dims(nearest_recon(te, hid), te, hid)
        uv = te * std[None, :, None]
        rec_uv = spline_recon(uv, hid) / std[None, :, None]
        r["spline_uV"] = dims(rec_uv, te, hid)["D0"]
        # the spline's constant term alone (c0 of the fit): how much of the 0.95 is the common component
        vis = [c for c in range(C.N_LEADS) if c not in hid]
        S = C.spline_matrix(hid)
        r["spline_rowsum"] = [float(x) for x in S.sum(1)]
        rows.append(r)
    summ = {}
    for k in ("spline", "mean_visible", "nearest"):
        for d in rows[0][k]:
            summ[f"{k}_{d}"] = float(np.mean([r[k][d] for r in rows]))
    summ["spline_shuf_D0"] = float(np.mean([r["spline_shuf"] for r in rows]))
    summ["spline_uV_D0"] = float(np.mean([r["spline_uV"] for r in rows]))
    # variance shares on the hidden leads: common (visible mean) vs rest
    vs = []
    for hid in P3:
        hid = list(hid); vis = [c for c in range(C.N_LEADS) if c not in hid]
        mv = te[:, vis].mean(1, keepdims=True); y = te[:, hid]
        vs.append(1 - ((y - mv) ** 2).mean() / ((y - y.mean()) ** 2).mean())
    summ["hidden_var_explained_by_visible_mean"] = float(np.mean(vs))
    refs, clf = decode_refs(tr, ytr, te, yte)
    mi = mi_patterns()
    summ.update({f"D3_{k}": v for k, v in refs.items()})
    summ.update({f"D3_spline_{n}": v for n, v in decode(clf, {n: spline_recon(te, list(h)) for n, h in mi.items()}, yte).items()})
    summ.update({f"D3b_{k}": v for k, v in lr_probe(tr, ytr, te, yte).items()})
    summ.update({f"D3b_spline_{k}": v for k, v in lr_probe(tr, ytr, te, yte, {n: spline_recon(te, list(h)) for n, h in mi.items()}).items()})
    return {"stage": "M13", "step": "修改6-audit", "subject": subject, "data": src, "patterns": rows, "summary": summ,
            "uV_std": std.tolist(), "e_seal": seal, "env": C.env_info(), "finished": C.now()}


def lake_for(kind, subject, seed):
    if kind == "v0":
        g, _ = M.load_lake(subject, seed); return g, M.dynamics(g, "static", {})
    if kind == "C":
        cfg = json.load(open(C.ROOT / "receipts" / "M13" / "C" / "selected.json"))["selected"]
        g, _ = M.load_lake(subject, seed); return g, M.dynamics(g, "field", cfg)
    if kind.startswith("L:"):
        from . import m13_learn as Lm
        stem = kind[2:].format(subject=subject, seed=seed)
        rec = json.load(open(Lm.OUT / "test" / f"{stem}.json"))
        g, _ = Lm.build(subject, seed, rec["cond"], rec["field"]["ell"], rec["field"]["lam"], der=rec["field"]["derangement"] or 0)
        z = np.load(Lm.OUT / "weights" / "test" / f"{stem}.npz")
        assert np.array_equal(z["pre"], g["conn"].pre) and np.array_equal(z["post"], g["conn"].post)
        from . import m12_s1 as S
        S.set_params(g["lr"], z["eff"], z["bias"])
        assert C.sha(np.asarray(g["lr"].brain.efficacy), np.asarray(g["lr"].brain.bias)) == rec["final_sha256"]
        return g, M.dynamics(g, "static", {})
    if kind.startswith("T:"):                       # §6/§7 in-time lakes: T:<test receipt stem>, scored in time
        from . import m13_learn as Lm, m13_time as Tm
        from . import m12_s1 as S
        stem = kind[2:].format(subject=subject, seed=seed)
        rec = json.load(open(Tm.OUT / "test" / f"{stem}.json"))
        g, _ = Lm.build(subject, seed, rec["cond"], rec["field"]["ell"], rec["field"]["lam"], der=rec["field"]["derangement"] or 0)
        z = np.load(Tm.OUT / "weights" / "test" / f"{stem}.npz")
        assert np.array_equal(z["pre"], g["conn"].pre) and np.array_equal(z["post"], g["conn"].post)
        S.set_params(g["lr"], z["eff"], z["bias"])
        assert C.sha(np.asarray(g["lr"].brain.efficacy), np.asarray(g["lr"].brain.bias)) == rec["final_sha256"]
        zeta = rec.get("zeta")
        return g, (M.dynamics(g, "dyn1", {}) if zeta is None else M.dynamics(g, "inert", {"zeta": zeta}))
    raise ValueError(kind)


def lake_job(kind, subject, seed):
    tr, ytr, te, yte, seal, src = data(subject)
    g, lake = lake_for(kind, subject, seed)
    mi = mi_patterns()
    pats = [tuple(h) for h in P3]
    rec = lake_recon(lake, g, te, pats + [h for h in mi.values() if h not in pats], every=1, in_time=kind.startswith("T:"))
    rows = [{"hidden": list(h), "lake": dims(rec[h], te, h)} for h in pats]
    summ = {f"lake_{d}": float(np.mean([r["lake"][d] for r in rows])) for d in rows[0]["lake"]}
    refs, clf = decode_refs(tr, ytr, te, yte)
    summ.update({f"D3_lake_{n}": v for n, v in decode(clf, {n: rec[h] for n, h in mi.items()}, yte).items()})
    summ.update({f"D3b_lake_{n}": v for n, v in lr_probe(tr, ytr, te, yte, {n: rec[h] for n, h in mi.items()}).items()})
    return {"stage": "M13", "step": "修改6-lake", "kind": kind, "subject": subject, "seed": seed, "data": src,
            "patterns": rows, "summary": summ, "e_seal": seal, "env": C.env_info(), "finished": C.now()}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--audit", action="store_true"); ap.add_argument("--lake"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv); t0 = time.time()
    if a.audit:
        path = OUT / "audit" / f"A{a.subject:02d}.json"
        rec = audit(a.subject)
    else:
        tag = a.lake.replace("A{subject:02d}_s{seed}_", "").replace("{subject:02d}", "").replace("{seed}", "").replace(":", "_")
        path = OUT / "lake" / f"A{a.subject:02d}_s{a.seed}_{tag}.json"
        rec = lake_job(a.lake, a.subject, a.seed)
    rec["wall_s"] = time.time() - t0
    C.write_receipt(path, rec)
    print(path.name, {k: round(v, 4) for k, v in rec["summary"].items()}, f"wall {rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
