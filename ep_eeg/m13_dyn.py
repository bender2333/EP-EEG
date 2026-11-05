"""M13 steps A (4.1), B (4.2), C (4.3): the trained v0 lakes run in time on the E0 reconstruction task.

No new learning. Each job loads the E0 checkpoint of (subject, seed) (U = 6 000, the official run), builds the
dynamics with cadence.experimental (fork bender2333/cadence, branch m13) and scores P3 (and optionally P1) hidden-
electrode reconstruction with the lake's own outlets (``lake_own``: fixed inverse of the target map, no fitting),
on the same rows as E0 (every 5th sample of each test trial).

Arms (one per job, ``--arm``):
  static   v0 as in E0: the static equilibrium at every row (warm-started fixed-point iteration); must reproduce
           E0's lake_own (brain.equilibrate per row) to rounding
  power    A: static with each lead's input replaced by its Hilbert amplitude (phase removed)
  dyn1     first-order lake integrated in time (tau 2 ms, dt 1 ms, 4 substeps per 4 ms sample): what running in
           time costs by itself, the reference that isolates B's oscillators
  inert    B: columns get a velocity partner, f0 = 10 Hz, damping ratio zeta
  field    C: a scalar field on a scalp mesh coupled to outlets or columns; quasi-static (c = inf, static
           equilibrium per row) or a wave (c m/s, integrated in time)
Controls computed in the same job: the same arm on the untrained lake (``--random``), and for ``field`` the
node assignment deranged (``--derange i``). spline / mean_visible come from the E0 receipts (same rows).

Spatial diagnostic (every job): regress the lake's reconstruction of each hidden lead on the visible leads
(least squares over the test rows) and compare the weights with the spherical-spline weights (cosine) and with
distance on the scalp (Spearman of |w| against -distance). The same numbers for the data itself (true hidden
lead on visible leads) are the reference.
"""
from __future__ import annotations

import argparse, json, os, sys, time, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from . import m12_s1 as S                        # noqa: E402
from . import m13_common as C                    # noqa: E402
from . import m13_e0 as E                        # noqa: E402
from cadence.experimental import InertialLake, ScalarField, graph_laplacian, sphere_mesh  # noqa: E402

OUT = C.ROOT / "receipts" / "M13"
CKPT = OUT / "E0" / "ckpt" / "official"
DT, SUB, TAU_FAST, F0 = 0.001, 4, 0.002, 10.0
P3 = [p for p in C.all_patterns() if p[0] == "P3"]


# ----------------------------------------------------------------------------- lakes
def load_lake(subject, seed, trained=True):
    init = 5000 + 100 * subject + seed
    g = E.build_lake(init)
    init_sha = C.sha(np.asarray(g["lr"].brain.efficacy))
    if trained:
        f = CKPT / f"{subject:02d}_s{seed}_eta0.05_U6000_{init_sha[:12]}.npz"
        z = np.load(f, allow_pickle=False)
        if int(z["u"]) != 6000:
            raise RuntimeError(f"{f.name}: checkpoint at u={int(z['u'])}, not finished")
        S.set_params(g["lr"], z["eff"], z["bias"])
    return g, init_sha


def mesh(n_mesh=260, min_gap=0.006):
    """Scalp mesh: Fibonacci cap + the 22 electrodes projected on the head sphere (electrodes are nodes 0..21)."""
    el = C.HEAD_CENTER + C.HEAD_R * C.unit(C.XYZ)
    zmin = ((el - C.HEAD_CENTER)[:, 2] / C.HEAD_R).min() - 0.15
    cap = sphere_mesh(n_mesh, C.HEAD_R, C.HEAD_CENTER, z_min=zmin)
    d = np.linalg.norm(cap[:, None] - el[None], axis=-1).min(1)
    xyz = np.concatenate([el, cap[d > min_gap]])
    return xyz, graph_laplacian(xyz, k=6)


def make_field(g, attach, alpha, ell, c, perm=None, zeta_f=0.5):
    xyz, L = mesh()
    node_of_lead = np.arange(C.N_LEADS) if perm is None else np.asarray(perm)
    if attach == "out":
        neurons = np.asarray(g["out"]); nodes = node_of_lead
    elif attach == "cols":
        neurons = np.asarray(g["cols"]); nodes = np.repeat(node_of_lead, g["k"])
    else:
        raise ValueError(attach)
    return ScalarField(L, neurons, nodes, alpha=alpha, ell=ell, c=c, zeta=zeta_f)


def dynamics(g, arm, cfg, perm=None):
    brain = g["lr"].brain; src = np.arange(g["ns"])
    if arm in ("static", "power", "dyn1"):
        return InertialLake(brain, src, (), tau_fast=TAU_FAST, dt=DT)
    if arm == "inert":
        return InertialLake(brain, src, g["cols"], f0=cfg.get("f0", F0), zeta=cfg["zeta"], tau_fast=TAU_FAST, dt=DT)
    if arm == "field":
        f = make_field(g, cfg["attach"], cfg["alpha"], cfg["ell"], cfg.get("c"), perm=perm)
        inert = g["cols"] if cfg.get("inert_zeta") is not None else ()
        return InertialLake(brain, src, inert, f0=F0, zeta=cfg.get("inert_zeta") or 0.2, tau_fast=TAU_FAST, dt=DT,
                            field=f)
    raise ValueError(arm)


def is_static(arm, cfg):
    """static equilibrium per row (v0's own semantics) unless the arm has time dynamics."""
    return arm in ("static", "power") or (arm == "field" and cfg.get("c") is None and cfg.get("inert_zeta") is None)


def envelope(seg):
    from scipy.signal import hilbert
    return np.abs(hilbert(seg, axis=-1))


# ----------------------------------------------------------------------------- one job
def spatial_diag(pred, Xvis, hid, vis):
    """pred (rows, h): lake reconstruction; returns per hidden lead (cos with spline weights, spearman vs -dist)."""
    from scipy.stats import spearmanr
    w, *_ = np.linalg.lstsq(np.c_[Xvis, np.ones(len(Xvis))], pred, rcond=None)
    w = w[:-1].T                                                     # (h, n_vis)
    ws = C.spline_matrix(hid)                                        # (h, n_vis)
    el = C.unit(C.XYZ)
    cos, rho = [], []
    for j, h in enumerate(hid):
        cos.append(float(w[j] @ ws[j] / (np.linalg.norm(w[j]) * np.linalg.norm(ws[j]) + 1e-12)))
        dist = np.arccos(np.clip(el[vis] @ el[h], -1, 1))
        rho.append(float(spearmanr(np.abs(w[j]), -dist).statistic))
    return cos, rho


def run_job(subject, seed, arm, cfg, split="test", random=False, derange=None, families=("P3",), n_trials=None):
    t0 = time.time()
    data = C.load_real(subject, split="t04_val45" if split == "test" else "dev_r02_r3")
    seal = C.e_seal()
    if seal["matches"]: raise RuntimeError(f"E files present: {seal['matches']}")
    seg = data["test"] if n_trials is None else data["test"][:n_trials]
    g, init_sha = load_lake(subject, seed, trained=not random)
    perm = None if derange is None else C.derangement(derange)
    lake = dynamics(g, arm, cfg, perm=perm)
    feed = envelope(seg) if arm == "power" else seg                # (trials, 22, 750)
    Xte = C.rows(seg)                                              # targets/visible: always the raw signal
    pats = [p for p in C.all_patterns() if p[0] in families]
    nt, T = len(seg), seg.shape[2]
    res, max_resid = [], []
    for i in range(0, len(pats), 11):
        chunk = pats[i:i + 11]
        D = np.empty((T, nt * len(chunk), g["ns"]))
        for j, (_, hid) in enumerate(chunk):
            M = np.zeros(C.N_LEADS, bool); M[list(hid)] = True
            val = np.where(M[None, :, None], 0.0, np.tanh(feed / 3.0))          # (trials, 22, T)
            D[:, j * nt:(j + 1) * nt, :C.N_LEADS] = val.transpose(2, 0, 1)
            D[:, j * nt:(j + 1) * nt, C.N_LEADS:] = M[None, None, :]
        with np.errstate(all="ignore"):
            if is_static(arm, cfg):
                out, resid = lake.track_static(D, g["out"], every=C.STEP)  # (T/5, B, 22)
                max_resid.append(float(np.nan_to_num(resid, nan=np.inf).max()))
            else:
                out = lake.run(D, SUB, g["out"], every=C.STEP)
                max_resid.append(None)
        for j, (fam, hid) in enumerate(chunk):
            hid = list(hid); vis = [c for c in range(C.N_LEADS) if c not in hid]
            s = out[:, j * nt:(j + 1) * nt][:, :, hid].transpose(1, 0, 2).reshape(-1, len(hid))
            finite = bool(np.isfinite(s).all())
            pred = E.own_inverse(np.nan_to_num(s)) if finite else np.zeros_like(s)
            y = Xte[:, hid]
            row = {"family": fam, "hidden": hid, "lake_own": C.r2_cols(pred, y), "finite": finite}
            if finite:
                row["cos_spline"], row["rho_dist"] = spatial_diag(pred, Xte[:, vis], hid, vis)
            row["data_cos_spline"], row["data_rho_dist"] = spatial_diag(y, Xte[:, vis], hid, vis)
            res.append(row)
    summ = {}
    for fam in families:
        rs = [r for r in res if r["family"] == fam]
        summ[f"lake_own_{fam}"] = float(np.mean([r["lake_own"].mean() for r in rs]))
        ok = [r for r in rs if r["finite"]]
        summ[f"cos_spline_{fam}"] = float(np.mean([np.mean(r["cos_spline"]) for r in ok])) if ok else None
        summ[f"rho_dist_{fam}"] = float(np.mean([np.mean(r["rho_dist"]) for r in ok])) if ok else None
        summ[f"data_cos_spline_{fam}"] = float(np.mean([np.mean(r["data_cos_spline"]) for r in rs]))
        summ[f"data_rho_dist_{fam}"] = float(np.mean([np.mean(r["data_rho_dist"]) for r in rs]))
        summ[f"finite_{fam}"] = len(ok) / len(rs)
    import cadence
    return {"stage": "M13", "arm": arm, "config": cfg, "split": split, "subject": subject, "seed": seed,
            "random": random, "derange": derange, "n_trials": nt, "data": data["source"],
            "data_sha256": C.sha(seg), "init_sha256": init_sha,
            "lake_sha256": C.sha(np.asarray(g["lr"].brain.efficacy), np.asarray(g["lr"].brain.bias)),
            "dyn": {"dt": DT, "substeps": SUB, "tau_fast": TAU_FAST, "f0": F0},
            "cadence": {"file": cadence.__file__, "commit": C.git("rev-parse", "HEAD", cwd=Path(cadence.__file__).parents[2])},
            "static": is_static(arm, cfg), "max_residual": max_resid, "patterns": res, "summary": summ, "e_seal": seal, "env": C.env_info(),
            "wall_s": time.time() - t0, "finished": C.now()}


def job_name(subject, seed, arm, cfg, random=False, derange=None):
    tag = "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))
    return f"A{subject:02d}_s{seed}_{arm}{'_' + tag if tag else ''}{'_rand' if random else ''}" \
           f"{'' if derange is None else f'_der{derange}'}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", required=True); ap.add_argument("--cfg", default="{}")
    ap.add_argument("--split", default="test", choices=["test", "dev"]); ap.add_argument("--random", action="store_true")
    ap.add_argument("--derange", type=int); ap.add_argument("--families", default="P3")
    ap.add_argument("--step", required=True); ap.add_argument("--n-trials", type=int)
    a = ap.parse_args(argv)
    cfg = json.loads(a.cfg)
    path = OUT / a.step / a.split / (job_name(a.subject, a.seed, a.arm, cfg, a.random, a.derange) + ".json")
    if path.exists():
        print("exists", path); return
    rec = run_job(a.subject, a.seed, a.arm, cfg, a.split, a.random, a.derange, tuple(a.families.split(",")), a.n_trials)
    rec["step"] = a.step
    C.write_receipt(path, rec)
    print(path.name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec["summary"].items()},
          f"wall {rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
