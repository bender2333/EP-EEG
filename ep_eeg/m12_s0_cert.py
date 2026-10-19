"""M12 S0 A2 (reports/M12.md S0.3 + amendment 1, B2): no false positives of the weighted cascade certificate, with power.

Configs (generator seed GEN; config k uses default_rng([GEN, k])):
  k in [0, 1000)      certified targets: weighted rho_free 70% U(0, 0.95), 30% U[0.95, 0.999)
  k in [1000, 1200)   uncertified, rho_free U(1, 1.5]  (descriptive)
  k in [1200, 1400)   positive controls with multiple fixed points: W_ff = w (11^T - I)/(n - 1), w L in [1.5, 3], c = 0
Each config: sources in {0, 6, 64} (linear, one-way), free in {8, 32, 128, 256}, free->free density {0.1, 0.5, 1}, symmetric
or not (half each), 1-4 free populations with their own (a, b) ~ U[0.2, 2] x U[-2, 2], dt in {0.2, 0.5, 1}, input amplitude
{0.5, 4, 16}, bias N(0, 1), batch 8. Two starts (rest; v ~ N(0, 25)), equilibrate damping=0, residual tol 1e-9, at most
1e6 steps in 1000-step chunks; at every chunk end the visited-state bound max_i sum_j |W_ji| |s'(v_j)| <= rho + 1e-12 is
checked with an independent slope formula. Independent recomputation rho_ref from the dense W with L_ref from a
4e6-point grid on [-60, 60] and a bounded scalar optimisation; mutation certificates (alpha) L = a/2, (beta) receiver
L_i instead of sender L_j must be flagged by the recomputation check.
"""
from __future__ import annotations
import argparse, math, os, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import minimize_scalar
from . import m12_common as C
import cadence as cd
from cadence.brain import BrainState
from cadence.learning import evenodd_neuron_model

N_CERT, N_UNCERT, N_MULTI = 1000, 200, 200
TOL, MAX_STEPS, CHUNK, BATCH = 1e-9, 1_000_000, 1000, 8
TGRID = np.tanh(0.5 * np.linspace(-60.0, 60.0, 4_000_001))


def l_ref(a, b):
    """Independent of cadence: dense-grid maximum and a bounded optimisation, the larger of the two."""
    g = float(np.abs(0.5 * (1 - TGRID ** 2) * (a + 2 * b * TGRID)).max())
    f = lambda v: -abs(0.5 * (1 - math.tanh(v / 2) ** 2) * (a + 2 * b * math.tanh(v / 2)))
    best = 0.0
    for lo, hi in ((-60, 0), (0, 60)):
        r = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        best = max(best, -r.fun)
    return max(g, best)


def slope(v, a, b, lin):
    t = np.tanh(0.5 * v); d = 0.5 * (1 - t * t) * (a + 2 * b * t)
    return np.where(lin, 1.0, d)


def config(gen, k):
    rng = np.random.default_rng([gen, k])
    kind = "cert" if k < N_CERT else ("uncert" if k < N_CERT + N_UNCERT else "multi")
    ns = 0 if kind == "multi" else int(rng.choice([0, 6, 64]))
    nf = int(rng.choice([8, 32, 128, 256])); n = ns + nf
    dens = float(rng.choice([0.1, 0.5, 1.0])); sym = bool(rng.integers(2)); dt = float(rng.choice([0.2, 0.5, 1.0]))
    amp = float(rng.choice([0.5, 4.0, 16.0]))
    ngroups = 1 if kind == "multi" else int(rng.integers(1, 5))
    ga, gb = rng.uniform(0.2, 2.0, ngroups), rng.uniform(-2.0, 2.0, ngroups)
    member = np.sort(rng.integers(0, ngroups, nf)) if ngroups > 1 else np.zeros(nf, int)
    a = np.r_[np.ones(ns), ga[member]]; b = np.r_[np.zeros(ns), gb[member]]
    lin = np.r_[np.ones(ns, bool), np.zeros(nf, bool)]
    lipf = np.array([l_ref(x, y) for x, y in zip(ga, gb)])[member]
    if kind == "multi":
        w = float(rng.uniform(1.5, 3.0)) / lipf[0]
        Wff = w * (np.ones((nf, nf)) - np.eye(nf)) / (nf - 1); dens, sym = 1.0, True; bias = np.zeros(nf); target = None
    else:
        M = rng.normal(size=(nf, nf)) * (rng.random((nf, nf)) < dens); np.fill_diagonal(M, 0.0)
        if sym: M = np.triu(M, 1); M = M + M.T
        target = float(rng.uniform(0, 0.95) if (kind == "cert" and rng.random() < 0.7) else
                       rng.uniform(0.95, 0.999) if kind == "cert" else rng.uniform(1.0, 1.5))
        rowm = (np.abs(M) * lipf[:, None]).sum(0).max()   # M[pre, post]: sum over senders j of |W_ji| L_j
        Wff = M * (target / rowm) if rowm > 0 else M; bias = rng.normal(size=nf)
    Win = rng.normal(size=(ns, nf)) * (rng.random((ns, nf)) < 0.5) if ns else np.zeros((0, nf))
    W = np.zeros((n, n)); W[:ns, ns:] = Win; W[ns:, ns:] = Wff
    X = np.zeros((BATCH, n))
    if ns: X[:, :ns] = amp * rng.normal(size=(BATCH, ns))
    start = rng.normal(0, 5.0, (BATCH, n))
    return dict(kind=kind, ns=ns, nf=nf, n=n, dens=dens, sym=sym, dt=dt, amp=amp, ngroups=ngroups, ga=ga, gb=gb, a=a, b=b, lin=lin,
                W=W, bias=np.r_[np.zeros(ns), bias], X=X, start=start, target=target, lipf_ref=lipf)


def make_brain(cf, backend):
    W = cf["W"]; pre, post = np.nonzero(W)
    conn = cd.Connectome.from_synapses(cf["n"], pre=pre, post=post)
    eff = W[conn.pre, conn.post]
    model = evenodd_neuron_model(cf["a"], cf["b"], linear=np.flatnonzero(cf["lin"]), dt=cf["dt"])
    kw = {} if backend == "cpu64" else {"backend": "torch", "device": "cpu", "precision": "float64"}
    return cd.NeuralGraph(conn, model, efficacy=eff, bias=cf["bias"], **kw), model


def rho_formula(W, ns, lip_sender, lip_receiver=None):
    """max over free receivers i of sum over free senders j of |W_ji| * weight; weight = L_j, or L_i for mutant beta."""
    A = np.abs(W[ns:, ns:])
    if lip_receiver is None: return float((A * lip_sender[:, None]).sum(0).max()) if A.size else 0.0
    return float((A.sum(0) * lip_receiver).max()) if A.size else 0.0


def solve(brain, cf, start):
    """Chunked equilibrate with the visited-state bound check; returns final v, steps, qualified, finite, worst bound."""
    ns, W = cf["ns"], cf["W"]; A = np.abs(W[ns:, ns:]); a, b, lin = cf["a"][ns:], cf["b"][ns:], cf["lin"][ns:]
    state = None if start is None else BrainState(v=start.copy(), activation=brain.neuron_model.activation(start),
                                                    adaptation=np.zeros_like(start), steps=0)
    used, worst = 0, 0.0
    while True:
        eq = brain.equilibrate(cf["X"], budget=CHUNK, chunk=32, tolerance=TOL, state=state, damping=0)
        used += eq.steps; state = eq.state; v = np.asarray(state.v)
        finite = bool(np.isfinite(v).all() and np.isfinite(np.asarray(eq.residual)).all())
        if not finite: return v, used, False, False, worst, np.asarray(eq.residual)
        if A.size:
            bound = (A[None] * np.abs(slope(v[:, ns:], a, b, lin))[:, :, None]).sum(1).max()
            worst = max(worst, float(bound))
        if np.all(eq.qualified) or used >= MAX_STEPS:
            return v, used, bool(np.all(eq.qualified)), True, worst, np.asarray(eq.residual)


def run_config(args):
    gen, k, backend, dev = args
    out = C.OUT / "S0" / ("dev" if dev else "A2") / C.run_commit() / backend; nm = f"g{gen}_k{k:04d}"
    prev = C.existing(out, nm)
    if prev: return prev
    t0 = time.time(); cf = config(gen, k); ns = cf["ns"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brain, model = make_brain(cf, backend)
        cert = cd.certificate(brain, sources=range(ns)) if ns else cd.certificate(brain)
    rho_c = cert.weighted_row_mass_free if ns else cert.weighted_row_mass
    lip_ref = np.r_[np.ones(ns), cf["lipf_ref"]]
    rho_ref = rho_formula(cf["W"], ns, lip_ref[ns:])
    tol_rc = 1e-12 * max(1.0, rho_ref) + 1e-9 * float(np.abs(cf["W"][ns:, ns:]).sum(0).max() if cf["nf"] else 0.0)
    rho_alpha = rho_formula(cf["W"], ns, 0.5 * cf["a"][ns:]); rho_beta = rho_formula(cf["W"], ns, None, lip_ref[ns:])
    certified = bool(cert.certified)
    near = abs(rho_ref - 1.0) <= 1e-9
    recompute_ok = abs(rho_c - rho_ref) <= tol_rc and (near or certified == (rho_ref < 1.0))
    va, sa, qa, fa, wa, ra = solve(brain, cf, None)
    vb, sb, qb, fb, wb, rb = solve(brain, cf, cf["start"])
    diff = float(np.abs(va - vb).max()) if fa and fb else float("inf")
    m_in = float((np.abs(cf["W"][:ns, ns:])).sum(0).max()) if ns else 0.0   # sources linear: L_j = 1
    uniq_thr = 2 * TOL * (1 + m_in) / (1 - rho_c) + 1e-9 if rho_c < 1 else 1e-6
    q = cert.rate
    rate_steps = (4 * math.ceil(math.log(1e-10 * (1 - q)) / math.log(q)) if 0 < q < 1 else 4) + \
                 (math.ceil(math.log(1e-10) / math.log(1 - cf["dt"])) if cf["dt"] < 1 else 1)
    flagged = (not (qa and qb)) or (not (fa and fb)) or diff > uniq_thr
    lc = model.lipschitz_per_neuron(cf["n"])[ns:]
    res = {"gen": gen, "k": k, "kind": cf["kind"], "backend": backend, "ns": ns, "nf": cf["nf"], "density": cf["dens"], "symmetric": cf["sym"],
           "dt": cf["dt"], "amp": cf["amp"], "groups": cf["ngroups"], "group_a": cf["ga"], "group_b": cf["gb"], "target": cf["target"],
           "W_sha256": C.sha(cf["W"]), "X_sha256": C.sha(cf["X"]), "bias_sha256": C.sha(cf["bias"]), "start_sha256": C.sha(cf["start"]),
           "certificate": cert.to_dict(), "lipschitz_cadence": sorted(set(lc.tolist())), "lipschitz_ref": sorted(set(cf["lipf_ref"].tolist())),
           "fixed_incoming_mass": cd.ep_structure(brain, fixed_inputs=range(ns)).fixed_incoming_mass,
           "rho_cadence": rho_c, "rho_ref": rho_ref, "recompute_tol": tol_rc, "recompute_ok": bool(recompute_ok), "near_one": near,
           "rho_mutant_alpha": rho_alpha, "rho_mutant_beta": rho_beta,
           "mutant_alpha_flagged": bool(abs(rho_alpha - rho_ref) > tol_rc or ((rho_alpha < 1) != (rho_ref < 1))),
           "mutant_beta_flagged": bool(abs(rho_beta - rho_ref) > tol_rc or ((rho_beta < 1) != (rho_ref < 1))),
           "steps": [sa, sb], "qualified": [qa, qb], "finite": [fa, fb], "residual_max": [float(np.max(ra)), float(np.max(rb))],
           "start_diff": diff, "m_in": m_in, "uniqueness_threshold": uniq_thr,
           "visited_bound_max": max(wa, wb), "visited_bound_ok": bool(max(wa, wb) <= rho_c + 1e-12),
           "detector_flagged": bool(flagged), "rate_bound_steps": rate_steps, "rate_violation": bool(max(sa, sb) > rate_steps),
           "false_positive": bool(certified and flagged), "wall_s": time.time() - t0}
    C.write_receipt(out, nm, res, dev=dev)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--backend", default="cpu64", choices=("cpu64", "torch64"))
    ap.add_argument("--dev", action="store_true"); ap.add_argument("--only", default=None, help="dev: comma list of k")
    a = ap.parse_args(); C.e_seal(); gen = 100 if a.dev else 0
    if not a.dev: C.start_clock(f"m12_s0_cert --backend {a.backend}")
    ks = [int(x) for x in a.only.split(",")] if a.only else list(range(N_CERT + N_UNCERT + N_MULTI))
    t0 = time.time(); fp = 0; done = 0
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(run_config, [(gen, k, a.backend, a.dev) for k in ks], chunksize=1):
            done += 1; fp += r["false_positive"]
            if a.dev or done % 50 == 0 or r["false_positive"] or not r["recompute_ok"] or not r["visited_bound_ok"]:
                print(f"[{done}/{len(ks)}] k{r['k']} {r['kind']} ns{r['ns']} nf{r['nf']} dt{r['dt']} rho {r['rho_cadence']:.4f} ref {r['rho_ref']:.4f} "
                      f"fp {r['false_positive']} recompute {r['recompute_ok']} bound {r['visited_bound_ok']} flagged {r['detector_flagged']} "
                      f"diff {r['start_diff']:.1e} steps {r['steps']} mutα {r['mutant_alpha_flagged']} mutβ {r['mutant_beta_flagged']} "
                      f"{r['wall_s']:.0f}s  (fp total {fp})", flush=True)
    if not a.dev:
        C.ledger("M12-S0", f"A2_{a.backend}", "NA", 1, time.time() - t0, "done", f"A2 {a.backend}: {done} configs, {fp} false positives (full judgement in m12_verify)")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
