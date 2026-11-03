"""M12 S0 (reports/M12.md S0.3 + amendment 1): D1 equivalence on the even/odd neuron, three variants.

  std     M11 D1 graph (6 sources -> 10 hidden one-way; hidden <-> hidden and hidden <-> 3 outputs reciprocal),
          sources linear (not through the activation), N1 = (a, b) = (1, 0.25) or N2 = (1, 1) on hidden and outputs.
  hetero  same graph, hidden 0-4 N1, hidden 5-9 N2, outputs N2 (amendment 1, N6).
  tied    D1-T: 3 channels x 3 positions = 9 linear sources; 10 hidden = filters 0, 1 at 3 positions (group size 3),
          filters 2, 3 at 2 positions (size 2); hidden (k, p) reads the 3 sources at p with W[k, c] tied across p;
          N2; Learner tie_groups with synapse_rate = group size (amendment 1, B4).
Free -> free edges scaled so the weighted cascade row mass max_i sum_j |W_ji| L_j = budget (sources excluded).
Physical coordinates: one weight per reciprocal pair, one per tie group, one bias per free neuron.
Parts (order fixed by amendment 1 / M11 M1-11): `valid` (IFT vs FD on numpy f64, torch-f64 IFT vs reference)
is run and committed before `ep` (EP grid on cpu64 / torch64 / torch32 against the reference IFT).
`regress` is the N12 pre-check (see m12_s0_regress.py).
"""
from __future__ import annotations
import argparse, itertools, os, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from . import m12_common as C
import cadence as cd
from cadence.learning import Learner, LearnerConfig, LearningPhaseError, evenodd_neuron_model, time_shared_entry, tie_rates
from .ift import implicit_grad, ep_equivalent, loss_grad_s

NEURONS = {"N1": (1.0, 0.25), "N2": (1.0, 1.0)}
DEV_NEURON = (1.0, 0.6)
BETAS, TOLS = (0.1, 0.04, 0.01, 0.0025), (3e-3, 1e-4, 1e-6, 1e-9)
BUDGETS, AMPS, NUDGES, SEEDS = (0.3, 0.6, 0.9), (0.5, 4.0), ("ce", "quad"), (0, 1, 2)
BACKENDS = ("cpu64", "torch64", "torch32")
NO, BATCH, PROD = 3, 16, (0.04, 3e-3)


def backend_kw(backend):
    return {"cpu64": {}, "torch64": {"backend": "torch", "device": "cpu", "precision": "float64"},
            "torch32": {"backend": "torch", "device": "cpu", "precision": "float32"}}[backend]


def graph(variant, seed, neuron):
    """Synapses, classes, tie groups, per-neuron (a, b), sources, outputs for one variant."""
    rng = np.random.default_rng(seed)
    if variant == "tied":
        pre, post, tie, entry = time_shared_entry(3, 3, 4, filter_positions=[[0, 1, 2], [0, 1, 2], [0, 1], [1, 2]])
        ns, nh = 9, 10
        shared = rng.normal(size=int(tie.max()) + 1)
        syn = {(int(p), int(q)): float(shared[g]) for p, q, g in zip(pre, post, tie)}
        grp = {(int(p), int(q)): int(g) for p, q, g in zip(pre, post, tie)}
        cls = {k: "deep" for k in syn}
    else:
        ns, nh = 6, 10
        syn, cls, grp = {}, {}, {}
        for a in range(ns):
            for h in range(ns, ns + nh): syn[(a, h)] = rng.normal(); cls[(a, h)] = "deep"
    hid = range(ns, ns + nh); out = list(range(ns + nh, ns + nh + NO))
    for a, b in itertools.combinations(hid, 2):
        w = rng.normal(); syn[(a, b)] = syn[(b, a)] = w; cls[(a, b)] = cls[(b, a)] = "lateral"
    for h in hid:
        for o in out:
            w = rng.normal(); syn[(h, o)] = syn[(o, h)] = w; cls[(h, o)] = cls[(o, h)] = "readout"
    n = ns + nh + NO
    a_n, b_n = np.ones(n), np.zeros(n)
    if variant == "hetero":
        (a1, b1), (a2, b2) = NEURONS["N1"], NEURONS["N2"]
        a_n[ns:ns + 5], b_n[ns:ns + 5] = a1, b1; a_n[ns + 5:], b_n[ns + 5:] = a2, b2
    else:
        a_n[ns:], b_n[ns:] = neuron
    return syn, cls, grp, a_n, b_n, ns, out, n


def build(variant, seed, neuron, budget, nudge, beta=0.04, tol=3e-3, backend="cpu64"):
    syn, cls, grp, a_n, b_n, ns, out, n = graph(variant, seed, neuron)
    keys = list(syn)
    conn = cd.Connectome.from_synapses(n, pre=np.array([p for p, _ in keys]), post=np.array([q for _, q in keys]))
    pairs = list(zip(conn.pre.tolist(), conn.post.tolist()))
    eff = np.array([syn[k] for k in pairs]); kind = np.array([cls[k] for k in pairs])
    groups = np.array([grp.get(k, -1) for k in pairs], dtype=np.int64)
    model = evenodd_neuron_model(a_n, b_n, linear=range(ns), dt=0.5)
    lip = model.lipschitz_per_neuron(n)
    counted = kind != "deep"
    rowm = np.bincount(conn.post, np.abs(eff) * lip[conn.pre] * counted, minlength=n)[ns:].max()
    eff = np.where(counted, eff * budget / rowm, eff)
    brain = cd.NeuralGraph(conn, model, efficacy=eff, bias=np.zeros(n), **backend_kw(backend))
    lc = LearnerConfig(beta=beta, eta=0.0, eta_bias=0.0, centered=True, free_steps=50000, nudged_steps=50000, tolerance=tol,
                       nudge="cross_entropy" if nudge == "ce" else "quadratic", temperature=0.1, normalize=0.0,
                       normalize_floor=1e-3, momentum=0.0, decay=0.0, scale_cap=1e9, qualified=True, damping=3, surprise=False)
    pn = np.zeros(n, bool); pn[ns:] = True
    tied = variant == "tied"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = Learner(brain, out, config=lc, plastic_synapses=np.ones(len(eff), bool), plastic_neurons=pn, reciprocal=True,
                     tie_groups=groups if tied else None, synapse_rate=tie_rates(groups) if tied else None)
    return lr, kind, groups, conn, ns, model


def batch(seed, amp, n, ns):
    rng = np.random.default_rng(seed + 1000); D = np.zeros((BATCH, n)); D[:, :ns] = amp * rng.normal(size=(BATCH, ns))
    return D, rng.integers(0, NO, BATCH)


def coords(lr, kind, groups, ns, n):
    """Physical parameters: list of synapse-index arrays (one per parameter) and their classes; then free biases."""
    rev = lr.reverse; params, pcls, seen = [], [], set()
    for g in np.unique(groups[groups >= 0]):
        m = np.flatnonzero(groups == g); params.append(m); pcls.append("deep"); seen.update(m.tolist())
    for i in range(len(rev)):
        if i in seen: continue
        if rev[i] >= 0 and rev[i] < i: continue
        params.append(np.array([i])); pcls.append(str(kind[i]))
    return params, np.array(pcls + ["bias"] * (n - ns))


def to_phys(edge_vec, bias_vec, params, ns):
    """Sum a per-synapse vector over each physical parameter's representative synapses (one direction per pair)."""
    return np.r_[[edge_vec[p].sum() for p in params], bias_vec[ns:]]


def cosine(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def compare(x, ref, classes):
    out = {"all": {"cos": cosine(x, ref), "rel_err": float(np.linalg.norm(x - ref) / (np.linalg.norm(ref) + 1e-300)),
                   "scale": float(np.linalg.norm(x) / (np.linalg.norm(ref) + 1e-300))}}
    for c in np.unique(classes):
        m = classes == c
        out[str(c)] = {"cos": cosine(x[m], ref[m]), "rel_err": float(np.linalg.norm(x[m] - ref[m]) / (np.linalg.norm(ref[m]) + 1e-300))}
    return out


def finite_loss(lr, D, y):
    eq = lr.brain.equilibrate(D, budget=50000, tolerance=1e-11, damping=3)
    if not np.all(eq.converged): raise ArithmeticError("free phase did not converge in the finite-difference check")
    return loss_grad_s(lr, np.asarray(eq.state.activation), np.asarray(y))[1]


def tight_ift(lr, D, y, params, ns):
    tight = lr.brain.equilibrate(D, budget=50000, tolerance=1e-11, damping=3)
    ok = bool(np.all(tight.converged))
    g, gb, loss, factor = implicit_grad(lr, tight.state, y); ce, cb = ep_equivalent(lr, g, gb)
    return ok, to_phys(ce, cb, params, ns), tight, loss


def name_of(variant, nname, seed, budget, amp, nudge): return f"{variant}_{nname}_s{seed}_b{budget}_amp{amp}_{nudge}"


def valid_cell(args):
    variant, nname, seed, budget, amp, nudge, dev = args
    out = C.OUT / "S0" / ("dev" if dev else "D1") / C.run_commit() / "valid"; nm = name_of(variant, nname, seed, budget, amp, nudge)
    prev = C.existing(out, nm)
    if prev: return prev
    neuron = DEV_NEURON if nname == "dev" else NEURONS.get(nname, (1.0, 1.0))
    t0 = time.time(); lr, kind, groups, conn, ns, model = build(variant, seed, neuron, budget, nudge)
    n = conn.n; D, y = batch(seed, amp, n, ns); params, classes = coords(lr, kind, groups, ns, n)
    ok, ift, tight, loss = tight_ift(lr, D, y, params, ns)
    v = np.asarray(tight.state.v)
    br = lr.brain; eff0 = np.asarray(br.efficacy).copy(); b0 = np.asarray(br.bias).copy(); h = 1e-4; fd = []
    for p in params:
        sel = np.unique(np.r_[p, lr.reverse[p][lr.reverse[p] >= 0]])
        vals = []
        for sgn in (1, -1):
            e = eff0.copy(); e[sel] += sgn * h; lr.brain = br.with_parameters(efficacy=e, bias=b0); vals.append(finite_loss(lr, D, y))
        fd.append(-(vals[0] - vals[1]) / (2 * h))
    for j in range(ns, n):
        vals = []
        for sgn in (1, -1):
            bb = b0.copy(); bb[j] += sgn * h; lr.brain = br.with_parameters(efficacy=eff0, bias=bb); vals.append(finite_loss(lr, D, y))
        fd.append(-(vals[0] - vals[1]) / (2 * h))
    lr.brain = br; fd = np.array(fd)
    # torch-f64 validity (amendment 1, N4): IFT at the torch tight state vs the numpy reference IFT
    lt, *_ = build(variant, seed, neuron, budget, nudge, backend="torch64")
    ok_t, ift_t, tight_t, _ = tight_ift(lt, D, y, params, ns)
    cert = cd.certificate(br, sources=range(ns))
    res = {"variant": variant, "neuron": nname, "seed": seed, "budget": budget, "amp": amp, "nudge": nudge, "loss": loss,
           "tight_converged": ok, "tight_converged_torch64": ok_t,
           "a": np.broadcast_to(np.asarray(model.odd, float), (n,)),
           "b": np.broadcast_to(np.asarray(model.even, float), (n,)), "linear": list(model.linear), "dt": model.dt,
           "lipschitz": model.lipschitz_per_neuron(n), "certificate": cert.to_dict(),
           "fixed_incoming_mass": cd.ep_structure(br, fixed_inputs=range(ns)).fixed_incoming_mass,
           "sat_frac": float((np.abs(v[:, ns:]) > 2).mean()),
           "grad_norm_ift": {c: float(np.linalg.norm(ift[classes == c])) for c in np.unique(classes)},
           "classes": classes, "g_ift": ift, "g_fd": fd, "g_ift_torch64": ift_t,
           "ift_vs_fd": compare(ift, fd, classes), "torch64_ift_vs_ref": compare(ift_t, ift, classes),
           "inputs_sha256": C.sha(D, y), "weights_sha256": C.sha(eff0, b0), "fd_h": h, "tight_tol": 1e-11,
           "tie_rate_rule": "synapse_rate = group size; group step = eta * sum of member contrasts" if variant == "tied" else None,
           "wall_s": time.time() - t0}
    C.write_receipt(out, nm, res, dev=dev)
    return res


def ep_cell(args):
    variant, nname, seed, budget, amp, nudge, backend, dev = args
    base = C.OUT / "S0" / ("dev" if dev else "D1") / C.run_commit()
    nm = name_of(variant, nname, seed, budget, amp, nudge)
    prev = C.existing(base / "ep" / backend, nm)
    if prev: return prev
    ref = C.existing(base / "valid", nm)
    if ref is None: raise RuntimeError(f"valid receipt missing for {nm}: run --part valid first")
    neuron = DEV_NEURON if nname == "dev" else NEURONS.get(nname, (1.0, 1.0))
    ift = np.asarray(ref["g_ift"]); fd = np.asarray(ref["g_fd"]); classes = np.asarray(ref["classes"])
    t0 = time.time(); res = {"variant": variant, "neuron": nname, "seed": seed, "budget": budget, "amp": amp, "nudge": nudge,
                             "backend": backend, "grid": []}
    for beta, tol in itertools.product(BETAS, TOLS):
        lr, kind, groups, conn, ns, model = build(variant, seed, neuron, budget, nudge, beta=beta, tol=tol, backend=backend)
        n = conn.n; D, y = batch(seed, amp, n, ns); params, _ = coords(lr, kind, groups, ns, n)
        cell = {"beta": beta, "tol": tol}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                free = lr.free(D); tgt = lr.targets(y)
                plus = lr.nudged(D, free, tgt, sign=1.0); minus = lr.nudged(D, free, tgt, sign=-1.0)
                e, nb = lr.contrast(free, plus, minus)
            e, nb = np.asarray(e, float), np.asarray(nb, float)
            ep = to_phys(e, nb, params, ns)
            finite = bool(np.isfinite(ep).all())
            cell.update({"refused": None, "finite": finite, "g_ep": ep,
                         "ep_vs_ift": compare(ep, ift, classes) if finite else None,
                         "ep_vs_fd": compare(ep, fd, classes) if finite else None,
                         "steps": {k: int(getattr(st, "steps", -1) or -1) for k, st in (("free", free), ("plus", plus), ("minus", minus))}})
            if variant == "tied" and (beta, tol) == PROD and finite:  # conversion check: Learner.apply's group step
                eta = 0.5; before = np.asarray(lr.brain.efficacy).copy()
                lr.apply(eta * e, np.zeros(n)); moved = np.asarray(lr.brain.efficacy) - before
                gs = [p for p in params if len(p) > 1]
                got = np.array([moved[p[0]] for p in gs]); want = np.array([eta * e[p].sum() for p in gs])
                cell["conversion"] = {"rel_err": float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-300)),
                                      "within_group_ptp": float(max(np.ptp(moved[p]) for p in gs)),
                                      "group_sizes": sorted({len(p) for p in gs})}
        except LearningPhaseError as err:
            cell.update({"refused": err.phase})
        res["grid"].append(cell)
    res["wall_s"] = time.time() - t0
    C.write_receipt(base / "ep" / backend, nm, res, dev=dev)
    return res


def jobs(part, dev):
    if dev:
        cells = [("std", "dev", 100, 0.6, 4.0, "quad"), ("tied", "dev", 100, 0.6, 0.5, "ce"), ("hetero", "dev", 100, 0.9, 0.5, "ce")]
    else:
        cells = [("std", nn, s, b, a, u) for nn, b, a, u, s in itertools.product(NEURONS, BUDGETS, AMPS, NUDGES, SEEDS)]
        cells += [("hetero", "N1N2", s, b, a, u) for b, a, u, s in itertools.product(BUDGETS, AMPS, NUDGES, SEEDS)]
        cells += [("tied", "N2", s, b, a, u) for b, a, u, s in itertools.product(BUDGETS, AMPS, NUDGES, SEEDS)]
    if part == "valid": return [(*c, dev) for c in cells]
    return [(*c, bk, dev) for c in cells for bk in BACKENDS]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--part", required=True, choices=("valid", "ep"))
    ap.add_argument("--dev", action="store_true", help="dev run: seed 100, (a, b) = (1, 0.6), receipts dev=true (amendment 1, N10)")
    a = ap.parse_args(); C.e_seal()
    if not a.dev: C.start_clock(f"m12_s0_d1 --part {a.part}")
    t0 = time.time(); todo = jobs(a.part, a.dev); n_ok = 0
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(valid_cell if a.part == "valid" else ep_cell, todo):
            n_ok += 1
            if a.part == "valid":
                print(f"[{n_ok}/{len(todo)}] {r['variant']} {r['neuron']} s{r['seed']} b{r['budget']} amp{r['amp']} {r['nudge']}: "
                      f"ift~fd {r['ift_vs_fd']['all']['cos']:.6f} torch {r['torch64_ift_vs_ref']['all']['cos']:.6f} {r['wall_s']:.0f}s", flush=True)
            else:
                prod = next(g for g in r["grid"] if (g["beta"], g["tol"]) == PROD)
                c = prod.get("ep_vs_ift") or {}
                print(f"[{n_ok}/{len(todo)}] {r['variant']} {r['neuron']} {r['backend']} s{r['seed']} b{r['budget']} amp{r['amp']} {r['nudge']}: "
                      f"prod ep~ift {c.get('all', {}).get('cos', float('nan')):.6f} refused {prod.get('refused')} {r['wall_s']:.0f}s", flush=True)
    if not a.dev:
        C.ledger("M12-S0", f"D1_{a.part}", "NA", 3, time.time() - t0, "done", f"S0 D1 part {a.part}: {n_ok} receipts (std 72 + hetero 36 + tied 36 cells)")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
