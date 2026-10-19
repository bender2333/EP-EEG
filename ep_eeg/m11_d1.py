"""M11 D1 (reports/M11.md §6.1): EP update vs implicit gradient vs central differences on small controllable
graphs (and the k=1 sandbox on A01), extending ep_eeg/gradcheck.py into a full numerical report.

Small graph: 6 source neurons -> (one-way) 10 hidden free neurons; hidden <-> hidden all reciprocal pairs;
hidden <-> 3 outputs reciprocal; no input -> output. Free -> free row mass scaled so that L * rho_W = budget
(ADR-0012 counting: sources excluded). Batch of 16 random inputs (stimulus N(0,1) * amp), random labels.
Parameters compared in physical coordinates: a reciprocal pair is one weight; classes deep (source->hidden),
lateral (hidden<->hidden), readout (hidden<->output), bias (free neurons).
  g_EP   centred +-beta contrast of cadence's Learner (qualified phases at the given tolerance)
  g_IFT  ep_eeg.ift (exact, at a free state solved to 1e-11)
  g_FD   central difference of the free-equilibrium loss, h = 1e-4, phases to 1e-11
Asymmetry sweep (secondary): reciprocal=False, the upper->lower direction of every free pair multiplied by
(1 + eps * xi), xi ~ N(0, 1), row mass rescaled back to the budget; EP is compared with the directed
gradient and with the pair-symmetrised gradient.
"""
from __future__ import annotations
import argparse, itertools, json, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cadence as cd
from cadence.learning import Learner, LearnerConfig, learning_neuron_model, LearningPhaseError
from cadence.certificate import lipschitz_constant
from .data import ROOT
from .ift import implicit_grad, ep_equivalent, free_loss
from .m11_common import write_receipt, git, M11_OUT, run_commit, existing

OUT = M11_OUT / "D1"
NS, NH, NO, BATCH = 6, 10, 3, 16
BETAS, TOLS = (0.1, 0.04, 0.01, 0.0025), (3e-3, 1e-4, 1e-6, 1e-9)
BUDGETS, AMPS, NUDGES, LEAKS = (0.3, 0.6, 0.9), (0.5, 4.0), ("ce", "quad"), (1.0, 0.1)
EPS = (0.0, 0.1, 0.3, 1.0)


def build(seed, leak, budget, nudge, beta=0.04, tol=3e-3, reciprocal=True, eps=0.0):
    rng = np.random.default_rng(seed); src = range(NS); hid = range(NS, NS + NH); out = list(range(NS + NH, NS + NH + NO))
    syn, cls = {}, {}
    for a in src:
        for h in hid: syn[(a, h)] = rng.normal(); cls[(a, h)] = "deep"
    for a, b in itertools.combinations(hid, 2): w = rng.normal(); syn[(a, b)] = syn[(b, a)] = w; cls[(a, b)] = cls[(b, a)] = "lateral"
    for h in hid:
        for o in out: w = rng.normal(); syn[(h, o)] = syn[(o, h)] = w; cls[(h, o)] = cls[(o, h)] = "readout"
    keys = list(syn); n = NS + NH + NO
    conn = cd.Connectome.from_synapses(n, pre=np.array([p for p, _ in keys]), post=np.array([q for _, q in keys]))
    eff = np.array([syn[(p, q)] for p, q in zip(conn.pre, conn.post)]); kind = np.array([cls[(p, q)] for p, q in zip(conn.pre, conn.post)])
    if eps > 0:  # feedback direction (pre > post) perturbed
        fb = (kind != "deep") & (conn.pre > conn.post); eff = np.where(fb, eff * (1 + eps * rng.normal(size=len(eff))), eff)
    model = learning_neuron_model(leak=leak); L = lipschitz_constant(model)
    counted = kind != "deep"; rowm = np.bincount(conn.post, np.abs(eff) * counted, minlength=n).max()
    eff = np.where(counted, eff * budget / (L * rowm), eff)
    brain = cd.NeuralGraph(conn, model, efficacy=eff, bias=np.zeros(n))
    lc = LearnerConfig(beta=beta, eta=0.0, eta_bias=0.0, centered=True, free_steps=50000, nudged_steps=50000, tolerance=tol,
                       nudge="cross_entropy" if nudge == "ce" else "quadratic", temperature=0.1, normalize=0.0,
                       normalize_floor=1e-3, momentum=0.0, decay=0.0, scale_cap=1e9, qualified=True, damping=3, surprise=False)
    pn = np.zeros(n, bool); pn[NS:] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = Learner(brain, out, config=lc, plastic_synapses=np.ones(len(eff), bool), plastic_neurons=pn, reciprocal=reciprocal)
    return lr, kind, conn


def batch(seed, amp, n):
    rng = np.random.default_rng(seed + 1000); D = np.zeros((BATCH, n)); D[:, :NS] = amp * rng.normal(size=(BATCH, NS))
    return D, rng.integers(0, NO, BATCH)


def physical(lr, kind):
    """One coordinate per physical parameter: unpaired synapses, one direction of each pair, free-neuron biases."""
    rev = lr.reverse; keep = (rev < 0) | (np.arange(len(rev)) < rev)
    return np.flatnonzero(keep), kind[keep]


def cosine(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def compare(ep, ref, classes):
    out = {"all": {"cos": cosine(ep, ref), "rel_err": float(np.linalg.norm(ep - ref) / (np.linalg.norm(ref) + 1e-300)),
                   "scale": float(np.linalg.norm(ep) / (np.linalg.norm(ref) + 1e-300))}}
    for c in np.unique(classes):
        m = classes == c; out[c] = {"cos": cosine(ep[m], ref[m]), "rel_err": float(np.linalg.norm(ep[m] - ref[m]) / (np.linalg.norm(ref[m]) + 1e-300))}
    return out


def ep_contrast(lr, D, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        free = lr.free(D); tgt = lr.targets(y)
        plus = lr.nudged(D, free, tgt, sign=1.0); minus = lr.nudged(D, free, tgt, sign=-1.0)
    return lr.contrast(free, plus, minus)


def cell(seed, leak, budget, amp, nudge, valid_only=False):
    prev = existing(OUT / run_commit() / ("valid" if valid_only else "small"), f"s{seed}_leak{leak}_b{budget}_amp{amp}_{nudge}")
    if prev: return prev
    t0 = time.time(); lr, kind, conn = build(seed, leak, budget, nudge); D, y = batch(seed, amp, conn.n)
    idx, pcls = physical(lr, kind); free_idx = np.arange(NS, conn.n)
    classes = np.r_[pcls, np.array(["bias"] * len(free_idx))]
    tight = lr.brain.equilibrate(D, budget=50000, tolerance=1e-11, damping=3); assert np.all(tight.converged)
    v = np.asarray(tight.state.v)
    g, gb, loss0, factor = implicit_grad(lr, tight.state, y); ce, cb = ep_equivalent(lr, g, gb)
    ift = np.r_[ce[idx], cb[free_idx]]
    # finite differences in physical coordinates
    br = lr.brain; eff0 = np.asarray(br.efficacy).copy(); b0 = np.asarray(br.bias).copy(); h = 1e-4; fd = []
    for i in idx:
        sel = [i] + ([lr.reverse[i]] if lr.reverse[i] >= 0 else [])
        vals = []
        for sgn in (1, -1):
            e = eff0.copy(); e[sel] += sgn * h; lr.brain = br.with_parameters(efficacy=e, bias=b0); vals.append(free_loss(lr, D, y))
        fd.append(-(vals[0] - vals[1]) / (2 * h))
    for j in free_idx:
        vals = []
        for sgn in (1, -1):
            bb = b0.copy(); bb[j] += sgn * h; lr.brain = br.with_parameters(efficacy=eff0, bias=bb); vals.append(free_loss(lr, D, y))
        fd.append(-(vals[0] - vals[1]) / (2 * h))
    lr.brain = br; fd = np.array(fd)
    res = {"seed": seed, "leak": leak, "budget": budget, "amp": amp, "nudge": nudge, "loss": loss0,
           "sat_frac": float((np.abs(v[:, NS:]) > 2).mean()), "gain_factor_max_dev": float(np.abs(factor - 1).max()),
           "ift_vs_fd": compare(ift, fd, classes), "grid": []}
    for beta, tol in ([] if valid_only else itertools.product(BETAS, TOLS)):
        lb, _, _ = build(seed, leak, budget, nudge, beta=beta, tol=tol)
        try:
            e, nb = ep_contrast(lb, D, y); ep = np.r_[e[idx], nb[free_idx]]
            res["grid"].append({"beta": beta, "tol": tol, "refused": None, "ep_vs_ift": compare(ep, ift, classes),
                                "ep_vs_fd": compare(ep, fd, classes)})
        except LearningPhaseError as err:
            res["grid"].append({"beta": beta, "tol": tol, "refused": err.phase})
    res["wall_s"] = time.time() - t0
    write_receipt(OUT / run_commit() / ("valid" if valid_only else "small"), f"s{seed}_leak{leak}_b{budget}_amp{amp}_{nudge}", res)
    return res


def asym_cell(seed, eps, amp, nudge, budget=0.6, leak=1.0):
    lr, kind, conn = build(seed, leak, budget, nudge, reciprocal=False, eps=eps); D, y = batch(seed, amp, conn.n)
    tight = lr.brain.equilibrate(D, budget=50000, tolerance=1e-11, damping=3); assert np.all(tight.converged)
    g, gb, _, _ = implicit_grad(lr, tight.state, y)
    pair = {(int(p), int(q)): i for i, (p, q) in enumerate(zip(conn.pre, conn.post))}
    rev = np.array([pair.get((int(q), int(p)), -1) for p, q in zip(conn.pre, conn.post)])
    sym = -g.copy(); m = rev >= 0; sym[m] = -(g[m] + g[rev[m]])
    try:
        e, _ = ep_contrast(lr, D, y)
    except LearningPhaseError as err:
        return {"seed": seed, "eps": eps, "amp": amp, "nudge": nudge, "refused": err.phase}
    return {"seed": seed, "eps": eps, "amp": amp, "nudge": nudge, "refused": None,
            "ep_vs_directed": compare(e, -g, kind), "ep_vs_pair_symmetrised": compare(e, sym, kind)}


def real_sandbox(beta, tol):
    """The M1 gradcheck graph (k=1 sandbox, A01 rows) with IFT added: every synapse, no sampling."""
    from .sandbox import Sandbox, SandboxConfig
    from .data import load_subject
    cfg = SandboxConfig(leak=1.0, rowmass_project=0.9, eta=0.5, temperature=0.01, k=1, sigma_col=0.05, sigma_out=0.1,
                        tolerance=tol, free_steps=50000, nudged_steps=50000, beta=beta)
    sb = Sandbox(cfg, 0); d = load_subject(1); X = d["train"]["track2_x"][:16, 5]; y = d["train"]["y"][:16]; D = sb.drive(X); L = sb.learner
    tight = L.brain.equilibrate(D, budget=50000, tolerance=1e-11, damping=3)
    g, gb, _, _ = implicit_grad(L, tight.state, y); ce, cb = ep_equivalent(L, g, gb)
    idx, pcls = physical(L, sb.kind)
    try:
        e, _ = ep_contrast(L, D, y)
    except LearningPhaseError as err:
        return {"beta": beta, "tol": tol, "refused": err.phase}
    return {"beta": beta, "tol": tol, "refused": None, "ep_vs_ift": compare(e[idx], ce[idx], pcls)}


def _cell(a): return cell(*a)
def _valid(a): return cell(*a, valid_only=True)
def _asym(a): return asym_cell(*a)
def _real(a): return real_sandbox(*a)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--part", default="small,asym,real", help="valid first (M1-11), then small,asym,real")
    a = ap.parse_args(); out = OUT / run_commit(); parts = a.part.split(",")
    with ProcessPoolExecutor(a.jobs) as ex:
        if "valid" in parts:  # amendment M1-11: IFT vs FD alone, committed before any EP-vs-IFT comparison
            for r in ex.map(_valid, list(itertools.product((0, 1, 2), LEAKS, BUDGETS, AMPS, NUDGES))):
                print(f"s{r['seed']} leak{r['leak']} b{r['budget']} amp{r['amp']} {r['nudge']}: ift~fd all {r['ift_vs_fd']['all']['cos']:.5f} "
                      f"min-class {min(v['cos'] for k, v in r['ift_vs_fd'].items() if k != 'all'):.5f} sat {r['sat_frac']:.2f} {r['wall_s']:.0f}s", flush=True)
        if "small" in parts:
            jobs = list(itertools.product((0, 1, 2), LEAKS, BUDGETS, AMPS, NUDGES))
            for r in ex.map(_cell, jobs):
                prod = next(g for g in r["grid"] if g["beta"] == 0.04 and g["tol"] == 3e-3)
                print(f"s{r['seed']} leak{r['leak']} b{r['budget']} amp{r['amp']} {r['nudge']}: ift~fd {r['ift_vs_fd']['all']['cos']:.4f} "
                      f"prod ep~ift {prod.get('ep_vs_ift', {}).get('all', {}).get('cos', float('nan')):.4f} {r['wall_s']:.0f}s", flush=True)
        if "asym" in parts:
            res = list(ex.map(_asym, itertools.product((0, 1, 2), EPS, AMPS, NUDGES)))
            write_receipt(out, "asym", {"cells": res})
        if "real" in parts:
            res = list(ex.map(_real, itertools.product(BETAS, TOLS)))
            write_receipt(out, "real_sandbox", {"cells": res, "subject": 1, "rows": "train[:16], sub-window 5"}, subject=1)


if __name__ == "__main__":
    main()
