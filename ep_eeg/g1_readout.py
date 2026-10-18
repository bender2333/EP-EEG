"""G1 (reports/M10-gates.md, pre-registered): can certified EP learn a fixed-feature 4-class readout to
within 0.03 kappa of ridge on the same features? Pure optimisation; no lake training.

Features (fixed): "lake" = column activations of the M6 random linear lake (seed 0), 88-d (352 readout
parameters); "raw" = Track-2, 66-d. Per-dim z-score on training rows; the network sees rho(z) = tanh(z/2)
(leak 1.0). Reference: RidgeClassifierCV on the trial mean of the same rho(z), standardised.
Row modes: "sub" trains on the 11 sub-window rows per trial (ADR-0008 mean of motor activations at
eval); "trial" trains on one row per trial whose drive is 2 artanh(m), m = trial mean of rho(z), so the
source activation equals the reference ridge feature exactly (updates matched: 1100 epochs).
Structures:
  recip   features drive d free relay neurons; relay <-> motor reciprocal pairs (sandbox readout topology)
  untied  same edges, Learner(reciprocal=False): forward and feedback efficacies are separate
  oneway  features are source neurons projecting one-way to motor (no free->free edges; ADR-0012
          certificate holds trivially; EP reduces to the delta rule)
  lateral oneway + motor <-> motor reciprocal plastic pairs (init 0): motor as a small cortex
Certificate: row mass over edges whose presynaptic neuron is not a source (ADR-0012), checked at init
and after every epoch; projection to rowmass 0.9/L over the same edges (sources are never scaled).
"""
from __future__ import annotations
import argparse, itertools, json, subprocess, time, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import cadence as cd
from cadence.certificate import ep_structure, lipschitz_constant
from cadence.learning import Learner, LearnerConfig, learning_neuron_model, LearningPhaseError
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import cohen_kappa_score
from eval.scoring import aggregate, fallback_class
from .data import load_subject, ROOT

M6 = dict(leak=1.0, rowmass_project=0.9, eta=0.5, temperature=0.01)
NUDGES = {"ce001": ("cross_entropy", 0.01), "ce01": ("cross_entropy", 0.1), "mse": ("quadratic", 0.2)}
STRUCTS = ("recip", "untied", "oneway", "lateral")
FEATS = ("lake", "raw")
GRID = [dict(feat=f, struct=s, nudge=n, eta=e, rows=r)
        for f, s, n, e, r in itertools.product(FEATS, STRUCTS, NUDGES, (0.05, 0.5), ("sub", "trial"))]
FEAT_DIR = ROOT / "receipts" / "G1" / "features"


def features(subject):
    """(train (192,11,d), val (96,11,d), ytr, yva) per feature set, cached."""
    p = FEAT_DIR / f"A{subject:02d}.npz"
    if not p.exists():
        from .sandbox import Sandbox, SandboxConfig
        d = load_subject(subject); sb = Sandbox(SandboxConfig(**M6), 0); out = {}
        for part in ("train", "val"):
            X = d[part]["track2_x"]; n, k, _ = X.shape
            eq = sb.free_states(X); assert np.asarray(eq.converged).all()
            out[f"lake_{part}"] = np.asarray(eq.state.activation)[:, sb.col0:sb.out0].reshape(n, k, -1)
            out[f"raw_{part}"] = X; out[f"y_{part}"] = d[part]["y"]
        FEAT_DIR.mkdir(parents=True, exist_ok=True); np.savez_compressed(p, **out)
    z = np.load(p); return {k: z[k] for k in z.files}


def prep(Ftr, Fva):
    mu, sd = Ftr.reshape(-1, Ftr.shape[-1]).mean(0), Ftr.reshape(-1, Ftr.shape[-1]).std(0) + 1e-9
    return (Ftr - mu) / sd, (Fva - mu) / sd


def rho(z): return np.tanh(z / 2)


def ridge_ref(Ztr, ytr, Zva, yva):
    A, B = rho(Ztr).mean(1), rho(Zva).mean(1); mu, sd = A.mean(0), A.std(0) + 1e-9
    r = RidgeClassifierCV(alphas=np.logspace(-3, 3, 13)).fit((A - mu) / sd, ytr)
    return float(cohen_kappa_score(yva, r.predict((B - mu) / sd)))


class Readout:
    def __init__(self, d, struct, nudge, eta, seed):
        rng = np.random.default_rng(seed); self.d, self.struct = d, struct
        n = d + 4; self.out = list(range(d, n)); syn = {}
        for a in range(d):
            for o in self.out:
                w = rng.normal(0, 0.01); syn[(a, o)] = w
                if struct in ("recip", "untied"): syn[(o, a)] = w
        if struct == "lateral":
            for o, q in itertools.combinations(self.out, 2): syn[(o, q)] = syn[(q, o)] = 0.0
        keys = list(syn); pre = np.array([p for p, _ in keys]); post = np.array([q for _, q in keys])
        self.conn = cd.Connectome.from_synapses(n, pre=pre, post=post)
        self.pre, self.post = self.conn.pre, self.conn.post
        eff = np.array([syn[(p, q)] for p, q in zip(self.pre, self.post)])
        self.sources = list(range(d)) if struct in ("oneway", "lateral") else []
        self.counted = ~np.isin(self.pre, self.sources)
        brain = cd.NeuralGraph(self.conn, learning_neuron_model(leak=1.0), efficacy=eff, bias=np.zeros(n))
        kind, T = NUDGES[nudge]
        self.lc = LearnerConfig(beta=0.04, eta=eta, eta_bias=0.01, centered=True, free_steps=1024, nudged_steps=1024,
                                tolerance=3e-3, nudge=kind, temperature=T, normalize=0.0, normalize_floor=1e-3,
                                momentum=0.0, decay=0.0, scale_cap=3.0, qualified=True, damping=3, surprise=False)
        pn = np.zeros(n, bool); pn[d:] = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.learner = Learner(brain, self.out, config=self.lc, plastic_synapses=np.ones(len(eff), bool),
                                   plastic_neurons=pn, reciprocal=(struct != "untied"))
        self.n_params = self.learner.parameters() if callable(getattr(self.learner, "parameters", None)) else None
        self.n_projections = 0

    @property
    def brain(self): return self.learner.brain

    def cert(self):
        br = self.brain; m = br.neuron_model; w = np.abs(np.asarray(br.weights))
        rho_f = float(np.bincount(self.post, w * self.counted, minlength=self.conn.n).max())
        fixed = ep_structure(br, fixed_inputs=self.sources).fixed_incoming_mass if self.sources else 0.0
        L = lipschitz_constant(m)
        return {"certified": bool(fixed == 0 and L * rho_f < 1 and m.adaptation is None), "row_mass": rho_f,
                "fixed_incoming_mass": float(fixed)}

    def project(self, limit=0.9 / 0.5):
        br = self.brain; eff = np.asarray(br.efficacy).copy(); g = np.asarray(br.weights) / np.where(eff == 0, 1, eff)
        inc = np.bincount(self.post, np.abs(eff * g) * self.counted, minlength=self.conn.n)
        over = inc > limit * (1 - 1e-9)
        if not over.any(): return
        f = np.ones(self.conn.n); f[over] = limit / inc[over] * (1 - 1e-6)
        fac = np.where(self.counted, np.minimum(f[self.post], f[self.pre]), 1.0)
        br.efficacy = eff * fac; self.n_projections += 1

    def drive(self, Z):
        D = np.zeros((len(Z), self.conn.n)); D[:, :self.d] = Z; return D

    def scores(self, Z):
        eq = self.brain.equilibrate(self.drive(Z), budget=1024, tolerance=3e-3, damping=3)
        a = np.asarray(eq.state.activation)[:, self.out]
        return a, np.asarray(eq.converged) & np.isfinite(a).all(1)

    def fit(self, Z, y, epochs, rng, batch=32):
        st = {"attempted": 0, "accepted": 0, "refused": 0, "cert_fail": None, "cert_init": self.cert()}
        if not st["cert_init"]["certified"]: st["cert_fail"] = "init"; return st
        D = self.drive(Z)
        for ep in range(epochs):
            idx = rng.permutation(len(D))
            for b in np.array_split(idx, max(1, len(idx) // batch)):
                st["attempted"] += len(b)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.learner.step(D[b], y[b])
                    st["accepted"] += len(b); self.project()
                except LearningPhaseError:
                    st["refused"] += len(b)
            if not self.cert()["certified"]: st["cert_fail"] = f"epoch_{ep}"; break
        st["cert_final"] = self.cert(); st["n_projections"] = self.n_projections
        return st


def evaluate(net, Zva, yva, fb, rows):
    if rows == "trial":
        a, ok = net.scores(trial_drive(Zva)); a, ok = a[:, None], ok[:, None]
    else:
        n, k, d = Zva.shape; a, ok = net.scores(Zva.reshape(-1, d)); a, ok = a.reshape(n, k, 4), ok.reshape(n, k)
    pred, rej = aggregate(a, ok, fb)
    return float(cohen_kappa_score(yva, pred)), int(rej.sum()), float(ok.mean())


def trial_drive(Z):
    return 2 * np.arctanh(np.clip(rho(Z).mean(1), -0.999, 0.999))


def run(subject, cfg, seed, split):
    """split "inner": train runs 0-2 (144 trials), eval run 3 (48); "final": train runs 0-3, eval runs 4-5."""
    t0 = time.time(); F = features(subject)
    Ftr, Fva, ytr, yva = F[f"{cfg['feat']}_train"], F[f"{cfg['feat']}_val"], F["y_train"], F["y_val"]
    if split == "inner": Ftr, Fva, ytr, yva = Ftr[:144], Ftr[144:], ytr[:144], ytr[144:]
    Ztr, Zva = prep(Ftr, Fva); ref = ridge_ref(Ztr, ytr, Zva, yva)
    net = Readout(Ztr.shape[-1], cfg["struct"], cfg["nudge"], cfg["eta"], seed); rng = np.random.default_rng(seed + 1000)
    if cfg["rows"] == "trial":
        X, y, epochs = trial_drive(Ztr), ytr, 1100
    else:
        X, y, epochs = Ztr.reshape(-1, Ztr.shape[-1]), np.repeat(ytr, Ztr.shape[1]), 100
    st = net.fit(X, y, epochs, rng)
    k, rej, settle = evaluate(net, Zva, yva, fallback_class(ytr), cfg["rows"]) if st["cert_fail"] is None else (0.0, len(yva), 0.0)
    return {"subject": subject, "seed": seed, "split": split, **cfg, "kappa_ep": k, "kappa_ridge": ref, "gap": ref - k,
            "rejected": rej, "settle_val": settle, "accept_rate": st["accepted"] / max(1, st["attempted"]),
            "cert_fail": st["cert_fail"], "row_mass_final": st.get("cert_final", {}).get("row_mass"),
            "n_projections": st.get("n_projections"), "wall_s": time.time() - t0}


def _job(a): return run(*a)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", choices=("select", "final", "smoke"), required=True)
    ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--configs", default=None, help="json list (final)")
    a = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    out = ROOT / "receipts" / "G1" / commit; out.mkdir(parents=True, exist_ok=True)
    for s in range(1, 10): features(s)
    if a.phase == "smoke":
        jobs = [(1, c, 0, "inner") for c in GRID if c["feat"] == "lake" and c["eta"] == 0.5 and c["nudge"] == "ce001"]
    elif a.phase == "select":
        jobs = [(s, c, 0, "inner") for c in GRID for s in range(1, 10)]
    else:
        cfgs = json.loads(a.configs); jobs = [(s, c, seed, "final") for c in cfgs for seed in (0, 1, 2) for s in range(1, 10)]
    with ProcessPoolExecutor(a.jobs) as ex: res = list(ex.map(_job, jobs, chunksize=1))
    (out / f"{a.phase}.json").write_text(json.dumps(res, indent=1, default=float)); print(out / f"{a.phase}.json")
    key = lambda r: (r["feat"], r["struct"], r["nudge"], r["eta"], r["rows"])
    groups = {}
    for r in res: groups.setdefault(key(r), []).append(r)
    rows = sorted(groups.items(), key=lambda kv: np.mean([r["gap"] for r in kv[1]]))
    for k, rs in rows:
        print(k, f"ep {np.mean([r['kappa_ep'] for r in rs]):.3f} ridge {np.mean([r['kappa_ridge'] for r in rs]):.3f} "
                 f"gap {np.mean([r['gap'] for r in rs]):+.3f} acc {np.mean([r['accept_rate'] for r in rs]):.3f} "
                 f"certfail {sum(r['cert_fail'] is not None for r in rs)} wall {np.mean([r['wall_s'] for r in rs]):.0f}s")


if __name__ == "__main__":
    main()
