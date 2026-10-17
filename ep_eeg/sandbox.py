"""EP sandbox adapter (ADR-0001..0003): Connectome.from_synapses + NeuralGraph + Learner.

Neurons: [inputs (d) | columns (22*k) | outlet (4)].
Inputs project one-way to the columns of their own lead; columns are reciprocal
(group-specific edge set / init); columns <-> outlet reciprocal.
"""
from __future__ import annotations
import time, warnings
from dataclasses import dataclass, asdict
import numpy as np
import cadence as cd
from cadence.learning import Learner, LearnerConfig, learning_neuron_model, LearningPhaseError
from cadence.certificate import certificate
from eval.t_only_bnci import CHANNEL_NAMES

LEADS = CHANNEL_NAMES[:22]
# ADR-0007 (proposed): five-region map for group 3
REGIONS = {
    "frontal": ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4"],
    "left": ["C5", "C3", "C1", "CP3", "CP1"],
    "mid": ["Cz", "CPz"],
    "right": ["C2", "C4", "C6", "CP2", "CP4"],
    "parietal": ["P1", "Pz", "P2", "POz"],
}
REGION_LINKS = {("frontal", "left"), ("frontal", "mid"), ("frontal", "right"),
                ("left", "mid"), ("right", "mid"),
                ("parietal", "left"), ("parietal", "mid"), ("parietal", "right")}


@dataclass
class SandboxConfig:
    track: str = "track2"
    group: str = "g1"          # g1 uniform, g2 distance-init, g3 five-region, sparse (g3-degree matched random)
    k: int = 4
    sigma_in: float = 0.1
    sigma_col: float = 0.01
    sigma_out: float = 0.01
    input_bias: float = 0.0    # fixed (non-plastic) bias on input neurons
    train_internal: bool = True  # False: column<->column frozen (fixed random sandbox)
    project_plastic_only: bool = False  # True: rowmass projection rescales only plastic synapses (strict freeze)
    epochs: int = 10
    batch: int = 32
    shuffle_labels: bool = False
    rowmass_project: float = 0.0  # >0: after every accepted update, symmetric per-neuron projection to row_mass <= this/L
    beta: float = 0.04
    eta: float = 0.1
    eta_bias: float = 0.01
    free_steps: int = 1024
    nudged_steps: int = 1024
    tolerance: float = 3e-3
    temperature: float = 0.2
    decay: float = 0.0
    momentum: float = 0.0
    normalize: float = 0.0
    scale_cap: float = 3.0
    damping: int = 3
    leak: float = 0.1          # learning_neuron_model leak (0.1 default; 1.0 = symmetric tanh(v/2))
    slots: int = 1             # Learner slots (M5 outlet extension; 1 = single action slot)
    use_warm: bool = False     # M7: per trial, sub-windows in time order, warm = previous free state
    n_classes: int = 4         # M8: 5 = four MI classes + inter-trial rest


def lead_xy():
    import mne
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    return np.array([pos[c] for c in LEADS])


def degree_matched_shuffle(edges, n, rng, swaps_per_edge=10):
    E = [tuple(e) for e in edges]; S = set(E)
    for _ in range(swaps_per_edge * len(E)):
        i, j = rng.integers(len(E), size=2)
        (a, b), (c, d) = E[i], E[j]
        if rng.random() < 0.5: c, d = d, c
        n1, n2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        if a == d or c == b or n1 in S or n2 in S or n1 == n2: continue
        S -= {E[i], E[j]}; S |= {n1, n2}; E[i], E[j] = n1, n2
    return E


def col_edges(cfg, rng):
    """Unordered column pairs (a<b) and an init multiplier per pair."""
    k = cfg.k; nc = 22 * k; lead = np.repeat(np.arange(22), k)
    pairs = [(a, b) for a in range(nc) for b in range(a + 1, nc)]
    mult = np.ones(len(pairs))
    if cfg.group in ("g3", "sparse"):
        reg = {c: r for r, cs in REGIONS.items() for c in cs}
        def allowed(a, b):
            ra, rb = reg[LEADS[lead[a]]], reg[LEADS[lead[b]]]
            return ra == rb or (ra, rb) in REGION_LINKS or (rb, ra) in REGION_LINKS
        g3 = [p for p in pairs if allowed(*p)]
        pairs = g3 if cfg.group == "g3" else degree_matched_shuffle(g3, nc, rng)
        mult = np.ones(len(pairs))
    elif cfg.group == "g2":
        xy = lead_xy(); dist = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        lam = np.median(dist[np.triu_indices(22, 1)])
        mult = np.array([np.exp(-dist[lead[a], lead[b]] ** 2 / (2 * lam ** 2)) for a, b in pairs])
        mult = mult / np.mean(np.abs(mult))  # same mean init magnitude as g1
    return pairs, mult


class Sandbox:
    def __init__(self, cfg: SandboxConfig, seed: int):
        self.cfg = cfg; rng = np.random.default_rng(seed)
        from eval.preprocess import channel_of_feature
        lead_of_feat = channel_of_feature(cfg.track); d = len(lead_of_feat)
        k = cfg.k; nc = 22 * k
        self.n_classes = int(cfg.n_classes)
        # slots>1: parallel 4-way readouts (cadence equal slots); total outlets = 4 * slots
        self.n_slots = int(cfg.slots)
        n_out = self.n_classes * self.n_slots
        n = d + nc + n_out
        self.d, self.nc, self.n, self.n_out = d, nc, n, n_out
        col0, out0 = d, d + nc
        self.col0, self.out0 = col0, out0
        syn = {}; kind = {}
        for f in range(d):
            for j in range(k):
                q = col0 + lead_of_feat[f] * k + j
                syn[(f, q)] = rng.normal(0, cfg.sigma_in); kind[(f, q)] = "in"
        pairs, mult = col_edges(cfg, rng)
        for (a, b), m in zip(pairs, mult):
            w = rng.normal(0, cfg.sigma_col) * m
            syn[(col0 + a, col0 + b)] = syn[(col0 + b, col0 + a)] = w
            kind[(col0 + a, col0 + b)] = kind[(col0 + b, col0 + a)] = "col"
        for a in range(nc):
            for o in range(n_out):
                w = rng.normal(0, cfg.sigma_out)
                syn[(col0 + a, out0 + o)] = syn[(out0 + o, col0 + a)] = w
                kind[(col0 + a, out0 + o)] = kind[(out0 + o, col0 + a)] = "out"
        keys = list(syn)
        pre = np.array([p for p, _ in keys]); post = np.array([q for _, q in keys])
        self.conn = cd.Connectome.from_synapses(
            n, pre=pre, post=post,
            populations={"inp": tuple(range(d)), "col": tuple(range(col0, out0)), "out": tuple(range(out0, n))})
        eff = np.array([syn[(p, q)] for p, q in zip(self.conn.pre, self.conn.post)])
        self.kind = np.array([kind[(p, q)] for p, q in zip(self.conn.pre, self.conn.post)])
        bias = np.zeros(n); bias[:d] = cfg.input_bias
        self.brain = cd.NeuralGraph(self.conn, learning_neuron_model(leak=cfg.leak), efficacy=eff, bias=bias)
        self.outputs = list(range(out0, n))
        lc = LearnerConfig(beta=cfg.beta, eta=cfg.eta, eta_bias=cfg.eta_bias, centered=True,
                           free_steps=cfg.free_steps, nudged_steps=cfg.nudged_steps, tolerance=cfg.tolerance,
                           nudge="cross_entropy", temperature=cfg.temperature, normalize=cfg.normalize,
                           normalize_floor=1e-3, momentum=cfg.momentum, decay=cfg.decay,
                           scale_cap=cfg.scale_cap, qualified=True, damping=cfg.damping, surprise=False)
        plastic_syn = np.ones(len(eff), bool)
        plastic_n = np.zeros(n, bool); plastic_n[col0:] = True
        if not cfg.train_internal:
            plastic_syn = self.kind == "out"
            plastic_n[col0:out0] = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.learner = Learner(self.brain, self.outputs, config=lc,
                                   plastic_synapses=plastic_syn, plastic_neurons=plastic_n,
                                   slots=cfg.slots)
        n_in = int((plastic_syn & (self.kind == "in")).sum())
        n_rec = int((plastic_syn & (self.kind != "in")).sum()) // 2
        self.n_params = n_in + n_rec + int(plastic_n.sum())
        self.rng = rng

    def drive(self, X):
        D = np.zeros((len(X), self.n)); D[:, :self.d] = X; return D

    def cert(self):
        c = certificate(self.learner.brain)
        return {"certified": bool(c.certified), "row_mass": float(c.row_mass), "rate": float(c.rate)}

    def project_rowmass(self):
        """Symmetric per-neuron projection: any neuron whose incoming |W| sum exceeds
        rowmass_project/L has all its incident synapses (both directions of each
        reciprocal pair, plus one-way inputs) scaled down. Keeps W symmetric."""
        br = self.learner.brain; lim = self.cfg.rowmass_project / 0.5
        eff = np.asarray(br.efficacy).copy(); post, pre = self.conn.post, self.conn.pre
        gscale = np.asarray(br.weights) / np.where(eff == 0, 1, eff)  # effective/efficacy factor
        smin = 1.0
        for _ in range(20):
            inc = np.bincount(post, weights=np.abs(eff * gscale), minlength=self.n)
            over = inc > lim * (1 - 1e-9)
            if not over.any(): break
            f = np.ones(self.n); f[over] = lim / inc[over] * (1 - 1e-6)
            fac = np.minimum(f[post], np.where(self.kind == "in", 1.0, f[pre]))
            if self.cfg.project_plastic_only: fac = np.where(self.learner.plastic_synapses, fac, 1.0)
            eff = eff * fac; smin = min(smin, float(fac.min()))
        if smin < 1.0:
            br.efficacy = eff
        return smin

    def fit(self, X, y, log):
        if self.cfg.use_warm: return self.fit_warm(X, y, log)
        cfg = self.cfg
        if cfg.shuffle_labels: y = self.rng.permutation(y)
        Xs = X.reshape(-1, X.shape[-1]); ys = np.repeat(y, X.shape[1])
        if self.n_slots > 1:
            # each slot is a 4-way classifier; same class label in every slot
            ys = np.column_stack([ys] * self.n_slots)
        D = self.drive(Xs)
        stats = {"attempted_batches": 0, "refused_batches": 0, "attempted_rows": 0, "accepted_rows": 0,
                 "row_sweeps": 0.0, "epochs": [], "refusal_phases": {}}
        c0 = self.cert(); stats["cert_init"] = c0
        if not c0["certified"]:
            stats["failed"] = "certificate_init"; return stats
        for ep in range(cfg.epochs):
            t = time.time(); idx = self.rng.permutation(len(D))
            for b in np.array_split(idx, max(1, len(idx) // cfg.batch)):
                stats["attempted_batches"] += 1; stats["attempted_rows"] += len(b)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _, rep = self.learner.step(D[b], ys[b])
                    stats["accepted_rows"] += len(b); stats["row_sweeps"] += rep["total_row_sweeps"]
                    if cfg.rowmass_project > 0: stats["min_projection"] = min(stats.get("min_projection", 1.0), self.project_rowmass())
                except LearningPhaseError as e:
                    stats["refused_batches"] += 1
                    stats["refusal_phases"][e.phase] = stats["refusal_phases"].get(e.phase, 0) + 1
            s = stats.get("min_projection", 1.0)
            c = self.cert(); c.update(epoch=ep, wall=time.time() - t, projection=s)
            stats["epochs"].append(c)
            log(f"epoch {ep} cert={c['certified']} row_mass={c['row_mass']:.3f} refused={stats['refused_batches']}/{stats['attempted_batches']}")
            if not c["certified"]:
                stats["failed"] = f"certificate_epoch_{ep}"; break
        return stats

    def fit_warm(self, X, y, log):
        """M7: batches of trials; within a trial time steps (sub-windows or raw chunks) t=0..T-1 in order, each
        Learner.step warm-started from the previous sub-window's free state; warm=None
        at trial start (reset between trials). A refused step resets warm to None.
        192 trials / batch 32 = 6 batches x 11 sub-windows = 66 steps/epoch (as M1)."""
        cfg = self.cfg
        if cfg.shuffle_labels: y = self.rng.permutation(y)
        n, T, d = X.shape
        stats = {"attempted_batches": 0, "refused_batches": 0, "attempted_rows": 0, "accepted_rows": 0,
                 "row_sweeps": 0.0, "epochs": [], "refusal_phases": {}, "warm_steps": 0}
        c0 = self.cert(); stats["cert_init"] = c0
        if not c0["certified"]:
            stats["failed"] = "certificate_init"; return stats
        for ep in range(cfg.epochs):
            t0 = time.time(); idx = self.rng.permutation(n)
            for b in np.array_split(idx, max(1, n // cfg.batch)):
                warm = None  # reset between trials
                yb = y[b] if self.n_slots == 1 else np.column_stack([y[b]] * self.n_slots)
                for t in range(T):
                    stats["attempted_batches"] += 1; stats["attempted_rows"] += len(b)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            st, rep = self.learner.step(self.drive(X[b, t]), yb, warm=warm)
                        stats["warm_steps"] += int(warm is not None)
                        warm = st.free  # carry activity to next sub-window
                        stats["accepted_rows"] += len(b); stats["row_sweeps"] += rep["total_row_sweeps"]
                        if cfg.rowmass_project > 0: stats["min_projection"] = min(stats.get("min_projection", 1.0), self.project_rowmass())
                    except LearningPhaseError as e:
                        warm = None
                        stats["refused_batches"] += 1
                        stats["refusal_phases"][e.phase] = stats["refusal_phases"].get(e.phase, 0) + 1
            c = self.cert(); c.update(epoch=ep, wall=time.time() - t0, projection=stats.get("min_projection", 1.0))
            stats["epochs"].append(c)
            log(f"epoch {ep} cert={c['certified']} row_mass={c['row_mass']:.3f} refused={stats['refused_batches']}/{stats['attempted_batches']}")
            if not c["certified"]:
                stats["failed"] = f"certificate_epoch_{ep}"; break
        return stats

    def predict_warm(self, X):
        """M7 inference: same warm chaining per trial (all trials as one batch)."""
        n, T, d = X.shape; acts, oks = [], []; state = None
        for t in range(T):
            eq = self.learner.brain.equilibrate(self.drive(X[:, t]), budget=self.cfg.free_steps,
                                                tolerance=self.cfg.tolerance, damping=self.cfg.damping, state=state)
            a = np.asarray(eq.state.activation)[:, self.outputs]
            ok = np.asarray(eq.converged) & np.all(np.isfinite(a), 1)
            if self.n_slots > 1: a = a.reshape(-1, self.n_slots, self.n_classes).mean(1)
            acts.append(a); oks.append(ok)
            state = eq.state if np.all(np.isfinite(np.asarray(eq.state.v))) else None
        return np.stack(acts, 1), np.stack(oks, 1)

    def free_states(self, X):
        n, k, d = X.shape
        eq = self.learner.brain.equilibrate(self.drive(X.reshape(-1, d)), budget=self.cfg.free_steps,
                                            tolerance=self.cfg.tolerance, damping=self.cfg.damping)
        return eq

    def predict(self, X):
        if self.cfg.use_warm: return self.predict_warm(X)
        n, k, d = X.shape
        eq = self.free_states(X)
        act = np.asarray(eq.state.activation)[:, self.outputs]
        ok = np.asarray(eq.converged) & np.all(np.isfinite(act), 1)
        if self.n_slots > 1:
            # (rows, slots, classes) -> mean over slots
            act = act.reshape(-1, self.n_slots, self.n_classes).mean(1)
        return act.reshape(n, k, self.n_classes), ok.reshape(n, k)


def train_sandbox(train, seed, **opts):
    cfg = SandboxConfig(**opts); sb = Sandbox(cfg, seed); logs = []
    stats = sb.fit(train[cfg.track + "_x"], train["y"], logs.append)
    return sb, stats, logs


def run_arm(train, val, seed, **opts):
    t = time.time()
    sb, stats, logs = train_sandbox(train, seed, **opts)
    cfg = sb.cfg
    extra = {"cfg": asdict(cfg), "params": sb.n_params, "train": stats, "log": logs, "train_wall": time.time() - t}
    n = len(val["y"])
    if "failed" in stats:  # failed certificate: all sub-windows refused (counted; fallback class)
        T = val[cfg.track + "_x"].shape[1]
        return np.zeros((n, T, 4)), np.zeros((n, T), bool), extra
    act, ok = sb.predict(val[cfg.track + "_x"])
    _, tr_ok = sb.predict(train[cfg.track + "_x"])
    extra["train_settle_rate"] = float(tr_ok.mean())
    extra["learn_accept_rate"] = stats["accepted_rows"] / max(1, stats["attempted_rows"])
    return act, ok, extra


def ridge_arm(train, val, seed, **opts):
    """C0 baseline: untrained (frozen random) sandbox; free-phase column states averaged over the 11
    sub-windows (qualified ones) -> standardised RidgeClassifierCV. Returned per sub-window as the
    linear decision function, so the frozen evaluation mean over sub-windows equals ridge on mean states."""
    from sklearn.linear_model import RidgeClassifierCV
    t = time.time(); cfg = SandboxConfig(**opts); sb = Sandbox(cfg, seed)
    def states(X):
        n, k, d = X.shape; eq = sb.free_states(X)
        S = np.asarray(eq.state.activation)[:, sb.col0:sb.out0].reshape(n, k, -1)
        ok = (np.asarray(eq.converged) & np.all(np.isfinite(S.reshape(n * k, -1)), 1)).reshape(n, k)
        return S, ok
    Str, oktr = states(train[cfg.track + "_x"]); Sva, okva = states(val[cfg.track + "_x"])
    w = oktr[..., None].astype(float); F = (Str * w).sum(1) / np.maximum(w.sum(1), 1)
    mu, sd = F.mean(0), F.std(0) + 1e-9
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 13)).fit((F - mu) / sd, train["y"])
    act = clf.decision_function(((Sva - mu) / sd).reshape(-1, F.shape[1])).reshape(*Sva.shape[:2], -1)
    extra = {"cfg": asdict(cfg), "ridge_alpha": float(clf.alpha_), "train_settle_rate": float(oktr.mean()),
             "cert": sb.cert(), "train_wall": time.time() - t}
    return act, okva, extra


PRESETS = {
    "ep_track2": {"track": "track2"},
    "ep_track1": {"track": "track1"},
    "ep_shuffled": {"track": "track2", "shuffle_labels": True},
    "ep_g1": {"group": "g1"}, "ep_g2": {"group": "g2"}, "ep_g3": {"group": "g3"},
    "ep_sparse": {"group": "sparse"}, "ep_fixed": {"train_internal": False},
    "ep_fixed_strict": {"train_internal": False, "project_plastic_only": True},  # C0: truly frozen sandbox
    "ep_slots4": {"slots": 4},
    "ep_raw_warm": {"track": "raw", "use_warm": True},   # M7
    "ep_raw_cold": {"track": "raw"},                     # superseded M7 draft
    "ep_rawseq_warm": {"track": "rawseq", "use_warm": True},  # M7 final: 30 raw steps, warm chained
    "ep_rawseq_cold": {"track": "rawseq"},                    # M7 ablation: same input, cold start
}


def arm(name):
    if name == "ep_fixed_ridge": return ridge_arm
    base = PRESETS[name]
    def f(train, val, seed, **opts):
        return run_arm(train, val, seed, **{**base, **opts})
    return f
