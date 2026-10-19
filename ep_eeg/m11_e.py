"""M11 step E (reports/M11.md §8): planted-truth positive controls.

E1 (Q1, closed form): synthetic IV-2a-format EEG with planted class-specific ERD sources, run through the
   frozen eval-layer preprocessing; truth oracle vs FBCSP / TS+LR / MDM / Track-2 sLDA / ceilings (B's
   procedure) on input_track2 and random_lake, under narrow vs broad volume conduction.
   Generator: 22 EEG leads at their standard_1005 positions; task sources at C4 (left hand), C3 (right hand),
   Cz (feet) and C5+C6 (tongue, both modulated), 8 background sources at random lead positions (jittered 1 cm);
   each source = 10 Hz (+-1 Hz per subject) and 20 Hz narrow-band noise (2 Hz wide) + 1/f noise; class c scales
   source c's oscillations by (1 - e) from cue+0.5 s to cue+4 s; Gaussian-blob forward model on 3-D lead
   positions, sigma_mix 1 cm (narrow) or 5 cm (broad); 1/f sensor noise; 3 EOG channels (blinks + slow drift)
   leaking into frontal leads. 6 runs x 48 trials (8 s each, cue at +2 s), 12 per class per run.
   e calibrated on separate synthetic subjects 101-109 (inner split runs 0-2 -> 3, narrow mixing):
   the value in {0.1, 0.2, 0.3, 0.5} whose truth-oracle mean kappa is closest to 0.6 within [0.5, 0.7]
   (closest to 0.6 overall if none falls inside).
E2 (Q3, static NeuralGraph): planted teacher in the recip_lat readout structure on G1's F_lake features;
   students trained by EP and by IFT (D2 hyper-parameters), agreement with the teacher and planted-edge AUC.
E3 (Q2, closed form): metric positive control; see ep_eeg/m11_c.py (shares C's data layer and metrics).
"""
from __future__ import annotations
import argparse, itertools, json, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.metrics import cohen_kappa_score
from .data import ROOT
from .m11_common import write_receipt, git, M11_OUT

OUT = M11_OUT / "E"
FS, TRIAL_S, N_RUN, N_TRIAL = 250, 8.0, 6, 48
EGRID = (0.1, 0.2, 0.3, 0.5)
SIGMA = {"narrow": 0.01, "broad": 0.05}


def pink(rng, n, size):
    f = np.fft.rfftfreq(size, 1 / FS); spec = (rng.normal(size=(n, len(f))) + 1j * rng.normal(size=(n, len(f)))) / np.maximum(f, 1.0) ** 0.5
    x = np.fft.irfft(spec, size, axis=1); return x / x.std(1, keepdims=True)


def narrow(rng, n, size, f0):
    sos = butter(4, [f0 - 1, f0 + 1], btype="band", fs=FS, output="sos")
    x = sosfilt(sos, rng.normal(size=(n, size + 500)), axis=1)[:, 500:]; return x / x.std(1, keepdims=True)


def synth_subject(subject, e, mix, with_truth=False):
    from .sandbox import lead_xy
    rng = np.random.default_rng(subject); pos = lead_xy()
    from .sandbox import LEADS as names
    task_pos = [pos[names.index(c)] for c in ("C4", "C3", "Cz", "C5", "C6")]; task_cls = [0, 1, 2, 3, 3]
    bg_pos = [pos[i] + rng.normal(0, 0.01, 3) for i in rng.choice(22, 8, replace=False)]
    P = np.array(task_pos + bg_pos); nsrc = len(P)
    G = np.exp(-np.linalg.norm(pos[:, None] - P[None], axis=-1) ** 2 / (2 * SIGMA[mix] ** 2))  # (22, nsrc)
    G /= G.max(0, keepdims=True)
    fmu = 10 + rng.uniform(-1, 1); fbe = 20 + rng.uniform(-1, 1)
    eog_w = np.exp(-np.linalg.norm(pos - pos[names.index("Fz")], axis=1) / 0.05) * 0.5
    runs, truths = [], []
    n = int(N_TRIAL * TRIAL_S * FS)
    for r in range(N_RUN):
        lab = rng.permutation(np.repeat(np.arange(4), N_TRIAL // 4)); on = (np.arange(N_TRIAL) * TRIAL_S * FS).astype(int)
        env = np.ones((nsrc, n))
        for o, c in zip(on, lab):
            a, b = o + int(2.5 * FS), o + int(6.0 * FS)
            for k, kc in enumerate(task_cls):
                if kc == c: env[k, a:b] = 1 - e
        osc = env * (narrow(rng, nsrc, n, fmu) + 0.7 * narrow(rng, nsrc, n, fbe))
        S = osc + pink(rng, nsrc, n)
        blinks = np.zeros((1, n)); blinks[0, rng.choice(n, n // (4 * FS), replace=False)] = 1
        blinks = np.convolve(blinks[0], np.hanning(100), "same")[None] * 5
        drift = np.cumsum(rng.normal(size=(1, n)), 1) / np.sqrt(n) * 3
        eog = np.vstack([blinks + drift, 0.6 * blinks + drift + 0.3 * rng.normal(size=(1, n)), 0.3 * blinks - drift])
        eeg = G @ S + 0.5 * pink(rng, 22, n) + eog_w[:, None] * eog[:1]
        runs.append((eeg * 10, eog * 10, on, lab))  # ~10 uV scale (irrelevant after z-scoring)
        if with_truth: truths.append((osc[:5], on, lab))
    return runs, truths


def to_d(runs):
    from eval import preprocess as P
    pp = P.fit_preproc(runs[:4]); out = {}
    for name, rr in (("train", runs[:4]), ("val", runs[4:])):
        seg, y = P.epochs(rr, pp); sw = P.subwindows(seg); out[name] = {"seg": seg, "y": y, "track2": P.track2(sw)}
    sc = P.FeatureScaler().fit(out["train"]["track2"])
    for name in out: out[name]["track2_x"] = sc.transform(out[name]["track2"])
    return out


def truth_oracle(truths, train_runs, val_runs):
    """Log band power of the noise-free task-source oscillations in the MI window -> shrinkage LDA."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    def feats(rs):
        F, Y = [], []
        for osc, on, lab in (truths[i] for i in rs):
            for o, c in zip(on, lab): F.append(np.log(osc[:, o + int(2.5 * FS):o + int(5.5 * FS)].var(1))); Y.append(c)
        return np.array(F), np.array(Y)
    Ftr, ytr = feats(train_runs); Fva, yva = feats(val_runs)
    return float(cohen_kappa_score(yva, LDA(solver="lsqr", shrinkage="auto").fit(Ftr, ytr).predict(Fva)))


def calibrate():
    res = {}
    for e in EGRID:
        ks = []
        for s in range(101, 110):
            _, tr = synth_subject(s, e, "narrow", with_truth=True); ks.append(truth_oracle(tr, (0, 1, 2), (3,)))
        res[e] = float(np.mean(ks))
    inside = [e for e in EGRID if 0.5 <= res[e] <= 0.7]
    pick = min(inside or EGRID, key=lambda e: abs(res[e] - 0.6))
    return pick, res


def e1_one(subject, mix, e, seed):
    from .baselines import fbcsp, lda
    from .m11_a import riemann
    from .m11_b import ceilings, fmaps_static, static_states, M6
    from .sandbox import Sandbox, SandboxConfig
    t0 = time.time(); runs, truths = synth_subject(subject, e, mix, with_truth=True); d = to_d(runs)
    ytr, yva = d["train"]["y"], d["val"]["y"]; K = lambda p: float(cohen_kappa_score(yva, p))
    out = {"truth_oracle": truth_oracle(truths, range(4), (4, 5))}
    p, _, _ = fbcsp(d["train"], d["val"], seed); out["fbcsp"] = K(p[:, 0].argmax(1))
    out["tslr"] = K(riemann(d["train"]["seg"], ytr, d["val"]["seg"], "tslr")); out["mdm"] = K(riemann(d["train"]["seg"], ytr, d["val"]["seg"], "mdm"))
    from eval.scoring import aggregate, fallback_class
    p, okl, _ = lda(d["train"], d["val"], seed); out["lda_track2"] = K(aggregate(p, okl, fallback_class(ytr))[0])
    ok = lambda X: np.ones(X.shape[:2], bool)
    F = dict(zip(("mean", "cat"), zip(fmaps_static(d["train"]["track2_x"], ok(d["train"]["track2_x"])), fmaps_static(d["val"]["track2_x"], ok(d["val"]["track2_x"])))))
    ct = ceilings(F, ytr, yva, seed); out["input_track2_cv"], out["input_track2_or"] = ct["kappa_cv"], ct["kappa_or"]
    sb = Sandbox(SandboxConfig(**M6), seed)
    Str, _, oktr = static_states(sb, d["train"]["track2_x"]); Sva, _, okva = static_states(sb, d["val"]["track2_x"])
    mtr, ctr = fmaps_static(Str, oktr); mva, cva = fmaps_static(Sva, okva, ref=mtr.mean(0))
    cl = ceilings({"mean": (mtr, mva), "cat": (ctr, cva)}, ytr, yva, seed)
    out.update(random_lake_cv=cl["kappa_cv"], random_lake_or=cl["kappa_or"], lake_settle=float(np.r_[oktr.ravel(), okva.ravel()].mean()),
               lake_rejected=int((~oktr).sum() + (~okva).sum()))
    rec = {"stage": "M11", "step": "E1", "arm": f"e1_{mix}", "subject": subject, "seed": seed, "e": e, "mix": mix, **out,
           "settle_rate": out["lake_settle"], "rejected": out["lake_rejected"], "wall_s": time.time() - t0}
    write_receipt(OUT / git("rev-parse", "--short", "HEAD") / f"e1_{mix}", f"S{subject:02d}_s{seed}", rec)
    return rec


# ---------- E2: planted teacher ----------
def planted_teacher(d, seed):
    from .g1_readout import Readout
    rng = np.random.default_rng(10_000 + seed); t = Readout(d, "recip_lat", "ce01", 0.05, seed)
    L = t.learner; br = L.brain; pre, post = t.pre, t.post; eff = np.zeros(len(pre)); out = set(t.out)
    io = np.flatnonzero((pre < d) & np.isin(post, list(out)))           # feature -> motor (pairs share one weight)
    per = int(round(0.1 * d))                                            # 10% density, the same count per motor
    planted = np.concatenate([rng.choice(io[post[io] == o], per, replace=False) for o in sorted(out)])
    val = rng.choice([-1, 1], len(planted)) * 0.1                        # motor row mass 9 * 0.1 + 3 * 0.2 = 1.5 < 1.8
    key = {(int(p), int(q)): i for i, (p, q) in enumerate(zip(pre, post))}
    for i, v in zip(planted, val): eff[i] = v; eff[key[(int(post[i]), int(pre[i]))]] = v
    lat = [i for i in range(len(pre)) if pre[i] in out and post[i] in out]; eff[lat] = -0.2  # lateral inhibition
    L.brain = br.with_parameters(efficacy=eff, bias=np.asarray(br.bias)); assert t.cert()["certified"], t.cert()
    return t, set(int(i) for i in planted) | {key[(int(post[i]), int(pre[i]))] for i in planted}, eff


def e2_one(subject, trainer, seed):
    from .g1_readout import features, prep, Readout, evaluate
    from eval.scoring import fallback_class
    from sklearn.metrics import roc_auc_score
    t0 = time.time(); F = features(subject); Ztr, Zva = prep(F["lake_train"], F["lake_val"]); d = Ztr.shape[-1]
    teacher, planted, eff_t = planted_teacher(d, seed)
    def labels(Z):
        n, k, _ = Z.shape; a, ok = teacher.scores(Z.reshape(-1, d)); return a.reshape(n, k, 4).mean(1).argmax(1)
    ytr, yva = labels(Ztr), labels(Zva)
    net = Readout(d, "recip_lat", "ce01", 0.05, seed, asym_eps=1.0 if trainer == "ep_asym" else 0.0)
    eff0 = np.asarray(net.brain.efficacy).copy()
    st = net.fit(Ztr.reshape(-1, d), np.repeat(ytr, Ztr.shape[1]), 100, np.random.default_rng(seed + 1000),
                 trainer="ep" if trainer == "ep_asym" else trainer)
    k, rej, settle = evaluate(net, Zva, yva, fallback_class(ytr), "sub") if st["cert_fail"] is None else (0.0, len(yva), 0.0)
    pred = np.asarray(evaluate.last_pred); eff = np.asarray(net.brain.efficacy)
    io = np.flatnonzero((net.pre < d) & (net.post >= d)); lab = np.isin(io, list(planted)).astype(int)
    auc = float(roc_auc_score(lab, np.abs(eff[io] - eff0[io])))
    lat = np.flatnonzero((net.pre >= d) & (net.post >= d))
    rec = {"stage": "M11", "step": "E2", "arm": f"e2_{trainer}", "subject": subject, "seed": seed, "agreement": float((pred == yva).mean()),
           "kappa_vs_teacher": k, "planted_auc": auc, "lateral_learned_mean": float(eff[lat].mean()), "lateral_planted": -0.2,
           "teacher_class_counts": np.bincount(ytr, minlength=4).tolist(), "accept_rate": st["accepted"] / max(1, st["attempted"]),
           "cert_fail": st["cert_fail"], "settle_rate": settle, "rejected": rej, "pred": pred, "label": yva, "wall_s": time.time() - t0}
    write_receipt(OUT / git("rev-parse", "--short", "HEAD") / f"e2_{trainer}", f"A{subject:02d}_s{seed}", rec, subject=subject)
    return rec


def _e1(a): return e1_one(*a)
def _e2(a): return e2_one(*a)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True, choices=("calibrate", "e1", "e2"))
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--e", type=float, default=None); a = ap.parse_args()
    out = OUT / git("rev-parse", "--short", "HEAD")
    if a.part == "calibrate":
        pick, res = calibrate(); write_receipt(out, "e1_calibration", {"pick": pick, "truth_oracle_inner": res}); print(pick, res); return
    with ProcessPoolExecutor(a.jobs) as ex:
        if a.part == "e1":
            for r in ex.map(_e1, itertools.product(range(1, 10), ("narrow", "broad"), (a.e,), (0, 1, 2))):
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("env",)}, flush=True)
        else:
            for r in ex.map(_e2, itertools.product(range(1, 10), ("ep", "ift", "ep_asym"), (0, 1, 2))):
                print(r["arm"], r["subject"], r["seed"], round(r["agreement"], 3), round(r["planted_auc"], 3), r["cert_fail"], flush=True)


if __name__ == "__main__":
    main()
