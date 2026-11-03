"""M11 E2 diagnostic (exploratory, not preregistered): why the known-wrong student e2_ep_asym is not wrong.

A01, seed 0, the E2 setting (planted recip_lat teacher on F_lake, D2 hyper-parameters, 100 epochs):
  1. pair asymmetry ||w_fb - w_fw|| (absolute and relative to ||w_fw||) along training of e2_ep_asym, recorded
     after every batch (EP step, then the row-mass projection, exactly as Readout.fit), with the step at which
     the projection first fires and the largest |increment_fw - increment_fb| of any single EP step;
  2. for the trained e2_ep and e2_ep_asym students: the share of validation predictions that change when every
     motor -> relay feedback efficacy is set to zero.
Writes receipts/M11/E/<run commit>/e2_diag.json.
"""
from __future__ import annotations
import time, warnings
import numpy as np
from .m11_e import planted_teacher, OUT
from .m11_common import write_receipt, run_commit


def main(subject=1, seed=0):
    from .g1_readout import features, prep, Readout, evaluate
    from eval.scoring import fallback_class
    warnings.simplefilter("ignore"); t0 = time.time()
    F = features(subject); Ztr, Zva = prep(F["lake_train"], F["lake_val"]); d = Ztr.shape[-1]
    teacher, _, _ = planted_teacher(d, seed)
    def labels(Z):
        n, k, _ = Z.shape; a, ok = teacher.scores(Z.reshape(-1, d)); return a.reshape(n, k, 4).mean(1).argmax(1)
    ytr, yva = labels(Ztr), labels(Zva); out = {}
    for name, eps in (("e2_ep", 0.0), ("e2_ep_asym", 1.0)):
        net = Readout(d, "recip_lat", "ce01", 0.05, seed, asym_eps=eps); L = net.learner
        key = {(int(p), int(q)): i for i, (p, q) in enumerate(zip(net.pre, net.post))}
        fw = np.flatnonzero((net.pre < d) & (net.post >= d)); fb = np.array([key[(int(net.post[i]), int(net.pre[i]))] for i in fw])
        X, y = Ztr.reshape(-1, d), np.repeat(ytr, Ztr.shape[1]); D = net.drive(X); rng = np.random.default_rng(seed + 1000)
        e = np.asarray(L.brain.efficacy).copy(); trace = [(0, float(np.linalg.norm(e[fb] - e[fw])), float(np.linalg.norm(e[fw])))]
        step, first_proj, max_incr_gap = 0, None, 0.0
        for _ in range(100):  # Readout.fit, inlined to record the asymmetry after every batch
            idx = rng.permutation(len(D))
            for b in np.array_split(idx, max(1, len(idx) // 32)):
                e0 = np.asarray(L.brain.efficacy).copy(); L.step(D[b], y[b]); e1 = np.asarray(L.brain.efficacy).copy()
                max_incr_gap = max(max_incr_gap, float(np.abs((e1[fw] - e0[fw]) - (e1[fb] - e0[fb])).max()))
                n0 = net.n_projections; net.project(); step += 1
                if first_proj is None and net.n_projections > n0: first_proj = step
                e = np.asarray(L.brain.efficacy)
                if step % 500 == 0: trace.append((step, float(np.linalg.norm(e[fb] - e[fw])), float(np.linalg.norm(e[fw]))))
        e = np.asarray(L.brain.efficacy).copy(); trace.append((step, float(np.linalg.norm(e[fb] - e[fw])), float(np.linalg.norm(e[fw]))))
        k, _, _ = evaluate(net, Zva, yva, fallback_class(ytr), "sub"); p = np.asarray(evaluate.last_pred)
        e0 = e.copy(); e0[fb] = 0.0; L.brain = L.brain.with_parameters(efficacy=e0, bias=np.asarray(L.brain.bias))
        k0, _, _ = evaluate(net, Zva, yva, fallback_class(ytr), "sub"); p0 = np.asarray(evaluate.last_pred)
        out[name] = {"asym_rel_init": trace[0][1] / trace[0][2], "asym_rel_final": trace[-1][1] / trace[-1][2], "trace_step_absdiff_normfw": trace,
                     "steps": step, "first_projection_step": first_proj, "n_projections": net.n_projections, "max_increment_gap": max_incr_gap,
                     "agreement": float((p == yva).mean()), "agreement_feedback_zeroed": float((p0 == yva).mean()),
                     "share_pred_changed_by_zeroing_feedback": float((p0 != p).mean()), "n_val": int(len(yva))}
        print(name, {k_: v for k_, v in out[name].items() if k_ != "trace_step_absdiff_normfw"}, flush=True)
    write_receipt(OUT / run_commit(), "e2_diag", {"exploratory": True, "subject": subject, "seed": seed, **out, "wall_s": time.time() - t0}, subject=subject)


if __name__ == "__main__":
    main()
