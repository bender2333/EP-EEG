"""M11 C diagnostic (exploratory, amendment 4): does the cadence temporal EP update go through for the C network?

For subjects A01-A04, seed 0: the first n observe attempts of the ep_noproj run exactly as run_arm makes them
(same net, ridge C, crop rng seed + 100, beta 0.05; refused attempts leave the parameters unchanged, so while
updates are refused these are literally the run's first attempts), at the selected encoding (gain from
settings.json) and at gain 3 (port 5). Records, per attempt, the reason and, per phase, the solver reason,
convergence, residual and wall time. Reads no prediction or generation metric.
Writes receipts/M11/C/accept_diag.json.
"""
from __future__ import annotations
import sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from . import m11_c as C
from .m11_common import write_receipt


def job(args):
    subject, gain, n = args; warnings.simplefilter("ignore"); s = C.settings(); port = s["port"]; runs = C.data(subject)["runs"]
    net = C.new_net(0, port); model = C.Net(C.params(net), gain, port).fit_readout(runs[:4]); P = net.parameters(); P["C"] = model.P["C"]; net.set_parameters(P)
    rng = np.random.default_rng(0 + 100); out = []
    for _ in range(n):
        zs, tgt = C.crops(runs, rng, C.OPT["batch"]); u = np.stack([C.encode(z, gain, port) for z in zs])
        free = net.imagine(u, state=np.zeros((len(u), C.H))).output; tgt = tgt.copy(); tgt[:, :C.BURN] = free[:, :C.BURN]
        t0 = time.time(); net.reset(); r = net.observe(u, tgt, beta=C.OPT["beta"], rate=0.0)
        ph = {k: {"reason": p.reason, "converged": bool(p.converged), "residual": p.residual}
              for k in ("free", "plus", "minus") if (p := getattr(r, k, None)) is not None}
        out.append({"updated": bool(r.updated), "reason": r.reason, "phases": ph, "wall_s": time.time() - t0})
    return {"subject": subject, "gain": gain, "port": port, "attempts": out}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5; s = C.settings()
    jobs = [(subj, g, n) for g in (s["gain"], 3.0) for subj in (1, 2, 3, 4)]
    with ProcessPoolExecutor(4) as ex: res = list(ex.map(job, jobs))
    for r in res: print(r["subject"], r["gain"], [a["reason"] for a in r["attempts"]], [round(a["wall_s"]) for a in r["attempts"]], flush=True)
    write_receipt(C.OUT, "accept_diag", {"exploratory": True, "n_attempts": n, "results": res})


if __name__ == "__main__":
    main()
