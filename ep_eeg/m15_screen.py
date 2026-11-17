"""M15 block 1b: four preregistered one-change screens on C, subjects 1/3/8, U=1500.

T session only. Receipts go to receipts/M15/screen and are not M14 grid receipts.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import numpy as np

from . import m14 as M
from . import m15_common as C

KINDS = ("beta_low", "beta_high", "one_sided", "eta_out")
SCREEN_SUBJECTS = (1, 3, 8)


def configure(g, kind):
    cfg = g["lr"].config
    if kind == "beta_low":
        g["lr"].config = replace(cfg, beta=0.01)
    elif kind == "beta_high":
        g["lr"].config = replace(cfg, beta=0.2)
    elif kind == "one_sided":
        g["lr"].config = replace(cfg, beta=0.04, centered=False)
    elif kind == "eta_out":
        rate = np.ones(len(g["conn"].pre)) if g["lr"].synapse_rate is None else np.array(g["lr"].synapse_rate, float)
        rate[g["path"] == M.PATHS["out"]] *= 5.0
        g["lr"].synapse_rate = rate
    else:
        raise ValueError(kind)
    return g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--subjects", nargs="+", type=int, default=list(SCREEN_SUBJECTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--U", type=int, default=1500)
    a = ap.parse_args(argv)
    C.seal_or_stop()
    for kind in a.kinds:
        for s in a.subjects:
            out = C.OUT / "screen" / f"{kind}_A{s:02d}_s{a.seed}.json"
            if out.exists():
                print("exists", out.name, flush=True)
                continue
            g = configure(M.build("C", 14500 + 100 * s + a.seed, eta=0.3), kind)
            data = C.load_t(s)
            tr = M.train(g, data, a.U, a.seed, eval_every=max(a.U, 1),
                         log=lambda m, k=kind, s=s: print(f"[{k} A{s:02d}] {m}", flush=True))
            ev, _, _ = M.evaluate(g, data["Xte"], data["yte"], data["ytr"])
            pred = ev.pop("pred")
            rec = {"stage": "M15", "block": "screen", "kind": kind, "split": "val", "subject": s,
                   "seed": a.seed, "U": a.U, "eta": 0.3, "dev": False,
                   "beta": g["lr"].config.beta, "centered": g["lr"].config.centered,
                   "test": ev, "train": {"curve": tr["curve"], "refused": tr["refused"],
                                         "certified_end": tr["certified_end"]},
                   "init_scale": g["init_scale"]}
            C.write_json(out, rec)
            print(f"{kind} A{s:02d} acc {ev['acc']:.3f} kappa {ev['kappa']:.3f}", flush=True)


if __name__ == "__main__":
    main()
