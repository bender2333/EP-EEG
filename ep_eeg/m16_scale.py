"""M16 block 1: temperature and outlet-budget variants of arm C. T session only."""
from __future__ import annotations
import argparse
from dataclasses import replace

from . import m14 as M
from . import m15_common as C
from .m14_probe import lda_scores, settle_split

KINDS = ("temp_match", "temp_mid", "out_budget")
SCREEN = (1, 3, 8)


def configure(kind, subject, seed, arm="C"):
    if kind == "out_budget":
        saved = M.BUDGET_INIT
        M.BUDGET_INIT = {**saved, "out": 0.7, "lat": 0.05}
        try:
            g = M.build(arm, 14500 + 100 * subject + seed, eta=0.3)
        finally:
            M.BUDGET_INIT = saved
    else:
        g = M.build(arm, 14500 + 100 * subject + seed, eta=0.3)
    temp = {"temp_match": 1e-3, "temp_mid": 3e-3}.get(kind, M.TEMP)
    g["lr"].config = replace(g["lr"].config, temperature=temp)
    return g


def run_one(kind, subject, seed, U, arm="C"):
    g = configure(kind, subject, seed, arm)
    cert = M.certificate(g)
    data = C.load_t(subject)
    if not cert.get("certified"):
        rec = {"stage": "M16", "block": "scale", "kind": kind, "arm": arm, "subject": subject,
               "seed": seed, "U": U, "split": "val", "crash": "cert", "cert_init": cert}
        path = C.OUT.parent / "M16" / "scale" / f"{arm}_{kind}_U{U}_A{subject:02d}_s{seed}.json"
        C.write_json(path, rec)
        return rec
    tr = M.train(g, data, U, seed, eval_every=min(250, U),
                 log=lambda m, k=kind, s=subject: print(f"[{arm} {k} A{s:02d}] {m}", flush=True))
    ev, _, _ = M.evaluate(g, data["Xte"], data["yte"], data["ytr"])
    ev.pop("pred", None)
    tr_f, _ = settle_split(g, data["Xtr"], data["ytr"], data["ytr"])
    va_f, _ = settle_split(g, data["Xte"], data["yte"], data["ytr"])
    lda = lda_scores(tr_f["outlets"], data["ytr"], va_f["outlets"], data["yte"])
    rec = {"stage": "M16", "block": "scale", "kind": kind, "arm": arm, "subject": subject,
           "seed": seed, "U": U, "split": "val", "dev": False, "crash": None,
           "temperature": g["lr"].config.temperature, "cert_init": cert,
           "init_scale": g["init_scale"], "test": ev,
           "lda_acc": lda["acc_val"], "lda_kappa": lda["kappa_val"],
           "gap": float(lda["acc_val"] - ev["acc"]),
           "train": {"refused": tr["refused"], "certified_end": tr["certified_end"],
                     "curve": tr["curve"]}}
    path = C.OUT.parent / "M16" / "scale" / f"{arm}_{kind}_U{U}_A{subject:02d}_s{seed}.json"
    C.write_json(path, rec)
    print(f"{arm} {kind} A{subject:02d} acc {ev['acc']:.3f} lda {lda['acc_val']:.3f} gap {rec['gap']:.3f}", flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--subjects", nargs="+", type=int, default=list(SCREEN))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--U", type=int, default=1500)
    ap.add_argument("--arm", default="C")
    a = ap.parse_args(argv)
    C.seal_or_stop()
    for kind in a.kinds:
        for s in a.subjects:
            path = C.OUT.parent / "M16" / "scale" / f"{a.arm}_{kind}_U{a.U}_A{s:02d}_s{a.seed}.json"
            if path.exists():
                print("exists", path.name, flush=True)
                continue
            run_one(kind, s, a.seed, a.U, a.arm)


if __name__ == "__main__":
    main()
