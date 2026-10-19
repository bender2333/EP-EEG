"""M11 D2 (reports/M11.md §6.2): readout micro-problem on G1's F_lake, non-degenerate structures, EP vs IFT.

Structures recip / lateral / recip_lat, trainers EP / IFT / delta (amendment M1-12), G1's inner-split choice for recip (cross-entropy T=0.1, eta 0.05,
sub rows, 100 epochs), trainers EP (cadence Learner) and IFT (ep_eeg.ift through Learner.apply), fresh seeds
3, 4, 5 (G1 saw recip on seeds 0-2), final split (runs 0-3 -> 4-5). One receipt per run, written on finish.
EP runs also record, every 20th batch, the cosine between the EP contrast and the IFT step at the same state.
"""
from __future__ import annotations
import argparse, itertools
from concurrent.futures import ProcessPoolExecutor
from .data import ROOT
from .g1_readout import run, features
from .m11_common import write_receipt, git, M11_OUT, existing

OUT = M11_OUT / "D2"
STRUCTS, TRAINERS, SEEDS = ("recip", "lateral", "recip_lat"), ("ep", "ift", "delta"), (3, 4, 5)


def job(a):
    struct, trainer, seed, subject = a
    prev = existing(OUT / git("rev-parse", "--short", "HEAD") / f"{struct}_{trainer}", f"A{subject:02d}_s{seed}")
    if prev: return prev
    cfg = dict(feat="lake", struct=struct, nudge="ce01", eta=0.05, rows="sub", trainer=trainer, probe_every=20 if trainer == "ep" else 0)
    r = run(subject, cfg, seed, "final")
    rec = {"stage": "M11", "step": "D2", "arm": f"{struct}_{trainer}", **r, "settle_rate": r["settle_val"]}
    write_receipt(OUT / git("rev-parse", "--short", "HEAD") / f"{struct}_{trainer}", f"A{subject:02d}_s{seed}", rec, subject=subject)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--structs", default=",".join(STRUCTS))
    ap.add_argument("--trainers", default=",".join(TRAINERS)); ap.add_argument("--seeds", default="3,4,5"); a = ap.parse_args()
    for s in range(1, 10): features(s)
    jobs = list(itertools.product(a.structs.split(","), a.trainers.split(","), map(int, a.seeds.split(",")), range(1, 10)))
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(job, jobs, chunksize=1):
            print(f"{r['arm']} A{r['subject']:02d} s{r['seed']} ep {r['kappa_ep']:.3f} ridge {r['kappa_ridge']:.3f} gap {r['gap']:+.3f} "
                  f"acc {r['accept_rate']:.3f} cert {r['cert_fail']} {r['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
