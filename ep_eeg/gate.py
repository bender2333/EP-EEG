"""Stage gate: the ONLY path that reads the E session. One evaluation per frozen candidate.

Models are trained exactly as in the frozen validation candidate (T runs 0-3,
same seeds, same options); E is scored with the frozen aggregation.
"""
from __future__ import annotations
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from eval.scoring import aggregate, fallback_class, score
from .data import ROOT


def one(arm, subject, seed, opts):
    os.environ["EPEEG_GATE"] = "1"
    from .data import load_subject_gate
    from .arms import get
    d = load_subject_gate(subject)
    scores, ok, extra = get(arm)(d["train"], d["val"], seed, **opts)
    pred, rej = aggregate(scores, ok, fallback_class(d["train"]["y"]))
    m = score(d["val"]["y"], pred, ok)
    m.update(subject=subject, seed=seed, pred=pred.tolist(), y=d["val"]["y"].tolist())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True); ap.add_argument("--candidate", required=True,
                    help="path to frozen validation summary.json")
    ap.add_argument("--read-holdout", action="store_true", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    summ = json.loads(open(a.candidate).read())
    cid = f'{a.stage}__{summ["arm"]}__{summ["commit"]}__{abs(hash(json.dumps(summ["opts"], sort_keys=True))) % 10**8}'
    out = ROOT / "holdout_receipts" / f"{cid}.json"
    if out.exists():
        raise SystemExit(f"HOLDOUT_ALREADY_READ: {cid}")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"status": "started", "candidate": a.candidate}))  # lock before reading E
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        res = [f.result() for f in [ex.submit(one, summ["arm"], s, k, summ["opts"]) for s in summ["subjects"] for k in summ["seeds"]]]
    kap = np.array([[r["kappa"] for r in res if r["subject"] == s] for s in summ["subjects"]])
    rec = {"candidate_id": cid, "stage": a.stage, "arm": summ["arm"], "commit": summ["commit"], "opts": summ["opts"],
           "kappa_E": float(kap.mean()), "kappa_E_seed_sd": float(kap.mean(0).std(ddof=1)) if kap.shape[1] > 1 else 0.0,
           "per_subject_kappa_E": kap.mean(1).tolist(), "acc_E": float(np.mean([r["acc"] for r in res])),
           "settle_rate_E": float(np.mean([r["settle_rate"] for r in res])),
           "rejected_trials_E": int(sum(r["rejected_trials"] for r in res)),
           "kappa_val": summ["kappa_val"], "wall_s": time.time() - t0, "runs": res}
    out.write_text(json.dumps(rec))
    f = ROOT / "holdout.tsv"
    if not f.exists():
        f.write_text("candidate_id\tstage\tarm\tkappa_E\tkappa_E_seed_sd\tacc_E\tsettle_rate_E\trejected_trials_E\tkappa_val\tstatus\n")
    with f.open("a") as h:
        h.write(f'{cid}\t{a.stage}\t{summ["arm"]}\t{rec["kappa_E"]:.4f}\t{rec["kappa_E_seed_sd"]:.4f}\t{rec["acc_E"]:.4f}\t'
                f'{rec["settle_rate_E"]:.4f}\t{rec["rejected_trials_E"]}\t{summ["kappa_val"]:.4f}\tfresh\n')
    print(json.dumps({k: v for k, v in rec.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
