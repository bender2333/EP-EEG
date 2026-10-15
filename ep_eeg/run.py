"""CLI runner for EP-EEG experiments across subjects, seeds, and arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from ep_eeg.baselines import run_eegnet, run_fbcsp, run_lda

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS_DIR = ROOT / "receipts"
RUNS_DIR = ROOT / "runs"
RESULTS_FILE = ROOT / "results.tsv"


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def append_result(
    commit: str,
    stage: str,
    arm: str,
    kappa_val: float,
    acc_val: float,
    settle_rate: float,
    seeds: str,
    wall_s: float,
    status: str,
    description: str,
) -> None:
    """Append a row to results.tsv adhering to AGENTS.md schema."""
    header_needed = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("commit\tstage\tarm\tkappa_val\tacc_val\tsettle_rate\tseeds\twall_s\tstatus\tdescription\n")
        f.write(
            f"{commit}\t{stage}\t{arm}\t{kappa_val:.4f}\t{acc_val:.4f}\t{settle_rate:.4f}\t{seeds}\t{wall_s:.2f}\t{status}\t{description}\n"
        )


def run_experiment(
    stage: str,
    arm: str,
    subject: int,
    seed: int,
) -> dict[str, Any]:
    t0 = time.time()
    if arm == "lda_track1":
        res = run_lda(subject_id=subject, track="track1", shuffle_labels=False, seed=seed)
    elif arm == "lda_track2":
        res = run_lda(subject_id=subject, track="track2", shuffle_labels=False, seed=seed)
    elif arm == "shuffled_labels":
        res = run_lda(subject_id=subject, track="track2", shuffle_labels=True, seed=seed)
    elif arm == "fbcsp":
        res = run_fbcsp(subject_id=subject, seed=seed)
    elif arm == "eegnet":
        res = run_eegnet(subject_id=subject, seed=seed)
    else:
        raise ValueError(f"Unknown arm: {arm}")

    wall_s = time.time() - t0
    res["wall_s"] = wall_s
    res["stage"] = stage
    res["arm"] = arm
    res["subject"] = subject
    res["seed"] = seed
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EP-EEG experiments")
    parser.add_argument("--stage", type=str, default="M0", help="Stage name (e.g. M0, M1)")
    parser.add_argument("--arm", type=str, required=True, help="Arm name")
    parser.add_argument("--subject", type=str, default="A01", help="Subject identifier (e.g. A01 or 1)")
    parser.add_argument("--all-subjects", action="store_true", help="Run across all 9 subjects")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="List of seeds to evaluate")
    args = parser.parse_args()

    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    subjects = list(range(1, 10)) if args.all_subjects else [int(args.subject.replace("A", ""))]
    seeds = args.seeds if args.seeds is not None else [args.seed]

    commit = get_git_commit()
    all_kappas = []
    all_accs = []
    total_wall = 0.0

    print(f"=== Starting Run: Stage={args.stage}, Arm={args.arm}, Subjects={subjects}, Seeds={seeds} ===")

    for sub in subjects:
        sub_kappas = []
        sub_accs = []
        for s in seeds:
            print(f"-> Running Subject A{sub:02d}, Seed {s}...")
            res = run_experiment(args.stage, args.arm, sub, s)
            total_wall += res["wall_s"]
            sub_kappas.append(res["kappa"])
            sub_accs.append(res["accuracy"])

            # Save individual receipt
            receipt_path = RECEIPTS_DIR / f"{args.stage}_{args.arm}_A{sub:02d}_seed{s}.json"
            receipt_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"   Acc: {res['accuracy']:.4f}, Kappa: {res['kappa']:.4f}, Time: {res['wall_s']:.2f}s")

        all_kappas.append(float(np.mean(sub_kappas)))
        all_accs.append(float(np.mean(sub_accs)))

    mean_kappa = float(np.mean(all_kappas))
    mean_acc = float(np.mean(all_accs))
    seed_str = ",".join(map(str, seeds))
    sub_desc = f"{len(subjects)} subjects (A01-A{len(subjects):02d})" if args.all_subjects else f"A{subjects[0]:02d}"

    append_result(
        commit=commit,
        stage=args.stage,
        arm=args.arm,
        kappa_val=mean_kappa,
        acc_val=mean_acc,
        settle_rate=1.0,
        seeds=seed_str,
        wall_s=total_wall,
        status="keep",
        description=f"{args.arm} on {sub_desc}",
    )

    print(f"\n=== Summary: {args.arm} | Mean Acc={mean_acc:.4f} | Mean Kappa={mean_kappa:.4f} | Total Time={total_wall:.2f}s ===")


if __name__ == "__main__":
    main()
