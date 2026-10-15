"""Fail-closed checks for the experiment control-plane manifest.

This validates protocol boundaries only. It does not load data or run models.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise SystemExit(f"MANIFEST_INVALID: {message}")


def git_head(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"cannot read cadence commit at {path}: {error}")
    return result.stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def validate(manifest: dict[str, Any]) -> None:
    require(manifest.get("project") == "EP-EEG", "project must be EP-EEG")
    require(
        manifest.get("readiness") in {"bootstrap_only", "ready"},
        "readiness must be bootstrap_only or ready",
    )
    cadence = manifest.get("cadence")
    require(isinstance(cadence, dict), "cadence section is required")
    cadence_path = Path(str(cadence.get("path", ""))).resolve()
    require(cadence_path.is_dir(), f"cadence path does not exist: {cadence_path}")
    expected_commit = str(cadence.get("commit", ""))
    actual_commit = git_head(cadence_path)
    require(
        actual_commit == expected_commit or actual_commit.startswith(expected_commit),
        f"cadence commit is {actual_commit}, expected {expected_commit}",
    )

    dataset = manifest.get("dataset")
    require(isinstance(dataset, dict), "dataset section is required")
    require(dataset.get("train_session") == "T", "training must use session T")
    require(dataset.get("holdout_session") == "E", "holdout must use session E")
    require(
        dataset.get("holdout_policy") == "stage_gate_only",
        "holdout policy must remain stage_gate_only",
    )
    training_runs = dataset.get("training_runs")
    validation_runs = dataset.get("validation_runs")
    require(
        isinstance(training_runs, list) and training_runs == [0, 1, 2, 3],
        "training runs must be T runs 0-3",
    )
    require(
        isinstance(validation_runs, list) and validation_runs == [4, 5],
        "validation runs must be T runs 4-5",
    )
    require(not set(training_runs) & set(validation_runs), "training and validation overlap")

    require(manifest.get("subjects") == 9, "the frozen protocol has 9 subjects")
    seeds = manifest.get("seeds")
    require(isinstance(seeds, list) and len(seeds) >= 3, "every config needs at least 3 seeds")
    require(all(isinstance(seed, int) for seed in seeds), "seeds must be integers")

    stages = manifest.get("stages")
    require(isinstance(stages, dict) and stages, "at least one stage is required")
    for name, stage in stages.items():
        require(isinstance(stage, dict), f"stage {name} must be an object")
        require(isinstance(stage.get("arms"), list) and stage["arms"], f"stage {name} has no arms")
        require(stage.get("requires_receipts") is True, f"stage {name} must require receipts")
        require(stage.get("requires_holdout") is True, f"stage {name} must declare its holdout gate")
        holdout_arms = stage.get("holdout_arms")
        require(
            isinstance(holdout_arms, list)
            and set(holdout_arms).issubset(set(stage["arms"])),
            f"stage {name} holdout arms must be a subset of declared arms",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=ROOT / "automation" / "stage_manifest.json",
    )
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read manifest: {error}")
    validate(manifest)
    if manifest.get("readiness") == "bootstrap_only":
        print(f"MANIFEST_VALID_FOR_BOOTSTRAP: {args.manifest}")
        print("EXECUTION_NOT_READY: experiment runner, measured budgets, and stage gate are pending")
    else:
        print(f"MANIFEST_VALID: {args.manifest}")


if __name__ == "__main__":
    main()
