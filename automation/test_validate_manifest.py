"""Boundary tests for the experiment manifest validator."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "manifest_validator", ROOT / "automation" / "validate_manifest.py"
)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)
BASE = json.loads((ROOT / "automation" / "stage_manifest.json").read_text(encoding="utf-8"))


class ManifestValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = copy.deepcopy(BASE)
        validator.git_head = lambda _path: "6499e56"  # type: ignore[method-assign]

    def test_current_manifest_is_valid_for_bootstrap(self) -> None:
        validator.validate(self.manifest)

    def test_rejects_validation_runs_outside_frozen_split(self) -> None:
        self.manifest["dataset"]["validation_runs"] = [0, 1]
        with self.assertRaisesRegex(SystemExit, "validation runs must be T runs 4-5"):
            validator.validate(self.manifest)

    def test_rejects_unregistered_holdout_candidate(self) -> None:
        self.manifest["stages"]["M0"]["holdout_arms"].append("ep_experiment")
        with self.assertRaisesRegex(SystemExit, "holdout arms must be a subset"):
            validator.validate(self.manifest)

    def test_rejects_unknown_readiness(self) -> None:
        self.manifest["readiness"] = "run_anyway"
        with self.assertRaisesRegex(SystemExit, "readiness must be"):
            validator.validate(self.manifest)


if __name__ == "__main__":
    unittest.main()
