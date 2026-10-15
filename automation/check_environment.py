"""Check the measured Python environment before scheduling an experiment."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform

from validate_manifest import git_head, validate


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise SystemExit(f"ENVIRONMENT_INVALID: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", type=Path,
                        default=ROOT / "automation" / "stage_manifest.json")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read manifest: {error}")
    validate(manifest)
    expected_python = str(manifest["environment"]["python"])
    if not platform.python_version().startswith(expected_python + "."):
        fail(f"Python is {platform.python_version()}, expected {expected_python}.x")
    for package, expected in manifest["environment"]["packages"].items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            fail(f"package {package} is not installed: {error}")
        if actual != expected:
            fail(f"package {package} is {actual}, expected {expected}")
    cadence_path = Path(manifest["cadence"]["path"]).resolve()
    cadence_commit = git_head(cadence_path)
    expected_commit = str(manifest["cadence"]["commit"])
    if cadence_commit != expected_commit and not cadence_commit.startswith(expected_commit):
        fail(f"cadence HEAD is {cadence_commit}, expected {expected_commit}")
    cadence_module = importlib.import_module("cadence")
    module_path = Path(cadence_module.__file__).resolve()
    if cadence_path not in module_path.parents:
        fail(f"cadence imports from {module_path}, outside {cadence_path}")
    print(f"ENVIRONMENT_VALID: Python {platform.python_version()}")
    print(f"CADENCE_VALID: {cadence_commit} ({module_path})")
    for package, expected in manifest["environment"]["packages"].items():
        print(f"PACKAGE_VALID: {package}=={expected}")


if __name__ == "__main__":
    main()
