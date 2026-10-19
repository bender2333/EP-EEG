"""M11 shared helpers: receipts bound to code, dependency versions and data hashes (docs/plan.md §10)."""
from __future__ import annotations
import hashlib, json, platform, subprocess, time
from functools import lru_cache
from pathlib import Path
from .data import ROOT

CADENCE = Path("/workspace/cadence")
GDF_DIR = Path("/workspace/bciciv2a_gdf_T")
MAT_DIR = Path("/workspace/mne_data/MNE-bnci-data/~bci/database/001-2014")
import os as _os
# smoke tests (code checks, not results) write under receipts/M11_smoke so they never mix with M11 receipts
M11_OUT = ROOT / "receipts" / ("M11_smoke" if _os.environ.get("M11_SMOKE") else "M11")


@lru_cache(None)
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def git(*a, cwd=ROOT):
    return subprocess.check_output(["git", *a], cwd=cwd, text=True).strip()


@lru_cache(None)
def env_info():
    import importlib.metadata as md
    vers = {}
    for p in ("numpy", "scipy", "scikit-learn", "torch", "mne", "moabb", "braindecode", "pyriemann", "cadence-net", "numba"):
        try: vers[p] = md.version(p)
        except md.PackageNotFoundError: vers[p] = None
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "cadence_commit": git("rev-parse", "HEAD", cwd=CADENCE),
            "cadence_dirty": bool(git("status", "--porcelain", "--untracked-files=no", cwd=CADENCE)),
            "python": platform.python_version(), "versions": vers, "host_cpus": __import__("os").cpu_count()}


def data_hashes(subject, kind="mat"):
    p = (MAT_DIR / f"A{subject:02d}T.mat") if kind == "mat" else (GDF_DIR / f"A{subject:02d}T.gdf")
    return {p.name: sha256(p)}


def write_receipt(out_dir, name, payload, subject=None, kind="mat"):
    """One JSON per finished run, written immediately (G3 lesson: a restart loses at most the runs in flight)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"env": env_info(), "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **payload}
    if subject is not None: rec["data"] = data_hashes(subject, kind)
    p = out_dir / f"{name}.json"; tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    tmp.replace(p); return p


def existing(out_dir, name):
    """Resume after a container restart: a finished run's receipt, or None."""
    p = Path(out_dir) / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def run_commit():
    """Receipt directory key: the code commit a queue was started from (M11_RUN_COMMIT), so commits of ledger or
    report text during a long queue do not split one step's receipts or break resume."""
    return __import__("os").environ.get("M11_RUN_COMMIT") or git("rev-parse", "--short", "HEAD")
