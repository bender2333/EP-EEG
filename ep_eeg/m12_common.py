"""M12 shared helpers: E-session seal (filename check + audit hook), receipts (reports/M12.md §0.6 and
amendment 1 N7), the results.tsv ledger, resume, and the 2-day S0 wall clock.

Importing this module installs a sys.addaudithook guard that refuses to open any file named A0[1-9]E.(mat|gdf)
(amendment 1, B1). No E file is ever hashed, read or listed by content; the seal checks file names only.
"""
from __future__ import annotations
import hashlib, json, os, platform, re, subprocess, sys, time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CADENCE = Path(os.environ.get("M12_CADENCE", "/workspace/cadence"))
M12_DATA = Path(os.environ.get("M12_DATA", "/workspace/m12_data"))
os.environ["MNE_DATA"] = str(M12_DATA)
OUT = ROOT / "receipts" / "M12"
E_PATTERN = re.compile(r"A0[1-9]E\.(mat|gdf)$", re.IGNORECASE)
CLOCK = OUT / "S0" / "clock.json"
CLOCK_LIMIT_S = 2 * 24 * 3600


def _guard(event, args):
    if event == "open" and args:
        p = args[0]
        if isinstance(p, (str, bytes, os.PathLike)):
            name = os.path.basename(os.fsdecode(p))
            if E_PATTERN.search(name):
                raise PermissionError(f"M12 E-session seal: refusing to open {name}")


sys.addaudithook(_guard)


def _selftest():
    try:
        open("/nonexistent/A01E.mat", "rb")
    except PermissionError:
        return True
    except FileNotFoundError:
        return False
    return False


HOOK_SELFTEST = _selftest()


def e_seal():
    """File-name-only check of the M12 data directory; never opens, hashes or stats E files' contents."""
    matches = sorted(str(p.relative_to(M12_DATA)) for p in M12_DATA.rglob("*") if E_PATTERN.search(p.name)) if M12_DATA.exists() else []
    seal = {"data_dir": str(M12_DATA), "method": "filename+audit_hook", "matches": matches,
            "hook_installed": True, "hook_selftest": HOOK_SELFTEST}
    if matches or not HOOK_SELFTEST:
        raise SystemExit(f"E-session seal failed: {seal}")
    return seal


def git(*a, cwd=ROOT):
    return subprocess.check_output(["git", *a], cwd=cwd, text=True).strip()


def threads():
    out = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    try:
        import torch
        out["torch_threads"] = torch.get_num_threads()
    except ImportError:
        out["torch_threads"] = None
    return out


@lru_cache(None)
def env_info():
    import importlib.metadata as md
    import cadence
    vers = {}
    for p in ("numpy", "scipy", "torch", "numba", "cadence-net"):
        try: vers[p] = md.version(p)
        except md.PackageNotFoundError: vers[p] = None
    cad_dir = Path(cadence.__file__).resolve().parents[2]
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "cadence_path": str(cad_dir), "cadence_commit": git("rev-parse", "HEAD", cwd=cad_dir),
            "cadence_dirty": bool(git("status", "--porcelain", "--untracked-files=no", cwd=cad_dir)),
            "cadence_version": cadence.__version__, "python": platform.python_version(), "versions": vers,
            "cpu_model": cpu, "cpus": os.cpu_count()}


def sha(*arrays):
    import numpy as np
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _default(o):
    if hasattr(o, "tolist"): return o.tolist()
    if isinstance(o, (set, tuple)): return list(o)
    return float(o)


def write_receipt(out_dir, name, payload, dev=False):
    """One JSON per finished run, written atomically and immediately."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"env": env_info(), "threads": threads(), "loadavg": os.getloadavg(), "e_seal": e_seal(), "dev": bool(dev),
           "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **payload}
    p = out_dir / f"{name}.json"; tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=_default)); tmp.replace(p)
    return p


def existing(out_dir, name):
    p = Path(out_dir) / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def run_commit():
    return os.environ.get("M12_RUN_COMMIT") or git("rev-parse", "--short", "HEAD")


def start_clock(what):
    """The S0 2-day wall clock starts at the first official acceptance run (amendment 1)."""
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    if not CLOCK.exists():
        CLOCK.write_text(json.dumps({"started": time.time(), "started_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "first_run": what}))
    c = json.loads(CLOCK.read_text())
    return time.time() - c["started"]


def clock_left():
    if not CLOCK.exists(): return CLOCK_LIMIT_S
    return CLOCK_LIMIT_S - (time.time() - json.loads(CLOCK.read_text())["started"])


def ledger(stage, arm, settle_rate, seeds, wall_s, status, description, kappa="NA", acc="NA"):
    row = "\t".join(str(x) for x in (run_commit(), stage, arm, kappa, acc, f"{settle_rate:.4f}" if isinstance(settle_rate, float) else settle_rate,
                                     seeds, int(wall_s), status, description))
    with open(ROOT / "results.tsv", "a") as f: f.write(row + "\n")
