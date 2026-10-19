"""M12 S0 pre-check: activation unit tests (amendment 1, N2) — runs the fork's tests/test_evenodd.py and records the outcome,
plus this repo's own closed-form-vs-grid check of L on the pre-registered grid ([-60, 60], 4e6 + 1 points)."""
from __future__ import annotations
import subprocess, sys, time
import numpy as np
from . import m12_common as C
from cadence.neuron import evenodd_lipschitz


def main():
    C.e_seal(); C.start_clock("m12_s0_unit"); t0 = time.time()
    cad = C.Path(C.env_info()["cadence_path"])
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_evenodd.py", "-rA"],
                       cwd=cad, capture_output=True, text=True)
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    t = np.tanh(0.5 * np.linspace(-60.0, 60.0, 4_000_001))
    rows = []
    for a, b in [(1.0, 0.25), (1.0, 1.0), (1.0, 0.6), (1.0, 0.0), (1.0, -1.0), (0.2, 2.0), (2.0, -2.0), (1.0, 0.49), (1.0, 0.51)]:
        L = evenodd_lipschitz(a, b); g = float(np.abs(0.5 * (1 - t * t) * (a + 2 * b * t)).max())
        rows.append({"a": a, "b": b, "L": L, "grid_max": g, "ok": bool(L >= g - 1e-12 and L - g <= 1e-8)})
    ok = p.returncode == 0 and all(r["ok"] for r in rows)
    C.write_receipt(C.OUT / "S0" / "unit" / C.run_commit(), "unit", {"pytest_returncode": p.returncode, "pytest_summary": tail,
                    "pytest_stdout": p.stdout[-20000:], "grid_check": rows, "pass": ok, "wall_s": time.time() - t0})
    C.ledger("M12-S0", "precheck_unit", "NA", 1, time.time() - t0, "pass" if ok else "fail", f"N2 activation unit tests: {tail}; L grid check {sum(r['ok'] for r in rows)}/{len(rows)}")
    print(tail, ok)


if __name__ == "__main__":
    main()
