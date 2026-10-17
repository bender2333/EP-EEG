"""M9c: is the lake operating in its linear regime? (zero-compute probe on saved readout_diag weights)

For each subject and net (random lake = untrained M6-config sandbox seed 0; "readout" and "full" = saved EP nets),
over all train+val sub-window rows of the free phase:
  column activation a: mean |a|, 5/95 percentiles;  potential v: share |v| > 1, share v < 0
  (learning_neuron_model rho = tanh(v/2) for v > 0, 0.1 tanh(v/2) for v < 0: a 10x slope kink at v = 0,
   so small |v| is leaky-ReLU-like, not linear, whenever v takes both signs)
  input drive per column neuron (sum of one-way input synapses x feature): std across rows, mean over neurons
  linearity: R^2 of an ordinary least-squares fit a ~ [X, 1] (66 inputs -> 88 columns), variance-weighted,
  plus the same for motor activations. R^2 ~ 1 means the lake is effectively a linear projection.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
from .data import load_subject, ROOT
from .sandbox import Sandbox, SandboxConfig

M6 = dict(leak=1.0, rowmass_project=0.9, eta=0.5, temperature=0.01, epochs=100)


def r2(Y, X):
    A = np.column_stack([X, np.ones(len(X))]); B, *_ = np.linalg.lstsq(A, Y, rcond=None)
    res = Y - A @ B; tot = Y - Y.mean(0)
    return float(1 - (res ** 2).sum() / (tot ** 2).sum())


def stats(sb, d):
    X = np.concatenate([d["train"]["track2_x"], d["val"]["track2_x"]]); n, k, f = X.shape; Xf = X.reshape(-1, f)
    eq = sb.free_states(X); a = np.asarray(eq.state.activation); v = np.asarray(eq.state.v)
    ac, vc = a[:, sb.col0:sb.out0], v[:, sb.col0:sb.out0]; am = a[:, sb.outputs]
    W = np.asarray(sb.learner.brain.dense()); drive = Xf @ W[:f, sb.col0:sb.out0]
    return {"col_mean_abs_a": float(np.abs(ac).mean()), "col_a_p5": float(np.percentile(ac, 5)),
            "col_a_p95": float(np.percentile(ac, 95)), "col_frac_absv_gt1": float((np.abs(vc) > 1).mean()),
            "col_frac_v_neg": float((vc < 0).mean()), "col_v_std": float(vc.std()),
            "col_input_drive_std": float(drive.std(0).mean()),
            "col_frac_sat": float((np.abs(ac) > 0.98).mean()), "motor_mean_abs_a": float(np.abs(am).mean()),
            "r2_col_linear_in_X": r2(ac, Xf), "r2_motor_linear_in_X": r2(am, Xf),
            "converged": float(np.asarray(eq.converged).mean())}


def load_into(sb, npz):
    z = np.load(npz)
    assert np.array_equal(z["pre"], sb.conn.pre) and np.array_equal(z["post"], sb.conn.post), "graph mismatch"
    sb.learner.brain = sb.learner.brain.with_parameters(efficacy=z["efficacy"], bias=z["bias"])
    return sb


def main(diag_dir):
    out = {}
    for s in range(1, 10):
        d = load_subject(s); row = {}
        row["random"] = stats(Sandbox(SandboxConfig(**M6), 0), d)
        for tag, opts in (("readout", {**M6, "train_internal": False, "project_plastic_only": True}), ("full", M6)):
            row[tag] = stats(load_into(Sandbox(SandboxConfig(**opts), 0), Path(diag_dir) / f"A{s:02d}_{tag}.npz"), d)
        out[s] = row; print(f"A{s:02d}", {t: round(row[t]["r2_col_linear_in_X"], 4) for t in row}, flush=True)
    p = Path(diag_dir) / "linearity.json"; p.write_text(json.dumps(out, indent=1)); print(p)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(str(ROOT / "receipts/M9c/readout_diag/*")))[-1])
