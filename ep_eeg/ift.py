"""Implicit (equilibrium) gradient for a cadence NeuralGraph learner (M11 D: the non-degenerate control).

Port of cadence benchmarks/local_equilibrium/run.py::implicit_gradient (read-only upstream, commit 6499e56),
generalised to every neuron as a state variable, to the quadratic nudge and to unpaired (asymmetric) edges.
At a free equilibrium v = W^T s(v) + d (W[pre, post], cadence docs/learning.md §2), with J = I - W^T diag(s'),
the adjoint solves J^T lam = s' * dL/ds, and dL/dW[i -> j] = lam_j s_i, dL/dbias_j = lam_j (effective weights;
the sandbox gain factor is 1, recorded by `factor`). Losses: cross-entropy nudge = T * CE(softmax(s_out / T), y)
(cadence omits 1/T), so dL/ds_out = softmax - onehot; quadratic nudge = 0.5 ||s_out - target||^2.
`ep_equivalent` maps the gradient onto what cadence's centred contrast estimates for each directed synapse:
a reciprocal pair has one weight, so both directions read -(dL/dW_ij + dL/dW_ji); an unpaired synapse reads
-dL/dW_ij. In the beta -> 0 limit EP contrast == ep_equivalent on a symmetric smooth branch.
"""
from __future__ import annotations
import numpy as np


def loss_grad_s(learner, s, labels):
    cfg = learner.config; out = learner.output_index; g = np.zeros_like(s)
    if cfg.nudge == "cross_entropy":
        z = s[:, out] / cfg.temperature; z = z - z.max(1, keepdims=True); lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        rows = np.arange(len(labels)); loss = float(-(cfg.temperature * lp[rows, labels]).mean())
        p = np.exp(lp); p[rows, labels] -= 1.0; g[:, out] = p
    else:
        t = learner.targets(labels)[:, out]; g[:, out] = s[:, out] - t; loss = float(0.5 * ((s[:, out] - t) ** 2).sum(1).mean())
    return g, loss


def implicit_grad(learner, state, labels, propagate=True):
    """Batch-mean dL/dW per directed synapse (effective weight), dL/dbias, and the loss at `state`.
    propagate=False is the M11 D2 "delta" control: lam = s' * dL/ds (J replaced by I, no credit through
    the recurrent / feedback couplings)."""
    br = learner.brain; conn = br.connectome
    s, v = np.asarray(state.activation), np.asarray(state.v)
    sp = br.neuron_model.slope_at(v)                                     # (batch, n)
    gs, loss = loss_grad_s(learner, s, np.asarray(labels))
    W = np.asarray(br.dense())                                           # W[pre, post]
    if propagate:
        JT = np.eye(conn.n)[None] - sp[:, :, None] * W[None]             # J^T = I - diag(s') W
        lam = np.linalg.solve(JT, (sp * gs)[..., None])[..., 0]
    else:
        lam = sp * gs
    g_edges = (s[:, conn.pre] * lam[:, conn.post]).mean(0)
    factor = np.asarray(br.weights) / np.where(np.asarray(br.efficacy) == 0, 1, np.asarray(br.efficacy))
    return g_edges, lam.mean(0), loss, factor


def ep_equivalent(learner, g_edges, g_bias):
    """What the centred EP contrast estimates for each synapse / neuron (see module docstring)."""
    rev = learner.reverse; c = -np.asarray(g_edges, float).copy(); p = rev >= 0
    c[p] = -(g_edges[p] + g_edges[rev[p]])
    return c, -np.asarray(g_bias, float)


def free_loss(learner, drive, labels, tolerance=1e-11, budget=20000):
    eq = learner.brain.equilibrate(drive, budget=budget, tolerance=tolerance, damping=3)
    if not np.all(eq.converged): raise ArithmeticError("free phase did not converge in the finite-difference check")
    return loss_grad_s(learner, np.asarray(eq.state.activation), np.asarray(labels))[1]
