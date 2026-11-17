"""M18 synthetic spatial task. The mixing matrix is fixed; only the draws and the init seed change.

Class information sits in the variance of four latent sources. A 22 x 22 orthogonal mix hides them.
"""
from __future__ import annotations
import numpy as np

N_CH = 22
N_TIME = 32
N_CLASS = 4
MIX_SEED = 18018


def mixing(seed=MIX_SEED):
    rng = np.random.default_rng(seed)
    A, _ = np.linalg.qr(rng.normal(size=(N_CH, N_CH)))
    return A, A[:, :N_CLASS]


def _draw(n, rng, A):
    y = np.tile(np.arange(N_CLASS), n // N_CLASS)
    y = rng.permutation(y)
    scale = np.ones((n, N_CH))
    for k in range(N_CLASS):
        on = y == k
        scale[on, k] = 2.0
        for j in range(N_CLASS):
            if j != k:
                scale[on, j] = 0.5
    S = rng.normal(size=(n, N_TIME, N_CH)) * scale[:, None, :]
    X = np.einsum("ck,ntk->ntc", A, S).reshape(n, N_TIME * N_CH)
    return X.astype(np.float64), y.astype(np.int64)


def make_synthetic(seed, n_train=192, n_val=96):
    """One window per trial, shape (n, 32*22). ``U`` is the discriminative subspace, shape (22, 4)."""
    A, U = mixing()
    Xtr, ytr = _draw(n_train, np.random.default_rng(1000 + seed), A)
    Xte, yte = _draw(n_val, np.random.default_rng(2000 + seed), A)
    return {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte, "U": U, "A": A}


def subspace_energy(W, U):
    """Fraction of the filter matrix's energy that lies in the column space of ``U``."""
    Uq, _ = np.linalg.qr(np.asarray(U, float))
    W = np.asarray(W, float)
    num = np.sum((W @ Uq) ** 2)
    den = np.sum(W ** 2) + 1e-12
    return float(num / den)


def principal_angle_deg(W, U):
    """Mean principal angle, in degrees, between the top singular subspace of W and U."""
    Uq, _ = np.linalg.qr(np.asarray(U, float))
    k = Uq.shape[1]
    _, _, vt = np.linalg.svd(np.asarray(W, float), full_matrices=False)
    V = vt[:k].T
    s = np.linalg.svd(Uq.T @ V, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.degrees(np.mean(np.arccos(s))))
