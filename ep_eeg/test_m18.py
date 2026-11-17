"""M18 harness. Structural checks do not load a session."""
import ast
from pathlib import Path

import numpy as np

from ep_eeg.m18_fix import class_prototypes, pathway_scale
from ep_eeg.m18_mech import hop_gain
from ep_eeg.m18_synth import make_synthetic, subspace_energy


def test_no_e_loader():
    root = Path(__file__).resolve().parent
    bad = ("load_e_session", "load_subject_gate", "holdout", "EPEEG_GATE")
    for path in root.glob("m18_*.py"):
        tree = ast.parse(path.read_text())
        names, constants = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                constants.add(node.value)
        for name in bad:
            assert name not in names and name not in constants, path.name


def test_subspace_energy_identities():
    data = make_synthetic(0)
    U = data["U"]
    assert data["Xtr"].shape == (192, 32 * 22)
    assert list(np.bincount(data["ytr"])) == [48, 48, 48, 48]
    assert subspace_energy(np.eye(22), U) == np.float64(4 / 22) or abs(subspace_energy(np.eye(22), U) - 4 / 22) < 1e-9
    W = np.zeros((22, 22))
    W[:4] = U.T
    assert subspace_energy(W, U) == np.float64(1.0) or abs(subspace_energy(W, U) - 1.0) < 1e-9


def test_pathway_scale_clips_and_prototypes_have_unit_rms():
    assert pathway_scale(3e-3) == 1.0
    assert pathway_scale(3e-5) == 100.0
    assert pathway_scale(1e-3) == 3.0
    R = class_prototypes(22)
    assert R.shape == (22, 4)
    assert np.allclose(np.sqrt(np.mean(R ** 2, axis=0)), 1.0)


def test_hop_gain_is_slope_times_weight():
    pre = np.array([0])
    post = np.array([1])
    weights = np.array([0.4])
    slope = np.array([1.0, 0.5])
    pre_id = np.array([0, 1])
    post_id = np.array([0, 1])
    assert hop_gain(pre, post, weights, slope, pre_id, post_id, 0, 1) == np.float64(0.2)
