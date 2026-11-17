"""M15 harness. No EEG session is loaded."""
import ast
from pathlib import Path

import numpy as np
import pytest

from ep_eeg import m15_align as A
from ep_eeg import m15_common as C
from ep_eeg import m15_screen as S


def test_no_e_loader():
    root = Path(C.__file__).resolve().parent
    bad = ("load_e_session", "load_subject_gate", "holdout", "EPEEG_GATE")
    for path in root.glob("m15_*.py"):
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


def test_paired_ties_and_win():
    z = C.paired([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    assert z["ties"] == 3 and z["p"] == 1.0 and z["mean_diff"] == 0.0
    w = C.paired([0.5] * 8, [0.2] * 8)
    assert w["wins"] == 8 and w["losses"] == 0 and w["p"] < 0.05


def test_ce_loss_prefers_the_labeled_outlet():
    act = np.zeros((4, 4))
    y = np.array([0, 1, 2, 3])
    act[np.arange(4), y] = 1.0
    wrong = np.zeros((4, 4))
    wrong[np.arange(4), (y + 1) % 4] = 1.0
    assert C.ce_loss(act, y, 0.01) < C.ce_loss(wrong, y, 0.01)


def test_cosine():
    assert A.cosine([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert A.cosine([0.0, 0.0], [1.0, 0.0]) is None


def test_ep_head_fits_a_separable_problem():
    rng = np.random.default_rng(0)
    y = np.tile(np.arange(4), 16)
    X = rng.normal(scale=0.05, size=(len(y), 4))
    X[np.arange(len(y)), y] += 1.0
    h = C.build_head(4, seed=1, eta=0.5)
    info = C.train_head(h, X, y, U=200, seed=1, batch=16)
    pred, ok = C.head_activity(h, X)
    assert ok.all()
    assert info["train_acc"] > 0.9
    assert (pred.argmax(1) == y).mean() > 0.9


def test_screen_kinds_only_change_the_named_knob():
    from ep_eeg import m14 as M
    base = M.build("C", 14600, eta=0.3)
    low = S.configure(M.build("C", 14600, eta=0.3), "beta_low")
    side = S.configure(M.build("C", 14600, eta=0.3), "one_sided")
    fast = S.configure(M.build("C", 14600, eta=0.3), "eta_out")
    assert low["lr"].config.beta == pytest.approx(0.01) and low["lr"].config.centered
    assert side["lr"].config.centered is False and side["lr"].config.beta == pytest.approx(0.04)
    assert fast["lr"].config.beta == pytest.approx(base["lr"].config.beta)
    out = fast["path"] == M.PATHS["out"]
    assert np.allclose(fast["lr"].synapse_rate[out], 5.0 * base["lr"].synapse_rate[out])
    assert np.allclose(fast["lr"].synapse_rate[~out], base["lr"].synapse_rate[~out])
