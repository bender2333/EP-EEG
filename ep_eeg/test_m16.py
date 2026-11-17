"""M16 harness. No EEG session is required for the structural checks."""
import ast
from pathlib import Path

import numpy as np
import pytest

from ep_eeg import m14 as M
from ep_eeg import m16_scale as S
from ep_eeg import m16_spatial as P


def test_no_e_loader():
    root = Path(S.__file__).resolve().parent
    bad = ("load_e_session", "load_subject_gate", "holdout", "EPEEG_GATE")
    for path in list(root.glob("m16_*.py")):
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


def test_temperature_and_budget_certificate():
    g = S.configure("temp_match", 1, 0)
    assert g["lr"].config.temperature == pytest.approx(1e-3)
    assert M.certificate(g)["certified"]
    b = S.configure("out_budget", 1, 0)
    cert = M.certificate(b)
    assert cert["certified"], cert
    assert b["init_scale"]["budget"]["out"] == pytest.approx(0.7)
    assert b["init_scale"]["budget"]["lat"] == pytest.approx(0.05)
    # the module constant used by ordinary builds is unchanged
    assert M.BUDGET_INIT["out"] == pytest.approx(0.5)
    assert M.BUDGET_INIT["lat"] == pytest.approx(0.15)


def test_set_front_roundtrip():
    g = M.build("C", 14600, eta=0.3)
    W = np.arange(M.NL * M.NL, dtype=float).reshape(M.NL, M.NL) / 100.0
    P.set_front(g, W)
    got = M.front_matrix(g)
    assert np.allclose(got, W)
