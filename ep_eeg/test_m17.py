"""M17 harness. Structural checks do not load a session."""
import ast
from pathlib import Path

import numpy as np
import pytest

from ep_eeg import m17_cspnet as N
from ep_eeg import m17_move as V


def test_no_e_loader():
    root = Path(V.__file__).resolve().parent
    bad = ("load_e_session", "load_subject_gate", "holdout", "EPEEG_GATE")
    for path in root.glob("m17_*.py"):
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


def test_coherence_extremes():
    same = np.ones((8, 3))
    assert np.allclose(V.coherence(same), 1.0)
    cancel = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert V.coherence(cancel)[0] == pytest.approx(0.0)


def test_csp_nets_certify_and_sources_have_no_incoming():
    import cadence as cd
    for kind in N.KINDS:
        g = N.build(kind, seed=0)
        cert = cd.certificate(g["lr"].brain, sources=range(g["ns"]))
        assert cert.certified, (kind, cert)
        pre, post = g["lr"].brain.connectome.pre, g["lr"].brain.connectome.post
        assert not np.any(post < g["ns"])
        _ = pre
