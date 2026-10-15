import numpy as np, pytest
from eval import preprocess as P, scoring as S


def test_subwindow_geometry():
    assert P.N_SUB == 11 and P.SEG_LEN == 750 and P.SUB_STARTS[-1] + P.WIN_LEN == 750
    sw = P.subwindows(np.random.randn(2, 22, 750))
    assert sw.shape == (2, 11, 22, 125)
    assert P.track1(sw).shape == (2, 11, 44) and P.track2(sw).shape == (2, 11, 66)


def test_aggregate_fallback_and_coverage():
    sc = np.zeros((2, 11, 4)); sc[0, :, 2] = 1
    ok = np.ones((2, 11), bool); ok[1] = False
    pred, rej = S.aggregate(sc, ok, fallback=3)
    assert pred.tolist() == [2, 3] and rej.tolist() == [False, True]
    m = S.score(np.array([2, 3]), pred, ok)
    assert m["rejected_trials"] == 1 and m["settle_rate"] == 0.5


def test_holdout_requires_gate(monkeypatch):
    from eval.holdout import load_e_session
    monkeypatch.delenv("EPEEG_GATE", raising=False)
    with pytest.raises(PermissionError):
        load_e_session(1, gate_token="STAGE_GATE")
