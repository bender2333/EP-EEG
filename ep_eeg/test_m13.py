"""M13 E0 harness checks (no EEG). Run: python -m pytest ep_eeg/test_m13.py -q"""
import numpy as np
import pytest

from ep_eeg import m13_common as C


def test_patterns():
    p = C.patterns()
    assert len(p["P1"]) == 22 and all(len(t) == 3 for t in p["P3"])
    assert set().union(*map(set, p["P3"])) == set(range(22))
    assert len(set(p["P3"])) == len(p["P3"])


def test_spline_constant_and_smooth_field():
    hid = [7, 8, 13]; vis = [c for c in range(22) if c not in hid]
    M = C.spline_matrix(hid)
    assert np.allclose(M @ np.ones(len(vis)), 1.0, atol=1e-6)          # constants are reproduced
    U = C.unit(C.XYZ); f = U @ np.array([0.3, -0.5, 0.8])               # a smooth (degree-1) field
    assert np.max(np.abs(M @ f[vis] - f[hid])) < 0.05 * np.ptp(f)


def test_derangement():
    for s in range(3):
        p = C.derangement(s); assert sorted(p) == list(range(22)) and not np.any(p == np.arange(22))


def test_r2():
    y = np.random.default_rng(0).normal(size=(500, 2))
    assert np.allclose(C.r2_cols(y, y), 1) and np.all(C.r2_cols(np.zeros_like(y) + y.mean(0), y) < 1e-12)


@pytest.fixture(scope="module")
def E():
    from ep_eeg import m13_e0
    return m13_e0


def test_no_leak(E):
    """A hidden lead's value never reaches the lake: changing it leaves every state unchanged."""
    g = E.build_lake(11); X = np.random.default_rng(1).normal(size=(64, 22)); hid = [7, 8, 13]
    a = E.lake_states(g, X, hid)
    X2 = X.copy(); X2[:, hid] += 5.0
    b = E.lake_states(g, X2, hid)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_v0_is_position_free(E):
    """v0 reads no position: a lake wired with leads relabelled by pi, fed data relabelled the same way,
    is the same network. Settled states and one EP update agree to round-off."""
    rng = np.random.default_rng(2); X = rng.normal(size=(64, 22)); hid = [7, 8, 13]
    pi = C.derangement(0)
    g, gp = E.build_lake(12), E.build_lake(12, perm=pi)
    inv = np.argsort(pi)
    Xp = X[:, pi]                          # lead c of the relabelled data carries original lead pi[c]
    hidp = sorted(int(inv[h]) for h in hid)
    Fa, Oa, _ = E.lake_states(g, X, hid); Fb, Ob, _ = E.lake_states(gp, Xp, hidp)
    assert np.allclose(Fa, Fb, atol=1e-10) and np.allclose(Oa[:, pi], Ob, atol=1e-10)
    Ma = np.zeros((64, 22), bool); Ma[:, hid] = True; Mb = Ma[:, pi]
    E.S.ep_step(g, E.drive_rows(g, X, Ma), E.target(g, X), E.ETA)
    E.S.ep_step(gp, E.drive_rows(gp, Xp, Mb), E.target(gp, Xp), E.ETA)
    Fa, _, _ = E.lake_states(g, X, hid); Fb, _, _ = E.lake_states(gp, Xp, hidp)
    assert np.allclose(Fa, Fb, atol=1e-8)
