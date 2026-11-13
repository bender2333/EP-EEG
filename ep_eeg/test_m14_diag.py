"""M14 diagnostic harness: C_skip topology and certificate, T-only probe. No EEG required.

Run: python -m pytest ep_eeg/test_m14_diag.py -q
"""
import ast
import csv
from pathlib import Path

import numpy as np
import pytest

from ep_eeg import m14 as M
from ep_eeg import m14_probe as P
from ep_eeg import m14_report as R


def edges(g):
    eff = np.asarray(g["lr"].brain.efficacy)
    tab = {}
    for i, (a, b) in enumerate(zip(g["conn"].pre, g["conn"].post)):
        key = (int(a), int(b))
        assert key not in tab
        tab[key] = (float(eff[i]), bool(g["ff"][i]), int(g["path"][i]), bool(g["pl"][i]))
    return tab


def crosses(g, left, right):
    L, Rgt = set(map(int, left)), set(map(int, right))
    return [(a, b) for a, b in edges(g) if (a in L and b in Rgt) or (a in Rgt and b in L)]


@pytest.fixture(scope="module")
def built():
    return {"C": M.build("C", 14600, eta=0.3), "C_skip": M.build("C_skip", 14600, eta=0.3),
            "C_nolake": M.build("C_nolake", 14600, eta=0.3)}


def test_c_has_no_pool_outlet(built):
    g = built["C"]
    assert g["pools"] is not None and len(g["pools"]) == 22 and len(g["cols"]) == 88 and len(g["out"]) == 4
    assert crosses(g, g["pools"], g["out"]) == []
    assert not np.any(g["path"] == M.PATHS["skip"])


def test_c_skip_pool_outlet_and_cert(built):
    g = built["C_skip"]
    pairs = crosses(g, g["pools"], g["out"])
    assert len(pairs) == 22 * 4 * 2
    want = {(int(p), int(o)) for p in g["pools"] for o in g["out"]}
    want |= {(b, a) for a, b in want}
    assert set(pairs) == want
    tab = edges(g)
    for a, b in pairs:
        w, ff, path, plastic = tab[(a, b)]
        assert path == M.PATHS["skip"] and ff and plastic
    pool = g["path"] == M.PATHS["pool"]
    assert pool.any() and not g["pl"][pool].any()
    cert = M.certificate(g)
    assert cert["certified"] is True and cert["adaptation"] is False
    assert "skip" in g["init_scale"]["pathway"]
    assert g["init_scale"]["budget"]["skip"] == M.BUDGET_INIT["skip"]
    assert g["init_scale"]["global"] <= 1.0
    assert g["init_scale"]["layer_max_before_global"]["outlet"] > g["init_scale"]["layer_max_before_global"]["column"]


def test_shared_edges_differ_only_by_recorded_global_scale(built):
    c, s = built["C"], built["C_skip"]
    tc, ts = edges(c), edges(s)
    assert set(tc) < set(ts)
    ratio = s["init_scale"]["global"] / c["init_scale"]["global"]
    for key, (w, ff, path, plastic) in tc.items():
        ws, ffs, paths, plastics = ts[key]
        assert ff is ffs and path == paths and plastic is plastics
        expect = w * ratio if ff else w
        assert ws == pytest.approx(expect, abs=1e-9)
    assert c["init_scale"]["global"] == pytest.approx(1.0)
    assert "skip" not in c["init_scale"]["pathway"]


def test_c_nolake_reads_pools(built):
    g = built["C_nolake"]
    assert len(g["cols"]) == 0 and g["pools"] is not None
    assert len(crosses(g, g["pools"], g["out"])) == 22 * 4 * 2
    assert not np.any(g["path"] == M.PATHS["skip"])


def test_c_skip_refuses_non_val_split():
    with pytest.raises(PermissionError, match="T session"):
        M.run_job(1, "C_skip", 0, 1, 0.3, split="gate")


def test_preregistered_report_does_not_adopt_c_skip():
    assert R.ARMS == M.PREREG_ARMS
    assert "C_skip" not in R.ARMS and "C_skip" in R.EXPLORATORY_ARMS
    assert all(a != "C_skip" and b != "C_skip" for a, b, _ in R.PAIRS)
    recs = {}
    for s in range(1, 10):
        recs[("C_skip", s, 0, 0.3)] = {"train": {"curve": [{"u": 3000, "acc": 0.99}]}}
        recs[("C", s, 0, 0.3)] = {"train": {"curve": [{"u": 3000, "acc": 0.40}]}}
    sel, _tab = R.select(recs)
    assert "C_skip" not in sel and sel["C"]["eta"] == 0.3 and sel["C"]["U"] == 3000


def test_exploratory_summary_skips_other_splits(tmp_path):
    keep = {"split": "val", "arm": "C_skip", "subject": 3, "seed": 0, "eta": 0.3, "U": 3000,
            "test": {"acc": 0.5, "kappa": 0.3, "settle_rate": 1.0, "rejected_trials": 0},
            "train": {"curve": [{"u": 3000, "acc": 0.5, "kappa": 0.3, "settle_rate": 1.0, "train_acc": 0.6, "rejected_trials": 0}]},
            "config": {"init_scale": {"global": 0.81}}}
    drop = dict(keep, split="gate", subject=4)
    (tmp_path / "keep.json").write_text(__import__("json").dumps(keep))
    (tmp_path / "drop.json").write_text(__import__("json").dumps(drop))
    rows = R.exploratory_summary(tmp_path)
    assert [r["subject"] for r in rows] == [3]
    assert rows[0]["kappa_val"] == pytest.approx(0.3) and rows[0]["init_global"] == pytest.approx(0.81)


def test_probe_source_has_no_e_loader():
    tree = ast.parse(Path(P.__file__).read_text())
    names = set()
    constants = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            constants.add(node.value)
    for bad in ("load_e_session", "load_subject_gate", "holdout", "EPEEG_GATE"):
        assert bad not in names and bad not in constants
    assert "val" in constants


def test_trial_mean_matches_frozen_aggregate():
    from eval.scoring import aggregate
    rng = np.random.default_rng(0)
    act = rng.normal(size=(20, 11, 4))
    ok = rng.random((20, 11)) > 0.2
    ok[0] = True
    pred, rej = aggregate(act, ok, fallback=0)
    assert not rej.any()
    assert np.array_equal(pred, P.trial_mean(act, ok).argmax(1))


def test_weight_file_matches_trainer():
    assert P.weight_file("C", 1, 0, 0.3, 3000) == M.OUT / "weights" / "C_A01_s0_eta0.3_T0.01" / "u3000.npz"
    assert P.weight_file("C_nolake", 9, 0, 0.3, 2000) == M.OUT / "weights" / "C_nolake_A09_s0_eta0.3_T0.01" / "u2000.npz"
    assert P.FROZEN_SELECTION["C"] == {"eta": 0.3, "U": 3000}
    assert P.FROZEN_SELECTION["C_nolake"] == {"eta": 0.3, "U": 2000}
    assert P.FROZEN_SELECTION["A"]["eta"] == 1.0 and P.FROZEN_SELECTION["A"]["U"] == 3000


def test_dev_probe_does_not_load_a_session(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("session loader called")
    monkeypatch.setattr(M, "load", boom)
    meta = P.run(["C", "C_nolake"], [0], seed=100, eta=0.3, U=1, dev=True, n_train=8, n_val=8, out_dir=tmp_path)
    assert meta["dev"] is True and meta["split"] == "val"
    rows = list(csv.DictReader((tmp_path / "probe.tsv").open(), delimiter="\t"))
    layers = {(r["arm"], r["layer"]) for r in rows}
    assert ("C", "pools") in layers and ("C", "columns") in layers and ("C", "outlets") in layers
    assert ("C_nolake", "pools") in layers and ("C_nolake", "outlets") in layers
    assert ("C_nolake", "columns") not in layers
    assert ("C", "readout") in layers
    for r in rows:
        assert r["dev"] == "True"
        for k in ("acc_val", "kappa_val", "settle_rate_val"):
            assert np.isfinite(float(r[k]))
    assert "dev=true" in (tmp_path / "probe.md").read_text()
