from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.mattr_ensemble.analysis import auc, classify, paired, relative_difference
from src.mattr_ensemble.core import mean_order, ordinal
from src.mattr_ensemble.evaluation import captured_ranking, ident, result
from src.mattr_fidelity_splice.core import marker
from src.mattr_fidelity_splice.statistics import official_areas, resamples, weights
from src.mattr_optimization.common import GRID, write_json


def test_equal_rank_weight_and_tie_break():
    members = [[2, 1, 0, 3], [0, 2, 1, 3], [1, 2, 0, 3]]
    expected = [2, 1, 0, 3]
    np.testing.assert_array_equal(mean_order(members), expected)
    np.testing.assert_array_equal(mean_order(members[::-1]), expected)
    np.testing.assert_array_equal(mean_order([[2, 0, 1], [1, 0, 2]]), [0, 1, 2])
    np.testing.assert_array_equal(mean_order([members[0]]), members[0])
    with pytest.raises(ValueError):
        mean_order([[0, 0]])


def test_ordinals_include_last_edge():
    assert torch.equal(ordinal(np.array([2, 0, 1])), torch.tensor([2., 1., 3.]))


@pytest.mark.parametrize("upper,lower,valid,want", [
    (0, .1, False, "Invalid"), (float("nan"), .1, True, "Invalid"),
    (.0049, .1, True, "Two-goal success"), (.0049, 0, True, "CMD non-inferior only"),
    (.005, .1, True, "Compact gain only"), (.005, 0, True, "Neither"),
])
def test_joint_boundaries(upper, lower, valid, want):
    assert classify(upper, lower, valid) == want


def test_relative_zero_reference():
    assert relative_difference(.005, .025) == 20
    assert relative_difference(.005, 0) is None


def test_shared_draws_recompute_each_seed_cmd_then_average():
    clean, empty = np.array([2., 4., 3., 5.]), np.array([0., 1., -1., 0.])
    indices = resamples(["a","a","b","b"], 100, 5648)
    w = weights(indices, 4)
    np.testing.assert_array_equal(indices, resamples(["a","a","b","b"], 100, 5648))
    margin = np.array([.1,.4,.2,.5])
    a = np.tile(clean+margin, (10,1))
    b = np.tile(clean-margin, (10,1))
    e = np.tile(clean, (10,1))
    cm = [official_areas(x @ w.T, w @ clean, w @ empty)[1] for x in (a,b,e)]
    summary = paired(0., [.1,.1], cm[2], cm[:2])
    expected = -(GRID[-1]-GRID[0]) * margin[indices].mean(1)/(clean-empty)[indices].mean(1)
    np.testing.assert_allclose(summary["interval"], np.quantile(expected,[.025,.975]))
    assert summary["unstable"] == 0
    assert auc(np.ones((10,2))).tolist() == [1.,1.]


def test_capped_kl_capture_calls_circuit_once(monkeypatch):
    calls = []
    values = dict(margin=np.array([-1.,2.]), kl=np.array([.2,.1]), top1_correct=np.array([False,True]),
                  sign_agreement=np.array([True,True]))
    def current():
        calls.append(116)
        return values
    engine = SimpleNamespace(current=current)
    def official_wrapper(engine, *args):
        arrays = engine.current()
        return {"points":[dict(selection="cap", target=116, kl=arrays["kl"].mean())]}, {}
    monkeypatch.setattr("src.mattr_ensemble.evaluation.run_ranking", official_wrapper)
    _, arrays = captured_ranking(engine, None, None, None, None)
    assert calls == [116]
    np.testing.assert_array_equal(arrays["cap_116_kl"], values["kl"])
    np.testing.assert_array_equal(arrays["cap_116_correct"], [False,True])


def test_resume_requires_matching_identity_and_content(tmp_path):
    (tmp_path/"gate6b_protocol.json").write_text("{}")
    key = "m1_ens5"
    p = dict(implementation_commit="abc", artifacts={"cohort.csv":"c",f"rankings/{key}.pt":"r"})
    folder = tmp_path/"evaluations"/key
    folder.mkdir(parents=True)
    assert result(tmp_path,p,key) is None
    path = folder/"metrics.json"
    write_json(path, {"cmd":.03})
    marker(folder, ident(tmp_path,p,key), [path])
    assert result(tmp_path,p,key) == {"cmd":.03}
    with pytest.raises(ValueError, match="identity mismatch"):
        result(tmp_path,dict(p,implementation_commit="def"),key)
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        result(tmp_path,p,key)
