import numpy as np
import pytest
import torch

from src.mattr_faithful.decisions import (
    ARMS, N, classify, confirmation_keys, choose_arm, loss_bins, paired_cmd,
    pilot_decision, reference_choice, relative_difference, stage1_keys,
)
from src.mattr_faithful.objectives import objective
from src.mattr_optimization.common import configure, GRID


@pytest.mark.parametrize("delta,effects,ci,valid,expected", [
    (-.002, [-.001] * 5, [-.004, -.001], False, "Invalid"),
    (-.002, [-.002] * 4 + [.001], [-.004, -.001], True, "Lower CMD"),
    (.002, [.002] * 5, [.001, .004], True, "Higher CMD"),
    (.001, [.001] * 5, [-.004, .004], True, "Comparable"),
    (-.001, [-.001] * 3 + [.001] * 2, [-.006, -.0001], True, "Inconclusive"),
    (.001, [.001] * 5, [-.005, .004], True, "Inconclusive"),
])
def test_classification_precedence(delta, effects, ci, valid, expected):
    assert classify(delta, effects, ci, valid) == expected


def test_relative_cmd_with_zero_reference():
    assert relative_difference(-.002, .030) == pytest.approx(-6.6666666667)
    assert relative_difference(0, 0) is None
    assert relative_difference(.002, 1e-10) is None


def test_pilot_uses_only_registered_windows_and_inclusive_cmd_limit():
    losses = np.r_[np.ones(500) * 2, np.ones(4000) * 99, np.ones(500)]
    passed = pilot_decision(losses, 3 * .03, .03, True)
    assert passed["decision"] == "continue"
    assert (passed["first_500_mean"], passed["last_500_mean"]) == (2, 1)
    losses[-500:] = 2
    assert pilot_decision(losses, .09, .03, True)["decision"] == "stop"


def test_pilot_stops_on_nonfinite_in_middle():
    losses = np.linspace(2, 1, 5000)
    losses[2000] = np.inf
    result = pilot_decision(losses, .01, .03, True)
    assert result["decision"] == "stop" and not result["checks"]["all_5000_losses_finite"]


def test_sparsity_bins_edges_and_empty():
    rows = loss_bins([0, .01 * N, np.nextafter(.1 * N, 0), .1 * N, N], [1, 2, 4, 6, 8])
    assert [x["count"] for x in rows] == [1, 2, 2]
    assert [x["mean"] for x in rows] == [1, 3, 7]
    assert loss_bins([0], [1])[-1]["mean"] is None


def test_confirmation_has_exact_five_raw_absolute_rankings():
    assert {k for k in confirmation_keys("K1") if k.startswith("mattr_abs")} == {
        f"mattr_abs_s{seed}_5000" for seed in range(42, 47)}
    assert "K1_s46_5000" in confirmation_keys("K1")
    assert "K0_s46_5000" not in confirmation_keys("K1")


def test_selection_tie_breaks():
    assert reference_choice(dict(eapig_abs=.031, eapig_signed=.03)) == "eapig_signed"
    assert reference_choice(dict(eapig_abs=.03 + 1e-9, eapig_signed=.03)) == "eapig_abs"
    scores = {key: [.03] * 3 for key in ARMS}
    scores["K1"] = [.0291] * 3
    assert choose_arm(scores)["winner"] == "M0"
    scores["K1"] = [.0288] * 3
    assert choose_arm(scores)["winner"] == "K1"


def test_signed_and_absolute_export_exact_official_pruning():
    configure()
    from src.mattr_matched_size.topology import graph_new
    from src.mattr_faithful.preflight import audit_export, order
    graph = graph_new()
    raw = torch.arange(N, dtype=torch.float32) - 19000
    raw[-1] = raw[-2]  # Pinned default sort handles an actual tie.
    signed, absolute = order(graph, raw), order(graph, raw, True)
    assert signed[0] in (N-2, N-1) and absolute[0] == 0
    assert np.array_equal(order(graph, audit_export(graph, raw, True)), absolute)


def test_bootstrap_recomputes_cohort_normalizer_and_pairs_seed():
    clean = np.array([3., 2., 5., 2.])
    empty = np.array([0., 1., -1., 0.])
    errors = np.array([.2, .1, .5, .6])
    selected = np.stack([np.broadcast_to(clean-errors*s, (10, 4)) for s in (1,2,3,4,5)])
    comparator = np.broadcast_to(clean-2*errors, (1, 10, 4))
    factor = (GRID[-1]-GRID[0]) * errors.mean() / (clean-empty).mean()
    summary, indices = paired_cmd(selected, comparator, clean, empty, ["a","a","b","b"],
                                  [factor*np.arange(1,6), [2*factor]], draws=100)
    expected = (GRID[-1]-GRID[0]) * errors[indices].mean(1)/(clean-empty)[indices].mean(1)
    assert summary["delta"] == pytest.approx(factor)
    assert summary["interval"] == pytest.approx(np.quantile(expected,[.025,.975]))
    assert summary["unstable"] == 0


def test_reference_reuse_only_reads_verified_receipts(monkeypatch, tmp_path):
    from src.mattr_faithful import pipeline
    from src.mattr_optimization.common import sha
    records, touched = {}, []
    for key in stage1_keys():
        path = tmp_path / "evaluations/development" / key / "completed.json"
        path.parent.mkdir(parents=True)
        path.write_text(key)
        records[key] = sha(path)
    monkeypatch.setattr(pipeline, "reference", lambda _: {"artifact_sha256": records})

    def load(out, config, role, key, verify_masks):
        assert verify_masks and role == "development"
        touched.append(key)
        return {"cmd": .03}
    assert set(pipeline.stage1_reuse(tmp_path, {}, load)) == set(stage1_keys())
    assert touched == list(stage1_keys())
    path.write_text("changed")
    with pytest.raises(ValueError, match="receipt changed"):
        pipeline.stage1_reuse(tmp_path, {}, load)


def test_kl_formula_gradients_and_detached_targets():
    configure()
    logits = torch.tensor([[1.0, 2., 3.]], requires_grad=True)
    reference = torch.tensor([[2., 1., 3.]]).log_softmax(-1).requires_grad_()
    labels = torch.tensor([[0, 1]])
    result = objective("kl", logits, labels, {"logp": reference}, None)
    p = reference.detach().exp().numpy()
    q = logits.detach().softmax(-1).numpy()
    assert float(result.detach()) == pytest.approx(float(np.sum(p * np.log(p / q))), abs=1e-7)
    result.backward()
    np.testing.assert_allclose(logits.grad.numpy(), q-p, atol=1e-7)
    assert reference.grad is None
    full = torch.tensor([[1., 2., 3.]], requires_grad=True)
    target = torch.tensor([-1.], requires_grad=True)
    loss = objective("margin", full, labels, {"margin": target}, 2.)
    assert float(loss.detach()) == 0
    loss.backward()
    assert target.grad is None


def test_target_mmap_reads_only_loss_specific_rows(tmp_path):
    from src.mattr_faithful.targets import Targets
    np.save(tmp_path / "margin.npy", np.array([1., 2., 3.], dtype=np.float32))
    targets = Targets(tmp_path, "margin")
    assert list(targets.arrays) == ["margin"]
    assert isinstance(targets.arrays["margin"], np.memmap)
    value = targets.row(1, "cpu")["margin"]
    assert value.shape == (1,) and value.item() == 2. and not value.requires_grad
    assert targets.reads == 1
