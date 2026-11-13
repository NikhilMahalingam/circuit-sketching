"""Frozen official evaluator with per-prompt read-only recording and exact mask reuse."""

from contextlib import contextmanager
from functools import partial
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.mattr_compact_validation.inference import assign_scores
from src.mattr_data_scaling.core import configure_runtime, load_model, close_meter
from src.mattr_fidelity_splice.core import ArrayCache, attempt, marker, read, verified
from src.mattr_fidelity_splice.inference import clean_reference, Engine, model_identity, position_identity, summarize
from src.mattr_matched_size.topology import graph_new, digest
from src.mattr_optimization.common import GRID, integration_weights, sha, write_json
from src.mattr_hierarchy.runner import weight_hash
from src.mattr_optimization.gate1 import LiveAccounting
from .core import CONFIRMATION, TRAINING, protocol
from .decisions import CAPS, N
from .preflight import dataset


def rank(out, key, p):
    path = out / "rankings" / f"{key}.pt"
    if f"rankings/{key}.pt" in p["artifacts"]:
        expected = p["artifacts"][f"rankings/{key}.pt"]
    else:
        folder = out / "training" / (key.rsplit("_", 1)[0] + "_5000")
        if not verified(folder):
            raise ValueError(f"Missing completed training for {key}")
        expected = read(folder / "rankings.json")[key]["ranking_sha256"]
    assert sha(path) == expected
    scores = torch.load(path, map_location="cpu", weights_only=True)
    assert scores.shape == (N,) and bool(torch.isfinite(scores).all())
    return scores, expected


def mask_at(graph, n, absolute):
    graph.apply_topn(n, absolute=absolute, level="edge", prune=True)
    return graph.in_graph[graph.real_edge_mask.bool()].numpy().copy()


def capped(graph, cap, absolute):
    # Adding ranked edges cannot destroy a connected input-to-output path.
    lower, upper = 0, N
    while upper - lower > 1:
        mid = (lower + upper) // 2
        if mask_at(graph, mid, absolute).sum() <= cap:
            lower = mid
        else:
            upper = mid
    after = mask_at(graph, upper, absolute)
    current = mask_at(graph, lower, absolute)
    assert current.sum() <= cap < after.sum() and not np.any(current & ~after)
    return lower, current


def evaluation_identity(out, p, role, key, ranking_sha):
    return dict(role=role, key=key, training_sha256=sha(out / TRAINING),
                    implementation_commit=p["implementation_commit"], source_status=p["source_status"],
                    ranking_sha256=ranking_sha,
                    cohort_sha256=sha(out / f"{role}.csv"), configuration=p["evaluator"],
                    confirmation_sha256=sha(out / CONFIRMATION) if role in ("confirmation", "public") else None,
                    absolute=key.startswith("mattr_abs_"))


def check_result(out, p, role, key, *, verify_masks=False):
    folder = out / "evaluations" / role / key
    if not (folder / "completed.json").exists():
        return None
    scores, ranking_sha = rank(out, key, p)
    ident = evaluation_identity(out, p, role, key, ranking_sha)
    assert verified(folder, ident)
    summary = read(folder / "metrics.json")
    if verify_masks:
        graph = graph_new()
        assign_scores(graph, scores)
        assert len(summary["points"]) == len(GRID) + len(CAPS)
        for point in summary["points"]:
            mask = mask_at(graph, point["nominal_edges"], ident["absolute"])
            assert digest(mask) == point["mask_sha256"] and int(mask.sum()) == point["actual_edges"]
            if point["selection"] == "cap":
                assert capped(graph, point["target"], ident["absolute"])[0] == point["nominal_edges"]
            else:
                assert point["nominal_edges"] == int(point["target"] * N)
    return summary


@contextmanager
def engine_session(out, role, frame, meter, expected_model, p):
    with meter.stage("model_loading_and_manifest"):
        model = load_model(meter, freeze=True)
        actual = model_identity(model)
        for field in ("weight_sha256", "tokenizer_class", "vocab_sha256", "vocab_size", "precision", "position"):
            assert actual[field] == expected_model[field], field
        before = weight_hash(model)
        ds = dataset(model, frame, "test" if role == "public" else "validation")
        positions = position_identity(model, ds, p["settings"]["evaluation_batch"])
    context = dict(training_sha256=sha(out / "gate6a_training_protocol.json"), model=actual,
                   role=role, cohort_sha256=sha(out / f"{role}.csv"), positions=positions,
                   evaluator=p["evaluator"], batch=p["settings"]["evaluation_batch"])
    cache = ArrayCache(out / "cache" / role / "circuits", context)
    batch = p["settings"]["evaluation_batch"]
    with meter.stage("cohort_references"), torch.no_grad():
        ref = clean_reference(model, ds, frame, ArrayCache(out / "cache" / role / "clean", context),
                              batch=batch, validate=False)
        engine = Engine(model, graph_new(), ds, frame, ref, cache, batch)
        mask_at(engine.graph, 0, False)
        empty = engine.current()
    refs = dict(baseline=float(torch.from_numpy(ref["margin"]).mean()),
                corrupted=float(torch.from_numpy(empty["margin"]).mean()))
    assert abs(refs["baseline"] - refs["corrupted"]) > 1e-8
    yield engine, refs, empty
    assert weight_hash(model) == before and all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert meter.counts["optimizer_updates"] == meter.counts["backward_calls"] == 0


def evaluate(out, role, key):
    p = protocol(out, live=True)
    from .pipeline import require_evaluation
    require_evaluation(out, p, role, key)
    cached = check_result(out, p, role, key, verify_masks=True)
    if cached is not None:
        return cached
    scores, ranking_sha = rank(out, key, p)
    ident = evaluation_identity(out, p, role, key, ranking_sha)
    folder = out / "evaluations" / role / key
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(out / f"{role}.csv", keep_default_na=False)
    expected_model = read(out / "preflight.json")["targets"]["target_identity"]["model"]
    configure_runtime()
    with attempt(out, role + "_" + key) as work:
        write_json(work / "configuration.json", ident)
        meter = LiveAccounting(work)
        try:
            with engine_session(out, role, frame, meter, expected_model, p) as (engine, refs, empty):
                summary, arrays = run_ranking(engine, refs, empty, scores, ident["absolute"], meter)
            summary["accounting"] = meter.counts
            summary["model_weights_unchanged"] = True
            summary["identity"] = ident
            write_json(folder / "metrics.json", summary)
            np.savez_compressed(folder / "vectors.npz", **arrays)
            marker(folder, ident, [folder / "metrics.json", folder / "vectors.npz"])
        finally:
            close_meter(meter)
    return summary


def run_ranking(engine, refs, empty, scores, absolute, meter):
    import MIB_circuit_track.evaluation as official

    graph = engine.graph
    assign_scores(graph, scores)
    points, vectors = [], {}
    margin_vectors = []
    cache_before = (engine.cache.hits, engine.cache.misses)

    def collect(selection, target, n, current):
        arrays = engine.current()
        vectors[f"{selection}_{target}_margin"] = arrays["margin"]
        clean = engine.reference["margin"]
        eligible = np.abs(clean) >= .5
        vectors[f"{selection}_{target}_ratio"] = (arrays["margin"][eligible] / clean[eligible])
        points.append(dict(selection=selection, target=target, nominal_edges=n, actual_edges=int(current.sum()),
                           mask_sha256=digest(current), **summarize(arrays, engine.reference)))
        return arrays

    def oracle_graph(model, current_graph, loader, metric, **kwargs):
        assert model is engine.model and current_graph is graph and loader is engine.loader and metric is engine.metric
        assert kwargs["intervention"] == "patching"
        i = len(margin_vectors)
        n = int(GRID[i] * N)
        current = graph.in_graph[graph.real_edge_mask.bool()].numpy().copy()
        arrays = collect("grid", GRID[i], n, current)
        margin_vectors.append(arrays["margin"])
        return torch.from_numpy(arrays["margin"].copy())

    with meter.stage("official_retention_grid"), torch.no_grad(), patch.object(official, "evaluate_graph", oracle_graph):
        result = official.evaluate_area_under_curve(engine.model, graph, engine.loader, engine.metric,
                 quiet=True, level="edge", absolute=absolute, intervention="patching", refs=refs)
    _, cpr, cmd, _, faith, accuracies, acc_auc = result
    for point, f, a, w in zip(points, faith, accuracies, integration_weights(), strict=True):
        point.update(faithfulness=f, official_accuracy=a, cmd_contribution=w * abs(1-f))
    np.testing.assert_allclose(sum(point["cmd_contribution"] for point in points), cmd, atol=1e-10, rtol=1e-10)
    with meter.stage("actual_edge_caps"), torch.no_grad():
        for cap in CAPS:
            n, current = capped(graph, cap, absolute)
            collect("cap", cap, n, current)
    vectors.update(grid_margins=np.asarray(margin_vectors), clean=engine.reference["margin"], empty=empty["margin"])
    summary = dict(cpr=cpr, cmd=cmd, accuracy_log_auc=acc_auc, faithfulness_above_one=sum(f > 1 for f in faith),
                   refs=refs, points=points, cache_hits=engine.cache.hits-cache_before[0],
                   cache_misses=engine.cache.misses-cache_before[1],
                   raw_score_stats=dict(mean=float(scores.mean()), median=float(np.median(scores.numpy())),
                                        min=float(scores.min()), max=float(scores.max())) if absolute else None)
    return summary, vectors
