"""Official circuit evaluation with per-prompt scalars captured on the same pass."""

from contextlib import contextmanager
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.mattr_data_scaling.core import close_meter, configure_runtime, load_model
from src.mattr_faithful.evaluate import mask_at, run_ranking
from src.mattr_faithful.preflight import dataset
from src.mattr_fidelity_splice.core import ArrayCache, attempt, marker, read, verified
from src.mattr_fidelity_splice.inference import Engine, clean_reference, model_identity, position_identity
from src.mattr_gate5g_addendum.common import checked_json, freeze_json
from src.mattr_hierarchy.runner import weight_hash
from src.mattr_matched_size.topology import graph_new
from src.mattr_optimization.common import ROOT, sha, write_json
from src.mattr_optimization.gate1 import LiveAccounting
from .core import FILE, protocol, prepare


def source_hashes():
    directory = ROOT / "src/mattr_ensemble"
    return {name: sha(directory / name) for name in ("core.py", "evaluation.py")}


@contextmanager
def session(out, frame, split, context, expected, meter):
    with meter.stage("model_loading"):
        model = load_model(meter, freeze=True)
        actual = model_identity(model)
        for field in ("weight_sha256", "tokenizer_class", "vocab_sha256", "vocab_size", "precision", "position"):
            assert actual[field] == expected[field], field
        ds = dataset(model, frame, split)
        context = dict(context, model=actual, position=position_identity(model, ds, 20), batch=20)
    cache = out / "cache" / context["role"]
    clean_cache = ArrayCache(cache / "clean", context)
    circuit_cache = ArrayCache(cache / "circuits", context)
    with meter.stage("references"), torch.no_grad():
        ref = clean_reference(model, ds, frame, clean_cache, batch=20, validate=False)
        engine = Engine(model, graph_new(), ds, frame, ref, circuit_cache, 20)
        mask_at(engine.graph, 0, False)
        empty = engine.current()
    refs = dict(baseline=torch.from_numpy(ref["margin"]).mean().item(),
                corrupted=torch.from_numpy(empty["margin"]).mean().item())
    assert abs(refs["baseline"] - refs["corrupted"]) >= 1e-8
    yield engine, refs, empty
    assert weight_hash(model) == actual["weight_sha256"]
    assert all(not param.requires_grad and param.grad is None for param in model.parameters())
    assert meter.counts["optimizer_updates"] == meter.counts["backward_calls"] == 0


def captured_ranking(engine, refs, empty, scores, meter):
    points = []
    original = engine.current

    def capture():
        arrays = original()
        points.append({k: arrays[k] for k in ("margin", "kl", "top1_correct", "sign_agreement")})
        return arrays

    with patch.object(engine, "current", capture):
        summary, vectors = run_ranking(engine, refs, empty, scores, False, meter)
    for point, arrays in zip(summary["points"], points, strict=True):
        prefix = f"{point['selection']}_{point['target']}"
        vectors[prefix + "_kl"] = arrays["kl"]
        vectors[prefix + "_correct"] = arrays["margin"] > 0
        vectors[prefix + "_top1_correct"] = arrays["top1_correct"]
        vectors[prefix + "_sign_agreement"] = arrays["sign_agreement"]
        np.testing.assert_allclose(vectors[prefix + "_kl"].mean(), point["kl"], atol=1e-9, rtol=1e-8)
    return summary, vectors


def smoke(out):
    p = prepare(out)
    path = out / "smoke.json"
    if path.exists():
        old = checked_json(path)
        if old["source_sha256"] == source_hashes():
            assert old["prepared_sha256"] == sha(out / "prepared.json")
            return old
        raise ValueError("Smoke source changed; preserve this campaign and use a new --out")
    from src.mattr_faithful.core import DEFAULT as A
    from src.mattr_fidelity_splice.inference import full_kl
    from src.mattr_fidelity_splice.statistics import official_areas
    from MIB_circuit_track.evaluation import evaluate_area_under_curve
    from src.mattr_compact_validation.inference import assign_scores
    from src.mattr_faithful.evaluate import capped
    from src.mattr_optimization.common import GRID
    configure_runtime()
    frame = pd.read_csv(A / "discovery.csv", keep_default_na=False).iloc[:8].reset_index(drop=True)
    cohort = pd.read_csv(out / "cohort_hashes.csv")
    assert not (set(frame.clean_sha256) | set(frame.corrupted_sha256)) & (set(cohort.clean_sha256) | set(cohort.corrupted_sha256))
    prompts = frame[["official_filtered_index", "clean_sha256", "corrupted_sha256"]].to_dict("records")
    with attempt(out, "smoke") as work:
        meter = LiveAccounting(work)
        try:
            scores = torch.load(out / "rankings/m1_ens5.pt", weights_only=True)
            context = dict(role="smoke_discovery", rows=prompts)
            with session(out, frame, "train", context, p["model"], meter) as (engine, refs, empty):
                start = time.perf_counter()
                summary, arrays = captured_ranking(engine, refs, empty, scores, meter)
                seconds = time.perf_counter() - start
                before = meter.counts["forward_batches"]
                duplicate, again = captured_ranking(engine, refs, empty, scores, meter)
                np.testing.assert_array_equal(arrays["cap_116_kl"], again["cap_116_kl"])
                assert meter.counts["forward_batches"] == before and duplicate["cache_misses"] == 0
                with meter.stage("official_oracles"), torch.no_grad():
                    assign_scores(engine.graph, scores)
                    official = evaluate_area_under_curve(engine.model, engine.graph, engine.loader, engine.metric,
                                quiet=True, level="edge", absolute=False, intervention="patching", refs=refs)
                    np.testing.assert_allclose(official[1:3], [summary["cpr"], summary["cmd"]], rtol=1e-8, atol=1e-8)
                    values = official_areas(arrays["grid_margins"].astype(float).mean(1), arrays["clean"].astype(float).mean(),
                                           arrays["empty"].astype(float).mean())
                    np.testing.assert_allclose(values, official[1:3], rtol=2e-6, atol=2e-6)
                    capped(engine.graph, 116, False)

                    def oracle_metric(logits, clean, lengths, labels):
                        from MIB_circuit_track.metrics import get_logit_positions
                        return full_kl(engine.reference["logp"], get_logit_positions(logits, lengths))

                    kl = engine.raw(engine.model, engine.graph, engine.loader, oracle_metric, quiet=True, intervention="patching")
                    np.testing.assert_allclose(kl.numpy(), arrays["cap_116_kl"], rtol=1e-7, atol=1e-7)
            record = dict(passed=True, prepared_sha256=sha(out / "prepared.json"), source_sha256=source_hashes(),
                          prompts=prompts, fresh_forwards=0, smoke_rank_seconds=seconds,
                          rank_forwards=before, counts=meter.counts, capped_116_kl_reused=True,
                          official_cmd_cpr_equivalent=True, weights_unchanged=True)
        finally:
            close_meter(meter)
        freeze_json(path, record)
    return record


def ident(out, p, key):
    return dict(key=key, protocol_sha256=sha(out / FILE), implementation_commit=p["implementation_commit"],
                ranking_sha256=p["artifacts"][f"rankings/{key}.pt"], cohort_sha256=p["artifacts"]["cohort.csv"],
                source_status="")


def result(out, p, key):
    folder = out / "evaluations" / key
    return read(folder / "metrics.json") if verified(folder, ident(out, p, key)) else None


def evaluate(out, only=None, approved=False):
    p = protocol(out, live=True)
    if only is not None and only not in p["jobs"]:
        raise ValueError(f"Unregistered ranking {only}")
    with attempt(out, "evaluation_session"):
        for key in [only] if only else p["jobs"]:
            if result(out, p, key) is not None:
                print("Verified result", key, flush=True)
                continue
            completed = [r for name in p["jobs"] if (r := result(out, p, name)) is not None]
            if completed:
                attempts = [read(path) for path in (out / "attempts").glob("rank_*/*/attempt.json")]
                spent = sum(a.get("seconds", 0.) for a in attempts)
                projection = spent + (len(p["jobs"]) - len(completed)) * max(r["seconds"] for r in completed)
                write_json(out / "runtime_projection.json", dict(projected_seconds=projection, spent_seconds=spent,
                           approved=approved, protocol_sha256=sha(out / FILE)))
                if projection > p["runtime_limit_seconds"] and not approved:
                    raise RuntimeError("Projected runtime exceeds two hours. Review runtime_projection.json and explicitly pass --approve-over-2h.")
            protocol(out, live=True)
            configure_runtime()
            frame = pd.read_csv(out / "cohort.csv", keep_default_na=False)
            identity = ident(out, p, key)
            folder = out / "evaluations" / key
            folder.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            with attempt(out, "rank_" + key) as work:
                write_json(work / "configuration.json", identity)
                meter = LiveAccounting(work)
                try:
                    context = dict(role="frozen_validation", protocol=identity["protocol_sha256"], cohort=identity["cohort_sha256"])
                    scores = torch.load(out / "rankings" / f"{key}.pt", weights_only=True)
                    with session(out, frame, "validation", context, p["model"], meter) as (engine, refs, empty):
                        summary, vectors = captured_ranking(engine, refs, empty, scores, meter)
                    summary.update(identity=identity, seconds=time.perf_counter()-start, counts=meter.counts,
                                   model_weights_unchanged=True)
                    write_json(folder / "metrics.json", summary)
                    np.savez_compressed(folder / "vectors.npz", **vectors)
                finally:
                    close_meter(meter)
                marker(folder, identity, [folder / "metrics.json", folder / "vectors.npz"])
