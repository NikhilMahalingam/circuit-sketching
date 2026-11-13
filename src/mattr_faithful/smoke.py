"""Discovery-only GPU checks; the full-pool target build is manual."""

import random
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.mattr_data_scaling.training import learn
from src.mattr_fidelity_splice.inference import Recorder
from src.mattr_fidelity_splice.core import read, pinned_extra_sources
from src.mattr_gate5g_addendum.common import freeze_json, checked_json
from src.mattr_hierarchy.adapter import Hierarchy
from src.mattr_optimization.common import provenance, sha
from src.mattr_optimization.gate0 import check_patching
from .core import GATE5G, head
from .objectives import objective
from .preflight import prepare
from .training import training_model, with_logits


def bounded(out):
    graph = prepare(out)
    path = out / "bounded.json"
    if path.exists():
        result = checked_json(path)
        if result["source_sha256"] == smoke_sources():
            return result
        # Before freeze, preserve superseded smoke evidence and verify the new source.
        old = out / "attempts" / "bounded_gpu" / ("superseded_" + sha(path)[:12])
        old.mkdir(parents=True, exist_ok=True)
        path.replace(old / path.name)
        path.with_suffix(".sha256").replace(old / "bounded.sha256")
    discovery = pd.read_csv(out / "discovery.csv", keep_default_na=False)
    dev = pd.read_csv(out / "development.csv")
    confirmation = pd.read_csv(out / "confirmation.csv")
    excluded = set(dev.clean_sha256) | set(dev.corrupted_sha256) | set(confirmation.clean_sha256) | set(confirmation.corrupted_sha256)
    full = pd.read_csv(out / "stream_n10000_s42.csv", float_precision="round_trip")
    check_rows = np.random.default_rng(6110).choice(len(discovery), 50, replace=False)
    rows = list(dict.fromkeys([*full.row.iloc[:30], *check_rows]))
    frame = discovery.iloc[rows].reset_index(drop=True)
    assert not (set(frame.clean_sha256) | set(frame.corrupted_sha256)) & excluded
    plan = full.iloc[:30].copy()
    plan["row"] = [rows.index(i) for i in plan.row]
    results = {}
    model_info = read(GATE5G / "gate5g_protocol.json")["model"]

    def checks(original, loss_fn, env, ds, meter, work, kwargs):
        from eap.evaluate import evaluate_baseline, evaluate_graph
        from MIB_circuit_track.evaluation import evaluate_area_under_curve
        from src.mattr_matched_size.topology import graph_new
        from src.mattr_fidelity_splice.statistics import official_areas
        from src.mattr_hierarchy.runner import tensor_state_equal
        from src.mattr_optimization.common import ROOT

        model = meter.model
        with meter.stage("bounded_forward_checks"), torch.no_grad():
            check_patching(loss_fn, SimpleNamespace(datasets={"train": ds}), model, work)
            ref = Recorder(model, frame, save_logp=True)
            loader = ds.to_dataloader(20)
            evaluate_baseline(model, loader, ref, quiet=True)
            targets = ref.arrays()
            empty_graph = graph_new()
            empty_graph.apply_topn(0, absolute=False, prune=True)
            empty = evaluate_graph(model, empty_graph, loader, ref.metric, quiet=True).numpy()
            denominator = float(targets["margin"].mean() - empty.mean())
            max_margin = max_kl = max_empty = max_cache = 0.
            for row in check_rows:
                i = rows.index(row)
                captured = []
                run = model.run_with_hooks

                def capture(*args, **kw):
                    value = run(*args, **kw)
                    captured.append(value[:, -1, :])
                    return value

                with patch.object(model, "run_with_hooks", capture), patch.object(random, "randint", return_value=i):
                    loss_fn(torch.ones(32491, device="cuda"))
                    empty_loss = loss_fn(torch.zeros(32491, device="cuda"))
                logits = captured[-2]
                label = torch.tensor([[frame.iloc[i].correct_token, frame.iloc[i].incorrect_token]], device="cuda")
                row_target = {key: torch.as_tensor(targets[key][i:i+1], device="cuda") for key in ("logp", "margin")}
                clean_tokens, attention_mask, *_ = env["pretok"][i]
                independent = model(clean_tokens, attention_mask=attention_mask)[:, -1].float().log_softmax(-1)
                error = float((independent - row_target["logp"]).abs().max())
                max_cache = max(max_cache, error)
                torch.testing.assert_close(independent, row_target["logp"], atol=1e-4, rtol=2e-5)
                max_margin = max(max_margin, float(objective("margin", logits, label, row_target, denominator)))
                max_kl = max(max_kl, abs(float(objective("kl", logits, label, row_target, denominator))))
                max_empty = max(max_empty, abs(float(-empty_loss) - empty[i]))
                np.testing.assert_allclose(float(-empty_loss), empty[i], atol=5e-5, rtol=1e-5)
            assert max_margin < 1e-7 and max_kl < 3e-6
            # Official hard ranking and separate per-prompt replay, with identical official pruning.
            mask_graph = graph_new()
            mask_graph.scores[mask_graph.real_edge_mask] = torch.arange(32491, dtype=torch.float32)
            oracle = evaluate_area_under_curve(model, mask_graph, loader, ref.metric, quiet=True, absolute=False)
            from src.mattr_optimization.common import GRID
            vectors = []
            for pct in GRID:
                mask_graph.apply_topn(int(pct * 32491), absolute=False, prune=True)
                vectors.append(evaluate_graph(model, mask_graph, loader, ref.metric, quiet=True).numpy())
            area = official_areas(np.asarray(vectors).mean(axis=1), targets["margin"].mean(), empty.mean())
            np.testing.assert_allclose(area, oracle[1:3], rtol=2e-6, atol=2e-6)
        from src.mattr_fidelity_splice.core import ArrayCache
        from src.mattr_fidelity_splice.inference import Engine
        from .evaluate import run_ranking
        with meter.stage("bounded_empty_cache"), torch.no_grad():
            cache = ArrayCache(work / "circuit_cache", dict(role="bounded_discovery"))
            engine = Engine(model, graph_new(), ds, frame, targets, cache, 20)
            engine.graph.apply_topn(0, absolute=False, prune=True)
            empty_arrays = engine.current()
        refs = dict(baseline=float(torch.from_numpy(targets["margin"]).mean()),
                    corrupted=float(torch.from_numpy(empty_arrays["margin"]).mean()))
        summary, _ = run_ranking(engine, refs, empty_arrays, torch.arange(32491, dtype=torch.float32), False, meter)
        np.testing.assert_allclose([summary["cpr"], summary["cmd"]], oracle[1:3], rtol=2e-6, atol=2e-6)
        circuit_seconds = next(row["seconds"] for row in meter.rows if row["stage"] == "official_retention_grid") / 10
        results.update(full_mask_margin_max=max_margin, full_mask_abs_kl_max=max_kl,
                       empty_margin_max_error=float(max_empty), cached_logp_max_error=max_cache,
                       full_mask_rows=50, oracle_cpr=oracle[1], oracle_cmd=oracle[2],
                       seconds_per_circuit_measured=circuit_seconds, measured_circuit_prompts=len(frame),
                       seconds_per_circuit_512=circuit_seconds * (512 / len(frame)))

        # Same rows and k values as the start of the historical full-data seed-42 run.
        hist = ROOT / graph["pairing"]["B0_s42_1000"]["checkpoint"]
        old = torch.load(hist, map_location="cpu", weights_only=False)["train_log"][:30]
        with meter.stage("historical_baseline_replay"):
            state = learn(original, 32491, loss_fn, Hierarchy(kind="bypass"), plan, target=30,
                          folder=work / "baseline", identity="bounded_replay", meter=meter, **kwargs)
        np.testing.assert_allclose([x[3] for x in state["train_log"]], [x[3] for x in old], atol=1e-4, rtol=2e-5)
        results["baseline_30_max_error"] = float(max(abs(x[3] - y[3]) for x, y in zip(state["train_log"], old)))
        for name, hierarchy in (("margin", False), ("kl", True), ("repo_cmd", False)):
            ad = Hierarchy(np.load(out / "H2_groups.npy"), device="cuda") if hierarchy else Hierarchy(kind="bypass")

            def callback(logits, labels, row):
                tar = {k: torch.as_tensor(v[row:row+1], device="cuda") for k, v in targets.items() if k in ("logp", "margin")}
                tar["empty"] = torch.as_tensor(empty[row:row+1], device="cuda")
                return objective(name, logits, labels, tar, denominator)

            apply = with_logits(loss_fn, model, env, callback)
            def fixed(scores):
                from matryoshka_attribution.masks import build_mask
                losses = []
                for row in plan.itertuples():
                    with torch.enable_grad(), patch.object(random, "randint", return_value=int(row.row)):
                        mask = build_mask(scores, float(row.k), "topk", T=.5, n_iters=30).mask
                        losses.append(float(apply(mask).detach()))
                return float(np.mean(losses))
            before_count = len(meter.grad_norms)
            with meter.stage("objective_before_" + name):
                before = fixed(torch.zeros(32491, device="cuda"))
            with meter.stage("bounded_" + name):
                state = learn(original, 32491, apply, ad, plan, target=30, folder=work / name,
                              identity=name, meter=meter, **kwargs)
            train_seconds = meter.rows[-1]["seconds"]
            with meter.stage("objective_after_" + name):
                after = fixed(state["effective_scores"].cuda())
            with meter.stage("bounded_resume_" + name):
                # Verify exact optimizer and RNG restoration with the same frozen streams.
                for stop in (15, 30):
                    replay = learn(original, 32491, apply, ad, plan, target=stop, folder=work / (name + "_resume"),
                                   identity=name, meter=meter, **kwargs)
                assert torch.equal(state["scores"], replay["scores"])
                assert tensor_state_equal(state["optimizer"], replay["optimizer"])
                assert tensor_state_equal(state["rng"], replay["rng"])
            norms = meter.grad_norms[before_count:]
            assert torch.count_nonzero(state["scores"]) and np.isfinite(norms).all() and max(norms) > 0
            # Upstream repo_cmd has a clamped-gap target, so only the two new matching
            # objectives are required to decrease in the prespecified smoke check.
            if name in ("margin", "kl") and not after < before:
                raise ValueError(f"{name}: mean over all 30 paired smoke queries did not decrease: {before}, {after}")
            results[name] = dict(first=before, final=after, score_variance=float(state["effective_scores"].var()),
                                 seconds_per_update=train_seconds / 30, exact_resume=True)

    with training_model(out, "bounded_gpu", frame, 42, 30, checks, model_info) as work:
        audit = read(work / "weight_audit.json")
        results.update(passed=True, model_weights_unchanged=audit["passed"], counts=audit["counts"],
                       confirmation_forwards=0, development_forwards=0,
                       source_sha256=smoke_sources(), evidence=work.relative_to(out).as_posix())
        freeze_json(path, results)
    return results


def smoke_sources():
    from .core import ROOT
    return {p.name: sha(p) for p in sorted((ROOT / "src/mattr_faithful").glob("*.py"))
            if p.name in ("smoke.py", "objectives.py", "training.py", "preflight.py")}


def manual_preflight(out):
    started = time.perf_counter()
    graph = prepare(out)
    smoke = bounded(out)
    from src.mattr_data_scaling.core import configure_runtime, load_model, close_meter
    from src.mattr_fidelity_splice.core import attempt
    from src.mattr_optimization.gate1 import LiveAccounting
    from src.mattr_hierarchy.runner import weight_hash
    from .targets import build, verified_targets
    configure_runtime()
    path = out / "preflight.json"
    if path.exists():
        record = checked_json(path)
        assert record["implementation_commit"] == head() and verified_targets(out / "targets")
        return record
    with attempt(out, "discovery_targets") as work:
        meter = LiveAccounting(work)
        try:
            model = load_model(meter, freeze=True)
            before = weight_hash(model)
            assert before == read(GATE5G / "gate5g_protocol.json")["model"]["weight_sha256"]
            targets = build(model, pd.read_csv(out / "discovery.csv", keep_default_na=False), out / "targets", meter)
            assert before == weight_hash(model) and all(p.grad is None for p in model.parameters())
        finally:
            close_meter(meter)
    record = dict(graph, passed=True, implementation_commit=head(), dependencies=provenance(),
                  pinned_extra=pinned_extra_sources(), targets=targets, bounded=smoke,
                  seconds=time.perf_counter()-started)
    for name in ("graph_preflight.json", "bounded.json", "targets/completed.json", "targets/metadata.json"):
        record["artifacts"][name] = sha(out / name)
    freeze_json(path, record)
    return record
