"""Train with the pinned MAttr edge closure, replacing only its scalar loss."""

from contextlib import contextmanager
import importlib.util
import inspect
import random
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.mattr_data_scaling.core import configure_runtime, close_meter
from src.mattr_data_scaling.training import learn
from src.mattr_fidelity_splice.core import attempt, marker, read, verified
from src.mattr_fidelity_splice.inference import model_identity
from src.mattr_hierarchy.adapter import Hierarchy
from src.mattr_hierarchy.runner import weight_hash
from src.mattr_optimization.common import MIB, UPSTREAM, sha, write_json
from src.mattr_optimization.gate1 import LiveAccounting
from .core import identity, protocol
from .decisions import ARMS, N, ranking_key
from .objectives import objective
from .preflight import audit_export, save_rank
from .targets import Targets, verified_targets


class Finished(Exception):
    """Exit before upstream enters its own evaluation phase."""


@contextmanager
def training_model(out, key, frame, seed, steps, callback, expected_model):
    configure_runtime()
    import matryoshka_attribution as package
    from transformer_lens import HookedTransformer
    from MIB_circuit_track.dataset import HFEAPDataset

    with attempt(out, key) as work:
        meter = LiveAccounting(work)
        old_model = HookedTransformer.from_pretrained
        old_data = HFEAPDataset.__init__
        old_backward = torch.Tensor.backward
        old_trainer = package.learn_scores
        state = SimpleNamespace()

        def model_load(*args, **kwargs):
            with meter.stage("model_loading"):
                model = meter.attach(old_model(*args, **kwargs))
                model.eval().requires_grad_(False)
                assert model.cfg.dtype == torch.float32
                state.weight = weight_hash(model)
                assert state.weight == expected_model["weight_sha256"]
                state.identity = model_identity(model)
                return model

        def data_load(ds, *args, **kwargs):
            assert kwargs["split"] == "train"
            old_data(ds, *args, **kwargs)
            ds.dataset = ds.dataset.select(frame.official_filtered_index.astype(int).tolist())
            for j, row in enumerate(frame.itertuples()):
                clean, donor, labels = ds[j]
                assert (clean, donor, list(labels)) == (row.clean_prompt, row.corrupted_prompt,
                                                      [row.correct_token, row.incorrect_token])
            state.dataset = ds

        def backward(tensor, *args, **kwargs):
            meter.counts["backward_calls"] += 1
            return old_backward(tensor, *args, **kwargs)

        def intercept(total, loss_fn, **kwargs):
            actual = state.identity
            for field in ("weight_sha256", "tokenizer_class", "vocab_sha256", "vocab_size", "precision", "position"):
                if actual[field] != expected_model[field]:
                    raise ValueError(f"Pinned model identity differs in {field}")
            assert total == N
            env = inspect.getclosurevars(loss_fn).nonlocals
            assert env["corrupt_topk"] is False and len(env["pretok"]) == len(frame)
            for name in ("steps", "on_step", "log_every"):
                kwargs.pop(name)
            callback(old_trainer, loss_fn, env, state.dataset, meter, work, kwargs)
            raise Finished

        argv = ["edge", "--mib-path", str(MIB), "--model", "gpt2", "--task", "ioi",
                "--steps", str(steps), "--k-schedule", "uniform", "--masking", "topk",
                "--mode", "iso", "--lr", ".05", "--T", ".5", "--n_iters", "30",
                "--optimizer", "adam", "--seed", str(seed), "--cf-cache-gb", "4",
                "--batch-size", "20", "--no-wandb", "--output", str(work / "unused_upstream")]
        spec = importlib.util.spec_from_file_location("gate6a_official_edge", UPSTREAM / "scripts/mib/eval_mib_edge.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            with patch.object(HookedTransformer, "from_pretrained", model_load), \
                 patch.object(HFEAPDataset, "__init__", data_load), \
                 patch.object(package, "learn_scores", intercept), \
                 patch.object(torch.Tensor, "backward", backward), patch.object(sys, "argv", argv):
                try:
                    module.main()
                except Finished:
                    pass
                else:
                    raise RuntimeError("Official training callback was not reached")
            after = weight_hash(meter.model)
            assert after == state.weight and all(not p.requires_grad and p.grad is None for p in meter.model.parameters())
            write_json(work / "weight_audit.json", dict(passed=True, before=state.weight, after=after, counts=meter.counts))
        finally:
            close_meter(meter)
        yield work


def with_logits(loss_fn, model, env, callback):
    """Observe the grad-tracked output of the original closure's final model call."""
    assert env["model"] is model
    original = model.run_with_hooks

    def loss(mask):
        logits = []
        index = []
        draw = random.randint

        def capture_row(a, b):
            value = draw(a, b)
            index.append(value)
            return value

        def capture_logits(*args, **kwargs):
            value = original(*args, **kwargs)
            if torch.is_grad_enabled():
                logits.append(value[:, -1, :])
            return value

        with patch.object(model, "run_with_hooks", capture_logits), patch.object(random, "randint", capture_row):
            baseline = loss_fn(mask)
        if baseline is None or len(index) != 1 or len(logits) != 1:
            raise RuntimeError("Pinned closure did not produce one differentiable IOI answer")
        label = torch.as_tensor(env["pretok"][index[0]][-1], device=mask.device, dtype=torch.long)[None, :]
        result = callback(logits[0], label, index[0])
        if not bool(torch.isfinite(result)):
            raise ValueError(f"Nonfinite loss for sampled row {index[0]}")
        return result
    return loss


def train(out, arm, seed, *, pilot=False):
    p = protocol(out, live=True)
    from .pipeline import require_training
    require_training(out, p, arm, seed, pilot=pilot)
    key = ranking_key(arm, seed)
    folder = out / "training" / key
    folder.mkdir(parents=True, exist_ok=True)
    name, hierarchy = ARMS[arm]
    stream = out / f"stream_n10000_s{seed}.csv"
    ident = identity(out, "training", key, stream_sha256=sha(stream), loss=name, hierarchy=hierarchy,
                     cache_sha256=sha(out / "targets/completed.json"))
    if verified(folder, ident):
        result = read(folder / "rankings.json")
        from .evaluate import rank
        for rkey in result:
            rank(out, rkey, p)
        return result
    assert verified_targets(out / "targets", full=False)
    targets = Targets(out / "targets", name)
    plan = pd.read_csv(stream, float_precision="round_trip")
    frame = pd.read_csv(out / "discovery.csv", keep_default_na=False)
    model_info = read(out / "preflight.json")["targets"]["target_identity"]["model"]
    outputs = {}

    def callback(original, loss_fn, env, ds, meter, work, kwargs):
        from src.mattr_data_scaling.training import checked_checkpoint
        from src.mattr_matched_size.topology import graph_new

        write_json(work / "configuration.json", ident)
        ad = Hierarchy(np.load(out / "H2_groups.npy"), device="cuda") if hierarchy else Hierarchy(kind="bypass")
        apply = with_logits(loss_fn, meter.model, env,
                            lambda logits, labels, row: objective(name, logits, labels, targets.row(row, logits.device),
                                                                  p["target_normalizer"]))
        # Retain each update's diagnostics separately, including interrupted-attempt compute.
        updates_path = work / "update_diagnostics.jsonl"
        original_update = meter.update
        existing = sorted(path for path in (folder / "checkpoints").glob("step_*.pt")
                          if path.with_suffix(".sha256").exists())
        offset = checked_checkpoint(existing[-1], ident)["completed"] if existing else 0
        elapsed_start = time.perf_counter()
        last = [elapsed_start]
        memory_path = folder / "checkpoint_memory.json"
        memory = read(memory_path) if memory_path.exists() else {}
        with updates_path.open("w", encoding="utf-8", buffering=1) as handle:
            def update(norm):
                if not np.isfinite(norm):
                    raise ValueError("Nonfinite score gradient")
                original_update(norm)
                now = time.perf_counter()
                step = offset + meter.counts["optimizer_updates"]
                handle.write(f'{{"step":{step},"norm":{norm:.17g},"seconds":{now-last[0]:.17g}}}\n')
                if step in (1000, 5000):
                    memory[str(step)] = dict(allocated=torch.cuda.memory_allocated(),
                                             reserved=torch.cuda.memory_reserved(),
                                             peak_allocated=torch.cuda.max_memory_allocated(),
                                             peak_reserved=torch.cuda.max_memory_reserved())
                    write_json(memory_path, memory)
                last[0] = now

            with meter.stage("training"), patch.object(meter, "update", update):
                state = learn(original, N, apply, ad, plan, target=5000, folder=folder / "checkpoints",
                              identity=ident, meter=meter, **kwargs)
        write_json(work / "rates.json", dict(updates=5000-offset,
                   training_seconds=time.perf_counter()-elapsed_start, cache_hits=targets.reads,
                   cache_misses=0, read_seconds=targets.read_seconds, transfer_seconds=targets.transfer_seconds,
                   donor_cache=env["cc_stats"]))
        rows = []
        for path in sorted((out / "attempts" / key).glob("*/update_diagnostics.jsonl"), key=lambda path: path.stat().st_mtime_ns):
            from json import loads
            rows.extend(loads(line) for line in path.read_text().splitlines() if line)
        diagnostics = pd.DataFrame(rows).drop_duplicates("step", keep="last").set_index("step").reindex(range(1, 5001))
        assert diagnostics.notna().all().all()
        result = plan.copy()
        result["loss"] = [row[3] for row in state["train_log"]]
        assert [row[1] for row in state["train_log"]] == plan.k.tolist()
        result["gradient_norm"] = diagnostics.norm.to_numpy()
        result["seconds"] = diagnostics.seconds.to_numpy()
        result.to_csv(folder / "steps.csv", index=False, float_format="%.17g")
        for step in (1000, 5000):
            checkpoint = folder / "checkpoints" / f"step_{step:06d}.pt"
            saved = checked_checkpoint(checkpoint, ident)
            effective = saved["effective_scores"]
            values = audit_export(graph_new(), effective)
            rkey = ranking_key(arm, seed, step)
            path = save_rank(out, rkey, values)
            raw = saved["scores"]
            outputs[rkey] = dict(ranking_sha256=sha(path), checkpoint_sha256=sha(checkpoint),
                                 effective_score_variance=float(effective.var()),
                                 raw_score_variance=float(raw.var()), group_variance=None, residual_variance=None,
                                 checkpoint_cuda_memory=memory[str(step)])
            if hierarchy:
                cpu = Hierarchy(ad.ids.cpu())
                _, centered = cpu.parts(raw)
                h = raw[:ad.group_count]
                outputs[rkey].update(group_variance=float(h.var()),
                                     residual_variance=float(centered.var()))

    with training_model(out, key, frame, seed, 5000, callback, model_info) as work:
        write_json(folder / "rankings.json", outputs)
        write_json(folder / "weight_audit.json", read(work / "weight_audit.json"))
        marker(folder, ident, [folder / name for name in ("steps.csv", "rankings.json", "weight_audit.json")])
    return outputs
