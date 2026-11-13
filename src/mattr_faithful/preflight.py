"""Read-only historical inputs and bounded discovery-only validation."""

import random
import time

import numpy as np
import pandas as pd
import torch

from src.mattr_data_scaling.design import official_rows
from src.mattr_fidelity_splice.core import check_files, read
from src.mattr_fidelity_splice.design import load_scores, positive_scores
from src.mattr_gate5g_addendum.common import freeze_json, checked_json
from src.mattr_matched_size.design import select_cohort
from src.mattr_matched_size.topology import graph_new, digest
from src.mattr_compact_validation.inference import assign_scores
from src.mattr_optimization.common import configure, sha, ROOT, GRID
from .core import HISTORY, GATE5G, ADDENDUM, TRAINING
from .decisions import SEEDS, N, ranking_key


def order(graph, scores, absolute=False):
    assign_scores(graph, scores)
    flat = graph.real_edge_mask.flatten().bool()
    slots = torch.full_like(flat, -1, dtype=torch.long)
    slots[flat] = torch.arange(N)
    values = graph.scores.flatten()
    values = (values.abs() if absolute else values).masked_fill(~flat, -torch.inf)
    return slots[torch.argsort(values, descending=True)][:N].numpy()


def audit_export(graph, raw, absolute=False, historical_masks=None):
    ordinal = positive_scores(order(graph, raw, absolute))
    assert torch.equal(ordinal.sort().values, torch.arange(1, N + 1, dtype=torch.float32))
    nominal = [int(f * N) for f in GRID]
    sizes = sorted(set([0, 1, *nominal, *np.random.default_rng(612).integers(2, N, 20)]))
    for n in sizes:
        assign_scores(graph, raw)
        graph.apply_topn(int(n), absolute=absolute, prune=True)
        mask = graph.in_graph[graph.real_edge_mask.bool()].numpy().copy()
        if historical_masks and n in historical_masks:
            assert digest(mask) == historical_masks[n]
        assign_scores(graph, ordinal)
        for mode in (False, True):
            graph.apply_topn(int(n), absolute=mode, prune=True)
            assert np.array_equal(mask, graph.in_graph[graph.real_edge_mask.bool()].numpy())
    return ordinal


def save_rank(out, key, values):
    path = out / "rankings" / (key + ".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert torch.equal(torch.load(path, weights_only=True), values)
    else:
        temp = path.with_suffix(".tmp")
        torch.save(values, temp)
        temp.replace(path)
    return path


def dataset(model, frame, split):
    from MIB_circuit_track.dataset import HFEAPDataset

    ds = HFEAPDataset("mib-bench/ioi", model.tokenizer, split=split, task="ioi", model_name="gpt2")
    ds.dataset = ds.dataset.select(frame.official_filtered_index.astype(int).tolist())
    for j, row in enumerate(frame.itertuples()):
        clean, donor, labels = ds[j]
        assert (clean, donor, list(labels)) == (row.clean_prompt, row.corrupted_prompt,
                                               [row.correct_token, row.incorrect_token])
    return ds


def prepare(out):
    """Tokenize manifests and audit historical artifacts without model construction."""
    configure()
    from transformers import AutoTokenizer
    from MIB_circuit_track.dataset import HFEAPDataset
    from src.mattr_optimization.gate4_init import edge_mapping
    from src.mattr_hierarchy.groups import construct
    from src.mattr_hierarchy.adapter import Hierarchy
    from src.mattr_data_scaling.evaluate import ranking

    out.mkdir(parents=True, exist_ok=True)
    if (out / TRAINING).exists():
        raise ValueError("Preflight inputs are already frozen")
    if (out / "graph_preflight.json").exists():
        record = checked_json(out / "graph_preflight.json")
        check_files(record["historical"])
        check_files(record["artifacts"], out)
        return record
    started = time.perf_counter()
    old = checked_json(GATE5G / "gate5g_protocol.json")
    previous = read(ADDENDUM / "confirmation_audit.json")
    gate5f = HISTORY / "gate5f_protocol.json"
    for file, audit in ((gate5f, HISTORY / "final_audit.json"),
                        (ADDENDUM / "stage2_protocol.json", ADDENDUM / "final_audit.json")):
        assert read(audit)["passed"]
        assert sha(file) == file.with_suffix(".sha256").read_text().strip()
    assert read(GATE5G / "final_audit.json")["passed"]
    history = {}

    def protect(path, expected=None):
        value = sha(path)
        if expected is not None and value != expected:
            raise ValueError(f"Historical artifact differs from its recorded hash: {path}")
        history[path.relative_to(ROOT).as_posix()] = value
        return path

    for path in (gate5f, HISTORY / "final_audit.json", GATE5G / "gate5g_protocol.json",
                 GATE5G / "final_audit.json", ADDENDUM / "stage2_protocol.json", ADDENDUM / "final_audit.json",
                 HISTORY / "known_ioi_heads.json"):
        protect(path)
    dev = pd.read_csv(protect(GATE5G / "cohort_manifest.csv", old["cohort_sha256"]), keep_default_na=False)
    assert len(dev) == 512 and dev.template_id.nunique() == 9 and old["cohort"]["seed"] == 5545
    full = read(gate5f)
    train_path = protect(HISTORY / "train_n10000.csv", full["artifact_sha256"]["train_n10000.csv"])
    train = pd.read_csv(train_path, keep_default_na=False)
    assert len(train) == 10000 and train.official_filtered_index.tolist() == list(range(10000))
    train.to_csv(out / "discovery.csv", index=False)
    dev.to_csv(out / "development.csv", index=False)
    seen, indices = set(), set()
    excluded = old["cohort"]["exclusions"] + [
        dict(path=(GATE5G / "cohort_manifest.csv").relative_to(ROOT).as_posix(), role="dev", sha256=old["cohort_sha256"]),
        dict(path=(ADDENDUM / "confirmation_cohort.csv").relative_to(ROOT).as_posix(), role="addendum", sha256=previous["sha256"])]
    for rec in excluded:
        path = protect(ROOT / rec["path"], rec["sha256"])
        if "historical_full" in rec["role"]:
            continue
        frame = pd.read_csv(path, keep_default_na=False)
        from src.mattr_compact_validation.design import text_hash
        for col in ("clean_prompt", "corrupted_prompt"):
            if col in frame:
                seen.update(text_hash(value) for value in frame[col])
        if "train" not in rec["role"] and "train" not in path.name and "official_filtered_index" in frame:
            indices.update(frame.official_filtered_index.astype(int))
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = HFEAPDataset("mib-bench/ioi", tokenizer, split="validation", task="ioi", model_name="gpt2")
    rows = official_rows(ds)
    confirm, eligible, quotas = select_cohort(rows, seen, indices, size=512, seed=5547)
    for row in confirm.itertuples():
        assert len(tokenizer(row.clean_prompt).input_ids) == len(tokenizer(row.corrupted_prompt).input_ids)
    confirm.insert(0, "example", range(len(confirm)))
    confirm.to_csv(out / "confirmation.csv", index=False)
    for role, frame in (("development", dev), ("confirmation", confirm)):
        frame[["official_filtered_index", "clean_sha256", "corrupted_sha256", "template_id"]].to_csv(
            out / f"{role}_hashes.csv", index=False)

    graph = graph_new()
    construct(graph, out)
    for name in ("edge_mapping.csv", "H2_groups.npy"):
        path = protect(HISTORY / name, full["artifact_sha256"][name])
        if name.endswith(".csv"):
            pd.testing.assert_frame_equal(edge_mapping(graph, graph), pd.read_csv(path, keep_default_na=False))
        else:
            np.testing.assert_array_equal(np.load(path), np.load(out / name))
    groups = Hierarchy(np.load(out / "H2_groups.npy"))
    inventory = pd.read_csv(protect(HISTORY / "checkpoint_inventory.csv"), keep_default_na=False)
    queries = pd.read_csv(protect(GATE5G / "queries.csv", old["artifacts"]["queries.csv"]))
    rankings = {}
    pairing = {}
    for seed in SEEDS:
        stream_name = f"stream_n10000_s{seed}.csv"
        plan = pd.read_csv(protect(HISTORY / stream_name, full["artifact_sha256"][stream_name]), float_precision="round_trip")
        rng = random.Random(seed)
        assert plan.row.tolist() == [rng.randint(0, 9999) for _ in range(5000)]
        assert plan.step.tolist() == list(range(1, 5001))
        plan.to_csv(out / stream_name, index=False, float_format="%.17g")
        for arm, method in (("B0", "baseline"), ("B1", "h2")):
            for step in (1000, 5000):
                key = ranking_key(arm, seed, step)
                _, path = ranking(HISTORY, method, 10000, seed, step)
                protect(path)
                job = next(j for j in full["training_jobs"] if j["key"] == f"{method}_n10000_s{seed}")
                checkpoint = path.parent / "checkpoints" / f"step_{step:06d}.pt"
                if checkpoint.exists():
                    check = protect(checkpoint.with_suffix(".sha256")).read_text().strip()
                else:
                    checkpoint = ROOT / job["endpoints"][str(step)]
                    check = inventory.loc[inventory.checkpoint == checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256"].item()
                state = torch.load(protect(checkpoint, check), map_location="cpu", weights_only=False)
                raw = torch.load(path, map_location="cpu", weights_only=True)
                assert state["completed"] == step and len(state["train_log"]) == step
                assert [line[1] for line in state["train_log"]] == plan.k.iloc[:step].tolist()
                if arm == "B1":
                    torch.testing.assert_close(groups.transform(state["scores"]), raw, atol=2e-6, rtol=1e-5)
                else:
                    assert torch.equal(state["scores"], raw)
                if not (path.parent / "checkpoints" / f"step_{step:06d}.pt").exists():
                    draws = pd.read_csv(protect(checkpoint.parent.parent / "training_example_draws.csv"))
                    assert draws.official_filtered_index.iloc[:step].tolist() == plan.row.iloc[:step].tolist()
                else:
                    from hashlib import sha256
                    prefix = plan.iloc[:step].to_csv(index=False, float_format="%.17g").encode()
                    assert state["stream_prefix_hash"] == sha256(prefix).hexdigest()
                pairing[key] = dict(verified_updates=step, row_rule="Gate 5F exact frozen stream",
                                    checkpoint=checkpoint.relative_to(ROOT).as_posix())
                rankings[key] = sha(save_rank(out, key, audit_export(graph, raw)))
                if arm == "B0" and step == 5000:
                    absolute_key = ranking_key("mattr_abs", seed)
                    rankings[absolute_key] = sha(save_rank(out, absolute_key, raw))
    eap = next(rec for rec in old["rankings"] if rec["key"] == "eap_ig_n10000")
    for name, value in eap["input_hashes"].items():
        protect(ROOT / name, value)
    scores = load_scores(eap)
    historical = {int(row.nominal_n): row.mask_sha256 for row in
                  queries[(queries.ranking == eap["key"]) & (queries.selection == "nominal")].itertuples()}
    assert len(historical) == len(GRID)
    for key, absolute in (("eapig_signed", False), ("eapig_abs", True)):
        values = audit_export(graph, scores, absolute, historical if not absolute else None)
        rankings[key] = sha(save_rank(out, key, values))
    files = [out / name for name in ("discovery.csv", "development.csv", "confirmation.csv",
             "development_hashes.csv", "confirmation_hashes.csv", "edge_mapping.csv", "H2_groups.npy")]
    files += [out / f"stream_n10000_s{seed}.csv" for seed in SEEDS]
    files += [out / "rankings" / f"{key}.pt" for key in rankings]
    record = dict(passed=True, historical=history, artifacts={p.relative_to(out).as_posix(): sha(p) for p in files},
                  rankings=rankings, cohort=dict(eligible=len(eligible), selected=len(confirm), quotas=quotas, seed=5547,
                  excluded=excluded, pre_freeze_confirmation_forwards=0), pairing=pairing,
                  eap_signed_historical_masks=len(historical), graph_forward_passes=0,
                  seconds=time.perf_counter()-started)
    freeze_json(out / "graph_preflight.json", record)
    return record
