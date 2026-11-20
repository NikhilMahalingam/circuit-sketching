"""Frozen inputs, ensemble construction and disjoint cohort selection."""

import shutil
import time

import numpy as np
import pandas as pd
import torch

from src.mattr_faithful import core as prior
from src.mattr_faithful.evaluate import check_result, mask_at, rank
from src.mattr_faithful.preflight import order, save_rank
from src.mattr_compact_validation.inference import assign_scores
from src.mattr_data_scaling.design import official_rows
from src.mattr_data_scaling.training import checked_checkpoint
from src.mattr_fidelity_splice.core import check_files, pinned_extra_sources, read
from src.mattr_gate5g_addendum.common import checked_json, freeze_json
from src.mattr_matched_size.design import select_cohort
from src.mattr_matched_size.topology import graph_new
from src.mattr_optimization.common import GRID, ROOT, provenance, sha
from src.mattr_faithful.decisions import CAPS, N

DEFAULT = ROOT / "results/mattr_ensemble/gate6b"
FILE = "gate6b_protocol.json"
MEMBERS = {name: [f"{arm}_s{seed}_5000" for seed in seeds] for name, arm, seeds in
           [("m1_ens5", "M1", range(42, 47)), ("m0_ens3", "M0", range(42, 45)),
            ("k0_ens3", "K0", range(42, 45))]}
JOBS = ["m1_ens5", "eapig_abs", *MEMBERS["m1_ens5"], "eapig_signed", "m0_ens3", "k0_ens3"]


def mean_order(orders):
    orders = np.asarray(orders)
    if orders.ndim != 2 or not len(orders) or not orders.shape[1]:
        raise ValueError("At least one complete ranking is required")
    indices = np.arange(orders.shape[1])
    if not np.all(np.sort(orders, axis=1) == indices):
        raise ValueError("Each member must be a permutation")
    # Sorting integer sums is exactly equivalent to sorting arithmetic means.
    return np.lexsort((indices, np.argsort(orders, axis=1).sum(axis=0)))


def ordinal(ranked):
    scores = torch.empty(len(ranked), dtype=torch.float32)
    scores[ranked] = torch.arange(len(ranked), 0, -1, dtype=torch.float32)
    return scores


def verify_prefixes(graph, ranked, scores, cutoffs):
    np.testing.assert_array_equal(scores.sort().values.numpy(), np.arange(1, N + 1))
    for n in cutoffs:
        expected = np.zeros(N, dtype=bool)
        expected[ranked[:n]] = True
        assign_scores(graph, scores)
        for absolute in (False, True):
            graph.apply_topn(n, absolute=absolute, level="edge", prune=False)
            np.testing.assert_array_equal(graph.in_graph[graph.real_edge_mask.bool()].numpy(), expected)
            actual = mask_at(graph, n, absolute)
            # Build the same prefix independently as a binary score vector, then prune officially.
            assign_scores(graph, torch.from_numpy(expected.astype(np.float32)))
            np.testing.assert_array_equal(actual, mask_at(graph, n, False))
            assign_scores(graph, scores)


def prepare(out):
    from transformers import AutoTokenizer
    from MIB_circuit_track.dataset import HFEAPDataset
    from src.mattr_faithful.pipeline import confirmation

    path = out / "prepared.json"
    if path.exists():
        record = checked_json(path)
        check_files(record["artifacts"], out)
        check_files(record["historical"])
        return record
    out.mkdir(parents=True, exist_ok=True)
    a = prior.DEFAULT
    p = prior.protocol(a)
    c = confirmation(a, p)
    audit = checked_json(a / "final_audit.json")
    assert audit["passed"] and audit["classification"] == "Inconclusive" and c["selected"] == "M1"
    assert audit["training_sha256"] == sha(a / prior.TRAINING)
    assert audit["confirmation_sha256"] == sha(a / prior.CONFIRMATION)
    check_files(audit["artifact_sha256"], a)
    check_files(p["sources"])
    assert provenance() == p["dependencies"] and pinned_extra_sources() == p["pinned_extra"]
    protected = dict(p["historical"])
    artifacts = {}

    def track(file):
        protected[file.relative_to(ROOT).as_posix()] = sha(file)
        return file

    for name in (prior.TRAINING, prior.CONFIRMATION, "preflight.json", "final_audit.json",
                 "graph_preflight.json", "confirmation.csv", "discovery.csv", "edge_mapping.csv"):
        track(a / name)
    graph = graph_new()
    assert len(graph.edges) == N
    from src.mattr_optimization.gate4_init import edge_mapping
    pd.testing.assert_frame_equal(edge_mapping(graph, graph), pd.read_csv(a / "edge_mapping.csv", keep_default_na=False))
    inputs, orders = {}, {}
    keys = [key for values in MEMBERS.values() for key in values] + ["eapig_abs", "eapig_signed"]
    cutoffs = sorted({*map(int, np.random.default_rng(612).choice(np.arange(1, N), 20, replace=False)), 1, N})
    started = time.perf_counter()
    for key in keys:
        scores, checksum = rank(a, key, p)
        assert checksum == c["rankings"][key]
        result = check_result(a, p, "confirmation", key, verify_masks=True)
        assert result and result["model_weights_unchanged"]
        track(a / "evaluations/confirmation" / key / "completed.json")
        for name in ("metrics.json", "vectors.npz"):
            track(a / "evaluations/confirmation" / key / name)
        track(a / "rankings" / f"{key}.pt")
        if not key.startswith("eapig"):
            folder = a / "training" / key
            receipt = read(track(folder / "completed.json"))
            checkpoint = track(folder / "checkpoints/step_005000.pt")
            track(checkpoint.with_suffix(".sha256"))
            state = checked_checkpoint(checkpoint, receipt["identity"])
            assert state["completed"] == 5000
            assert sha(checkpoint) == read(track(folder / "rankings.json"))[key]["checkpoint_sha256"]
            assert torch.equal(ordinal(order(graph, state["effective_scores"])), scores)
            weights = read(track(folder / "weight_audit.json"))
            assert weights["passed"] and weights["before"] == weights["after"] == p["dependencies"].get(
                "weight_sha256", read(a / "preflight.json")["targets"]["target_identity"]["model"]["weight_sha256"])
        inputs[key] = checksum
        orders[key] = order(graph, scores)
        if key in JOBS:
            saved = save_rank(out, key, scores)
            artifacts[saved.relative_to(out).as_posix()] = sha(saved)
    single = MEMBERS["m1_ens5"][0]
    ranked = mean_order([orders[single]])
    np.testing.assert_array_equal(ranked, orders[single])
    single_scores = torch.load(out / "rankings" / f"{single}.pt", weights_only=True)
    for n in cutoffs:
        assign_scores(graph, single_scores)
        expected = mask_at(graph, n, False)
        assign_scores(graph, ordinal(ranked))
        np.testing.assert_array_equal(expected, mask_at(graph, n, False))
    ensembles = {}
    for key, members in MEMBERS.items():
        selected = mean_order([orders[name] for name in members])
        scores = ordinal(selected)
        verify_prefixes(graph, selected, scores, cutoffs)
        saved = save_rank(out, key, scores)
        ordering = saved.with_suffix(".npy")
        np.save(ordering, selected, allow_pickle=False)
        for file in (saved, ordering):
            artifacts[file.relative_to(out).as_posix()] = sha(file)
        ensembles[key] = dict(members={name: inputs[name] for name in members},
                              score_sha256=sha(saved), ordering_sha256=sha(ordering))
    ensemble_seconds = time.perf_counter() - started
    old = checked_json(a / "graph_preflight.json")["cohort"]["excluded"]
    paths = [(ROOT / rec["path"], rec["sha256"], "train" in rec["role"] or "train" in rec["path"].split("/")[-1])
             for rec in old if "historical_full" not in rec["role"]]
    paths.extend([(a / "confirmation.csv", c["confirmation_manifest_sha256"], False),
                  (a / "discovery.csv", p["artifacts"]["discovery.csv"], True)])
    from src.mattr_compact_validation.design import text_hash
    seen, indices = set(), set()
    for file, checksum, training in paths:
        assert sha(track(file)) == checksum
        frame = pd.read_csv(file, keep_default_na=False)
        for column in ("clean_prompt", "corrupted_prompt"):
            seen.update(text_hash(s) for s in frame[column])
        if not training and "official_filtered_index" in frame:
            indices.update(frame.official_filtered_index.astype(int))
    tok = AutoTokenizer.from_pretrained("gpt2")
    ds = HFEAPDataset("mib-bench/ioi", tok, split="validation", task="ioi", model_name="gpt2")
    cohort, eligible, quotas = select_cohort(official_rows(ds), seen, indices, size=512, seed=5548)
    assert (cohort.clean_prompt.map(lambda s: len(tok(s).input_ids)) ==
            cohort.corrupted_prompt.map(lambda s: len(tok(s).input_ids))).all()
    cohort.to_csv(out / "cohort.csv", index=False)
    cohort[["official_filtered_index", "clean_sha256", "corrupted_sha256", "template_id"]].to_csv(
        out / "cohort_hashes.csv", index=False)
    for file in (a / "edge_mapping.csv", prior.HISTORY / "known_ioi_heads.json"):
        shutil.copyfile(track(file), out / file.name)
    for name in ("cohort.csv", "cohort_hashes.csv", "edge_mapping.csv", "known_ioi_heads.json"):
        artifacts[name] = sha(out / name)
    cost = pd.read_csv(track(a / "compute_accounting.csv"))
    historical_seconds = float(cost.loc[cost.job.isin(MEMBERS["m1_ens5"]), "seconds"].sum())
    record = dict(inputs=inputs, ensembles=ensembles, artifacts=artifacts, historical=protected,
                  eligible=len(eligible), selected=len(cohort), quotas=quotas, cutoffs=cutoffs,
                  historical_training_seconds=historical_seconds, preparation_seconds=ensemble_seconds,
                  model=read(a / "preflight.json")["targets"]["target_identity"]["model"], graph_forwards=0)
    freeze_json(path, record)
    return record


def protocol(out, live=False):
    p = checked_json(out / FILE)
    check_files(p["artifacts"], out)
    check_files(p["historical"])
    if live:
        if prior.head() != p["implementation_commit"] or prior.source_status():
            raise ValueError("Frozen implementation commit or clean source mismatch")
        if provenance() != p["dependencies"] or pinned_extra_sources() != p["pinned_extra"]:
            raise ValueError("Pinned dependencies changed")
    return p


def freeze(out):
    if (out / FILE).exists():
        return protocol(out, live=True)
    if prior.source_status():
        raise ValueError("Commit all experiment source before freezing")
    p = prepare(out)
    smoke = checked_json(out / "smoke.json")
    assert smoke["passed"] and smoke["prepared_sha256"] == sha(out / "prepared.json")
    from .evaluation import source_hashes
    assert smoke["source_sha256"] == source_hashes()
    p["artifacts"].update({name: sha(out / name) for name in ("prepared.json", "smoke.json")})
    assert not (out / "evaluations").exists()
    p.update(implementation_commit=prior.head(), dependencies=provenance(), pinned_extra=pinned_extra_sources(),
             jobs=JOBS, cohort_seed=5548, grid=GRID, caps=CAPS, batch=20, precision="float32",
             selection="positive ordinal with official signed top-n and connectivity pruning",
             metrics="official MIB linear CMD/CPR and log-retention accuracy AUC; cap uses largest eligible prefix",
             kl_points={"grid": [.1,.2,.5], "cap": [116]}, bootstrap={"draws": 5000, "seed": 5648, "denominator_epsilon": 1e-8},
             cmd_margin=.005, compact_threshold=0, runtime_limit_seconds=7200,
             classifications=["Invalid", "Two-goal success", "CMD non-inferior only", "Compact gain only", "Neither"],
             registered={"first":"m1_ens5","second":"eapig_abs","cap":116},
             descriptive=["paired M1 ensemble versus seed mean", "signed EAP-IG at every cap",
                          "M0 and K0 ensembles", "full-rank Spearman and cap-116 Jaccard"],
             output_schema={"metrics.json":"official metrics, points, mask hashes, counters, weight checks",
                            "vectors.npz":"aligned clean/empty/grid margins and per-point margin/correctness/KL"})
    freeze_json(out / FILE, p)
    return p
