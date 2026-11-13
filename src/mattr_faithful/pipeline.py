"""Registered stage transitions and safe per-job orchestration."""

import time

import numpy as np
import pandas as pd

from src.mattr_fidelity_splice.core import read, verified, attempt
from src.mattr_gate5g_addendum.common import freeze_json, checked_json
from src.mattr_optimization.common import GRID, sha, write_json
from .core import CONFIRMATION, TRAINING, protocol, reference
from .decisions import (ARMS, SEEDS, REFERENCES, ranking_key, reference_choice, pilot_decision,
                        choose_arm, confirmation_keys, stage1_keys)


def linked(out, name):
    record = checked_json(out / f"{name}.json")
    if record["training_sha256"] != sha(out / TRAINING):
        raise ValueError(f"{name} protocol mismatch")
    return record


def freeze_record(out, filename, **record):
    record = dict(training_sha256=sha(out / TRAINING), **record)
    path = out / f"{filename}.json"
    if path.exists():
        if checked_json(path) != record:
            raise ValueError(f"Frozen {filename} differs on replay")
    else:
        freeze_json(path, record)
    return record


def selected(out):
    return linked(out, "selection")


def continued(out):
    reference(out)
    pilot = linked(out, "pilot_decision")
    if pilot["decision"] != "continue":
        raise ValueError(f"Pilot stopped: {pilot['failures']}")


def require_evaluation(out, p, role, key):
    if role == "development" and key in p["reference_jobs"]:
        return
    if role == "development" and key in p["development_jobs"]:
        reference(out)
        if key != ranking_key("M0", 42):
            continued(out)
        return
    if role in ("confirmation", "public"):
        c = confirmation(out, p)
        if role == "confirmation" and key in c["rankings"]:
            from .evaluate import rank
            if rank(out, key, p)[1] != c["rankings"][key]:
                raise ValueError(f"Frozen ranking changed: {key}")
            return
        if role == "public":
            public = linked(out, "public_authorization")
            if public["confirmation_sha256"] == sha(out / CONFIRMATION) and key in public["keys"]:
                from .evaluate import rank
                assert rank(out, key, p)[1] == c["rankings"][key]
                return
    raise ValueError(f"Unregistered evaluation: {role}/{key}")


def require_training(out, p, arm, seed, pilot=False):
    reference(out)
    if pilot and (arm, seed) == ("M0", 42):
        return
    continued(out)
    if [arm, seed] in p["training_jobs"] or (seed in (45, 46) and arm == selected(out)["winner"]):
        return
    raise ValueError(f"Unregistered training job: {arm}/{seed}")


def reference_selection(out):
    from .evaluate import evaluate, check_result
    p = protocol(out, live=True)
    path = out / "reference_selection.json"
    existing = linked(out, "reference_selection") if path.exists() else None
    metrics = {}
    for key in stage1_keys():
        metrics[key] = check_result(out, p, "development", key, verify_masks=True)
        if metrics[key] is None:
            if existing:
                raise ValueError(f"Frozen Stage 1 output is missing: {key}")
            metrics[key] = evaluate(out, "development", key)
    cmd = {key: metrics[key]["cmd"] for key in REFERENCES}
    key = reference_choice(cmd)
    record = freeze_record(out, "reference_selection", name=key, cmd=cmd,
                  ranking_sha256=metrics[key]["identity"]["ranking_sha256"],
                  development_sha256=sha(out / "development.csv"),
                  evaluation_source_commit=p["implementation_commit"], tie_break=p["reference_rule"],
                  status="continue" if cmd[key] <= p["reference_max_cmd"] else "stop",
                  artifact_sha256={k: sha(out / "evaluations/development" / k / "completed.json") for k in metrics})
    if record["status"] == "stop":
        raise ValueError(f"EAP-IG sanity stop: {cmd}")
    return record


def stage1_reuse(out, p, load=None):
    """Verify frozen Stage 1 receipts, with no inference code path."""
    from .evaluate import check_result
    load = check_result if load is None else load
    ref = reference(out)
    data = {}
    for key in stage1_keys():
        folder = out / "evaluations/development" / key
        if sha(folder / "completed.json") != ref["artifact_sha256"][key]:
            raise ValueError(f"Stage 1 receipt changed: {key}")
        value = load(out, p, "development", key, verify_masks=True)
        if value is None:
            raise ValueError(f"Stage 1 output missing: {key}")
        data[key] = value
    return data


def pilot(out):
    from .training import train
    from .evaluate import evaluate
    p = protocol(out, live=True)
    require_training(out, p, "M0", 42, pilot=True)
    reference_selection(out)
    if (out / "pilot_decision.json").exists():
        return linked(out, "pilot_decision")
    ref = reference(out)
    value = ref["cmd"][ref["name"]]
    key = ranking_key("M0", 42)
    try:
        train(out, "M0", 42, pilot=True)
        losses = pd.read_csv(out / "training" / key / "steps.csv").loss
        metrics = evaluate(out, "development", key)
    except (KeyboardInterrupt, InterruptedError):
        raise
    except Exception as exc:
        decision = pilot_decision([], None, value, False)
        freeze_record(out, "pilot_decision", **decision, error=f"{type(exc).__name__}: {exc}")
        raise
    decision = pilot_decision(losses, metrics["cmd"], value, checked_json(out / "preflight.json")["passed"])
    return freeze_record(out, "pilot_decision", **decision)


def development(out, only=None):
    from .evaluate import evaluate
    p = protocol(out, live=True)
    continued(out)
    prior = stage1_reuse(out, p)
    if only:
        if only not in p["development_jobs"]:
            raise ValueError(f"Unregistered development ranking {only}")
        return prior[only] if only in prior else evaluate(out, "development", only)
    scores = {}
    for key in p["development_jobs"]:
        scores[key] = prior[key] if key in prior else evaluate(out, "development", key)
    result = choose_arm({arm: [scores[ranking_key(arm, s)]["cmd"] for s in SEEDS[:3]] for arm in ARMS})
    return freeze_record(out, "selection", **result, reference_sha256=sha(out / "reference_selection.json"),
                         evaluation_sha256={key: sha(out / "evaluations/development" / key / "completed.json")
                                            for key in p["development_jobs"]})


def freeze_confirmation(out):
    from .evaluate import rank
    p = protocol(out, live=True)
    continued(out)
    s = selected(out)
    ref = reference(out)
    for key, value in s["evaluation_sha256"].items():
        assert sha(out / "evaluations/development" / key / "completed.json") == value
    if not (out / CONFIRMATION).exists():
        assert not list((out / "attempts").glob("confirmation_*")) and not (out / "evaluations/confirmation").exists()
    rankings = {key: rank(out, key, p)[1] for key in confirmation_keys(s["winner"])}
    return freeze_record(out, "gate6a_confirmation_protocol",
                         implementation_commit=p["implementation_commit"], selected=s["winner"],
                         selection_sha256=sha(out / "selection.json"), reference=ref["name"],
                         reference_sha256=sha(out / "reference_selection.json"), rankings=rankings,
                         confirmation_manifest_sha256=sha(out / "confirmation.csv"),
                         confirmation_hashes_sha256=sha(out / "confirmation_hashes.csv"),
                         bootstrap=p["bootstrap"], primary="five-seed mean CMD minus selected EAP-IG CMD",
                         classifications=["Invalid", "Lower CMD", "Higher CMD", "Comparable", "Inconclusive"],
                         lower_cmd_requires="negative mean, at least four negative seed effects, upper CI < 0",
                         comparable="entire CI strictly inside (-.005,+.005)",
                         descriptive="paired comparison versus all five raw absolute MAttr rankings",
                         pre_freeze_confirmation_forwards=0)


def confirmation(out, p):
    c = linked(out, "gate6a_confirmation_protocol")
    assert c["implementation_commit"] == p["implementation_commit"]
    assert c["selection_sha256"] == sha(out / "selection.json")
    assert c["reference_sha256"] == sha(out / "reference_selection.json")
    assert c["confirmation_manifest_sha256"] == sha(out / "confirmation.csv")
    assert c["confirmation_hashes_sha256"] == sha(out / "confirmation_hashes.csv")
    assert set(c["rankings"]) == set(confirmation_keys(c["selected"]))
    return c


def confirm(out, only=None):
    from .evaluate import evaluate, rank
    p = protocol(out, live=True)
    c = confirmation(out, p)
    keys = [only] if only else list(c["rankings"])
    for key in keys:
        if key not in c["rankings"] or rank(out, key, p)[1] != c["rankings"][key]:
            raise ValueError(f"Unregistered or changed confirmation ranking {key}")
        evaluate(out, "confirmation", key)


def runtime_guard(out, approved=False, report_only=False):
    p = protocol(out)
    pre = checked_json(out / "preflight.json")
    attempts = [read(file) for file in (out / "attempts").glob("*/*/attempt.json")]
    spent = sum(a.get("seconds", 0) for a in attempts if not a["key"].startswith("command_"))
    spent += checked_json(out / "graph_preflight.json")["seconds"]
    pilot_key = ranking_key("M0", 42)
    rates = [read(file) for file in (out / "attempts" / pilot_key).glob("*/rates.json")]
    default = max(pre["bounded"][loss]["seconds_per_update"] for loss in ("margin", "kl", "repo_cmd"))
    seconds_per_step = max([default, *(r["training_seconds"] / r["updates"] for r in rates if r["updates"] >= 100)])
    completed_train = sum(verified(folder) for folder in (out / "training").glob("*"))
    expected_training = len(p["training_jobs"]) + 2
    done_eval = sum(verified(folder) for role in ("development", "confirmation")
                    for folder in (out / "evaluations" / role).glob("*"))
    remaining_rankings = len(p["development_jobs"]) + len(confirmation_keys("M0")) - done_eval
    times = []
    for path in (out / "attempts" / ("development_" + pilot_key)).glob("*/compute_accounting.csv"):
        stages = pd.read_csv(path)
        for row in stages.itertuples():
            if row.stage == "official_retention_grid" and row.success and row.forward_batches > 0:
                # Patching performs two official root forwards per evaluation batch.
                times.append(row.seconds / (row.forward_batches / (2 * np.ceil(512 / 20))))
    smoke = pre["bounded"]
    stages = pd.read_csv(out / smoke["evidence"] / "compute_accounting.csv")
    cap = stages.loc[stages.stage == "actual_edge_caps"].iloc[0]
    cap_rate = cap.seconds / len(p["caps"]) * (512 / smoke["measured_circuit_prompts"])
    seconds_per_circuit = max([smoke["seconds_per_circuit_512"], cap_rate, *times])
    # Allow a cold process/model load for each remaining training or ranking job.
    per_job_setup = 20.
    projection = (spent + (expected_training-completed_train) * (5000 * seconds_per_step + per_job_setup)
                  + remaining_rankings * (15 * seconds_per_circuit + per_job_setup))
    record = dict(elapsed_attempt_seconds=spent, projected_total_seconds=projection,
                  seconds_per_update=seconds_per_step, projected_seconds_per_512_prompt_circuit=seconds_per_circuit,
                  remaining_training_jobs=expected_training-completed_train, remaining_rankings=remaining_rankings,
                  limit_seconds=p["runtime_limit_seconds"], pilot_finished=(out / "pilot_decision.json").exists())
    write_json(out / "runtime_projection.json", record)
    if record["pilot_finished"] and projection > p["runtime_limit_seconds"] and not report_only:
        if not approved:
            raise RuntimeError("Projected cumulative runtime exceeds 16 hours. Inspect runtime_projection.json and pass --approve-over-16h.")
        write_json(out / "runtime_approval.json", dict(training_sha256=sha(out / TRAINING), approved_at=time.time(), **record))
    return record


def public_test(out, approved):
    from transformers import AutoTokenizer
    from MIB_circuit_track.dataset import HFEAPDataset
    from src.mattr_data_scaling.design import official_rows
    from .evaluate import evaluate
    if not approved:
        raise ValueError("Explicit --i-approve-public-test required")
    p = protocol(out, live=True)
    c = confirmation(out, p)
    if not checked_json(out / "final_audit.json")["passed"]:
        raise ValueError("Successful confirmation audit required")
    if (out / "public_complete.json").exists():
        linked(out, "public_complete")
        raise ValueError("Public check already completed for this protocol")
    keys = [c["reference"], *(ranking_key(c["selected"], s) for s in SEEDS)]
    with attempt(out, "public_once"):
        freeze_record(out, "public_authorization", confirmation_sha256=sha(out / CONFIRMATION), keys=keys)
        if not (out / "public_manifest.json").exists():
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            ds = HFEAPDataset("mib-bench/ioi", tokenizer, split="test", task="ioi", model_name="gpt2")
            frame = official_rows(ds)
            frame.to_csv(out / "public.csv", index=False)
            freeze_record(out, "public_manifest", cohort_sha256=sha(out / "public.csv"), rows=len(frame))
        assert linked(out, "public_manifest")["cohort_sha256"] == sha(out / "public.csv")
        for key in keys:
            evaluate(out, "public", key)
        freeze_record(out, "public_complete", confirmation_sha256=sha(out / CONFIRMATION), keys=keys)
