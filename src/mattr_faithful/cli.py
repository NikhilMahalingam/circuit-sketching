"""Manual Gate 6A stage entry points."""

import argparse
from pathlib import Path

from src.mattr_optimization.common import ROOT, configure, git, sha, write_json
from .core import DEFAULT
from .decisions import ARMS, SEEDS, development_keys

STAGES = {
    "preflight": "Run CPU tests, bounded checks and build discovery-only clean/empty targets",
    "graph": "Audit canonical mappings, historical rankings, streams and cohort manifests",
    "smoke": "Run bounded correctness checks on discovery prompts",
    "freeze-training": "Freeze the clean implementation and training protocol",
    "select-reference": "Run Stage 1 development evaluations and freeze the EAP-IG reference",
    "pilot": "Run and evaluate the registered M0 seed-42 pilot",
    "train": "Train one authorized arm/seed and resume exact checkpoints",
    "eval-dev": "Evaluate development rankings and freeze arm selection when complete",
    "freeze-confirmation": "Freeze the confirmation cohort, ranking set and decision rules",
    "eval-confirm": "Evaluate frozen confirmation rankings",
    "analyze": "Generate results and report without model forwards",
    "audit": "Recompute and verify the final research artifacts",
    "public-test": "Run the explicitly approved optional public-test evaluation",
    "package": "Stage allowlisted audited results, with an optional local commit",
    "run-all": "Run registered stages after preflight and training freeze",
}


def package(out, *, commit=False, dry_run=False):
    from src.mattr_fidelity_splice.core import check_files
    from src.mattr_gate5g_addendum.common import checked_json
    from .report import REPORT
    import subprocess

    record = checked_json(out / "final_audit.json")
    assert record["passed"]
    check_files(record["artifact_sha256"], out)
    for filename, key in (("gate6a_training_protocol.json", "training_sha256"),
                          ("gate6a_confirmation_protocol.json", "confirmation_sha256")):
        checked_json(out / filename)
        assert sha(out / filename) == record[key]
    names = [*record["artifact_sha256"], "gate6a_training_protocol.json", "gate6a_training_protocol.sha256",
             "gate6a_confirmation_protocol.json", "gate6a_confirmation_protocol.sha256",
             "reference_selection.json", "reference_selection.sha256", "pilot_decision.json", "pilot_decision.sha256",
             "selection.json", "selection.sha256", "final_audit.json", "final_audit.sha256",
             "development_hashes.csv", "confirmation_hashes.csv", "sparsity_loss_bins.csv", "ranking_stability.csv",
             "edge_composition.csv"]
    paths = sorted({out / name for name in names})
    for path in paths:
        assert path.is_file() and path.stat().st_size <= 5_000_000
        print(path.relative_to(ROOT).as_posix())
    if dry_run:
        return
    existing = git(ROOT, "diff", "--cached", "--name-only", "-z").decode().split("\0")
    allowed = {path.relative_to(ROOT).as_posix() for path in paths} | {REPORT}
    if set(existing) - allowed - {""}:
        raise ValueError("Unrelated staged files; preserve them and package from an empty index")
    report_body = (out / REPORT).read_text(encoding="utf-8")
    directory = out.relative_to(ROOT).as_posix()
    (ROOT / REPORT).write_text(report_body.replace("](figures/", f"]({directory}/figures/"), encoding="utf-8")
    relative = [path.relative_to(ROOT).as_posix() for path in paths] + [REPORT]
    subprocess.run(["git", "add", "-f", "--", *relative], cwd=ROOT, check=True)
    subprocess.run(["git", "diff", "--cached", "--stat"], cwd=ROOT, check=True)
    if commit:
        subprocess.run(["git", "commit", "-m", "research: report Gate 6A faithfulness-trained MAttr results"],
                       cwd=ROOT, check=True)
    print("git push -u origin research/mattr-faithful-objective")


def run_all(out, approved):
    from .pipeline import reference_selection, pilot, runtime_guard, development, freeze_confirmation, confirm
    from .training import train
    from .report import analyze, audit
    from .core import protocol

    p = protocol(out, live=True)
    reference_selection(out)
    decision = pilot(out)
    if decision["decision"] != "continue":
        analyze(out)
        raise RuntimeError(f"Pilot stopped: {decision['failures']}")
    runtime_guard(out, approved)
    for arm, seed in p["training_jobs"]:
        if (arm, seed) != ("M0", 42):
            train(out, arm, seed)
    result = development(out)
    for seed in SEEDS[-2:]:
        train(out, result["winner"], seed)
    freeze_confirmation(out)
    confirm(out)
    analyze(out)
    return audit(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage, description in STAGES.items():
        options = sub.add_parser(stage, help=description, description=description)
        options.add_argument("--out", type=Path, default=DEFAULT)
        options.add_argument("--dry-run", action="store_true")
        if stage in ("train", "eval-dev", "eval-confirm", "run-all"):
            options.add_argument("--approve-over-16h", action="store_true")
        if stage == "train":
            options.add_argument("--arm", choices=ARMS, required=True)
            options.add_argument("--seed", choices=SEEDS, type=int, required=True)
        if stage in ("eval-dev", "eval-confirm"):
            options.add_argument("--ranking", help="Execute or verify only this registered ranking")
        if stage == "public-test":
            options.add_argument("--i-approve-public-test", action="store_true")
        if stage == "package":
            options.add_argument("--commit", action="store_true")
    a = parser.parse_args()
    out = a.out.resolve()
    if not out.is_relative_to(ROOT / "results/mattr_faithful"):
        parser.error("--out must be under results/mattr_faithful/")
    if a.dry_run and a.stage != "package":
        print(f"{a.stage}: {STAGES[a.stage]}\nCampaign: {out}")
        if a.stage in ("eval-dev", "run-all"):
            print("Development ranking jobs:", ", ".join(development_keys()))
        return
    configure()
    from . import pipeline, report
    from .core import freeze_training, head, protocol, source_status
    from .preflight import prepare
    from .smoke import bounded, manual_preflight
    from .training import train
    from src.mattr_fidelity_splice.core import attempt
    if a.stage == "package" and a.dry_run:
        return package(out, dry_run=True)
    with attempt(out, "command_" + a.stage):
        try:
            if a.stage == "train":
                p = protocol(out, live=True)
                pipeline.require_training(out, p, a.arm, a.seed)
                pipeline.runtime_guard(out, a.approve_over_16h)
                value = train(out, a.arm, a.seed)
            elif a.stage in ("eval-dev", "eval-confirm"):
                pipeline.runtime_guard(out, a.approve_over_16h)
                fn = pipeline.development if a.stage == "eval-dev" else pipeline.confirm
                value = fn(out, a.ranking)
            elif a.stage == "pilot":
                value = pipeline.pilot(out)
                if value["decision"] != "continue":
                    report.analyze(out)
                    raise RuntimeError(f"Pilot stopped: {value['failures']}")
                pipeline.runtime_guard(out, report_only=True)
            elif a.stage == "preflight":
                import pytest
                if pytest.main(["-q", str(ROOT / "tests/test_mattr_faithful.py")]):
                    raise RuntimeError("Gate 6A CPU unit tests failed")
                value = manual_preflight(out)
            elif a.stage == "public-test":
                value = pipeline.public_test(out, a.i_approve_public_test)
            elif a.stage == "package":
                value = package(out, commit=a.commit)
            elif a.stage == "run-all":
                value = run_all(out, a.approve_over_16h)
            else:
                functions = {"graph": prepare, "smoke": bounded, "freeze-training": freeze_training,
                             "select-reference": pipeline.reference_selection,
                             "freeze-confirmation": pipeline.freeze_confirmation, "analyze": report.analyze, "audit": report.audit}
                value = functions[a.stage](out)
            status = "complete"
            if isinstance(value, dict) and "classification" in value:
                print(value["classification"])
        except BaseException:
            status = "failed_or_stopped"
            raise
        finally:
            frozen = out / "gate6a_training_protocol.json"
            write_json(out / f"status_{a.stage}.json", dict(status=status, commit=head(),
                       source_status=source_status(), training_sha256=sha(frozen) if frozen.exists() else None))


if __name__ == "__main__":
    main()
