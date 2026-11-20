"""Manual Gate 6B stages."""

import argparse
from pathlib import Path
import subprocess

from src.mattr_faithful.core import head, source_status
from src.mattr_fidelity_splice.core import attempt, check_files
from src.mattr_gate5g_addendum.common import checked_json
from src.mattr_optimization.common import ROOT, configure, git, sha, write_json
from .core import DEFAULT, FILE, JOBS, freeze, protocol


def package(out, commit, dry_run):
    from .analysis import REPORT

    protocol(out)
    audit = checked_json(out / "final_audit.json")
    assert audit["passed"] and audit["protocol_sha256"] == sha(out / FILE)
    check_files(audit["outputs"], out)
    paths = [out / name for name in [*audit["outputs"], FILE, "gate6b_protocol.sha256",
             "final_audit.json", "final_audit.sha256", "cohort_hashes.csv", "runtime_projection.json"]]
    for path in paths:
        assert path.is_file() and path.stat().st_size <= 5_000_000, path
        print(path.relative_to(ROOT).as_posix())
    print(REPORT)
    if dry_run:
        return
    assert head() and git(ROOT, "branch", "--show-current").decode().strip() == "research/mattr-faithful-ensemble"
    names = [p.relative_to(ROOT).as_posix() for p in paths] + [REPORT]
    staged = set(filter(None, git(ROOT, "diff", "--cached", "--name-only", "-z").decode().split("\0")))
    if staged - set(names):
        raise ValueError("Unrelated staged files are present")
    body = (out / REPORT).read_text(encoding="utf-8")
    (ROOT / REPORT).write_text(body.replace("](figures/", f"]({out.relative_to(ROOT).as_posix()}/figures/"), encoding="utf-8")
    subprocess.run(["git", "add", "-f", "--", *names], cwd=ROOT, check=True)
    subprocess.run(["git", "diff", "--cached", "--stat"], cwd=ROOT, check=True)
    if commit:
        subprocess.run(["git", "commit", "-m", "research: report Gate 6B seed ensemble results"], cwd=ROOT, check=True)
    print("git push -u origin research/mattr-faithful-ensemble")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("preflight", "freeze", "evaluate", "analyze", "audit", "package", "run-all"):
        options = sub.add_parser(stage)
        options.add_argument("--out", type=Path, default=DEFAULT)
        options.add_argument("--dry-run", action="store_true")
        if stage == "evaluate":
            options.add_argument("--ranking", choices=JOBS)
        if stage in ("evaluate", "run-all"):
            options.add_argument("--approve-over-2h", action="store_true")
        if stage == "package":
            options.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "results/mattr_ensemble") or out == ROOT / "results/mattr_ensemble":
        parser.error("--out must be a campaign directory under results/mattr_ensemble")
    if args.dry_run and args.stage != "package":
        print(args.stage, out, getattr(args, "ranking", None) or ", ".join(JOBS))
        return
    configure()
    from .evaluation import evaluate, smoke
    from .analysis import analyze, audit
    if args.dry_run:
        return package(out, False, True)
    with attempt(out, "command_" + args.stage):
        status = "failed_or_stopped"
        try:
            if args.stage == "preflight":
                import pytest
                if pytest.main(["-q", str(ROOT / "tests/test_mattr_ensemble.py")]):
                    raise RuntimeError("Gate 6B CPU tests failed")
                value = smoke(out)
            elif args.stage == "freeze":
                value = freeze(out)
            elif args.stage == "evaluate":
                value = evaluate(out, args.ranking, args.approve_over_2h)
            elif args.stage == "analyze":
                value = analyze(out)
            elif args.stage == "audit":
                value = audit(out)
            elif args.stage == "package":
                value = package(out, args.commit, False)
            else:
                evaluate(out, approved=args.approve_over_2h)
                analyze(out)
                value = audit(out)
            status = "complete"
            if isinstance(value, dict):
                print(value.get("classification", value.get("passed", args.stage)), flush=True)
        finally:
            write_json(out / f"status_{args.stage}.json", dict(status=status, commit=head(), source_status=source_status(),
                       protocol_sha256=sha(out / FILE) if (out / FILE).exists() else None))


if __name__ == "__main__":
    main()
