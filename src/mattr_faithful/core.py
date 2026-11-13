"""Source locking and immutable Gate 6A artifacts."""

from src.mattr_fidelity_splice.core import check_files
from src.mattr_gate5g_addendum.common import checked_json, freeze_json
from src.mattr_optimization.common import GRID, ROOT, git, provenance, sha
from .decisions import ARMS, SEEDS, CAPS, development_keys, stage1_keys

DEFAULT = ROOT / "results/mattr_faithful/gate6a"
HISTORY = ROOT / "results/mattr_data_scaling/gate5f"
GATE5G = ROOT / "results/mattr_fidelity_splice/gate5g"
ADDENDUM = ROOT / "results/mattr_gate5g_addendum/gate5g_addendum"
AUDIT = ROOT / "GATE6A_PRIOR_WORK_AUDIT.md"
TRAINING = "gate6a_training_protocol.json"
CONFIRMATION = "gate6a_confirmation_protocol.json"
SETTINGS = dict(model="gpt2-small", task="ioi", n=10000, steps=5000,
                masking="topk", k_schedule="uniform", mode="iso", temperature=.5,
                n_iters=30, optimizer="Adam", lr=.05, adam_betas=[.9, .999],
                adam_eps=1e-8, gamma=0, precision="float32", training_batch=1,
                evaluation_batch=20, cf_cache_gb=4)
SOURCES = ("src", "scripts", "configs", "tests", "requirements.txt", ".gitattributes", AUDIT.name)


def head():
    return git(ROOT, "rev-parse", "HEAD").decode().strip()


def source_status():
    return git(ROOT, "status", "--porcelain", "--untracked-files=all", "--", *SOURCES).decode().strip()


def freeze_training(out):
    from .targets import verified_targets
    if (out / TRAINING).exists():
        return protocol(out, live=True)
    if source_status():
        raise ValueError("Commit Gate 6A sources before freezing the training protocol")
    preflight = checked_json(out / "preflight.json")
    if not preflight["passed"] or preflight["implementation_commit"] != head():
        raise ValueError("Preflight must pass at the current clean implementation commit")
    check_files(preflight["artifacts"], out)
    assert verified_targets(out / "targets")
    check_files(preflight["historical"])
    check_files(preflight["dependencies"]["source_sha256"])
    check_files(preflight["pinned_extra"])
    if provenance() != preflight["dependencies"]:
        raise ValueError("Runtime differs from verified preflight")
    sources = git(ROOT, "ls-files", "-z", "--", *SOURCES).decode().split("\0")
    record = dict(gate="6A", implementation_commit=head(), source_status=source_status(),
                  sources={name: sha(ROOT / name) for name in sources if name},
                  audit_sha256=sha(AUDIT), dependencies=preflight["dependencies"],
                  pinned_extra=preflight["pinned_extra"], settings=SETTINGS,
                  artifacts=preflight["artifacts"], historical=preflight["historical"],
                  preflight_sha256=sha(out / "preflight.json"), arms=ARMS, seeds=SEEDS,
                  grid=GRID, caps=CAPS, target_strategy=preflight["targets"]["strategy"],
                  target_normalizer=preflight["targets"]["D"],
                  target_metadata_sha256=sha(out / "targets/metadata.json"),
                  cohort=preflight["cohort"], historical_streams=preflight["pairing"],
                  training_jobs=[[arm, seed] for arm in ARMS for seed in SEEDS[:3]],
                  extension_rule="only selected arm, seeds 45 and 46",
                  reference_jobs=stage1_keys(), development_jobs=development_keys(),
                  reference_rule="minimum development CMD, absolute if tied within 1e-8",
                  reference_max_cmd=.10,
                  pilot=dict(arm="M0", seed=42, windows=[[1, 500], [4501, 5000]],
                             require_finite=True, strict_loss_decrease=True, max_cmd_ratio=3,
                             correctness_required=True),
                  selection="minimum three-seed mean CMD; within <0.001 of best prefer ordinary, then margin, repo_cmd, KL, then arm ID",
                  bootstrap=dict(seed=5647, draws=5000, stratification="exact template", paired=True,
                                 denominator_epsilon=1e-8),
                  evaluator=dict(level="edge", intervention="patching", prune=True, log_scale=False,
                                 ranking="positive ordinal except raw mattr_abs"),
                  runtime_limit_seconds=16 * 3600)
    freeze_json(out / TRAINING, record)
    return protocol(out, live=True)


def protocol(out, live=False):
    p = checked_json(out / TRAINING)
    if sha(out / "preflight.json") != p["preflight_sha256"]:
        raise ValueError("Preflight record changed after freeze")
    check_files(p["artifacts"], out)
    check_files(p["historical"])
    if live:
        if head() != p["implementation_commit"] or source_status():
            raise ValueError("Frozen implementation commit or clean tracked sources mismatch")
        check_files(p["sources"])
        check_files(p["pinned_extra"])
        if provenance() != p["dependencies"]:
            raise ValueError("Pinned dependencies differ from the frozen protocol")
    return p


def identity(out, role, key, **fields):
    return dict(role=role, key=key, training_sha256=sha(out / TRAINING),
                implementation_commit=head(), source_status=source_status(), **fields)


def reference(out):
    record = checked_json(out / "reference_selection.json")
    if record["training_sha256"] != sha(out / TRAINING):
        raise ValueError("Reference selection is from a different protocol")
    if record["status"] != "continue":
        raise ValueError("EAP-IG CMD exceeded the frozen 0.10 sanity threshold")
    return record
