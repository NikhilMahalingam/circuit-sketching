# Gate 5G addendum manual runbook

Everything is post hoc/exploratory. Original Gate5G protocol and classifications
remain unchanged. Stage1 is saved-artifact CPU analysis; Stage2 must be explicitly
started by the researcher. No training, new attribution, model-weight updates,
leaderboard submission or Codex/LLM dependency is present.

## Existing pinned WSL environment

```bash
cd '/mnt/h/GenAI Research/sketch_prune'
git switch research/mattr-gate5g-addendum
source "$HOME/.venvs/mattr-optimization-af82b2e/bin/activate"
```

Wrappers use the same pinned Python, offline HuggingFace cache and dependency
setup as Gate5G. They resolve the repository from their own path, work from any
directory, support `--help`, `--dry-run`, `--out`, and fail nonzero. Scripts remain
foreground-only. Do not upgrade dependencies or alter original source/artifacts.

## Stage1 (already executed by Codex)

The committed `stage1_analysis_plan.md` predates computation of the diagnostics.
It limits flexibility but does not make a post hoc analysis confirmatory.

```bash
bash scripts/gate5g_addendum/00_stage1.sh --dry-run
bash scripts/gate5g_addendum/00_stage1.sh
```

Successful reruns verify completion/output hashes and skip. Source/configuration
changes require a separate campaign via `--out`; never overwrite frozen evidence.
Stage1 reads only saved outputs/rankings and CPU graphs, then constructs and hashes
the fresh Stage2 cohort with seed5546. This preparation performs no model forwards.
Raw prompts and large edge inventories stay local. Small summaries and figures
are committed with the implementation. All denominators, ratios and accounting
shares retain their documented caveats. Researcher interpretation remains empty.

## Bounded correctness (optional)

```bash
bash scripts/gate5g_addendum/08_smoke.sh
```

This uses two excluded historical Gate5F prompts only. It checks a custom selected
set, exact official pruned mask, cached output reuse with zero repeat forwards,
official one-point normalized faithfulness and the unchanged validated full-vocab
KL recorder. Prompt hashes are recorded before forwards. Fresh confirmation prompts
are never used. CPU tests:

```bash
python -m pytest tests/test_mattr_gate5g_addendum.py -q
```

## Step1: adaptive seed42 search — manual GPU execution

```bash
bash scripts/gate5g_addendum/03_seed42_search.sh --dry-run
bash scripts/gate5g_addendum/03_seed42_search.sh
```

Search refuses new confirmation-cohort inputs: its only dataset path is the old
Gate5G manifest, checked disjoint from confirmation. On first execution it freezes
`search_protocol.json` with the implementation commit, source/dependency hashes,
model identity, old cohort, R and deterministic rules before model outputs.
Relevant tracked source must be clean. The candidate pool is exactly the original
rank interval between capped116 and296 prefixes; it is not all32491 edges.

Level1: eight contiguous balanced R blocks, evaluate C116+block and C296−block.
Levels2/3: split only the winning current addition block into four contiguous
balanced sub-blocks; stop at singleton or three levels. Choose the final candidate
from **all evaluated additions** using maximum old-cohort accuracy gain, then fewer
actual retained edges, lowest original rank, then mask hash. This retains a parent
block if refinement loses its benefit. At most60 unique circuit masks; typically
24block requests plus two references. Cache reuse and failed/resumed computation
are recorded separately. Full selected/pruned sets, ranks, count changes, topology
flags and search trace are saved. No source/cohort choice is exposed as an override.

Rerun the same search command after interruption. Identical masks reuse verified
outputs; the deterministic search is replayed without new forwards for those masks.
Completed candidate/trace/circuits are hash-checked before skipping.

## Step2: freeze fresh confirmation and four-circuit interventions

```bash
bash scripts/gate5g_addendum/01_freeze_stage2.sh
```

This requires the completed candidate search. It freezes `stage2_protocol.json`
and SHA256, including the candidate, source commit, all63 registered requests,
data and model identities, original parent/splice mask checks, explicit unpruned
sets and official pruned masks, statistics, bootstrap5000/seed5646, projections and
stopping rules. The new cohort has512 rows (minimum256 if insufficient eligible),
proportional largest-remainder nine-template quotas and all historical/train/
Gate5G clean/corrupted/cross-role exclusions. No forward may use it before freeze.

Keep HEAD and all experiment sources fixed between search, freeze and GPU jobs.
Every registered command checks source bytes, dependencies, protocol/config,
candidate and mask hashes before reuse/execution. No silent override exists.
Do not commit ignored protocol/results before evaluation: changing HEAD locks jobs.
For a code/resource amendment preserve the campaign, document the deviation,
commit a tested implementation, and use a new `--out` campaign throughout.

## Steps3/4: manual displacement and independent confirmation

```bash
# All five seeds, one foreground process per seed:
bash scripts/gate5g_addendum/02_displacement.sh
# Optional single seed/fraction:
bash scripts/gate5g_addendum/02_displacement.sh --seed 42 --fraction 0.1

# Only the frozen seed42 candidate and its two references:
bash scripts/gate5g_addendum/04_seed42_confirm.sh
```

There are60 displacement requests (5seeds×3fractions×4circuits) and3 confirmation
requests. Identical officially pruned masks reuse outputs. Parent, Remove, Add and
Splice refer to their explicitly recorded unpruned sets; official pruning is
unchanged. Remove/Add change budgets; non-additivity is circuit-level and includes
pruning, actual-size and intervention effects. Three points do not define new CPR.

After the first complete displacement seed, the measured projection stops if the
cumulative GPU-job process wall estimate exceeds2hours. To approve arbitrary
duration explicitly while keeping settings fixed:

```bash
bash scripts/gate5g_addendum/02_displacement.sh --approve-over-2h
bash scripts/gate5g_addendum/04_seed42_confirm.sh --approve-over-2h
```

These flags propagate to every seed. OOM/correctness/hash failures still stop.
Timers include initialization/references and failed/resumed attempts; allocator
peaks are not total physical VRAM occupancy or GPU-utilization time. No automatic
batch/sample reduction occurs. Repeat identical commands/flags to resume.

## Steps5/6: offline aggregation and audit

```bash
bash scripts/gate5g_addendum/05_analyze.sh
bash scripts/gate5g_addendum/06_audit.sh
less GATE5G_ADDENDUM_REPORT.md
```

One shared5000-draw exact-template paired prompt bootstrap, seed5646, is reused
across all circuits/seeds. Normalize replicate cohort means with the official
formula. Near-zero denominators fail; ratios with an unstable C296−C116 accuracy
gap are counted and flagged. Prompt intervals condition on fixed trained seeds.
All numerical results and overlapping/absent descriptive labels are retained.
Candidate selection remains frozen; no confirmation-driven refinement is allowed.

Both scripts perform zero new model forwards. Audits validate saved execution
commits without requiring current HEAD to remain the original implementation
after results commits. The researcher-written interpretation is not auto-filled.

## Step7: review and package

Fill the researcher interpretation after reviewing actual results, then:

```bash
bash scripts/gate5g_addendum/07_package.sh --dry-run
bash scripts/gate5g_addendum/07_package.sh
git diff --cached --stat
bash scripts/gate5g_addendum/07_package.sh --commit
git push origin research/mattr-gate5g-addendum
```

Only allowlisted small summaries, protocols, report and PNGs are staged; files
larger than10MB or unrelated staged work are rejected. Raw prompts, model weights,
score/mask tensors, full-vocabulary reference caches and large edge inventories
remain local. Packaging never pushes automatically. No further research gate runs.
