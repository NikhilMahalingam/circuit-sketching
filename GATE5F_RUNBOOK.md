# Gate 5F: manual execution

Implementation and bounded correctness tests only have been run by Codex.
There are no Gate 5F research results yet. You launch the registered jobs yourself.
No command requires Codex, an agent, an LLM API, or an online service once the
pinned model/dataset cache is present. The wrappers default to offline mode.

## Environment and first command

From your WSL Ubuntu terminal:

```bash
cd "/mnt/h/GenAI Research/sketch_prune"
source "$HOME/.venvs/mattr-optimization-af82b2e/bin/activate"
python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'unavailable')"
bash scripts/gate5f/00_preflight.sh
```

The first experiment command is `bash scripts/gate5f/00_preflight.sh`.
It runs only inventory/data preparation, CPU schedule tests, bounded 30-update
smoke tests, a 16-example total EAP throughput/partition check, and a four-training-
prompt evaluator check. It never launches registered training or512-row sweeps.
`--cpu-only` omits GPU checks and cannot qualify a new protocol for freezing.
Already frozen preflight inputs are verified instead of regenerated.

Every wrapper resolves the project root from its own absolute location. For example,
the following works from any directory:

```bash
bash "/mnt/h/GenAI Research/sketch_prune/scripts/gate5f/00_preflight.sh" --help
```

The wrappers use your activated `VIRTUAL_ENV`, otherwise the existing pinned
`$HOME/.venvs/mattr-optimization-af82b2e`. Set `GATE5F_PYTHON` only to another
installation of the same pinned environment; dependency checks reject drift.
No packages or upstream sources are installed or modified.
The original local historical checkpoint directories and cached official data
must remain available; large checkpoint/data files are intentionally not in Git.
One historical reporting helper uses frozen CRLF bytes while Git stores LF.
The exact bytes and hash are archived in `source_at_freeze/` and
`source_byte_manifest.json` under the campaign directory. This existing working
copy already matches. If a new checkout fails specifically on
`src/mattr_hierarchy/reconcile_accounting.py`, inspect the byte manifest and run:

```bash
python scripts/gate5f/restore_archived_line_endings.py --restore
```

The helper refuses any source-logic difference and only restores the archived
line endings. Historical source logic, frozen criteria and protocol remain unchanged.

## Review and freeze

Artifacts default to `results/mattr_data_scaling/gate5f/`. Review:

- `checkpoint_inventory.csv`: source/data/draw compatibility, hashes, rejected methods.
- `replacement_job_manifest.json`: reusable checkpoints and missing trajectories.
- `historical_ranking_references.json`: the Gate5D/5E references to15 saved rankings.
- `TRAINING_SCHEDULE_AUDIT.md` and `schedule_audit.json`.
- `cohort_audit.json`, both cohort hash tables, and `stream_convention.json`.
- `eap_preflight.json`: measured throughput, batch partition equivalence, memory and projection.
- `preflight_cpu.json`, `preflight_gpu.json`, smoke logs and source pins.

```bash
bash scripts/gate5f/01_freeze_protocol.sh
cat results/mattr_data_scaling/gate5f/gate5f_protocol.sha256
```

Freeze fails without passing preflight. The28 endpoints, cohorts, sources,
data/RNG streams, reuse decisions, known-head reference, statistical definitions,
command list and criteria become immutable. The five-seed primary analysis uses
42–46 at N10000/5000 only. Factorial analyses use42–44 only. The Gate5E cohort is
secondary development data and cannot determine classification.

The compatibility inventory identifies12 missing training trajectories in the
current checkout: three H2 N128 continuations from1000 to5000, and nine full-pool
trajectories. The three N128 baseline trajectories and full-data seed42 baseline
are historical reuses. Intermediate1000-update checkpoints are valid because
the pinned trainer has constant learning rate/temperature and its duration only
sets the loop bound; the CPU test checks scores and Adam state explicitly.
Historical Gate2 lacks before/after model hashes: its reuse evidence is the
verified completion marker, source/stream/data agreement and score-only Adam
state, disclosed separately from Gate5C's explicit weight-hash audits.

## Complete sequential pipeline

After reviewing/freezing, run this yourself:

```bash
bash scripts/gate5f/run_all.sh
```

It reads the frozen command manifest, executes missing work sequentially in the
foreground, stops on the first failure, then analyzes, audits and packages.
It does not start preflight or freeze implicitly. It never selects favorable
seeds, changes settings, launches background jobs, or submits to a leaderboard.
`Ctrl+C` interrupts execution; completed files remain intact. Rerun the same
command to resume verified missing jobs. No new Codex session is needed.

## Individual training jobs

These are real implemented CLI arguments:

```bash
bash scripts/gate5f/02_train.sh --dataset-size 128 --method h2 --seed 42 --max-steps 5000 --resume
bash scripts/gate5f/02_train.sh --dataset-size 10000 --method baseline --seed 43 --max-steps 5000 --resume
bash scripts/gate5f/02_train.sh --dataset-size 10000 --method h2 --seed 43 --max-steps 5000 --resume
```

Use seeds42–44 for N128 and42–46 for N10000. Default max steps is5000. You can
stop deliberately at1000 using `--max-steps 1000` and later extend with
`--max-steps 5000 --resume`. Extra seeds45/46 may have1000 checkpoints but are
rejected as factorial evaluation endpoints. Reused1000 checkpoints are exported
without training. Historical files are never overwritten.

Each process records a unique attempt ID, PID, host, UTC timestamps and config
identity, writes append-only logs and counters, and locks its own job. Checkpoints
are written atomically every100 updates and at targets, with sidecar SHA-256.
They preserve raw scores, Adam state, global RNG states and frozen stream
position/prefix hashes. Exported rankings use effective H2 scores. Cold restarts
rebuild deterministic donor caches and their extra work is charged.

A failed checkpoint write may leave an unreferenced `.tmp` file. The previous
verified checkpoint is usable. A `.pt` missing its hash sidecar is preserved in
the checkpoint folder's `incomplete/` directory with a unique name, logged, and
excluded from resumption. A checksum mismatch stops for inspection. No valid
checkpoint is silently replaced.

## EAP-IG attribution

```bash
bash scripts/gate5f/03_eap_ig.sh --dataset-size 128 --seed 42
bash scripts/gate5f/03_eap_ig.sh --dataset-size 128 --seed 43
bash scripts/gate5f/03_eap_ig.sh --dataset-size 128 --seed 44
bash scripts/gate5f/03_eap_ig.sh --dataset-size 10000
```

This calls the pinned official `EAP-IG-inputs`, five integration points, patching,
signed scores, batch4 and the official mean loss. Both128 and10000 divide by4.
Preflight tests that averaging equal-batch chunk outputs by example count
preserves the complete reference computation. Under the pinned mean-loss and
`total_items` reduction, changing batch size changes raw scale; do not do so.
The full-data output is one ranking, not five seeds. Attribution restarts after
an interrupted attempt; the failed attempt's costs/logs are preserved. An OOM
stops with the original settings, rather than silently changing batch/IG steps.

## Evaluate saved rankings

```bash
bash scripts/gate5f/04_evaluate.sh --method baseline --dataset-size 128 --seed 42 --step 1000 --cohort primary
bash scripts/gate5f/04_evaluate.sh --method h2 --dataset-size 10000 --seed 44 --step 5000 --cohort primary
bash scripts/gate5f/04_evaluate.sh --method h2 --dataset-size 10000 --seed 44 --step 5000 --cohort development
bash scripts/gate5f/04_evaluate.sh --method eap_ig --dataset-size 128 --seed 42 --step 0 --cohort primary
bash scripts/gate5f/04_evaluate.sh --method eap_ig --dataset-size 10000 --step 0 --cohort development
```

Each job independently saves full official CPR/CMD/log-AUC curves, per-prompt
margins/positive-margin correctness/vocabulary-top1, actual-edge cap curves,
nominal prefix/shortfall/mask hashes and compute. Completed outputs are skipped
only after hash/config checks. Interrupted evaluation reuses verified immutable
per-mask outputs; no mutable activations are reused. Batch20 and float32 remain
fixed. Full-model/clean references and model weight integrity are checked.

## Twelve-hour stop rule

Preflight rates provide initial projections. The first new trajectory reaching
5000 calibrates actual training cost (a continuation may execute4000 new updates).
Before every subsequent expensive command, the guard charges all attempts and
projects remaining training, attribution and evaluation using measured rates.
If projected cumulative wall time exceeds12hours, it stops and writes
`runtime_projection.json`. It does not reduce the matrix or budgets.

The implementation preflight's initial projection is about21.2hours, dominated
by a deliberately conservative evaluation extrapolation from four prompts.
This is not a measured full-campaign duration. Expect the guard to require review
after the first new5000-update trajectory unless updated measurements bring it
below the threshold. Read the saved component estimates before approving.

After reading that report, explicitly authorize continuing unchanged:

```bash
bash scripts/gate5f/run_all.sh --approve-over-12h
```

The same flag works on individual expensive wrappers. Approval is recorded with
the protocol hash; it is never inferred or applied automatically. The projection
is an estimate, not GPU utilization time, and includes failed/restarted work.

## Offline analysis, audit and packaging

```bash
bash scripts/gate5f/05_analyze.sh
bash scripts/gate5f/06_audit.sh
bash scripts/gate5f/07_package.sh
```

Analysis does not construct a model or execute forwards. It requires all64
evaluation jobs (56 paired MAttr/H2 jobs plus8 EAP references). If incomplete,
it writes the missing-job table and exits nonzero without a classification.
`05_analyze.sh --allow-incomplete` writes status only, never fabricated metrics.

The completed generator writes `GATE5F_DATA_SCALING_REPORT.md`, statistical
tables and eight plots. Known-head overlap uses the frozen26-head reference;
topology controls are64 layer-preserving head relabelings checked under official
pruning. They exactly match actual size and source/destination roles, but test
head identity conditional on layer structure. This limitation is explicit.

The final audit rechecks protocol/artifact hashes, paired streams, old-file
preservation, cohort exclusions, unchanged weights, counters and classification.
Packaging copies only small reports/configs/summaries/plots into `package/` with
a checksum manifest. It excludes model weights, checkpoints, raw prompt text and
large caches. For the pre-execution implementation package only:

```bash
bash scripts/gate5f/07_package.sh --implementation-only
```

Every wrapper supports `--help`, exits nonzero on failure, and saves detailed logs
under `results/mattr_data_scaling/gate5f/logs/`. Do not edit frozen source/config
files mid-campaign. Resolve a demonstrated blocker transparently before creating
a newly identified protocol; never silently weaken the existing criteria.
