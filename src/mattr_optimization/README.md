# MAttr optimization — independent execution gates

The upstream edge CLI, trainer and pinned official MIB evaluator are called
directly. No adaptive-sampling, gradient-initialization or refinement algorithm
has been implemented. The previous node-level reproduction is a separate study.

Run from the project root in WSL Ubuntu:

```bash
bash src/mattr_optimization/setup_wsl.sh
bash src/mattr_optimization/run_wsl.sh tests
OUT="results/mattr_optimization/$(date -u +%Y%m%dT%H%M%SZ)"
bash src/mattr_optimization/run_wsl.sh --out "$OUT"
```

Setup creates `~/.venvs/mattr-optimization-af82b2e` from the existing official
frozen uv lock and offline dependency cache. It never syncs older environments.
If the uv bootstrap/cache is missing, setup fails; it does not change versions
or silently install alternative dependencies. No downloads or test split access.

Default smoke: 30 updates, 16 official train rows, 32 official validation rows,
seed 42, one training example per step, evaluation batch 2, donor budget 1 GB.
The runner refuses more than 100 updates. Completed output folders cannot be
overwritten. An interrupted run can resume using the exact same configuration;
score, Adam and Python/NumPy/Torch/CUDA RNG state are restored. Donor caches are
rebuilt and their real forward calls are counted. Checkpoint equivalence is
unit-tested against the upstream trainer, not inferred from final circuit size.

The full reference must be a **new, separately authorized Gate-1 run**, with
5,000 steps, uncapped official data and the documented upstream settings.
`--batch-size` in the official edge CLI is evaluation batch size. There is no
`--train-examples` or `--adam-eps` flag in that CLI. Do not present a capped
smoke as the full reference. The independently callable Gate-1 runner is:

```bash
OUT="results/mattr_optimization/$(date -u +%Y%m%dT%H%M%SZ)_gate1"
bash src/mattr_optimization/run_wsl.sh gate1 --out "$OUT"
```

Gate 1 requires saved passing edge-level Gate-0 evidence, preserves all official
rows in their original order, runs 5,000 iterations with seed 42, evaluation batch
20 and a 4 GB donor cache, then stops. It never dispatches Gate 2 or a variant.
`--batch-size` and `--cf-cache-gb` are explicit resource overrides recorded in
the configuration; none is changed automatically. The run's source/configuration
identity is frozen. On interruption, repeat the exact same command to resume.
The WSL wrapper first compares current official rows with saved train/validation
manifests, rejecting a changed dataset or row order before loading a checkpoint.
Forwards and incomplete attempts remain charged. Scores/Adam/RNG are checkpointed
every 100 iterations. Official validation vectors are exported and can be reused
only with matching ranking/configuration/data identities and file hashes.

`live_status.json` records optimizer milestones and final process state; during
validation, follow `console.log` and the newly saved vectors in `raw_validation/`.
The console log is available when the shell command is piped to a log.
`raw_validation/` holds per-example official
clean, empty-circuit corrupted and ten circuit-grid output vectors. Use:

```bash
bash src/mattr_optimization/run_wsl.sh gate1 --out "$OUT" --report-only
```

to regenerate the completed Gate-1 report without executing model forwards.

Audit a completed run without loading model weights or executing forwards:

```bash
bash src/mattr_optimization/run_wsl.sh verify-gate1 --out "$OUT"
```

The verifier checks raw scores, checkpoint/optimizer state, actual data order,
official edge selection/tie order, saved per-example metrics and forward counts.

## Gate 2: launch discovery-data scaling yourself

Run from the project root in **WSL Ubuntu**. The default interpreter is the
existing isolated `~/.venvs/mattr-optimization-af82b2e/bin/python`; set
`MATTR_OPT_PYTHON` to another compatible interpreter if desired. Keep the official
source pins and frozen dependency versions. Gate-1 completion/audit evidence
must be available locally (or give `--gate1-evidence /path/to/gate1_passed.json`).

```bash
OUT="results/mattr_optimization/$(date -u +%Y%m%dT%H%M%SZ)_gate2"
bash src/mattr_optimization/run_wsl.sh gate2 --out "$OUT" \
  --sizes 16 32 64 128 --seeds 42 --validation-size 128
```

This is **four separate 5,000-iteration edge-MAttr runs**, not one run split
among four pools. Each job uses a fresh CUDA subprocess, unchanged uniform/topk/
iso/Adam-lr-.05 training, nested deterministic train subsets and one fixed
128-example validation cohort. It evaluates frozen 500/1,000/5,000-iteration
rankings and final train quality with official MIB. The eligible pool, actual
visited unique examples and actual presentations are separate outputs. Seed 42
is the first feasibility campaign; use a new output directory for seeds 43/44
after reviewing it. Gate 3 is never dispatched.

Append `--dry-run` to preview without loading a model or starting training.
CPU subset/manifest tests: `bash src/mattr_optimization/run_wsl.sh tests-gate2`.
The full Gate-2 CUDA sweep was not run during preparation; only CLI and CPU
tests were executed. Expect roughly a couple of hours for the four seed-42
runs, with hardware-dependent variation. No settings change after OOM.

To resume, keep the **same OUT and arguments** and rerun the final command;
do not generate another timestamp. Completed jobs and evaluations are reused
after identity/hash checks, and incomplete jobs restore score/Adam/RNG
checkpoints. Do not change source or package versions mid-campaign. Per-job
logs are saved automatically. Changed manifests are rejected before overwrite.
Force-killed attempts missing final timing logs are flagged as incomplete cost.

The campaign exports `sample_efficiency.csv`, `quality_by_step.csv`, complete
faithfulness curves, per-example reuse counts, compute logs, plots and its own
`MATTR_SAMPLE_EFFICIENCY_REPORT.md`. All Gate-0/Gate-1 campaign files are left
untouched. To regenerate analysis only:

```bash
bash src/mattr_optimization/run_wsl.sh gate2 --out "$OUT" --report-only
```

Use `notebooks/mattr_optimization.ipynb` to inspect results. All execution
switches default off. Generate it with the Windows environment:

```powershell
.venv/Scripts/python.exe -m src.mattr_optimization.notebook
.venv/Scripts/python.exe -m src.mattr_optimization.execute_analysis --out <campaign>
```

Source evidence: `MATTR_FINDINGS.md`. The root `MATTR_BASELINE_REPORT.md` reflects
the latest completed baseline gate; each campaign retains its own report. Other reports explicitly mark
unrun gates. Metric tests use synthetic curves only as unit fixtures; all model
benchmark metrics use the seven-value official MIB evaluator. No leaderboard
submission or multi-model work is authorized by this Gate-0 runner.

## Gate 2 final analysis (no algorithm changes)

`gate2_controls.py` runs four conditional N=128 controls: training seed 43/44
with subset seed 42, and training seed 42 with subset seed 43/44. The existing
diagonal runs supply the remaining anchor/comparisons. Only the subset RNG is
decoupled; the upstream 5,000-update training recipe is unchanged. These controls
measure conditional subset and training variation; they are not a full factorial
variance decomposition or a significance test.

`gate2_frozen.py` evaluates Gate 1's saved score tensor on the exact Gate 2
validation cohort, without retraining. It also evaluates the original Gate 2
500/1,000-step frozen checkpoints on their respective training pools to complete
the generalization-gap tables. Official per-example vectors and real compute
counts are saved separately from the original campaigns.

The serial workflow for the authorized October 21 follow-up is:

```bash
bash src/mattr_optimization/gate2_followup.sh
```

It waits for `20261021_gate2_seeds43_44/gate2_passed.json`, then runs the four
controls, frozen evaluations and final analysis in sequence. Do not start a
second copy alongside the active workflow. Resume only after the old process has
ended; completed jobs and frozen evaluations are identity/hash checked.

Outputs live under `results/mattr_optimization/20261021_gate2_final/`.
`GATE2_FINAL_ANALYSIS.md` contains all checkpoints, paired contrasts, conditional
variability, compact curves and compute. `gate3_success_criteria.json` is frozen
before any modified sampling schedule is tested. Passing a CPR-specific rule
does not establish better fidelity when CMD increases. Gate 3 is not launched.

CPU-only follow-up tests:

```bash
~/.venvs/mattr-optimization-af82b2e/bin/python -m unittest discover -s tests -p 'test_mattr_gate2_*.py' -v
```
