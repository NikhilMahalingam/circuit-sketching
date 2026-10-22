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

## Gate 3: CPR-aware sampling, with frozen stopping rules

The new sampler uses only the official trainer's existing `k_sampler` hook.
Patching, loss, mask gradients, optimizer and MIB evaluator remain unchanged.
This is a static quadrature-aware mixture; it does not adapt to validation or
per-step loss. The exact distribution, floor rounding, 5% truncated jitter,
full-graph endpoint exclusion/renormalization and one-draw Torch RNG mapping
are documented in the saved `GATE3_SAMPLING_PROTOCOL.md` and JSON before training.

From WSL Ubuntu with the existing isolated runtime:

```bash
bash src/mattr_optimization/run_gate3_wsl.sh tests
OUT="results/mattr_optimization/20261022_gate3"
bash src/mattr_optimization/run_gate3_wsl.sh --out "$OUT" --freeze-only
bash src/mattr_optimization/run_gate3_wsl.sh --out "$OUT"
```

The campaign first runs three official uniform controls for matched runtime and
memory, checking their scores bitwise against Gate 2's 1,000-step checkpoints.
Then it screens exactly lambda 0.5 and 0.25 for paired seeds 42/43/44 at N=128.
Quality is compared with the saved Gate 2 checkpoints. All original success
thresholds and their SHA-256 are preserved. The additional conservative screen
policy is frozen before execution: provisional primary numerical gates against
1,000-step baselines OR the original efficiency alternative against 5,000-step
baselines. Highest mean paired relative CPR gain selects at most one eligible
mixture; a tie selects 0.5. If neither qualifies, the campaign stops.

A selected winner continues its existing score/Adam/RNG checkpoints to 5,000
updates for all three seeds. Donor-cache rebuilds, restarts, rejected variants
and all baseline controls remain charged. No further hyperparameters, Gate 4
or leaderboard submission are dispatched. Repeat the exact command only after
the previous process has ended to resume; source/protocol changes are rejected.

To regenerate the final report without model calls:

```bash
bash src/mattr_optimization/run_gate3_wsl.sh --out "$OUT" --report-only
```

The root `GATE3_ADAPTIVE_SAMPLING_REPORT.md` and campaign copy contain paired
CPR/CMD/compact-accuracy/log-AUC results, full curves and actual compute. Original
Gate 2 files are hash protected; `gate3_success_criteria.json` is copied byte for
byte, never rewritten. Model checkpoints and runtime caches stay local; curated
curves, rankings, logs and protocol evidence are retained for version control.
The ranking CSV is for inspection; use the saved exact score tensor and official
MIB graph sorting when reproducing circuits, so tied-score order is preserved.
After completion, run the CPU artifact audit with the same isolated interpreter:

```bash
~/.venvs/mattr-optimization-af82b2e/bin/python -m src.mattr_optimization.verify_gate3 --out "$OUT"
```

## Gate 4: EAP-IG score initialization

Gate 4 changes only the initial MAttr score vector. The pinned official EAP-IG
inputs adapter uses five left-endpoint integration points and signed loss
attributions. Its canonical edge audit covers every real edge, including the
official graph's negative logits-column index. The trainer-local Parameter
constructor receives centered midranks scaled by 0.1, 0.5, 1 or 2; original Adam,
sigmoid top-k, uniform-k, edge patching and evaluation stay unchanged.

```bash
bash src/mattr_optimization/run_gate4_wsl.sh tests
OUT="results/mattr_optimization/20261022_gate4"
bash src/mattr_optimization/run_gate4_wsl.sh smoke --out "$OUT/smoke_seed42" --seed 42
bash src/mattr_optimization/run_gate4_wsl.sh --out "$OUT" --freeze-only
bash src/mattr_optimization/run_gate4_wsl.sh --out "$OUT"
```

Read the measured smoke attribution cost before the final command. A frozen
protocol and SHA record the exact scale-selection and success criteria before
modified training. The exploratory seed-42 pilot selects at most two scales;
paired seeds 42/43/44 screen at 1,000 steps. At most one eligible winner continues
to 5,000 steps, with checkpoints evaluated at 500/1,000/2,500/5,000. A negative
screen stops confirmation and any extra sweep. EAP-IG-only and approximately
time-matched frozen ordinary-MAttr rankings are separate controls, not new trained
variants. Attribution and all pilot/failed-trial costs are recorded.

Report regeneration performs no model forwards:

```bash
bash src/mattr_optimization/run_gate4_wsl.sh --out "$OUT" --report-only
```

The root and campaign `GATE4_EAPIG_INITIALIZATION_REPORT.md` summarize actual
quality, full curves, compact accuracy, exact rankings, cross-seed retained-edge
Jaccard overlap, timing and memory. A disjoint official validation cohort is
accessed only after a single final development candidate qualifies. No Gate 5,
full-data follow-up or leaderboard submission is launched. Historical Gates 1–3
and the Gate 3 criteria hash remain protected.

After the campaign finishes, verify byte preservation, all mapped edges, paired
row/k streams, real computation counters and inference gradient mode without any
new model calls:

```bash
~/.venvs/mattr-optimization-af82b2e/bin/python -m src.mattr_optimization.verify_gate4 --out "$OUT"
```

The wrapper runs this CPU audit automatically for a completed campaign, including
after report regeneration. It also reconstructs 500-step display times from the
measured checkpoint prefix plus fixed setup; those prefixes were not separately
launched processes. This corrects the frozen initial reporter's full-job timing
for early display points. The protocol, 1,000-step screening costs/decisions and
all quality measurements are unchanged; the report records the correction.
