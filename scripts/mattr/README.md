# Official MAttr reproduction

Run from **WSL Ubuntu** with a working CUDA driver. These scripts use the
official locked Python 3.12 `lm`/`dev` environment, not the Windows
Sketch-ACDC virtual environment. The RTX 3060 passed a real CUDA tensor check.

```bash
cd '/mnt/h/GenAI Research/sketch_prune'
bash scripts/mattr/setup_wsl.sh
OUT="$PWD/results/mattr_reproduction/$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/mattr/campaign_wsl.sh --out "$OUT" --stage tests
bash scripts/mattr/campaign_wsl.sh --out "$OUT" --stage smoke
bash scripts/mattr/campaign_wsl.sh --out "$OUT" --stage reference
bash scripts/mattr/campaign_wsl.sh --out "$OUT" --stage scaling
bash scripts/mattr/campaign_wsl.sh --out "$OUT" --stage baselines
bash scripts/mattr/campaign_wsl.sh --out "$OUT" --stage checkpoints
bash scripts/mattr/campaign_wsl.sh --out "$OUT" --stage report
bash scripts/mattr/campaign_wsl.sh --out "$OUT" --stage confirmation
```

Keep the same `OUT` when restarting. `--stage all` runs these stages in order.
Each job has a console log, source/configuration identity, immutable Adam/RNG
checkpoints, saved rankings, actual forward event log, stage accounting and
completion marker. Do not run multiple GPU jobs concurrently. Changing a
configuration requires a new output directory. A completed job is reused;
retraining does not happen when analyzing existing artifacts.

Official source: <https://github.com/aryamanarora/matryoshka-attribution>,
commit `af82b2eef145dd3e66c793a413352d6931a152d7`.
The official setup pins its MIB fork to `582fdbc8a2b0f7063a265c8f94851fd2a3aa5480`.
That fork's seven-value evaluator is required; the older Sketch-ACDC clone
is incompatible. Package/source/lock hashes are recorded per run. Setup
aborts if the official checkout has local edits.

The uncapped reference follows the documented 500-step node command. The
scaling comparison uses nested eligible training pools of 16/32/64/128,
three seeds, 500 updates and one fixed 128-example validation cohort.
MAttr samples with replacement: the eligible pool, actually visited unique
examples and total presentations are separate quantities. EAP/EAP-IG use
the same eligible pools and evaluation cohort, but different compute budgets.
Their attribution batch size is 1 to give every example equal weight under
the official mean-loss implementation; evaluation batch size remains 20.
The public test split is reserved for a frozen final confirmation: after all
validation experiments, the highest mean MAttr validation-CPR pool is locked,
and all methods use that pool and the three preset seeds on a fixed new
128-example public-test cohort. The pinned official README explicitly permits
`--split test`. Ranking hashes and the test protocol are locked before loading
test data; test scores cannot alter the selection. The reference ranking is also
reevaluated on the common validation cohort.

The local code calls the official trainer, node EAP functions and evaluator.
It does not redefine CPR/CMD. Checkpoint resumption restores raw scores,
Adam and Python/Torch RNG state before the next upstream update. Donor
activation caches are rebuilt on restart; their extra real forwards remain
part of measured cost. Interrupted work since the last checkpoint is
repeated and is not claimed as a compute saving.

Direct single-job commands are available in `run_wsl.sh` and
`evaluate_wsl.sh`; run with `--help`. Batch sizes and the donor cache budget
are explicit. CUDA OOM fails with guidance; no dataset or optimization
budget is silently reduced. The notebook only dispatches these WSL scripts
and reads output tables. Optional interaction experiments remain disabled.
