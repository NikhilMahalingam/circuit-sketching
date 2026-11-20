# Gate 6B seed ensemble runbook

Use the existing pinned WSL Python environment and local Gate 6A campaign. Preflight needs the historical checkpoints, rankings, evaluation receipts, and official dataset cache. Keep the full campaign local.

```bash
cd '/mnt/h/GenAI Research/sketch_prune'
git switch research/mattr-faithful-ensemble
bash scripts/gate6b/00_preflight.sh
bash scripts/gate6b/01_freeze.sh
```

Preflight verifies the 13 input rankings and 11 training checkpoints, builds the three rank ensembles, and selects the disjoint validation cohort. It also runs a bounded CUDA check on eight discovery rows. Freeze binds the clean implementation commit and writes the protocol and SHA-256.

```bash
bash scripts/gate6b/run_all.sh
```

The equivalent individual stages are:

```bash
bash scripts/gate6b/02_eval.sh
bash scripts/gate6b/03_analyze.sh
bash scripts/gate6b/04_audit.sh
```

Use `02_eval.sh --ranking m1_ens5` for one registered ranking. Rerun the same command after an interruption. Completed ranking receipts and exact circuit caches are checked before reuse. For a fresh campaign, pass the same absolute `--out` under `results/mattr_ensemble/` to each command. All commands support `--help` and `--dry-run`. The default campaign is `results/mattr_ensemble/gate6b`.

If the projection exceeds two hours, inspect `runtime_projection.json`. To approve that runtime, rerun the same evaluation command or `run_all.sh` with `--approve-over-2h`. On an integrity, CUDA memory, or code-lock failure, preserve the campaign and inspect its failed attempt. Changes to source after freeze require a new implementation commit and a fresh protocol.

After successful audit, review the campaign report and stage the small outputs.

```bash
bash scripts/gate6b/05_package.sh --dry-run
bash scripts/gate6b/05_package.sh --commit
git push -u origin research/mattr-faithful-ensemble
```

The completed report includes the frozen classification and an empty researcher interpretation section. The repository package uses summary tables, plots and protocol metadata. Per-prompt evaluation arrays and model artifacts stay local.
