# MAttr edge discovery-data scaling

Campaign: `20261021_gate2_seeds43_44`. Completed jobs: 8. Seeds: [43, 44]. Each job preserves the official 5,000-iteration edge recipe; only eligible discovery-pool size and preset seed differ. Training samples with replacement; per-example reuse counts and actually visited unique rows are saved.

All jobs use the same deterministic official validation cohort (seed 42). Pool subsets are nested within seed. Validation metrics are from the official MIB evaluator; training quality is diagnostic and never updates scores. Frozen rankings at 500, 1,000 and 5,000 iterations are evaluated after training, without perturbing the optimizer trajectory. No test split or algorithm variant is used. These cohort scores are not directly compared with Gate 1's full 10,000-example validation score.

`sample_efficiency.csv` contains final train/validation CPR and CMD, gaps and measured compute. `quality_by_step.csv`, `faithfulness_curves.csv`, `compute_accounting.csv` and `training_example_usage.csv` retain the detailed measurements. Model/data/evaluation stage timing includes real work and archived measured attempts. Interpreter startup, source verification, report rendering and gaps between stages are additional overhead; process summaries retain broader per-job elapsed time. Force-killed attempts without terminal timers are flagged, so their reported time must be treated as a lower bound.

One seed is exploratory. Even three seeds provide preliminary evidence only. No automatic success threshold, best-pool selection, algorithm modification or claim that 128 examples are sufficient is made by this report. Compare CPR, CMD, compact-circuit faithfulness and variability before deciding the next experiment. Gate 3 is not dispatched.
