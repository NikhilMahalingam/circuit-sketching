# Gate 5C: regularized hierarchical MAttr

Uses the existing pinned WSL environment and original official MAttr training
loop, mask gradients, edge patcher and MIB evaluator. Every one of the 32,491
effective edge scores remains available. GPT-2 weights never enter the optimizer;
their full SHA-256 is checked before/after. Original model-gradient flags are
preserved for paired numerical behavior. No model weights are updated.

```bash
bash src/mattr_hierarchy/run_wsl.sh tests
OUT=results/mattr_hierarchy/20261022_gate5c
bash src/mattr_hierarchy/run_wsl.sh --out "$OUT" --prepare-only
bash src/mattr_hierarchy/run_wsl.sh --out "$OUT"
bash src/mattr_hierarchy/run_wsl.sh diagnostics --out "$OUT"
bash src/mattr_hierarchy/run_wsl.sh audit --out "$OUT"
```

Preparation freezes a training-only magnitude-check rule and all source hashes.
The pilot performs 20 gradient probes on training prompts and frozen ordinary
500-step scores, with no updates or validation. It then freezes the complete
protocol/scale before any modified CPR result. Run order is correctness smoke,
fresh dedicated baseline calibration, four seed-42 variants, and only the
registered conditional controls/confirmation. No qualifying screen means stop.
The original gamma values are .001/.01; a replacement scale is permitted only by
the frozen train-gradient/loss magnitude rule, never by development CPR.

H1 has 157 source groups; H2 has 445 destination groups separating Q/K/V. Input
and logits are explicit official schema categories. Both have one disjoint
canonical edge-to-group map. Shuffle5701 preserves exact group-size counts and
is fixed across seeds. Effective scores are h[group] + centered residual, with
gamma * scale * mean(centered_residual squared). Components are not penalized.
One packed parameter contains h and r and receives the unchanged elementwise
official Adam update. A deterministic padded indexed sum implements both the
group projection and its transpose, avoiding atomic scatter ordering.

Raw packed checkpoints save Adam, RNG, group identity and complete histories.
They are never treated as rankings. Official evaluation/export always receives
the effective 32,491 scores. Gamma-zero hierarchy is a different parameterization;
only the explicit bypass claims ordinary-MAttr equivalence. Exact CPU and CUDA
resume checks restore optimizer, score and Python/NumPy/CPU/CUDA RNG state.

Run `--report-only`, followed by `diagnostics --out "$OUT"` and `audit --out "$OUT"`, to regenerate analysis
without model forwards. The notebook defaults to saved-artifact analysis and
never launches training unless its switches are changed explicitly. Small
summaries, logs, configs and plots are versioned. Packed checkpoints, residual
and effective vectors, full rankings, memberships, adjacency/group arrays and
prompt manifests remain local. Compressed canonical edge/group mappings are
included for reproducibility. No leaderboard submission or subsequent algorithm.

Final reporting: after diagnostics and audit, run Python module src.mattr_hierarchy.reconcile_accounting from the repository root. It preserves original stage aggregates and removes the nested pilot counter overlap using unique process counters; no frozen training source or selection decision changes.
