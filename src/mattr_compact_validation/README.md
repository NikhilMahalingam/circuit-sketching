# Gate 5D: compact-circuit generalization

Inference only: all five frozen Gate 5C methods, all three seeds, exactly
1,000-step checkpoints. No trainer or optimizer is invoked.

Use the existing pinned WSL environment; no installation or environment changes:

```bash
bash src/mattr_compact_validation/run_wsl.sh tests
bash src/mattr_compact_validation/run_wsl.sh --stage prepare
bash src/mattr_compact_validation/run_wsl.sh --stage inference
bash src/mattr_compact_validation/run_wsl.sh --stage report
bash src/mattr_compact_validation/run_wsl.sh --stage audit
```

Preparation selects the official validation cohort and freezes checkpoint/source,
cohort, template, grid, bootstrap, success and stopping rules before any new-cohort
model inference. Existing completed inference outputs cannot be overwritten.
Batch size is 20, as in Gate 5C. OOM or correctness failures stop the run; no
automatic changes to the frozen protocol or evaluation behavior are made.

The unchanged official MIB evaluator computes CPR/CMD, positive-margin accuracy
and accuracy log-AUC. A read-only metric recorder captures per-prompt margins and
the official full-vocabulary top-1 diagnostic from the same logits. Only
graph-independent official reference scalars are cached. No clean or mutable
patched activations are reused.

Bootstrap draws are shared over prompts across seeds and methods, stratified by
the exact official `template` column. The trained seeds are fixed paired
replications, never independent prompt rows. Primary simultaneous intervals use
Bonferroni over two endpoints. Control comparisons have a separate six-endpoint
family. Secondary AUC intervals are unadjusted.

Prompt manifests, raw checkpoints, and activation/model data stay local. The
versioned cohort file contains official row indices, hashes, token IDs and exact
template hashes rather than prompt text. All historical results are hash-checked.
The notebook defaults to saved-results analysis with inference disabled.
