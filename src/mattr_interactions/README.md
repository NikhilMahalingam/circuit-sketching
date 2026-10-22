# Gate 5B interaction diagnostics

Uses the existing pinned WSL MAttr/MIB/TransformerLens environment. No trainer,
optimizer, learned masking variant, dependency change, or leaderboard submission.

```bash
bash src/mattr_interactions/run_wsl.sh tests
OUT=results/mattr_interactions/20261022_gate5b
bash src/mattr_interactions/run_wsl.sh --out "$OUT" --prepare-only
bash src/mattr_interactions/run_wsl.sh --out "$OUT"
bash src/mattr_interactions/run_wsl.sh --out "$OUT" --report-only
bash src/mattr_interactions/run_wsl.sh audit --out "$OUT"
```

Preparation uses saved 5,000-step Gate 2 scores and the canonical official graph.
It freezes candidate edges, pairs, protocol and source hashes before intervention
effects. After train-only screening it freezes confirmation pairs and matched
random controls before loading the new validation cohort. Reusing a completed
campaign regenerates analysis with no model forwards. A failed attempt is
preserved and requires inspection rather than silent parameter changes.

All interventions call the official `eap.evaluate.evaluate_graph`, including its
normal connectivity cleanup. We exclude toggles where cleanup would alter any
additional edge. The corrected official MAttr live loss closure is captured as a
hard-mask correctness oracle; `learn_scores` is intercepted and never executed.
Dependent paths, cache equivalence, determinism and singleton batching are tested
on real CUDA. No historical frozen-clean sender patcher is imported.

Donor-only matrices are copied into a fresh official activation-difference tensor.
Live patched source outputs are always recomputed. The CPU donor LRU is capped at
2 GiB; exact evaluation caches include model/protocol, cohort and mask identities.
CUDA timers synchronize, actual forwards and examples are counted independently,
and no inference gradients are stored. OOM does not change settings automatically.

The notebook defaults to analysis of existing artifacts. Raw masks, score vectors,
checkpoints, official prompt manifests and cached source activations stay outside
Git. Small summaries, numerical per-example effects, protocol, source hashes,
official curves where applicable, plots and the report are published. Gate 5B
measures margins and vocabulary top-1 accuracy; it does not create CPR/CMD proxies.

Component-size controls use the same random permutation of edge identities across
all three seeds, preserving existing cross-seed covariance. Prompt bootstraps use
shared draws across pairs. Fresh validation means disjoint from small development
cohorts; historical Gate 1 evaluated the complete official validation split.
