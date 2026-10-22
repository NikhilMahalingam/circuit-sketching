# Prefix-cached ACDC: independent MIB adaptation

Campaign: `results/acdc_prefix_cache/20261019T173713Z`.

## Scope and independent reference

The supplied Downloads/RESEARCH_LOG.md, Stage J, reports 2.35× faster execution, bitwise loss agreement and a later replay of all 32,491 decisions with the same 686-edge GPT-2 circuit. It describes auto-circuit single-template IOI, ABC donors, and squared full-vocabulary probability distance. The reported full replay took 2.5 hours under power capping. Those are independent author-reported measurements, not measurements from this repository. The complete Stage-J implementation and original trace are not supplied or present in the inspected clones; source-log identity and the relevant excerpt are saved in independent_source.json. Exact reconstruction of that original experiment has not been performed.

This implementation instead uses official MIB IOI train/validation data, GPT-2 Small, float32, and the official negative logit difference (get_metric(..., loss=True)). A deletion is accepted when proposed mean loss minus current mean loss is strictly below the fixed threshold. This signed rule matches the default ACDC abs_value_threshold=False; the historical restricted Sketch-ACDC absolute-change decision rule is not silently reused. The threshold, traversal and starting graph are identical between the two new arms. This is an optimization reproduction on MIB IOI, not a numerical reproduction of the independent 2.35× experiment.

## Implementation and safety

The standard executor always runs the full transformer. The cached executor uses TransformerLens.forward(start_at_layer=L) with a cloned full-model residual stream before block L, the original token IDs and the original padding mask. Residual connections and all attention/MLP operations within the suffix are executed by TransformerLens itself. A final-logit receiver uses the last block rather than bypassing its hook_resid_post. Block zero falls back to a full forward.

The prefix proof uses the actual full MIB DAG. Every removed edge changes its receiver; its descendant closure is propagated through all graph dependencies. Cached sender nodes before the proposed boundary must have empty intersection with that closure, and no removed receiver hook may be skipped. The union includes all previously accepted removals, not only the current candidate. Arbitrary out-of-order states therefore move the boundary earlier when necessary; explicitly unsafe boundaries are rejected. Unknown nodes, unsupported architectures, model/config changes and incompatible checkpoint identities fail rather than silently reusing stale state.

Both arms cache identical fixed corrupted donors. Only the optimized arm also retains clean sender tensors and residual boundaries. At a skipped, proven-unaffected prefix, donor-minus-clean is valid. At every sender in the suffix, donor-minus-CURRENT-intervened output is recomputed. The matrix update is checked against the corrected shared receiver_patch function, corrected EdgeExperiment on dependent real-model queries, and the independent official MIB evaluate_graph. The historical FrozenSenderEdgeExperiment is never used. No gradients are tracked during inference. Repeated intervention queries are executed, not memoized.

Traversal sorts by descending receiver DAG phase, descending sender index, then edge name. Attention and MLP have separate within-block phases. The compatible exact search tests each real edge independently and performs no automatic cascading deletions or node-skipping shortcuts. The official MIB oracle does its own graph.prune(); its losses are independently checked, including a disconnected-head case. Saved retained identities refer to the explicit tested edge mask, not a relabeled canonical graph. This traversal/control flow is not claimed to match an unavailable auto-circuit trace.

Configuration: discovery 16, validation 16, batch 2, seed 42, threshold 0.05, atol 5e-05, rtol 1e-05. Public test is not used. Existing installed WSL packages are reused without installing dependencies or importing MAttr experiment code. The older corrected MIB clone is selected explicitly; all previously tracked experiment artifacts are protected by SHA-256 checks.

## Stage A measurements

**Stage A passed.** 30 single-edge and 8 multi-edge queries cover receiver layers 0,2,4,6,8,10,11, q/k/v inputs, MLP inputs, final residual receivers, dependent senders, randomized masks, and an early previous deletion paired with a late candidate.

Across 608 per-example train comparisons: max absolute loss error **0**, mean absolute error **0**, exact matches **608/608**. Maximum meaningful relative error: 0. Maximum error against the independent official MIB oracle: **1.6212463e-05**.

The restricted trajectory checks 32 edges selected across early/middle/late layers in reverse-topological subsequence order. The longer trajectory checks the first 64 edges of the full deterministic traversal. Candidate IDs, proposed losses, keep/remove decisions, before/after state hashes, and complete retained edge lists match. Interrupting, rebuilding caches and resuming the longer trajectory also matches all 64 decisions. In total, 160 saved comparison records include the resume replay; these are not 160 unique full-graph decisions.

**No full 32,491-edge search was executed.** Matching these partial traces does not establish full-trajectory correctness or full-search speedup. Stage A uses diagnostic timings, not a repeated, isolated end-to-end performance experiment.

Ten real small-TransformerLens unit tests passed. The GPU gate recorded 3,467 actual root forward batches, 6,934 example-forwards and 26,948 actual block executions across both arms, independent oracles, warmups, replay and held-out checks. Every recorded inference forward disabled gradients.

Held-out validation checks of the two final partial circuits:

| Trajectory | Examples | Standard loss | Cached loss | Max discrepancy |
|---|---:|---:|---:|---:|
| restricted_32 | 16 | -2.4759810 | -2.4759810 | 0 |
| partial_prefix_64 | 16 | -8.1599121 | -8.1599121 | 0 |

Maximum sampled Stage-A preparation allocator peak: 1226.6 MiB allocated / 1380.0 MiB reserved. Standard donor cache: 138.9 MiB; optimized donor + clean-prefix cache: 288.4 MiB. Both executors coexist for correctness testing, so these allocator peaks are **not** isolated per-method VRAM benchmarks. Stage B measures isolated allocations.

## Planning estimates before Stage B/C

These are rough planning estimates, not measured benchmark results, speedup claims, confidence intervals, or upper bounds. They scale Stage-A diagnostic query times linearly with discovery size and interpolate missing layers from nearby measured boundaries. Growing deletion-mask bookkeeping, checkpoint I/O, validation, microbenchmark warmup, GPU clocks and caching memory pressure can change runtime substantially. Stage B is needed to calibrate them. Stage-C rows below are one standard/cached full-graph pair; repeating all sizes is potentially hours-long.

| Discovery examples | B medium pairs (configured repeats) | C one full pair | C configured repeats |
|---|---:|---:|---:|
| 16 | 3.3 min | 3.71 h | 11.14 h |
| 32 | 6.7 min | 7.43 h | 22.28 h |
| 64 | 13.4 min | 14.85 h | 44.55 h |
| 128 | 26.8 min | 29.70 h | 89.10 h |

Stage B additionally includes seven-layer repeated microbenchmarks and held-out evaluations. The first traversal prefixes are concentrated at late receivers and must be labeled partial trajectories. Their end-to-end speedup cannot be promoted to a full-search speedup.

## Performance status and next action

Stage B: completed. Stage C: NOT RUN; disabled by default. Only completed stages support the measured comparisons below; partial trajectories do not establish full-search performance. The small real-model unit tests are structural correctness tests, not a second standardized model/task benchmark.

Next authorized checkpoint: review the implementation, numerical agreement and estimates before executing Stage B or C. No sketching, Hessians, group deletion or MAttr algorithm has been implemented. Possible later optimization: avoid conservative full-graph dependency propagation for nodes proven unable to affect logits, but that requires a separate correctness proof and is not implemented.

Measured end-to-end discovery speedup includes per-arm tokenization, constructor, donor/cache construction, synchronization, mask bookkeeping, exact decisions, checkpoint I/O and previous failed attempts. Common model/dataset loading and warmups are separate; validation time and including-validation totals are also saved. Medians and interquartile ranges are reported across paired trials.

- Stage B, n=16, checks=64: median speedup 3.382×, IQR [3.175, 3.516], 3 paired trials. Partial trajectory.
- Stage B, n=16, checks=128: median speedup 3.084×, IQR [2.987, 3.190], 3 paired trials. Partial trajectory.
- Stage B, n=32, checks=64: median speedup 3.494×, IQR [3.467, 3.730], 3 paired trials. Partial trajectory.
- Stage B, n=32, checks=128: median speedup 3.545×, IQR [3.523, 3.596], 3 paired trials. Partial trajectory.
- Stage B, n=64, checks=64: median speedup 3.607×, IQR [3.396, 3.794], 3 paired trials. Partial trajectory.
- Stage B, n=64, checks=128: median speedup 4.033×, IQR [3.897, 4.046], 3 paired trials. Partial trajectory.
- Stage B, n=128, checks=64: median speedup 4.015×, IQR [3.943, 4.117], 3 paired trials. Partial trajectory.
- Stage B, n=128, checks=128: median speedup 4.078×, IQR [4.024, 4.176], 3 paired trials. Partial trajectory.
- Stage C, n=16, checks=32491: median speedup 1.723×, IQR [1.723, 1.723], 1 paired trials. Full graph.
