# MAttr optimization: Gate 1 official edge baseline

Campaign: `20261021T134352Z_gate1`. **Gate 1 completed.** Only the official baseline was run; Gate 2 and all algorithm modifications remain unrun.

## Protocol and provenance

Pinned MAttr `af82b2eef145dd3e66c793a413352d6931a152d7`, required MIB fork `582fdbc8a2b0f7063a265c8f94851fd2a3aa5480`; exact EAP commit, source/lock hashes, installed versions, command argv and execution-source snapshots are saved. A dedicated WSL Python 3.12 environment uses the official frozen dependency lock on RTX 3060 12 GB. Older environments and experiments were not changed.

GPT-2 Small, official MIB IOI, edge granularity (32,491 real scores), seed 42, 5,000 requested iterations; 5000 logged updates. The uncapped official train pool has 10000 rows and the uncapped validation split has 10000 rows, both in original official order. Clean prompts are distinct across splits and answer names were checked to be single tokens. No test split or custom IOI dataset was accessed.

Uniform floating k; sigmoid top-k; iso/denoising; zero score initialization; Adam lr 0.05, epsilon 1e-8, betas (0.9,0.999); T=.5; 30 bisection iterations. One example is sampled with replacement per training iteration. Evaluation batch=20, donor-cache budget=4.0 GB; explicit resource overrides relative to upstream defaults: {'evaluation_batch_size': False, 'counterfactual_cache_gb': False}. Float32/TF32 disabled. Upstream model-weight gradient accumulation is preserved; no freezing or algorithm optimization was introduced.

The adapter calls the upstream edge CLI, trainer, live-sender intervention closure and official signed edge evaluator. It adds counters, checkpointing and audited raw-vector exports. Versioned checkpoints restore scores, Adam and Python/NumPy/Torch/CUDA RNGs. Cached evaluation vectors are reused only after exact ranking, source/configuration, data fingerprint and vector-hash checks; reuses incur zero model forwards. Setup/replayed work is included in the complete accounting. No settings change automatically after OOM.

## Official validation results

- **CPR AUC: 10.58801242**.
- **CMD AUC: 9.60080915** (lower absolute deviation from full-model behavior).
- Accuracy log-AUC: 0.73720560.
- Clean mean logit difference: 3.24274755; official empty-circuit corrupted reference: -2.72592449.
- Maximum discrepancy in four post-training hard-mask checks against official MIB: 1.9073486e-06, within atol=5e-5/rtol=1e-5.

| Requested fraction | Requested edges | Faithfulness | Accuracy |
|---:|---:|---:|---:|
| 0.001 | 32 | 0.000000 | 0.0481 |
| 0.002 | 64 | -0.000011 | 0.0481 |
| 0.005 | 162 | 0.150415 | 0.1198 |
| 0.01 | 324 | 4.640131 | 0.9970 |
| 0.02 | 649 | 10.454853 | 1.0000 |
| 0.05 | 1624 | 13.024123 | 1.0000 |
| 0.1 | 3249 | 13.768633 | 1.0000 |
| 0.2 | 6498 | 14.088088 | 1.0000 |
| 0.5 | 16245 | 14.305707 | 1.0000 |
| 1 | 32491 | 1.000000 | 0.9740 |

Requested counts are before official connectivity pruning; weighted edge counts are saved separately. CPR uses the official linear retained-fraction trapezoids, with no added zero endpoint and no clipping. Accuracy log-AUC is a different metric. Values above one reflect overshoot; a larger CPR alone does not establish improved mechanistic fidelity. Raw per-example clean, corrupted and ten grid-point validation vectors, the learned ranking and score tensors are saved. Independent integration of the saved curve agrees with official CPR/CMD within 1e-6.

The post-run verifier exports `official_nested_ranking.csv` with the actual official Torch tie ordering, and `official_circuit_membership_summary.csv` with pruned circuit sizes and identity hashes at each grid point. `ranking.csv` is a score-sorted presentation table; use the official ranking/graph procedure when ties matter. This export changes no trained scores or benchmark results and executes no model forwards.

## Published comparison and research conclusion

The [paper v1, Table 4](https://arxiv.org/html/2609.25518v1) reports GPT-2/IOI edge validation CPR **10.59** for its 5,000-step recipe. Our result differs from that rounded target by **-0.001988**. Rounded-value agreement, if present, cannot establish precision beyond the published rounding; otherwise this is a reproduction gap to investigate before claiming an improvement. Hardware/library behavior, precise evaluation protocol, revision and numerical precision must be considered. Our earlier node-level CPR 1.83 and Gate-0 small-cohort CPR are not matched edge-baseline comparisons.

This one seed establishes the measured reference and feasibility, not improvement over MAttr, statistical significance, or leaderboard performance. Sampling, initialization and ranking refinement have not been tested. No success threshold has been selected from variant results.

## Compute and memory

- Training, including checkpoint work: 1616.4 seconds (26.94 min).
- Full official validation evaluation, including raw-vector export: 2536.3 seconds (42.27 min).
- Instrumented model/data/correctness/training/evaluation stages: 4180.3 seconds (69.67 min); interpreter startup/source verification/reporting are outside these stage totals.
- Measured process work after source/config preparation: 4217.3 seconds (70.29 min), including upstream graph construction, pretokenization, exports and time between instrumented stages. Source checks, interpreter startup and report generation are additional overhead.
- Resume data-preflight CPU work outside that process timer: 0.0 seconds; zero model forwards. The WSL wrapper verifies saved row contents/order before any restart.
- Maximum allocated CUDA memory across stages: 6.799 GiB; maximum reserved: 12.393 GiB.
- Maximum sampled CPU RSS across stages: 2.905 GiB.
- Real root forward batches: 21788; individual example-forwards: 249788; backward calls: 5000; optimizer updates this and archived attempts: 5000.
- Eligible train rows: 10000; actual unique sampled rows across attempts: 3925; actual presentations including repeated work: 5000.

CUDA synchronizes at stage boundaries. Counters observe actual root forwards, not counterfactual-cache hits; activation-checkpoint recomputation inside backward is not a new root forward and remains part of measured backward/training cost. Training has required score-gradient graphs; held-out evaluation and hard-mask correctness checks use no-grad. No full-model weight updates occur. Allocation peaks measure PyTorch allocator usage, not all system GPU memory.

The raw upstream console uses calendar/time.time timestamps; WSL clock jumps were observed during this run, including a negative printed pretokenization duration. Those calendar-based durations and progress rates are not used for compute conclusions. The accounting and process timers above use monotonic perf_counter with CUDA synchronization.

## Next gate

Stop here. Gate 2 should run unchanged edge MAttr on nested 16/32/64/128 discovery pools with one fixed validation cohort, seed 42 first, then 43/44 after feasibility. Use these runs to characterize baseline noise and freeze improvement criteria before testing algorithm variants. No automatic sweep, multi-model evaluation or leaderboard submission is scheduled.
