## Outcome interpretation

Frozen classification: **Ranking-quality effect**. Primary capped-T1 success: **True**. Official pruning passed monotonicity and nestedness for every audited prefix. The fresh nominal 0.5% gain was **36.85 percentage points** (paired 95% interval 34.11 to 39.58). At capped **T1=116**, hierarchy averaged **54.30%** accuracy versus **19.79%**, a **34.51-point** difference (95% interval 31.84 to 37.17); seed changes 42/43/44 were **+40.82, +7.03, +55.66 points**. **3/3** seeds improved. These are positive-margin IOI accuracies, separate from vocabulary top-1.

The **advantage-retention ratio R is 93.64% (paired 95% interval 90.94% to 96.28%)**. It is not a causal proportion explained by pruning. The registered primary comparison is capped: seed 42 compares 116 against 116 actual edges; seeds 43/44 compare 114/113 hierarchy edges against 116 baseline edges. Only seed 42 is exact at T1. Across all 10 targets and 3 paired seeds, **13/30** comparisons are exact. The exact subsets and their prompt intervals are reported below; capped success does not establish three-seed exact-size superiority.

At secondary **T2=296**, the pooled hierarchy-minus-baseline difference was **14.26 points** (descriptive 95% interval 12.76 to 15.63). Actual sizes at T2 are 296/295 for baseline/hierarchy seed 42, and 296/296 for seeds 43/44. This pooled gain is driven by seed 42; the exactly matched seeds 43/44 average a small -0.39-point difference (interval includes zero). At 400 actual edges the exactly matched mean gain is +1.89 points; at 640 it is zero. Dense curves therefore show a compact benefit that fades as accuracy saturates. These secondary endpoints do not change the primary decision.

Control mean positive-margin accuracy:

| method | 116 | 296 |
| --- | --- | --- |
| baseline | 0.1979 | 0.7520 |
| edge_l2 | 0.1979 | 0.7461 |
| gamma_zero | 0.5547 | 0.8958 |
| hierarchy | 0.5430 | 0.8945 |
| shuffled | 0.3366 | 0.8561 |

Secondary paired contrasts (hierarchy minus comparator, accuracy fractions):

| comparator | target | mean | lower | upper | positive_seeds |
| --- | --- | --- | --- | --- | --- |
| baseline | 116 | 0.3451 | 0.3184 | 0.3717 | 3.0000 |
| gamma_zero | 116 | -0.0117 | -0.0208 | -0.0026 | 0.0000 |
| shuffled | 116 | 0.2064 | 0.1849 | 0.2285 | 2.0000 |
| edge_l2 | 116 | 0.3451 | 0.3184 | 0.3717 | 3.0000 |
| baseline | 296 | 0.1426 | 0.1276 | 0.1563 | 2.0000 |
| gamma_zero | 296 | -0.0013 | -0.0059 | 0.0033 | 0.0000 |
| shuffled | 296 | 0.0384 | 0.0221 | 0.0540 | 3.0000 |
| edge_l2 | 296 | 0.1484 | 0.1335 | 0.1634 | 2.0000 |

A gamma-zero tie or advantage means the residual penalty is not established as the cause of the compact gain. Real grouping beat shuffled grouping by 20.64 points at T1 and 3.84 points at T2 on these capped comparisons; the secondary prompt intervals above exclude zero. These findings favor a structural parameter-sharing explanation over residual regularization alone, conditional on the fixed rankings and cohort. They do not establish training-seed significance. No group-stability correlation alone establishes causal fidelity.

The baseline ensemble reached **93.95%** accuracy at capped T2 (actual **296** edges); the prespecified 90% diagnostic **passed**. Its component seed 42/43/44 accuracies there were **26.37%/99.80%/99.41%**. This reduces the poor seed-42 outcome but does not beat the strongest seed. All ensemble comparisons and their three-run training costs remain separate from marginal construction/inference costs.

Closure mean changes in official CPR/CMD:

| method | CPR_delta | CMD_delta |
| --- | --- | --- |
| baseline | 0.0072 | 0.0028 |
| edge_l2 | 0.0049 | 0.0024 |
| gamma_zero | 0.0063 | 0.0029 |
| hierarchy | 0.0059 | 0.0036 |
| shuffled | 0.0045 | 0.0028 |

MIB integrates the nominal fraction coordinates, so closure changes are **nominal-grid counting effects**, with actual sizes listed in closure_nominal_effects.csv. They are not matched-size ranking-quality evidence. No matched-size closure evaluations were run. Closure audit covers all known first-appearance boundaries from original prefixes <= 1624; appended-tail nesting samples do not establish a pruning-free tail. Diagnostic rank correlations use canonical ordinal tie breaks; original ranking evaluation retains the pinned official Torch sort.

Original nominal mean CPR was **10.006862 baseline versus 9.962688 hierarchy**; CMD was **9.023285 versus 8.973372**. Thus the compact accuracy benefit did not improve leaderboard CPR. CPR, CMD, compact decision recovery and full-model fidelity remain distinct. Gate 5D's mixed decision is preserved.

Measured inference-process wall time: **58.11 minutes**, evaluation-stage wall time **57.61 minutes**, CPU topology/construction **200.77 seconds**. There were **19,956** real root forwards and **391,332** example-forwards, with **166** exact-output cache hits. CUDA peaks: **2.40 GiB allocated**, **5.91 GiB reserved**; these are not physical GPU occupancy or kernel-utilization timers. Zero optimizer updates or backward calls were executed.

Registered-target skip lists for all 18 original/ensemble rankings:

| ranking | exact_registered_targets | skipped_registered_targets | exact_count | registered_targets |
| --- | --- | --- | --- | --- |
| baseline_s42 | [80, 100, 116, 180, 296, 400, 640] | [60, 140, 240] | 7 | 10 |
| baseline_s43 | [60, 100, 116, 140, 180, 240, 296, 400, 640] | [80] | 9 | 10 |
| baseline_s44 | [60, 116, 140, 180, 240, 296, 400, 640] | [80, 100] | 8 | 10 |
| hierarchy_s42 | [100, 116, 240, 400, 640] | [60, 80, 140, 180, 296] | 5 | 10 |
| hierarchy_s43 | [80, 100, 140, 240, 296, 400, 640] | [60, 116, 180] | 7 | 10 |
| hierarchy_s44 | [80, 296, 400, 640] | [60, 100, 116, 140, 180, 240] | 4 | 10 |
| gamma_zero_s42 | [100, 180, 240, 296, 640] | [60, 80, 116, 140, 400] | 5 | 10 |
| gamma_zero_s43 | [60, 100, 180, 296, 400] | [80, 116, 140, 240, 640] | 5 | 10 |
| gamma_zero_s44 | [100, 296, 400, 640] | [60, 80, 116, 140, 180, 240] | 4 | 10 |
| shuffled_s42 | [60, 80, 100, 140, 180, 240, 296, 400, 640] | [116] | 9 | 10 |
| shuffled_s43 | [60, 100, 140, 180, 240, 296, 640] | [80, 116, 400] | 7 | 10 |
| shuffled_s44 | [60, 140, 180, 400, 640] | [80, 100, 116, 240, 296] | 5 | 10 |
| edge_l2_s42 | [80, 100, 116, 140, 180, 296, 400, 640] | [60, 240] | 8 | 10 |
| edge_l2_s43 | [60, 100, 116, 140, 240, 296, 400, 640] | [80, 180] | 8 | 10 |
| edge_l2_s44 | [60, 80, 140, 180, 240, 400, 640] | [100, 116, 296] | 7 | 10 |
| baseline_ensemble | [80, 100, 116, 140, 180, 240, 296, 640] | [60, 400] | 8 | 10 |
| hierarchy_ensemble | [80, 100, 116, 140, 296, 400] | [60, 180, 240, 640] | 6 | 10 |
| gamma_zero_ensemble | [100, 116, 180, 240, 296, 640] | [60, 80, 140, 400] | 6 | 10 |

Recommendation: prioritize a future independent evaluation of the existing gamma-zero component parameterization and rank ensembles for compact circuits, while retaining standard MAttr as the CPR reference. The penalty has not demonstrated value here. Treat closure as a separate nominal-counting direction and charge ensembles for three trained runs. This is a recommendation only. Stop after Gate 5E; no further gate, training, new mask/refinement algorithm, or leaderboard submission has run.
