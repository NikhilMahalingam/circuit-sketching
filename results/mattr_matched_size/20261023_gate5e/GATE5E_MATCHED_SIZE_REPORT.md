# Gate 5E: Matched-size circuit evaluation and ranking robustness

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

## Actual frozen decision

**Ranking-quality effect**. Primary capped-T1 success: **True**. T1=**116**, T2=**296**, computed as medians over all15 saved Gate5D actual counts, not hardcoded estimates. Full decision and advantage-retention ratio:

```json
{
  "primary_success": true,
  "checks": {
    "positive_mean": true,
    "positive_at_least2_seeds": true,
    "paired_prompt_lower_positive": true,
    "correct_evaluator_and_masks": true
  },
  "cap_T1": {
    "mean": 0.3450520833333333,
    "lower": 0.318359375,
    "upper": 0.37174479166666663
  },
  "nominal_0.5pct": {
    "mean": 0.3684895833333333,
    "lower": 0.3411458333333333,
    "upper": 0.3958333333333333
  },
  "positive_cap_seeds": 3,
  "per_seed_cap_changes": [
    0.408203125,
    0.0703125,
    0.556640625
  ],
  "per_seed_nominal_changes": [
    0.400390625,
    0.150390625,
    0.5546875
  ],
  "advantage_retention_ratio": {
    "stable": true,
    "point": 0.9363957597173145,
    "lower": 0.9094201289674878,
    "upper": 0.9628318584070796,
    "denominator_nearzero_threshold": 0.001953125,
    "invalid_bootstrap_denominators": 0,
    "causal_explained_fraction": false
  },
  "classification": "Ranking-quality effect",
  "scope": "capped-size comparison; no causal fraction or training-seed significance claim",
  "new_training_authorized": false
}
```

Gate5D remains mixed/inconclusive; its protocol, results and failed seed-consistency criterion were not revised. This inference-only gate tests whether the nominal advantage remains when circuits are capped by actual edge count. It does not train/optimize MAttr, modify scores by gradient, fine-tune model weights or introduce differentiable masks. Closure and equal-rank ensembles are the only prespecified exploratory ranking constructions.

## Official pruning and size counting

iteratively remove nodes lacking incoming or outgoing edges, then edges with absent parents/children, until both masks stabilize; input treated as having incoming edges, logits receiver forced present. Q/K/V map to their shared head node for node-presence checks; no new edges are added.

**MIB counts grid size nominally**, as `int(percentage * len(graph.edges))`, then runs official pruning. The evaluator returns actual weighted counts for reporting but integrates over the original percentage coordinates. CPR and CMD use the nominal linear grid; accuracy log-AUC uses nominal log-grid coordinates. About **1.401%** of the CPR quadrature weight lies at1% and below. Closure changes on that grid are legitimate training-free ranking/counting effects; they do not establish better equal-size circuits.

Pinned source references:

- `external/matryoshka-attribution/deps/MIB-circuit-track/EAP-IG/src/eap/graph.py:443` through line529 (apply_topn).
- `external/matryoshka-attribution/deps/MIB-circuit-track/EAP-IG/src/eap/graph.py:567` through line599 (prune).
- `external/matryoshka-attribution/deps/MIB-circuit-track/MIB_circuit_track/evaluation.py:12` through line140 (evaluate_area_under_curve).

All15 original signed rankings and18 total original/ensemble mappings retain every canonical edge identity. Official `apply_topn(...,absolute=False,prune=True)` alone determines every circuit. Source/model/checkpoint/effective-score hashes match Gate5D. Original tie semantics follow the pinned flattened PyTorch sort; only the prespecified ensembles use canonical-index tie breaks on signed ordinal ranks.

## CPU topology and closure correctness

PhaseA verified A(n) nondecreasing and P(n) nested for every n1..1624 for15 original rankings and3 ensembles, reproducing all saved Gate5D counts AND masks through5% exactly. All real edges are audited against the canonical32491-edge graph. Complete A(n), mask hashes, packed P(n), first appearances and jump statistics were saved. Maximum jumps and flat intervals are in jump_summary.csv/jump_distribution.csv. CPU topology/construction time: **200.767seconds**.

All **15** closure rankings passed their mandatory correctness tests. Every prefix through the last known first-appearance group was tested, not merely50samples: each group boundary returns exactly its originalP(n), with zero edges pruned; every nonboundary prefix obeys the lower/upper group sandwich, with consecutive nesting. Partial-group pruning is allowed. An additional50full-range samples check nesting only. Export order is deterministic and all32491 edges appear once. The initial group audit uses original n<=1624; **the appended tail is not claimed pruning-free**. Detailed scope and per-prefix checks are saved in closure_correctness.csv and topology/. No official pruning code was changed. Invalid closure rankings would have been skipped while retaining originals.

## Protocol and fresh cohort

Protocol SHA-256: `bb6421a5e5481a172559d0634ab9e71a90b942192f046a824b818bb4ecc90b39`. All original/derived ranking hashes, source/dependency pins, selections, targets, interpolation rules, cohort/hash manifests, metrics, bootstrap, thresholds and stop conditions were frozen before any fresh-cohort forward pass.

Official MIB IOI validation cohort: **512** prompts, selection seed5543, proportional stratification over nine exact official template categories. Exclusions cover all historical small development/confirmation cohorts, **all512 Gate5D prompts**, and discovery training prompts. Both clean and corrupted hashes and cross-role collisions are excluded/deduplicated, alongside official row IDs. Labels are verified single tokens. Raw prompts remain local; IDs/hashes and generic template labels are versioned. Gate1 historically evaluated the entire10000-example validation split: this cohort is fresh relative to smaller campaigns, **not untouched across all historical research or a private test**.

## Original nominal-size results

| ranking | kind | method | seed | CPR | CMD | accuracy_log_AUC |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_s42 | original | baseline | 42 | 9.8325 | 8.8585 | 0.6409 |
| baseline_s43 | original | baseline | 43 | 10.1458 | 9.1569 | 0.7637 |
| baseline_s44 | original | baseline | 44 | 10.0423 | 9.0545 | 0.7443 |
| hierarchy_s42 | original | hierarchy | 42 | 9.9584 | 8.9739 | 0.7530 |
| hierarchy_s43 | original | hierarchy | 43 | 9.9815 | 8.9909 | 0.7883 |
| hierarchy_s44 | original | hierarchy | 44 | 9.9482 | 8.9553 | 0.8244 |
| gamma_zero_s42 | original | gamma_zero | 42 | 9.9093 | 8.9248 | 0.7459 |
| gamma_zero_s43 | original | gamma_zero | 43 | 10.0219 | 9.0314 | 0.7862 |
| gamma_zero_s44 | original | gamma_zero | 44 | 9.9460 | 8.9532 | 0.8244 |
| shuffled_s42 | original | shuffled | 42 | 9.8078 | 8.8231 | 0.7239 |
| shuffled_s43 | original | shuffled | 43 | 9.7700 | 8.7769 | 0.8139 |
| shuffled_s44 | original | shuffled | 44 | 9.6059 | 8.6173 | 0.7435 |
| edge_l2_s42 | original | edge_l2 | 42 | 9.8100 | 8.8360 | 0.6414 |
| edge_l2_s43 | original | edge_l2 | 43 | 10.1124 | 9.1236 | 0.7633 |
| edge_l2_s44 | original | edge_l2 | 44 | 10.0286 | 9.0408 | 0.7443 |

These use the complete official ten-point grid. CPR/CMD and accuracy log-AUC come from the unchanged official MIB evaluator. Positive-margin accuracy and separate full-vocabulary top1 are recorded from the same logits. New-cohort clean/empty references are shared safely, without mutable patched-activation caches.

## Capped comparisons and exact subset

For each target, the largest audited n with A(n)<=T is used. There is no separate exact-size evaluation. An exact comparison means **both** circuits attain T; otherwise it is a capped comparison and the remaining actual-count difference is retained. At any attained count, nestedness makes the circuit unique. There are **675/1530** exact comparisons across all unordered pairs of18original/ensemble rankings and10targets (cross-seed pairs descriptive), including **13/30** paired baseline/hierarchy target comparisons. The latter span3fixed seeds x10targets; they are not30independent trials.

| target | seed | mean | lower | upper |
| --- | --- | --- | --- | --- |
| 116 | 42 | 0.4082 | 0.3672 | 0.4473 |
| 116 | 43 | 0.0703 | 0.0332 | 0.1055 |
| 116 | 44 | 0.5566 | 0.5156 | 0.5996 |
| 116 | equal_seed_mean | 0.3451 | 0.3184 | 0.3717 |
| 296 | 42 | 0.4355 | 0.3945 | 0.4766 |
| 296 | 43 | 0.0020 | 0.0000 | 0.0059 |
| 296 | 44 | -0.0098 | -0.0195 | 0.0000 |
| 296 | equal_seed_mean | 0.1426 | 0.1276 | 0.1563 |

All target curves/control differences are in paired_cap_comparisons.csv. Per-ranking caps, selected n, actual sizes, shortfalls and mask hashes are in matched_results.csv. Every count1..640 has an exact-attainability flag for each original and ensemble; skipped counts are explicitly listed in skipped_exact_targets_1_640.json. Pair-by-pair exact/capped labels and size differences are in exact_comparisons_all_pairs.csv; primary pairs in exact_primary_comparisons.csv.

Exact-primary subsets of the same measured cap queries:

| target | exact_seeds | n_fixed_seeds | mean | lower | upper | scope |
| --- | --- | --- | --- | --- | --- | --- |
| 60 | [] | 0 | nan | nan | nan | no exact paired comparison attained by both original rankings |
| 80 | [] | 0 | nan | nan | nan | no exact paired comparison attained by both original rankings |
| 100 | [42, 43] | 2 | 0.2910 | 0.2588 | 0.3223 | descriptive exact subset of already evaluated capped circuits; no separate exact forwards |
| 116 | [42] | 1 | 0.4082 | 0.3672 | 0.4473 | descriptive exact subset of already evaluated capped circuits; no separate exact forwards |
| 140 | [43] | 1 | 0.1133 | 0.0762 | 0.1523 | descriptive exact subset of already evaluated capped circuits; no separate exact forwards |
| 180 | [] | 0 | nan | nan | nan | no exact paired comparison attained by both original rankings |
| 240 | [43] | 1 | 0.0020 | -0.0078 | 0.0117 | descriptive exact subset of already evaluated capped circuits; no separate exact forwards |
| 296 | [43, 44] | 2 | -0.0039 | -0.0098 | 0.0010 | descriptive exact subset of already evaluated capped circuits; no separate exact forwards |
| 400 | [42, 43, 44] | 3 | 0.0189 | 0.0104 | 0.0280 | descriptive exact subset of already evaluated capped circuits; no separate exact forwards |
| 640 | [42, 43, 44] | 3 | 0.0000 | 0.0000 | 0.0000 | descriptive exact subset of already evaluated capped circuits; no separate exact forwards |

Intervals above apply only to the indicated exact fixed-seed subset, which can contain fewer than three seeds. No broader exact-size superiority is inferred from a capped result.

## Interpolation at T1 and T2

For each ranking, lower/upper attainable circuits bracket the target. Linear interpolation of measured accuracy uses weight (T-lower)/(upper-lower); equal counts use the exact circuit. These are **interpolated estimates, not directly evaluated exact circuits**. Gaps exceeding max(10edges,10%ofT) are flagged without hiding them. Estimates, brackets/gaps and flags are in interpolated_estimates.csv.

| target | seed | mean | lower | upper | interpolated |
| --- | --- | --- | --- | --- | --- |
| 116 | 42 | 0.4082 | 0.3672 | 0.4473 | True |
| 116 | 43 | 0.0850 | 0.0508 | 0.1201 | True |
| 116 | 44 | 0.5555 | 0.5145 | 0.5973 | True |
| 116 | equal_seed_mean | 0.3495 | 0.3230 | 0.3757 | True |
| 296 | 42 | 0.4355 | 0.3945 | 0.4766 | True |
| 296 | 43 | 0.0020 | 0.0000 | 0.0059 | True |
| 296 | 44 | -0.0098 | -0.0195 | 0.0000 | True |
| 296 | equal_seed_mean | 0.1426 | 0.1276 | 0.1563 | True |

## Paired prompt uncertainty and advantage retention

The sole primary hypothesis is the capped T1 comparison. It requires positive equally seed-weighted accuracy gain, positive changes in>=2/3 seeds, paired template-stratified95% bootstrap lower>0, and valid masks/evaluator. **5000** shared prompt resamples, seed**5643**, are used across methods, seeds, targets and the nominal/cap contrast. Training seeds are fixed, never treated as independently sampled prompt rows or resampled. T2, control and interpolation intervals are descriptive secondary/unadjusted. Prompt uncertainty conditions on the fixed official-template mix, not unseen-template generalization or training-seed significance.

R is the cappedT1 accuracy gain divided by the fresh nominal0.5% gain. Shared resamples preserve numerator/denominator covariance. A denominator with95%CI containing0 or absolute point<1/N is marked unstable/undefined; invalid zero-denominator draws are counted. R is an **advantage-retention ratio, not a causal fraction explained by pruning**. Classification follows the frozen order: non-replication if nominal lower<=0; ranking-quality effect if primary passes and stableR>=.5; mostly connectivity if nominal positive, primary fails, and capCIincludes0 or stableR<.25; mixed otherwise. None proves a causal mechanism.

## Structural controls at T1/T2

| target | comparator | seed | mean | lower | upper | corrected_prompts | broken_prompts | positive_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 116 | baseline | equal_seed_mean | 0.3451 | 0.3184 | 0.3717 | nan | nan | 3.0000 |
| 116 | gamma_zero | equal_seed_mean | -0.0117 | -0.0208 | -0.0026 | nan | nan | 0.0000 |
| 116 | shuffled | equal_seed_mean | 0.2064 | 0.1849 | 0.2285 | nan | nan | 2.0000 |
| 116 | edge_l2 | equal_seed_mean | 0.3451 | 0.3184 | 0.3717 | nan | nan | 3.0000 |
| 296 | baseline | equal_seed_mean | 0.1426 | 0.1276 | 0.1563 | nan | nan | 2.0000 |
| 296 | gamma_zero | equal_seed_mean | -0.0013 | -0.0059 | 0.0033 | nan | nan | 0.0000 |
| 296 | shuffled | equal_seed_mean | 0.0384 | 0.0221 | 0.0540 | nan | nan | 3.0000 |
| 296 | edge_l2 | equal_seed_mean | 0.1484 | 0.1335 | 0.1634 | nan | nan | 2.0000 |

Gamma-zero, shuffled and edge-L2 controls remain included regardless of performance. A gamma-zero tie favors a parameterization/optimization interpretation over a residual-penalty benefit. Real-versus-shuffled contrasts probe grouping, with exploratory secondary prompt intervals; smoother group scores alone do not establish causal fidelity.

## Closure ranking: nominal-grid effects

| original | CPR_delta | CMD_delta |
| --- | --- | --- |
| baseline | 0.0072 | 0.0028 |
| edge_l2 | 0.0049 | 0.0024 |
| gamma_zero | 0.0063 | 0.0029 |
| hierarchy | 0.0059 | 0.0036 |
| shuffled | 0.0045 | 0.0028 |

Every closure ranking was evaluated on the full official CPR/CMD grid. closure_nominal_effects.csv reports original and closure actual counts beside **every** accuracy difference, plus CPR/CMD changes and exact mask identity. These are **nominal-grid leaderboard-counting effects**, not evidence of improved equal-size ranking quality. No matched-size closure evaluations were run: at counts attained by the original, the constructed closure boundary has the same originalP(n), so that evaluation is redundant. closure_rank_stability.csv provides parent/transformed signed-order Spearman; closure_cross_seed_stability.csv reports before/after cross-seed rank correlations and official pruned-edge Jaccard. Mandatory group/tail nesting checks document scope. An improvement does not authorize another campaign.

## Equal-weight rank ensembles

| method | target | actual_edges | ensemble_accuracy | seed42_accuracy | seed43_accuracy | seed44_accuracy | component_min | component_mean | ensemble_minus_component_min | baseline_T2_90pct_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 116 | 116 | 0.1367 | 0.0469 | 0.3574 | 0.1895 | 0.0469 | 0.1979 | 0.0898 | False |
| baseline | 296 | 296 | 0.9395 | 0.2637 | 0.9980 | 0.9941 | 0.2637 | 0.7520 | 0.6758 | True |
| hierarchy | 116 | 116 | 0.4668 | 0.4551 | 0.4277 | 0.7461 | 0.4277 | 0.5430 | 0.0391 | False |
| hierarchy | 296 | 296 | 0.9844 | 0.6992 | 1.0000 | 0.9844 | 0.6992 | 0.8945 | 0.2852 | False |
| gamma_zero | 116 | 116 | 0.4590 | 0.4648 | 0.4531 | 0.7461 | 0.4531 | 0.5547 | 0.0059 | False |
| gamma_zero | 296 | 296 | 0.9863 | 0.7031 | 1.0000 | 0.9844 | 0.7031 | 0.8958 | 0.2832 | False |

Original score magnitudes are converted to signed ordinal ranks with canonical-index ties, averaged equally across seeds, then ties broken by canonical index. All edges appear once; no validation-dependent weights. All3ensembles were evaluated on the full official grid and registered cap targets. The baseline90% diagnostic atT2 is explicitly recorded above. Compare ensemble performance against the component minimum and each seed, not only a favorable average. ensemble_training_cost.csv charges the **three preexisting1000-step runs** separately from new inference/construction. Ensembling is not a cheaper discovery method.

| method | separate_training_runs | optimizer_updates_total | existing_training_seconds | existing_adaptation_seconds | scope |
| --- | --- | --- | --- | --- | --- |
| baseline | 3 | 3000 | 826.6077 | 857.8911 | reused three1000-step runs; not newGate5E compute or a cheaper discovery claim |
| hierarchy | 3 | 3000 | 833.8396 | 868.2720 | reused three1000-step runs; not newGate5E compute or a cheaper discovery claim |
| gamma_zero | 3 | 3000 | 826.6268 | 859.4405 | reused three1000-step runs; not newGate5E compute or a cheaper discovery claim |

Separate CPU construction replay (values match frozen exports exactly; original construction is already charged in PhaseA):

| method | CPU_construction_seconds_replay | reproduced_export | model_forwards | scope |
| --- | --- | --- | --- | --- |
| baseline | 0.0096 | True | 0 | timed deterministic replay; original construction included in PhaseA total; file loading excluded |
| hierarchy | 0.0093 | True | 0 | timed deterministic replay; original construction included in PhaseA total; file loading excluded |
| gamma_zero | 0.0097 | True | 0 | timed deterministic replay; original construction included in PhaseA total; file loading excluded |

## Actual compute and correctness

Inference-process wall time **58.110minutes**, of which nominal/matched evaluation stages total **57.614minutes**. Root model forwards **19,956**, individual-example forwards **391,332**, optimizer updates/backward calls **0/0**. Unique exact full-cohort masks **381**, output-cache hits **166**, reference-scalar hits **64**. Cache hits execute no model forward; no activation caches are used. CPU topology/construction time is separate and charged above. Preparation, bootstrap/reporting and audits are additional CPU work, not GPU training.

CUDA-synchronized monotonic timers measure actual stage wall time; allocation peaks are **2452.53MiB allocated**, **6056.00MiB reserved**, sampled CPU RSS **2968.49MiB**. These are allocator/RSS values, not total physical VRAM occupancy or CUDA kernel-utilization time. Per-ranking nominal and matched-stage costs are saved. First-ranking projection was checked against7200seconds before any remaining full-cohort evaluation; its measured inputs are in first_ranking_projection.json. Exact per-prompt output reuse explains differences between requested and unique circuits.

Existing official batching/repeat/reference checks and new cached-versus-uncached MIB comparisons passed, with zero repeat forwards on cache hits. Full circuit matches clean reference, all evaluated masks match frozen official masks, no gradients are stored, and complete model weight hashes match before/after. Historical artifacts and Gate5C/5D decisions remain preserved.

## Limitations and recommendation

This is a capped-size experiment when actual counts differ. Exact subsets and interpolated estimates must retain their labels. Size matching does not establish a causal explanation of the nominal advantage; circuits can differ in composition and task behavior. Closure only audits first-appearance groups built from n<=1624, and nominal closure gains compare different actual sizes. Ensembles consume3trained runs. The same three trained/subset seeds and known IOI template families remain; prompt intervals do not establish training-seed significance or unseen-template generalization. Historical full-validation exposure remains, and no public/private test or leaderboard submission was used.

Use the frozen classification and exact-size evidence above to judge whether compact-ranking quality remains plausible. Treat closure as a separate deterministic nominal-grid direction and ensemble robustness as a three-run tradeoff. Any future direction is a recommendation only. Stop after Gate5E: no new training, score optimization, differentiable masks, further gates or unregistered ranking refinement.


![accuracy_actual_size](figures/accuracy_actual_size.png)

![paired_target_difference](figures/paired_target_difference.png)

![nominal_vs_capped](figures/nominal_vs_capped.png)

![retention_curves](figures/retention_curves.png)

![ensemble_vs_seeds](figures/ensemble_vs_seeds.png)

![closure_cpr_cmd](figures/closure_cpr_cmd.png)


## Final preservation audit

```json
{
  "passed": true,
  "protocol_sha256": "bb6421a5e5481a172559d0634ab9e71a90b942192f046a824b818bb4ecc90b39",
  "historical_files_unchanged": 7834,
  "full_grid_rankings": 33,
  "capped_upper_queries": 216,
  "cohort_prompts": 512,
  "clean_donor_crossrole_overlap": 0,
  "no_matched_closure_queries": true,
  "original_checkpoint_hashes_verified": true,
  "model_weights_unchanged": true,
  "actual_forwards_verified": true,
  "optimizer_updates": 0,
  "model_backwards": 0,
  "Gate5D_decision_unchanged": true,
  "additional_model_forwards": 0,
  "seconds": 107.861834222
}
```
