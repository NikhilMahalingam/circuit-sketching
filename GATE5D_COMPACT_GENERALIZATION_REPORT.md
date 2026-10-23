# Gate 5D: Hierarchical MAttr compact-circuit generalization

## Interpretation of the completed experiment

The registered outcome is **mixed/inconclusive**, not a compact-generalization success under all frozen criteria. At **0.5% nominal retention**, mean positive-margin accuracy increased from **16.67% to 53.58%** (+36.91 percentage points; simultaneous95% interval **+33.79 to +40.01 points**), with positive changes in all three seeds. At **1%**, it increased from **70.96% to 88.74%** (+17.77 points; interval **+16.05 to +19.40**), but the changes were **+53.91, 0.00 and -0.59 points** for seeds42/43/44. Thus only one seed improved at1%, failing the prespecified at-least-two-seeds condition. The pooled bootstrap result does not override that condition. Baseline seeds43/44 were already at99.80%/100% at1%; this ceiling is context for the negative consistency decision, not a reason to relax it.

The gamma-zero hierarchy reached **53.52%/89.13%** at0.5%/1%, essentially matching or slightly exceeding the regularized hierarchy. This supports parameterization/optimization geometry as the main explanation rather than a demonstrated benefit from the residual penalty. The real hierarchy exceeded shuffled groups by **15.69 and4.04 points**, respectively, passing the prespecified pooled control comparisons with two-seed consistency. Generic edge regularization stayed near ordinary MAttr (**16.47%/71.03%**) and does not explain the large pooled compact gains; however, the real-versus-edge-L2 comparison also misses two-positive-seed consistency at1%. The complete structural-advantage criterion does not pass because all required control safeguards must pass.

Mean accuracy log-AUC rose **0.710848 to0.782791** (+0.071943; secondary95% paired prompt interval +0.067091 to+0.076649). Mean CPR fell **9.904998 to9.869015**, and mean CMD fell **8.921454 to8.879688**. There is no CPR or leaderboard improvement claim. Full-vocabulary top1, captured separately from the same logits, rose from12.17% to34.83% at0.5% and69.47% to78.26% at1%; these are not the positive-margin endpoints.

All nine exact official template strata had positive equally seed-weighted mean primary differences at both cutoffs. This is a descriptive result within existing template families, conditional on the seed mixture; it does not cure the1% seed inconsistency or establish unseen-template generalization. `official_template_labels.csv` maps each hash to its exact generic benchmark template, with no individual prompt text. Independent exclusion and optimizer-step audits supplement the frozen-protocol checks without changing the cohort, metrics or decisions.

The evidence supports a **narrow research motivation for very compact circuits, particularly the unregularized hierarchical parameterization**, rather than a validated regularization-specific improvement. Keep ordinary MAttr as the CPR reference and preserve the failed Gate5C/5D decisions. No further training, algorithm, ranking refinement, or leaderboard submission was performed.

## Executed decision

**Mixed or inconclusive**. All15 saved rankings were evaluated at exactly1000 training updates; Gate5D performs **zero optimizer updates and zero model backward calls**. This gate evaluates a new compact-accuracy hypothesis; it does not override Gate5C's failed CPR gate or authorize5000-step continuation. Ordinary MAttr remains the CPR reference unless independently demonstrated otherwise.

| fraction | delta | lower | upper | corrected | broken |
| --- | --- | --- | --- | --- | --- |
| 0.0050 | 0.3691 | 0.3379 | 0.4001 | 194.3333 | 5.3333 |
| 0.0100 | 0.1777 | 0.1605 | 0.1940 | 92.0000 | 1.0000 |

Differences are accuracy fractions, not percentage points. Pooled values equally weight the three FIXED trained seeds. Corrected/broken pooled counts are means of per-seed counts, not independent3N observations. The confidence intervals are paired, exact-template-stratified prompt bootstrap with5000 shared resamples, seed5642, Bonferroni over the two primary endpoints (97.5% marginal percentile intervals, simultaneous coverage at least95%). These intervals describe prompt sampling conditional on the fixed template mix, not training-seed significance or unseen-template performance.

## Protocol, historical exposure and cohort audit

Protocol SHA-256: `ec2bf0f0a2a4fd73f4699b3040e939ae64b629d46e715cf9e531dd92c4e1ad43`. Frozen before any model forward on the new cohort. All methods/seeds,15 checkpoint/effective-score hashes, original32491-edge mapping, exclusions, exact row IDs and UTF8 hashes, circuit grid, bootstrap and thresholds were recorded. Source pins, checkpoint metadata, group mappings, model configuration and effective exports were audited. No methods were selected or discarded using fresh-cohort performance.

Selected **512** official MIB IOI validation prompts from **9808** eligible unique strings, stratified proportionally across **9** exact official `template` values using largest-remainder quotas and seed5542. Exclusions use both filtered official row IDs and exact prompt SHA256, including prior small Gate2/3/4/5A development evaluations, Gate5B held-out confirmation, Gate5C development, and all located discovery-training manifests. Clean prompt duplicates are removed even if row IDs differ. Corrupted-prompt hashes from historical manifests are also excluded conservatively. Cohort audit records every input manifest/hash. Official names/labels are verified single tokens. No public/private test data or generated substitute is used.

**Historical Gate1 evaluated the full official10000-example validation split. These prompts are fresh relative to the smaller development/confirmation cohorts, not untouched by all previous research.** Template labels come directly from the dataset's `template` field; none are invented from text. Some templates can remain correlated, and stratified prompt intervals condition on the observed family mixture. This is not a held-out-family design.

## Frozen checkpoint provenance

All methods use Gate5C1000-step scores (baseline exact-bypass scores verified bitwise against Gate2). Effective scores are evaluated for hierarchical checkpoints; raw packed h/residual parameters are never interpreted as edge scores. Per-checkpoint hashes and config/cohort provenance are in the protocol and checkpoint_provenance.csv. No model training, score adaptation, ranking refinement or mask changes occurred.

## Official per-seed and descriptive means

| method | seed | CPR | CMD | accuracy_log_AUC | accuracy_0.005 | accuracy_0.01 | accuracy_0.02 | accuracy_0.05 | top1_0.005 | top1_0.01 | top1_0.02 | top1_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 42 | 9.7178 | 8.7438 | 0.6359 | 0.0293 | 0.1309 | 0.9922 | 1.0000 | 0.0039 | 0.0859 | 0.9902 | 1.0000 |
| baseline | 43 | 10.0441 | 9.0553 | 0.7575 | 0.3164 | 0.9980 | 1.0000 | 1.0000 | 0.2695 | 0.9980 | 1.0000 | 1.0000 |
| baseline | 44 | 9.9531 | 8.9652 | 0.7391 | 0.1543 | 1.0000 | 1.0000 | 1.0000 | 0.0918 | 1.0000 | 1.0000 | 1.0000 |
| hierarchy | 42 | 9.8524 | 8.8679 | 0.7423 | 0.3828 | 0.6699 | 1.0000 | 1.0000 | 0.1953 | 0.4082 | 1.0000 | 1.0000 |
| hierarchy | 43 | 9.8978 | 8.9072 | 0.7818 | 0.4766 | 0.9980 | 1.0000 | 1.0000 | 0.3438 | 0.9980 | 1.0000 | 1.0000 |
| hierarchy | 44 | 9.8568 | 8.8640 | 0.8242 | 0.7480 | 0.9941 | 1.0000 | 1.0000 | 0.5059 | 0.9414 | 1.0000 | 1.0000 |
| gamma_zero | 42 | 9.8023 | 8.8179 | 0.7353 | 0.3906 | 0.6816 | 1.0000 | 1.0000 | 0.2051 | 0.4375 | 1.0000 | 1.0000 |
| gamma_zero | 43 | 9.9333 | 8.9428 | 0.7796 | 0.4668 | 0.9980 | 1.0000 | 1.0000 | 0.3359 | 0.9961 | 1.0000 | 1.0000 |
| gamma_zero | 44 | 9.8541 | 8.8612 | 0.8242 | 0.7480 | 0.9941 | 1.0000 | 1.0000 | 0.5059 | 0.9414 | 0.9980 | 1.0000 |
| shuffled | 42 | 9.6641 | 8.6796 | 0.7130 | 0.1816 | 0.6875 | 0.9980 | 1.0000 | 0.0762 | 0.5273 | 0.9961 | 1.0000 |
| shuffled | 43 | 9.6480 | 8.6552 | 0.8085 | 0.7559 | 0.9590 | 1.0000 | 1.0000 | 0.6562 | 0.8691 | 1.0000 | 1.0000 |
| shuffled | 44 | 9.5383 | 8.5497 | 0.7364 | 0.1992 | 0.8945 | 1.0000 | 1.0000 | 0.0820 | 0.8086 | 0.9902 | 1.0000 |
| edge_l2 | 42 | 9.6986 | 8.7246 | 0.6361 | 0.0293 | 0.1328 | 0.9922 | 1.0000 | 0.0039 | 0.0879 | 0.9902 | 1.0000 |
| edge_l2 | 43 | 10.0266 | 9.0379 | 0.7568 | 0.3105 | 0.9980 | 1.0000 | 1.0000 | 0.2637 | 0.9980 | 1.0000 | 1.0000 |
| edge_l2 | 44 | 9.9380 | 8.9501 | 0.7391 | 0.1543 | 1.0000 | 1.0000 | 1.0000 | 0.0918 | 1.0000 | 1.0000 | 1.0000 |

| method | CPR | CMD | accuracy_log_AUC | accuracy_0.005 | accuracy_0.01 | accuracy_0.02 | accuracy_0.05 | top1_0.005 | top1_0.01 | top1_0.02 | top1_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 9.9050 | 8.9215 | 0.7108 | 0.1667 | 0.7096 | 0.9974 | 1.0000 | 0.1217 | 0.6947 | 0.9967 | 1.0000 |
| edge_l2 | 9.8877 | 8.9042 | 0.7107 | 0.1647 | 0.7103 | 0.9974 | 1.0000 | 0.1198 | 0.6953 | 0.9967 | 1.0000 |
| gamma_zero | 9.8632 | 8.8739 | 0.7797 | 0.5352 | 0.8913 | 1.0000 | 1.0000 | 0.3490 | 0.7917 | 0.9993 | 1.0000 |
| hierarchy | 9.8690 | 8.8797 | 0.7828 | 0.5358 | 0.8874 | 1.0000 | 1.0000 | 0.3483 | 0.7826 | 1.0000 | 1.0000 |
| shuffled | 9.6168 | 8.6282 | 0.7526 | 0.3789 | 0.8470 | 0.9993 | 1.0000 | 0.2715 | 0.7350 | 0.9954 | 1.0000 |

The original official signed `apply_topn(..., absolute=False, prune=True)` selects nominal circuit fractions. Rankings are nested before pruning; actual retained-edge counts can be smaller and are not forced to nominal counts. `faithfulness_curves.csv` and `compact_accuracy_and_edge_counts.csv` retain every official grid point and mask hash. Official CPR/CMD use ratios of cohort means and the unchanged evaluator's integrals; no homemade replacement is used. Accuracy is the proportion with correct-minus-incorrect margin>0. Full-vocabulary top1 is a separate official `acc` diagnostic captured from the same logits, with no extra model pass. CPR rewards amplification and is not interchangeable with model fidelity or compact accuracy; CMD is reported separately.

## Paired uncertainty, corrections and controls

| seed | fraction | delta | lower | upper | corrected | broken |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.0050 | 0.3535 | 0.3086 | 0.3984 | 181.0000 | 0.0000 |
| 43 | 0.0050 | 0.1602 | 0.1181 | 0.2031 | 98.0000 | 16.0000 |
| 44 | 0.0050 | 0.5938 | 0.5449 | 0.6426 | 304.0000 | 0.0000 |
| equal_seed_mean | 0.0050 | 0.3691 | 0.3379 | 0.4001 | 194.3333 | 5.3333 |
| 42 | 0.0100 | 0.5391 | 0.4883 | 0.5879 | 276.0000 | 0.0000 |
| 43 | 0.0100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 44 | 0.0100 | -0.0059 | -0.0137 | 0.0000 | 0.0000 | 3.0000 |
| equal_seed_mean | 0.0100 | 0.1777 | 0.1605 | 0.1940 | 92.0000 | 1.0000 |

| comparator | fraction | delta | lower | upper |
| --- | --- | --- | --- | --- |
| baseline | 0.0050 | 0.3691 | 0.3379 | 0.4001 |
| baseline | 0.0100 | 0.1777 | 0.1605 | 0.1940 |
| baseline | 0.0200 | 0.0026 | 0.0000 | 0.0059 |
| baseline | 0.0500 | 0.0000 | 0.0000 | 0.0000 |
| gamma_zero | 0.0050 | 0.0007 | -0.0091 | 0.0104 |
| gamma_zero | 0.0100 | -0.0039 | -0.0104 | 0.0026 |
| gamma_zero | 0.0200 | 0.0000 | 0.0000 | 0.0000 |
| gamma_zero | 0.0500 | 0.0000 | 0.0000 | 0.0000 |
| shuffled | 0.0050 | 0.1569 | 0.1255 | 0.1888 |
| shuffled | 0.0100 | 0.0404 | 0.0162 | 0.0645 |
| shuffled | 0.0200 | 0.0007 | 0.0000 | 0.0026 |
| shuffled | 0.0500 | 0.0000 | 0.0000 | 0.0000 |
| edge_l2 | 0.0050 | 0.3711 | 0.3345 | 0.4082 |
| edge_l2 | 0.0100 | 0.1771 | 0.1569 | 0.1967 |
| edge_l2 | 0.0200 | 0.0026 | 0.0000 | 0.0065 |
| edge_l2 | 0.0500 | 0.0000 | 0.0000 | 0.0000 |

| comparator | seed | delta | lower | upper |
| --- | --- | --- | --- | --- |
| baseline | 42 | 0.1064 | 0.0986 | 0.1144 |
| baseline | 43 | 0.0243 | 0.0198 | 0.0289 |
| baseline | 44 | 0.0851 | 0.0796 | 0.0905 |
| baseline | equal_seed_mean | 0.0719 | 0.0671 | 0.0766 |
| gamma_zero | 42 | 0.0070 | 0.0037 | 0.0105 |
| gamma_zero | 43 | 0.0023 | -0.0002 | 0.0050 |
| gamma_zero | 44 | 0.0000 | 0.0000 | 0.0000 |
| gamma_zero | equal_seed_mean | 0.0031 | 0.0016 | 0.0046 |
| shuffled | 42 | 0.0294 | 0.0234 | 0.0352 |
| shuffled | 43 | -0.0266 | -0.0322 | -0.0211 |
| shuffled | 44 | 0.0878 | 0.0828 | 0.0927 |
| shuffled | equal_seed_mean | 0.0302 | 0.0267 | 0.0337 |
| edge_l2 | 42 | 0.1062 | 0.0984 | 0.1142 |
| edge_l2 | 43 | 0.0250 | 0.0205 | 0.0296 |
| edge_l2 | 44 | 0.0851 | 0.0796 | 0.0905 |
| edge_l2 | equal_seed_mean | 0.0721 | 0.0672 | 0.0769 |

Primary success requires positive mean differences and positive corrected pooled lower bounds at both0.5%/1%, at least2/3 positive seeds at each, mean accuracy-log-AUC change>=-.02, no mean loss>.02 at2%/5%, and valid semantics. Per-seed primary intervals correct for two endpoints within each seed, not all six seed-specific tests. Control intervals use a distinct Bonferroni family of6 pooled contrasts (3controls x2 primary cutoffs). Accuracy-log-AUC95% intervals are secondary/unadjusted; bootstrap replicates use the linear functional of per-prompt correctness with weights verified against the official point estimate. No bootstrap treats seeds as independently sampled prompts.

Frozen decision details:

```json
{
  "primary": {
    "passed": false,
    "checks": {
      "valid_conventions": true,
      "positive_mean_0.005": true,
      "corrected_lower_0.005": true,
      "positive_2seeds_0.005": true,
      "positive_mean_0.01": true,
      "corrected_lower_0.01": true,
      "positive_2seeds_0.01": false,
      "accuracy_auc": true,
      "compact_2_5_safeguard": true
    }
  },
  "controls": {
    "gamma_zero": {
      "passed": false,
      "checks": {
        "valid_conventions": true,
        "positive_mean_0.005": true,
        "corrected_lower_0.005": false,
        "positive_2seeds_0.005": false,
        "positive_mean_0.01": false,
        "corrected_lower_0.01": false,
        "positive_2seeds_0.01": false
      }
    },
    "shuffled": {
      "passed": true,
      "checks": {
        "valid_conventions": true,
        "positive_mean_0.005": true,
        "corrected_lower_0.005": true,
        "positive_2seeds_0.005": true,
        "positive_mean_0.01": true,
        "corrected_lower_0.01": true,
        "positive_2seeds_0.01": true
      }
    },
    "edge_l2": {
      "passed": false,
      "checks": {
        "valid_conventions": true,
        "positive_mean_0.005": true,
        "corrected_lower_0.005": true,
        "positive_2seeds_0.005": true,
        "positive_mean_0.01": true,
        "corrected_lower_0.01": true,
        "positive_2seeds_0.01": false
      }
    }
  },
  "structural_advantage": false,
  "classification": "Mixed or inconclusive",
  "training_seed_significance_claim": false,
  "CPR_improvement_is_separate": true,
  "further_training_authorized": false
}
```

If gamma-zero hierarchy shows the same gain, evidence favors parameterization/optimization geometry rather than the residual penalty. Actual component grouping must beat shuffled grouping, gamma-zero hierarchy AND generic edge regularization at both primary cutoffs to support the registered structural claim. Group-score stability alone is not evidence of causal fidelity. Controls remain included regardless of outcome; no further sweep or training is authorized.

## Exact-template descriptive analysis

Full per-template/seed/method accuracy and top1 are in template_family_accuracy.csv; paired template contrasts are in template_family_paired_differences.csv. Template counts and IDs are frozen. Small per-template samples and repeated lexical patterns limit precision; the template plot is descriptive, with no post-hoc family selection or unseen-family significance claim.

## Actual retained edges

| method | seed | fraction | requested_edges | actual_edges | accuracy | top1_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 42 | 0.0050 | 162 | 92 | 0.0293 | 0.0039 |
| baseline | 42 | 0.0100 | 324 | 273 | 0.1309 | 0.0859 |
| baseline | 42 | 0.0200 | 649 | 638 | 0.9922 | 0.9902 |
| baseline | 42 | 0.0500 | 1624 | 1601 | 1.0000 | 1.0000 |
| baseline | 43 | 0.0050 | 162 | 118 | 0.3164 | 0.2695 |
| baseline | 43 | 0.0100 | 324 | 309 | 0.9980 | 0.9980 |
| baseline | 43 | 0.0200 | 649 | 638 | 1.0000 | 1.0000 |
| baseline | 43 | 0.0500 | 1624 | 1612 | 1.0000 | 1.0000 |
| baseline | 44 | 0.0050 | 162 | 116 | 0.1543 | 0.0918 |
| baseline | 44 | 0.0100 | 324 | 296 | 1.0000 | 1.0000 |
| baseline | 44 | 0.0200 | 649 | 636 | 1.0000 | 1.0000 |
| baseline | 44 | 0.0500 | 1624 | 1612 | 1.0000 | 1.0000 |
| hierarchy | 42 | 0.0050 | 162 | 114 | 0.3828 | 0.1953 |
| hierarchy | 42 | 0.0100 | 324 | 279 | 0.6699 | 0.4082 |
| hierarchy | 42 | 0.0200 | 649 | 631 | 1.0000 | 1.0000 |
| hierarchy | 42 | 0.0500 | 1624 | 1613 | 1.0000 | 1.0000 |
| hierarchy | 43 | 0.0050 | 162 | 129 | 0.4766 | 0.3438 |
| hierarchy | 43 | 0.0100 | 324 | 298 | 0.9980 | 0.9980 |
| hierarchy | 43 | 0.0200 | 649 | 635 | 1.0000 | 1.0000 |
| hierarchy | 43 | 0.0500 | 1624 | 1614 | 1.0000 | 1.0000 |
| hierarchy | 44 | 0.0050 | 162 | 118 | 0.7480 | 0.5059 |
| hierarchy | 44 | 0.0100 | 324 | 290 | 0.9941 | 0.9414 |
| hierarchy | 44 | 0.0200 | 649 | 639 | 1.0000 | 1.0000 |
| hierarchy | 44 | 0.0500 | 1624 | 1609 | 1.0000 | 1.0000 |
| gamma_zero | 42 | 0.0050 | 162 | 114 | 0.3906 | 0.2051 |
| gamma_zero | 42 | 0.0100 | 324 | 285 | 0.6816 | 0.4375 |
| gamma_zero | 42 | 0.0200 | 649 | 628 | 1.0000 | 1.0000 |
| gamma_zero | 42 | 0.0500 | 1624 | 1614 | 1.0000 | 1.0000 |
| gamma_zero | 43 | 0.0050 | 162 | 121 | 0.4668 | 0.3359 |
| gamma_zero | 43 | 0.0100 | 324 | 295 | 0.9980 | 0.9961 |
| gamma_zero | 43 | 0.0200 | 649 | 637 | 1.0000 | 1.0000 |
| gamma_zero | 43 | 0.0500 | 1624 | 1616 | 1.0000 | 1.0000 |
| gamma_zero | 44 | 0.0050 | 162 | 119 | 0.7480 | 0.5059 |
| gamma_zero | 44 | 0.0100 | 324 | 291 | 0.9941 | 0.9414 |
| gamma_zero | 44 | 0.0200 | 649 | 639 | 1.0000 | 0.9980 |
| gamma_zero | 44 | 0.0500 | 1624 | 1607 | 1.0000 | 1.0000 |
| shuffled | 42 | 0.0050 | 162 | 123 | 0.1816 | 0.0762 |
| shuffled | 42 | 0.0100 | 324 | 309 | 0.6875 | 0.5273 |
| shuffled | 42 | 0.0200 | 649 | 643 | 0.9980 | 0.9961 |
| shuffled | 42 | 0.0500 | 1624 | 1586 | 1.0000 | 1.0000 |
| shuffled | 43 | 0.0050 | 162 | 140 | 0.7559 | 0.6562 |
| shuffled | 43 | 0.0100 | 324 | 304 | 0.9590 | 0.8691 |
| shuffled | 43 | 0.0200 | 649 | 623 | 1.0000 | 1.0000 |
| shuffled | 43 | 0.0500 | 1624 | 1611 | 1.0000 | 1.0000 |
| shuffled | 44 | 0.0050 | 162 | 111 | 0.1992 | 0.0820 |
| shuffled | 44 | 0.0100 | 324 | 304 | 0.8945 | 0.8086 |
| shuffled | 44 | 0.0200 | 649 | 635 | 1.0000 | 0.9902 |
| shuffled | 44 | 0.0500 | 1624 | 1618 | 1.0000 | 1.0000 |
| edge_l2 | 42 | 0.0050 | 162 | 93 | 0.0293 | 0.0039 |
| edge_l2 | 42 | 0.0100 | 324 | 274 | 0.1328 | 0.0879 |
| edge_l2 | 42 | 0.0200 | 649 | 638 | 0.9922 | 0.9902 |
| edge_l2 | 42 | 0.0500 | 1624 | 1604 | 1.0000 | 1.0000 |
| edge_l2 | 43 | 0.0050 | 162 | 116 | 0.3105 | 0.2637 |
| edge_l2 | 43 | 0.0100 | 324 | 310 | 0.9980 | 0.9980 |
| edge_l2 | 43 | 0.0200 | 649 | 640 | 1.0000 | 1.0000 |
| edge_l2 | 43 | 0.0500 | 1624 | 1616 | 1.0000 | 1.0000 |
| edge_l2 | 44 | 0.0050 | 162 | 115 | 0.1543 | 0.0918 |
| edge_l2 | 44 | 0.0100 | 324 | 305 | 1.0000 | 1.0000 |
| edge_l2 | 44 | 0.0200 | 649 | 636 | 1.0000 | 1.0000 |
| edge_l2 | 44 | 0.0500 | 1624 | 1612 | 1.0000 | 1.0000 |

## Computation and inference correctness

Official full-grid evaluation wall time **22.330minutes**. Inference-process wall time including loading/correctness **22.749minutes**. Actual model forward calls **7,950**, individual example-forwards **155,720**. Unique evaluation prompts **512**, reused identically for each of15 rankings. CUDA timers synchronize and use monotonic perf_counter; root-forward counters count real calls, not cache hits. Official graph-independent reference scalar/result cache hits **28**; donor activation cache disabled, no mutable patched activations reused. References computed once for the same model/cohort/intervention are safe to share across rankings.

Peak allocated CUDA **2384.65MiB**, peak reserved **8796.00MiB**, sampled peak CPU RSS **2884.03MiB**. These are allocator/RSS measurements, not total physical VRAM occupancy or CUDA kernel utilization time. Per-method timing and forwards are in per_method_compute.csv; the first method bears one-time reference computation. All methods use batch20; no silent resource adaptation. Model parameters have no stored gradients and complete before/after weight hashes match. Correctness checks compare repeated and singleton batching, full-circuit versus clean reference, nested/pruned masks, and official reference-cache equivalence before the full evaluation.

| method_seed | seconds | forward_batches | example_forwards | peak_allocated_mib | peak_reserved_mib | sampled_peak_cpu_rss_mib |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_s42 | 103.3860 | 624 | 12288 | 2384.6470 | 8796.0000 | 1698.7031 |
| baseline_s43 | 87.0790 | 520 | 10240 | 2384.6470 | 8796.0000 | 1696.8477 |
| baseline_s44 | 88.0017 | 520 | 10240 | 2384.6470 | 8796.0000 | 1696.9375 |
| hierarchy_s42 | 87.6462 | 520 | 10240 | 2384.6470 | 8796.0000 | 1697.0703 |
| hierarchy_s43 | 88.0407 | 520 | 10240 | 2384.6470 | 8796.0000 | 1697.2109 |
| hierarchy_s44 | 88.2559 | 520 | 10240 | 2384.6470 | 8796.0000 | 1697.3359 |
| gamma_zero_s42 | 88.4307 | 520 | 10240 | 2384.6470 | 8796.0000 | 1694.7305 |
| gamma_zero_s43 | 87.4828 | 520 | 10240 | 2384.6470 | 8796.0000 | 1695.0117 |
| gamma_zero_s44 | 88.3476 | 520 | 10240 | 2384.6470 | 8796.0000 | 1695.0898 |
| shuffled_s42 | 88.2577 | 520 | 10240 | 2384.6470 | 8796.0000 | 1695.1289 |
| shuffled_s43 | 88.3856 | 520 | 10240 | 2384.6470 | 8796.0000 | 1695.1328 |
| shuffled_s44 | 88.7464 | 520 | 10240 | 2384.6470 | 8796.0000 | 1695.1367 |
| edge_l2_s42 | 87.5824 | 520 | 10240 | 2384.6470 | 8796.0000 | 1695.2188 |
| edge_l2_s43 | 87.2518 | 520 | 10240 | 2384.6470 | 8796.0000 | 1695.2500 |
| edge_l2_s44 | 92.9190 | 520 | 10240 | 2384.6470 | 8796.0000 | 1695.3320 |

## Recommendation and limitations

Use the frozen primary/control decisions above to interpret compact-accuracy generalization separately from structural attribution and CPR. A primary success warrants further independent-cohort research as a recommendation only; it does not demonstrate a leaderboard victory. A gamma-zero tie prevents attributing gains to regularization. Only three fixed paired training/subset seeds were measured, the official templates are reused, historical full-validation exposure remains, and no full-data/multi-model generalization was tested. Preserve the original MAttr and historical failed gates. Stop after Gate5D; no new training, algorithms, further gates or leaderboard submission.


![compact_accuracy](results/mattr_compact_validation/20261023_gate5d/figures/compact_accuracy.png)

![per_seed_accuracy_difference](results/mattr_compact_validation/20261023_gate5d/figures/per_seed_accuracy_difference.png)

![paired_bootstrap_intervals](results/mattr_compact_validation/20261023_gate5d/figures/paired_bootstrap_intervals.png)

![corrected_vs_broken](results/mattr_compact_validation/20261023_gate5d/figures/corrected_vs_broken.png)

![cpr_vs_cmd](results/mattr_compact_validation/20261023_gate5d/figures/cpr_vs_cmd.png)

![template_family_accuracy](results/mattr_compact_validation/20261023_gate5d/figures/template_family_accuracy.png)


## Final preservation audit

```json
{
  "passed": true,
  "historical_files_unchanged": 7724,
  "evaluated_rankings": 15,
  "protocol_sha256": "ec2bf0f0a2a4fd73f4699b3040e939ae64b629d46e715cf9e531dd92c4e1ad43",
  "cohort_size": 512,
  "actual_forwards_verified": true,
  "weight_updates": 0,
  "model_backwards": 0,
  "frozen_sources_and_checkpoints_verified": true,
  "additional_model_forwards": 0
}
```
