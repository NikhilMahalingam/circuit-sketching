# Sketch-ACDC final research report

Results directory: `results/mib_sketch_diagnostics/final/20261015T070721Z`.

**Recommendation: PAUSE the Hessian-based Sketch-ACDC approach. This restricted experiment does not satisfy both decision-quality and amortization requirements. Finish documentation before separately considering MAttr; no MAttr experiment was started.**

## 1. Research question and fixed setup

Does second-order Hessian information improve context-dependent MIB IOI edge-removal prediction enough to repay its construction? GPT-2 Small, 16 fixed EAP-selected MIB edges, official train/validation data, float32 model/derivatives, batch size 2. Detached 16-dimensional accumulation and pseudoinverse use float64. Discovery subsets are nested prefixes of the seed-42 train permutation. The same 128 validation prompts and the same queries are used for every size and estimator. Public test was not loaded. Validation was not used for query selection, thresholds, fitting, or seed-extension selection.

Frozen training-only query set: 35 edge/state queries, 3 partial states, 8435 query/threshold pairs, including **321 difficult pairs across 35 distinct queries**. Near-boundary pairs are within max(1e-6, 5.1% of exact train-16 effect magnitude); this selection remains fixed across data sizes. Thresholds extend beyond 0.01–0.30 and are derived from train effects only. Of the 321 fixed difficult pairs, 200 have train-16 effect magnitude below 5e-4. These numerical-scale decisions dominate the unstratified count; substantive-effect strata are descriptive checks, not replacement success criteria. The bootstrap resamples the same validation prompt indices jointly across queries (200 replicates); it conditions on the fitted estimator.

## 2. Previous experiments and what they established

- Toy factual baseline: feasibility of singleton/pairwise prediction; no MIB leaderboard implication.
- Standardized head-level IOI experiment: held-out mask / validation-prompt MAE 0.736744 (learned first order) versus 0.367987 (learned second order); mean absolute pair interaction 0.4830. These used `IOIScorer`, not the defective edge adapter, so this bug does not invalidate them. A globally fitted quadratic over head masks is distinct from a local Hessian approximation for an edge removal.
- Head-level circuit selection: mixed quality despite prediction improvement; it also used `IOIScorer` and is unaffected by this edge-specific defect.
- Historical edge Sketch-ACDC Stage A: loaded 32/64 train/validation and removed 10 of 16 candidates at threshold 0.05. Isolated endpoint checks were insufficient to establish dependent-mask correctness.
- Historical B/C pilot (16 train, 8 validation, one threshold): MAE 0.152787 first order, 0.224677 rank 2, 0.207928 rank 4. All methods reached the same six-edge state; verified arms checked all 16 decisions. H1 collection took 413.37 s and Stage C 173.64 s. These are preserved measurements of the frozen-sender function, not validated general results for official simultaneous MIB interventions. They require corrected reruns for such claims; historical timings include three forward passes per batch, whereas the corrected adapter uses two.
- Earlier diagnostic smoke: only two effective prompts per split and two queries. It discovered the dependency bug; it could not establish a rank trend or scientific advantage. No prior notebook or numeric artifact has been overwritten.

## 3. Intervention correctness and historical implications

The old adapter cached `corrupted_sender - original_clean_sender` before any intervention. MIB adds `corrupted_sender - current_intervened_sender` at each removed connection during the patched pass. When an upstream removal changes a downstream sender, the old update subtracts the wrong value. At z=1 a removed connection must carry the fixed corrupted donor, independent of the current sender. Detaching current senders would additionally break mixed derivatives. At the intact state, agreement along individual coordinate axes can preserve singleton effects and diagonal curvature while still giving incorrect mixed derivatives; randomized HVP sketches depend on those mixed derivatives. Thus an HVP finite-difference check against the historical function alone was insufficient to establish correct MIB semantics.

The corrected shared `EdgeExperiment` now captures current sender outputs without detaching their derivative. `DiagnosticEdgeExperiment` adds persistent caching without another patching implementation. `FrozenSenderEdgeExperiment` is retained solely as an explicitly named historical control; cache identities include adapter versions. **All 24 audited masks passed** the independent MIB evaluator, including random densities, partial states, dependent edges, and sampled prior pilot queries. Maximum corrected discrepancy: 1.90735e-06; maximum historical discrepancy: 3.3123.

Regression tests cover the complete shared adapter on a two-edge dependent DAG, all binary states, its analytic mixed derivative, receiver isolation, version-separated cache identities, full-Hessian recovery, nested rank-16 recovery, and streamed weighted derivatives. Real-model endpoint and HVP finite-difference checks are recorded in `correctness_passed.json`. Prior single-edge intact-state measurements remain valid by construction; dependent-state quantities need rechecking. Spot checks of prior queries do not certify every historical query.

## 4. Completed 16/32/64/128 results

Completed (seed, discovery size): [(42, 128), (42, 16), (42, 32), (42, 64), (43, 128), (44, 128)]. Incomplete primary sizes: [].

At anchor 0, full second order lowers held-out MAE at 2 of 4 completed primary sizes and improves difficult-decision accuracy at 3 of 4 sizes. These are descriptive signs, not independent significance tests. Held-out sketch MAE decreases monotonically with rank at 0 of 4 primary sizes. Recovering the reference Hessian more accurately need not improve finite-removal prediction. With 128 discovery prompts, first-order MAE is 0.0486489 versus 0.047534 for full second order; difficult-decision accuracy is 86.6% versus 88.5%.

| examples | first_MAE | full_MAE | first_difficult_accuracy | full_difficult_accuracy | full_MAE_gain | full_difficult_gain |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.06379 | 0.04675 | 0.5265 | 0.4984 | 0.01704 | -0.02804 |
| 32 | 0.07686 | 0.0846 | 0.7539 | 0.7757 | -0.007741 | 0.02181 |
| 64 | 0.07011 | 0.07565 | 0.8162 | 0.8349 | -0.005535 | 0.01869 |
| 128 | 0.04865 | 0.04753 | 0.866 | 0.8847 | 0.001115 | 0.01869 |

## 5. Estimator comparison, midpoint, and error sources

| discovery_examples | method | mae | spearman | difficult_accuracy | difficult_decision_count | false_prune_rate | false_retain_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | first_order | 0.06379 | 0.6703 | 0.5265 | 321 | 0.02513 | 0.09745 |
| 16 | full_second_order | 0.04675 | 0.6784 | 0.4984 | 321 | 0.0243 | 0.09887 |
| 16 | sketch_rank_12 | 0.04675 | 0.6647 | 0.5732 | 321 | 0.02359 | 0.09674 |
| 16 | sketch_rank_16 | 0.04675 | 0.6728 | 0.5607 | 321 | 0.02407 | 0.09674 |
| 16 | sketch_rank_4 | 0.04528 | 0.67 | 0.5483 | 321 | 0.02336 | 0.09793 |
| 16 | sketch_rank_8 | 0.04676 | 0.6661 | 0.5639 | 321 | 0.02359 | 0.0971 |
| 32 | first_order | 0.07686 | 0.7529 | 0.7539 | 321 | 0.02537 | 0.08275 |
| 32 | full_second_order | 0.0846 | 0.7527 | 0.7757 | 321 | 0.02205 | 0.08453 |
| 32 | sketch_rank_12 | 0.0846 | 0.7527 | 0.7632 | 321 | 0.0243 | 0.08417 |
| 32 | sketch_rank_16 | 0.0846 | 0.7527 | 0.7632 | 321 | 0.02371 | 0.08417 |
| 32 | sketch_rank_4 | 0.08424 | 0.7527 | 0.7632 | 321 | 0.02442 | 0.08405 |
| 32 | sketch_rank_8 | 0.0846 | 0.7527 | 0.7632 | 321 | 0.0243 | 0.08417 |
| 64 | first_order | 0.07011 | 0.6451 | 0.8162 | 321 | 0.05584 | 0.0594 |
| 64 | full_second_order | 0.07565 | 0.6619 | 0.8349 | 321 | 0.05335 | 0.06177 |
| 64 | sketch_rank_12 | 0.07565 | 0.649 | 0.8411 | 321 | 0.05465 | 0.06189 |
| 64 | sketch_rank_16 | 0.07565 | 0.6583 | 0.8411 | 321 | 0.05513 | 0.06165 |
| 64 | sketch_rank_4 | 0.07531 | 0.6448 | 0.838 | 321 | 0.05465 | 0.06046 |
| 64 | sketch_rank_8 | 0.07565 | 0.6448 | 0.8411 | 321 | 0.05489 | 0.06082 |
| 128 | first_order | 0.04865 | 0.5838 | 0.866 | 321 | 0.02336 | 0.04635 |
| 128 | full_second_order | 0.04753 | 0.5734 | 0.8847 | 321 | 0.01849 | 0.05086 |
| 128 | sketch_rank_12 | 0.04753 | 0.5737 | 0.8847 | 321 | 0.02063 | 0.05062 |
| 128 | sketch_rank_16 | 0.04753 | 0.5765 | 0.8847 | 321 | 0.01921 | 0.05074 |
| 128 | sketch_rank_4 | 0.04721 | 0.5835 | 0.8847 | 321 | 0.02229 | 0.04742 |
| 128 | sketch_rank_8 | 0.04753 | 0.5835 | 0.8847 | 321 | 0.02229 | 0.04742 |

All method rows share exact endpoints and data. Full second order is an analysis reference. Sketch HVPs are measured directly and are never computed by multiplying the reference Hessian. Nested orthonormal randomized directions are shared across ranks; rank-16 is a compression sanity reference, not scalable sketching.

MIB supplies binary circuit evaluation; the dynamic linear edge-strength interpolation is our explicitly defined continuous extension, not an invented official API. The shared midpoint sets all intact coordinates to 0.5; it can lie far from either binary endpoint. Its Taylor difference includes cross terms. At a candidate-only midpoint, the pure quadratic term cancels, so a midpoint benefit would not alone establish a benefit of second order. A large gain against the midpoint first-order estimator should also be compared with the stronger anchor-0 first-order control. Both anchors at n=128, seed 42:

| anchor_strength | method | mae | difficult_accuracy | wall_seconds |
| --- | --- | --- | --- | --- |
| 0 | first_order | 0.04865 | 0.866 | 28.42 |
| 0 | full_second_order | 0.04753 | 0.8847 | 319.6 |
| 0 | sketch_rank_12 | 0.04753 | 0.8847 | 243.4 |
| 0 | sketch_rank_16 | 0.04753 | 0.8847 | 316.8 |
| 0 | sketch_rank_4 | 0.04721 | 0.8847 | 99.68 |
| 0 | sketch_rank_8 | 0.04753 | 0.8847 | 171.2 |
| 0.5 | first_order | 0.08816 | 0.8598 | 26.22 |
| 0.5 | full_second_order | 0.04576 | 0.8972 | 319.8 |
| 0.5 | sketch_rank_12 | 0.04576 | 0.8972 | 247.8 |
| 0.5 | sketch_rank_16 | 0.04576 | 0.8972 | 320 |
| 0.5 | sketch_rank_4 | 0.04598 | 0.8972 | 101.3 |
| 0.5 | sketch_rank_8 | 0.04576 | 0.8972 | 174.7 |

Signed sketch error = full-Taylor error on train + sketch-minus-full compression error + exact train-minus-validation shift. Absolute terms do not add because signs can cancel. The full-Taylor column is the common reference error; estimator train MAE also shows the actual first-order error. At n=128, seed 42, anchor 0, exact train-to-validation shift MAE is 0.0355653, while full-Taylor truncation MAE on train is 0.0120934. Rank-4 and rank-16 compression MAE are 0.00175401 and 1.34649e-06. Increasing rank therefore addresses a smaller error source in this configuration; it cannot remove Taylor truncation or a mismatch between discovery and validation effects. Compression is not uniformly harmless: at the seed-44 midpoint, rank-4 compression MAE rises to 0.0468306, exceeding the full-Taylor train error 0.0239452. Its validation MAE is 0.1462; exact train-to-validation shift MAE is 0.115923. The seed/anchor tables retain this failure case.

| discovery_examples | method | estimator_mae_train | full_taylor_mae_train | compression_mae | exact_train_validation_shift_mae | total_validation_mae |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | first_order | 0.02172 | 0.01209 | NA | 0.03557 | 0.04865 |
| 128 | full_second_order | 0.01209 | 0.01209 | NA | 0.03557 | 0.04753 |
| 128 | sketch_rank_12 | 0.01209 | 0.01209 | 2.624e-06 | 0.03557 | 0.04753 |
| 128 | sketch_rank_16 | 0.01209 | 0.01209 | 1.346e-06 | 0.03557 | 0.04753 |
| 128 | sketch_rank_4 | 0.01286 | 0.01209 | 0.001754 | 0.03557 | 0.04721 |
| 128 | sketch_rank_8 | 0.0121 | 0.01209 | 1.426e-05 | 0.03557 | 0.04753 |
| 16 | first_order | 0.02517 | 0.01085 | NA | 0.04355 | 0.06379 |
| 16 | full_second_order | 0.01085 | 0.01085 | NA | 0.04355 | 0.04675 |
| 16 | sketch_rank_12 | 0.01085 | 0.01085 | 2.702e-06 | 0.04355 | 0.04675 |
| 16 | sketch_rank_16 | 0.01085 | 0.01085 | 1.505e-06 | 0.04355 | 0.04675 |
| 16 | sketch_rank_4 | 0.01207 | 0.01085 | 0.002776 | 0.04355 | 0.04528 |
| 16 | sketch_rank_8 | 0.01086 | 0.01085 | 2.841e-05 | 0.04355 | 0.04676 |
| 32 | first_order | 0.02415 | 0.01577 | NA | 0.06896 | 0.07686 |
| 32 | full_second_order | 0.01577 | 0.01577 | NA | 0.06896 | 0.0846 |
| 32 | sketch_rank_12 | 0.01577 | 0.01577 | 2.812e-06 | 0.06896 | 0.0846 |
| 32 | sketch_rank_16 | 0.01577 | 0.01577 | 1.194e-06 | 0.06896 | 0.0846 |
| 32 | sketch_rank_4 | 0.0164 | 0.01577 | 0.001636 | 0.06896 | 0.08424 |
| 32 | sketch_rank_8 | 0.01578 | 0.01577 | 1.71e-05 | 0.06896 | 0.0846 |
| 64 | first_order | 0.02138 | 0.01418 | NA | 0.06159 | 0.07011 |
| 64 | full_second_order | 0.01418 | 0.01418 | NA | 0.06159 | 0.07565 |
| 64 | sketch_rank_12 | 0.01419 | 0.01418 | 3.572e-06 | 0.06159 | 0.07565 |
| 64 | sketch_rank_16 | 0.01418 | 0.01418 | 1.192e-06 | 0.06159 | 0.07565 |
| 64 | sketch_rank_4 | 0.01481 | 0.01418 | 0.001493 | 0.06159 | 0.07531 |
| 64 | sketch_rank_8 | 0.01419 | 0.01418 | 8.37e-06 | 0.06159 | 0.07565 |

Decision accuracy and false-prune/retain rates use all query/threshold pairs as denominator; difficult accuracy uses the fixed train-defined subset. Difficult effect-scale strata are saved separately in `difficult_scale_strata.csv`. Near-zero effects can be below a useful floating-point decision margin; those rows remain in the primary metric, and the descriptive strata do not change the success criteria.

Error on difficult query/threshold pairs (rows are weighted equally, so a query near several thresholds appears several times):

| discovery_examples | seed | anchor_strength | method | difficult_mae | difficult_accuracy | decision_count |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 42 | 0 | first_order | 0.06283 | 0.5265 | 321 |
| 16 | 42 | 0 | full_second_order | 0.04656 | 0.4984 | 321 |
| 16 | 42 | 0 | sketch_rank_12 | 0.04656 | 0.5732 | 321 |
| 16 | 42 | 0 | sketch_rank_16 | 0.04656 | 0.5607 | 321 |
| 16 | 42 | 0 | sketch_rank_4 | 0.0453 | 0.5483 | 321 |
| 16 | 42 | 0 | sketch_rank_8 | 0.04657 | 0.5639 | 321 |
| 32 | 42 | 0 | first_order | 0.07551 | 0.7539 | 321 |
| 32 | 42 | 0 | full_second_order | 0.08417 | 0.7757 | 321 |
| 32 | 42 | 0 | sketch_rank_12 | 0.08417 | 0.7632 | 321 |
| 32 | 42 | 0 | sketch_rank_16 | 0.08417 | 0.7632 | 321 |
| 32 | 42 | 0 | sketch_rank_4 | 0.08391 | 0.7632 | 321 |
| 32 | 42 | 0 | sketch_rank_8 | 0.08418 | 0.7632 | 321 |
| 64 | 42 | 0 | first_order | 0.07134 | 0.8162 | 321 |
| 64 | 42 | 0 | full_second_order | 0.07734 | 0.8349 | 321 |
| 64 | 42 | 0 | sketch_rank_12 | 0.07734 | 0.8411 | 321 |
| 64 | 42 | 0 | sketch_rank_16 | 0.07734 | 0.8411 | 321 |
| 64 | 42 | 0 | sketch_rank_4 | 0.0771 | 0.838 | 321 |
| 64 | 42 | 0 | sketch_rank_8 | 0.07734 | 0.8411 | 321 |
| 128 | 42 | 0 | first_order | 0.04892 | 0.866 | 321 |
| 128 | 42 | 0 | full_second_order | 0.0484 | 0.8847 | 321 |
| 128 | 42 | 0 | sketch_rank_12 | 0.0484 | 0.8847 | 321 |
| 128 | 42 | 0 | sketch_rank_16 | 0.0484 | 0.8847 | 321 |
| 128 | 42 | 0 | sketch_rank_4 | 0.0482 | 0.8847 | 321 |
| 128 | 42 | 0 | sketch_rank_8 | 0.0484 | 0.8847 | 321 |

Illustrative first-order mistakes chosen on training labels only at the largest completed primary size. Rows are ranked by full-second-order training error reduction, then limited to distinct queries with train effect magnitude at least 0.01. This is a descriptive subset, not another success criterion. Validation columns show whether that training-selected correction transfers:

| edge | state | threshold | train_effect | validation_effect | first_prediction | full_prediction | validation_first_correct | validation_full_correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a9.h9->a11.h10<q> | 0000001000000000 | 1.386 | 1.633 | 1.524 | 1.183 | 1.832 | False | True |
| a5.h5->a8.h6<v> | 0000001000000000 | 1.613 | -1.589 | -1.517 | -1.792 | -1.737 | False | False |

Seed extension at n=128, both anchors. Gain intervals compare with first order at the same anchor:

| seed | anchor_strength | method | mae | difficult_accuracy | difficult_accuracy_gain_ci_low | difficult_accuracy_gain_ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 0 | first_order | 0.04865 | 0.866 | 0 | 0 |
| 42 | 0 | full_second_order | 0.04753 | 0.8847 | -0.01869 | 0.03115 |
| 42 | 0 | sketch_rank_12 | 0.04753 | 0.8847 | -0.01869 | 0.03115 |
| 42 | 0 | sketch_rank_16 | 0.04753 | 0.8847 | -0.01869 | 0.03115 |
| 42 | 0 | sketch_rank_4 | 0.04721 | 0.8847 | -0.01869 | 0.03115 |
| 42 | 0 | sketch_rank_8 | 0.04753 | 0.8847 | -0.01869 | 0.03115 |
| 42 | 0.5 | first_order | 0.08816 | 0.8598 | 0 | 0 |
| 42 | 0.5 | full_second_order | 0.04576 | 0.8972 | -0.07508 | 0.1246 |
| 42 | 0.5 | sketch_rank_12 | 0.04576 | 0.8972 | -0.07508 | 0.1246 |
| 42 | 0.5 | sketch_rank_16 | 0.04576 | 0.8972 | -0.07508 | 0.1246 |
| 42 | 0.5 | sketch_rank_4 | 0.04598 | 0.8972 | -0.07508 | 0.1246 |
| 42 | 0.5 | sketch_rank_8 | 0.04576 | 0.8972 | -0.07508 | 0.1246 |
| 43 | 0 | first_order | 0.06221 | 0.785 | 0 | 0 |
| 43 | 0 | full_second_order | 0.06182 | 0.7913 | -0.03738 | 0.02492 |
| 43 | 0 | sketch_rank_16 | 0.06182 | 0.7944 | -0.03427 | 0.02181 |
| 43 | 0 | sketch_rank_4 | 0.06029 | 0.7975 | -0.03115 | 0.02492 |
| 43 | 0.5 | first_order | 0.09196 | 0.785 | 0 | 0 |
| 43 | 0.5 | full_second_order | 0.05955 | 0.8193 | -0.04673 | 0.1215 |
| 43 | 0.5 | sketch_rank_16 | 0.05955 | 0.8193 | -0.04673 | 0.1215 |
| 43 | 0.5 | sketch_rank_4 | 0.05863 | 0.7882 | -0.07788 | 0.09034 |
| 44 | 0 | first_order | 0.1242 | 0.8193 | 0 | 0 |
| 44 | 0 | full_second_order | 0.1279 | 0.8349 | -0.02196 | 0.02804 |
| 44 | 0 | sketch_rank_16 | 0.1279 | 0.8349 | -0.02196 | 0.02804 |
| 44 | 0 | sketch_rank_4 | 0.1155 | 0.8349 | -0.02196 | 0.02804 |
| 44 | 0.5 | first_order | 0.1292 | 0.8629 | 0 | 0 |
| 44 | 0.5 | full_second_order | 0.1366 | 0.8785 | -0.02804 | 0.0405 |
| 44 | 0.5 | sketch_rank_16 | 0.1366 | 0.8785 | -0.02804 | 0.0405 |
| 44 | 0.5 | sketch_rank_4 | 0.1462 | 0.7601 | -0.1341 | 0.06542 |

Across-seed summaries at n=128 (seed standard deviation is descriptive; seed count differs for unreplicated ranks):

| anchor_strength | method | seeds | mae_mean | mae_seed_sd | difficult_accuracy_mean | difficult_accuracy_seed_sd |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | first_order | 3 | 0.07836 | 0.04028 | 0.8235 | 0.04066 |
| 0 | full_second_order | 3 | 0.07909 | 0.04289 | 0.837 | 0.04676 |
| 0 | sketch_rank_12 | 1 | 0.04753 | NA | 0.8847 | NA |
| 0 | sketch_rank_16 | 3 | 0.07909 | 0.04289 | 0.838 | 0.04525 |
| 0 | sketch_rank_4 | 3 | 0.07435 | 0.03627 | 0.839 | 0.04376 |
| 0 | sketch_rank_8 | 1 | 0.04753 | NA | 0.8847 | NA |
| 0.5 | first_order | 3 | 0.1031 | 0.02268 | 0.8359 | 0.04409 |
| 0.5 | full_second_order | 3 | 0.08062 | 0.04893 | 0.865 | 0.04066 |
| 0.5 | sketch_rank_12 | 1 | 0.04576 | NA | 0.8972 | NA |
| 0.5 | sketch_rank_16 | 3 | 0.08062 | 0.04893 | 0.865 | 0.04066 |
| 0.5 | sketch_rank_4 | 3 | 0.0836 | 0.05458 | 0.8152 | 0.07241 |
| 0.5 | sketch_rank_8 | 1 | 0.04576 | NA | 0.8972 | NA |

## 6. Compute accounting and break-even

| discovery_examples | method | wall_seconds | forward_passes_batches | backward_calls | hvp_operations | peak_cuda_allocated_mib | hypothetical_break_even_avoided_queries | available_candidate_queries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | first_order | 3.499 | 48 | 24 | 0 | 879 | 5 | 35 |
| 16 | full_second_order | 42.05 | 48 | 408 | 384 | 1175 | 58 | 35 |
| 16 | sketch_rank_12 | 31.71 | 48 | 312 | 288 | 1175 | 44 | 35 |
| 16 | sketch_rank_16 | 41.55 | 48 | 408 | 384 | 1175 | 58 | 35 |
| 16 | sketch_rank_4 | 13.28 | 48 | 120 | 96 | 1175 | 19 | 35 |
| 16 | sketch_rank_8 | 22.5 | 48 | 216 | 192 | 1175 | 32 | 35 |
| 32 | first_order | 7.721 | 96 | 48 | 0 | 879 | 6 | 35 |
| 32 | full_second_order | 82.82 | 96 | 816 | 768 | 1175 | 55 | 35 |
| 32 | sketch_rank_12 | 63.88 | 96 | 624 | 576 | 1175 | 43 | 35 |
| 32 | sketch_rank_16 | 83.14 | 96 | 816 | 768 | 1175 | 55 | 35 |
| 32 | sketch_rank_4 | 26.46 | 96 | 240 | 192 | 1175 | 18 | 35 |
| 32 | sketch_rank_8 | 44.91 | 96 | 432 | 384 | 1175 | 30 | 35 |
| 64 | first_order | 12.8 | 192 | 96 | 0 | 879 | 5 | 35 |
| 64 | full_second_order | 160.4 | 192 | 1632 | 1536 | 1175 | 61 | 35 |
| 64 | sketch_rank_12 | 126.4 | 192 | 1248 | 1152 | 1175 | 48 | 35 |
| 64 | sketch_rank_16 | 164.4 | 192 | 1632 | 1536 | 1175 | 63 | 35 |
| 64 | sketch_rank_4 | 51.53 | 192 | 480 | 384 | 1175 | 20 | 35 |
| 64 | sketch_rank_8 | 89.3 | 192 | 864 | 768 | 1175 | 34 | 35 |
| 128 | first_order | 28.42 | 384 | 192 | 0 | 887.4 | 5 | 35 |
| 128 | full_second_order | 319.6 | 384 | 3264 | 3072 | 1196 | 56 | 35 |
| 128 | sketch_rank_12 | 243.4 | 384 | 2496 | 2304 | 1196 | 43 | 35 |
| 128 | sketch_rank_16 | 316.8 | 384 | 3264 | 3072 | 1196 | 55 | 35 |
| 128 | sketch_rank_4 | 99.68 | 384 | 960 | 768 | 1196 | 18 | 35 |
| 128 | sketch_rank_8 | 171.2 | 384 | 1728 | 1536 | 1196 | 30 | 35 |

Costs include corrupted-donor preparation, patched forwards, backwards, direct HVPs, and matrix reconstruction. Only one batch graph is retained, making 128 prompts feasible on 12 GB. Full columns and randomized products reuse batch graph preparation; per-method costs allocate the measured shared preparation and corresponding directional work to each standalone method. These allocations are **not additive**. Raw `runtime_breakdown.csv` counts the actual campaign work once; peak memory for curvature methods is a measured shared-stage upper bound.

The exact-query reference is measured using three fresh train candidate evaluations with current-state score already known. Break-even = ceiling(standalone construction seconds / measured fresh-query seconds). It is a **hypothetical no-verification lower bound** on queries that must be avoided, not a measured speedup. Every refresh adds construction again; verification adds exact query cost and can only worsen this bound. This final diagnostic did not execute another hybrid search or claim verification savings. The historical verified pilot saved zero exact decisions. A separately labeled sensitivity column assumes a 2.35x faster exact baseline based on the independent report; that speedup was not measured in this repository.

Matched-compute first-order control: use the largest actually measured first-order discovery set whose construction fits the curvature method budget, chosen by sample count and cost rather than validation quality. This directly tests spending compute on more data instead of curvature. Across 20 measured primary curvature/budget comparisons, curvature has lower MAE in 10 and higher difficult-decision accuracy in 5. These paired rows reuse data and are not independent trials. The available first-order control is capped at 128 examples even when more would fit the budget.

| curvature_examples | curvature_method | curvature_seconds | curvature_mae | curvature_difficult_accuracy | first_order_examples_at_budget | first_order_seconds | first_order_mae | first_order_difficult_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | full_second_order | 42.05 | 0.04675 | 0.4984 | 128 | 28.42 | 0.04865 | 0.866 |
| 16 | sketch_rank_12 | 31.71 | 0.04675 | 0.5732 | 128 | 28.42 | 0.04865 | 0.866 |
| 16 | sketch_rank_16 | 41.55 | 0.04675 | 0.5607 | 128 | 28.42 | 0.04865 | 0.866 |
| 16 | sketch_rank_4 | 13.28 | 0.04528 | 0.5483 | 64 | 12.8 | 0.07011 | 0.8162 |
| 16 | sketch_rank_8 | 22.5 | 0.04676 | 0.5639 | 64 | 12.8 | 0.07011 | 0.8162 |
| 32 | full_second_order | 82.82 | 0.0846 | 0.7757 | 128 | 28.42 | 0.04865 | 0.866 |
| 32 | sketch_rank_12 | 63.88 | 0.0846 | 0.7632 | 128 | 28.42 | 0.04865 | 0.866 |
| 32 | sketch_rank_16 | 83.14 | 0.0846 | 0.7632 | 128 | 28.42 | 0.04865 | 0.866 |
| 32 | sketch_rank_4 | 26.46 | 0.08424 | 0.7632 | 64 | 12.8 | 0.07011 | 0.8162 |
| 32 | sketch_rank_8 | 44.91 | 0.0846 | 0.7632 | 128 | 28.42 | 0.04865 | 0.866 |
| 64 | full_second_order | 160.4 | 0.07565 | 0.8349 | 128 | 28.42 | 0.04865 | 0.866 |
| 64 | sketch_rank_12 | 126.4 | 0.07565 | 0.8411 | 128 | 28.42 | 0.04865 | 0.866 |
| 64 | sketch_rank_16 | 164.4 | 0.07565 | 0.8411 | 128 | 28.42 | 0.04865 | 0.866 |
| 64 | sketch_rank_4 | 51.53 | 0.07531 | 0.838 | 128 | 28.42 | 0.04865 | 0.866 |
| 64 | sketch_rank_8 | 89.3 | 0.07565 | 0.8411 | 128 | 28.42 | 0.04865 | 0.866 |
| 128 | full_second_order | 319.6 | 0.04753 | 0.8847 | 128 | 28.42 | 0.04865 | 0.866 |
| 128 | sketch_rank_12 | 243.4 | 0.04753 | 0.8847 | 128 | 28.42 | 0.04865 | 0.866 |
| 128 | sketch_rank_16 | 316.8 | 0.04753 | 0.8847 | 128 | 28.42 | 0.04865 | 0.866 |
| 128 | sketch_rank_4 | 99.68 | 0.04721 | 0.8847 | 128 | 28.42 | 0.04865 | 0.866 |
| 128 | sketch_rank_8 | 171.2 | 0.04753 | 0.8847 | 128 | 28.42 | 0.04865 | 0.866 |

The fixed query benchmark uses 38 distinct binary states (current states plus candidate removals), catalogued in `diagnostic_binary_states.csv`; each is evaluated on the relevant train subset and the identical validation subset. Validation states are cached across jobs. Raw ledger mask/cache trackers describe their live process session; their cumulative-mask field does not become a campaign-wide count when saved rows are restored after interruption. Forward-batch and example-forward logs remain reconciled across restarts.

The fixed candidate bank was selected by official EAP on 16 training prompts in the prior pilot (4.16 s attribution stage). That historical common prerequisite is reused and is not counted as new campaign work.

The full six-configuration campaign was executed through the batch-streamed CLI. `notebooks/mib_sketch_diagnostics.smoke.executed.ipynb` separately records execution of every notebook code cell on the explicitly labeled small smoke configuration, including the optional Hessian and the final campaign dashboard. It is not presented as the full campaign execution transcript.

## 7. Limitations and reproducibility

Restricted 16-edge graph; no full-model circuit, CPR, CMD, or leaderboard score. Three train-selected partial states and fixed candidate rankings limit generality. Query selection uses the first 16 training examples, so sample-efficiency curves condition on that fixed query set. The fixed validation subset has appeared in earlier project experiments and is not a fresh private confirmation set. Bootstrap intervals represent prompt sampling conditional on each estimator, not an independent replication or a multiple-testing-adjusted population claim. Additional seeds change train subsets and sketch directions together; those effects are not separately identified. The larger datasets remain nested within each seed. Seed extensions replicate the training-selected sketch, the rank-4 budget control from the same HVP prefix, and first-order/full-Hessian controls, not every rank. Public test remains unused.

Reproduce with `scripts/run_sketch_diagnostics_final.py --sizes 16 32 64 128 --out <run-directory>`. Sizes and seeds are independently configurable; context bundles and completion markers support restart. Exact source hashes, package/model provenance, train/validation manifests, random seeds, selected edges, thresholds, source reports, and incomplete configurations are saved. No smoke fallback occurs. Model/device errors stop the configuration.

Actual completed campaign stages: 86; forward batches 32884; example-forwards 65768; backward calls 50596; HVP operations 47617; measured stage wall seconds 6248.54. This excludes initial model/data loading and report rendering, not omitted algorithm work.

## 8. Independent ACDC reports: attributed context, not our measurements

Source snapshots and SHA-256 values are saved in `results/mib_sketch_diagnostics/final/20261015T070721Z/external_reports/` and the configuration. [Stage G](results/mib_sketch_diagnostics/final/20261015T070721Z/external_reports/ACDC_FP32_SCALE_REPORT.md) reports no reliable sketch quality benefit over prior-order contiguous groups at 7,588 edges. [Stage H](results/mib_sketch_diagnostics/final/20261015T070721Z/external_reports/ACDC_GPT2_REPORT.md) reports 32,491-edge GPT-2 batching at about 6% of its ACDC reference cost but 20–30% worse held-out loss; sketch bank cost offsets most saved checks. [Stage I](results/mib_sketch_diagnostics/final/20261015T070721Z/external_reports/ACDC_GPT2_GAP_REPORT.md) reports that 64 discovery pairs substantially improve held-out quality, while a good first-order prior plus contiguous grouping remains difficult to beat and the sketch costs about 1.2x for similar quality. [RESEARCH_LOG](results/mib_sketch_diagnostics/final/20261015T070721Z/external_reports/RESEARCH_LOG.md) describes input-direction failures on sparse deletions, improved local ranking with output-space Gauss–Newton sketches, and a 2.35x prefix-cache optimization preserving ACDC decisions.

Those experiments use a squared probability-distance loss, different graph scales and data roles, and exact-checked group deletions. Our scalar IOI logit-difference and restricted one-edge queries do not reproduce them or support a direct numeric comparison. They motivate larger discovery sets, strong first-order and optimized-exact controls, and a demanding end-to-end cost test. No output-space or group-deletion algorithm was implemented in this wrap-up.

## 9. Research decision

PAUSE the Hessian-based Sketch-ACDC approach. This restricted experiment does not satisfy both decision-quality and amortization requirements. Finish documentation before separately considering MAttr; no MAttr experiment was started.

Continue requires repeatable difficult-decision gains across prompts/seeds and a plausible amortization route. Full Taylor accuracy with poor/expensive Hessian compression would motivate a separate output-space investigation. Weak incremental value or construction/verification cost beyond plausible savings supports pausing input-Hessian Sketch-ACDC and considering MAttr as a separately authorized next study. These are the supplied qualitative criteria. The conservative evidence checks below operationalize them for this report; they are not a formally preregistered hypothesis test. The training-only seed-extension rule was recorded in the campaign configuration before the primary runs.

Training-only seed-extension selection: {"method": "sketch_rank_16", "anchor_strength": 0.0, "train_mae_gain_fraction": 0.44314580693314987, "train_difficult_gain": 0.0311526479750778, "train_mae": 0.0120945571286722}. Pending extension seeds: []. The selection takes the lowest training MAE among qualifying rows; tiny differences among nearly identical high-rank estimators do not establish a scientifically optimal rank. The supplied consistency criterion is interpreted conservatively: directionally consistent difficult-decision gains across discovery sizes/seeds, paired-prompt intervals supporting the seed results, and a no-verification break-even within available candidate queries. Mixed or uncertain evidence does not establish the requested continuation case.

## Measured figures

![Prediction MAE versus discovery examples](<results/mib_sketch_diagnostics/final/20261015T070721Z/sample_efficiency_mae.png>)

![Difficult-decision accuracy versus discovery examples](<results/mib_sketch_diagnostics/final/20261015T070721Z/sample_efficiency_difficult_accuracy.png>)

![Sketch rank versus prediction error](<results/mib_sketch_diagnostics/final/20261015T070721Z/rank_vs_error.png>)

![Measured computation versus prediction quality](<results/mib_sketch_diagnostics/final/20261015T070721Z/compute_vs_quality.png>)

![Difficult-decision accuracy versus computation](<results/mib_sketch_diagnostics/final/20261015T070721Z/accuracy_vs_cost.png>)

![Decision accuracy versus threshold](<results/mib_sketch_diagnostics/final/20261015T070721Z/accuracy_vs_threshold.png>)

![Prediction error on difficult decisions](<results/mib_sketch_diagnostics/final/20261015T070721Z/difficult_decision_errors.png>)

![Midpoint comparison](<results/mib_sketch_diagnostics/final/20261015T070721Z/midpoint_comparison.png>)

![Error across partially pruned states](<results/mib_sketch_diagnostics/final/20261015T070721Z/error_vs_pruned_edges.png>)
