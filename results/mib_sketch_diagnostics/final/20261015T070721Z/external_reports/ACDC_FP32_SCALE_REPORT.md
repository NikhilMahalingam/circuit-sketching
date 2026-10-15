# Float32 scale-up with 16 discovery pairs: Stage G report

**With 16 discovery pairs on the 7,588-edge graph, sketch-composed large groups remain the cheapest exact-checked search, but plain prior-order groups give better circuits at the same cost.**

- **Cost.** Sketch-composed groups up to 256 used about 0.20× ACDC's compute, tied with the no-sketch contiguous groups (0.21×) and below random proposals (0.38×). They used 86% fewer exact checks than ACDC and 28% fewer than contiguous groups.
- **Quality against ACDC.** They met the 10% held-out allowance in 6/6 runs.
- **Quality against the controls.** Their held-out loss was higher than contiguous groups' in all 6 seed comparisons (median 1.13×, 3 clearly) and slightly higher than random proposals' (median 1.06×).
- **The Stage F frontier result does not survive.** Its quality advantage on the 4 original pairs disappears with adequate discovery data.

The registered float32 agreement gate passed exactly. These results follow the registered [plan](ACDC_FP32_SCALE_PLAN.md). Raw results are in [results/acdc_fp32_scale_20261003/](results/acdc_fp32_scale_20261003/), with every run in [tables.md](results/acdc_fp32_scale_20261003/tables.md).

## Precision change and agreement gate

Float32 was adopted for this stage with James's approval. In float64, 16 pairs cost 1.20 s per exact check and backward passes ran out of GPU memory. On Stage B's data, float32 ACDC reproduced the float64 run decision for decision:
- the same 785-edge circuit (Jaccard 1.0);
- 100% decision agreement;
- held-out ratio 1.000.

Losses differed by at most 2×10⁻⁶ over 20 random masks, and the gate-path IG prior ranking was unchanged (Spearman 1.000). All four registered criteria passed. The float32 helpers reproduce the registered float64 helpers exactly when run in float64 (unit test).

## Integrity

- **Data.** Two fresh 16-pair discovery sets (fresh-pool prompts 304–319 and 320–335) and a fresh 128-pair holdout (336–463), all disjoint from every earlier use.
- **Completion.** All 14 runs and the holdout completed with no failures, on AC power. Validation-only selections were frozen before the holdout opened.
- **Power capping.** The GPU was power-capped for several runs (per-check cost up to 1.06 s against 0.343 s uncapped). Costs are compared in each run's per-check units.
- **Disclosures.** A smoke run checked every stage before registration. [analyze.py](acdc_fp32_study/analyze.py) was written during the run.

## Results

ACDC references with 16 pairs. These are far more faithful than the 4-pair reference in Stage F, which reached 53% of the empty-circuit loss.

| Block | Edges | Held-out loss | Share of empty-circuit loss (0.645) |
| --- | --- | --- | --- |
| Set 0 | 614 | 0.249 | 39% |
| Set 1 | 803 | 0.158 | 24% |

| Arm | Meets 10% | Held-out / ACDC at k (median) | At the 10% point | Exact checks at k | Cost at k (× ACDC) | Banks |
| --- | --- | --- | --- | --- | --- | --- |
| `random_prior` (4 runs) | 3/4 | 0.918 | 0.872 | 2,774 | 0.38 | 0 |
| `contig_G256` (2 runs) | 2/2 | **0.858** | **0.848** | 1,530 | 0.21 | 0 |
| `greedy_G256` (6 runs) | **6/6** | 0.971 | 0.917 | **1,096** | **0.20** | 15 |

**Comparisons at k** (held-out ratio with 95% paired-example intervals):

| Comparison | Pairs | First lower | Clearly lower | Clearly higher | Median ratio |
| --- | --- | --- | --- | --- | --- |
| `greedy_G256` vs `contig_G256` | 6 | 0 | 0 | 3 | 1.132 |
| `greedy_G256` vs `random_prior` | 6 | 2 | 0 | 2 | 1.061 |
| `contig_G256` vs `random_prior` | 2 | 1 | 1 | 0 | 0.958 |

## Registered expectations

- **E1, `greedy_G256` uses under half of `random_prior`'s checks in both blocks: held** (31–40% in set 0 and 40–47% in set 1).
- **E2, `greedy_G256` meets the allowance in most runs: held** (6/6).
- **E3, `greedy_G256` is no worse than `random_prior` (median paired ratio ≤ 1.05): did not hold,** narrowly (1.061).
- **E4, `greedy_G256` costs less than `random_prior` and ACDC: held** (0.20× against 0.38× and 1).

## Interpretation

With adequate discovery data, three things hold on this graph:
- **Every exact-checked large-group method is comparable to or better than ACDC** at about a fifth of its compute.
- **The sketch's role is reducing exact checks.** It saves 28% of checks against contiguous groups, but 16-pair banks cost more, so total cost is a tie. It does not improve circuits: plain contiguous groups in gate-path IG prior order are better in every paired comparison here.
- **The sketch is the most consistent arm** against ACDC (6/6 within the allowance), but its typical circuit is a little worse than the cheap controls'.

On the 1,281-edge graph with 16 pairs (Stage E), contiguous groups beat the sketch in one of three discovery sets and lost in the other two. Across Stages E–G, the robust conclusion is that discovery-set choice and size matter more than sketch-versus-no-sketch, and the sketch shows no reliable quality benefit.

**What remains open for sketching:**
- **Larger graphs.** Exact checks saved scale with the graph while bank cost does not, so whether the saved checks become a net cost saving at, for example, GPT-2 small's 32,491 edges is untested.
- **Other uses.** Roles other than composing deletion groups.

Not established: results with more than two discovery sets at this scale, other tasks, or population-level quality.
