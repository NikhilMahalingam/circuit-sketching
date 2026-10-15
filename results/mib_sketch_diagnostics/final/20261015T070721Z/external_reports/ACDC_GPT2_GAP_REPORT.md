# The GPT-2 quality gap: Stage I report

**More discovery data closes most of the gap; ACDC's edge order makes it worse; and the sketch adds nothing in the best configuration.**

- **Data.** With 64 discovery pairs instead of 16, contiguous large-group search reached the 16-pair ACDC reference's held-out quality: 0.98× ACDC's loss at its size (median of two sets). It did so at 0.15–0.18× ACDC's compute. Every 64-pair arm improved on its 16-pair counterpart: median ratios 0.78–0.86, with 7 of 15 pairs clearly lower.
- **Order.** In ACDC's reverse-topological order, group search was much worse: 1.33–1.47× the held-out loss of prior order (8 of 10 pairs clearly higher). It also needed about twice the checks.
- **Sketch.** At 64 pairs in prior order, sketch-composed and lazy groups matched contiguous groups in quality (median 1.05×, no pair clearly worse) but cost 1.2× as much. They saved no checks there. The sketch saved checks and cost (0.76× checks, 0.89× cost) only in reverse-topological order, the worse order.

These results follow the registered [plan](ACDC_GPT2_GAP_PLAN.md) with one compute-window [amendment](results/acdc_gpt2_gap_20261004/amendment_1.json). Raw results are in [results/acdc_gpt2_gap_20261004/](results/acdc_gpt2_gap_20261004/), with every run in [tables.md](results/acdc_gpt2_gap_20261004/tables.md).

## Integrity

- **Completion.** 23 of 24 runs completed. The last, lowest-priority run (`d1_m16/lazy_G1024_m64_s2`) hit the cutoff and is recorded as failed. The holdout of 128 fresh pairs (prompts 176–303, empty-circuit loss 0.336) was opened only after validation-only selections were frozen.
- **Pause.** At James's request the campaign was paused for about 8.5 hours after 13 runs. The process was stopped rather than suspended, so no paused time entered any run's timing. The interrupted run restarted from scratch. Amendment 1 extended the cutoff by the paused time and changed nothing else.
- **Miscount.** The amendment record says "13 of 22 runs"; the schedule has 24 runs. An [erratum](results/acdc_gpt2_gap_20261004/amendment_1_erratum.json) records the correction.
- **Chunked computation.** The 64-pair loss, prior and sketch were computed in exact chunks of 16 pairs. On real GPT-2 the chunked loss matched the unchunked one to 4×10⁻⁷.
- **Power capping.** The GPU was power-capped for most 64-pair runs: about 3.4 s per check against 1.15 s at full power. Costs are in 16-pair check units, using each run's own mean check time and a check-cost ratio measured at its start (3.94–4.10).
- **Reuse from Stage H.** Stage H's ACDC references and all of its runs were re-scored on this holdout. Their selections are identical to Stage H's.
- **Disclosure.** [analyze.py](acdc_gpt2_gap/analyze.py) was written before the run started.

## Results

ACDC references on the fresh holdout, both from 16 pairs:

| Block | Edges | Held-out loss | Share of empty-circuit loss |
| --- | --- | --- | --- |
| Set 0 | 686 | 0.057 | 17% |
| Set 1 | 717 | 0.067 | 20% |

| Arm | Pairs | Order | Meets 10% (set 0, set 1) | Held-out / ACDC at k (median) | Discovery loss at k | Exact checks at k | Cost at k (× ACDC) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `contig_G1024` (Stage H) | 16 | prior | 0/2 (0, 0) | 1.26 | 0.011 | 1,826 | 0.059 |
| `greedy_G1024` (Stage H) | 16 | prior | 2/6 (0, 2) | 1.24 | 0.014 | 1,492 | 0.061 |
| `lazy_G1024` (Stage H) | 16 | prior | 2/6 (0, 2) | 1.18 | 0.014 | 1,924 | 0.067 |
| `contig_G1024_m64` | 64 | prior | 1/2 (0, 1) | **0.98** | 0.039 | 1,297 | 0.166 |
| `greedy_G1024_m64` | 64 | prior | 3/6 (0, 3) | 1.04 | 0.036 | 1,247 | 0.206 |
| `lazy_G1024_m64` | 64 | prior | 2/5 (0, 2) | 1.10 | 0.038 | 1,506 | 0.214 |
| `contig_topo_G1024` | 16 | reverse topological | 0/2 | 1.67 | 0.037 | 3,664 | 0.114 |
| `greedy_topo_G1024` | 16 | reverse topological | 0/6 | 1.66 | 0.029 | 2,794 | 0.101 |
| `contig_topo_G1024_m64` | 64 | reverse topological | 0/2 | 1.43 | 0.064 | 3,558 | 0.435 |

The 64-pair prior-order arms are close to ACDC in set 1 (0.78–0.99×, all 6 runs meet the allowance) but not in set 0 (1.05–1.19×, 0 of 7 meet). The set 0 contiguous run is within 10% at k (1.05×), but its validation-selected point is not (1.13×).

**Paired comparisons at k:**

| Question | Comparison | Pairs | First lower | Clearly lower | Clearly higher | Median loss ratio | Median cost ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Data | `contig_G1024_m64` vs `contig_G1024` | 2 | 2 | 2 | 0 | 0.776 | 2.88 |
| Data | `greedy_G1024_m64` vs `greedy_G1024` | 6 | 6 | 2 | 0 | 0.853 | 3.97 |
| Data | `lazy_G1024_m64` vs `lazy_G1024` | 5 | 4 | 2 | 0 | 0.840 | 3.06 |
| Data | `contig_topo_G1024_m64` vs `contig_topo_G1024` | 2 | 2 | 1 | 0 | 0.861 | 3.81 |
| Order | `contig_topo_G1024` vs `contig_G1024` | 2 | 0 | 0 | 1 | 1.330 | 1.97 |
| Order | `greedy_topo_G1024` vs `greedy_G1024` | 6 | 0 | 0 | 5 | 1.348 | 1.62 |
| Order | `contig_topo_G1024_m64` vs `contig_G1024_m64` | 2 | 0 | 0 | 2 | 1.465 | 2.63 |
| Sketch | `greedy_G1024_m64` vs `contig_G1024_m64` | 6 | 2 | 0 | 0 | 1.048 | 1.20 |
| Sketch | `lazy_G1024_m64` vs `contig_G1024_m64` | 5 | 2 | 1 | 0 | 1.049 | 1.21 |
| Sketch | `greedy_topo_G1024` vs `contig_topo_G1024` | 6 | 2 | 0 | 1 | 1.016 | 0.89 |
| Sketch | `greedy_G1024` vs `contig_G1024` (Stage H, re-scored) | 6 | 3 | 0 | 0 | 0.987 | 0.99 |

## Registered expectations

- **E1, every 64-pair arm beats its 16-pair counterpart in most pairs: held** (2/2, 6/6, 4/5, 2/2).
- **E2, `contig_G1024_m64` meets the allowance in both blocks at under 0.5× ACDC's compute: did not hold.** It met it in set 1, and in set 0 it was within 10% at k but not at its selected point. Its cost was 0.15–0.18×.
- **E3, `greedy_G1024_m64` uses fewer checks than `contig_G1024_m64` in every pair: did not hold** (3 of 6; median check ratio 0.97).
- **Open questions.** Order: reverse-topological was clearly worse. Sketch at 64 pairs: no quality difference, and higher cost.

## Interpretation

- **The gap was mostly overfitting to 16 discovery pairs, and cheap search can buy more data.**
  - At 64 pairs, circuits fit their discovery pairs about 3× less tightly (0.031–0.046 against 0.010–0.014) but generalize better.
  - Contiguous groups in prior order then match the 16-pair ACDC reference at about a sixth of its compute.
  - This compares methods at different data sizes, roughly at matched compute. A 64-pair ACDC, about 4× the cost of the 16-pair one, would very likely also improve, as every method did in Stage E. So it is not a claim that group search beats ACDC on equal data.
- **ACDC's order is not what makes ACDC good.** Reverse-topological order made group search fit its own discovery data worse (0.037 against 0.011 at 16 pairs) and generalize worse. ACDC's advantage at equal data, which showed up in Stage H, likely comes from deciding edges one at a time, not from its order.
- **The prior already provides what the sketch adds.** With a good order, contiguous groups are accepted as often as sketch-composed ones, so the sketch saves no checks and its banks are pure cost. It helps only when the order is poor: in reverse-topological order it cut checks 24% and cost 11%. Stage H's slight quality edge for the sketch (0.94× on Stage H's holdout) shrank to 0.99× on this fresh holdout, consistent with noise.

Not established: a 64-pair ACDC reference, more discovery sets, other tasks, and why set 0 stays harder than set 1.
