# GPT-2 small scale-up: Stage H report

**On GPT-2 small (32,491 edges, IOI), exact-checked large-group search costs about 6% of ACDC's compute, but its circuits are clearly worse than ACDC's. The sketch is cost-neutral at best, with a small quality edge that is mostly within noise.**

- **Cost.** At ACDC's circuit size, every batch arm used 0.04–0.13× ACDC's compute in per-check units, and 1,100–2,300 exact checks against ACDC's 32,492. The saving grew with the graph: at 7,588 edges it was about 0.2×.
- **Quality against ACDC.** The ACDC references are much stronger here: 21% and 27% of the empty-circuit loss. At their size, batch circuits had 1.2–1.3× ACDC's held-out loss (median by arm). Only 10/22 runs met the 10% allowance, and nine of those ten came from discovery set 1 (set 0: 1/11).
- **The sketch.**
  - With groups up to 1,024, sketch composition was cost-neutral against contiguous groups (median 0.99×), used 18% fewer checks, and had slightly lower held-out loss (median 0.94×; 1 pair clearly lower, 1 clearly higher).
  - The lazy sketch arm had the best quality against contiguous groups (median 0.92×; 5/6 lower, none clearly higher) but cost 1.18× as much.
  - With groups up to 256, the sketch rebuilt 64 banks and cost 1.6× contiguous groups.

The precision gate passed, and all 24 runs and the holdout completed with no failures. These results follow the registered [plan](ACDC_GPT2_PLAN.md). Raw results are in [results/acdc_gpt2_20261003/](results/acdc_gpt2_20261003/), with every run in [tables.md](results/acdc_gpt2_20261003/tables.md).

## Integrity

- **Precision gate.** Float32 matched float64 to 1.7×10⁻⁶ over 20 random masks, the prior ranking agreed (Spearman 1.000), and the first 2,000 ACDC decisions were identical.
- **Threshold.** Block 0's ACDC at τ = 0.0003 kept 686 edges with validation loss 0.088, against an empty-circuit loss of 0.441, so the fallback did not trigger.
- **Completion.** All runs finished before the cutoff, on AC power. Validation-only selections were frozen before the 128-pair holdout opened.
- **Power capping.** The GPU was power-capped for much of the campaign: per-check time ranged from 0.287 s to 0.817 s, against 0.286 s at full power. Search took 20.5 h of wall time instead of the projected 10–16 h. Costs are compared in each run's own per-check units. Measured seconds are in tables.md.
- **Disclosures.**
  - Feasibility measurements and a full-pipeline smoke run used prompts 900–915, which are excluded from the study.
  - [analyze.py](acdc_gpt2/analyze.py) was written before the run started.
  - `lazy_G1024` was added after the smoke run and before registration.
  - Sketch arms are charged one 8-second warm-up bank, including lazy runs.

## Results

| Block | ACDC edges (k) | ACDC held-out loss | Share of empty-circuit loss (0.321) | ACDC measured time |
| --- | --- | --- | --- | --- |
| Set 0 | 686 | 0.066 | 21% | 3.3 h |
| Set 1 | 717 | 0.087 | 27% | 6.0 h (power-capped) |

| Arm | Meets 10% | Held-out / ACDC at k (median) | Exact checks at k | Cost at k (× ACDC) | Banks |
| --- | --- | --- | --- | --- | --- |
| `contig_G256` (no sketch) | 1/2 | 1.30 | 1,878 | 0.059 | 0 |
| `contig_G1024` (no sketch) | 1/2 | 1.30 | 1,826 | 0.059 | 0 |
| `greedy_G256` | 3/6 | 1.27 | 1,308 | 0.088 | 64 |
| `greedy_G1024` | 3/6 | 1.21 | 1,492 | 0.061 | 16 |
| `lazy_G1024` | 2/6 | 1.22 | 1,924 | 0.067 | 8 |

The two contiguous arms found identical circuits at k in both blocks. Raising the cap from 256 to 1,024 changed only the check count.

**Sketch arms against their contiguous control at k** (held-out ratio with 95% paired-example intervals):

| Comparison | Pairs | Sketch lower | Clearly lower | Clearly higher | Median loss ratio | Median cost ratio | Median check ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `greedy_G256` vs `contig_G256` | 6 | 3 | 0 | 2 | 0.998 | 1.63 | 0.67 |
| `greedy_G1024` vs `contig_G1024` | 6 | 4 | 1 | 1 | 0.943 | 0.99 | 0.82 |
| `lazy_G1024` vs `contig_G1024` | 6 | 5 | 1 | 0 | 0.924 | 1.18 | 1.10 |
| `greedy_G1024` vs `greedy_G256` | 6 | 3 | 3 | 1 | 0.931 | 0.62 | 1.14 |

## Registered expectations

- **E1, `greedy_G256` uses fewer checks than `contig_G256` in every pair: held** (6/6; 1,206–1,377 against 1,684 and 2,073).
- **E2, `contig_G256` beats `greedy_G256` on held-out loss in most pairs: did not hold** (3/6, median ratio 0.998). Stage G's contiguous advantage did not replicate.
- **E3, every batch arm meets the allowance in most runs and costs under 0.3× ACDC: cost held, quality did not.** All costs were 0.04–0.13×, but no arm met the allowance in most runs.
- **E4, `lazy_G1024` builds fewer than half of `greedy_G1024`'s banks: did not hold, narrowly** (median 8 against 16; strictly fewer than half in 3 of 6 pairs).
- **Open question, whether any sketch arm costs less than its contiguous control: no.** `greedy_G1024` tied (median 0.99×, 3 of 6 pairs cheaper), and the other sketch arms cost more.

## Interpretation

- **Scale makes batching much cheaper.** Most of the graph is deleted in a few large accepted groups, so checks grow far more slowly than the edge count: about 1,900 checks at 32,491 edges against about 1,500 at 7,588.
- **Quality is now the binding constraint.** Against a strong ACDC reference, the batch circuits are 20–30% worse at the same size, and the shortfall depends heavily on the discovery set (set 0: 1/11 runs met the allowance; set 1: 9/11). Two candidate causes are untested:
  - the prior order instead of ACDC's reverse-topological order;
  - the allowance, under which each accepted group may spend up to 4τ.
- **The sketch's cost case is not established.** Bank cost still offsets the saved checks. Larger groups with a larger radius help: `greedy_G1024` costs 0.62× `greedy_G256` with 16 banks instead of 64. The lazy variant cut banks to 8, but it needed more checks.
- **The sketch's quality signal reversed from Stage G.** All three sketch arms had median held-out loss at or below contiguous groups', with the lazy arm the most consistent. The effects are small, few pairs are clearly separated, and Stages E–G have shown such directions flip between discovery sets. This is suggestive, not established.

Not established: behavior on other tasks, more discovery sets, or any fix for the quality gap.
