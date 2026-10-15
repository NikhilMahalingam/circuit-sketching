# Sparse circuit-discovery pilot

The [Stage I report](ACDC_GPT2_GAP_REPORT.md) diagnoses Stage H's quality gap on GPT-2 small.
Stage H's batch circuits matched ACDC on discovery loss but not held-out loss, so Stage I tests
64 discovery pairs and ACDC's reverse-topological order against Stage H's 16-pair ACDC references,
on a fresh holdout. With 64 pairs, contiguous group search matched the 16-pair ACDC's held-out
quality (0.98× at its size) at 0.15–0.18× its compute. Every 64-pair arm improved on its 16-pair
version (median 0.78–0.86×). Reverse-topological order made group search much worse (1.33–1.47×).
The sketch added nothing with the prior order: equal quality at 1.2× the cost. It cut checks and
cost only in the worse, reverse-topological order.

The [Stage H report](ACDC_GPT2_REPORT.md) scales exact-checked large-group search to GPT-2 small
(32,491 edges, IOI) in float32, against two 16-pair ACDC references. Every batch arm cost
0.04–0.13× ACDC's compute at ACDC's circuit size, about 1,900 checks against 32,492. But the
circuits were 1.2–1.3× worse than ACDC's on held-out data, and only 10/22 runs met the 10%
allowance (1/11 on discovery set 0). Sketch-composed groups of up to 1,024 tied contiguous groups
on cost, with 18% fewer checks and a slight, mostly non-significant quality edge (median 0.94×).
A lazy sketch arm had the best quality against contiguous groups (median 0.92×) at 1.18× their cost.
A running record of all stages is in [RESEARCH_LOG.md](RESEARCH_LOG.md).

The [Stage G report](ACDC_FP32_SCALE_REPORT.md) repeats the 7,588-edge comparison in float32,
using two fresh 16-pair discovery sets and a fresh 128-pair holdout. A registered agreement gate
passed exactly: float32 ACDC matched float64 decision for decision. Sketch-composed groups of up
to 256 edges met the 10% held-out allowance in 6/6 runs, at 0.20× ACDC's compute with 86% fewer
exact checks. That cost ties contiguous groups (0.21×), even though the sketch used 28% fewer
checks. But contiguous prior-order groups had lower held-out loss in all 6 paired comparisons
(median 1.13×, 3 clearly), so Stage F's frontier result does not survive adequate discovery data.

The [Stage F report](ACDC_SOLU_GROUPS_REPORT.md) scales the large-group comparison to the
7,588-edge SoLU-6L graph. Sketch-composed groups of up to 256 edges (radius 512) met the 10%
held-out allowance in 3/3 seeds, with held-out loss 0.925× ACDC's at 785 edges. They used 868
exact checks against ACDC's 7,589 (3× fewer than random proposals) at about 0.20× ACDC's compute,
or 3.6× faster in measured uncapped runs. That puts the sketch arm on the cost–quality frontier
among batch methods: tied with contiguous groups on cost, and matching or beating the controls on
quality. The margins are small, with three seeds and the original 4 discovery pairs, which
Stage E showed flatter batch methods.

The [Stage E report](ACDC_DISCOVERY_SIZE_REPORT.md) reruns the large-group comparison on the
1,281-edge graph with three resampled discovery sets of 4 and 16 pairs, drawn from a fresh pool
of 3,000 prompts made with ACDC's own docstring generator. Going from 4 to 16 discovery pairs
lowered held-out loss for every method in every set (median about 39%) and brought all batch
methods to within about ±15% of ACDC. The sketch-versus-contiguous comparison changes direction
across discovery sets, and the overfitting hypothesis is not consistently supported. On fresh
4-pair sets, batch methods were usually worse than ACDC (1.2–1.6× in two of three sets), so their
earlier wins on the original 4 pairs do not generalize.

The [Stage D report](ACDC_LARGE_GROUPS_REPORT.md) uses the output sketch to compose large
deletion groups (up to 16 or 64 edges), each accepted only under the strict allowance
τ·min(|G|, 4), following a [registered plan](ACDC_LARGE_GROUPS_PLAN.md). Sketch-composed groups
passed exact checks far more often, cutting exact checks by 44–47% against cap-4 random batching.
With groups up to 64, total compute was 0.54× ACDC, about the same as random batching. Every run of
every arm met the 10% held-out allowance. The no-sketch control, contiguous large groups in prior
order, was cheaper (0.47× ACDC) and had better held-out quality at 157 edges (0.88× against
0.92–1.00×). That control is a single deterministic run. Overfitting to the 4 discovery pairs is
a plausible, untested explanation.

The [Stage C report](ACDC_OUTPUT_SEARCH_REPORT.md) runs complete exact-checked searches on the
1,281-edge graph with group proposals from the output-space sketch, following a
[registered plan](ACDC_OUTPUT_SEARCH_PLAN.md). At radius 64 with the gradient-prior order, these
searches met the 10% held-out allowance in 5/5 seeds, with held-out loss 0.87–0.90× ACDC's at 157
edges and 0.83–0.86× ACDC's compute. They used 10–14% fewer exact checks than random proposals,
the first time a sketch has reduced checks. Against prior-ordered random proposals, though, the
gain is small and within noise: median paired ratio 0.96, while random reaches about 0.87× ACDC at
0.57× ACDC's compute. Radius 16 costs 1.7× ACDC with no quality gain.

The [output-space sketch diagnostic](ACDC_OUTPUT_SKETCH_REPORT.md) tests a different sketch on
the ranking audit's 18 frozen pools: a Gaussian sketch of the loss residual (S·J) under the
Gauss–Newton loss model, whose relative error is the same for every query. At the same cost as the
archived 4-direction input sketch, it cuts mean regret about 225× (257 edges) and 37× (1,281 edges),
raises rank correlation from 0.25/−0.04 to 0.60/0.72, and matches the oracle's acceptance on the larger
pools. Its remaining error is Gauss–Newton truncation, not sketch noise. These are local ranking
results only, with no search and no holdout.

The [Stage B report](ACDC_MLP_SCALE_REPORT.md) scales the sketch-ranked hybrid, in which
every deletion is still exact-checked, to a 7,588-edge graph with heads and MLPs
(`SoLU_6L768W_C4_Code`), following a [registered plan](ACDC_MLP_SCALE_PLAN.md) with one
compute-window amendment. With the gradient-prior order, the hybrid met the 10% held-out
allowance against ACDC in 3/3 seeds (median 0.95× ACDC), the best of all arms. In ACDC's
order it met it in 1/3, against 0/3 for random proposals. Sketch ranking beat same-seed random
proposals in 5 of 6 pairs, which is suggestive with three seeds. Each run rebuilt about 440
banks, roughly one full ACDC sweep of compute, so the hybrid cost 1.15–1.6× ACDC while
exact-checked batching without a sketch cost 0.35–0.51×. The reference ACDC itself is weak on
this task (65% of the empty-circuit loss). GPU power capping varied during the run, so costs are
normalized by each run's measured per-check time.

The [group-testing Stage A report](ACDC_GROUP_TESTING_REPORT.md) treats exact-checked
batch deletion as adaptive group testing on the 1,281-edge graph, following a
[registered plan](ACDC_GROUP_TESTING_PLAN.md). The generalized search reproduces the
archived random-batching trajectories exactly. Ordering groups by a 1-second gate-path
integrated-gradients prior lets cap-4 batching meet the 10% held-out allowance in 5/5
seeds, against 2/5 for the topological order, at about 45% less complete time than ACDC.
One deterministic uncapped prior-ordered search reached comparable fidelity with 56
checks instead of 1,282. That is a single run, and it is not yet replicated. Lifting the
cap with random groups failed: the size-scaled allowance τ·|G| becomes vacuous for
large groups. No sketch banks were used.

The [proposal ranking value synthesis](ACDC_RANKING_VALUE_REPORT.md) reuses the
completed controls and adds one registered 18-pool diagnostic on saved persistent
search states. Noiseless cubic ranking often reduces local regret, but acceptance
headroom is limited and better immediate ranking does not guarantee better final
circuits. Current construction costs greatly exceed measured checking savings.
No new search, bank construction, tuning sweep or holdout evaluation was run;
the random-batching speedup remains the practical result. Edge Pruning stays paused.

The [directional-bank construction audit](ACDC_BUILDER_AUDIT_REPORT.md)
tests scalar-constant reuse and CUDA graph replay while preserving cubic dense
R4 mathematics. Replay roughly halves small-model bank time but improves the
larger model by only 4–5%, below the preregistered promotion threshold.
Setup-inclusive searches, numerical checks, profiler limitations and the
unchanged quality evidence are reported separately. The earlier comparable-quality
random-batching speedup and negative sketch conclusion remain preserved.

The [persistence-overhead ablation](ACDC_PERSISTENCE_ABLATION_REPORT.md)
completed 54 primary AC-powered timing runs, six contemporaneous history-writing
controls, and nine preserved transition-block runs. Every trajectory, mask,
cache count and bank coefficient matches the archived experiment. With terminal
export, random batching achieves the locked 10% quality allowance and beats
ordinary and same-size ACDC in two of three seeds at both scales: roughly
41–43% lower complete time on the larger model and 7% on the small model.
The larger sketch control improves search time by 29%, but bank construction
still exceeds an entire ACDC search. Quality evaluations and operating points
are reused unchanged; these speedups belong to batching. All original timings
remain preserved as measurements of the earlier implementation.

The [persistent-batch scaling study](ACDC_BATCH_SCALING_REPORT.md) completed
both existing models (257 and 1,281 edges), three persistent seeds per method,
paired reset controls, six ACDC timing runs, and fresh 128-pair holdouts at each
scale. Scaling did not produce a sketch-assisted speedup: qualifying larger-model
sketch circuits cost about 213 seconds complete versus 98.6 seconds for ACDC.
Persistence saved total checks on the small paired control but increased them
on the larger control. Random batching remained the stronger runtime control;
quality varied across seeds and the new holdouts. All work counts, matched-size
and matched-fidelity comparisons, checkpoint overhead, and uncertainty are in
the report. The older interrupted scaling campaign remains separate.

The [controlled persistent-batch follow-up](ACDC_PERSISTENT_BATCH_REPORT.md)
completed five paired seeds per policy and evaluator, five ACDC timing repeats,
and a fresh 128-pair holdout. Persistence reduced rejection work in every pair.
Validation-selected random batching met the 10% quality allowance and reduced
charged total time by about 1% in four of five seeds. Sketch ranking qualified
in five seeds but remained slower than ACDC; the speed benefit belongs to random
batching with exact acceptance. All seeds, failures, timings, and uncertainty
are included in the report.

The [joint batch-deletion pilot](ACDC_JOINT_BATCH_REPORT.md) completed 23 searches
and a new 128-pair holdout. Sketch-ranked groups with exact acceptance produced
64–66-edge circuits meeting the 10% joint quality allowance across three seeds,
but took longer than ordinary exact ACDC. Rejected groups, exact checks, and
rebuilding erased the savings from fewer accepted steps. All earlier reports
remain unchanged; the separate scaling report remains paused.

The [bounded overnight recentering campaign](ACDC_OVERNIGHT_REPORT.md) completed
31 operational search paths, five deterministic-cubic search diagnostics and a
fresh 128-example small-model holdout. Genuine rebuilding did not produce a
competitive size/fidelity/runtime tradeoff. The larger-model replication is
explicitly partial after a wall-clock interruption and deadline timeout; saved
checkpoints and accounting are in `results/acdc_overnight_20261001/`.

The [genuine adaptive cubic experiment](ACDC_ADAPTIVE_REPORT.md) recomputed
directional coefficients at changing masks and enforced every candidate's
radius. Local errors improved, but six adaptive searches returned larger,
less faithful circuits and took longer than exact ACDC. All rebuild costs and
exploratory held-out results are reported separately from the preserved
fixed-center experiments.

The [recentering audit](ACDC_RECENTERING_AUDIT.md) confirms that all 564 completed
reusable-sketch search paths used one fixed center and zero rebuilds. Adaptive
recentered directional sketching was not tested. The older explicit-quadratic
experiment did rebuild; the audit separates its distances and costs. No new
model evaluations or searches were run for this audit.

The primary objective is now **lower total discovery cost at comparable circuit
size and held-out fidelity**. Exact ACDC reproduction is a secondary diagnostic.
The [quality-first report](ACDC_QUALITY_REPORT.md) and
[preregistered protocol](ACDC_QUALITY_PLAN.md) compare 696 sketch candidates with
39 exact-ACDC candidates across both graphs and three sketch seeds. Pure threshold
and ranking searches followed their own trajectories without decision fallback.
No tested sketch achieved a comparable-quality speedup at the registered
0/5/10/25% allowances. Fresh held-out losses, matched-size/fidelity curves,
uncertainty, tuning costs and reproduction commands are saved in the report.

The preceding experiment extends the reusable Gunn sketches to **1,000 directions**
and tests a **1,281-edge graph**. See
[the results and figures](ACDC_1000_DIRECTIONS_REPORT.md) and
[methods, configurations and commands](ACDC_1000_DIRECTIONS_METHODS.md).
All 32 selected discovery runs completed. More directions reduced some
reconstruction errors, but no tested configuration produced a competitive
circuit faster than exact ACDC. The larger graph improved construction
amortization without reaching a useful accuracy/cost regime.

The preceding audit diagnoses variance and stability using the six saved Gunn
coefficient banks, with no new model evaluations. See
[ACDC_VARIANCE_REPORT.md](ACDC_VARIANCE_REPORT.md) and
[methods and reproduction](ACDC_VARIANCE_METHODS.md).
Extreme directions dominate high-degree estimates, but substantial noise remains
after trimming; robust aggregation did not qualify for further discovery runs.

The preceding experiment benchmarks reusable higher-order Gunn sketches
at degrees 2, 3, 4, 5 and 8, both expansion centers, three direction counts and
three seeds. See [ACDC_HIGHER_ORDER_REPORT.md](ACDC_HIGHER_ORDER_REPORT.md),
[methods and reproduction](ACDC_HIGHER_ORDER_METHODS.md), and the
[complete results table](results/acdc_higher_order/results_table.csv).
All 184 discovery runs completed: small cubic sketches were faster but inaccurate;
screening recovered exact ACDC with full fallback and higher total cost.
The [earlier plan](ACDC_HIGHER_ORDER_PLAN.md) and quadratic baselines are preserved.

The preceding diagnostic tests full-model centering and periodic
recentering with explicit Taylor and Gauss–Newton quadratics. See
[ACDC_RECENTERING_REPORT.md](ACDC_RECENTERING_REPORT.md) for results, plots and
reproduction, and [ACDC_RECENTERING_METHODS.md](ACDC_RECENTERING_METHODS.md) for
derivative construction, calibration, cache isolation and cost accounting.
It preserves the prior 257-edge model and squared-error task and adds no new
randomized sketches.

The preceding experiment asks whether sketch screening can accelerate ACDC while
returning the same circuit. See [ACDC_SCREENING_REPORT.md](ACDC_SCREENING_REPORT.md)
for measured results and reproduction, and
[ACDC_SCREENING_THEORY.md](ACDC_SCREENING_THEORY.md) for the conditional emulation
proof and the distinction between finite-toy certificates and empirical bounds.
It uses the entire existing 257-edge model graph, with separate squared-error
and KL comparisons. Earlier discovery algorithms and results are preserved.

This repository now implements and measures a Taylor / random complex-direction
tensor sketch for fixed-dataset edge-patching loss. Read
[SKETCH_REPORT.md](SKETCH_REPORT.md) for results and
[SKETCH_METHODS.md](SKETCH_METHODS.md) for conventions, costs and reproduction.
The sketch is constructed with truncated-series arithmetic, and compared with
exact patching and explicit Taylor tensors on enumerable toys and a restricted
pretrained Redwood transformer graph. All 38 tests pass in the saved run.

The earlier baseline pilot and its cancelled long-running job are preserved in
[REPORT.md](REPORT.md). The sections and commands below describe that older
baseline configuration; use the **sketch methods reproduction commands** for the
new experiment. No cancelled baseline job needs to be restarted.

## Reproduce on Windows / NVIDIA CUDA

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1
.venv\Scripts\python.exe scripts/download_assets.py
.venv\Scripts\python.exe -m pytest -q tests
.venv\Scripts\python.exe -m circuit_pilot.run_toys
.venv\Scripts\python.exe -m circuit_pilot.smoke
.venv\Scripts\python.exe -m circuit_pilot.run_transformer --config configs/induction.json
.venv\Scripts\python.exe -m circuit_pilot.analyze
```

The install is local to `.venv/` and `.cache/`. `requirements-lock.txt` freezes the
working environment, including PyTorch 2.7.1+cu128 and TransformerLens 2.15.4.
The CUDA 12.8 wheel supports this machine's RTX 5070 Laptop GPU. A CPU environment
can install torch 2.7.1 from the CPU index followed by `requirements.txt`; run the
toy suite and the CPU transformer tests. The full transformer command requires CUDA
unless `device` is changed in the configuration.

For one seed, add `--seeds 0`. `--timeout 120` bounds each search to two minutes and
preserves its last completed state. Use `--output results/another_run` to preserve
existing results. The default is a 900-second per-search and 6,000-second global
search budget. Deadlines are checked between exact evaluations: one evaluation
can overrun the deadline. Held-out evaluation and diagnostics finish afterward.
The three-seed configuration is a bounded pilot, not a guarantee every method will
finish on slower hardware.

## Intervention and graph

Every retained edge reads its source from the current patched run. Every excluded
edge reads the original, immutable donor trace. Every receiver recomputes. One
whole donor sequence is paired with each recipient, and a binary mask is shared
over examples and token positions. No zero ablation, weight pruning, weight
refitting, or automatic connectivity cleanup is used.

The transformer graph contains **257 selectable additive edges**: 24 token
embedding to layer-0 Q/K/V inputs, 216 token-embedding/layer-0-head to layer-1
Q/K/V inputs, and 17 embedding/head to readout inputs. Q/K/V projections,
attention, layer norms, biases, positional inputs, and unembedding are mandatory.
There are no MLPs in this model. Mandatory shortformer position embeddings are
identical for aligned recipient/donor sequences, so the empty graph reproduces
the donor logits exactly. This differs from the paper's 305-edge graph by keeping
48 direct Q/K/V computation edges mandatory. Saved graph JSON lists every edge.

`EdgePatcher` is the only binary transformer patcher. It reconstructs each
receiver as its cached donor residual plus retained current-minus-donor source
contributions. The last residual is gathered at the scored position before
unembedding, preserving full-vocabulary logits while avoiding unused positions.

## Data and upstream provenance

The exact checkpoint is `ArthurConmy/redwood_attn_2l/final.pth`, revision
`2edc292b8a1eb1104a3d847cbbcf30c4e96f7d42`. Assets are immutable, with SHA-256 hashes
in [asset-manifest.json](asset-manifest.json); upstream code commits are in
[upstream-lock.json](upstream-lock.json).

The [ACDC loader](https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/bc99ace817974b5584b7ee203d596a8e2bbcd399/acdc/induction/utils.py)
loads `validation_data.pt` and `mask_repeat_candidates.pkl`, takes shifted
next-token labels, and shuffles complete rows with `torch.randperm` to obtain
donors. We follow that stored-data procedure, without a tokenizer or regenerated
random-token sequences. We preserve permutation fixed points and record them.
This is the **ACDC shuffled-donor condition**, not a separate iid-donor experiment.

For affordable repeated exact search, this pilot uses 32-token prefixes and the
first upstream-marked position in each eligible row, with 32/128/256 disjoint
discovery/validation/test row IDs per seed. Prefix length 64 was also smoke-tested.
The [paper, Appendix J](https://arxiv.org/html/2304.14997v4#A10) describes
OpenWebText induction data and scores all marked positions in 300-token sequences.
Our prefix/position sampling is an explicit reduction, not a reproduction of that
full experiment. The downloaded mask is authoritative here; the upstream loader
does not regenerate it from raw OpenWebText. The report audits how often selected
positions have a literal repeated bigram in the retained prefix.

All examples within a pair have the same tokenization and length. Split IDs and
prediction positions are saved. Model weights are loaded strictly without
centering, folding layer norm/value biases, or refactoring. Tests compare logits
before/after enabling split hooks and after the readout optimization.

## Search and evaluation

`search.LossEvaluator` is the replaceable mask-to-loss interface. `Evaluator`
memoizes exact losses only within a run and logs all candidates, including rejected
ones. There is no cross-method loss cache. Ties use deterministic edge index,
reverse index, or a seeded random order; the numeric tolerance is saved.

* ACDC: one reverse-topological sweep; delete iff **incremental** loss increase
  is strictly below tau, following the
  [published implementation](https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/bc99ace817974b5584b7ee203d596a8e2bbcd399/acdc/TLACDCExperiment.py).
  Within a destination we use descending edge IDs. This is labeled `acdc`, never
  confused with the next method. Redundant-node removal is disabled.
* Backward: test every retained edge, choose the least harmful deletion, and
  accept only if **total** discovery loss stays within budget.
* Forward: test every missing edge and choose smallest resulting loss. Continue
  through zero or negative gains until feasible, full, or timed out.
* Doubling: restart at empty for caps 1, 2, 4, ... (last cap truncated to graph
  size), run each cap fully, and prune the first feasible candidate backward.
  Restarts reuse exact mask losses within that run; queries/cache hits are reported.
* EAP / EAP-IG5: rank by absolute signed attribution, then add ranked edges until
  the same discovery fidelity budget is met. Scoring uses next-token NLL, not
  zero-gradient clean-output KL. The
  [EAP-IG reference](https://github.com/hannamw/EAP-IG/blob/0edbdd72e3683db69c363a23deb1775f44ec8376/eap/attribute.py)
  supplies the algorithm: donor-minus-clean endpoint source differences times
  receiver gradients; sum tokens/features, mean examples. EAP evaluates gradients
  at clean embeddings; IG averages at alpha=1/5,...,1 along donor-to-clean input
  embeddings. Our adapter avoids the reference's hard-coded CUDA buffers and
  tokenizer/graph-rendering dependencies. It is tested against exact mask
  gradients (EAP) and finite receiver perturbations (IG); it is not presented as
  an unmodified upstream executable.

Transformer fidelity is mean full-vocabulary KL(p_full || p_mask), plus squared
probability distance. Task diagnostics are next-token NLL, top-1 accuracy, and
correct-token versus donor-top-incorrect-token logit difference. All final binary
circuits use the same evaluator. Tiny negative KL values near zero are float32
roundoff and are saved as measured, not interpreted as better-than-full fidelity.

The default discovery budget is 5% of discovery empty-graph KL. Validation chooses
the smallest evaluated circuit within 5% of validation empty-graph KL, using
predeclared evenly spaced trajectory checkpoints and endpoints. For ACDC it chooses
between completed sweep endpoints for tau=.01 and .1. If no circuit qualifies,
the lowest-validation-KL circuit is explicitly marked infeasible. Test results
are computed after selection is frozen and never affect it. The selected result
includes the complete cost of all candidate runs for its method. Three random
masks per matched final/selected size are evaluated without tuning them.

## Outputs and checks

`results/toys/` stores every mask, exact finite-distribution frontier, histories,
and a fidelity-versus-size/runtime figure. `results/induction/seed_*/` stores the
graph, split IDs, binary masks, accepted histories, every queried mask, gradients,
timings, computation counts, peak allocated GPU memory, held-out metrics, and
timeout/completion status. Donor caches are rebuilt reproducibly and kept in RAM;
the original downloaded tensors stay in `data/redwood/`.

Tests cover full/empty/self-donor endpoints, child isolation, immutable original
donor messages, exhaustive toy losses, zero-gain progression, strict incremental
versus total-budget acceptance, finite-difference gradients, EAP/IG attribution,
and unchanged pretrained weights. Timings synchronize CUDA; cache/model loading,
search, attribution, validation, and evaluator overhead are separated. CPU toy
milliseconds are illustrative, with cold autograd initialization reported apart.

The older baseline pilot did not include a tensor sketch. The new
`sketch_pilot/` implementation supplies the measured approximate-evaluator
comparison described above. No GPT-2 IOI experiment, reweighted compression
comparison, or population certificate is claimed.
