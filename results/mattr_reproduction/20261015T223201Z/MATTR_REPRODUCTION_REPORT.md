# MAttr reproduction on GPT-2 Small / MIB IOI

Campaign: `results/mattr_reproduction/20261015T223201Z`. Branch: `codex/mattr-reproduction`. Public test is used only for the frozen final confirmation; discovery and selection use train/validation.

## Reproduction and provenance

Attribution calls the unmodified official MAttr trainer and CLI; evaluation calls the pinned official MIB evaluator. The local adapter selects rows, records compute, and saves/restores scores, Adam state and RNGs. Official tests (6) and adapter tests (5) passed; fresh and resumed updates matched the upstream training loop bit for bit in deterministic tensor tests. The CUDA smoke trained and evaluated GPT-2 before the reference.

Pinned MAttr: `af82b2eef145dd3e66c793a413352d6931a152d7`. Exact MIB/EAP commits, package versions, lock/source hashes, commands, prompt manifests and counters are in each run directory. WSL Ubuntu / RTX 3060 12 GB uses an isolated Python 3.12 environment and the official locked lm/dev dependencies. Existing Sketch-ACDC artifacts were preserved.

Source: [official repository](https://github.com/aryamanarora/matryoshka-attribution). Published comparison: [paper, Appendix E.5 Table 4](https://arxiv.org/html/2609.25518v1) reports GPT-2/IOI node validation CPR AUC 1.83 for the 500-step lr 0.05 recipe. That is an author-reported result, distinct from our measurements. The paper also gives public-test results; they are not our validation target.

Our uncapped seed-42 reference completed 500 optimization steps on an eligible pool of 10,000 official train rows, evaluated on 10,000 validation rows. Official CPR AUC **1.8320**, CMD AUC **0.9731**, accuracy AUC **0.4106**. Difference from the rounded reported 1.83: **+0.0020**. This reproduces the authors’ validation result to its reported two-decimal precision; the rounded target cannot establish agreement beyond that precision.

The reference actually sampled 489 unique training rows in 500 presentations. Its eligible pool size is not its number of unique rows visited by optimization.

Clean/corrupted mean logit differences: 3.2427 / -2.7259. Attribution: 37.6s; official validation evaluation: 2021.5s. Peak allocated CUDA memory: 4.30 GiB; peak reserved CUDA memory: 9.50 GiB; sampled CPU RSS peak: 3.76 GiB.

The same reference ranking scores 1.8408 CPR AUC on the fixed 128-prompt comparison cohort. This is a separate cohort result and is not substituted for the uncapped reference.

## Metric definitions and comparison protocol

`area_under` integrates normalized logit-difference faithfulness over the official linear retained-fraction grid; the official CLI calls it CPR AUC. At each size the implementation uses (mean circuit − mean corrupted)/(mean clean − mean corrupted). `area_from_1` integrates absolute deviation from one (official CMD AUC). Accuracy AUC uses positive logit difference over log sparsity. These three quantities are distinct. CPR can exceed one through behavioral overshoot; larger CPR alone does not prove a more faithful mechanism. CMD closer to zero means less deviation from full-model behavior.

The ten retained fractions are 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 and 1.0, applied to the same 157-node basis (input, 144 heads, 12 MLPs). Requested counts are 0,0,0,1,3,7,15,31,78,157; post-pruning retained counts are recorded separately for each graph. Weighted edge counts are never labeled as node counts.

The reference follows the uncapped documented command. All scaling and baseline comparisons use exactly one fixed 128-prompt validation cohort (seed 42), the same nested discovery pools within seeds 42/43/44, node granularity, signed ranking and counterfactual patching. The reference ranking has a separate reevaluation on that cohort (pending until its artifact exists). No custom IOI data or homemade leaderboard metric is used.

Pool size is the number of eligible unique examples; MAttr samples with replacement for 500 one-example updates. Actually visited unique rows and total presentations are recorded. EAP sees the pool once; EAP-IG uses five official integration steps. Both baselines use attribution batch size 1: the official mean-loss implementation accumulates per-batch averages, so a smaller final batch would otherwise receive different per-example weight. Evaluation batch size stays 20 for every compared method. The official attribution functions and loss are unchanged. These are equal eligible-data comparisons, **not compute-matched** experiments. Cost-quality plots expose their different budgets; no new compute-matched tuning was performed.

## Discovery-data size and baselines

| Method | Pool | Runs | Validation CPR mean ± SD | CMD | Accuracy AUC | Construction s | Train−validation CPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| EAP | 16 | 3 | 0.2500 ± 0.0000 | 0.7490 | 0.1017 | 23.3 | 0.0000 |
| EAP | 32 | 3 | 0.2500 ± 0.0000 | 0.7490 | 0.1017 | 26.1 | -0.0000 |
| EAP | 64 | 3 | 0.2500 ± 0.0000 | 0.7490 | 0.1017 | 31.1 | 0.0000 |
| EAP | 128 | 3 | 0.2500 ± 0.0000 | 0.7490 | 0.1017 | 46.4 | -0.0000 |
| EAP-IG | 16 | 3 | 1.5952 ± 0.0348 | 0.7880 | 0.3863 | 27.7 | 0.0082 |
| EAP-IG | 32 | 3 | 1.5941 ± 0.0391 | 0.7829 | 0.3910 | 40.9 | 0.0815 |
| EAP-IG | 64 | 3 | 1.5876 ± 0.0147 | 0.7750 | 0.3926 | 60.1 | 0.0625 |
| EAP-IG | 128 | 3 | 1.5804 ± 0.0038 | 0.7731 | 0.3884 | 110.0 | 0.0548 |
| MAttr | 16 | 3 | 1.8397 ± 0.0283 | 0.9790 | 0.4173 | 51.1 | 0.0804 |
| MAttr | 32 | 3 | 1.8409 ± 0.0101 | 0.9932 | 0.4120 | 52.3 | 0.1345 |
| MAttr | 64 | 3 | 1.8392 ± 0.0102 | 0.9859 | 0.4146 | 54.8 | 0.1267 |
| MAttr | 128 | 3 | 1.8530 ± 0.0039 | 0.9964 | 0.4139 | 54.3 | 0.1095 |

For completed MAttr runs, validation CPR changed from 1.8397 at 16 examples to 1.8530 at 128 examples (+0.0133). Seed variation measures optimization/subset differences on the same validation cohort; it does not establish generalization to other tasks or models.
Across-seed SD was 0.0283 at the smallest pool and 0.0039 at the largest. With only three seeds, these are descriptive variability estimates.

16-row pool: 16–16 unique rows visited; 32-row pool: 32–32 unique rows visited; 64-row pool: 64–64 unique rows visited; 128-row pool: 125–128 unique rows visited. Actual presentations per MAttr run: 500–500.

Exploratory selection among the final 500-step MAttr configurations uses mean validation CPR: pool 128. All raw runs remain saved. Public-test confirmation uses this same pool for all methods and the three preset seeds.

MAttr has higher mean validation CPR than EAP at 4/4 pool sizes; mean paired difference over pool sizes is +1.5932. This comparison uses each method’s prescribed budget, not matched compute.
It has lower mean CMD than EAP at 0/4 pool sizes.
It has higher mean official accuracy AUC than EAP at 4/4 pool sizes.

MAttr has higher mean validation CPR than EAP-IG at 4/4 pool sizes; mean paired difference over pool sizes is +0.2539. This comparison uses each method’s prescribed budget, not matched compute.
It has lower mean CMD than EAP-IG at 0/4 pool sizes.
It has higher mean official accuracy AUC than EAP-IG at 4/4 pool sizes.

MAttr therefore wins the CPR comparison against EAP-IG while deviating more from full-model behavior by CMD at every pool size. This is a metric tradeoff, not an unqualified fidelity win; the circuit-size curves show the overshoot that contributes to the CPR score.

At the 128-row pool, construction plus the identical validation protocol takes 81.8s for MAttr versus 141.8s for EAP-IG (means over three seeds). These are measured costs of the tested recipes, not results from tuning both methods to a common time budget.

Observed input-node ranking ranges (out of 157): EAP 134–153; EAP-IG 1–1; MAttr 4–8.
In 12/12 EAP runs, the input ranks below the 78 nodes requested at 50% retention. This is an observed limitation of the signed node-ranking protocol and is consistent with sparse circuits losing clean input information. An input-held-clean control was not run, so it does not prove the cause or account for the MAttr–EAP-IG difference. No input score was forced or adjusted.

MAttr mean train−validation CPR gaps range from +0.0804 to +0.1345 across pool sizes.
The smallest pool does not show a larger positive train−validation gap than the 128-example pool; these measurements do not support a simple monotonic small-data overfitting claim.

EAP: mean attribution 12.47s, total construction 31.71s, fixed-validation evaluation 28.47s; attribution uses 180.0 forward batches and 60.0 backward calls on average.

EAP-IG: mean attribution 40.12s, total construction 59.69s, fixed-validation evaluation 28.05s; attribution uses 420.0 forward batches and 300.0 backward calls on average.

MAttr: mean attribution 29.02s, total construction 53.13s, fixed-validation evaluation 28.14s; attribution uses 559.6 forward batches and 500.0 backward calls on average.

## Generalization and computational cost

Train-subset and validation quality use the same official circuit sweep. A positive train−validation gap is consistent with overfitting but can also reflect cohort variation and normalization by different clean/corrupted reference scores. Loss is noisy because sparsity and prompts vary every update; it is not a held-out-quality metric. Checkpoints at 100, 250 and 500 steps are evaluated without changing the optimizer trajectory.

Root-model forwards are counted at actual execution, including counterfactual donor forwards. Cached donors do not increment forward counts. Backward calls and Adam updates are separate counters. Construction cost includes interpreter/import startup, data/model preparation, all mask optimization, checkpoint overhead and ranking; circuit evaluation is separate. The total-compute plot adds the same fixed-validation evaluation to construction cost. Extra train-quality and checkpoint diagnostics are recorded separately, rather than charging their unequal query counts to that comparison. All actual work remains in the stage/process accounting tables. The final checkpoint’s identical validation query is reused after score/cohort/configuration checks and adds zero forwards. CUDA stages synchronize before timing and report allocator peaks; CPU RSS is sampled. Timers use monotonic perf_counter rather than console calendar timestamps. Timing includes instrumentation and cold-start overhead, so it is not an author-hardware speed comparison.

Across completed model jobs, instrumented stages total 115.6 minutes, 43,809 forward batches, 556,545 example-forwards and 10,825 backward calls. The largest accumulated stage is `validation_evaluation`: 62.9 minutes (54.4% of stage time). This includes reference and diagnostic sweeps as well as the comparison jobs; unit tests, notebook rendering and analysis are outside these model-stage counters.

Artifacts: `training_results.csv`, `dataset_scaling.csv`, `mattr_vs_eap.csv`, `quality_vs_compute.csv`, `training_curves.csv`, `checkpoint_quality.csv`, `compute_accounting.csv`, per-run rankings/tensors/manifests/checkpoints/raw per-example evaluations, logs and PNG figures.

## Frozen public-test confirmation

The pinned [MIB README](https://github.com/aryamanarora/MIB-circuit-track/blob/582fdbc8a2b0f7063a265c8f94851fd2a3aa5480/README.md#evaluation) explicitly permits local public-test evaluation using `--split test`. After all validation experiments finish, `confirmation_selection.json` freezes the discovery pool, all nine source-ranking hashes, seeds, checkpoint, and a new deterministic 128-example test cohort before test data is loaded. The common pool maximizes mean MAttr validation CPR over seeds 42/43/44; all three methods use that pool. Test results cannot change this selection.

| Method | Test runs | Public-test CPR mean ± SD | CMD mean | Accuracy AUC mean |
|---|---:|---:|---:|---:|
| EAP | 3 | 0.2500 ± 0.0000 | 0.7490 | 0.0650 |
| EAP-IG | 3 | 1.4839 ± 0.0038 | 0.6826 | 0.3701 |
| MAttr | 3 | 1.7561 ± 0.0035 | 0.9004 | 0.3937 |

Public-test minus validation mean CPR at the frozen pool: EAP +0.0000; EAP-IG -0.0964; MAttr -0.0969. This cohort change also changes prompt difficulty and normalization; it is not a pure estimate of optimization overfitting.

These are local measurements on one fixed public-test subset, not a private leaderboard submission or a full-public-test reproduction. Error bars describe ranking/subset-seed variability on that same cohort, not population confidence intervals. The official evaluator supplies all reported metrics.

## Research decision

Most promising next hypothesis: validate a shorter fixed optimization budget. The 250-step checkpoint is within 0.02 mean CPR AUC of the best measured step when averaging these configurations; this exploratory observation needs an independent confirmation cohort. Repeated full validation sweeps are costly, so an online early-stopping scheme would have to include their overhead before claiming a saving.

No proposed algorithmic improvement has been implemented. Equal-compute claims require separate predeclared budgets. Optional interaction analysis remains gated and cannot establish explicit Hessian learning.

**Execution status:** complete requested reference, scaling, baseline, checkpoint and frozen public-test measurements.
