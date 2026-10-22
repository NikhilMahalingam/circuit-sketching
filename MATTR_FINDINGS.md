# MAttr edge optimization: implementation findings (Gates 0–1)

The supplied task is the complete plan; the user confirmed there is no separate agent runbook. The initial request authorized Gate 0 only; the subsequent request explicitly authorized Gate 1. Both have now completed. Earlier node-MAttr results, Sketch-ACDC and prefix-caching artifacts are preserved, including previously modified local prefix-caching results.

## Pinned implementation and environment

MAttr is pinned to `af82b2eef145dd3e66c793a413352d6931a152d7`; its required MIB fork is pinned to `582fdbc8a2b0f7063a265c8f94851fd2a3aa5480`. The fork's evaluator returns seven values; the older Sketch-ACDC MIB checkout is not interchangeable. See [deps.py:21](external/matryoshka-attribution/src/matryoshka_attribution/deps.py#L21). Exact EAP submodule revision, package versions, executable, raw source hashes and the uv lock hash are saved per campaign.

A new WSL environment, `~/.venvs/mattr-optimization-af82b2e`, uses the official frozen `uv.lock` and default `lm`/`dev` groups. Offline setup uses already cached dependencies. It does not sync or install into the older environments. Source verification compares against the pinned Git blobs, allowing only CRLF/LF differences without rewriting the checkout. The environment's editable upstream package remains pinned and source guarded.

## Scores, masks, objective and sampling

| Question | Actual implementation | Source |
|---|---|---|
| Initialization | One zero-initialized floating score per real edge; 32,491 for GPT-2 Small. | `trainer.py:74`; `eval_mib_edge.py:132–157` |
| Update | Adam on scores, lr set by CLI. Shared trainer defaults: eps=1e-8, betas=(0.9,0.999), no clipping, one k draw per step, no LR schedule. The edge CLI does not expose `--adam-eps`, so it inherits that default. | `trainer.py:35–57,74–90,122,172`; `eval_mib_edge.py:359–364` |
| Soft top-k | Bisection finds tau so the sum of sigmoid((score−tau)/T) is approximately k. Backward implicitly differentiates tau, coupling score gradients. It is not an independent sigmoid or hard top-k. | `sigmoid_topk.py:7–66`; `masks.py:48–55` |
| Defaults for the requested recipe | T=.5; 30 bisection iterations from the edge CLI (the shared trainer default is 50); uniform k; iso; Adam lr=.05; 5,000 requested reference steps. | `eval_mib_edge.py:52–85`; `README.md:87–92` |
| k | The shared trainer samples a floating k uniformly in [1,total) using Torch RNG; log and log_both also exist. The edge script's local Python-RNG `sample_k` definition is unused by the shared training call. | `schedules.py:13–35`; `trainer.py:101–108`; `eval_mib_edge.py:46–49,359–364` |
| Examples | One official training example drawn with replacement per optimizer iteration; unequal clean/donor token lengths return None and skip the update. The CLI batch size controls evaluation, not training. | `eval_mib_edge.py:178–216,383`; `trainer.py:111–113,127–128` |
| Objective | Iso keeps the selected soft edge contributions clean, replaces the complement with counterfactual contributions, and minimizes the negative correct-minus-incorrect final-position logit difference. No task-example normalization or CPR-grid weighting occurs in the baseline loss. | `eval_mib_edge.py:249–255,349–355` |
| Patching | Fixed corrupted donor outputs minus live CURRENT clean/intervened sender outputs. Differentiable expansion from real-edge score vector into adjacency; per-destination stacks use activation checkpointing. It does not use the historical frozen-clean-sender patcher. | `eval_mib_edge.py:249–347` |
| Donor cache | Device-resident per-example corrupted activations with an explicit CLI memory budget, default 4 GB. Cache hits skip real donor forwards. Gate-0 smoke explicitly uses 1 GB; this is a resource setting, not the full reference configuration. | `eval_mib_edge.py:84–86,196–247` |

The adapter calls the upstream CLI and trainer; it does not copy the edge intervention implementation. Hard-mask checks compare the upstream closure against official MIB on the same already-pruned graph, including states with many dependent deletions. These do not prove a separate soft-mask oracle or universal correctness on every graph.

## Official IOI data and metrics

The upstream edge command uses `gpt2` → TransformerLens `gpt2-small`, dataset `mib-bench/ioi`, explicit official `train` and `validation` splits. HFEAPDataset filters compatible answer token lengths and returns clean `prompt`, default `s2_io_flip_counterfactual.prompt`, and token IDs from metadata's indirect object and subject. It indexes the first token; the local smoke additionally verifies the selected names are single-token answers and fails rather than substituting a homemade metric. See `MIB_circuit_track/dataset.py:40–97,118–125,173–183` and `eval_mib_edge.py:30–43,149–152,379–390`.

The exact default retained-fraction grid is:

`0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0`.

For 32,491 edges the requested counts are `32,64,162,324,649,1624,3249,6498,16245,32491`. Official connectivity pruning may retain fewer edges. Weighted edge counts returned by MIB are saved separately; they must not be mislabeled as requested edge counts.

At each fraction, official normalized faithfulness is:

`(mean circuit logit difference − mean corrupted reference) / (mean clean reference − mean corrupted reference)`.

Normalization is a ratio of dataset means, **not** a mean of per-example ratios. The corrupted reference is the officially evaluated empty graph. There is no clipping to [0,1]; faithfulness and CPR can exceed 1. A small denominator or behavioral overshoot can inflate CPR, so improvement in CPR alone does not establish improved mechanistic fidelity. Source: `MIB_circuit_track/evaluation.py:35–43,98–109`.

CPR (`area_under`) is the linear-x trapezoidal integral of faithfulness by default. CMD (`area_from_1`) uses trapezoids of the pointwise absolute distance from one. The corresponding CPR point weights are:

`0.0005, 0.002, 0.004, 0.0075, 0.02, 0.04, 0.075, 0.2, 0.4, 0.25`.

They sum to **0.999**, because integration starts at .001. There is no added zero endpoint and no division by .999. A constant-one curve therefore gives CPR=.999 and CMD=0. Endpoints are measured, not forced; a one-point custom grid returns NaN integrals. Accuracy AUC is a separate normalized log-x integral. Source: `evaluation.py:61–69,111–143`. Unit tests call this actual evaluator with artificial input curves and independently calculate trapezoids; unit numbers are never reported as model measurements.

Official edge evaluation calls `apply_topn(..., absolute=False, level='edge', prune=True)` at each cutoff: signed scores define a single underlying nested ranking, and connectivity cleanup is then applied separately at each budget. The evaluated surviving graph is not guaranteed to have exactly the requested size. Source: `evaluation.py:75–82`; `EAP-IG/src/eap/graph.py:440–531`; `eval_mib_edge.py:374–390`. A final method must retain one ranking; no validation-driven independent circuit selection is introduced.

## Existing related experiments and novelty boundaries

The inspected [paper v1, Appendix D.2 and Table 4](https://arxiv.org/html/2609.25518v1) already compares uniform/log schedules, hard forward masks, detached thresholds, straight-through estimators, optimizers and extra training. Its Table 4 reports GPT-2/IOI **edge** validation CPR **10.59** for the 5,000-step headline recipe; this is an author-reported target, not our measurement. It is distinct from the earlier 1.83 node result.

The repository additionally implements `FixedK` and `AdaptiveLogK` (`schedules.py:38–106`) and exposes a `k_sampler` hook (`trainer.py:101–105`). AdaptiveLogK tracks an accuracy frontier, rather than the proposed CPR quadrature weights. Thus “adaptive sampling” or “hard top-k” alone is not a novelty claim. The inspected paper/repository did not establish a published test of this task's exact CPR-grid mixture, EAP-IG rank initialization, or trained-ranking boundary swaps. This is a bounded source-inspection finding, not a literature-wide novelty claim. Source scripts: `scripts/mib/launch/submit_mib_edge_ablations.sh:14–35`, `scripts/mib/launch/submit_edge_lr05.sh:15–36`; `src/matryoshka_attribution/masks.py:65–81`.

Future A should compare CPR-grid weighting with existing schedule controls, tracking small circuits separately. Future B requires an explicit initialization adapter because the current shared trainer always creates zeros; EAP-IG cost must be charged. Future C1 must distinguish post-training continuation from already-tested hard-mask training. No A/B/C algorithm modification is implemented or executed at Gate 0. Example-normalization changes have no established equivalence to the ratio-of-means evaluator and remain disabled.

The observed upstream edge smoke also accumulated model-weight gradients, even though Adam updates only edge scores. This behavior was preserved for the baseline. Freezing model weights could reduce unnecessary gradient storage and computation while retaining derivatives through edge masks, but requires an explicit gradient/update equivalence test before becoming a separately documented memory optimization. No such optimization is enabled in Gate 0.

## Gate status

The full seed-42, 5,000-step official edge baseline completed in `results/mattr_optimization/20261021T134352Z_gate1/`: validation CPR 10.58801242, which rounds to the paper's reported 10.59, and CMD 9.60080915. Both official train and validation splits contain 10,000 rows in their original order. Actual sampled train rows: 3,925 unique in 5,000 presentations. Training took 26.94 minutes and full validation 42.27 minutes; allocated GPU memory peaked at 6.799 GiB. These are baseline reproduction measurements, not improvements. See `MATTR_BASELINE_REPORT.md` and the raw campaign for the exact configuration, counters and limits.

The prior Gate-0 campaign retains its own report and executed notebook. Edge discovery-data scaling, noise-based success thresholds, variant comparisons and submissions remain pending. All notebook execution switches default off. No automatic next-gate dispatch occurs.
