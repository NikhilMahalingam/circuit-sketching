# Gate 6A prior-work audit

## Publication check

Arora et al. v1 Appendix G.1, Table 8 includes KL and CMD-shaped training, scored by CPR and Compactness on node-level SVA+ tasks. The IOI cell uses Qwen-2.5. Figure 2 and Appendix E.3–E.4 report the official edge benchmark for the headline logit-difference training and other optimizer/mask/schedule variants. I found no official edge-level MIB CMD result for faithfulness-trained MAttr in v1. The document's prior-work stop condition does not apply. KL and the CMD-shaped loss are prior work. [MAttr paper v1](https://arxiv.org/html/2609.25518v1#A7.SS1)

## Training objective and targets

At pinned MAttr `af82b2eef145dd3e66c793a413352d6931a152d7`, let d be the base-token minus source-token margin, q the intervened distribution, t the base label for iso and source label for cause, and x = d for iso and −d for cause. All listed losses average their per-example values over the batch.

| Upstream loss | Per-example formula |
| --- | --- |
| logit_diff | −x |
| ce | −log q[t] |
| logit | −logit[t] |
| prob | −q[t] |
| hinge | max(0, hinge_margin − x), default margin 2 |
| acc | sigmoid(−x / acc_temp), default temperature 1 |
| ld_tanh | tanh(−x / ld_scale), default scale 2 |
| kl | sum over the full vocabulary of p_clean × (log p_clean − log q), iso only |
| cmd | abs(1 − (d − d_empty)/g), g = max(1, d_clean − d_empty) for nonnegative gaps and min(−1, d_clean − d_empty) otherwise, iso only |

The shared helper accepts targets as supplied. The SVA caller computes and caches clean margins, all-patched margins and clean log probabilities under no-grad. SVA exposes `--loss`, `--hinge-margin`, `--acc-temp` and `--ld-scale`. The MIB node script calls the helper with a fixed logit-difference loss. The MIB edge MAttr script exposes no loss flag and computes negative IOI margin directly. The separate edge-pruning trainer exposes `--loss kl|logit_diff` and uses a different training algorithm. [Loss source](https://github.com/aryamanarora/matryoshka-attribution/blob/af82b2eef145dd3e66c793a413352d6931a152d7/src/matryoshka_attribution/losses.py#L22-L103) · [SVA CLI](https://github.com/aryamanarora/matryoshka-attribution/blob/af82b2eef145dd3e66c793a413352d6931a152d7/scripts/sva/eval_sva.py#L794-L800) · [SVA targets](https://github.com/aryamanarora/matryoshka-attribution/blob/af82b2eef145dd3e66c793a413352d6931a152d7/scripts/sva/eval_sva.py#L1030-L1097) · [Edge objective](https://github.com/aryamanarora/matryoshka-attribution/blob/af82b2eef145dd3e66c793a413352d6931a152d7/scripts/mib/eval_mib_edge.py#L349-L365)

The proposed squared margin mismatch with a fixed discovery-pool denominator has no equivalent in this helper. R0 therefore calls the upstream CMD loss unchanged. The gap floor can make R0's loss nonzero even at the clean margin. K0/K1 can reuse the upstream KL helper with detached float32 log probabilities. The margin objective is a soft-mask surrogate for discrete-circuit CMD.

## Training streams

The edge closure draws one example with replacement per call using Python `random.randint`, independently of the Torch uniform-k draw. `--batch-size 20` controls evaluation. The official trainer supports a sampler hook, starts scores at zero and uses Adam for the registered recipe. Gate 5F freezes the separate row and k streams and rejects length skips. Drawing 5,000 times with replacement from 10,000 rows has expected unique coverage 10,000 × (1 − (1 − 1/10,000)^5,000) ≈ 3,935. Gate 5F observed 3,890–3,932. [Edge sampling source](https://github.com/aryamanarora/matryoshka-attribution/blob/af82b2eef145dd3e66c793a413352d6931a152d7/scripts/mib/eval_mib_edge.py#L179-L209) · [Trainer](https://github.com/aryamanarora/matryoshka-attribution/blob/af82b2eef145dd3e66c793a413352d6931a152d7/src/matryoshka_attribution/trainer.py#L35-L129) · [Gate 5F stream construction](src/mattr_data_scaling/design.py#L30-L51)

## Official evaluation

At MIB fork `582fdbc8a2b0f7063a265c8f94851fd2a3aa5480` and EAP-IG `3fb341eafdaba63632954aecbbb96b4b08e13d02`, the evaluator uses cohort means of clean, empty-patched and circuit IOI margins at the official last-token position. It selects `int(p × number_of_edges)` scores and prunes disconnected structure. At each grid point, F = (mean circuit − mean empty)/(mean clean − mean empty). CPR applies linear trapezoidal quadrature to F and CMD applies it to the pointwise values |1−F| on the same ten-point grid, with no 0% integration endpoint or extra area normalization. Accuracy is the fraction of positive IOI margins, with log-retention trapezoidal integration normalized by the log span. [Evaluator](https://github.com/aryamanarora/MIB-circuit-track/blob/582fdbc8a2b0f7063a265c8f94851fd2a3aa5480/MIB_circuit_track/evaluation.py#L32-L140) · [IOI metric](https://github.com/aryamanarora/MIB-circuit-track/blob/582fdbc8a2b0f7063a265c8f94851fd2a3aa5480/MIB_circuit_track/metrics.py#L138)

The MIB paper recommends signed scores for CPR and absolute scores for CMD. The pinned evaluator uses one ordering per invocation for both integrals. Its Python API defaults to absolute, its CLI defaults to signed unless `--absolute` is supplied, and the official MAttr edge script explicitly uses signed ordering for both. The registered EAP-IG reference therefore prefers absolute in a numerical tie. The original leaderboard submission configuration was outside this audit. The pinned edge selector uses the default descending `torch.argsort` on the flattened official score matrix, without a stability flag. Canonical exports must use that exact CPU sort path to preserve historical ties before assigning ordinal scores. [MIB paper](https://proceedings.mlr.press/v267/mueller25a.html) · [MIB CLI](https://github.com/aryamanarora/MIB-circuit-track/blob/582fdbc8a2b0f7063a265c8f94851fd2a3aa5480/run_evaluation.py#L20-L26) · [MAttr scoring call](https://github.com/aryamanarora/matryoshka-attribution/blob/af82b2eef145dd3e66c793a413352d6931a152d7/scripts/mib/eval_mib_edge.py#L383-L389)
