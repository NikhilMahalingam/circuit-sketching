# Gate 5G addendum: Stage 1 analysis plan

This is post hoc and exploratory: Gate 5G outcomes have already been observed.
Commit this plan before running the new analyses. Preserve original protocol
c051cf92d775c92693205b22d07684b911268b96e41d99e777efc1018e8c3462 and decisions
`EAP-IG closer` / `Compact gain with CPR cost`. Zero model forwards, backwards or
optimizer updates are permitted in Stage 1. Subsequent additions must be labeled
post-plan exploratory.

Inputs: existing Gate5G frozen protocol, ranking catalog, original Gate5F full-data
signed EAP scores, canonical edge mapping, EAP order, splice exports, saved official
masks, queries, nominal/per-prompt results, completion markers, shared bootstrap
indices, structural features, final audit and offline-repair records. Verify hashes,
original implementation/dependencies, cohort, canonical grid and masks before use.
No measurements may be substituted if source artifacts are unavailable.

Prespecified analyses:

1. Inspect pinned attribution/metric source for sign semantics and record line
   references. Count positive/zero/negative original EAP scores in ranks1..324,
   1..1624 and all32491. Record first nonpositive rank and prefix score distributions.
   Separately count incidence with the pinned negative-name-mover class and the
   negative-attribution proportion among those edges. Never interpret MAttr signs
   as excitation/suppression or equate head-class incidence with causal function.
2. Read official MIB normalization and trapezoidal CPR/CMD quadrature. Decompose
   each of20 parent/splice comparisons into signed/absolute weighted grid-point
   contributions. Reproduce unrounded official differences at rtol1e-9 (atol1e-10
   for zero differences); fail beyond justified tolerance. Bands: <=1%,2–5%,10–50%,
   100%. Verify full-model endpoint contribution zero. Save per-seed, equal-seed
   means, signed accounting shares and paired5000-draw template-bootstrap band CIs
   from the exact saved Gate5G indices. Signed shares are not probabilities and may
   be negative or exceed100%; unstable near-zero totals get no share.
3. For every splice/grid point, reconstruct unpruned top-n sets using frozen signed
   order. Inventory additions/displacements and assert equal cardinality. Record
   changed fraction, parent overlap with EAP prefix, actual counts and differences
   in official pruned masks. At10/20/50%, characterize original EAP signs/score
   quantiles, parent MAttr ranks, known-head classes, negative-head incidence,
   source/destination layers and receiver components. Save identities separately.
4. At10/20/50%, report per-prompt splice-minus-parent margins and KL: mean, median,
   q05/q25/q75/q95, fraction with smaller margin or lower KL, template summaries,
   Spearman association and top-ceil(10%N) negative-loss concentration. Define net
   margin change near zero as abs(mean)<1e-8; then do not divide. Save negative mass
   separately. Normalized mean-margin/faithfulness relations are algebra, not an
   independent finding.
5. Diagnose baseline full-data seed42 at capped116/296 using saved outputs: IOI
   sign accuracy, vocabulary top1 accuracy, wrong-name/other-token failure counts,
   exact-template failure distribution, known-head coverage, newly represented
   components/edges, EAP overlap, graph paths and pruning/count differences. These
   descriptors do not identify a missing edge's causal role.

Outputs: input integrity/source-line audit; sign tables and histogram/rank plot;
CPR/CMD grid/band/pooled-CI tables and contribution figures; pre/post-pruning edge
inventories and composition figures; margin/KL distributions and template tables;
seed42 diagnostics; CPU accounting; GATE5G_ADDENDUM_REPORT.md. Report only measured
values and mechanical comparisons. If top324 has no negative EAP scores, include
the task's exact required statement ruling out direct negative-prefix attribution,
without ruling out suppressor-associated components or negative causal effects.
Leave the researcher-written interpretation section empty. Stage2 results must be
marked not run. No new attribution, training, adaptive search or confirmation runs.
