# Research log: sketched circuit discovery

A running, chronological record of what was done, what came out, how it was read, and what was decided next. Each stage's registered plan and full report hold the details; this log ties them together. The README lists the same reports newest first.

## Current state (updated 2026-10-04)

- **Goal.** Use sketches to cut the cost of ACDC-style circuit discovery without losing circuit quality. Quality first; then scale and time savings. Algorithms need not be ACDC-like or one edge at a time.
- **What works on cost.** Exact-checked large-group deletion in a cheap attribution-prior order. It costs about 0.2× ACDC's compute at 7,588 edges and about 0.06× at 32,491 edges (GPT-2 small).
- **What doesn't yet work.** On GPT-2 small, against a strong ACDC reference, these circuits are 1.2–1.3× worse than ACDC's on held-out data, and only 10/22 runs met the 10% allowance. Quality is now the binding constraint.
- **The sketch's current role.**
  - An output-space Gauss–Newton sketch composes deletion groups and reliably cuts exact checks (18–33% fewer than contiguous groups).
  - Bank costs have so far offset the saved checks: a tie at best (groups up to 1,024 on GPT-2).
  - On GPT-2 the sketch arms' held-out quality was slightly better than contiguous groups' (median 0.92–0.94× for the 1,024 arms). Stage G found the opposite, so this is not established.
- **Stage I result.** On GPT-2 small, 64-pair contiguous group search matched the 16-pair ACDC's held-out quality (0.98×) at about a sixth of its compute. ACDC's reverse-topological order made group search much worse. The sketch added nothing with a good prior order: equal quality, 1.2× the cost.
- **Paused (2026-10-07), resuming automatically at 18:00 local.** Stage J runs ACDC on the same 64 pairs (the equal-data question; no sketch arm). Its gate passed, and set 0's ACDC is checkpointed at 27,000 of 32,491 steps.
- **Pending.** GitHub push needs an empty `sketched-circuit-discovery` repo URL, or the `gh` CLI installed and logged in.

## Conventions used in every stage

- **Edges and deletion.** Components are attention heads and (from Stage B on) MLPs. An edge carries one component's output into another's input. Deleting an edge replaces what it carries with the activation from a corrupted "donor" prompt.
- **Loss.** Squared distance between the patched and full model's full-vocabulary output probabilities, averaged over prompt pairs. The full model has loss 0; the empty circuit (every edge deleted) is the ceiling.
- **ACDC reference.** Visit edges in reverse topological order; delete an edge if the exact loss increase is below τ. Its final size is k.
- **Exact check.** One patched forward pass over the discovery pairs. Every method here accepts a deletion only after an exact check.
- **Data roles.** Discovery pairs drive the search. A validation set picks each run's operating point, and those picks are frozen before a fresh holdout is opened. Holdout losses are the quality measure.
- **"Meets the 10% allowance".** The validation-selected circuit has at most 1.1k edges and held-out loss at most 1.1× ACDC's.
- **Cost.** Complete time, including preparation, the prior and sketch banks, in units of that run's own exact-check time, relative to ACDC. This normalization corrects for GPU power capping, which varied between runs.

## Log

### Before 2026-10-01: input-direction sketches (inherited work)

- **Tried.** Reusable Taylor sketches of the loss along random input directions (Gunn), at many degrees, direction counts (up to 1,000), centers, recentering and adaptive rebuilds. They were used to screen, rank or batch ACDC deletions on 257- and 1,281-edge attention-only graphs.
- **Found.** No tested sketch gave a comparable-quality speedup. Bank construction often cost more than a full ACDC search. Exact-checked random batching was the one practical speedup (about 41–45% at 1,281 edges).
- **Read as.** The sketch was retired from the proposal-ranking role ([ACDC_RANKING_VALUE_REPORT.md](ACDC_RANKING_VALUE_REPORT.md)). Details are in the README's older entries.

### 2026-10-01: handoff and packaging

- Read the handoff and packaged the repo into git (`1bb2245`). `.npz` arrays are untracked but checksummed in `results/npz_sha256sums.txt`, and `.gitattributes` keeps files byte-exact so hash manifests stay valid.
- James proposed scaling up; the agreed path was to treat batching as adaptive group testing first.

### Stage A, 2026-10-01: group testing (no sketch)

- **Question.** Can ordering deletion groups by a cheap prior, and lifting the group-size cap, make exact-checked batching faster and more reliable?
- **Ran.** 25 searches on the 1,281-edge graph ([plan](ACDC_GROUP_TESTING_PLAN.md), [report](ACDC_GROUP_TESTING_REPORT.md)).
- **Found.**
  - A 1-second gate-path integrated-gradients prior made cap-4 batching meet the allowance in 5/5 seeds, against 2/5 in topological order, at about 45% less time than ACDC.
  - One deterministic uncapped run needed 56 checks instead of 1,282.
  - Uncapped random groups collapsed, because the allowance τ·|G| grows with the group and becomes vacuous.
- **Decision.** James stressed that the project is about sketching. Sketches stay central from here on; any sketch-free control must be labeled as such.

### Stage B, 2026-10-02: the sketch-ranked hybrid at 7,588 edges

- **Question.** Does sketch-ranked, exact-checked batching keep its quality on a 6× larger graph with MLPs (SoLU-6L, docstring task)?
- **Ran.** 4 arms × 3 seeds ([plan](ACDC_MLP_SCALE_PLAN.md), [report](ACDC_MLP_SCALE_REPORT.md)). The τ fallback triggered, so τ = 0.0003 and k = 785.
- **Found.**
  - The hybrid in prior order met the allowance 3/3 (0.95× ACDC), the best quality of any arm, and beat same-seed random proposals in 5/6 pairs.
  - About 440 bank rebuilds cost roughly one ACDC sweep, so the hybrid cost 1.15–1.6× ACDC. Batching without a sketch cost 0.35–0.51×.
- **Read as.** The input sketch's estimates are too noisy for sparse queries. Its error scales with the whole derivative tensor, not with the few edges in a query.

### 2026-10-02: output-space sketch diagnostic

- **Question.** Can a different sketch rank sparse deletions well at the same cost?
- **Method.** A Gaussian sketch S of the loss residual under the Gauss–Newton model: ℓ(c+h) ≈ (1/k)‖S r_c + S J_c h‖². Its relative variance is 2/k for every query h. It is built with 1 forward and k backward passes ([report](ACDC_OUTPUT_SKETCH_REPORT.md)).
- **Found.** On 18 frozen pools, regret fell 225× (257 edges) and 37× (1,281 edges), and Spearman rose to 0.60 and 0.72. The remaining error is Gauss–Newton truncation, not sketch noise.
- **Decision.** Use the output sketch in full searches.

### Stage C, 2026-10-02: output-sketch searches at 1,281 edges

- **Ran.** Exact-checked searches with output-sketch group proposals ([plan](ACDC_OUTPUT_SEARCH_PLAN.md), [report](ACDC_OUTPUT_SEARCH_REPORT.md)).
- **Found.**
  - At radius 64 in prior order: meets 5/5, held-out 0.87–0.90× ACDC at 157 edges, 0.83–0.86× ACDC's compute.
  - It used 10–14% fewer checks than random proposals, the first sketch ever to cut checks.
  - The quality gain over prior-ordered random proposals is within noise (median 0.96), and random is cheaper (0.57×).
- **Decision.** Use the sketch to compose large groups under a non-diluting allowance τ·min(|G|, 4).

### Stage D, 2026-10-02: sketch-composed large groups at 1,281 edges

- **Ran.** Groups up to 16 or 64 edges ([plan](ACDC_LARGE_GROUPS_PLAN.md), [report](ACDC_LARGE_GROUPS_REPORT.md)).
- **Found.**
  - Sketch composition cut checks 44–47% against cap-4 random batching, at about 0.54× ACDC.
  - Every run met the allowance.
  - The no-sketch control, contiguous groups in prior order, was cheaper (0.47×) and better (0.88× ACDC at 157 edges), but it was a single deterministic run.
- **Hypothesis.** The 4 original discovery pairs may favor some methods by chance.

### Stage E, 2026-10-02/03: discovery-set size and resampling (1,281 edges)

- **Ran.** Three fresh discovery sets of 4 and 16 pairs, from a new 3,000-prompt docstring pool made with ACDC's own generator ([plan](ACDC_DISCOVERY_SIZE_PLAN.md), [report](ACDC_DISCOVERY_SIZE_REPORT.md)).
- **Found.**
  - 16 pairs lowered held-out loss for every method in every set (median about 39%), and brought batch methods within about ±15% of ACDC.
  - Sketch versus contiguous flipped direction between sets.
  - On fresh 4-pair sets, batch methods were 1.2–1.6× worse than ACDC in two of three sets, so earlier wins on the original 4 pairs do not generalize.
- **Read as.** Discovery data matters more than the search method. Later stages use 16 pairs.

### Stage F, 2026-10-03: large groups at 7,588 edges (4 original pairs)

- **Ran.** Groups up to 256 edges with the sketch's radius 512 ([plan](ACDC_SOLU_GROUPS_PLAN.md), [report](ACDC_SOLU_GROUPS_REPORT.md)).
- **Found.** The sketch arm met the allowance 3/3, at 0.925× ACDC's held-out loss, with 868 checks against ACDC's 7,589, about 0.20× ACDC's compute (3.6× faster measured). It tied contiguous groups on cost and matched or beat the controls on quality.
- **Caveat.** It used the 4 original pairs, which Stage E had just shown to be unreliable.

### Stage G, 2026-10-03: float32, 16 pairs, 7,588 edges

- **Question.** Does Stage F's frontier result survive adequate discovery data?
- **Precision.** James approved float32: float64 at 16 pairs cost 1.2 s per check and ran out of GPU memory. A registered gate passed exactly: float32 ACDC matched float64 decision for decision.
- **Ran.** Two fresh 16-pair sets, 14 runs, fresh 128-pair holdout ([plan](ACDC_FP32_SCALE_PLAN.md), [report](ACDC_FP32_SCALE_REPORT.md); `09ceed7`).
- **Found.**
  - The sketch arm met the allowance 6/6 at 0.20× ACDC with 86% fewer checks than ACDC and 28% fewer than contiguous groups.
  - Contiguous groups (no sketch) had lower held-out loss in all 6 paired comparisons (median 1.13×, 3 clearly) at the same cost (0.21×).
- **Read as.** Stage F's result does not survive. The sketch saves checks, not quality, and bank costs offset the saved checks at this size.
- **Decision.** Scale to GPT-2 small, where saved checks grow with the graph while bank cost does not.

### Stage H, 2026-10-03: GPT-2 small, IOI (in progress)

- **Feasibility.** Measured before registration on prompts 900–915 of the IOI file, which are excluded from the study:
  - 32,491 edges; clean top-1 accuracy 15/16; donors never predict the answer; empty-circuit loss 0.394.
  - Float32 exact check 0.286 s at 16 pairs, so one ACDC sweep is about 2.6 h. Float64 costs 0.77 s per check.
  - Float32 matches float64 to 1.2×10⁻⁶ in loss.
  - Prior 6.0 s (peak 4.3 GiB); one output-sketch bank 8.0 s (about 28 checks; peak 4.6 GiB).
- **Data.** The standard single-template IOI file from auto-circuit (`ioi_vanilla_template_prompts.json`, 1,000 unique prompts, all 14 tokens, ABC donors).
- **Composer speed.** The sketch's group composer took 0.20 s per 1,024-edge group, about 0.7 of an exact check, which would have unfairly inflated the sketch arms' cost. A rewrite (`acdc_gpt2/search.py`) makes the same choices in 0.04 s. Tests show it reproduces the registered search's trace exactly, including forced ties.
- **Smoke finding.**
  - **Observation.** Early in the prior order, contiguous groups of 1,024 deleted 3,000 edges with 4 exact checks. Over the same stretch, the sketch arms spent 17–49 s building banks (`greedy_G256` built 6, about 170 checks' worth). Contiguous groups then cleared 12,000 edges with 52 checks.
  - **Read as.** Most of the graph is easy to delete, and there an always-on sketch is pure overhead.
  - **Response.** Added an exploratory `lazy_G1024` arm. It uses contiguous groups first and calls the sketch only for retries after a rejection, so banks are built only where cheap guesses fail.
- **Design.** Registered 2026-10-04 02:02 UTC ([plan](ACDC_GPT2_PLAN.md), results in `results/acdc_gpt2_20261003/`).
  - **Gate and data.** A float32 precision gate, then two 16-pair discovery sets (prompts 0–15 and 16–31), validation on 32–47 and a 128-pair holdout on 48–175.
  - **Threshold.** τ = 0.0003, falling back to 0.0001 if block 0's ACDC circuit keeps more than half the empty-circuit validation loss.
  - **Arms.**
    - Contiguous groups up to 256 and 1,024 (no sketch).
    - Sketch-composed groups up to 256 and 1,024 (3 seeds each).
    - `lazy_G1024` (3 seeds).
  - **Question.** Primarily, whether any sketch arm costs less than its contiguous control at ACDC's circuit size, and at what quality.
- **Compute.** The projection is 10–16 h at full power. The cutoff is 2026-10-05 02:02 UTC.
- **Precision gate passed (2026-10-04 02:13 UTC).** Float32 losses matched float64 to 1.7×10⁻⁶ over 20 random masks, and the prior ranking agreed (Spearman 1.000). The first 2,000 ACDC decisions were identical in both precisions (1,909 deletions each). The search started immediately, beginning with block 0's ACDC reference.
- **Completed (2026-10-04, [report](ACDC_GPT2_REPORT.md)).** All 24 runs and the holdout finished with no failures. Heavy GPU power capping stretched the search to 20.5 h; costs are per-check normalized.
  - **ACDC references.** 686 and 717 edges, at 21% and 27% of the empty-circuit held-out loss. These are much stronger references than on SoLU.
  - **Cost.** Every batch arm cost 0.04–0.13× ACDC at k, with about 1,100–2,300 checks against 32,492.
  - **Quality.** Batch circuits had 1.2–1.3× ACDC's held-out loss at k. 10/22 runs met the allowance: 1/11 on set 0 and 9/11 on set 1.
  - **Sketch.**
    - `greedy_G1024` tied `contig_G1024` on cost (0.99×), used 18% fewer checks, and had median loss 0.94× (1 pair clearly lower, 1 clearly higher).
    - `lazy_G1024` had median loss 0.92× (5/6 lower, none clearly higher) at 1.18× cost, with 8 banks against 16.
    - `greedy_G256` cost 1.6× contiguous groups (64 banks).
    - Contiguous caps of 256 and 1,024 gave identical circuits.
  - **Read as.**
    - Scale makes exact-checked batching dramatically cheaper (0.2× at 7,588 edges, 0.06× at 32,491). But quality against a strong reference is now the binding constraint, and it depends heavily on the discovery set.
    - The sketch's cost case is still not established: it ties at best.
    - The quality direction reversed from Stage G, with sketch arms at or below contiguous groups. The effects are small and possibly set-dependent.
  - **Open.** Why batch circuits trail ACDC here. The untested candidates are the prior order (instead of reverse topological) and the per-group allowance of up to 4τ.

### Stage I, 2026-10-04 to 2026-10-06: the GPT-2 quality gap

- **Diagnosis from Stage H's saved losses (no new compute).** At ACDC's size, batch circuits fit their 16 discovery pairs as well as ACDC's did:

  | Block | Contiguous groups, discovery | ACDC, discovery | Contiguous groups, held-out | ACDC, held-out |
  | --- | --- | --- | --- | --- |
  | Set 0 | 0.0097 | 0.0102 | 0.104 | 0.066 |
  | Set 1 | 0.0123 | 0.0121 | 0.090 | 0.087 |

  Every circuit fits 16 pairs about 7–10× more tightly than it generalizes.
- **Read as.** The gap is generalization, not search quality. The stricter-allowance idea was dropped, because the circuits already fit their discovery data as tightly as ACDC's.
- **Decision (James: "proceed").** Test two explanations, reusing Stage H's ACDC references:
  - **More discovery data (64 pairs).** Batch search is cheap enough to afford it.
  - **ACDC's reverse-topological order.** It may pick edges that generalize better.

  Sketch arms run in each condition. The plan is [ACDC_GPT2_GAP_PLAN.md](ACDC_GPT2_GAP_PLAN.md).
- **Feasibility (excluded prompts 900–963).** At 64 pairs, the loss, prior and sketch are computed in exact chunks of 16 pairs, because whole-batch gradients don't fit in GPU memory.
  - A 64-pair check costs 1.145 s, 3.98× a 16-pair check. A bank takes 31 s.
  - A 64-pair batch search should cost about 0.25× a 16-pair ACDC.
- **Smoke run.** Clearing the first 3,000 edges in reverse-topological order took contiguous groups about 400 checks, against 5–39 in prior order. Sketch-composed groups in that order needed about 100, so the sketch matters much more there.
- **Registered 2026-10-05 00:53 UTC** (`results/acdc_gpt2_gap_20261004/`). Runs take place in this order:
  1. contiguous groups, 64 pairs;
  2. reverse-topological contiguous groups and sketch-composed groups (3 seeds), 16 pairs;
  3. sketch-composed groups, 64 pairs (3 seeds);
  4. reverse-topological contiguous groups, 64 pairs;
  5. lazy sketch, 64 pairs (3 seeds).

  All of Stage H's runs are re-scored on a fresh 128-pair holdout (prompts 176–303). The projection is 13–20 h at full power, and the cutoff is 2026-10-06 12:51 UTC.
- **Paused at James's request (2026-10-05 13:49 UTC).** 13 of 24 runs are complete (an earlier version of this entry said 22). The GPU was heavily power-capped: a 64-pair check took about 3.4 s, against 1.15 s at full power.
  - **How.** The process was stopped rather than suspended, so no paused time enters any run's timing.
  - **Lost.** The interrupted run, `d1_m16/greedy_G1024_m64_s1`, had been going for about 12 minutes and will rerun from scratch on resume.
  - **Record.** Details are in `results/acdc_gpt2_gap_20261004/pause.json`.
- **Resumed (2026-10-05 22:40 UTC).**
  - **Amendment 1.** The cutoff and hard deadline were extended by the paused interval plus 5 minutes (516 minutes), so the registered compute budget is unchanged. The new cutoff is 2026-10-06 21:27 UTC. Methods, arms, seeds, data and holdout are unchanged, and no selection had been frozen nor the holdout opened.
  - **Restart.** The relaunch skipped the 13 completed runs and restarted at the interrupted one. The first session's console log is preserved as `console_part1.log`.
- **Completed (2026-10-06, [report](ACDC_GPT2_GAP_REPORT.md)).** 23 of 24 runs completed. The last lazy run hit the cutoff after renewed power capping.
  - **Data (H1): supported.** With 64 discovery pairs, contiguous groups reached the 16-pair ACDC's held-out quality (0.98× at k, median) at 0.15–0.18× ACDC's compute. Every 64-pair arm improved on its 16-pair version (median 0.78–0.86×). Set 1 met the allowance in all 6 prior-order 64-pair runs; set 0 in none of 7 (1.05–1.19×).
  - **Order (H2): rejected.** Reverse-topological order was much worse: 1.33–1.47× held-out, worse discovery fit, about twice the checks.
  - **Sketch.** At 64 pairs in prior order it matched contiguous groups' quality (1.05×) at 1.2× the cost, with no check savings. It helped only in the bad order (0.76× checks, 0.89× cost). Stage H's slight sketch quality edge shrank to 0.99× on the fresh holdout.
  - **Read as.**
    - The GPT-2 gap was mostly overfitting to 16 pairs, and cheap group search can afford more data. This is a roughly matched-compute comparison: a 64-pair ACDC would likely improve too.
    - ACDC's order isn't its advantage. One-edge-at-a-time decisions may be.
    - With a good attribution prior, the sketch adds nothing to group composition. Its value showed up only where the order was poor.

### Stage J, 2026-10-06: the equal-data question (paused)

- **Decision (James: "settle the equal data question").** Run ACDC itself on Stage I's 64-pair sets, so that group search and ACDC are compared on the same data. This stage contains no sketch arm.
- **A faster ACDC with identical decisions.**
  - **Idea.** In reverse topological order, while ACDC tests edges into block L, every lower block computes exactly what it does under the full mask. Each check can start at block L from one cached full-mask forward.
  - **Exactness.**
    - Losses are bitwise identical to standard ACDC's: tiny-model tests, and 0/120 structured checks on real GPT-2.
    - Replaying the first 4,000 steps of Stage H's real ACDC gave identical decisions, including across an interrupt and resume.
  - **Speed.** 2.35× faster.
  - **Read as.** The standard ACDC used as the cost baseline in Stages H and I was 2.35× more expensive than it needs to be. Group search's reported cost advantage over ACDC shrinks by that factor. From now on, costs are reported against both implementations.
  - **Checkpointing.** ACDC checkpoints every 1,000 steps, so a pause loses little.
- **Registered 2026-10-07 02:24 UTC** ([plan](ACDC_GPT2_EQUAL_PLAN.md), `results/acdc_gpt2_equal_20261006/`).
  - **Gate.** The fast ACDC replays Stage H's whole block-0 ACDC decision for decision.
  - **Runs.** ACDC on 64 pairs, set 0 then set 1. Contiguous group search on the same 64 pairs at that ACDC's stopping budgets.
  - **Holdout.** Everything from Stages H–J is re-scored on a fresh holdout (prompts 400–527).
  - **Compute.** About 11 h at full power. The cutoff is 2026-10-08 18:21 UTC.
- **Gate passed (2026-10-07).** The prefix-cached ACDC replayed all 32,491 of Stage H's block-0 decisions with 0 unequal losses and the identical 686-edge circuit. It took 2.5 h under power capping.
- **Paused at James's request (2026-10-07 13:54 UTC).** Set 0's 64-pair ACDC had checkpointed 27,000 of 32,491 steps. About 7 minutes of work since that checkpoint is lost, and the run resumes from step 27,000 with identical decisions. Set 0's contiguous run, all of set 1 and the holdout have not started. The record is in `results/acdc_gpt2_equal_20261006/pause.json`.
- **Resume scheduled (James's request): 2026-10-07 18:00 local, 22:00 UTC.**
  - **How.** A one-time Windows scheduled task, `SketchedCircuitDiscovery_StageJ_Resume`, runs `scripts/resume_stage_j.cmd`.
  - **Amendment 1.** It extends the cutoff by the scheduled pause plus 5 minutes (491 minutes), to 2026-10-09 02:32 UTC, so the registered compute budget is unchanged. Nothing else changes.

## Cross-stage conclusions so far

- **Discovery data.** Its size and choice move quality more than any search-method difference (Stages E–I). On GPT-2, 64 pairs closed most of the gap to a 16-pair ACDC.
- **Exact-checked large groups.** In prior order, they are the strongest practical method on cost: comparable to ACDC at about a fifth of its compute on 7,588 edges, and about 6% of its compute on GPT-2 small. On GPT-2, though, they are 20–30% worse than a strong ACDC reference.
- **Input-direction sketches.** They fail for sparse queries.
- **The output-space sketch.** It ranks and composes groups well enough to cut exact checks by 28–47%. Its quality effect has flipped between stages (worse than contiguous groups in Stage G, slightly better in Stage H, equal in Stage I), and its bank cost has so far offset the saved checks. With a good attribution prior it saves no checks at all (Stage I, 64 pairs). It helps only when the order is poor.

## Operational notes

- **Windows CUDA.** It spills to system RAM unless `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7,max_split_size_mb:256` is set with a memory-fraction cap.
- **CPU complex kernels.** Multi-threaded batched complex kernels give wrong jets on CPU (torch 2.7.1). Use 1 thread or CUDA.
- **Long runs.** The GPU is power-capped intermittently (24 W against 55–70 W), so costs are normalized per check. Long runs are launched detached, so they outlive the 2-hour tool limit.
- **Folder rename.** The local folder was renamed to `sketched-circuit-discovery` on 2026-10-02, and the `.venv` was repointed. Run tests with `.venv/Scripts/python.exe -m pytest`.
