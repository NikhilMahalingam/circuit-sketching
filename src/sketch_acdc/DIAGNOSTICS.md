# Sketch diagnostics

Open `notebooks/mib_sketch_diagnostics.ipynb` in the existing project kernel, or
run `.\.venv\Scripts\python.exe scripts\run_sketch_diagnostics.py` from the
repository root. No dependencies are installed or upstream repositories changed.

The default smoke loads 16 official train prompts and 32 validation prompts,
then computes on two prompts per split, one partial ACDC state, two candidates,
and rank 4. `MIB_DIAGNOSTICS_FULL_HESSIAN=1` optionally exercises the full 16-by-16
Hessian on that tiny prompt/query set. Outputs are timestamped under
`results/mib_sketch_diagnostics/`.

For the larger diagnostic run, set `MIB_DIAGNOSTICS_FULL_RUN=1`. This uses all
16/32 prompts, several saved partially pruned states, every intact candidate,
ranks 4/8/12/16, and shared anchors 0 and 0.5. The full Hessian still requires
the separate `MIB_DIAGNOSTICS_FULL_HESSIAN=1` flag. Both flags can be set in the
first notebook cell instead of the environment. Read the printed forward/HVP
estimates before launching a larger run.

## Intervention audit finding

The historical version of `core.EdgeExperiment` captured clean sender outputs in a separate
forward pass. Official MIB evaluation subtracts current sender outputs during
the intervened pass. These definitions can differ for dependent edge removals.
The shared `core.EdgeExperiment` now captures differentiable current senders in
the intervened forward. `DiagnosticEdgeExperiment` inherits that corrected
implementation and adds persistent caching. `FrozenSenderEdgeExperiment` keeps
the old behavior solely for historical audits. No previous result is rewritten.

The smoke audit found a maximum historical-versus-official discrepancy of
0.411971 logit-difference units on a dependent six-edge stress mask. The new
adapter matched the independent official evaluator on baseline, single-edge,
partial-state, and dependent stress masks. This stress test does not establish
that the discrepancy caused the earlier pilot's sketch underperformance.

## Reading the diagnostics

Same-train effects distinguish Taylor truncation from prompt shift. With the
optional full Hessian, signed sketch error equals full-Taylor error plus
compression error, and held-out error adds train-to-validation prompt shift.
Absolute errors do not add: their signs can cancel.

Thresholds near exact effects are selected on training data only. Difficult
validation decisions are a retrospective first-order-disagreement subset,
shared by all methods. If the existing edges have no effects near 0.01–0.30,
the notebook reports insufficient threshold coverage rather than inventing
hard cases. A higher-rank trend need not be monotonic for indefinite Nyström
sketches or binary Taylor predictions.

`method_costs.csv` allocates the common gradient to each estimator; the raw
ledger counts each real forward/backward/HVP once. Exact intervention labels
are included as an oracle with their measured shared train+validation query
cost, not as a learned predictor. No cost-to-accuracy plot establishes an ACDC
speedup by itself.

## Final campaign

`scripts/run_sketch_diagnostics_final.py --sizes 16 32 64 128` runs the official
multi-edge correctness gate, then a full 16-example diagnostic followed by the
larger discovery sizes against a fixed 128-prompt validation set. All subsets
are nested within each train seed; thresholds and query selection are frozen
using train-16 seed 42 only. `--out <directory> --sizes 32` resumes one size.
Full Hessians and nested rank-4/8/12/16 direct-HVP sketches are computed one
batch at a time. Context bundles checkpoint intermediate work. The full
Hessian is an analysis reference, never used to manufacture sketch HVPs.

`scripts/build_sketch_final_report.py <directory>` aggregates completed jobs,
records incomplete configurations, renders measured figures, and writes
`SKETCH_ACDC_FINAL_REPORT.md`. The training-only seed-extension selection and
the independent Stage G–I report sources are recorded separately.
