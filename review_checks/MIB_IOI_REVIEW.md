# IOI notebook implementation review

The source notebook is `notebooks/mib_ioi_experiments.ipynb` (39 cells, 19 code cells).
It imports experiment helpers from `mib_ioi/` and official MIB/EAP source from
the ignored local checkout at `external/MIB-circuit-track`.

## Repository and API evidence

- MIB revision: `b759df34433c9e31043ba9e02908ce0bf20e894f`.
- EAP revision: `7af394a5662de8b23ad6154716a0cd3993d447a3`.
- Inspected README, attribution/evaluation scripts, model/task mappings, dataset
  filtering and IOI item construction, metrics, tokenization, graph hooks,
  node attribution, exact graph evaluation, and official curve integration.
- Official identifiers: `gpt2` -> `gpt2-small`, `ioi` -> `mib-bench/ioi`.
- Official splits downloaded: 10,000 train, 10,000 validation, 1,000 public test.
  The notebook uses the official loader/filter, then selects seeded subsets.
- The actual node attribution API accepts uppercase `EAP`.
- Upstream `Graph.to_json` raises an out-of-range logits-node score error for
  node scores. The notebook uses official `to_pt`/`from_pt` with a tensor round-trip
  assertion. Reloading also avoids unsupported deep-copying of graph node sets.
  No upstream source was modified.

## Correctness checks

- The metric and last-unpadded-token positioning come directly from MIB/EAP.
- Every selected name label is verified to be a single token matching MIB's IDs.
- Whole filtered train/validation clean-prompt sets are checked for overlap.
- Counterfactual token lengths and attention masks must align with clean inputs.
- Head replacement validates `[batch, position, head, d_model]`, clones the input,
  and edits only requested slices. The original activation tensor is preserved.
- A fresh intervention check tests hook removal, determinism, batched versus
  per-example outputs, official baseline agreement, and exact graph-patching agreement.
- Pair features use only `i < j`; mask pools are unique across mask splits and
  cover five intervention densities. Quadratic training designs must have full rank.
- Learned fits use only training masks and official training prompts. Same-split
  and cross-prompt-split held-out metrics are labeled separately. Exact singleton
  and pair diagnostics are explicitly distinguished from learned-fit targets.
- Bootstrap resampling is across matched examples, jointly across pairs/masks.
  Exploratory signed pair tests use BY multiple-testing correction. Effect-size
  and descriptive ranking criteria are specified before execution.
- All inference uses no-grad; only official EAP enables gradients. Parameter
  gradients are cleared between attribution batches and after attribution.
- Model-level instrumentation counts real forward batches and example-forwards,
  including calls inside MIB/EAP. CUDA synchronization brackets timed calls/stages.
- Detached CPU donor slices and per-example scalar results are cached. Exact-score
  disk caches include data/code/model/component provenance in their identity.
- Official evaluation is called unchanged; raw returned fields are preserved.
  Local subset evaluation is not represented as private leaderboard performance.

## Validation

Five focused invariant tests pass in `test_mib_ioi.py`: head isolation/dimensions,
disjoint multi-density designs for 12 and 16 heads, recovery of known quadratic
interactions, example-level bootstrap behavior, and compilation/schema checks for
every notebook code cell.

A 16-train/16-validation integration run executed all 19 code cells on the RTX
3060. Its final rerun reused cached exact effects; its measured forward count
therefore describes only the resumed work. Independently recomputed held-out
metrics and regression coefficients passed, as did the official evaluator output
checks. This smoke run is separately labeled in its executed notebook.

The default run's results and independent validation are recorded separately
in its timestamped results directory and executed notebook.

## Preserved baseline files

SHA-256 before and after implementation:

- `mib_sketching_baseline.ipynb`:
  `CD0C18A72EB9865B442DD0A71BAEC191F6509E00F5A5C47E42C0483D4F6D20E4`
- `mib_sketching_baseline.executed.ipynb`:
  `6CD96360114C07E429071070ECEC7C9F547BA0D16CE16800042B125B148DAFD8`

Other pre-existing untracked experiment files were not changed.
