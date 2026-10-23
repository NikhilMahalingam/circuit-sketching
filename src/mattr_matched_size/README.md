# Gate 5E: frozen-ranking matched-size evaluation

Inference only. No training, model fine-tuning, score optimization or new
differentiable masking. The only derived rankings are the prespecified
first-appearance closure and equally weighted three-seed ordinal-rank ensembles.
Gate 5D implementation and its mixed decision remain unchanged.

Use the existing pinned WSL environment:

```bash
bash src/mattr_matched_size/run_wsl.sh tests
bash src/mattr_matched_size/run_wsl.sh --stage topology
bash src/mattr_matched_size/run_wsl.sh --stage prepare
bash src/mattr_matched_size/run_wsl.sh --stage inference
bash src/mattr_matched_size/run_wsl.sh --stage report
bash src/mattr_matched_size/run_wsl.sh --stage audit
```

Preparation requires the completed CPU topology audit and freezes the complete
protocol before fresh-cohort forwards. Inference runs correctness checks and the
first complete ranking, then stops if measured projected workload exceeds two
hours. Completed/partial attempts are never silently overwritten. Batch20 and
the official patcher, signed sorter, pruning and MIB integrals remain unchanged.

Exact output caching stores only immutable per-prompt margins/top1, keyed by
mask, model, source protocol and cohort. No activations are cached. Cache hits
execute no forwards. All original and derived rankings use the same new512
official validation prompts selected with5543, excluding prior small campaigns,
Gate5D and training using both clean/donor hashes and canonical row IDs.

Cap queries use the largest audited nominal prefix with A(n)<=T. Exact matches
are a labeled subset, never separate evaluations. Closure is evaluated only on
the nominal official grid, with original/closure actual counts side by side;
no matched-size closure evaluations are allowed. The appended closure tail is
not claimed pruning-free. Ensembles reuse3training runs and are not cheaper
discovery methods.

Raw prompts, tensor checkpoints, packed circuit sets, model data and caches stay
local. Lossless compressed CSV exports of the 18 derived signed-score rankings
are versioned with tensor/export hashes. Version small tables, protocol/hashes,
plots, tests, report and executed notebook.
The notebook defaults to saved-result analysis with all expensive flags off.

The completed campaign is `results/mattr_matched_size/20261023_gate5e/`.
Its `interpretation_generator.py` adds descriptive interpretation after the
registered report and audit; run it from the repository root. It adds no
evaluation, selection rule, or hypothesis test. `export_derived_rankings.py`
produces lossless CSV exports from the frozen local tensors. Supplementary
report/export script hashes are in `post_execution_provenance.json`, separate
from the immutable pre-evaluation protocol. Replaying the registered report
replaces the descriptive overlay; rerun the interpretation helper afterward.
