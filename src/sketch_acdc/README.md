# Restricted Sketch-ACDC experiment

Run `notebooks/mib_sketch_acdc.ipynb` from the repository root with the existing CUDA environment. The notebook defaults to **Stage A only**. To investigate H1, set `RUN_STAGE_B = True`; Stage C additionally requires `RUN_STAGE_C = True`. Stage B may require thousands of forward batches and HVPs on the 16-edge graph, so inspect its printed estimate before execution.

The edge adapter imports the official MIB EAP `Graph`, `make_hooks_and_matrices`, `get_scores_eap`, metric, dataset, and `evaluate_graph`. It implements a continuous scaling of MIB's corrupted-minus-clean edge update. Its binary endpoints were checked numerically against the official evaluator, including a 10-edge terminal state. The sequential pruning traversal and strict absolute-effect threshold follow the inspected original ACDC code, but only 16 candidates in an otherwise complete graph can be removed. This is **not** full ACDC or an official MIB circuit score.

`core.py` contains the intervention adapter, exact restricted pruning, HVP, and randomized Nyström sketch. `research.py` contains opt-in H1/H2 collection, train-only empirical calibration, and hybrid pruning. Validation prompts are held out for assessment. The empirical residual bound is not a formal confidence interval. All accuracy or speed claims require the Stage B/C runs; Stage A only establishes mechanics.

Run numerical unit tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_sketch_acdc.py -v
```

Run the default CUDA Stage A and save an executed notebook under a timestamped result directory with:

```powershell
.\.venv\Scripts\python.exe scripts\run_sketch_acdc_stage_a.py
```

The measured B/C pilot uses 16 official train prompts, 8 validation prompts,
16 EAP-selected edges, one threshold (0.05), and sketch ranks 2 and 4. It runs
all three stages with:

```powershell
$env:SKETCH_ACDC_PROFILE='pilot'
$env:SKETCH_ACDC_RUN_B='1'
$env:SKETCH_ACDC_RUN_C='1'
.\.venv\Scripts\python.exe scripts\run_sketch_acdc_stage_a.py
```

The pilot's original per-method times exclude H1 data collection. Run
`scripts/reconcile_sketch_acdc_costs.py <result-directory>` to add the measured
H1 cost as separate columns and regenerate cost plots. The columns retain both
algorithm-only and end-to-end research costs. The pilot is a functional test,
not enough evidence to claim a general pruning speedup.
