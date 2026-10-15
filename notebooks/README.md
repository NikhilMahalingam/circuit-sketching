# Circuit discovery notebook

`mib_circuit_discovery.ipynb` is a controlled head-level follow-up to
`mib_ioi_experiments.ipynb`. It imports the official MIB data/model through
`mib_ioi/runtime.py` and reuses the completed IOI run's 12 EAP-ranked heads,
train singleton effects, and train pair effects. Clone the pinned official MIB
repository as described in `../mib_ioi/README.md` before running it in a fresh
checkout.

Run cells 1–4 first to verify official data, intervention reproduction, and
train-only coefficients. Cell 5 evaluates EAP, first-order, and second-order
selected circuits on train and validation; it took under a minute of model
forwards on the RTX 3060 after model/data startup. Cell 6 is the exhaustive
oracle: `RUN_ORACLE = False` by default. At 12 heads and sizes 2, 4, 6, 8,
and 10 it requires 2,046 candidate circuits, about 32,736 train forward
batches plus donor passes; enable only after inspecting the printed estimate.
The remaining cells produce comparisons, cost accounting, and results-only
conclusions. The saved executed notebook ran with the oracle disabled.

For headless execution from the repository root:

```powershell
.\.venv\Scripts\python.exe run_mib_circuit_discovery.py
```

If the Jupyter kernel cannot start in the current shell, use `--direct` to
execute the same Python cells sequentially and save their outputs:

```powershell
.\.venv\Scripts\python.exe run_mib_circuit_discovery.py --direct
```

To deliberately run the exact oracle, add `--oracle` and optionally set
`--max-oracle-candidates-per-size`. The cap rejects a circuit size entirely
when the full combination count exceeds it. Validate a completed run with
`review_checks/validate_mib_discovery_results.py RESULTS_DIRECTORY`.

## Sketch-ACDC diagnostics

`mib_sketch_diagnostics.ipynb` reuses the official IOI loader and the corrected
shared edge adapter. Its default execution is an explicitly labeled lightweight
smoke test. The full wrap-up runs the same utilities through a checkpointed CLI:

```powershell
.\.venv\Scripts\python.exe scripts\run_sketch_diagnostics_final.py --sizes 16 32 64 128
```

Keep the printed result directory. Resume a single discovery size with
`--out <directory> --sizes 128`. Thresholds and queries are chosen on training
prompts; all sizes share the same 128 validation prompts. After the primary
runs, `scripts/build_sketch_final_report.py <directory>` writes the report and
`seed_extension_selection.json`. Run the selected ranks with
`--out <directory> --sizes 128 --seeds 43 44 --ranks <selected ranks>`, then rebuild
the report and run `scripts/verify_sketch_final_results.py <directory>`.

The notebook's final dashboard displays completed campaign results without
launching another model run. Full numerical artifacts and the research decision
are linked from [the final report](../SKETCH_ACDC_FINAL_REPORT.md). Earlier
executed notebooks and frozen-sender results remain unchanged; the report
explains their limitations.
