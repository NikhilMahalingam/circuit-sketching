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
