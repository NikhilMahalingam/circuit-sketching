# Circuit sketching

The [Sketch-ACDC research wrap-up](SKETCH_ACDC_FINAL_REPORT.md) records corrected
MIB edge semantics, the nested discovery-data diagnostics, measured compute,
and the research recommendation. The shared edge adapter now uses current
sender activations during simultaneous patching. Earlier frozen-sender edge
results are preserved and explicitly qualified in that report; earlier
head-level experiments used a separate adapter.

The follow-up [circuit discovery notebook](notebooks/mib_circuit_discovery.ipynb)
and its [run guide](notebooks/README.md) compare EAP, first-order, and
second-order selection on official MIB IOI examples. Exact combinatorial
search is available but disabled by default.

An exploratory GPT-2 attention-head intervention notebook comparing first-order
regression, explicit second-order regression, and a toy directional sketch on
held-out intervention masks.

## Setup

The pinned environment uses **Python 3.11** and **CUDA 12.6 PyTorch**. The full
notebook was validated on an NVIDIA GeForce RTX 3060. These Windows PowerShell
commands create an isolated project environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name mib-sketching --display-name "Python (mib-sketching CUDA)"
.\.venv\Scripts\python.exe -m jupyterlab
```

Open `notebooks/mib_sketching_baseline.ipynb` and select **Python (mib-sketching CUDA)**.
Use the project interpreter explicitly so that a different system Python does
not select a CPU-only PyTorch installation. The first run downloads GPT-2 weights
from Hugging Face.

## Execute and validate

For a complete run without opening JupyterLab:

```powershell
.\.venv\Scripts\python.exe run_notebook.py
.\.venv\Scripts\python.exe review_checks\validate_cuda_results.py
```

The runner requires CUDA and the registered kernel. It saves an executed copy
to `notebooks/mib_sketching_baseline.executed.ipynb`, exports tables and plots to a new
directory under `results/`, and records execution details in `review_checks/`.
The validator reads those local outputs, recomputes held-out metrics, and checks
forward-pass accounting and train/test separation. Run it after the runner.

The clean source notebooks, completed executed notebooks, and published result
tables are tracked in Git. Environments, downloaded models, failed runs, and
unpublished result directories remain local.

## Review checks

These checks exercise notebook definitions without downloading GPT-2 weights:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s review_checks -p test_notebook_dataflow.py -v
.\.venv\Scripts\python.exe review_checks\test_hook_contracts.py
```

The tokenizer provenance check downloads the canonical GPT-2 vocabulary and
merges:

```powershell
.\.venv\Scripts\python.exe review_checks\verify_gpt2_candidates.py
```

See [the review report](review_checks/REVIEW.md) for the ten correctness checks
and validation evidence. On October 14, 2026, all 23 code cells executed on the
RTX 3060 without errors; 218 real forwards matched an independent count,
held-out metrics were recomputed, and train/test overlap was zero. Peak allocated
GPU memory was approximately 647 MiB. Outputs referenced by the report are
generated locally and are not included in the repository.

`build_notebook.py` regenerates the clean source notebook and overwrites that
file. Keep notebook content changes in the generator if using it to rebuild.

## Matryoshka Attribution reproduction

[mib_mattr_reproduction.ipynb](notebooks/mib_mattr_reproduction.ipynb) runs and
analyzes the official GPT-2 Small / MIB IOI node-attribution recipe, discovery
data-size experiments, and matched-data EAP/EAP-IG comparisons. Model execution
uses a separate **WSL Ubuntu / Python 3.12** environment with the official
dependency lock. Follow the [MAttr run guide](scripts/mattr/README.md), rather
than the Windows environment instructions above.

The [reproduction report](MATTR_REPRODUCTION_REPORT.md) records actual results,
metric definitions, compute accounting, and limitations. Timestamped outputs
are saved under `results/mattr_reproduction/`. Previous Sketch-ACDC notebooks
and reports remain preserved.

The [executed analysis notebook](notebooks/mib_mattr_reproduction.executed.ipynb)
and [saved campaign](results/mattr_reproduction/20261015T223201Z/) include the
rankings, checkpoints, raw evaluations, figures, and verification evidence.

## Prefix-cached ACDC

[acdc_prefix_caching.ipynb](notebooks/acdc_prefix_caching.ipynb) compares full
forwards with dependency-checked cached-prefix execution on official MIB IOI.
The [implementation guide](src/acdc_prefix_cache/README.md) documents the
independent stages and WSL commands. The [report](ACDC_PREFIX_CACHE_REPORT.md)
records passing Stage A correctness tests and planning estimates; performance
Stages B and C have not run.

The [executed analysis notebook](notebooks/acdc_prefix_caching.executed.ipynb)
reads the [saved Stage A campaign](results/acdc_prefix_cache/20261019T173713Z/).
All experiment flags remain disabled by default. The independent reference's
2.35× speedup has not been reproduced or claimed by this campaign.
