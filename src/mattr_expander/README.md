# Gate5A: expander-coupled MAttr

This module uses the existing pinned WSL runtime and official GPT-2 Small/IOI
edge CLI. Only the score vector passed to the official mask builder changes.
All 32,491 real transformer edges remain eligible. Raw scores initialize at zero;
Adam, uniform-k sampling, implicit sigmoid top-k, model-weight gradient behavior,
precision, edge patches and official MIB evaluation are unchanged.

```bash
bash src/mattr_expander/run_wsl.sh tests
OUT="results/mattr_expander/20261022_gate5a"
bash src/mattr_expander/run_wsl.sh --out "$OUT" --prepare-only
bash src/mattr_expander/run_wsl.sh --out "$OUT"
```

The first preparation freezes graph algorithms before generating graph spectra.
It then freezes the complete protocol/source/mapping/adjacency hashes before
any modified CPR evaluation. Graphs are degree8, undirected, simple, connected,
seed1701. The ring has +/-1..4 links; G1 uses E//4 successful degree-preserving
swaps; G2 uses the pinned NetworkX random-regular generator. Spectra use sparse
ARPACK or the ring's analytic circulant spectrum. No dense production adjacency.

PhaseA runs CPU tests, GPU value/gradient/mass checks and real official patching,
beta0 and 30-update smoke tests. PhaseB runs exactly the six graph/beta variants,
both scale-only controls and one fresh beta0 calibration at1000 updates. Beta is
0.1 or0.25. At most two graph variants may advance. Conditional PhaseC repeats
paired seeds43/44 with all registered topology/scale controls at selected betas.
Only a qualifying PhaseC winner permits PhaseD at5000 and the registered control
continuations. No extra betas, graph seeds, model/tasks, ranking refinement or
leaderboard submission. Exact selection thresholds/budgets are in the frozen JSON.

For beta<.5, B=(1-beta)I+beta P is invertible: optimization reparameterization is
not a richer hypothesis class. Scale-only controls distinguish score scaling;
ring and rewired controls distinguish connectivity. An optional fixed G2
permutation is permitted only for a selected G2 winner. No causal spectral claim
is justified by a single graph instance.

Checkpoints contain **raw** scores and original optimizer/RNG state. Every mask,
official ranking and exported result uses **effective** scores. The symmetric
custom backward uses P^T=P and fixed-degree gathers to avoid CUDA scatter-add
ordering. CPU gradient checks, full-graph GPU reference tests, exact beta0 and
resume tests verify the adapter. Graph construction is charged; spectra are a
separate diagnostic cost. Every control has the same logging cadence.

Report regeneration and completed-campaign audit run no model forwards:

```bash
bash src/mattr_expander/run_wsl.sh --out "$OUT" --report-only
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OMP_NUM_THREADS=4 \
  ~/.venvs/mattr-optimization-af82b2e/bin/python -m src.mattr_expander.audit --out "$OUT"
```

Use `notebooks/mattr_expander.ipynb` to inspect the frozen configuration, launch
registered phases explicitly or read the saved analysis. Running its default
cells does not launch training. Raw checkpoints, score vectors, adjacency arrays,
full rankings and generated prompt manifests remain under ignored `results/`.
Git retains small summaries, complete faithfulness curves, logs, figures, protocol,
graph/mapping hashes and the compressed canonical model-edge mapping.

The upstream `training_metrics_*.csv` step column is zero-based, matching Gate2.
Gradient diagnostics and example-draw files count completed updates starting at1;
add1 to the upstream step when joining these tables. Ranking CSVs are presentation
tables: official retained-set exports and exact effective tensors preserve the
official tie ordering used for circuit evaluation.
