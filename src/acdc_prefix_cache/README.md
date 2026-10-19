# Prefix-cached ACDC

This is an independent MIB adaptation of Stage J in the supplied research log.
The original implementation and decision trace were not supplied. The 2.35×
number is an independent result, not a target numerical result for MIB's
different dataset and loss.

Run from the project root in **WSL Ubuntu**, using the existing CUDA packages
without modifying their installation:

```bash
bash src/acdc_prefix_cache/run_wsl.sh tests
OUT="$PWD/results/acdc_prefix_cache/$(date -u +%Y%m%dT%H%M%SZ)"
bash src/acdc_prefix_cache/run_wsl.sh --stage a --out "$OUT"
```

Stage A includes real GPT-2/official MIB checks. Stop and review the report
and planning estimates before running either command below:

```bash
bash src/acdc_prefix_cache/run_wsl.sh --stage b --out "$OUT"
bash src/acdc_prefix_cache/run_wsl.sh --stage c --out "$OUT" --allow-full-graph
```

All default stages are independently configurable using `--help`. Stage C is
explicitly disabled and additionally requires completed A and B. An existing
passing A is reused without extra forwards. B/C use versioned per-trial
checkpoints and complete result markers; aborted attempts remain in their
accounting and are charged when a run resumes. No experimental settings are
changed after OOM. Pick a new directory and explicit settings when changing
the core protocol. `--sizes`, `--checks`, and `--repeats` configure a new
performance phase before it starts; a started phase's protocol is immutable.

The default signed deletion criterion is `proposed_loss - current_loss < 0.05`
using the official negative IOI logit difference. This matches ACDC's default
signed rule; it differs from the old absolute-change Sketch-ACDC search and
the independent log's squared probability distance. Both new arms share the
same rule and fixed reverse-topological traversal. Connectivity cleanup is
not a search shortcut; each explicit edge is tested. Independent MIB oracle
evaluation performs its own pruning and must agree within saved tolerances.

The cache is guarded by the complete graph's descendant closure, every
previous deletion, model parameter versions and relevant model settings.
Only standard learned-position, serial attention/MLP architectures are
supported. MLP-only deletions conservatively restart their entire block.
Output-edge deletions restart the last block so its residual hook executes.
No current suffix sender is replaced by a frozen full-model sender.

`PREFIX_PYTHON` can select another compatible WSL interpreter explicitly.
The default reuses the installed WSL Python packages read-only; it does not
run uv/pip or import MAttr experiments. The original MIB/EAP clone and commit
hashes are checked separately. Models and datasets use the existing offline
cache. There is no custom IOI fallback or large-model download.

Generate the notebook with
`python -m src.acdc_prefix_cache.notebook`. Its default flags read saved
artifacts only; enabling A/B/C dispatches the corresponding WSL command.
The executed notebook is an analysis interface over real saved GPU results.
