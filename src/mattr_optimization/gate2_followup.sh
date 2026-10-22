#!/usr/bin/env bash
# Serial follow-up: never contend with the authorized main sweep for CUDA memory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/mattr-optimization-matplotlib"
PYTHON="${MATTR_OPT_PYTHON:-$HOME/.venvs/mattr-optimization-af82b2e/bin/python}"
MAIN="results/mattr_optimization/20261021_gate2_seeds43_44"
FIRST="results/mattr_optimization/20261021T172035Z_gate2"
FINAL="results/mattr_optimization/20261021_gate2_final"
mkdir -p "$FINAL"
while [[ ! -f "$MAIN/gate2_passed.json" ]]; do
  if find "$MAIN" -name failure.json -print -quit | grep -q .; then
    echo "Main sweep failed; refusing follow-up." >&2
    exit 1
  fi
  sleep 20
done
for PAIR in "43 42" "44 42" "42 43" "42 44"; do
  read -r TRAIN SUBSET <<< "$PAIR"
  "$PYTHON" -u -m src.mattr_optimization.gate2_controls \
    --out "$FINAL/controls/train${TRAIN}_subset${SUBSET}" \
    --training-seed "$TRAIN" --subset-seed "$SUBSET"
done
"$PYTHON" -u -m src.mattr_optimization.gate2_frozen --out "$FINAL/frozen" \
  --campaigns "$FIRST" "$MAIN" --gate1 results/mattr_optimization/20261021T134352Z_gate1
"$PYTHON" -u -m src.mattr_optimization.gate2_final_analysis --out "$FINAL" \
  --campaigns "$FIRST" "$MAIN" --controls "$FINAL/controls" --frozen "$FINAL/frozen"
echo "Gate 2 final analysis complete. No Gate 3 launched."
