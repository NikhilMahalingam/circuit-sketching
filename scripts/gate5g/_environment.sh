#!/usr/bin/env bash
set -euo pipefail
GATE5G_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$GATE5G_ROOT"
export PYTHONPATH="$GATE5G_ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$GATE5G_ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export MPLBACKEND=Agg MPLCONFIGDIR="$GATE5G_ROOT/.cache/mattr-optimization-matplotlib" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
GATE5G_PYTHON="${GATE5G_PYTHON:-$HOME/.venvs/mattr-optimization-af82b2e/bin/python}"
if [[ ! -x "$GATE5G_PYTHON" ]]; then echo "Pinned Python not found: $GATE5G_PYTHON" >&2; exit 1; fi
