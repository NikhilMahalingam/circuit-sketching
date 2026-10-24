#!/usr/bin/env bash
set -euo pipefail
GATE5F_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$GATE5F_ROOT"
export PYTHONPATH="$GATE5F_ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$GATE5F_ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export MPLBACKEND=Agg MPLCONFIGDIR="$GATE5F_ROOT/.cache/mattr-optimization-matplotlib" OMP_NUM_THREADS=4
GATE5F_PYTHON="${GATE5F_PYTHON:-${VIRTUAL_ENV:-$HOME/.venvs/mattr-optimization-af82b2e}/bin/python}"
if [[ ! -x "$GATE5F_PYTHON" ]]; then echo "Existing pinned Python not found: $GATE5F_PYTHON" >&2; exit 1; fi
