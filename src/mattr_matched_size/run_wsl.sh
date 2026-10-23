#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.cache/mattr-optimization-matplotlib" OMP_NUM_THREADS=4
PYTHON="${MATTR_OPT_PYTHON:-$HOME/.venvs/mattr-optimization-af82b2e/bin/python}"
if [[ "${1:-}" == topology ]]; then
  shift; exec "$PYTHON" -u -m src.mattr_matched_size.topology "$@"
fi
if [[ "${1:-}" == tests ]]; then
  exec "$PYTHON" -m unittest discover -s tests -p test_mattr_matched_size.py -v
fi
exec "$PYTHON" -u -m src.mattr_matched_size.campaign "$@"
