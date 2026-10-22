#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/mattr-optimization-matplotlib" OMP_NUM_THREADS=4
PYTHON="${MATTR_OPT_PYTHON:-$HOME/.venvs/mattr-optimization-af82b2e/bin/python}"
if [[ "${1:-}" == tests ]]; then
  exec "$PYTHON" -m unittest discover -s tests -p test_mattr_interactions.py -v
fi
if [[ "${1:-}" == audit ]]; then
  shift
  exec "$PYTHON" -u -m src.mattr_interactions.audit "$@"
fi
exec "$PYTHON" -u -m src.mattr_interactions.campaign "$@"
