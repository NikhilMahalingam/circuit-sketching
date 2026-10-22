#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export MPLBACKEND=Agg
export MPLCONFIGDIR="$ROOT/.cache/mattr-optimization-matplotlib"
cd "$ROOT"
PYTHON="${MATTR_OPT_PYTHON:-$HOME/.venvs/mattr-optimization-af82b2e/bin/python}"
if [[ "${1:-}" == tests ]]; then
  exec "$PYTHON" -m unittest discover -s tests -p test_mattr_optimization.py -v
fi
if [[ "${1:-}" == tests-gate2 ]]; then
  exec "$PYTHON" -m unittest discover -s tests -p test_mattr_gate2.py -v
fi
if [[ "${1:-}" == gate2 ]]; then
  shift
  exec "$PYTHON" -u -m src.mattr_optimization.gate2 "$@"
fi
if [[ "${1:-}" == verify-gate1 ]]; then
  shift
  exec "$PYTHON" -m src.mattr_optimization.verify_gate1 "$@"
fi
if [[ "${1:-}" == gate1 ]]; then
  shift
  "$PYTHON" -m src.mattr_optimization.resume_data_check "$@"
  exec "$PYTHON" -m src.mattr_optimization.gate1 "$@"
fi
exec "$PYTHON" -m src.mattr_optimization.gate0 "$@"
