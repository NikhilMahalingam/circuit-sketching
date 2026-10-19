#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MPLCONFIGDIR="$ROOT/.cache/prefix-matplotlib"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
# Read-only use of installed packages: no uv sync, pip install, or MAttr imports.
PYTHON="${PREFIX_PYTHON:-$HOME/.venvs/mattr-af82b2e/bin/python}"
if [[ "${1:-}" == tests ]]; then
  exec "$PYTHON" -m unittest discover -s tests -p test_acdc_prefix_cache.py -v
fi
exec "$PYTHON" -m src.acdc_prefix_cache.runner "$@"
