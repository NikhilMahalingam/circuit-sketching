#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/mattr-af82b2e}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/.cache/huggingface"
export MPLCONFIGDIR="$ROOT/.cache/mattr-matplotlib"
export MPLBACKEND=Agg
export WANDB_MODE=disabled
cd "$ROOT/external/matryoshka-attribution"
exec "$ROOT/.cache/mattr-bootstrap/bin/uv" run --frozen python -m src.mattr_reproduction.runner "$@"
