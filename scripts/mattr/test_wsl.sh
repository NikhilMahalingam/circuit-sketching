#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for script in "$ROOT"/scripts/mattr/*.sh; do bash -n "$script"; done
export PYTHONPATH="$ROOT:$ROOT/external/matryoshka-attribution/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/mattr-af82b2e}/bin/python"
cd "$ROOT/external/matryoshka-attribution"
"$PYTHON" -m pytest tests -q
cd "$ROOT"
"$PYTHON" -m unittest discover -s tests -p test_mattr_reproduction.py -v
