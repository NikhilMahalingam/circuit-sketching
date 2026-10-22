#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export UV_PROJECT_ENVIRONMENT="${MATTR_OPT_ENV:-$HOME/.venvs/mattr-optimization-af82b2e}"
export UV_LINK_MODE=copy
export UV_PYTHON=3.12.15
# A new environment; never sync the older MAttr or Sketch-ACDC environments.
"$ROOT/.cache/mattr-bootstrap/bin/uv" sync --frozen --offline \
  --project "$ROOT/external/matryoshka-attribution"
