#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../gate5g/_environment.sh"
exec "$GATE5G_PYTHON" -u -m src.mattr_ensemble.cli preflight "$@"
