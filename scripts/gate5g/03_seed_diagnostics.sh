#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_environment.sh"
exec "$GATE5G_PYTHON" -u -m src.mattr_fidelity_splice.cli diagnose "$@"
