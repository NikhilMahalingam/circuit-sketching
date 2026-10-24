#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_environment.sh"
exec "$GATE5F_PYTHON" -u -m src.mattr_data_scaling.cli freeze "$@"
