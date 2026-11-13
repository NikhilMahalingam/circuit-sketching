#!/usr/bin/env bash
set -euo pipefail
if (($# == 0)) || [[ "$1" == "--help" ]]; then
    printf 'Usage: bash _run.sh STAGE [options]\n'
    exit 0
fi
gate6a_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$gate6a_dir/../gate5g/_environment.sh"
exec "$GATE5G_PYTHON" -u -m src.mattr_faithful.cli "$@"
