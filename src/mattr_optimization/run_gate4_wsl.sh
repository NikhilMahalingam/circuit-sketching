#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/mattr-optimization-matplotlib"
PYTHON="${MATTR_OPT_PYTHON:-$HOME/.venvs/mattr-optimization-af82b2e/bin/python}"
if [[ "${1:-}" == smoke ]]; then
  shift
  exec "$PYTHON" -u -m src.mattr_optimization.gate4_attribution --smoke "$@"
fi
if [[ "${1:-}" == tests ]]; then
  exec "$PYTHON" -m unittest discover -s tests -p 'test_mattr_gate4*.py' -v
fi
campaign_dir=""
previous_arg=""
freeze_only=false
for task_arg in "$@"; do
  if [[ "$previous_arg" == --out ]]; then campaign_dir="$task_arg"; fi
  if [[ "$task_arg" == --freeze-only ]]; then freeze_only=true; fi
  previous_arg="$task_arg"
done
"$PYTHON" -u -m src.mattr_optimization.gate4 "$@"
if [[ "$freeze_only" == false && -n "$campaign_dir" && -f "$campaign_dir/gate4_completed.json" ]]; then
  # CPU-only audit also corrects the frozen reporter's early display-time scope;
  # never changes protocol bytes, screen costs, quality values, or decisions.
  "$PYTHON" -u -m src.mattr_optimization.verify_gate4 --out "$campaign_dir"
fi
