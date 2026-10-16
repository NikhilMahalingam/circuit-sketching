#!/usr/bin/env bash
# Isolated official Python/uv environment; no Sketch-ACDC environment changes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$ROOT/external/matryoshka-attribution"
BOOTSTRAP="$ROOT/.cache/mattr-bootstrap"
COMMIT=af82b2eef145dd3e66c793a413352d6931a152d7
if [[ ! -d "$REPO/.git" ]]; then
  git clone https://github.com/aryamanarora/matryoshka-attribution.git "$REPO"
  git -C "$REPO" checkout "$COMMIT"
fi
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] || { echo 'Unexpected official revision' >&2; exit 1; }
if [[ ! -x "$BOOTSTRAP/bin/uv" ]]; then
  python3 -m pip install --target "$BOOTSTRAP" uv==0.13.1
fi
export PATH="$BOOTSTRAP/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/mattr-af82b2e}"
export UV_LINK_MODE=copy
export UV_PYTHON=3.12.15
export WANDB_MODE=disabled
# The pristine Windows clone has CRLF files. Check out its indexed bytes as LF
# before invoking upstream bash. Abort rather than overwrite upstream edits.
if [[ -n "$(git -c core.autocrlf=true -C "$REPO" status --porcelain)" ]]; then
  echo 'Official checkout has edits; refusing to overwrite it.' >&2
  exit 1
fi
git -C "$REPO" config core.autocrlf false
git -C "$REPO" archive HEAD | tar -x -C "$REPO"
git -C "$REPO" diff --exit-code --quiet
# Public submodules may use SSH URLs; HTTPS needs no private-key setup.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=url.https://github.com/.insteadOf
export GIT_CONFIG_VALUE_0=git@github.com:
cd "$REPO"
uv --version
uv python install 3.12.15
bash scripts/setup.sh
uv run --frozen python -m pytest tests -q
uv run --frozen python -c 'import torch,sys,importlib.metadata as m; print(sys.version); print({p:m.version(p) for p in ["torch","transformers","transformer-lens","datasets","pytest"]}); assert torch.cuda.is_available(); x=torch.ones(8,device="cuda"); print(torch.cuda.get_device_name(),x.sum().item())'
uv run --frozen python scripts/mib/eval_mib.py --help
