# Gate 6A WSL runbook

## Environment and prerequisites

Run in Ubuntu WSL on the RTX 3060. Reuse the Gate 5F/G Python 3.12 environment, offline Hugging Face cache, pinned clones, and local historical artifacts. The Git remote does not contain the historical raw checkpoints or prompts needed by preflight. Keep those artifacts available in the existing workspace. For environment provisioning, see the [pinned MAttr setup](scripts/mattr/README.md).

```bash
cd '/mnt/h/GenAI Research/sketch_prune'
git switch research/mattr-faithful-objective
source scripts/gate5g/_environment.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
"$GATE5G_PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name()); assert torch.cuda.is_available()'
git status --short -- src scripts configs tests requirements.txt GATE6A_PRIOR_WORK_AUDIT.md
```

Use the same campaign path in every command. The default is `results/mattr_faithful/gate6a`. Pass an absolute `--out` under `results/mattr_faithful/` to every script if you need a separate campaign.

## Full registered run

Preflight checks the exact historical streams and checkpoint identities, runs discovery-only smoke checks, builds the full discovery cache, and computes D from the official clean and all-patched outputs. It needs about 2 GB of local storage for float32 vocabulary targets. Confirmation manifest construction performs no model forwards. Commit all experiment source before the protocol freeze.

```bash
bash scripts/gate6a/00_preflight.sh
bash scripts/gate6a/01_freeze_training.sh
bash scripts/gate6a/run_all.sh
```

`run_all.sh` selects and freezes the EAP-IG reference before running the pilot. It stops if reference CMD exceeds 0.10 or the pilot fails. If projected cumulative time exceeds 16 hours after the pilot, inspect `runtime_projection.json`. With explicit manual approval, resume the same campaign using:

```bash
bash scripts/gate6a/run_all.sh --approve-over-16h
```

## Run one stage at a time

```bash
bash scripts/gate6a/02_select_reference.sh
bash scripts/gate6a/03_pilot.sh
```

Proceed only when `pilot_decision.json` records `continue`. The following block stops on any failure and reuses the pilot trajectory.

```bash
(
  set -e
  for arm in M0 M1 K0 K1 R0; do
    for seed in 42 43 44; do
      if [[ "$arm:$seed" != M0:42 ]]; then
        bash scripts/gate6a/04_train.sh --arm "$arm" --seed "$seed"
      fi
    done
  done
  bash scripts/gate6a/05_eval_dev.sh
  selected="$("$GATE5G_PYTHON" -c 'import json; print(json.load(open("results/mattr_faithful/gate6a/selection.json"))["winner"])')"
  for seed in 45 46; do
    bash scripts/gate6a/04_train.sh --arm "$selected" --seed "$seed"
  done
  bash scripts/gate6a/06_freeze_confirmation.sh
  bash scripts/gate6a/07_eval_confirm.sh
  bash scripts/gate6a/08_analyze.sh
  bash scripts/gate6a/09_audit.sh
)
```

For the manual route, pass `--approve-over-16h` to each costly `04_train.sh`, `05_eval_dev.sh`, or `07_eval_confirm.sh` command when you have approved the projection. `05_eval_dev.sh --ranking M0_s42_5000` verifies a single development result. After freezing confirmation, `07_eval_confirm.sh --ranking mattr_abs_s42_5000` executes or verifies that one ranking. Omit `--ranking` for all registered jobs. Each stage exposes `--help` and `--dry-run`.

## Interruption and integrity

After an ordinary interruption, rerun the same command and campaign. Training restores the latest verified 100-update checkpoint, including Adam, scores, RNG and exact row/k offset. Evaluation reuses only verified completed rankings and exact mask outputs under the same cohort and configuration. An interrupted attempt can incur additional compute. Inspect `status_*.json`, `attempts/`, and `runtime_projection.json` for evidence.

For a protocol mismatch, nonfinite loss, failed correctness check, or CUDA OOM, stop and inspect the error. Preserve the old campaign. A source fix requires a new commit, a fresh campaign, and a new freeze before registered execution. The frozen settings include batch sizes and precision.

## Optional public test and results

Public test requires successful confirmation audit, explicit approval and the same frozen implementation HEAD. Run this before any results commit if you choose to use it.

```bash
bash scripts/gate6a/10_public_test.sh --i-approve-public-test
```

After audit, inspect the generated report under the campaign directory. The packager stages an allowlist and rejects files over 5 MB. The final interpretation remains researcher-written.

```bash
bash scripts/gate6a/11_package.sh --dry-run
bash scripts/gate6a/11_package.sh --commit
git push -u origin research/mattr-faithful-objective
```
