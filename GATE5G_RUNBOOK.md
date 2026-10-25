# Gate 5G manual execution

Inference only. No training, backward calls, model updates, attribution reruns or leaderboard submissions. All evaluations are manual foreground jobs in WSL on the existing pinned environment. Never use fresh Gate5G prompts for model checks before freeze.

## Activate and check CUDA

```bash
cd '/mnt/h/GenAI Research/sketch_prune'
git switch research/mattr-fidelity-splice
source "$HOME/.venvs/mattr-optimization-af82b2e/bin/activate"
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

Wrappers resolve the repository from their own location and use that exact Python environment. `GATE5G_PYTHON` may specify the same pinned environment at another path, but freeze records and enforces its identity. Do not upgrade packages or change historical sources.

For this shared Windows/WSL checkout, repository-local `core.autocrlf=true` makes both Git clients interpret the existing CRLF worktree consistently; no historical files were rewritten. If another shared checkout appears clean in Windows but dirty in WSL, first inspect `git -c core.autocrlf=true status --short`. Only when that confirms a clean tree, use `git config --local core.autocrlf true` before freeze. The experiment still hashes exact source bytes and rejects real code changes.

## Preflight and freeze

```bash
bash scripts/gate5g/00_preflight.sh
bash scripts/gate5g/01_freeze_protocol.sh
```

Preflight inventories checkpoints and effective exports, reconstructs the signed EAP ordering, verifies splices/topology, constructs the new cohort, and runs bounded correctness on **two historical Gate5F prompts only**. CPU-only inspection is available with `--cpu-only`, but cannot authorize freeze. Runtime extrapolation from two prompts is conservative and uncertain. `pre_freeze_test_prompts.csv` records identities before forwards. Fresh prompts are only hashed/tokenized until freeze.

Freeze requires a clean tracked implementation commit and passing preflight. It writes ignored `results/mattr_fidelity_splice/gate5g/gate5g_protocol.json` and `.sha256`. The protocol references the already committed implementation; the implementation commit need not include the protocol. Do not commit the protocol separately before evaluation: doing so changes HEAD and intentionally locks registered execution. Result packaging occurs later.

## Inspect planned jobs

```bash
bash scripts/gate5g/run_all.sh --dry-run
```

All registered execution verifies exact HEAD, relevant source bytes/cleanliness, pinned dependencies, protocol, rankings, cohort and frozen query hashes. No override exists for this code lock.

## Individual Parts A, B and C

```bash
# One complete Part A ranking; calibrates the cumulative runtime guard.
bash scripts/gate5g/02_fidelity.sh --job baseline_n10000_s42_step5000
# All Part A core rankings:
bash scripts/gate5g/02_fidelity.sh
# Extended seed diagnostics (four caps); reuses globally identical mask outputs:
bash scripts/gate5g/03_seed_diagnostics.sh
# A single prespecified splice:
bash scripts/gate5g/04_splice.sh --method baseline --seed 42 --k 324
# Or all20 registered splices:
bash scripts/gate5g/04_splice.sh
```

Registered caps are60,80,100,116,140,180,240,296,400,640; the official nominal grid is unchanged. PartB uses60,116,180,296. Splices additionally measure K−1/K/K+1 for fixed boundary diagnostics. Full-vocabulary KL is recorded for every evaluated mask, including all three primary nominal endpoints and the prominent secondary cap116 endpoint. No logits for circuit distributions are saved.

## Full sequential run and safe resume

```bash
bash scripts/gate5g/run_all.sh
```

This runs only frozen registered A→B→C jobs, sequentially, without starting analysis or a new gate. Ctrl+C stops the foreground process. Rerun the identical command to resume. Completed jobs are skipped only after identity and byte-hash checks. Individual immutable mask outputs are shared across jobs; interrupted writes are preserved in an `incomplete` subdirectory and recomputed. Corrupt completed outputs/cache hashes cause a hard failure, not silent reuse. Failed attempts and their actual forwards remain charged.

After the first complete PartA ranking the CLI projects cumulative research runtime. If it exceeds five hours, inspect `runtime_projection.json`, then explicitly approve unchanged settings:

```bash
bash scripts/gate5g/run_all.sh --approve-over-5h
```

The same flag applies to individual PartA/B/C commands. It only authorizes runtime; it never overrides code, hash, correctness, cohort or memory checks. OOM stops; no automatic batch-size reduction occurs.

## Analyze, audit and review

```bash
bash scripts/gate5g/05_analyze.sh
bash scripts/gate5g/06_audit.sh
less GATE5G_FIDELITY_SPLICE_REPORT.md
```

These stages perform **zero model forwards**. Analysis uses one shared5000-draw exact-template paired prompt bootstrap (seed5645). PartA uses98.333333% marginal intervals at10/20/50%; PartC uses its joint CPR/compact-accuracy rule. Training seeds are fixed descriptive units. The audit does not fail solely because researcher interpretation is empty.

Write your conclusions in `## Interpretation and recommendation (researcher-written)` in the campaign report `results/mattr_fidelity_splice/gate5g/GATE5G_FIDELITY_SPLICE_REPORT.md`, then mirror that text in the root report (or rerun offline analysis; it preserves the campaign interpretation). Do not edit protocol, thresholds or source to influence classification.

## Review, stage, commit and push results

```bash
bash scripts/gate5g/07_package.sh --dry-run
bash scripts/gate5g/07_package.sh
git diff --cached --stat
# After reviewing the staged allowlist and filling the interpretation:
bash scripts/gate5g/07_package.sh --commit
git push origin research/mattr-fidelity-splice
```

Packaging requires a successful final audit, displays and stages only allowlisted small summaries/figures/reports, rejects files larger than10MB, and refuses unrelated staged changes. It warns when researcher interpretation is empty. `--commit` is required to create `research: Gate 5G fidelity and splice results`; it never pushes automatically. Raw prompts, score/checkpoint tensors, full-vocabulary clean log-probabilities and caches remain ignored/local.

Analysis/audit/packaging validate saved execution commits but permit current HEAD to advance for a results commit. Registered evaluation intentionally stops after HEAD changes; completed experiments are still valid.

## Protocol deviations

If a correctness fix or resource amendment is necessary after freeze: stop; preserve the old campaign, logs and protocol; document the reason; make a new tested implementation commit; use a **new** campaign directory via `--out results/mattr_fidelity_splice/gate5g_amendment1` on every stage; rerun historical-only preflight and freeze a new hash before any fresh-cohort forward. Do not overwrite or loosen the original protocol. Previously viewed fresh outcomes must be disclosed; an amendment does not restore historical freshness.

Every public wrapper supports `--help`, accepts `--out`, works from any directory when called by absolute path, returns nonzero on failure and records stage status/logs. Preflight/freeze also have no-forward `--dry-run` descriptions; registered job wrappers validate their lock even in dry-run mode. Nothing invokes Codex or an LLM API.
