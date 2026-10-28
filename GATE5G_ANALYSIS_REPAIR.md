# Gate 5G offline analysis repair

The registered 58 GPU evaluation jobs completed under implementation commit
`60e98c15d7ed7c8fa96485ae5a2b0a7e7b1f9133` and protocol SHA-256
`c051cf92d775c92693205b22d07684b911268b96e41d99e777efc1018e8c3462`.
Analysis subsequently failed in the structural extractor: both its explicit keyword
and the unchanged Gate5F overlap helper supplied `actual_edges` to `dict()`.
The original end-to-end synthetic test mocked that extractor, missing the collision.

This is a documented post-freeze **offline analysis repair**. Frozen source files,
the original protocol, masks, rankings, prompts, inference outputs, metric definitions,
bootstrap draws and success criteria remain unchanged. A new scoped adapter calls
the real original extractor/helper and removes only the redundant helper count.
Before doing so it validates the explicit query counts against all frozen structural
masks. It also checks that any existing registered A/C decisions are reproduced.
The new regression test invokes the real structural extractor and overlap helper.

No inference, attribution, model training, new sampling, modified thresholds or new
scientific experiment is authorized or executed by this repair. A new protocol hash
is unnecessary for this offline-only correction: original execution provenance is
retained, and repair commit/source hashes are separately recorded in
`analysis_repair.json`, `analysis_metadata.json` and the generated report.
No further registered GPU evaluations may run from the repair commit: the original
live code lock intentionally still requires the frozen inference commit.

## Resume after the completed evaluations

```bash
cd '/mnt/h/GenAI Research/sketch_prune'
source "$HOME/.venvs/mattr-optimization-af82b2e/bin/activate"
bash scripts/gate5g/05_analyze_repaired.sh
bash scripts/gate5g/06_audit.sh
```

Use the repaired analysis entry point instead of `05_analyze.sh`; do not rerun
preflight, freeze or GPU evaluations. `--dry-run` verifies completed artifacts/counts
without regenerating analysis. The wrapper supports `--help` and `--out`.
The original offline audit and packaging scripts work after the repair commit;
they validate saved inference commits without requiring current HEAD to match.

The researcher-written interpretation remains empty. After reviewing the actual
results, fill that section in the campaign report, then run:

```bash
bash scripts/gate5g/07_package.sh --dry-run
bash scripts/gate5g/07_package.sh --commit
git push origin research/mattr-fidelity-splice
```

Repair provenance is embedded in the already allowlisted `analysis_metadata.json`
and report, so the frozen packaging source needs no changes. No automatic result
commit or leaderboard submission is performed.
