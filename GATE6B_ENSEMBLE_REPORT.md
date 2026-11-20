# Gate 6B seed ensemble results

Registered results are pending. The analysis stage generates the completed report after evaluating the ten frozen rankings. See the [WSL runbook](GATE6B_RUNBOOK.md).

Graph preflight verified 13 Gate 6A rankings, including both EAP-IG conventions, and replayed the 11 input training checkpoint exports. All three ensembles cover the canonical 32,491 edges with positive scores from 1 to 32,491. The one-member reproduction and signed/absolute ensemble prefix checks passed at 22 cutoffs.

Selection seed 5548 yielded 6,736 eligible validation rows, of which 512 were selected across nine templates. On eight excluded discovery prompts, the bounded CUDA run reproduced the official CMD and CPR, verified capped-116 KL, and replayed all 15 circuits from the cache with zero additional root forwards. The entire check recorded 56 root forwards and zero updates. The first combined grid and cap sweep took 3.09 seconds on the eight prompts. The registered run is estimated at 25–35 minutes based on the historical 512-prompt evaluations and will reproject after its first completed ranking.

[Input and ensemble hashes](results/mattr_ensemble/gate6b/implementation_checks/ensemble_manifest.json)

## Interpretation and recommendation (researcher-written)
