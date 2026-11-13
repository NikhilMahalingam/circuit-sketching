# Gate 6A faithfulness-trained MAttr

Registered results are pending. After the manual run, the analysis script generates the campaign report using the frozen protocol and saved outputs. See the [prior-work audit](GATE6A_PRIOR_WORK_AUDIT.md) and [WSL runbook](GATE6A_RUNBOOK.md).

## Implementation checks

- 17 synthetic CPU checks passed. The WSL script syntax, help and dry-run checks passed from outside the repository.
- The graph-only preflight verified all 32,491 edge identities, historical row and sparsity streams, and the ten signed EAP-IG masks. The confirmation selection found 7,248 eligible rows and selected 512 across nine templates.
- On 50 discovery prompts, the maximum all-ones margin loss was 1.98e-11 and the maximum absolute all-ones KL was 3.87e-7. The largest all-zero margin discrepancy against official patching was 2.57e-5. Independent clean log probabilities agreed within 3.82e-5.
- The first 30 historical ordinary MAttr losses matched exactly. Mean loss across the fixed 30 smoke queries decreased from 0.7594 to 0.1318 for margin, 0.8310 to 0.1062 for KL, and 0.8015 to 0.3403 for upstream CMD. The 15+15 update resume matched uninterrupted scores, optimizer and RNG state. GPT-2 weights were unchanged.
- The passing GPU attempt used 880 root forwards, 5,592 example-forwards and 210 bounded updates, including baseline replay and checkpoint replay. Peaks were 2,655 MiB allocated and 4,464 MiB reserved. Updates took 0.26–0.30 seconds. Uncached capped-circuit evaluation averaged 1.46 seconds per circuit on 80 prompts, including mask construction.
- The 10,000-row target build and its denominator D are pending manual WSL preflight. No development, confirmation, or public-test model forwards were executed.

## Interpretation and recommendation (researcher-written)
