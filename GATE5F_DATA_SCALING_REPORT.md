# Gate 5F: data scaling and training duration

**Registered experiment not executed. No research classification or performance results yet.**

The implementation, historical compatibility inventory, bounded correctness
tests, EAP throughput preflight, and reproducible manual pipeline are provided.
Follow [GATE5F_RUNBOOK.md](GATE5F_RUNBOOK.md) to launch the experiment yourself.

After all registered jobs complete, `scripts/gate5f/05_analyze.sh` replaces this
status document with the actual five-seed primary result, factorial contrasts,
official metrics, topology-aware controls, compute accounting and eight figures.
It uses only local outputs and does not execute model forwards. Incomplete runs
produce a missing-job report rather than fabricated performance numbers.
