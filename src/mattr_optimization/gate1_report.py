"""Report the measured full official edge baseline; no sweeps or variant claims."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .common import ROOT, GRID, integration_weights, write_json


def generate(out):
    out = Path(out)
    config = json.loads((out/'experiment_config.json').read_text())
    marker = json.loads((out/'gate1_passed.json').read_text())
    metrics = json.loads((out/'official_metrics.json').read_text())
    train = pd.read_csv(out/'training_metrics.csv')
    curve = pd.read_csv(out/'faithfulness_curve.csv')
    frames = [pd.read_csv(p) for p in sorted((out/'attempts').glob('*/compute_accounting.csv'))]
    frames.append(pd.read_csv(out/'compute_accounting.csv'))
    costs = pd.concat(frames, ignore_index=True)
    costs.to_csv(out/'complete_compute_accounting.csv', index=False)
    summaries = [json.loads(p.read_text()) for p in sorted((out/'attempts').glob('*/process_summary.json'))]
    summaries.append(json.loads((out/'process_summary.json').read_text()))
    process_seconds = sum(s['process_seconds'] for s in summaries)
    preflight_path = out/'resume_data_checks.jsonl'
    preflight_seconds = sum(json.loads(line)['seconds'] for line in preflight_path.read_text().splitlines()) if preflight_path.exists() else 0.
    draw_frames = [pd.read_csv(p) for p in sorted((out/'attempts').glob('*/training_example_draws.csv'))]
    draw_frames.append(pd.read_csv(out/'training_example_draws.csv'))
    draws = pd.concat(draw_frames, ignore_index=True)
    usage = draws.official_filtered_index.value_counts().reindex(range(marker['train_examples']), fill_value=0)
    usage.rename_axis('official_filtered_index').rename('actual_presentations_including_repeated_work').to_csv(out/'training_example_usage.csv')
    pd.DataFrame({'fraction': GRID, 'linear_cpr_weight': integration_weights(),
        'requested_edges': [int(x*32491) for x in GRID]}).to_csv(out/'cpr_grid_weights.csv', index=False)
    unique = int((usage > 0).sum())
    training = costs.loc[costs.stage=='training', 'seconds'].sum()
    evaluating = costs.loc[costs.stage=='official_validation_evaluation', 'seconds'].sum()
    full_stage = costs.seconds.sum()
    patching = pd.read_csv(out/'patching_checks.csv')
    # Independent audit only; reported benchmark metrics remain the official evaluator's.
    audit_area = float(np.trapezoid(curve.faithfulness, curve.fraction))
    audit_cmd = float(np.trapezoid(np.abs(1-curve.faithfulness), curve.fraction))
    if not np.isclose(audit_area, metrics['cpr'], atol=1e-6, rtol=1e-7):
        raise AssertionError('Saved official curve disagrees with official CPR.')
    if not np.isclose(audit_cmd, metrics['cmd'], atol=1e-6, rtol=1e-7):
        raise AssertionError('Saved official curve disagrees with official CMD.')
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].plot(train.step+1, train.loss, alpha=.2, lw=.5)
    axes[0].plot(train.step+1, train.loss.rolling(100, min_periods=1).mean(), lw=1)
    axes[0].set(xlabel='Iteration', ylabel='Negative logit difference', title='Training (100-step rolling mean)')
    axes[1].plot(curve.fraction, curve.faithfulness, marker='o')
    axes[1].axhline(1, color='gray', linestyle='--')
    axes[1].set(xscale='log', xlabel='Requested edge fraction (log display)',
        ylabel='Official normalized faithfulness', title='Full official validation')
    axes[2].hist(usage, bins=range(int(usage.max())+2), align='left', edgecolor='black')
    axes[2].set(xlabel='Actual train-example presentations', ylabel='Number of eligible rows', title='Sampling with replacement')
    fig.tight_layout()
    fig.savefig(out/'gate1_diagnostics.png', dpi=160)
    plt.close(fig)
    refs = metrics['official_references']
    rows = '\n'.join(f'| {r.fraction:g} | {int(r.requested_edges)} | {r.faithfulness:.6f} | {r.accuracy:.4f} |'
                     for r in curve.itertuples())
    text = f'''# MAttr optimization: Gate 1 official edge baseline

Campaign: `{out.name}`. **Gate 1 completed.** Only the official baseline was run; Gate 2 and all algorithm modifications remain unrun.

## Protocol and provenance

Pinned MAttr `af82b2eef145dd3e66c793a413352d6931a152d7`, required MIB fork `582fdbc8a2b0f7063a265c8f94851fd2a3aa5480`; exact EAP commit, source/lock hashes, installed versions, command argv and execution-source snapshots are saved. A dedicated WSL Python 3.12 environment uses the official frozen dependency lock on RTX 3060 12 GB. Older environments and experiments were not changed.

GPT-2 Small, official MIB IOI, edge granularity (32,491 real scores), seed 42, 5,000 requested iterations; {len(train)} logged updates. The uncapped official train pool has {marker['train_examples']} rows and the uncapped validation split has {marker['validation_examples']} rows, both in original official order. Clean prompts are distinct across splits and answer names were checked to be single tokens. No test split or custom IOI dataset was accessed.

Uniform floating k; sigmoid top-k; iso/denoising; zero score initialization; Adam lr 0.05, epsilon 1e-8, betas (0.9,0.999); T=.5; 30 bisection iterations. One example is sampled with replacement per training iteration. Evaluation batch={config['evaluation_batch_size']}, donor-cache budget={config['counterfactual_cache_gb']} GB; explicit resource overrides relative to upstream defaults: {config['resource_overrides']}. Float32/TF32 disabled. Upstream model-weight gradient accumulation is preserved; no freezing or algorithm optimization was introduced.

The adapter calls the upstream edge CLI, trainer, live-sender intervention closure and official signed edge evaluator. It adds counters, checkpointing and audited raw-vector exports. Versioned checkpoints restore scores, Adam and Python/NumPy/Torch/CUDA RNGs. Cached evaluation vectors are reused only after exact ranking, source/configuration, data fingerprint and vector-hash checks; reuses incur zero model forwards. Setup/replayed work is included in the complete accounting. No settings change automatically after OOM.

## Official validation results

- **CPR AUC: {metrics['cpr']:.8f}**.
- **CMD AUC: {metrics['cmd']:.8f}** (lower absolute deviation from full-model behavior).
- Accuracy log-AUC: {metrics['accuracy_auc']:.8f}.
- Clean mean logit difference: {refs['baseline']:.8f}; official empty-circuit corrupted reference: {refs['corrupted']:.8f}.
- Maximum discrepancy in four post-training hard-mask checks against official MIB: {patching.absolute_error.max():.8g}, within atol=5e-5/rtol=1e-5.

| Requested fraction | Requested edges | Faithfulness | Accuracy |
|---:|---:|---:|---:|
{rows}

Requested counts are before official connectivity pruning; weighted edge counts are saved separately. CPR uses the official linear retained-fraction trapezoids, with no added zero endpoint and no clipping. Accuracy log-AUC is a different metric. Values above one reflect overshoot; a larger CPR alone does not establish improved mechanistic fidelity. Raw per-example clean, corrupted and ten grid-point validation vectors, the learned ranking and score tensors are saved. Independent integration of the saved curve agrees with official CPR/CMD within 1e-6.

The post-run verifier exports `official_nested_ranking.csv` with the actual official Torch tie ordering, and `official_circuit_membership_summary.csv` with pruned circuit sizes and identity hashes at each grid point. `ranking.csv` is a score-sorted presentation table; use the official ranking/graph procedure when ties matter. This export changes no trained scores or benchmark results and executes no model forwards.

## Published comparison and research conclusion

The [paper v1, Table 4](https://arxiv.org/html/2609.25518v1) reports GPT-2/IOI edge validation CPR **10.59** for its 5,000-step recipe. Our result differs from that rounded target by **{metrics['cpr']-10.59:+.6f}**. Rounded-value agreement, if present, cannot establish precision beyond the published rounding; otherwise this is a reproduction gap to investigate before claiming an improvement. Hardware/library behavior, precise evaluation protocol, revision and numerical precision must be considered. Our earlier node-level CPR 1.83 and Gate-0 small-cohort CPR are not matched edge-baseline comparisons.

This one seed establishes the measured reference and feasibility, not improvement over MAttr, statistical significance, or leaderboard performance. Sampling, initialization and ranking refinement have not been tested. No success threshold has been selected from variant results.

## Compute and memory

- Training, including checkpoint work: {training:.1f} seconds ({training/60:.2f} min).
- Full official validation evaluation, including raw-vector export: {evaluating:.1f} seconds ({evaluating/60:.2f} min).
- Instrumented model/data/correctness/training/evaluation stages: {full_stage:.1f} seconds ({full_stage/60:.2f} min); interpreter startup/source verification/reporting are outside these stage totals.
- Measured process work after source/config preparation: {process_seconds:.1f} seconds ({process_seconds/60:.2f} min), including upstream graph construction, pretokenization, exports and time between instrumented stages. Source checks, interpreter startup and report generation are additional overhead.
- Resume data-preflight CPU work outside that process timer: {preflight_seconds:.1f} seconds; zero model forwards. The WSL wrapper verifies saved row contents/order before any restart.
- Maximum allocated CUDA memory across stages: {costs.peak_allocated_mib.max()/1024:.3f} GiB; maximum reserved: {costs.peak_reserved_mib.max()/1024:.3f} GiB.
- Maximum sampled CPU RSS across stages: {costs.sampled_peak_cpu_rss_mib.max()/1024:.3f} GiB.
- Real root forward batches: {int(costs.forward_batches.sum())}; individual example-forwards: {int(costs.example_forwards.sum())}; backward calls: {int(costs.backward_calls.sum())}; optimizer updates this and archived attempts: {int(costs.optimizer_updates.sum())}.
- Eligible train rows: {marker['train_examples']}; actual unique sampled rows across attempts: {unique}; actual presentations including repeated work: {len(draws)}.

CUDA synchronizes at stage boundaries. Counters observe actual root forwards, not counterfactual-cache hits; activation-checkpoint recomputation inside backward is not a new root forward and remains part of measured backward/training cost. Training has required score-gradient graphs; held-out evaluation and hard-mask correctness checks use no-grad. No full-model weight updates occur. Allocation peaks measure PyTorch allocator usage, not all system GPU memory.

The raw upstream console uses calendar/time.time timestamps; WSL clock jumps were observed during this run, including a negative printed pretokenization duration. Those calendar-based durations and progress rates are not used for compute conclusions. The accounting and process timers above use monotonic perf_counter with CUDA synchronization.

## Next gate

Stop here. Gate 2 should run unchanged edge MAttr on nested 16/32/64/128 discovery pools with one fixed validation cohort, seed 42 first, then 43/44 after feasibility. Use these runs to characterize baseline noise and freeze improvement criteria before testing algorithm variants. No automatic sweep, multi-model evaluation or leaderboard submission is scheduled.
'''
    (out/'MATTR_BASELINE_REPORT.md').write_text(text, encoding='utf-8')
    (ROOT/'MATTR_BASELINE_REPORT.md').write_text(text, encoding='utf-8')
    write_json(out/'curve_audit.json', {'official_cpr': metrics['cpr'], 'audit_cpr': audit_area,
        'official_cmd': metrics['cmd'], 'audit_cmd': audit_cmd, 'passed': True})
