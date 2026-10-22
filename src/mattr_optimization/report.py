"""Generate Gate-0 measured findings without predicting algorithm improvements."""
import json
import pandas as pd
import matplotlib.pyplot as plt

from .common import ROOT, GRID, integration_weights, write_json


def generate(out):
    out = __import__('pathlib').Path(out)
    metrics = json.loads((out/'official_metrics.json').read_text())
    config = json.loads((out/'experiment_config.json').read_text())
    cost_frames = [pd.read_csv(p) for p in sorted((out/'attempts').glob('*/compute_accounting.csv'))]
    cost_frames.append(pd.read_csv(out/'compute_accounting.csv'))
    cost = pd.concat(cost_frames, ignore_index=True)
    cost.to_csv(out/'complete_compute_accounting.csv', index=False)
    diagnostics = json.loads((out/'training_diagnostics.json').read_text())
    curve = pd.read_csv(out/'faithfulness_curve.csv')
    train = pd.read_csv(out/'training_metrics.csv')
    patching = pd.read_csv(out/'patching_checks.csv')
    draw_frames = [pd.read_csv(p) for p in sorted((out/'attempts').glob('*/training_example_draws.csv'))]
    draw_frames.append(pd.read_csv(out/'training_example_draws.csv'))
    draws = pd.concat(draw_frames, ignore_index=True)
    usage = draws.subset_index.value_counts().reindex(range(16), fill_value=0)
    usage.rename_axis('subset_index').rename('actual_presentations_including_repeated_work').to_csv(out/'training_example_usage.csv')
    training = float(cost.loc[cost.stage=='training', 'seconds'].sum())
    evaluation = float(cost.loc[cost.stage=='official_validation_evaluation', 'seconds'].sum())
    steps = len(train)
    projection = training/steps*5000
    # The uncapped official train/validation cohorts are determined by the official loader.
    # This is only a throughput projection from the small cohort, not a runtime bound.
    raw_estimate = {'warm_small_pool_5000_step_projection_seconds': projection,
        'validation_seconds_per_example_projection': evaluation/32,
        'limitations': 'Warm small-pool training underestimates donor misses/pretokenization of uncapped reference. '
            'Sequence lengths, cache cap, allocator fragmentation and evaluation batch size change costs. '
            'Not a full-run VRAM bound or guaranteed runtime.'}
    write_json(out/'resource_estimates.json', raw_estimate)
    pd.DataFrame({'fraction': GRID, 'linear_cpr_weight': integration_weights(),
        'requested_edges': [int(x*32491) for x in GRID]}).to_csv(out/'cpr_grid_weights.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(train.step+1, train.loss, marker='.', lw=1)
    axes[0].set(xlabel='Optimizer step', ylabel='Sampled negative logit difference', title='Smoke training loss (varying k/prompts)')
    axes[1].plot(curve.fraction, curve.faithfulness, marker='o')
    axes[1].axhline(1, color='gray', linestyle='--')
    axes[1].set(xscale='log', xlabel='Requested retained edge fraction (log display)',
        ylabel='Official normalized faithfulness', title='32-example validation; smoke only')
    fig.tight_layout()
    fig.savefig(out/'gate0_diagnostics.png', dpi=160)
    plt.close(fig)
    text = f'''# MAttr optimization: Gate 0 baseline status

Campaign: `{out.name}`. Only Gate 0 executed; the full official 5,000-step edge baseline has **not** run. Earlier node-level results are not substituted for edge-level results.

## Implementation and execution

The adapter invokes the pinned official `scripts/mib/eval_mib_edge.py` and shared trainer. It preserves zero initialization, uniform k, sigmoid top-k, iso mode, Adam lr 0.05, epsilon 1e-8, betas (0.9, 0.999), T=0.5 and 30 bisection iterations. It adds deterministic official data subsets, source guards, actual computation counters, versioned score/Adam/RNG checkpoints and a hard-mask comparison with official MIB. A separate Python 3.12 environment was created from the existing official frozen uv lock/cache; no older environment was synced or changed.

Smoke settings explicitly differ from the full reference: seed 42, {steps} updates, 16 eligible train examples, 32 fixed validation examples, evaluation batch 2 and a 1 GB donor-cache budget. Training still samples one example with replacement per update. Train and validation selected clean prompts are distinct; selected IOI answer names are checked to be single tokens. No test split was loaded. All previously tracked artifacts were hashed before/after and preserved.

The hard-mask closure was compared with official MIB at four pruned graph states (0%, 10%, 50%, 100% requested retention), including simultaneous dependent removals: maximum loss discrepancy {patching.absolute_error.max():.8g}, tolerance atol=5e-5, rtol=1e-5. This validates canonical hard-mask execution; it does not claim a soft-mask oracle exists. No custom edge patcher is introduced.

## Measured smoke results

- Official validation CPR: **{metrics['cpr']:.6f}**.
- Official validation CMD: **{metrics['cmd']:.6f}** (lower means less deviation).
- Accuracy log-AUC: {metrics['accuracy_auc']:.6f}.
- Training: {training:.2f} seconds; official evaluation: {evaluation:.2f} seconds.
- Maximum sampled stage GPU allocation: {cost.peak_allocated_mib.max():.1f} MiB; reserved: {cost.peak_reserved_mib.max():.1f} MiB.
- Maximum sampled CPU RSS: {cost.sampled_peak_cpu_rss_mib.max():.1f} MiB.
- Actual sampled unique training rows: {diagnostics['actually_sampled_unique']}/16; presentations this attempt: {diagnostics['sampled_presentations_this_attempt']}.
- Actual forward batches: {int(cost.forward_batches.sum())}; example-forwards: {int(cost.example_forwards.sum())}; backward calls: {int(cost.backward_calls.sum())}; optimizer updates: {int(cost.optimizer_updates.sum())}.
- Model parameter elements requiring gradients: {diagnostics['model_parameters_requiring_grad']}; with retained gradients: {diagnostics['model_parameters_with_grad']}.

Each stage synchronizes CUDA for wall time. Inference and oracle evaluation use no-grad; training retains the live activations required for score gradients. Upstream also accumulates model-weight gradients although Adam only updates edge scores; this behavior was preserved, and any proposed freezing optimization requires a separate gradient-equivalence test. Counterfactual cache hits add no phantom forward calls. Unit-only artificial metric curves are never treated as benchmark measurements. Raw counters include patching checks separately from training and evaluation. Archived interrupted attempts, if present, are included in the complete accounting. On restart, donors are rebuilt; repeated forwards are charged to the new attempt.

These are feasibility measurements on a tiny cohort and short training budget. They do **not** establish improvement, convergence, replication of a published edge score, statistical significance, or leaderboard performance. CPR can exceed one through behavioral overshoot, so CMD and small-circuit faithfulness must be tracked alongside CPR.

The [paper's Table 4](https://arxiv.org/html/2609.25518v1) reports GPT-2/IOI edge validation CPR 10.59 for its 5,000-step recipe. Our short run is not a matched comparison with that published target. Full reference reproduction is still pending. Four metric/mask/resume unit tests passed; their log is saved independently of model results.

## Planning estimates and next gate

At the measured small-pool rate, 5,000 updates project to {projection/60:.1f} minutes of training. This is a warm-cache throughput projection, **not** an uncapped-reference prediction. The full official training pool creates more donor-cache misses and pretokenization work. Evaluation projects to {evaluation/32*10000/60:.1f} minutes per 10,000 validation examples at batch 2, before cohort-length effects. Full-run peak VRAM is unverified; a 4 GB donor cache can add memory beyond this 1 GB smoke setting. No settings will change automatically after OOM.

Next: Gate 1, one uncapped official 5,000-step edge baseline on seed 42, with documented memory controls and full evaluation accounting. Stop here before starting it. Gate 2 must then measure unchanged MAttr on nested 16/32/64/128 pools before any algorithm modification. No adaptive sampler, initialization, refinement, multi-model run or leaderboard submission has been executed.
'''
    (ROOT/'MATTR_BASELINE_REPORT.md').write_text(text, encoding='utf-8')
    (out/'MATTR_BASELINE_REPORT.md').write_text(text, encoding='utf-8')
    for name, body in {
        'MATTR_SAMPLE_EFFICIENCY_REPORT.md': 'Gate 2 has not run. The prior node-level scaling study does not answer edge-level data scaling. Required next protocol: paired nested train pools 16/32/64/128; fixed validation cohort; unchanged 5,000-step algorithm; seed 42 first, then 43/44 after feasibility. No sample-efficiency conclusions are available.',
        'MATTR_OPTIMIZATION_REPORT.md': 'Gates 3–5 have not run. No modifications have been compared and no improvement is claimed. First measure full edge baseline noise and data scaling; freeze success thresholds before testing sampling, then initialization, then optional refinement separately. Existing upstream AdaptiveLogK and hard_topk ablations must be acknowledged. Multi-model validation and leaderboard submission require explicit approval.'}.items():
        (ROOT/name).write_text('# '+name.removesuffix('.md').replace('_', ' ').title()+'\n\n'+body+'\n', encoding='utf-8')
