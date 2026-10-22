"""Analyze completed scaling jobs; all quality values come from official MIB."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

from .common import ROOT


def generate(out):
    out = Path(out)
    quality, costs, usage, faith = [], [], [], []
    manifests = {}
    for marker in sorted(out.glob('seed*_n*/completed.json')):
        job = marker.parent
        config = json.loads((job/'experiment_config.json').read_text())
        seed, pool = config['seed'], config['pool']
        train = pd.read_csv(job/'train_manifest.csv')
        validation = pd.read_csv(job/'validation_manifest.csv')
        ids = validation.official_filtered_index.tolist()
        prompts = validation[['clean_prompt', 'corrupted_prompt', 'correct_token', 'incorrect_token']].to_dict('records')
        validation_id = json.dumps({'ids': ids, 'rows': prompts}, sort_keys=True)
        if manifests and validation_id != next(iter(manifests.values())):
            raise ValueError('Validation cohort differs between scaling jobs.')
        manifests[(seed,pool)] = validation_id
        q = pd.read_csv(job/'quality.csv')
        quality.append(q)
        attempts = sorted((job/'attempts').glob('*/compute_accounting.csv'))+[job/'compute_accounting.csv']
        c = pd.concat([pd.read_csv(p) for p in attempts], ignore_index=True)
        c['seed'], c['pool'] = seed, pool
        # A force-killed process can lack its final timers: expose incomplete cost.
        incomplete = [p for p in (job/'attempts').glob('*') if not (p/'process_summary.json').exists()]
        c['cost_may_be_incomplete'] = bool(incomplete)
        costs.append(c)
        draw_paths = sorted((job/'attempts').glob('*/training_example_draws.csv'))+[job/'training_example_draws.csv']
        draws = pd.concat([pd.read_csv(p) for p in draw_paths], ignore_index=True)
        presentations = draws.subset_index.value_counts().reindex(range(pool), fill_value=0)
        frame = pd.DataFrame({'subset_index': range(pool), 'actual_presentations': presentations.to_numpy(),
            'official_filtered_index': train.official_filtered_index.to_numpy(), 'seed': seed, 'pool': pool})
        frame.to_csv(job/'training_example_usage.csv', index=False)
        usage.append(frame)
        for file in sorted((job/'evaluation').glob('*/faithfulness_curve.csv')):
            curve = pd.read_csv(file)
            split, step = file.parent.name.split('_step')
            curve['split'], curve['step'], curve['seed'], curve['pool'] = split, int(step), seed, pool
            faith.append(curve)
    if not quality:
        print('No completed Gate-2 jobs to summarize.')
        return
    q = pd.concat(quality, ignore_index=True)
    c = pd.concat(costs, ignore_index=True)
    f = pd.concat(faith, ignore_index=True)
    u = pd.concat(usage, ignore_index=True)
    q.to_csv(out/'quality_by_step.csv', index=False)
    c.to_csv(out/'compute_accounting.csv', index=False)
    f.to_csv(out/'faithfulness_curves.csv', index=False)
    u.to_csv(out/'training_example_usage.csv', index=False)
    final = q[(q.split=='validation') & (q.step==5000)].copy()
    training = q[(q.split=='train') & (q.step==5000)][['seed','pool','cpr','cmd']]
    summary = final.merge(training, on=['seed','pool'], suffixes=('_validation','_train'))
    summary['train_minus_validation_cpr'] = summary.cpr_train-summary.cpr_validation
    time_rows = c.groupby(['seed','pool'], as_index=False).agg(
        measured_stage_seconds=('seconds','sum'), forward_batches=('forward_batches','sum'),
        example_forwards=('example_forwards','sum'), peak_allocated_mib=('peak_allocated_mib','max'))
    summary = summary.merge(time_rows, on=['seed','pool'])
    training_cost = c[c.stage=='training'].groupby(['seed','pool'], as_index=False).seconds.sum().rename(columns={'seconds':'training_seconds'})
    summary = summary.merge(training_cost, on=['seed','pool'])
    summary.to_csv(out/'sample_efficiency.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for seed, rows in summary.groupby('seed'):
        rows = rows.sort_values('pool')
        axes[0,0].plot(rows.pool, rows.cpr_validation, marker='o', label=f'seed {seed}')
        axes[0,2].plot(rows.pool, rows.cpr_train, marker='o', label=f'train seed {seed}')
        axes[0,2].plot(rows.pool, rows.cpr_validation, marker='x', linestyle='--', label=f'val seed {seed}')
        axes[1,0].plot(rows.pool, rows.training_seconds/60, marker='o', label=f'seed {seed}')
    for (seed,pool), rows in q[q.split=='validation'].groupby(['seed','pool']):
        axes[0,1].plot(rows.step, rows.cpr, marker='o', label=f'N={pool}, seed={seed}')
    for (seed,pool), rows in f[(f.split=='validation') & (f.step==5000)].groupby(['seed','pool']):
        axes[1,1].plot(rows.fraction, rows.faithfulness, marker='.', label=f'N={pool}, seed={seed}')
    axes[1,1].set_xscale('log')
    gaps = summary.groupby('pool').train_minus_validation_cpr.mean()
    axes[1,2].plot(gaps.index, gaps.values, marker='o')
    labels = [('Discovery pool size','Official validation CPR'), ('Optimizer iterations','Official validation CPR'),
        ('Discovery pool size','Official CPR'), ('Discovery pool size','Training minutes (incl. checkpoints)'),
        ('Requested edge fraction (log display)','Official faithfulness'), ('Discovery pool size','Mean train − validation CPR')]
    for ax,(x,y) in zip(axes.flat, labels):
        ax.set(xlabel=x, ylabel=y)
        if ax is not axes[1,2]: ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out/'sample_efficiency.png', dpi=160)
    plt.close(fig)
    seeds = sorted(final.seed.unique().tolist())
    text = f'''# MAttr edge discovery-data scaling

Campaign: `{out.name}`. Completed jobs: {len(summary)}. Seeds: {seeds}. Each job preserves the official 5,000-iteration edge recipe; only eligible discovery-pool size and preset seed differ. Training samples with replacement; per-example reuse counts and actually visited unique rows are saved.

All jobs use the same deterministic official validation cohort (seed 42). Pool subsets are nested within seed. Validation metrics are from the official MIB evaluator; training quality is diagnostic and never updates scores. Frozen rankings at 500, 1,000 and 5,000 iterations are evaluated after training, without perturbing the optimizer trajectory. No test split or algorithm variant is used. These cohort scores are not directly compared with Gate 1's full 10,000-example validation score.

`sample_efficiency.csv` contains final train/validation CPR and CMD, gaps and measured compute. `quality_by_step.csv`, `faithfulness_curves.csv`, `compute_accounting.csv` and `training_example_usage.csv` retain the detailed measurements. Model/data/evaluation stage timing includes real work and archived measured attempts. Interpreter startup, source verification, report rendering and gaps between stages are additional overhead; process summaries retain broader per-job elapsed time. Force-killed attempts without terminal timers are flagged, so their reported time must be treated as a lower bound.

One seed is exploratory. Even three seeds provide preliminary evidence only. No automatic success threshold, best-pool selection, algorithm modification or claim that 128 examples are sufficient is made by this report. Compare CPR, CMD, compact-circuit faithfulness and variability before deciding the next experiment. Gate 3 is not dispatched.
'''
    (out/'MATTR_SAMPLE_EFFICIENCY_REPORT.md').write_text(text, encoding='utf-8')
    (ROOT/'MATTR_SAMPLE_EFFICIENCY_REPORT.md').write_text(text, encoding='utf-8')
