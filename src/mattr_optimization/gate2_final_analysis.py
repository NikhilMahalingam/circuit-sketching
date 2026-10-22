"""Final descriptive Gate-2 analysis and pre-variant Gate-3 protocol freeze."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .common import ROOT, sha, write_json


def table(frame, decimals=4):
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f'{x:.{decimals}f}'
        return str(x)
    lines = ['| '+' | '.join(map(str,frame.columns))+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
    lines.extend('| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaigns', type=Path, nargs='+', required=True)
    p.add_argument('--controls', type=Path, required=True)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out, a.controls, a.frozen = a.out.resolve(), a.controls.resolve(), a.frozen.resolve()
    a.campaigns = [path.resolve() for path in a.campaigns]
    a.out.mkdir(parents=True, exist_ok=True)
    qs,fs,us,cs,ss = [],[],[],[],[]
    manifests, jobs, hashes, provenances = {},{},{},[]
    for campaign in a.campaigns:
        for name in ('sample_efficiency.csv','quality_by_step.csv','faithfulness_curves.csv','training_example_usage.csv','compute_accounting.csv'):
            path = campaign/name
            hashes[str(path.relative_to(ROOT))] = sha(path)
        qs.append(pd.read_csv(campaign/'quality_by_step.csv'))
        fs.append(pd.read_csv(campaign/'faithfulness_curves.csv'))
        us.append(pd.read_csv(campaign/'training_example_usage.csv'))
        cs.append(pd.read_csv(campaign/'compute_accounting.csv'))
        ss.append(pd.read_csv(campaign/'sample_efficiency.csv'))
        for marker in sorted(campaign.glob('seed*_n*/completed.json')):
            job = marker.parent
            c = json.loads((job/'experiment_config.json').read_text())
            provenances.append(c['source_provenance'])
            key = (c['seed'],c['pool'])
            if key in jobs:
                raise ValueError('Duplicate pool/seed job.')
            saved = json.loads(marker.read_text())
            if not saved['passed'] or any(sha(job/n)!=h for n,h in saved['sha256'].items()):
                raise ValueError('Completed source artifacts changed.')
            jobs[key] = job
            manifests[key] = pd.read_csv(job/'validation_manifest.csv')
    if set(jobs)!=set(itertools.product((42,43,44),(16,32,64,128))):
        raise ValueError('Require all twelve unchanged Gate-2 jobs.')
    if any(prov != provenances[0] for prov in provenances):
        raise ValueError('Official sources, package versions or execution environment differ between jobs.')
    for frame in manifests.values():
        pd.testing.assert_frame_equal(frame,next(iter(manifests.values())))
    quality,faith,usage,cost,summary = [pd.concat(x,ignore_index=True) for x in (qs,fs,us,cs,ss)]
    frozen_marker = json.loads((a.frozen/'completed.json').read_text())
    if not frozen_marker.get('passed') or not frozen_marker.get('gate1_unchanged') or frozen_marker.get('training_updates') != 0:
        raise ValueError('Require completed frozen-only evaluations with an unchanged Gate 1 ranking.')
    extra = pd.read_csv(a.frozen/'quality.csv')
    train_extra = extra[extra.kind=='checkpoint_train_diagnostic'][quality.columns]
    q = pd.concat([quality,train_extra],ignore_index=True)
    if q.duplicated(['seed','pool','split','step']).any():
        raise ValueError('Duplicate quality measurements.')
    val = q[q.split=='validation'].copy()
    train = q[q.split=='train'][['seed','pool','step','cpr','cmd']]
    all_steps = val.merge(train,on=['seed','pool','step'],suffixes=('_validation','_train'),validate='one_to_one')
    all_steps['cpr_gap'] = all_steps.cpr_train-all_steps.cpr_validation
    all_steps['cmd_gap'] = all_steps.cmd_train-all_steps.cmd_validation
    # Cumulative times are checkpoint measurements, not linear extrapolations.
    import torch
    timing, reference_rows = [], []
    for (seed,pool),job in jobs.items():
        for split in ('train','validation'):
            refs = json.loads((job/f'evaluation/{split}_step5000/official_result.json').read_text())['references']
            reference_rows.append({'seed':seed,'pool':pool,'split':split,'clean_logit_difference':refs['baseline'],
                                   'empty_circuit_logit_difference':refs['corrupted'],
                                   'normalization_denominator':refs['baseline']-refs['corrupted']})
        if list((job/'attempts').glob('*')):
            raise ValueError('Checkpoint timing/unique-prefix analysis needs explicit resumed-attempt reconciliation.')
        draws = pd.read_csv(job/'training_example_draws.csv')
        if len(draws)!=5000:
            raise ValueError('Expected 5,000 actual presentations per original job.')
        actual_counts = draws.subset_index.value_counts().reindex(range(pool),fill_value=0).to_numpy()
        usage_counts = usage[(usage.seed==seed)&(usage.pool==pool)].sort_values('subset_index').actual_presentations.to_numpy()
        if not np.array_equal(actual_counts,usage_counts):
            raise ValueError('Reported per-example usage disagrees with actual draw logs.')
        for step in (500,1000,5000):
            path = job/f'checkpoints/step_{step:06d}.pt'
            state = torch.load(path,map_location='cpu',weights_only=False)
            timing.append({'seed':seed,'pool':pool,'step':step,'cumulative_training_seconds':state['training_seconds'],
                           'actual_unique_visited':draws.iloc[:step].subset_index.nunique(),
                           'actual_presentations':step})
    all_steps = all_steps.merge(pd.DataFrame(timing),on=['seed','pool','step'],validate='one_to_one').sort_values(['pool','seed','step'])
    if len(all_steps)!=36:
        raise ValueError('Missing train/validation checkpoint measurements.')
    all_steps.to_csv(a.out/'quality_all_steps.csv',index=False)
    cohort_references = pd.DataFrame(reference_rows)
    cohort_references.to_csv(a.out/'cohort_references.csv',index=False)
    total_usage = usage.groupby(['seed','pool'],as_index=False).agg(actual_unique_visited=('actual_presentations',lambda s:int((s>0).sum())),
             actual_presentations=('actual_presentations','sum'),min_reuse=('actual_presentations','min'),max_reuse=('actual_presentations','max'))
    total_usage['mean_reuse'] = total_usage.actual_presentations/total_usage.actual_unique_visited
    total_usage.to_csv(a.out/'example_usage.csv',index=False)
    detailed_cost = summary.merge(total_usage,on=['seed','pool'],validate='one_to_one')
    detailed_cost.to_csv(a.out/'compute_by_job.csv',index=False)
    q.to_csv(a.out/'all_quality.csv',index=False)
    faith.to_csv(a.out/'all_faithfulness_curves.csv',index=False)
    cost.to_csv(a.out/'all_compute_stages.csv',index=False)
    reference = extra[extra.kind=='full_data_reference'].iloc[0]
    ref_curve = pd.read_csv(a.frozen/'gate1_on_gate2_validation/faithfulness_curve.csv')
    paired = []
    for row in val.itertuples(index=False):
        n128 = val[(val.seed==row.seed)&(val.pool==128)&(val.step==row.step)].iloc[0]
        end = val[(val.seed==row.seed)&(val.pool==row.pool)&(val.step==5000)].iloc[0]
        paired.append({'seed':row.seed,'pool':row.pool,'step':row.step,
            'cpr_minus_full_data_reference':row.cpr-reference.cpr,'cmd_minus_full_data_reference':row.cmd-reference.cmd,
            'accuracy_auc_minus_full_data_reference':row.accuracy_auc-reference.accuracy_auc,
            'cpr_minus_n128_same_seed_step':row.cpr-n128.cpr,'cmd_minus_n128_same_seed_step':row.cmd-n128.cmd,
            'accuracy_auc_minus_n128_same_seed_step':row.accuracy_auc-n128.accuracy_auc,
            'cpr_5000_minus_this_step':end.cpr-row.cpr,'cmd_5000_minus_this_step':end.cmd-row.cmd,
            'accuracy_auc_5000_minus_this_step':end.accuracy_auc-row.accuracy_auc})
    paired = pd.DataFrame(paired).sort_values(['pool','seed','step'])
    paired.to_csv(a.out/'paired_baseline_differences.csv',index=False)
    stats = val.groupby(['pool','step']).agg(cpr_mean=('cpr','mean'),cpr_sd=('cpr','std'),cpr_min=('cpr','min'),cpr_max=('cpr','max'),
        cmd_mean=('cmd','mean'),cmd_sd=('cmd','std'),accuracy_auc_mean=('accuracy_auc','mean'),accuracy_auc_sd=('accuracy_auc','std')).reset_index()
    stats.to_csv(a.out/'descriptive_variability.csv',index=False)
    compact = []
    for (seed,pool,step),group in faith[faith.split=='validation'].groupby(['seed','pool','step']):
        d = {'seed':seed,'pool':pool,'step':step}
        for pct in (.001,.002,.005,.01,.02,.05):
            row = group[np.isclose(group.fraction,pct)].iloc[0]
            suffix = f'{pct*100:g}pct'
            d['faithfulness_'+suffix] = row.faithfulness
            d['accuracy_'+suffix] = row.accuracy
            d['surviving_edges_'+suffix] = row.weighted_edge_count
        compact.append(d)
    compact = pd.DataFrame(compact).sort_values(['pool','seed','step'])
    compact.to_csv(a.out/'compact_circuits.csv',index=False)
    # Separate conditional axes rather than pretending the diagonal seeds identify a variance decomposition.
    control_rows = []
    for marker in sorted(a.controls.glob('train*_subset*/completed.json')):
        job = marker.parent
        saved = json.loads(marker.read_text())
        if not saved.get('passed') or any(sha(job/n)!=h for n,h in saved['sha256'].items()):
            raise ValueError('Completed conditional-control artifacts changed.')
        job_config = json.loads((job/'experiment_config.json').read_text())
        if job_config['source_provenance'] != provenances[0]:
            raise ValueError('Conditional control has different official source/environment provenance.')
        config = job_config['campaign_protocol']
        training_seed,subset_seed = config['training_seed'],config['subset_seed']
        manifest = pd.read_csv(job/'validation_manifest.csv')
        pd.testing.assert_frame_equal(manifest,next(iter(manifests.values())))
        original_job = jobs[(subset_seed,128)]
        pd.testing.assert_frame_equal(pd.read_csv(job/'train_manifest.csv'),pd.read_csv(original_job/'train_manifest.csv'))
        official = json.loads((job/'upstream_command.json').read_text())
        if not official['unchanged_algorithm']:
            raise ValueError('Algorithm control changed.')
        for r in pd.read_csv(job/'quality.csv').query("split == 'validation'").itertuples():
            control_rows.append({'training_seed':training_seed,'subset_seed':subset_seed,'step':r.step,'cpr':r.cpr,'cmd':r.cmd,'accuracy_auc':r.accuracy_auc})
    if len(control_rows)!=12:
        raise ValueError('Require four completed conditional controls x three checkpoints.')
    for r in val[val.pool==128].itertuples():
        control_rows.append({'training_seed':r.seed,'subset_seed':r.seed,'step':r.step,'cpr':r.cpr,'cmd':r.cmd,'accuracy_auc':r.accuracy_auc})
    controls = pd.DataFrame(control_rows)
    anchor = controls[(controls.training_seed==42)&(controls.subset_seed==42)][['step','cpr','cmd','accuracy_auc']]
    controls = controls.merge(anchor,on='step',suffixes=('','_anchor'))
    controls['cpr_minus_anchor'] = controls.cpr-controls.cpr_anchor
    controls['cmd_minus_anchor'] = controls.cmd-controls.cmd_anchor
    controls['accuracy_auc_minus_anchor'] = controls.accuracy_auc-controls.accuracy_auc_anchor
    controls.to_csv(a.out/'conditional_seed_controls.csv',index=False)
    conditional = []
    for label,g in [('training randomness at fixed subset 42',controls[controls.subset_seed==42]),
                    ('subset variation at fixed training seed 42',controls[controls.training_seed==42]),
                    ('joint diagonal variation',controls[controls.training_seed==controls.subset_seed])]:
        for step,rows in g.groupby('step'):
            conditional.append({'axis':label,'step':step,'n':len(rows),'cpr_mean':rows.cpr.mean(),'cpr_sd':rows.cpr.std(ddof=1),
                                'cmd_mean':rows.cmd.mean(),'cmd_sd':rows.cmd.std(ddof=1),
                                'accuracy_auc_mean':rows.accuracy_auc.mean(),'accuracy_auc_sd':rows.accuracy_auc.std(ddof=1)})
    conditional = pd.DataFrame(conditional)
    conditional.to_csv(a.out/'conditional_variability.csv',index=False)
    final_conditional = conditional[conditional.step==5000].set_index('axis')
    subset_variation = final_conditional.loc['subset variation at fixed training seed 42']
    training_variation = final_conditional.loc['training randomness at fixed subset 42']
    interaction_rows = []
    for step in (500,1000,5000):
        step_rows = controls[controls.step==step].set_index(['training_seed','subset_seed'])
        for seed in (43,44):
            row = {'alternate_seed':seed,'step':step}
            for metric in ('cpr','cmd','accuracy_auc'):
                row[metric+'_interaction_contrast'] = (step_rows.loc[(seed,seed),metric]
                    -step_rows.loc[(seed,42),metric]-step_rows.loc[(42,seed),metric]+step_rows.loc[(42,42),metric])
            interaction_rows.append(row)
    seed_interactions = pd.DataFrame(interaction_rows)
    seed_interactions.to_csv(a.out/'seed_interaction_contrasts.csv',index=False)
    overlap = []
    for n in (16,32,64,128):
        for s,t in itertools.combinations((42,43,44),2):
            x = set(pd.read_csv(jobs[(s,n)]/'train_manifest.csv').official_filtered_index)
            y = set(pd.read_csv(jobs[(t,n)]/'train_manifest.csv').official_filtered_index)
            overlap.append({'pool':n,'seed_a':s,'seed_b':t,'shared_rows':len(x&y),'jaccard':len(x&y)/len(x|y)})
    pd.DataFrame(overlap).to_csv(a.out/'subset_overlap.csv',index=False)
    # Plots retain seed trajectories: SD over three runs is descriptive, not a confidence interval.
    fig,axs = plt.subplots(2,3,figsize=(15,9))
    for seed,g in val[val.step==5000].groupby('seed'):
        g = g.sort_values('pool')
        axs[0,0].plot(g.pool,g.cpr,'o-',label=f'seed {seed}')
        axs[0,1].plot(g.pool,g.cmd,'o-',label=f'seed {seed}')
    axs[0,0].axhline(reference.cpr,color='black',linestyle='--',label='Frozen full-data reference')
    axs[0,1].axhline(reference.cmd,color='black',linestyle='--',label='Frozen full-data reference')
    for n,g in stats.groupby('pool'):
        axs[0,2].errorbar(g.step,g.cpr_mean,yerr=g.cpr_sd,marker='o',capsize=3,label=f'N={n}')
    for n,g in all_steps.groupby('pool'):
        gaps=g.groupby('step').cpr_gap.mean()
        axs[1,0].plot(gaps.index,gaps.values,'o-',label=f'N={n}')
        time_g=g.groupby('step').agg(seconds=('cumulative_training_seconds','mean'),cpr=('cpr_validation','mean'))
        axs[1,1].plot(time_g.seconds/60,time_g.cpr,'o-',label=f'N={n}')
    for label,g in conditional.groupby('axis'):
        axs[1,2].plot(g.step,g.cpr_sd,'o-',label=label)
    for ax,(x,y) in zip(axs.flat,[('Discovery prompts','Validation CPR'),('Discovery prompts','Validation CMD (lower = closer to full behavior)'),
          ('Updates','Validation CPR; descriptive seed SD'),('Updates','Mean train minus validation CPR'),
          ('Cumulative training minutes','Validation CPR'),('Updates','Conditional CPR sample SD')]):
        ax.set(xlabel=x,ylabel=y);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(a.out/'gate2_comparison.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for n,g in faith[(faith.split=='validation')&(faith.step==5000)].groupby('pool'):
        means=g.groupby('fraction')[['faithfulness','accuracy']].mean()
        axs[0].plot(means.index,means.faithfulness,'o-',label=f'N={n}')
        axs[1].plot(means.index,means.accuracy,'o-',label=f'N={n}')
    for ax,y in zip(axs,['faithfulness','accuracy']):
        ax.plot(ref_curve.fraction,ref_curve[y],color='black',linestyle='--',label='Frozen full-data')
        ax.set(xscale='log',xlabel='Requested edge fraction',ylabel='Official '+y);ax.legend()
    axs[0].axhline(1,color='grey',linewidth=.6)
    fig.tight_layout();fig.savefig(a.out/'compact_faithfulness.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for label,g in conditional.groupby('axis'):
        axs[0].plot(g.step,g.cpr_sd,'o-',label=label)
        axs[1].plot(g.step,g.accuracy_auc_sd,'o-',label=label)
    for ax,y in zip(axs,['CPR sample SD','Accuracy log-AUC sample SD']):
        ax.set(xlabel='Optimizer updates',ylabel=y);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(a.out/'conditional_variability.png',dpi=180);plt.close(fig)
    # Conservative research recommendation: largest tested discovery pool; retain both short screen and official budget.
    n128=stats[stats.pool==128].set_index('step')
    gain=n128.loc[5000,'cpr_mean']-n128.loc[1000,'cpr_mean']
    speed=all_steps[(all_steps.pool==128)&(all_steps.step==5000)].cumulative_training_seconds.mean()/all_steps[(all_steps.pool==128)&(all_steps.step==1000)].cumulative_training_seconds.mean()
    final_stats = stats[stats.step==5000]
    best_cpr = int(final_stats.loc[final_stats.cpr_mean.idxmax(),'pool'])
    best_cmd = int(final_stats.loc[final_stats.cmd_mean.idxmin(),'pool'])
    gap_means = all_steps[all_steps.step==5000].groupby('pool').cpr_gap.mean()
    smallest_gap = int(gap_means.idxmin())
    cpr_cmd_correlation = val[val.step==5000][['cpr','cmd']].corr().iloc[0,1]
    criteria = {
      'gate':3,'status':'frozen_before_any_modified_sampling_test','model':'gpt2','task':'ioi','level':'edge',
      'discovery_pool':128,'nested_subset_seeds':[42,43,44],'training_seeds':[42,43,44],
      'validation_size':128,'validation_seed':42,'screen_steps':1000,'confirmation_steps':5000,
      'reference':'Unchanged uniform-k at the identical pool, seed, step budget and validation cohort; retain official full-data 5000-step Gate1 as external frozen reference.',
      'unchanged_defaults':{'optimizer':'adam','lr':.05,'eps':1e-8,'betas':[.9,.999],'T':.5,'n_iters':30,'masking':'topk','mode':'iso','initialization':'zeros','evaluation_batch':20,'donor_cache_gb':4.0},
      'primary_cpr_specific_success':{'mean_paired_relative_CPR_gain_at_5000_min':.02,'positive_seed_differences_min':2,'seed_count':3,
        'mean_compact_accuracy_drop_max_absolute':.02,'compact_fractions':[.005,.01,.02,.05],
        'mean_accuracy_log_auc_drop_max_absolute':.02,
        'mean_CMD_increase_max': 'max(0, mean paired CPR increase) + 0.05',
        'training_time_ratio_max':1.10,'peak_allocated_memory_ratio_max':1.10},
      'interpretation':'A CPR-specific success can amplify behavior and worsen CMD. Call it a general fidelity improvement only if mean paired CMD also decreases and compact accuracy guardrails pass. Three seeds are descriptive, not statistical significance.',
      'sample_efficiency_success_alternative':{'CPR_relative_loss_max':.01,'CMD_increase_max_absolute':.10,'compact_accuracy_drop_max_absolute':.02,'accuracy_log_auc_drop_max_absolute':.02,'total_training_compute_reduction_min':.20,
        'confirmation':'Compare candidate at 1000 with unchanged uniform at 5000 on identical subsets/seeds; charge adaptation/attribution overhead.'},
      'variant_budget':{'schedule_only':True,'mixture_lambdas':[.5,.25],'weight_exponent':1.0,'jitter':'Freeze exact distribution in a separate variant protocol before first run; no post-result tuning.',
                        'selection':'Screen both prespecified mixtures at 1000; carry at most one to 5000 using mean paired CPR gain, subject to guards. No tests in this task.'},
      'no_test_split_access':True,'no_leaderboard_claim':True,'gate3_executed':False,
      'validation_reuse':'Development cohort used for budget/variant selection; successful confirmation must later be checked on a disjoint official validation cohort before any generalization or leaderboard claim.'}
    freeze=a.out/'gate3_success_criteria.json'
    if freeze.exists():
        if json.loads(freeze.read_text())!=criteria:
            raise ValueError('Gate 3 criteria already frozen; do not change in place.')
    else:
        write_json(freeze,criteria)
    write_json(a.out/'analysis_provenance.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'source_csv_sha256':hashes,
         'criteria_sha256':sha(freeze),'analysis_source_sha256':sha(Path(__file__)),'gate3_executed':False})
    total_seconds = cost.seconds.sum()
    control_cost=[]
    for path in a.controls.glob('train*_subset*/**/compute_accounting.csv'):
        c=pd.read_csv(path);c['artifact']=str(path.relative_to(a.controls));control_cost.append(c)
    control_cost=pd.concat(control_cost,ignore_index=True)
    control_cost.to_csv(a.out/'control_compute.csv',index=False)
    frozen_cost=pd.concat([pd.read_csv(path) for path in a.frozen.rglob('compute_accounting.csv')],ignore_index=True)
    frozen_cost.to_csv(a.out/'frozen_evaluation_compute.csv',index=False)
    compact_show=compact[compact.step==5000][['seed','pool','faithfulness_0.5pct','faithfulness_1pct','faithfulness_2pct','faithfulness_5pct','accuracy_0.5pct','accuracy_1pct','accuracy_2pct','accuracy_5pct']]
    sections=[
      '# Gate 2 final analysis',
      'All results below are measured with pinned official edge-MAttr and the official seven-value MIB evaluator. No MAttr algorithm has been modified. No Gate 3 run, public/private test access or submission was performed. All twelve pool/seed jobs and four conditional controls use 5,000 updates; checkpoints are evaluated without optimizer feedback.',
      '## Findings and metric interpretation',
      f'The frozen full-data Gate 1 ranking on this exact 128-prompt validation cohort has CPR **{reference.cpr:.6f}**, CMD **{reference.cmd:.6f}**, accuracy log-AUC **{reference.accuracy_auc:.6f}**. This is the directly comparable reference; its original full-10,000-validation score is a separate result. No retraining was performed.',
      'CPR integrates normalized logit-difference recovery; it is unbounded and rewards amplification. CMD integrates absolute distance from faithfulness 1, so lower means closer to the original model. A rise in CPR together with CMD is a real tradeoff, not automatically better circuit fidelity. Scores are ratios of dataset means, not means of per-example ratios. We do not clip curves, invert signs or replace the official metrics.',
      f'At 5,000 updates, the highest three-seed mean CPR is at **N={best_cpr}**, the lowest mean CMD is at **N={best_cmd}**, and the smallest mean train–validation CPR gap is at **N={smallest_gap}**. Across the twelve final runs CPR and CMD have descriptive correlation **{cpr_cmd_correlation:.6f}**; this is a metric tradeoff diagnostic, not a significance test.',
      f'At N=128, the mean CPR increment from 1,000 to 5,000 updates is **{gain:.6f}**; the measured cumulative-training-time ratio is **{speed:.2f}x**. See the exact step and compact-circuit tables before interpreting this increment as worth its cost.',
      f'Mean N=128 accuracy log-AUC rises from **{n128.loc[1000,"accuracy_auc_mean"]:.6f}** at 1,000 updates to **{n128.loc[5000,"accuracy_auc_mean"]:.6f}** at 5,000. Thus a short run is useful for screening CPR, but cannot replace the longer compact-quality confirmation. At 500 updates the seed-43/44 N=128 runs had visited only 125/126 rows; all main runs had visited their full eligible pool by 1,000 updates.',
      '## All measured checkpoints',
      'Train scores at 500/1,000 are additional frozen-checkpoint evaluations. Every gap uses the train pool and the same validation cohort at that step. Times are actual cumulative checkpoint training timers; they exclude the subsequent official evaluation. Unique rows and presentations come from actual draw logs. The 5,000-step usage totals were cross-checked against training_example_usage.csv.',
      'Each split uses its own clean/empty-circuit reference denominator. Train–validation CPR gaps therefore combine generalization with cohort normalization differences and are not standalone proof of overfitting. The measured references are saved in cohort_references.csv; compact-circuit accuracy and behavior curves provide additional context.',
      table(all_steps[['seed','pool','step','cpr_validation','cmd_validation','cpr_train','cmd_train','cpr_gap','cmd_gap','actual_unique_visited','actual_presentations','cumulative_training_seconds']].rename(columns={
          'pool':'N','step':'updates','cpr_validation':'val CPR','cmd_validation':'val CMD','cpr_train':'train CPR','cmd_train':'train CMD',
          'cpr_gap':'CPR gap','cmd_gap':'CMD gap','actual_unique_visited':'unique rows','actual_presentations':'presentations','cumulative_training_seconds':'train seconds'})),
      '## Three-seed descriptive variability', table(stats),
      'Sample SD describes three coupled pool/training-seed runs. These are not confidence intervals, a significance test, or evidence that 128 prompts suffice for a leaderboard run.',
      '## Paired baseline differences',
      'Subtractions below share exactly the same validation prompts. The full-data ranking is one frozen seed-42 reference, so its contrasts are paired on evaluation data, not on independent full-data training repetitions. N=128 contrasts match nominal training seed and checkpoint. The 5,000-minus-earlier contrasts hold the discovery pool and optimizer trajectory fixed.',
      table(paired.rename(columns={'pool':'N','cpr_minus_full_data_reference':'ΔCPR full','cmd_minus_full_data_reference':'ΔCMD full',
          'accuracy_auc_minus_full_data_reference':'ΔaccAUC full','cpr_minus_n128_same_seed_step':'ΔCPR N128',
          'cmd_minus_n128_same_seed_step':'ΔCMD N128','accuracy_auc_minus_n128_same_seed_step':'ΔaccAUC N128',
          'cpr_5000_minus_this_step':'CPR gain to 5k','cmd_5000_minus_this_step':'CMD change to 5k','accuracy_auc_5000_minus_this_step':'accAUC gain to 5k'})),
      '## Separating subset and training variation',
      'Four additional unchanged N=128 controls hold subset seed 42 fixed while training seeds vary (42/43/44), or hold training seed 42 fixed while subset seeds vary. Diagonal results are reused from the main experiment. Thus conditional variation is measured separately rather than attributed from confounded diagonal seeds.',
      f'At 5,000 steps, subset-seed versus training-seed conditional CPR SD is **{subset_variation.cpr_sd:.6f} versus {training_variation.cpr_sd:.6f}**. Corresponding accuracy-log-AUC SD is **{subset_variation.accuracy_auc_sd:.6f} versus {training_variation.accuracy_auc_sd:.6f}**. This separates the measured sensitivity of margin recovery from compact decision recovery at this anchor configuration.',
      table(controls.sort_values(['step','subset_seed','training_seed'])), table(conditional),
      'The two complete 2x2 anchor blocks permit signed interaction contrasts: F(seed,seed) − F(seed,42) − F(42,seed) + F(42,42). These measured contrasts show departures from additive seed effects; they are descriptive, not significance tests.',
      table(seed_interactions),
      'The conditional SDs are not an additive decomposition of total variance: subset–training interactions are present, two off-axis cells of the 3x3 design are unmeasured, and only three values lie on each axis. The controls are conditional at N=128; they do not establish the decomposition at smaller N. Data-subset variation includes which prompt is drawn by each training RNG index. Source-index overlap is saved in subset_overlap.csv. No statistical-significance claim is made.',
      '## Compact circuits at 5,000 steps',
      'Fractions are requested ranking cutoffs, followed by official connectivity pruning. The CSV includes actual surviving edge counts and all 0.1/0.2/0.5/1/2/5% checkpoints; complete ten-point curves at every step are retained. High faithfulness above 1 is amplification, while accuracy reports the fraction with positive IOI logit difference.',
      table(compact_show),
      'Frozen full-data reference curve:', table(ref_curve),
      '## Reuse and compute', table(total_usage.sort_values(['pool','seed'])),
      table(detailed_cost[['seed','pool','training_seconds','measured_stage_seconds','forward_batches','example_forwards','peak_allocated_mib']].sort_values(['pool','seed'])),
      f'Twelve main jobs: **{total_seconds/3600:.3f} h** of measured stages, including the preexisting seed-42 campaign. Four conditional controls: **{control_cost.seconds.sum()/3600:.3f} h**. Frozen Gate 1 reference plus additional train diagnostics: **{frozen_cost.seconds.sum()/60:.3f} min**, including their model/data loading. These scopes are separate and must be added for total study cost. Process startup, interstage gaps and analysis/plot rendering are additional overhead. Actual forward batches/example forwards and measured allocation peaks are in the CSVs; elapsed time is CUDA-synchronized. Cached evaluations do not count as forwards.',
      '## Proposed Gate 3 configuration and frozen decision rules',
      'Use **128 discovery prompts**, the same nested subsets and seeds 42/43/44, and the fixed 128-prompt development cohort. This is the largest tested small-data configuration, not a claim of sufficiency. Screen at **1,000 updates**, then confirm at the official **5,000-update** budget. Preserve the official full-10,000-training configuration as the reference and do not replace its label with a small-data result. Keep the unchanged uniform-k counterpart at every matched budget.',
      'Before a variant is run, success criteria are frozen in gate3_success_criteria.json: a CPR-specific confirmation needs mean paired CPR improvement of at least 2%, positive CPR differences in at least two of three seeds, no mean accuracy loss above 0.02 at any of the 0.5/1/2/5% compact cutoffs or in accuracy log-AUC, CMD increase no greater than the positive CPR increase plus 0.05, and at most 10% training-time and allocated-memory overhead. Report the paired CMD change even when this gate passes; call it a general fidelity improvement only if CMD also decreases. This explicitly prevents interpreting amplification alone as better fidelity.',
      'An alternative efficiency success needs at least 20% less total training/adaptation cost, CPR loss at most 1%, CMD increase at most 0.10 and compact-accuracy/accuracy-log-AUC loss at most 0.02 versus the matched 5,000-step uniform baseline. Compare the 1,000-step candidate explicitly against that baseline. Candidate-screen overhead and all failed trials remain charged in research compute, even if a deployable run is cheaper.',
      'At most two prespecified schedule-only mixtures (lambda 0.5 and 0.25, weight exponent 1) may be screened later; carry at most one to confirmation. Freeze the exact sampling/jitter distribution in a separate variant protocol before testing it. No initialization or refinement change is permitted in that comparison. Any apparent winner is exploratory after development-cohort reuse and requires a disjoint official validation check before a generalization/leaderboard claim.',
      f'Criteria SHA-256: `{sha(freeze)}`. **Gate 3 has not been launched.**',
      '## Plots', '![Gate 2 comparisons](gate2_comparison.png)', '![Compact faithfulness and accuracy](compact_faithfulness.png)',
      '![Conditional seed variability](conditional_variability.png)']
    audit_path=a.out/'final_validation.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        if not audit['passed'] or audit['criteria_sha256']!=sha(freeze):
            raise ValueError('Artifact audit or frozen-criteria identity failed.')
        totals=audit['study_totals_including_preexisting_seed42']
        sections.extend(['## Artifact audit',
          f'Passed: {audit["main_jobs"]} main jobs, {audit["conditional_controls"]} conditional controls, '
          f'{audit["frozen_vector_files"]} frozen-evaluation vector files and {audit["protected_files_unchanged"]} unchanged protected baseline files. '
          'The Gate 2 study, including the preexisting seed-42 campaign, used '
          f'{totals["forward_batches"]:,} real forward batches, {totals["example_forwards"]:,} example-forwards and '
          f'{totals["optimizer_updates"]:,} optimizer updates. Frozen evaluation had no gradients/backward calls, '
          'and the audit itself executed zero model forwards. Nested subsets, disjoint train/validation prompts, identical '
          'validation manifests, raw vector hashes and the frozen Gate 3 criteria hash were checked. '
          'See final_validation.json and complete_study_accounting.csv in the result directory.'])
    text='\n\n'.join(sections)+'\n'
    (a.out/'GATE2_FINAL_ANALYSIS.md').write_text(text,encoding='utf-8')
    # Root report links resolve to the retained result artifacts.
    relative=str(a.out.relative_to(ROOT)).replace('\\','/')
    root_text=text
    for name in ('gate2_comparison.png','compact_faithfulness.png','conditional_variability.png'):
        root_text=root_text.replace(f']({name})',f']({relative}/{name})')
    (ROOT/'GATE2_FINAL_ANALYSIS.md').write_text(root_text,encoding='utf-8')
    print('Final analysis complete; Gate 3 criteria frozen; no Gate 3 model execution.')


if __name__ == '__main__':
    main()
