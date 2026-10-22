"""Frozen Gate-3 gates and paired result report. No model forwards here."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .common import ROOT,GRID,sha,write_json
from .gate2_final_analysis import table
from .gate3 import baseline,verify_criteria,CRITERIA_SHA


def costs(job):
    files=sorted((job/'attempts').glob('*/compute_accounting.csv'))+[job/'compute_accounting.csv']
    frames=[pd.read_csv(p) for p in files]
    c=pd.concat(frames,ignore_index=True)
    if not c.success.all():raise ValueError('Failed attempts present: report explicitly before interpreting gates.')
    incomplete=[p for p in (job/'attempts').glob('*') if not (p/'process_summary.json').exists()]
    if incomplete:raise ValueError('Incomplete timing audit; do not claim compute success.')
    return {'training_seconds':float(c[c.stage=='training'].seconds.sum()),'stage_seconds':float(c.seconds.sum()),
            'peak_allocated_mib':float(c.peak_allocated_mib.max()),'forward_batches':int(c.forward_batches.sum()),
            'example_forwards':int(c.example_forwards.sum()),'backward_calls':int(c.backward_calls.sum()),
            'optimizer_updates':int(c.optimizer_updates.sum()),
            'launch_seconds':sum(json.loads(p.read_text())['seconds'] for p in job.glob('launch_elapsed_*.json'))}


def baseline_row(seed,step):
    folder=baseline(seed)
    q=pd.read_csv(folder/'quality.csv').query("split == 'validation' and step == @step").iloc[0]
    curve=pd.read_csv(folder/f'evaluation/validation_step{step}/faithfulness_curve.csv')
    import torch
    state=torch.load(folder/f'checkpoints/step_{step:06d}.pt',map_location='cpu',weights_only=False)
    c=pd.read_csv(folder/'compute_accounting.csv')
    row={'seed':seed,'step':step,'cpr':float(q.cpr),'cmd':float(q.cmd),'accuracy_auc':float(q.accuracy_auc),
         'training_seconds':float(state['training_seconds']),'peak_allocated_mib':float(c.peak_allocated_mib.max())}
    for fraction in (.005,.01,.02,.05):
        r=curve[np.isclose(curve.fraction,fraction)].iloc[0]
        row[f'accuracy_{fraction}']=float(r.accuracy)
        row[f'faithfulness_{fraction}']=float(r.faithfulness)
    return row


def passes(rows,criteria,efficiency=False):
    """All numeric limits come from the original frozen JSON; no tuning here."""
    p=criteria['sample_efficiency_success_alternative' if efficiency else 'primary_cpr_specific_success']
    if efficiency:
        checks={'cpr_noninferiority':rows.relative_cpr_gain.mean()>=-p['CPR_relative_loss_max'],
                'cmd_noninferiority':rows.cmd_delta.mean()<=p['CMD_increase_max_absolute'],
                'compute_saving':rows.training_seconds.mean()/rows.baseline_training_seconds.mean()<=1-p['total_training_compute_reduction_min']}
        acc=p['compact_accuracy_drop_max_absolute'];auc=p['accuracy_log_auc_drop_max_absolute']
    else:
        checks={'cpr_gain':rows.relative_cpr_gain.mean()>=p['mean_paired_relative_CPR_gain_at_5000_min'],
                'positive_seeds':int((rows.cpr_delta>0).sum())>=p['positive_seed_differences_min'],
                'cmd_guard':rows.cmd_delta.mean()<=max(0.,rows.cpr_delta.mean())+.05,
                'training_time_guard':rows.training_seconds.mean()/rows.baseline_training_seconds.mean()<=p['training_time_ratio_max'],
                'memory_guard':rows.peak_allocated_mib.mean()/rows.baseline_peak_allocated_mib.mean()<=p['peak_allocated_memory_ratio_max']}
        # Assert the frozen textual expression before evaluating its constants.
        if p['mean_CMD_increase_max']!='max(0, mean paired CPR increase) + 0.05':raise ValueError('Unknown frozen CMD expression.')
        acc=p['mean_compact_accuracy_drop_max_absolute'];auc=p['mean_accuracy_log_auc_drop_max_absolute']
    checks['accuracy_auc_guard']=rows.accuracy_auc_delta.mean()>=-auc
    for f in criteria['primary_cpr_specific_success']['compact_fractions']:
        checks[f'compact_accuracy_{f}']=rows[f'accuracy_delta_{f}'].mean()>=-acc
    return {'passed':all(bool(v) for v in checks.values()),'checks':{k:bool(v) for k,v in checks.items()}}


def comparison(candidate,base,resource):
    row={**candidate,'baseline_cpr':base['cpr'],'baseline_cmd':base['cmd'],'baseline_accuracy_auc':base['accuracy_auc'],
         'cpr_delta':candidate['cpr']-base['cpr'],'relative_cpr_gain':(candidate['cpr']-base['cpr'])/base['cpr'],
         'cmd_delta':candidate['cmd']-base['cmd'],'accuracy_auc_delta':candidate['accuracy_auc']-base['accuracy_auc'],
         'baseline_training_seconds':resource['training_seconds'],'baseline_peak_allocated_mib':resource['peak_allocated_mib']}
    for f in (.005,.01,.02,.05):row[f'accuracy_delta_{f}']=candidate[f'accuracy_{f}']-base[f'accuracy_{f}']
    return row


def generate(out):
    started=time.perf_counter();out=Path(out).resolve();criteria=verify_criteria()
    if sha(out/'gate3_success_criteria.json')!=CRITERIA_SHA:raise ValueError('Copied frozen criteria changed.')
    protocol=json.loads((out/'gate3_protocol.json').read_text())
    if sha(out/'gate3_protocol.json')!=(out/'gate3_protocol.sha256').read_text().strip():raise ValueError('Sampling protocol hash changed.')
    quality=[];curves=[];compute=[]
    for job in sorted(out.glob('*_seed*')):
        if not job.is_dir():continue
        config=json.loads((job/'experiment_config.json').read_text())
        for path in sorted(job.glob('quality_*.json')):
            q=json.loads(path.read_text());step=q['step']
            marker=job/f'completed_{step}.json'
            if not marker.exists():continue
            saved=json.loads(marker.read_text())
            if not saved['passed'] or any(sha(job/n)!=h for n,h in saved['sha256'].items()):raise ValueError('Completed artifacts changed.')
            curve=pd.read_csv(job/f'evaluation_step{step}/faithfulness_curve.csv')
            for f in (.005,.01,.02,.05):
                r=curve[np.isclose(curve.fraction,f)].iloc[0]
                q[f'accuracy_{f}']=float(r.accuracy);q[f'faithfulness_{f}']=float(r.faithfulness)
            q['job']=job.name
            # Current total includes screen+continuation. Recover screen cost from
            # first archived attempt for its separate comparison after confirmation.
            if step==1000 and (job/'completed_5000.json').exists():
                original=sorted((job/'attempts').glob('*/compute_accounting.csv'))
                if len(original)!=1:raise ValueError('Screen/continuation cost needs explicit reconciliation.')
                c=pd.read_csv(original[0])
                resource={'training_seconds':float(c[c.stage=='training'].seconds.sum()),'stage_seconds':float(c.seconds.sum()),
                          'peak_allocated_mib':float(c.peak_allocated_mib.max()),'forward_batches':int(c.forward_batches.sum()),
                          'example_forwards':int(c.example_forwards.sum()),'optimizer_updates':int(c.optimizer_updates.sum()),
                          'backward_calls':int(c.backward_calls.sum()),'launch_seconds':json.loads((job/'launch_elapsed_1000.json').read_text())['seconds']}
            else:resource=costs(job)
            q.update(resource);quality.append(q)
            curve['seed']=q['seed'];curve['lambda']=q['lambda'];curve['step']=step;curves.append(curve)
        compute.append({'job':job.name,'lambda':config['lambda'],'seed':config['seed'],**costs(job)})
    q=pd.DataFrame(quality)
    if len(q[q.step==1000])!=9:raise ValueError('Require all three uniforms and six variant screens.')
    f=pd.concat(curves,ignore_index=True);c=pd.DataFrame(compute)
    paired=[];efficiency=[]
    for row in q[q['lambda']!=1].to_dict('records'):
        seed,step=row['seed'],row['step'];b=baseline_row(seed,step)
        # Current controls are only for the short screen; 5k uses actual Gate2 training timers.
        resource=q[(q['lambda']==1)&(q.seed==seed)&(q.step==1000)].iloc[0].to_dict() if step==1000 else b
        paired.append(comparison(row,b,resource))
        if step==1000:
            full=baseline_row(seed,5000)
            efficiency.append(comparison(row,full,full))
    paired=pd.DataFrame(paired);eff=pd.DataFrame(efficiency)
    screen=paired[paired.step==1000]
    decisions={}
    for lam in (.5,.25):
        g=screen[screen['lambda']==lam]
        primary=passes(g,criteria)
        fast=passes(eff[eff['lambda']==lam],criteria,True)
        decisions[str(lam)]={'provisional_primary_1000':primary,'efficiency_1000_vs_5000':fast,
                            'eligible':primary['passed'] or fast['passed'],'mean_paired_relative_cpr_gain':float(g.relative_cpr_gain.mean())}
    eligible=[lam for lam in (.5,.25) if decisions[str(lam)]['eligible']]
    winner=max(eligible,key=lambda lam:(decisions[str(lam)]['mean_paired_relative_cpr_gain'],lam)) if eligible else None
    decision={'criteria_sha256':CRITERIA_SHA,'winner':winner,'screen_decisions':decisions,'selection_rule':protocol['screen_rule']}
    decision_path=out/'screen_decision.json'
    if decision_path.exists() and json.loads(decision_path.read_text())!=decision:raise ValueError('Screen decision changed after selection.')
    write_json(decision_path,decision)
    final=None
    confirmed=paired[paired.step==5000]
    if len(confirmed):
        if winner is None or len(confirmed)!=3 or set(confirmed['lambda'])!={winner}:raise ValueError('Unexpected confirmation sweep.')
        final=passes(confirmed,criteria)
        final['general_model_fidelity_improvement']=bool(final['passed'] and confirmed.cmd_delta.mean()<0)
    write_json(out/'success_assessment.json',{'screen':decision,'confirmation':final,'gate4_executed':False,'leaderboard_submitted':False})
    q.to_csv(out/'all_results.csv',index=False);paired.to_csv(out/'paired_results.csv',index=False)
    eff.to_csv(out/'efficiency_comparisons.csv',index=False);f.to_csv(out/'faithfulness_curves.csv',index=False)
    c.to_csv(out/'compute_accounting.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(14,4.5))
    for lam,g in screen.groupby('lambda'):
        axs[0].plot(g.seed,g.cpr_delta,'o-',label=f'lambda={lam}')
        axs[1].plot(g.seed,g.accuracy_auc_delta,'o-',label=f'lambda={lam}')
        means=f[(f['lambda']==lam)&(f.step==1000)].groupby('fraction').faithfulness.mean()
        axs[2].plot(means.index,means.values,'o-',label=f'lambda={lam}')
    uniform=f[(f['lambda']==1)&(f.step==1000)].groupby('fraction').faithfulness.mean()
    axs[2].plot(uniform.index,uniform.values,'k--',label='official uniform')
    for ax,label in zip(axs,['Paired CPR change vs Gate2 1000','Paired accuracy log-AUC change','Official faithfulness (mean of three seeds)']):
        ax.set(ylabel=label);ax.legend(fontsize=8)
    for ax in axs[:2]:ax.set(xlabel='Paired data/training seed',xticks=[42,43,44]);ax.axhline(0,color='grey',linewidth=.6)
    axs[2].set(xlabel='Requested edge fraction',xscale='log')
    fig.tight_layout();fig.savefig(out/'gate3_screen.png',dpi=180);plt.close(fig)
    summary=screen.groupby('lambda').agg(cpr_delta=('cpr_delta','mean'),relative_cpr_gain=('relative_cpr_gain','mean'),
         cmd_delta=('cmd_delta','mean'),accuracy_auc_delta=('accuracy_auc_delta','mean'),training_seconds=('training_seconds','mean'),
         baseline_training_seconds=('baseline_training_seconds','mean'),peak_allocated_mib=('peak_allocated_mib','mean'),
         baseline_peak_allocated_mib=('baseline_peak_allocated_mib','mean')).reset_index()
    summary['training_ratio']=summary.training_seconds/summary.baseline_training_seconds
    summary['memory_ratio']=summary.peak_allocated_mib/summary.baseline_peak_allocated_mib
    summary.to_csv(out/'screen_summary.csv',index=False)
    status=('Both mixtures failed the prespecified screening gates. Negative result: no confirmation, no added hyperparameters, no Gate 4.' if winner is None else
            f'Only lambda={winner} selected by the frozen screen rule. '+('Confirmation complete.' if final is not None else 'Only the prespecified 5000-step confirmation remains.'))
    text='\n\n'.join([
       '# Gate 3 CPR-aware sparsity sampling',status,
       f'Original Gate2 criteria SHA-256: `{CRITERIA_SHA}`. Frozen sampling protocol SHA-256: `{sha(out/"gate3_protocol.json")}`.',
       '## Protocol and invariants',
       'Official trainer k_sampler hook only; edge patching, optimizer, mask differentiation and official MIB evaluation sources remain pinned and unmodified. This is a static CPR-aware mixture, not an online loss-adaptive method. N=128 paired discovery/training seeds 42/43/44; unchanged seed-42 validation cohort of 128 prompts. No public/private test or leaderboard access.',
       (out/'GATE3_SAMPLING_PROTOCOL.md').read_text(),
       'Three fresh uniform timing controls reproduced Gate2 score tensors bitwise, as well as row/k draws and official faithfulness/accuracy (numeric tolerance 1e-6 for faithfulness). Quality comparisons use the original Gate2 checkpoints; fresh controls provide matched cold-reference timing and memory. Baseline replay work is charged in research compute.',
       '## 1000-step paired comparisons',table(screen[['lambda','seed','cpr','baseline_cpr','cpr_delta','relative_cpr_gain','cmd','baseline_cmd','cmd_delta','accuracy_auc','baseline_accuracy_auc','accuracy_auc_delta']]),
       'Positive CPR changes are better for the published CPR objective. Positive CMD changes mean worse reproduction of full-model margins. Accuracy/log-AUC and the 0.5/1/2/5% compact guardrails are evaluated independently; amplification alone is not general fidelity improvement.',
       '## Screening means and resource overhead',table(summary),
       '## All frozen screening checks', '```json\n'+json.dumps(decisions,indent=2)+'\n```',
       'At 1000 steps the primary numerical thresholds are conservative provisional screening gates, not a claim of passing the frozen 5000-step criterion. The frozen alternative compares 1000-step candidates with 5000-step uniforms, charging sampling overhead. Winner selection uses only the frozen rule. Three paired seeds are descriptive, not statistical significance.',
       '## Compact decisions',table(screen[['lambda','seed']+[f'accuracy_delta_{x}' for x in (.005,.01,.02,.05)]]),
       'Complete un-clipped faithfulness/accuracy curves and post-pruning edge counts are in faithfulness_curves.csv; per-job rankings, exact scores, all optimizer logs, k draws, example draws and gradients are retained.',
       '## 5000-step confirmation', 'Not run: no screened winner.' if winner is None else ('Pending.' if final is None else table(confirmed[['lambda','seed','cpr','baseline_cpr','cpr_delta','relative_cpr_gain','cmd_delta','accuracy_auc_delta','training_seconds','baseline_training_seconds']])+'\n\n```json\n'+json.dumps(final,indent=2)+'\n```'),
       '## Compute',table(c),
       f'Actual sum of job end-to-end timers: **{c.launch_seconds.sum()/60:.3f} min**; synchronized model/data/training/evaluation stages: **{c.stage_seconds.sum()/60:.3f} min**. '
       f'Real root forward batches: **{c.forward_batches.sum():,}**; example-forwards: **{c.example_forwards.sum():,}**; optimizer updates/backwards: **{c.optimizer_updates.sum():,}/{c.backward_calls.sum():,}**. '
       'All baseline replays, both screen variants and continuation restarts are charged. CPU tests, protocol preparation and report rendering are additional overhead; report timing is recorded separately. Cached completed jobs execute no forwards. Model allocation peaks are torch.cuda.max_memory_allocated, not physical total GPU occupancy.',
       'Confirmation training cost, if run, includes both the 1000-step prefix and actual continuation (with rebuilt donor cache); full-data Gate1 remains the unchanged reference. Gate2 5000-step training timers are used for confirmation, whose hardware/run-order timing variability must be kept in mind. Screen training/peak overhead uses contemporaneous bitwise-equivalent controls.',
       '## Interpretation and stop condition',
       'Leaderboard CPR and model fidelity are different outcomes: a CPR-specific success is labeled as such; general fidelity improvement additionally requires lower CMD. Any apparent success remains development-cohort evidence and requires the separately specified disjoint-validation check before a broader claim. No such check or later gate is launched here.',
       '![Gate3 screening](gate3_screen.png)',
       '**No Gate 4. No leaderboard submission. No expansion beyond lambda 0.5 and 0.25.**'])+'\n'
    (out/'GATE3_ADAPTIVE_SAMPLING_REPORT.md').write_text(text,encoding='utf-8')
    relative=str(out.relative_to(ROOT)).replace('\\','/')
    (ROOT/'GATE3_ADAPTIVE_SAMPLING_REPORT.md').write_text(text.replace('](gate3_screen.png)',f']({relative}/gate3_screen.png)'),encoding='utf-8')
    with (out/'analysis_accounting.jsonl').open('a') as log:log.write(json.dumps({'seconds':time.perf_counter()-started,'model_forwards':0,'confirmation_complete':final is not None})+'\n')
    return decision
