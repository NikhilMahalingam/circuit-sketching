"""Aggregate saved official outputs; never substitute locally invented metrics."""
from itertools import combinations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .runtime import ROOT, digest, write_json


def load(path):
    return json.loads(Path(path).read_text())


def costs(path):
    frame=pd.read_csv(path/'compute_accounting.csv')
    attribution=frame.stage.isin(['attribution_training','attribution_construction'])
    evaluation=frame.stage.str.endswith('_evaluation')
    construction=~evaluation
    result={}
    for label,mask in [('attribution',attribution),('evaluation',evaluation),('construction',construction),('total',np.ones(len(frame),dtype=bool))]:
        subset=frame[mask]
        for field in ['wall_seconds','forward_batches','example_forwards','backward_calls','optimizer_steps']:
            result[label+'_'+field]=float(subset[field].sum())
    for field in ['peak_cuda_allocated_mib','peak_cuda_reserved_mib','peak_cpu_rss_mib']:
        result[field]=float(frame[field].max())
    process=path/'process_accounting.json'
    result['process_wall_seconds']=sum(x['wall_seconds'] for x in load(process)) if process.exists() else float('nan')
    result['startup_and_exit_seconds']=max(0.,result['process_wall_seconds']-result['total_wall_seconds'])
    if process.exists():
        result['construction_wall_seconds']+=result['startup_and_exit_seconds']
    result['validation_wall_seconds']=float(frame.loc[frame.stage=='validation_evaluation','wall_seconds'].sum())
    result['train_diagnostic_wall_seconds']=float(frame.loc[frame.stage=='train_evaluation','wall_seconds'].sum())
    result['matched_protocol_wall_seconds']=result['construction_wall_seconds']+result['validation_wall_seconds']
    return result


def metric_row(path):
    config=load(path/'experiment_config.json');complete=load(path/'complete.json')
    metric=load(path/'official_evaluation.json')
    row={'method':config['method'],'discovery_pool':complete['eligible_unique_train_examples'],
        'seed':config['seed'],'validation_examples':metric['examples'],
        'sampled_unique':complete['sampled_unique_train_examples'],
        'presentations':complete.get('total_training_presentations',complete.get('total_attribution_example_presentations')),
        'cpr_auc':metric['cpr_auc_area_under'],'cmd_auc':metric['cmd_auc_area_from_1'],
        'accuracy_auc':metric['accuracy_auc_logspace'],
        'clean_logit_diff':metric['clean_logit_diff'],'corrupted_logit_diff':metric['corrupted_logit_diff'],
        'train_cpr_auc':float('nan'),'train_validation_gap':float('nan'),
        'job':str(path),**costs(path)}
    row['cpr_auc_per_construction_second']=row['cpr_auc']/row['construction_wall_seconds']
    row['cpr_auc_per_matched_protocol_second']=row['cpr_auc']/row['matched_protocol_wall_seconds']
    ranking=pd.read_csv(path/'node_ranking.csv')
    row['input_node_rank']=int(ranking.loc[ranking.component=='input','rank'].iloc[0])
    if (path/'official_train_evaluation.json').exists():
        train=load(path/'official_train_evaluation.json')
        row['train_cpr_auc']=train['cpr_auc_area_under']
        row['train_validation_gap']=row['train_cpr_auc']-row['cpr_auc']
    return row


def savefig(out,name):
    plt.tight_layout();plt.savefig(out/(name+'.png'),dpi=160);plt.close()


def generate(out):
    out=Path(out)
    training=[];matched=[];curves=[];checkpoint_rows=[];all_costs=[]
    for source in [out/'reference',*sorted((out/'scaling').glob('*'))]:
        if not (source/'complete.json').exists():continue
        row=metric_row(source);row['kind']='reference' if source.name=='reference' else 'scaling'
        training.append(row)
        curve=pd.read_csv(source/'training_curve.csv')
        curve['discovery_pool']=row['discovery_pool'];curve['seed']=row['seed'];curve['kind']=row['kind']
        curves.append(curve)
        if row['kind']=='scaling':
            final=out/f"checkpoint_evaluations/n{row['discovery_pool']}_seed{row['seed']}/step500"
            if (final/'official_train_evaluation.json').exists():
                row['train_cpr_auc']=load(final/'official_train_evaluation.json')['cpr_auc_area_under']
                row['train_validation_gap']=row['train_cpr_auc']-row['cpr_auc']
            matched.append(row.copy())
    for path in sorted((out/'baselines').glob('*/*')):
        if (path/'complete.json').exists():matched.append(metric_row(path))
    for path in sorted((out/'checkpoint_evaluations').glob('*/*')):
        if not (path/'complete.json').exists():continue
        cfg=load(path/'experiment_config.json');value=load(path/'official_evaluation.json')
        train=load(path/'official_train_evaluation.json')
        source=Path(cfg['source']);step=cfg['step']
        if not source.exists():source=out/f"scaling/n{cfg['train_size']}_seed{cfg['seed']}"
        ckpt=torch_load(source/f'checkpoints/step_{step:06d}.pt')
        checkpoint_rows.append({'discovery_pool':cfg['train_size'],'seed':cfg['seed'],'step':step,
            'train_cpr_auc':train['cpr_auc_area_under'],'validation_cpr_auc':value['cpr_auc_area_under'],
            'train_validation_gap':train['cpr_auc_area_under']-value['cpr_auc_area_under'],
            'training_seconds_to_checkpoint':ckpt['training_seconds'],
            'diagnostic_evaluation_seconds':costs(path)['process_wall_seconds']})
    for path in sorted(out.rglob('compute_accounting.csv')):
        if path.parent==out:continue
        if not (path.parent/'complete.json').exists():continue
        frame=pd.read_csv(path);frame.insert(0,'job',str(path.parent.relative_to(out)));all_costs.append(frame)
    empty_columns=['method','discovery_pool','seed','validation_examples','cpr_auc','cmd_auc',
                   'accuracy_auc','train_cpr_auc','train_validation_gap','construction_wall_seconds']
    train_frame=pd.DataFrame(training) if training else pd.DataFrame(columns=empty_columns)
    comparison=pd.DataFrame(matched) if matched else pd.DataFrame(columns=empty_columns)
    train_frame.to_csv(out/'training_results.csv',index=False)
    comparison.to_csv(out/'mattr_vs_eap.csv',index=False)
    comparison.to_csv(out/'quality_vs_compute.csv',index=False)
    loss_curves=pd.concat(curves,ignore_index=True) if curves else pd.DataFrame(columns=['step','k','k_fraction','loss','discovery_pool','seed','kind'])
    loss_curves.to_csv(out/'training_curves.csv',index=False)
    cp=pd.DataFrame(checkpoint_rows) if checkpoint_rows else pd.DataFrame(columns=['discovery_pool','seed','step','train_cpr_auc','validation_cpr_auc','train_validation_gap'])
    cp.to_csv(out/'checkpoint_quality.csv',index=False)
    if all_costs:
        accounting=pd.concat(all_costs,ignore_index=True)
        accounting.to_csv(out/'compute_accounting.csv',index=False)
        accounting.groupby('stage').agg(wall_seconds=('wall_seconds','sum'),
            forward_batches=('forward_batches','sum'),example_forwards=('example_forwards','sum'),
            backward_calls=('backward_calls','sum'),optimizer_steps=('optimizer_steps','sum'),
            peak_cuda_allocated_mib=('peak_cuda_allocated_mib','max'),
            peak_cuda_reserved_mib=('peak_cuda_reserved_mib','max'),
            peak_cpu_rss_mib=('peak_cpu_rss_mib','max')).to_csv(out/'stage_summary.csv')
        processes=[]
        for path in sorted(out.rglob('process_accounting.json')):
            if not (path.parent/'complete.json').exists():continue
            for attempt,row in enumerate(load(path)):
                processes.append({'job':str(path.parent.relative_to(out)),'attempt':attempt,**row})
        pd.DataFrame(processes).to_csv(out/'process_accounting.csv',index=False)
    summary=pd.DataFrame(columns=['method','discovery_pool','runs','validation_cpr_mean','validation_cpr_std'])
    if not comparison.empty:
        summary=comparison.groupby(['method','discovery_pool']).agg(
            runs=('cpr_auc','size'),validation_cpr_mean=('cpr_auc','mean'),validation_cpr_std=('cpr_auc','std'),
            cmd_mean=('cmd_auc','mean'),accuracy_auc_mean=('accuracy_auc','mean'),
            construction_seconds_mean=('construction_wall_seconds','mean'),
            attribution_seconds_mean=('attribution_wall_seconds','mean'),
            evaluation_seconds_mean=('evaluation_wall_seconds','mean'),
            validation_seconds_mean=('validation_wall_seconds','mean'),
            matched_protocol_seconds_mean=('matched_protocol_wall_seconds','mean'),
            train_cpr_mean=('train_cpr_auc','mean'),gap_mean=('train_validation_gap','mean')).reset_index()
        summary.to_csv(out/'dataset_scaling.csv',index=False)
        for method,frame in summary.groupby('method'):
            err=frame.validation_cpr_std if np.isfinite(frame.validation_cpr_std).any() else None
            plt.errorbar(frame.discovery_pool,frame.validation_cpr_mean,yerr=err,
                         marker='o',capsize=3,label=method)
        plt.xlabel('Eligible unique discovery examples');plt.ylabel('Official validation CPR AUC');plt.legend()
        savefig(out,'quality_vs_discovery_data')
        for method,frame in comparison.groupby('method'):
            plt.scatter(frame.construction_wall_seconds,frame.cpr_auc,label=method,alpha=.8)
        plt.xscale('log');plt.xlabel('Total construction seconds (preparation + attribution + ranking)')
        plt.ylabel('Official validation CPR AUC');plt.legend();savefig(out,'quality_vs_construction_compute')
        for method,frame in comparison.groupby('method'):
            plt.scatter(frame.matched_protocol_wall_seconds,frame.cpr_auc,label=method,alpha=.8)
        plt.xscale('log');plt.xlabel('Construction + identical validation evaluation (seconds)')
        plt.ylabel('Official validation CPR AUC');plt.legend();savefig(out,'quality_vs_compute')
        for method,frame in comparison.groupby('method'):
            plt.scatter(frame.matched_protocol_wall_seconds,frame.accuracy_auc,label=method,alpha=.8)
        plt.xscale('log');plt.xlabel('Construction + identical validation evaluation (seconds)')
        plt.ylabel('Official accuracy AUC over log sparsity');plt.legend();savefig(out,'accuracy_vs_compute')
        # Same fixed validation cohort and requested circuit-size grid for every line.
        for method,frame in comparison[comparison.discovery_pool==128].groupby('method'):
            values=[pd.read_csv(Path(job)/'validation_curve.csv') for job in frame.job]
            points=np.stack([x.official_normalized_logit_diff_cpr.to_numpy() for x in values])
            plt.plot(values[0].retained_fraction,points.mean(axis=0),'o-',label=method+' (pool 128)')
        common=out/'reference_common_validation/validation_curve.csv'
        if common.exists():
            values=pd.read_csv(common)
            plt.plot(values.retained_fraction,values.official_normalized_logit_diff_cpr,'--',label='MAttr (full pool)')
        plt.xscale('log');plt.xlabel('Requested retained-node fraction');plt.ylabel('Official normalized logit difference')
        plt.legend(fontsize=8);savefig(out,'circuit_size_curves')
        for method,frame in summary.groupby('method'):
            plt.plot(frame.discovery_pool,frame.train_cpr_mean,'--o',label=method+' train')
            plt.plot(frame.discovery_pool,frame.validation_cpr_mean,'-o',label=method+' validation')
        plt.xlabel('Eligible unique discovery examples');plt.ylabel('Official CPR AUC');plt.legend(fontsize=8)
        savefig(out,'train_vs_validation')
        for method,frame in summary.groupby('method'):
            plt.plot(frame.discovery_pool,frame.gap_mean,'o-',label=method)
        plt.axhline(0,color='grey',lw=1);plt.xlabel('Eligible unique discovery examples');plt.ylabel('Train minus validation CPR AUC')
        plt.legend();savefig(out,'generalization_gap')
    else:summary.to_csv(out/'dataset_scaling.csv',index=False)
    if not cp.empty:
        for n,frame in cp.groupby('discovery_pool'):
            stat=frame.groupby('step').validation_cpr_auc.agg(['mean','std'])
            err=stat['std'] if np.isfinite(stat['std']).any() else None
            plt.errorbar(stat.index,stat['mean'],yerr=err,marker='o',label=f'n={n}')
        plt.xlabel('Optimization steps');plt.ylabel('Official fixed-validation CPR AUC');plt.legend();savefig(out,'quality_vs_steps')
        for n,frame in cp.groupby('discovery_pool'):
            points=frame.groupby('step')[['training_seconds_to_checkpoint','validation_cpr_auc']].mean()
            plt.plot(points.training_seconds_to_checkpoint,points.validation_cpr_auc,'o-',label=f'n={n}')
        plt.xlabel('Measured training seconds through checkpoint');plt.ylabel('Official validation CPR AUC')
        plt.legend();savefig(out,'quality_vs_training_runtime')
    if training:
        for n,frame in loss_curves[loss_curves.kind=='scaling'].groupby('discovery_pool'):
            frame=frame.sort_values(['seed','step']).copy()
            frame['smoothed']=frame.groupby('seed').loss.transform(lambda x:x.rolling(25,min_periods=1).mean())
            stat=frame.groupby('step').smoothed.agg(['mean','std'])
            line=plt.plot(stat.index,stat['mean'],label=f'pool {n}')[0]
            plt.fill_between(stat.index,stat['mean']-stat['std'].fillna(0),
                stat['mean']+stat['std'].fillna(0),color=line.get_color(),alpha=.12)
        ref=loss_curves[loss_curves.kind=='reference']
        if not ref.empty:
            plt.plot(ref.step,ref.loss.rolling(25,min_periods=1).mean(),'--',color='black',label='full-pool reference')
        plt.xlabel('Optimization step');plt.ylabel('Training loss (25-step rolling mean)')
        plt.legend(fontsize=8,title='Shading: ±1 seed SD',title_fontsize=8);savefig(out,'training_loss')
    confirmation=[]
    for path in sorted((out/'confirmation').glob('*/*')):
        if not (path/'complete.json').exists():continue
        cfg=load(path/'experiment_config.json');metric=load(path/'official_test_evaluation.json')
        confirmation.append({'method':cfg['method'],'seed':cfg['seed'],
            'discovery_pool':cfg['discovery_pool'],'test_examples':metric['examples'],
            'cpr_auc':metric['cpr_auc_area_under'],'cmd_auc':metric['cmd_auc_area_from_1'],
            'accuracy_auc':metric['accuracy_auc_logspace'],'job':str(path),**costs(path)})
    cf=pd.DataFrame(confirmation) if confirmation else pd.DataFrame(columns=['method','seed','discovery_pool','test_examples','cpr_auc','cmd_auc','accuracy_auc'])
    cf.to_csv(out/'public_test_confirmation.csv',index=False)
    if not cf.empty:
        stat=cf.groupby('method').cpr_auc.agg(['mean','std'])
        plt.bar(stat.index,stat['mean'],yerr=stat['std'] if stat['std'].notna().any() else None,capsize=4)
        plt.ylabel('Official CPR AUC on fixed 128-example public-test cohort')
        savefig(out,'public_test_confirmation')
    _report(out,train_frame,comparison,summary,cp,cf)
    print('Report and aggregate tables saved:',out,flush=True)


def torch_load(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=False)


def _report(out,training,comparison,summary,checkpoints,confirmation):
    missing=[]
    if not (out/'reference/complete.json').exists():missing.append('full reference')
    if len(comparison)!=36:missing.append(f'equal-data comparisons ({len(comparison)}/36)')
    if len(checkpoints)!=36:missing.append(f'checkpoint diagnostics ({len(checkpoints)}/36)')
    if len(confirmation)!=9:missing.append(f'frozen public-test confirmation ({len(confirmation)}/9)')
    test_access=bool(list((out/'confirmation').glob('*/*/test_access_started.json')))
    test_status=('Public test is used only for the frozen final confirmation; discovery and selection use train/validation.'
                 if test_access else 'Public test has not been used for discovery, selection, or evaluation.')
    tests=load(out/'tests_passed.json') if (out/'tests_passed.json').exists() else {}
    test_text=(f'Official tests ({tests.get("official_tests")}) and adapter tests ({tests.get("adapter_tests")}) passed; fresh and resumed updates matched the upstream training loop bit for bit in deterministic tensor tests.'
               if tests.get('passed') else 'Completed test evidence is not available in this campaign.')
    smoke_text=('The CUDA smoke trained and evaluated GPT-2 before the reference.'
                if (out/'smoke/complete.json').exists() else 'The CUDA smoke has not completed.')
    lines=['# MAttr reproduction on GPT-2 Small / MIB IOI','',
        f'Campaign: `{out.relative_to(ROOT)}`. Branch: `codex/mattr-reproduction`. '+test_status,'',
        '## Reproduction and provenance','',
        'Attribution calls the unmodified official MAttr trainer and CLI; evaluation calls the pinned official MIB evaluator. The local adapter selects rows, records compute, and saves/restores scores, Adam state and RNGs. '+test_text+' '+smoke_text,'',
        'Pinned MAttr: `af82b2eef145dd3e66c793a413352d6931a152d7`. Exact MIB/EAP commits, package versions, lock/source hashes, commands, prompt manifests and counters are in each run directory. WSL Ubuntu / RTX 3060 12 GB uses an isolated Python 3.12 environment and the official locked lm/dev dependencies. Existing Sketch-ACDC artifacts were preserved.','',
        'Source: [official repository](https://github.com/aryamanarora/matryoshka-attribution). Published comparison: [paper, Appendix E.5 Table 4](https://arxiv.org/html/2609.25518v1) reports GPT-2/IOI node validation CPR AUC 1.83 for the 500-step lr 0.05 recipe. That is an author-reported result, distinct from our measurements. The paper also gives public-test results; they are not our validation target.','']
    reference=training[training.kind=='reference'] if not training.empty else pd.DataFrame()
    if not reference.empty:
        r=reference.iloc[0]
        match=('This reproduces the authors’ validation result to its reported two-decimal precision; the rounded target cannot establish agreement beyond that precision.'
               if round(r.cpr_auc,2)==1.83 else 'The recipe completed, but its numerical result differs from the published value.')
        lines += [f'Our uncapped seed-42 reference completed 500 optimization steps on an eligible pool of {int(r.discovery_pool):,} official train rows, evaluated on {int(r.validation_examples):,} validation rows. Official CPR AUC **{r.cpr_auc:.4f}**, CMD AUC **{r.cmd_auc:.4f}**, accuracy AUC **{r.accuracy_auc:.4f}**. Difference from the rounded reported 1.83: **{r.cpr_auc-1.83:+.4f}**. '+match,
            '',f'The reference actually sampled {int(r.sampled_unique)} unique training rows in {int(r.presentations)} presentations. Its eligible pool size is not its number of unique rows visited by optimization.',
            '',f'Clean/corrupted mean logit differences: {r.clean_logit_diff:.4f} / {r.corrupted_logit_diff:.4f}. Attribution: {r.attribution_wall_seconds:.1f}s; official validation evaluation: {r.evaluation_wall_seconds:.1f}s. Peak allocated CUDA memory: {r.peak_cuda_allocated_mib/1024:.2f} GiB; peak reserved CUDA memory: {r.peak_cuda_reserved_mib/1024:.2f} GiB; sampled CPU RSS peak: {r.peak_cpu_rss_mib/1024:.2f} GiB.','']
        common=out/'reference_common_validation/official_evaluation.json'
        if common.exists():
            value=load(common)
            lines += [f'The same reference ranking scores {value["cpr_auc_area_under"]:.4f} CPR AUC on the fixed {value["examples"]}-prompt comparison cohort. This is a separate cohort result and is not substituted for the uncapped reference.','']
    lines += ['## Metric definitions and comparison protocol','',
        '`area_under` integrates normalized logit-difference faithfulness over the official linear retained-fraction grid; the official CLI calls it CPR AUC. At each size the implementation uses (mean circuit − mean corrupted)/(mean clean − mean corrupted). `area_from_1` integrates absolute deviation from one (official CMD AUC). Accuracy AUC uses positive logit difference over log sparsity. These three quantities are distinct. CPR can exceed one through behavioral overshoot; larger CPR alone does not prove a more faithful mechanism. CMD closer to zero means less deviation from full-model behavior.','',
        'The ten retained fractions are 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 and 1.0, applied to the same 157-node basis (input, 144 heads, 12 MLPs). Requested counts are 0,0,0,1,3,7,15,31,78,157; post-pruning retained counts are recorded separately for each graph. Weighted edge counts are never labeled as node counts.','',
        'The reference follows the uncapped documented command. All scaling and baseline comparisons use exactly one fixed 128-prompt validation cohort (seed 42), the same nested discovery pools within seeds 42/43/44, node granularity, signed ranking and counterfactual patching. The reference ranking has a separate reevaluation on that cohort (pending until its artifact exists). No custom IOI data or homemade leaderboard metric is used.','',
        'Pool size is the number of eligible unique examples; MAttr samples with replacement for 500 one-example updates. Actually visited unique rows and total presentations are recorded. EAP sees the pool once; EAP-IG uses five official integration steps. Both baselines use attribution batch size 1: the official mean-loss implementation accumulates per-batch averages, so a smaller final batch would otherwise receive different per-example weight. Evaluation batch size stays 20 for every compared method. The official attribution functions and loss are unchanged. These are equal eligible-data comparisons, **not compute-matched** experiments. Cost-quality plots expose their different budgets; no new compute-matched tuning was performed.','',
        '## Discovery-data size and baselines','']
    if not summary.empty:
        lines += ['| Method | Pool | Runs | Validation CPR mean ± SD | CMD | Accuracy AUC | Construction s | Train−validation CPR |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in summary.itertuples():
            lines.append(f'| {r.method} | {r.discovery_pool} | {r.runs} | {r.validation_cpr_mean:.4f} ± {r.validation_cpr_std:.4f} | {r.cmd_mean:.4f} | {r.accuracy_auc_mean:.4f} | {r.construction_seconds_mean:.1f} | {r.gap_mean:.4f} |')
        ma=summary[summary.method=='MAttr'].sort_values('discovery_pool')
        if not ma.empty:
            first,last=ma.iloc[0],ma.iloc[-1]
            lines += ['',f'For completed MAttr runs, validation CPR changed from {first.validation_cpr_mean:.4f} at {int(first.discovery_pool)} examples to {last.validation_cpr_mean:.4f} at {int(last.discovery_pool)} examples ({last.validation_cpr_mean-first.validation_cpr_mean:+.4f}). Seed variation measures optimization/subset differences on the same validation cohort; it does not establish generalization to other tasks or models.']
            lines += [f'Across-seed SD was {first.validation_cpr_std:.4f} at the smallest pool and {last.validation_cpr_std:.4f} at the largest. With only three seeds, these are descriptive variability estimates.']
            sampled=comparison[comparison.method=='MAttr']
            coverage=[f'{int(n)}-row pool: {int(g.sampled_unique.min())}–{int(g.sampled_unique.max())} unique rows visited'
                      for n,g in sampled.groupby('discovery_pool')]
            lines += ['', '; '.join(coverage)+f'. Actual presentations per MAttr run: {int(sampled.presentations.min())}–{int(sampled.presentations.max())}.']
            best=ma.loc[ma.validation_cpr_mean.idxmax()]
            if len(ma)==4 and (ma.runs==3).all():
                write_json(out/'validation_selected_configuration.json',{'method':'MAttr','discovery_pool':int(best.discovery_pool),
                    'steps':500,'selection_metric':'mean fixed-validation official CPR AUC over seeds 42,43,44',
                    'validation_cpr_mean':float(best.validation_cpr_mean),'public_test_accessed':False,
                    'selection_is_exploratory':True})
                lines += ['',f'Exploratory selection among the final 500-step MAttr configurations uses mean validation CPR: pool {int(best.discovery_pool)}. All raw runs remain saved. Public-test confirmation uses this same pool for all methods and the three preset seeds.']
            paired=summary.pivot(index='discovery_pool',columns='method',values='validation_cpr_mean')
            if len(comparison)==36 and {'MAttr','EAP','EAP-IG'}<=set(paired.columns):
                for baseline in ('EAP','EAP-IG'):
                    difference=paired.MAttr-paired[baseline]
                    wins=int((difference>0).sum())
                    lines += ['',f'MAttr has higher mean validation CPR than {baseline} at {wins}/{len(difference)} pool sizes; mean paired difference over pool sizes is {difference.mean():+.4f}. This comparison uses each method’s prescribed budget, not matched compute.']
                    cmd=summary.pivot(index='discovery_pool',columns='method',values='cmd_mean')
                    lines += [f'It has lower mean CMD than {baseline} at {int((cmd.MAttr<cmd[baseline]).sum())}/{len(cmd)} pool sizes.']
                    acc=summary.pivot(index='discovery_pool',columns='method',values='accuracy_auc_mean')
                    lines += [f'It has higher mean official accuracy AUC than {baseline} at {int((acc.MAttr>acc[baseline]).sum())}/{len(acc)} pool sizes.']
                ig=summary[summary.method=='EAP-IG'].set_index('discovery_pool')
                matched_ma=ma.set_index('discovery_pool')
                if (matched_ma.validation_cpr_mean>ig.validation_cpr_mean).all() and (matched_ma.cmd_mean>ig.cmd_mean).all():
                    lines += ['', 'MAttr therefore wins the CPR comparison against EAP-IG while deviating more from full-model behavior by CMD at every pool size. This is a metric tradeoff, not an unqualified fidelity win; the circuit-size curves show the overshoot that contributes to the CPR score.']
                largest=int(matched_ma.index.max())
                a,b=matched_ma.loc[largest],ig.loc[largest]
                lines += ['',f'At the {largest}-row pool, construction plus the identical validation protocol takes {a.matched_protocol_seconds_mean:.1f}s for MAttr versus {b.matched_protocol_seconds_mean:.1f}s for EAP-IG (means over three seeds). These are measured costs of the tested recipes, not results from tuning both methods to a common time budget.']
                ranks=comparison.groupby('method').input_node_rank.agg(['min','max'])
                lines += ['', 'Observed input-node ranking ranges (out of 157): '+
                    '; '.join(f'{method} {int(r["min"])}–{int(r["max"])}' for method,r in ranks.iterrows())+'.']
                late=comparison[(comparison.method=='EAP') & (comparison.input_node_rank>78)]
                if len(late):
                    lines += [f'In {len(late)}/12 EAP runs, the input ranks below the 78 nodes requested at 50% retention. This is an observed limitation of the signed node-ranking protocol and is consistent with sparse circuits losing clean input information. An input-held-clean control was not run, so it does not prove the cause or account for the MAttr–EAP-IG difference. No input score was forced or adjusted.']
            if len(ma)==4 and ma.gap_mean.notna().all():
                lines += ['',f'MAttr mean train−validation CPR gaps range from {ma.gap_mean.min():+.4f} to {ma.gap_mean.max():+.4f} across pool sizes.']
                if ma.gap_mean.iloc[0]>max(0.,ma.gap_mean.iloc[-1]):
                    lines += ['The positive gap at the smallest pool is larger than at 128 examples, consistent with the small-data overfitting concern; three seeds and one held-out cohort limit the strength of this evidence.']
                else:
                    lines += ['The smallest pool does not show a larger positive train−validation gap than the 128-example pool; these measurements do not support a simple monotonic small-data overfitting claim.']
            else:
                lines += ['', 'Training-quality/gap evaluations are still pending; no overfitting conclusion is drawn.']
            for method,frame in comparison.groupby('method'):
                lines += ['',f'{method}: mean attribution {frame.attribution_wall_seconds.mean():.2f}s, total construction {frame.construction_wall_seconds.mean():.2f}s, fixed-validation evaluation {frame.validation_wall_seconds.mean():.2f}s; attribution uses {frame.attribution_forward_batches.mean():.1f} forward batches and {frame.attribution_backward_calls.mean():.1f} backward calls on average.']
    else:lines.append('No complete matched measurements yet.')
    budget_text=''
    if (out/'stage_summary.csv').exists():
        stages=pd.read_csv(out/'stage_summary.csv')
        total=float(stages.wall_seconds.sum())
        biggest=stages.loc[stages.wall_seconds.idxmax()]
        budget_text=(f'Across completed model jobs, instrumented stages total {total/60:.1f} minutes, '
            f'{int(stages.forward_batches.sum()):,} forward batches, '
            f'{int(stages.example_forwards.sum()):,} example-forwards and '
            f'{int(stages.backward_calls.sum()):,} backward calls. '
            f'The largest accumulated stage is `{biggest.stage}`: {biggest.wall_seconds/60:.1f} minutes '
            f'({100*biggest.wall_seconds/total:.1f}% of stage time). '
            'This includes reference and diagnostic sweeps as well as the comparison jobs; unit tests, notebook rendering and analysis are outside these model-stage counters.')
    lines += ['', '## Generalization and computational cost','',
        'Train-subset and validation quality use the same official circuit sweep. A positive train−validation gap is consistent with overfitting but can also reflect cohort variation and normalization by different clean/corrupted reference scores. Loss is noisy because sparsity and prompts vary every update; it is not a held-out-quality metric. Checkpoints at 100, 250 and 500 steps are evaluated without changing the optimizer trajectory.','',
        'Root-model forwards are counted at actual execution, including counterfactual donor forwards. Cached donors do not increment forward counts. Backward calls and Adam updates are separate counters. Construction cost includes interpreter/import startup, data/model preparation, all mask optimization, checkpoint overhead and ranking; circuit evaluation is separate. The total-compute plot adds the same fixed-validation evaluation to construction cost. Extra train-quality and checkpoint diagnostics are recorded separately, rather than charging their unequal query counts to that comparison. All actual work remains in the stage/process accounting tables. The final checkpoint’s identical validation query is reused after score/cohort/configuration checks and adds zero forwards. CUDA stages synchronize before timing and report allocator peaks; CPU RSS is sampled. Timers use monotonic perf_counter rather than console calendar timestamps. Timing includes instrumentation and cold-start overhead, so it is not an author-hardware speed comparison.','',
        budget_text,'',
        'Artifacts: `training_results.csv`, `dataset_scaling.csv`, `mattr_vs_eap.csv`, `quality_vs_compute.csv`, `training_curves.csv`, `checkpoint_quality.csv`, `compute_accounting.csv`, per-run rankings/tensors/manifests/checkpoints/raw per-example evaluations, logs and PNG figures.','',
        '## Frozen public-test confirmation','',
        'The pinned [MIB README](https://github.com/aryamanarora/MIB-circuit-track/blob/582fdbc8a2b0f7063a265c8f94851fd2a3aa5480/README.md#evaluation) explicitly permits local public-test evaluation using `--split test`. After all validation experiments finish, `confirmation_selection.json` freezes the discovery pool, all nine source-ranking hashes, seeds, checkpoint, and a new deterministic 128-example test cohort before test data is loaded. The common pool maximizes mean MAttr validation CPR over seeds 42/43/44; all three methods use that pool. Test results cannot change this selection.','']
    if not confirmation.empty:
        lines += ['| Method | Test runs | Public-test CPR mean ± SD | CMD mean | Accuracy AUC mean |',
                  '|---|---:|---:|---:|---:|']
        for method,frame in confirmation.groupby('method'):
            lines.append(f'| {method} | {len(frame)} | {frame.cpr_auc.mean():.4f} ± {frame.cpr_auc.std():.4f} | {frame.cmd_auc.mean():.4f} | {frame.accuracy_auc.mean():.4f} |')
        changes=[]
        for method,frame in confirmation.groupby('method'):
            pool=int(frame.discovery_pool.iloc[0])
            validation=comparison[(comparison.method==method)&(comparison.discovery_pool==pool)]
            if len(frame)==3 and len(validation)==3:
                changes.append(f'{method} {frame.cpr_auc.mean()-validation.cpr_auc.mean():+.4f}')
        if changes:
            lines += ['', 'Public-test minus validation mean CPR at the frozen pool: '+
                '; '.join(changes)+'. This cohort change also changes prompt difficulty and normalization; it is not a pure estimate of optimization overfitting.']
    else:lines.append('Confirmation is pending; no public-test conclusion is drawn.')
    lines += ['', 'These are local measurements on one fixed public-test subset, not a private leaderboard submission or a full-public-test reproduction. Error bars describe ranking/subset-seed variability on that same cohort, not population confidence intervals. The official evaluator supplies all reported metrics.','',
        '## Research decision','']
    candidate='No candidate modification is justified until the pending measurements finish.'
    if not missing and not summary.empty and not checkpoints.empty:
        ma=summary[summary.method=='MAttr'].sort_values('discovery_pool')
        step_means=checkpoints.groupby('step').validation_cpr_auc.mean()
        earliest=int(step_means[step_means>=step_means.max()-.02].index.min())
        if len(ma)==4 and ma.validation_cpr_mean.iloc[-1]-ma.validation_cpr_mean.iloc[0]>.05:
            candidate='Most promising next hypothesis: improve discovery-prompt coverage at a fixed presentation budget. The measured larger-pool advantage motivates a predeclared coverage experiment; it does not establish a new method.'
        elif earliest<500:
            candidate=f'Most promising next hypothesis: validate a shorter fixed optimization budget. The {earliest}-step checkpoint is within 0.02 mean CPR AUC of the best measured step when averaging these configurations; this exploratory observation needs an independent confirmation cohort. Repeated full validation sweeps are costly, so an online early-stopping scheme would have to include their overhead before claiming a saving.'
        else:
            candidate='The measured endpoints do not establish a simple data-coverage or early-stopping win. Prioritize replication on additional held-out cohorts before proposing an attribution change.'
    lines += [candidate,'',
        'No proposed algorithmic improvement has been implemented. Equal-compute claims require separate predeclared budgets. Optional interaction analysis remains gated and cannot establish explicit Hessian learning.','',
        '**Execution status:** '+('complete requested reference, scaling, baseline, checkpoint and frozen public-test measurements.' if not missing else 'incomplete: '+', '.join(missing)+'.')]
    text='\n'.join(lines)+'\n'
    (ROOT/'MATTR_REPRODUCTION_REPORT.md').write_text(text,encoding='utf-8')
    (out/'MATTR_REPRODUCTION_REPORT.md').write_text(text,encoding='utf-8')
