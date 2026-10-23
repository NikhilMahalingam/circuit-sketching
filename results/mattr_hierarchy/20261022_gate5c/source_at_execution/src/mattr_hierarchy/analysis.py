"""Frozen selection/confirmation decisions and actual-results reporting."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from src.mattr_optimization.common import ROOT,sha,write_json
from src.mattr_optimization.gate3 import baseline
from src.mattr_optimization.gate2_final_analysis import table
from src.mattr_optimization.gate4_report import job_cost
from src.mattr_expander.analysis import quality,base_quality
from .adapter import Hierarchy,group_sum

FRACTIONS=(.005,.01,.02,.05)
def job(out,family,gamma,seed,shuffled=False):return out/f'jobs/{family}_g{gamma:g}_s{seed}{"_shuffled" if shuffled else ""}'
def protocol(out):
    assert sha(out/'gate5c_protocol.json')==(out/'gate5c_protocol.sha256').read_text().strip()
    return json.loads((out/'gate5c_protocol.json').read_text())
def cost(out,folder,step):
    c=job_cost(folder,1000 if step<=1000 else None)
    state=torch.load(folder/f'checkpoints/step_{step:06d}.pt',map_location='cpu',weights_only=False)
    c['adaptation_seconds']=c['adaptation_seconds']-c['training_seconds']+state['training_seconds']+json.loads((out/'mapping_audit.json').read_text())['construction_seconds']
    return c
def compare(out,folder,step):
    q=quality(folder,step);cfg=json.loads((folder/'experiment_config.json').read_text());b=base_quality(out,cfg['seed'],step)
    c=cost(out,folder,step);fresh=job(out,'bypass',0,cfg['seed'])
    if step<=1000:
        bc=cost(out,fresh,step)
    else:
        bc=job_cost(fresh,1000);state=torch.load(baseline(cfg['seed'])/f'checkpoints/step_{step:06d}.pt',map_location='cpu',weights_only=False)
        bc['adaptation_seconds']=bc['adaptation_seconds']-bc['training_seconds']+state['training_seconds']
    row={**q,'job':folder.name,'gamma':cfg['gamma'],'shuffled':cfg['shuffled'],
         'baseline_cpr':b['cpr'],'baseline_cmd':b['cmd'],'cpr_delta':q['cpr']-b['cpr'],
         'relative_cpr_gain':q['cpr']/b['cpr']-1,'cmd_delta':q['cmd']-b['cmd'],
         'accuracy_auc_delta':q['accuracy_auc']-b['accuracy_auc'],'time_ratio':c['adaptation_seconds']/bc['adaptation_seconds'],
         'adaptation_seconds':c['adaptation_seconds'],'baseline_adaptation_seconds':bc['adaptation_seconds'],
         'memory_ratio':c['peak_allocated_mib']/bc['peak_allocated_mib']}
    for f in FRACTIONS:row[f'accuracy_delta_{f}']=q[f'accuracy_{f}']-b[f'accuracy_{f}']
    return row
def gates(rows,screen=False):
    gain=.005 if screen else .02
    checks={'CPR':rows.relative_cpr_gain.mean()>=gain,'accuracy_auc':rows.accuracy_auc_delta.mean()>=-.02,
       'CMD':rows.cmd_delta.mean()<=max(0.,rows.cpr_delta.mean())+.05,'adaptation_compute':rows.time_ratio.mean()<=1.10,
       'memory':rows.memory_ratio.mean()<=1.10}
    if not screen:checks['positive_seeds']=int((rows.cpr_delta>0).sum())>=2
    for f in FRACTIONS:checks[f'compact_{f}']=rows[f'accuracy_delta_{f}'].mean()>=-.02
    return {'passed':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()}}
def select_b(out):
    rows=pd.DataFrame([compare(out,job(out,f,g,42),1000) for f in ('H1','H2') for g in (.001,.01)])
    result=[];eligible=[]
    for _,row in rows.iterrows():
        decision=gates(pd.DataFrame([row]),screen=True)
        result.append({'family':row.family,'gamma':row.gamma,'relative_CPR_gain':row.relative_cpr_gain,**decision})
        if decision['passed']:eligible.append(row)
    eligible=sorted(eligible,key=lambda x:(-x.relative_cpr_gain,-sum(x[f'accuracy_delta_{f}'] for f in FRACTIONS),x.gamma,x.family))
    selected={'family':eligible[0].family,'gamma':float(eligible[0].gamma)} if eligible else None
    write_json(out/'phaseB_decision.json',{'selected':selected,'results':result,'exploratory_seed':42,'no_extra_gamma':True})
    return selected
def controls_for(selected):
    f,g=selected['family'],selected['gamma']
    return [(f,0.,False),(f,g,True),('edge',g,False)]
def select_c(out,selected,step=1000):
    real=pd.DataFrame([compare(out,job(out,selected['family'],selected['gamma'],s),step) for s in (42,43,44)])
    primary=gates(real);controls={}
    for family,gamma,shuffled in controls_for(selected):
        c=pd.DataFrame([compare(out,job(out,family,gamma,s,shuffled),step) for s in (42,43,44)])
        differences=real.cpr.to_numpy()/c.cpr.to_numpy()-1
        checks={'CPR_advantage':float(differences.mean())>=.005,'positive_seeds':int((differences>0).sum())>=2,
                'accuracy_auc':float((real.accuracy_auc.to_numpy()-c.accuracy_auc.to_numpy()).mean())>=-.02}
        for f in FRACTIONS:checks[f'compact_{f}']=float((real[f'accuracy_{f}'].to_numpy()-c[f'accuracy_{f}'].to_numpy()).mean())>=-.02
        controls[f'{family}_gamma{gamma:g}_shuffled{shuffled}']={'passed':bool(all(checks.values())),
             'checks':{k:bool(v) for k,v in checks.items()},'mean_relative_CPR_advantage':float(differences.mean()),
             'per_seed_relative_CPR_advantage':differences.tolist()}
    passed=primary['passed'] and all(c['passed'] for c in controls.values())
    result={'winner':selected if passed else None,'primary':primary,'structural_controls':controls,
            'passed':passed,'step':step,'claim':'structural hierarchy only if primary AND all3 controls pass'}
    write_json(out/f'phase{"C" if step==1000 else "D"}_decision.json',result)
    return passed

def structural(out,folders):
    mapping=pd.read_csv(out/'edge_mapping.csv',keep_default_na=False);sizes=pd.read_csv(out/'group_sizes.csv')
    rows=[];vectors={};stability=[]
    for folder in folders:
        cfg=json.loads((folder/'experiment_config.json').read_text())
        for p in sorted(folder.glob('quality_*.json')):
            step=int(p.stem.split('_')[-1]);raw=torch.load(folder/f'checkpoints/step_{step:06d}.pt',map_location='cpu',weights_only=False)['scores']
            ids=np.load(out/f'{cfg["family"]}{"_shuffled" if cfg["shuffled"] else ""}_groups.npy') if cfg['family'] in ('H1','H2') else None
            h=Hierarchy(ids,kind='hierarchy' if ids is not None else cfg['family'])
            effective,residual=h.parts(raw);vectors[folder.name,step]=effective.numpy()
            for hierarchy in ('H1','H2'):
                true_ids=torch.tensor(np.load(out/f'{hierarchy}_groups.npy'),dtype=torch.long)
                count=torch.bincount(true_ids);means=torch.bincount(true_ids,weights=effective.double())/count
                rc=effective.double()-means[true_ids]
                square=torch.bincount(true_ids,weights=rc.square())/count
                labels=sizes[sizes.hierarchy==hierarchy].sort_values('group_id')
                order=np.argsort(-effective.numpy())
                for label in labels.itertuples():
                    group=label.group_id
                    row={'job':folder.name,'family':cfg['family'],'gamma':cfg['gamma'],'shuffled':cfg['shuffled'],'seed':cfg['seed'],
                        'step':step,'hierarchy':hierarchy,'component':label.component,'type':label.type,'edges':int(count[group]),
                        'mean_effective_score':float(means[group]),'within_component_variance':float(square[group])}
                    for f in FRACTIONS:row[f'top_edges_{f}']=int((true_ids[order[:int(f*32491)]]==group).sum())
                    rows.append(row)
            if ids is not None:
                actual_h=raw[:h.group_count];actual_r=residual
                pd.DataFrame({'group_id':range(h.group_count),'learned_h':actual_h.numpy(),
                    'centered_residual_mean':(group_sum(actual_r,h.padded,h.valid)/h.sizes).numpy(),
                    'residual_variance':(group_sum(actual_r.square(),h.padded,h.valid)/h.sizes).numpy()}).to_csv(folder/f'component_scores_{step}.csv',index=False)
                torch.save(actual_r,folder/f'centered_residuals_{step}.pt')
    table_rows=pd.DataFrame(rows);table_rows.to_csv(out/'structural_component_statistics.csv',index=False)
    for (family,gamma,shuffled,step),group in table_rows.groupby(['family','gamma','shuffled','step']):
        seeds=sorted(group.seed.unique())
        for a,b in itertools.combinations(seeds,2):
            fa=job(out,family,gamma,a,shuffled);fb=job(out,family,gamma,b,shuffled)
            va=vectors[fa.name,step];vb=vectors[fb.name,step]
            stability.append({'family':family,'gamma':gamma,'shuffled':shuffled,'step':step,'seed_a':a,'seed_b':b,'level':'edge','spearman':spearmanr(va,vb).statistic})
            for hierarchy in ('H1','H2'):
                x=group[(group.seed==a)&(group.hierarchy==hierarchy)].sort_values('component')
                y=group[(group.seed==b)&(group.hierarchy==hierarchy)].sort_values('component')
                stability.append({'family':family,'gamma':gamma,'shuffled':shuffled,'step':step,'seed_a':a,'seed_b':b,'level':hierarchy,'spearman':spearmanr(x.mean_effective_score,y.mean_effective_score).statistic})
            for f in FRACTIONS:
                ma=pd.read_csv(fa/f'membership_{step}.csv');mb=pd.read_csv(fb/f'membership_{step}.csv')
                ea=set(ma[np.isclose(ma.fraction,f)].edge);eb=set(mb[np.isclose(mb.fraction,f)].edge)
                stability.append({'family':family,'gamma':gamma,'shuffled':shuffled,'step':step,'seed_a':a,'seed_b':b,'level':f'pruned_edges_{f}','jaccard':len(ea&eb)/len(ea|eb) if ea|eb else np.nan})
    pd.DataFrame(stability,columns=['family','gamma','shuffled','step','seed_a','seed_b','level','spearman','jaccard']).to_csv(out/'ranking_stability.csv',index=False)

def generate(out):
    p=protocol(out);folders=[x for x in sorted((out/'jobs').glob('*')) if (x/'completed_1000.json').exists()]
    paired=[];costs=[];gradients=[];curves=[]
    for folder in folders:
        cfg=json.loads((folder/'experiment_config.json').read_text())
        for path in sorted(folder.glob('quality_*.json')):
            step=int(path.stem.split('_')[-1])
            if step in (500,1000,5000):paired.append(compare(out,folder,step))
            curve=pd.read_csv(folder/f'evaluation_step{step}/faithfulness_curve.csv');curves.append(curve.assign(job=folder.name,step=step,seed=cfg['seed']))
        for path in sorted(folder.glob('gradient_diagnostics_*.csv')):gradients.append(pd.read_csv(path).assign(job=folder.name))
        costs.append({'job':folder.name,**job_cost(folder)})
    for name in ('pilot','phaseA_smoke'):
        if (out/name/'process_summary.json').exists():costs.append({'job':name,**job_cost(out/name)})
    rows=pd.DataFrame(paired);compute=pd.DataFrame(costs)
    rows.to_csv(out/'paired_results.csv',index=False);compute.to_csv(out/'compute_accounting.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(out/'faithfulness_curves.csv',index=False)
    grad=pd.concat(gradients,ignore_index=True);grad.to_csv(out/'gradient_diagnostics.csv',index=False)
    structural(out,folders)
    decisions={name:json.loads((out/f'phase{name}_decision.json').read_text()) if (out/f'phase{name}_decision.json').exists() else None for name in ('B','C','D')}
    fig=out/'figures';fig.mkdir(exist_ok=True)
    def save(name):plt.tight_layout();plt.savefig(fig/f'{name}.png',dpi=150);plt.close()
    for metric,name,label in [('cpr','cpr_vs_steps','Official CPR'),('cmd','cmd_vs_steps','Official CMD')]:
        plt.figure(figsize=(8,4))
        for labeljob,g in rows.groupby('job'):plt.plot(g.step,g[metric],marker='o',label=labeljob)
        plt.xlabel('Completed optimizer updates');plt.ylabel(label);plt.legend(fontsize=7);save(name)
    plt.figure(figsize=(8,4))
    for name,g in rows.groupby('job'):plt.plot(g.adaptation_seconds,g.cpr,marker='o',label=name)
    plt.xlabel('Adaptation seconds (checkpoint prefix plus fixed setup)');plt.ylabel('Official CPR');plt.legend(fontsize=7);save('cpr_vs_compute')
    plt.figure(figsize=(8,4))
    latest=rows[rows.step==rows.step.max()]
    for _,row in latest.iterrows():plt.plot(np.array(FRACTIONS)*100,[row[f'accuracy_{f}'] for f in FRACTIONS],marker='o',label=row.job)
    plt.xlabel('Nominal retained edges (%) before official pruning');plt.ylabel('Official IOI margin > 0 accuracy');plt.legend(fontsize=7);save('compact_accuracy')
    plt.figure(figsize=(8,4));stability=pd.read_csv(out/'ranking_stability.csv')
    if stability.empty:plt.text(.5,.5,'No paired-seed run authorized:\ncross-seed hierarchy stability unmeasured',ha='center',va='center');plt.axis('off')
    else:stability[stability.level.isin(['edge','H1','H2'])].groupby('level').spearman.mean().plot.bar();plt.ylabel('Mean paired rank Spearman')
    save('component_vs_edge_stability')
    figure,axes=plt.subplots(1,2,figsize=(11,4))
    for name,g in grad.groupby('job'):
        axes[0].plot(g.step,g.gradient_ratio,label=name)
        axes[1].plot(g.step,g.adam_update_regularizer_relative_contribution,label=name)
    axes[0].set_ylabel('Regularizer / task gradient norm');axes[1].set_ylabel('Same-state Adam update change / total');axes[0].set_xlabel('Updates');axes[1].set_xlabel('Updates');axes[0].legend(fontsize=6);save('regularization_diagnostics')
    plt.figure(figsize=(8,4))
    for name,g in rows[rows.step==1000].groupby(['family','gamma','shuffled']):plt.plot(g.seed,g.cpr_delta,'o-',label=str(name))
    plt.axhline(0,color='black',linestyle='--');plt.xlabel('Paired training / subset seed');plt.ylabel('CPR difference from Gate2');plt.legend(fontsize=7);save('paired_cpr_differences')
    summary=rows[rows.step==1000].groupby(['family','gamma','shuffled']).agg(seeds=('seed','nunique'),cpr=('cpr','mean'),CPR_gain=('relative_cpr_gain','mean'),CMD_change=('cmd_delta','mean'),accuracy_log_AUC_change=('accuracy_auc_delta','mean'),time_ratio=('time_ratio','mean')).reset_index()
    summary.to_csv(out/'screen_summary.csv',index=False)
    body=f'''# Gate 5C: Regularized hierarchical MAttr

## Actual decision

{json.dumps(decisions,indent=2)}

## Hypothesis and mathematical objective

Shared component scores may stabilize edge importance without eliminating edge-specific deviations. Effective scores are h[g(e)] + r[e] - group_mean(r)[g(e)]. Every original32,491 edge remains eligible and trainable through this full-score-space parameterization. Zero h/r initialization; loss is official negative one-example IOI logit difference plus gamma * scale * mean(centered_residual^2). The component vector is unpenalized. A deterministic padded indexed reduction and its symmetric projection backward avoid scatter-atomic reduction ordering. No per-edge Python training loop, new top-k mask, edge restriction, model fine-tuning, expander or interaction training.

The official optimizer, uniform CPU Torch k RNG, Python example draws, temperature.5, Adam lr.05/eps1e-8/betas(.9,.999), iso live patching, float32 and MIB evaluator stay pinned. One packed Adam parameter holds h and r; this is mathematically the same elementwise Adam rule on both vectors. Model parameters never enter Adam, and their complete before/after SHA hashes must match. Official model-weight gradient flags/accumulation are preserved for exact baseline pairing; producing those gradients is not a weight update. Inference is no-grad.

## Prior research and mappings

Complete Gate1/2/3/4/5A/5B reports were read and hashed. Gate1 CPR10.588012 is a full-data/full-validation reference; matched N128 Gate2 checkpoints remain the primary comparator. Gate5B's stable components explain little edge-specific variance, so all residual directions remain available. H1 groups by official source component (including explicit input); H2 separates every head Q/K/V receiver, each MLP receiver and logits. Canonical ordering is unchanged. Shuffled controls use fixed seed5701 and preserve exact group-size distributions.

{table(pd.read_csv(out/'group_sizes.csv').groupby('hierarchy').edges.agg(['count','min','max','sum']).reset_index())}

## Training-only scale inspection and correctness

{json.dumps(p['regularization_scale'],indent=2)}

The probe uses saved500-step baseline scores, four training prompts, and five fixed k values, with20 model-gradient probes and zero optimizer updates. No development metric selects the scale. Original gamma values .001/.01 are retained unless both are numerically negligible or dominant under the previously frozen training-only rule. All pilot work is charged. CPU/GPU tests cover disjoint mappings, singleton groups, zero means, full score-space rank, finite differences, official soft-mask mass/order, exact checkpoint/Adam/RNG/group identity, deterministic export and a dedicated original-parameter bypass. Gamma-zero hierarchy is NOT claimed identical to ordinary MAttr. Real30-update smoke additionally verifies patching, exact resume and unchanged GPT2 weights.

Protocol SHA-256 `{sha(out/'gate5c_protocol.json')}`; source, cohort, dependency, group mapping and historical criteria hashes are frozen before CPR screening. The preflight/probe protocol is separate and contains no outcome-driven hyperparameter replacement. Historical Gate3/4 stopping criteria are unchanged.

## Official held-out development results

{table(summary)}

{table(rows[['job','seed','step','cpr','baseline_cpr','relative_cpr_gain','cmd_delta','accuracy_auc_delta','time_ratio',*[f'accuracy_delta_{f}' for f in FRACTIONS]]])}

Complete official faithfulness/accuracy curves and pruned counts are retained. Compact fractions are nominal ranking cutoffs followed by official connectivity pruning. Official accuracy here is positive IOI margin, not vocabulary top1. CPR rewards normalized logit amplification and is unbounded; higher CPR accompanied by higher CMD is not improved full-model fidelity. No homemade CPR/CMD is used.

## Controls, stopping and structural analysis

Seed42 screening requires>=.5% CPR gain, all compact and accuracy-log-AUC drops<=.02, CMD increase<=positive CPR gain+.05, <=10% adaptation/memory overhead. At most one candidate advances by CPR, compact accuracy, lower gamma and hierarchy tie breaks. PhaseC requires mean paired CPR>=2%, at least2 positive seeds, the same mean accuracy/CMD safeguards, <=10% adaptation overhead, and real hierarchy beating gamma-zero, size-matched shuffle and generic edge L2 controls by>=.5% mean CPR and>=2 seeds without accuracy regressions. Control failure is parameterization/regularization evidence at most, never a structural claim. Only a full PhaseC pass authorizes5000-step continuation with all necessary controls. Failed screening means no extra gamma values, no additional seeds or confirmation. Modified5000-step performance remains untested when stopped.

Learned h, centered residuals, their variances, effective score vectors, true-component score means and top-edge counts are saved. ranking_stability.csv reports paired edge/component Spearman and official pruned-edge Jaccard only when seeds are actually run. A one-seed pilot cannot establish ranking transferability. structural_component_statistics.csv distinguishes attention, MLP and schema categories with size-normalized variances. Smoother/stabler scores alone do not establish causal fidelity. Existing paired seeds vary both discovery subsets and optimizer RNG. No statistical significance claim from three seeds or the repeatedly reused128-prompt development cohort.

## Computation and limitations

{table(compute)}

Instrumented stages total **{compute.stage_seconds.sum()/60:.3f}min**; optimizer updates **{int(compute.optimizer_updates.sum()):,}**, backward calls **{int(compute.backward_calls.sum()):,}**, actual root forwards **{int(compute.forward_batches.sum()):,}**, example-forwards **{int(compute.example_forwards.sum()):,}**. All rejected variants, probes, bypass/reference checks, exact-resume checks and restarts are charged. Launch timers additionally include interpreter/source checks. Group construction is charged to deployable adaptation, CPU report/tests/audits are additional research work. The score-operation microbenchmark reports transform/MSE/backward overhead separately and does not isolate regularization time within the full model backward. Gradient diagnostics use analytic penalty gradients and a read-only same-state Adam counterfactual, not extra model backpropagation or a changed optimizer. Prefix500/2500 compute uses measured checkpoint training plus fixed setup; those are not separate launches. Cold restarts rebuild donor caches and are charged.

Peak allocated CUDA memory **{compute.peak_allocated_mib.max():.1f}MiB** is PyTorch allocator usage, not total physical VRAM occupancy. Sampled CPU RSS is saved. Exact evaluation cache hits add no forwards. No setting changes silently after OOM. Raw vectors/checkpoints and prompt data remain local; small summaries/configurations/logs/figures are versioned. Full-data Gate1 follow-up, fresh-cohort generalization, deferred ranking refinement, OptionA, further algorithms and leaderboard submission are not run automatically.

## Recommendation

Apply the frozen decisions above. A negative screen supports retaining ordinary MAttr and does not disprove hierarchy under all possible schedules. A positive development result still needs independent validation and a separately authorized full-data comparison. Stop at Gate5C.
'''
    for name in ('cpr_vs_steps','cpr_vs_compute','cmd_vs_steps','compact_accuracy','component_vs_edge_stability','regularization_diagnostics','paired_cpr_differences'):body+=f'\n![{name}](figures/{name}.png)\n'
    (out/'GATE5C_HIERARCHICAL_MATTR_REPORT.md').write_text(body)
    (ROOT/'GATE5C_HIERARCHICAL_MATTR_REPORT.md').write_text(body.replace('(figures/',f'({out.relative_to(ROOT).as_posix()}/figures/'))
