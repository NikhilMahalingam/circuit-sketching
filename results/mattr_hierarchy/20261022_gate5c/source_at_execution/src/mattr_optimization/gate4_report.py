"""Gate4 frozen decisions and artifact report; no model forwards."""
import json
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from .common import ROOT,sha,write_json
from .gate3 import baseline
from .gate2_final_analysis import table

FRACTIONS=(.005,.01,.02,.05)
SCALES=(.1,.5,1.,2.)
SEEDS=(42,43,44)


def protocol(out):
    p=out/'gate4_protocol.json'
    if sha(p)!=(out/'gate4_protocol.sha256').read_text().strip():raise ValueError('Protocol SHA changed.')
    return json.loads(p.read_text())


def metrics(folder,step,quality=None):
    quality=quality or folder/f'quality_{step}.json'
    q=json.loads(quality.read_text())
    curve=pd.read_csv(folder/f'evaluation_step{step}/faithfulness_curve.csv')
    for f in FRACTIONS:
        r=curve[np.isclose(curve.fraction,f)].iloc[0]
        q[f'accuracy_{f}']=float(r.accuracy);q[f'faithfulness_{f}']=float(r.faithfulness)
    return q


def base_metrics(out,seed,step):
    if step==2500:return metrics(out/f'baseline2500_seed{seed}',step,out/f'baseline2500_seed{seed}/quality.json')
    job=baseline(seed)
    q=pd.read_csv(job/'quality.csv').query("split=='validation' and step==@step").iloc[0].to_dict()
    curve=pd.read_csv(job/f'evaluation/validation_step{step}/faithfulness_curve.csv')
    for f in FRACTIONS:q[f'accuracy_{f}']=float(curve[np.isclose(curve.fraction,f)].iloc[0].accuracy)
    return q


def job_cost(folder,target=None):
    files=sorted((folder/'attempts').glob('*/compute_accounting.csv'))+[folder/'compute_accounting.csv']
    frames=[];processes=[]
    for f in files:
        if not f.exists():continue
        proc=json.loads((f.parent/'process_summary.json').read_text())
        if target==1000 and proc.get('target',1000)>1000:continue
        if not proc['success']:raise ValueError('Failed attempt requires explicit cost reconciliation.')
        frames.append(pd.read_csv(f));processes.append(proc)
    c=pd.concat(frames,ignore_index=True)
    is_eval=c.stage.str.contains('evaluation')
    return {'adaptation_seconds':sum(p.get('seconds',p.get('process_seconds',0)) for p in processes)-float(c[is_eval].seconds.sum()),
            'training_seconds':float(c[c.stage=='training'].seconds.sum()),'evaluation_seconds':float(c[is_eval].seconds.sum()),
            'stage_seconds':float(c.seconds.sum()),'peak_allocated_mib':float(c.peak_allocated_mib.max()),
            'cpu_rss_mib':float(c.sampled_peak_cpu_rss_mib.max()),
            **{k:int(c[k].sum()) for k in ('forward_batches','example_forwards','backward_calls','optimizer_updates')}}


def base_cost(seed,step):
    # Gate3 replay is bitwise identical and gives matched cold/setup scope.
    control=ROOT/f'results/mattr_optimization/20261022_gate3/uniform_seed{seed}'
    control_cost=job_cost(control)
    state=torch.load(baseline(seed)/f'checkpoints/step_{step:06d}.pt',map_location='cpu',weights_only=False)
    return {**control_cost,'adaptation_seconds':control_cost['adaptation_seconds']-control_cost['training_seconds']+state['training_seconds']}


def compare(out,scale,seed,step,baseline_step=None):
    baseline_step=baseline_step or step;folder=out/f'scale{scale:g}_seed{seed}'
    q=metrics(folder,step);b=base_metrics(out,seed,baseline_step)
    c=job_cost(folder,step);a=job_cost(out/f'attribution_seed{seed}');bc=base_cost(seed,baseline_step)
    total=c['adaptation_seconds']+a['adaptation_seconds']
    row={**q,'scale':scale,'baseline_step':baseline_step,'baseline_cpr':b['cpr'],'baseline_cmd':b['cmd'],
         'baseline_accuracy_auc':b['accuracy_auc'],'cpr_delta':q['cpr']-b['cpr'],
         'relative_cpr_gain':q['cpr']/b['cpr']-1,'cmd_delta':q['cmd']-b['cmd'],
         'accuracy_auc_delta':q['accuracy_auc']-b['accuracy_auc'],
         'adaptation_seconds':total,'baseline_adaptation_seconds':bc['adaptation_seconds'],
         'time_ratio':total/bc['adaptation_seconds'],
         'memory_ratio':max(a['peak_allocated_mib'],c['peak_allocated_mib'])/bc['peak_allocated_mib']}
    for f in FRACTIONS:row[f'accuracy_delta_{f}']=q[f'accuracy_{f}']-b[f'accuracy_{f}']
    return row


def gates(rows,p,efficiency=False,quality_only=False):
    s=p['success']['efficiency' if efficiency else 'primary']
    if efficiency:
        checks={'cpr':rows.relative_cpr_gain.mean()>=-s['CPR_relative_loss_max'],
                'cmd':rows.cmd_delta.mean()<=s['CMD_increase_max']}
        if not quality_only:checks['compute']=rows.adaptation_seconds.mean()/rows.baseline_adaptation_seconds.mean()<=1-s['total_adaptation_compute_reduction_min']
        acc=s['accuracy_drop_max'];auc=s['accuracy_auc_drop_max']
    else:
        checks={'cpr':rows.relative_cpr_gain.mean()>=s['mean_paired_relative_CPR_gain_min'],
                'positive_seeds':int((rows.cpr_delta>0).sum())>=s['positive_seeds_min'],
                'cmd':rows.cmd_delta.mean()<=max(0.,rows.cpr_delta.mean())+.05}
        if not quality_only:
            checks['compute']=rows.adaptation_seconds.mean()/rows.baseline_adaptation_seconds.mean()<=s['total_adaptation_time_ratio_max']
            checks['memory']=rows.memory_ratio.mean()<=s['peak_allocated_ratio_max']
        acc=s['mean_compact_accuracy_drop_max'];auc=s['mean_accuracy_log_auc_drop_max']
    checks['accuracy_auc']=rows.accuracy_auc_delta.mean()>=-auc
    for f in FRACTIONS:checks[f'accuracy_{f}']=rows[f'accuracy_delta_{f}'].mean()>=-acc
    return {'passed':all(bool(v) for v in checks.values()),'checks':{k:bool(v) for k,v in checks.items()}}


def frozen_decision(path,decision):
    if path.exists() and json.loads(path.read_text())!=decision:raise ValueError('Decision changed after freeze: '+str(path))
    if not path.exists():write_json(path,decision)
    return decision


def select_pilot(out):
    protocol(out)
    rows=[compare(out,s,42,1000) for s in SCALES]
    ranking=sorted(rows,key=lambda r:(-r['relative_cpr_gain'],r['scale']))
    decision={'exploratory':True,'selection_data':'seed42 development1000','scales':[r['scale'] for r in ranking[:2]],
              'rows':rows,'no_confirmatory_claim':True}
    frozen_decision(out/'pilot_decision.json',decision)
    return decision['scales']


def select_screen(out):
    p=protocol(out);scales=json.loads((out/'pilot_decision.json').read_text())['scales']
    results={};eligible=[]
    for scale in scales:
        rows=pd.DataFrame([compare(out,scale,s,1000) for s in SEEDS])
        eff=pd.DataFrame([compare(out,scale,s,1000,5000) for s in SEEDS])
        primary=gates(rows,p);efficiency=gates(eff,p,True)
        item={'primary_provisional':primary,'efficiency':efficiency,'mean_relative_cpr_gain':float(rows.relative_cpr_gain.mean())}
        results[str(scale)]=item
        if primary['passed'] or efficiency['passed']:eligible.append((float(rows.relative_cpr_gain.mean()),scale))
    winner=sorted(eligible,key=lambda r:(-r[0],r[1]))[0][1] if eligible else None
    frozen_decision(out/'screen_decision.json',{'winner':winner,'results':results,'exploratory':True})
    return winner


def confirm(out,winner):
    p=protocol(out)
    primary=gates(pd.DataFrame([compare(out,winner,s,5000) for s in SEEDS]),p)
    efficiency=gates(pd.DataFrame([compare(out,winner,s,1000,5000) for s in SEEDS]),p,True)
    return frozen_decision(out/'development_confirmation.json',{'scale':winner,'primary':primary,'efficiency':efficiency,
      'development_passed':primary['passed'] or efficiency['passed'],'selected_budget':5000 if primary['passed'] else 1000})


def compute_matches(out):
    selections=[];records=[]
    decision=json.loads((out/'screen_decision.json').read_text())
    scale=decision['winner']
    if scale is None:
        # A negative screen still gets one prespecified diagnostic: strongest paired
        # CPR screen, not a new candidate, with no additional training.
        candidates=decision['results'];scale=min(candidates,key=lambda k:(-candidates[k]['mean_relative_cpr_gain'],float(k)))
        scale=float(scale)
    for seed in SEEDS:
        target=compare(out,scale,seed,1000)['adaptation_seconds']
        choices=[]
        control=job_cost(ROOT/f'results/mattr_optimization/20261022_gate3/uniform_seed{seed}')
        setup=control['adaptation_seconds']-control['training_seconds']
        for f in baseline(seed).glob('checkpoints/step_*.pt'):
            step=int(f.stem.split('_')[1])
            if step<100 or step>5000:continue
            state=torch.load(f,weights_only=False,map_location='cpu')
            choices.append((abs(setup+state['training_seconds']-target),step,setup+state['training_seconds']))
        diff,step,seconds=min(choices)
        records.append({'scale':scale,'seed':seed,'initialized_step':1000,'baseline_step':step,
                        'initialized_adaptation_seconds':target,'baseline_adaptation_seconds':seconds,'absolute_time_mismatch_seconds':diff})
        selections.append((seed,step))
    pd.DataFrame(records).to_csv(out/'compute_matched_configuration.csv',index=False)
    return selections


def report(out):
    p=protocol(out);quality=[];curves=[];paired=[];costs=[];correlations=[];membership_frames=[]
    for seed in SEEDS:
        for step in (500,1000,5000):
            b=base_metrics(out,seed,step);quality.append({**b,'method':'zero','seed':seed,'step':step})
            c=pd.read_csv(baseline(seed)/f'evaluation/validation_step{step}/faithfulness_curve.csv')
            curves.extend(c.assign(method='zero',seed=seed,step=step).to_dict('records'))
        attr=out/f'attribution_seed{seed}'
        if (attr/'completed.json').exists():costs.append({'job':attr.name,**job_cost(attr)})
    for folder in sorted(out.iterdir()):
        if not folder.is_dir() or not (folder/'process_summary.json').exists() or folder.name.startswith('attribution'):continue
        if folder.name=='source_at_execution':continue
        costs.append({'job':folder.name,**job_cost(folder)})
        if folder.name.startswith('scale'):
            scale=float(folder.name.split('_')[0][5:]);seed=int(folder.name.split('seed')[1])
            for f in sorted(folder.glob('quality_*.json')):
                step=int(f.stem.split('_')[1]);q=metrics(folder,step)
                quality.append({**q,'method':f'scale{scale:g}'})
                c=pd.read_csv(folder/f'evaluation_step{step}/faithfulness_curve.csv')
                curves.extend(c.assign(method=f'scale{scale:g}',seed=seed,step=step).to_dict('records'))
                if step!=2500 or (out/f'baseline2500_seed{seed}/completed.json').exists():
                    paired.append(compare(out,scale,seed,step))
                initial=torch.load(out/f'attribution_seed{seed}/rank_normalized.pt',weights_only=True)
                final=torch.load(folder/f'checkpoints/step_{step:06d}.pt',weights_only=False,map_location='cpu')['scores']
                correlations.append({'scale':scale,'seed':seed,'step':step,'initial_final_spearman':float(spearmanr(initial.numpy(),final.numpy()).statistic)})
                m=pd.read_csv(folder/f'membership_{step}.csv');m['method']=f'scale{scale:g}';m['seed']=seed;membership_frames.append(m)
        elif folder.name.startswith('eap_only'):
            seed=int(folder.name.split('seed')[1]);q=metrics(folder,0,folder/'quality.json')
            quality.append({**q,'method':'EAP-IG only'})
            c=pd.read_csv(folder/'evaluation_step0/faithfulness_curve.csv');curves.extend(c.assign(method='EAP-IG only',seed=seed,step=0).to_dict('records'))
            m=pd.read_csv(folder/'membership_0.csv');m['method']='EAP-IG only';m['seed']=seed;membership_frames.append(m)
    q=pd.DataFrame(quality);c=pd.DataFrame(curves);d=pd.DataFrame(paired);cost=pd.DataFrame(costs);corr=pd.DataFrame(correlations)
    for name,frame in [('quality_by_step',q),('faithfulness_curves',c),('paired_results',d),('compute_accounting',cost),('ranking_correlations',corr)]:frame.to_csv(out/f'{name}.csv',index=False)
    compact=c[c.fraction.isin(FRACTIONS)];compact.to_csv(out/'compact_circuit_accuracy.csv',index=False)
    # Zero-reference retained sets are reconstructed with the pinned graph and
    # original score tensors, never a presentation CSV's arbitrary tie order.
    from .common import configure
    configure()
    from eap.graph import Graph
    for seed in SEEDS:
        graph=Graph.from_model({'n_layers':12,'n_heads':12,'d_model':768,'parallel_attn_mlp':False})
        for step in (500,1000,5000):
            scores=torch.load(baseline(seed)/f'checkpoints/step_{step:06d}.pt',weights_only=False,map_location='cpu')['scores']
            graph.scores[:]=-torch.inf;graph.scores[graph.real_edge_mask.bool()]=scores
            rows=[]
            for f in FRACTIONS:
                graph.apply_topn(int(f*32491),absolute=False,prune=True)
                rows.extend({'method':'zero','seed':seed,'step':step,'fraction':f,'edge':e.name} for e in graph.edges.values() if e.in_graph)
            membership_frames.append(pd.DataFrame(rows))
    stability=[]
    if membership_frames:
        members=pd.concat(membership_frames,ignore_index=True)
        for (method,step,fraction),frame in members.groupby(['method','step','fraction']):
            for a,b in itertools.combinations(sorted(frame.seed.unique()),2):
                x=set(frame[frame.seed==a].edge);y=set(frame[frame.seed==b].edge)
                stability.append({'method':method,'step':step,'fraction':fraction,'seed_a':int(a),'seed_b':int(b),
                                  'jaccard':len(x&y)/len(x|y) if x|y else 1.,'edges_a':len(x),'edges_b':len(y)})
    pd.DataFrame(stability).to_csv(out/'retained_edge_stability.csv',index=False)
    figs=out/'figures';figs.mkdir(exist_ok=True)
    def save(name):
        plt.gcf().suptitle('Development cohort — exploratory selection',fontsize=9)
        plt.tight_layout();plt.savefig(figs/(name+'.png'),dpi=150);plt.close()
    for metric in ('cpr','cmd'):
        plt.figure(figsize=(8,4))
        for name,g in q.groupby('method'):
            mean=g.groupby('step')[metric].mean();plt.plot(mean.index,mean.values,'o-',label=name)
        plt.xlabel('Optimizer steps');plt.ylabel('Official '+metric.upper());plt.legend(fontsize=7);save(metric+'_vs_steps')
    plt.figure(figsize=(8,4))
    if not d.empty:
        for scale,g in d.groupby('scale'):
            plt.scatter(g.adaptation_seconds,g.cpr,label=f'scale {scale:g}')
        plt.legend()
    for seed in SEEDS:
        for step in (1000,5000):
            plt.scatter(base_cost(seed,step)['adaptation_seconds'],base_metrics(out,seed,step)['cpr'],marker='x',color='black')
        eap_folder=out/f'eap_only_seed{seed}'
        if (eap_folder/'quality.json').exists():
            plt.scatter(job_cost(out/f'attribution_seed{seed}')['adaptation_seconds'],json.loads((eap_folder/'quality.json').read_text())['cpr'],marker='s',color='gray')
    plt.xlabel('EAP-IG + MAttr + required setup seconds');plt.ylabel('Official development CPR');save('cpr_vs_compute')
    plt.figure(figsize=(8,4))
    for name,g in compact[compact.step.isin([0,1000,5000])].groupby(['method','step']):
        mean=g.groupby('fraction').accuracy.mean();plt.plot(mean.index,mean.values,'o-',label=f'{name[0]} / {name[1]}')
    plt.xscale('log');plt.xlabel('Retained edge fraction');plt.ylabel('IOI decision accuracy');plt.legend(fontsize=6);save('compact_accuracy')
    plt.figure(figsize=(8,4))
    if not corr.empty:
        for scale,g in corr.groupby('scale'):
            mean=g.groupby('step').initial_final_spearman.mean();plt.plot(mean.index,mean.values,'o-',label=str(scale))
        plt.legend(title='Scale')
    plt.xlabel('Optimizer steps');plt.ylabel('Initial/final Spearman');save('initial_final_correlation')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    if not d.empty:
        for scale,g in d[d.step==1000].groupby('scale'):
            for ax,metric in zip(axes,('cpr_delta','cmd_delta')):
                ax.plot(g.seed,g[metric],'o-',label=str(scale));ax.axhline(0,color='gray');ax.set_ylabel(metric);ax.set_xlabel('Paired seed');ax.legend()
    save('paired_differences')
    screen=json.loads((out/'screen_decision.json').read_text()) if (out/'screen_decision.json').exists() else None
    final=json.loads((out/'development_confirmation.json').read_text()) if (out/'development_confirmation.json').exists() else None
    if screen is None:outcome='Preliminary campaign; screening is incomplete. No conclusion.'
    elif screen['winner'] is None:outcome='Negative result: neither prespecified paired screen passed. No 5,000-step confirmation, extra scales, Gate5, full-data follow-up or submission.'
    elif final is None:outcome='One exploratory screen winner selected; 5,000-step confirmation pending. No final-quality claim.'
    else:outcome='5,000-step development confirmation completed; see frozen checks below. Generalization requires the separate disjoint cohort checks.'
    summary=d[d.step==1000].groupby('scale').agg(seeds=('seed','count'),mean_cpr_delta=('cpr_delta','mean'),
      mean_relative_cpr_gain=('relative_cpr_gain','mean'),mean_cmd_delta=('cmd_delta','mean'),
      mean_accuracy_auc_delta=('accuracy_auc_delta','mean'),mean_time_ratio=('time_ratio','mean')) if not d.empty else pd.DataFrame()
    eap=q[q.method=='EAP-IG only'][['seed','cpr','cmd','accuracy_auc']]
    text='# Gate 4 EAP-IG initialization for MAttr\n\n'+outcome+'\n\n'
    text+=f'Frozen protocol SHA-256: `{sha(out/"gate4_protocol.json")}`. Historical Gate3 criteria remain `{p["historical_gate3_criteria_sha256"]}`. Parent commit `{p["parent_commit"]}`.\n\n'
    text+='## Implementation and attribution audit\n\nOfficial pinned EAP-IG inputs, five **left-endpoint** alphas 0/.2/.4/.6/.8, patching, signed negative-logit-difference loss, batch4. Official activation difference is corrupted minus clean; multiplication by the loss gradient gives estimated loss increase on corruption. Positive scores favor retention; no sign flipping or absolute values. Matrix logits column -1 is normalized to its positive flatten coordinate. Every one of 32,491 attention-q/k/v, MLP and logits edges is mapped bijectively and sentinel-tested. Rank initialization uses ascending average ranks centered in [-1,1]; ties remain equal. Only trainer-local Parameter construction changes; Adam, implicit mask derivatives, edge patches and evaluator are untouched. EAP model weights are frozen, with gradients propagated from interpolated embeddings. Model weights are never optimized.\n\n'
    text+='The four-training-prompt smoke cost, training-only sign diagnostics and all scale/mask saturation/soft-mass/gradient checks are saved under smoke_seed42/. Attribution vectors and exact canonical mappings are under attribution_seed*/. Smoke used a prefix only and was never reused as a full128 attribution.\n\n'
    text+='## Selection and evidence scope\n\nSeed42 chooses at most two scales exploratorily by1000-step CPR, with lower scale breaking ties; these two must pass all paired-seed gates. Seed42 is reused, not an independent confirmation. No significance claim from three seeds. At most one winner continues, preserving full Adam/RNG state. Success requires mean paired relative CPR gain>=2%, positive differences in>=2 seeds, mean compact and log-AUC losses<=.02, CMD guard, <=10% total adaptation-time and peak-allocation overhead. Efficiency requires>=20% total adaptation saving, CPR loss<=1%, CMD increase<=.10 and accuracy losses<=.02 versus5000. A primary CPR gain is not general fidelity improvement unless CMD also decreases.\n\n'
    text+='## Screening summary\n\n'+table(summary.reset_index())+'\n\n## Per-seed matched-budget comparisons\n\n'
    columns=['scale','seed','step','cpr','baseline_cpr','cpr_delta','relative_cpr_gain','cmd_delta','accuracy_auc_delta','time_ratio']
    text+=table(d[columns]) if not d.empty else 'Pending.'
    text+='\n\n## Attribution-only control (zero MAttr updates)\n\n'+table(eap)+'\n\n'
    text+='Learned-initialization gains must be distinguished from what the EAP-IG ranking already achieves at step0. Quality/curves retain this control separately. Full10,000-example Gate1 remains the historical reference, not replaced by N128.\n\n'
    text+='## Frozen decisions\n\n```json\n'+json.dumps({'screen':screen,'development_confirmation':final},indent=2)+'\n```\n\n'
    text+='## Actual research computation\n\n'+table(cost)+'\n\n'
    if not cost.empty:text+=f'Summed instrumented stages: {cost.stage_seconds.sum()/60:.3f}min; real forward batches {cost.forward_batches.sum():,}; example-forwards {cost.example_forwards.sum():,}; backward calls {cost.backward_calls.sum():,}; MAttr updates {cost.optimizer_updates.sum():,}. All attribution, pilot, rejected scales, continuation restarts and frozen control evaluations are charged. Peak CUDA allocation and sampled CPU RSS are measured per stage; allocator reservation differs from physical GPU occupancy.\n\n'
    text+='Deployable adaptation cost includes one full128 EAP-IG job, initial model/data/setup and MAttr (including continuation cold restarts); validation evaluation is separate. Ordinary baseline setup uses the contemporaneous, bitwise Gate2-equivalent Gate3 replay, combined with the actual Gate2 checkpoint training timer. Hardware/run-order variability remains; these are measured times, not FLOP estimates. Pure CPU protocol/report analysis is additional research overhead. Initial1000 versus5000 comparisons are in efficiency_comparisons.csv. Nearest100-step ordinary checkpoints give a frozen, approximately time-matched control; actual timing mismatch is recorded, not hidden. No ordinary training is extended beyond the registered5000 budget.\n\n'
    efficiencies=[]
    for scale in json.loads((out/'pilot_decision.json').read_text())['scales'] if (out/'pilot_decision.json').exists() else []:
        for seed in SEEDS:
            if (out/f'scale{scale:g}_seed{seed}/completed_1000.json').exists():efficiencies.append(compare(out,scale,seed,1000,5000))
    pd.DataFrame(efficiencies).to_csv(out/'efficiency_comparisons.csv',index=False)
    matches=out/'compute_matched_configuration.csv'
    if matches.exists():
        cm=pd.read_csv(matches);rows=[]
        for r in cm.to_dict('records'):
            f=out/f'compute_matched_seed{int(r["seed"])}'
            if (f/'quality.json').exists():rows.append({**r,**json.loads((f/'quality.json').read_text())})
        pd.DataFrame(rows).to_csv(out/'compute_matched_results.csv',index=False)
        text+='## Approximate fair-compute ordinary reference\n\n'+table(pd.DataFrame(rows))+'\n\n'
    confirmations=[]
    for seed in SEEDS:
        a=out/f'confirmation_initialized_seed{seed}';b=out/f'confirmation_baseline_seed{seed}'
        if (a/'quality.json').exists() and (b/'quality.json').exists():
            qa=json.loads((a/'quality.json').read_text());qb=json.loads((b/'quality.json').read_text())
            x=metrics(a,qa['step'],a/'quality.json');y=metrics(b,5000,b/'quality.json')
            row={'seed':seed,'cpr_delta':x['cpr']-y['cpr'],'relative_cpr_gain':x['cpr']/y['cpr']-1,
                 'cmd_delta':x['cmd']-y['cmd'],'accuracy_auc_delta':x['accuracy_auc']-y['accuracy_auc']}
            for f in FRACTIONS:row[f'accuracy_delta_{f}']=x[f'accuracy_{f}']-y[f'accuracy_{f}']
            confirmations.append(row)
    if confirmations:
        frame=pd.DataFrame(confirmations);frame.to_csv(out/'disjoint_confirmation_results.csv',index=False)
        result=gates(frame,p,efficiency=final['selected_budget']==1000,quality_only=True) if len(frame)==3 else {'passed':False,'incomplete':True}
        write_json(out/'disjoint_confirmation_decision.json',result)
        text+='## Disjoint official validation confirmation\n\n'+table(frame)+'\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n\n'
    text+='## Plots\n\n'
    for name in ('cpr_vs_steps','cpr_vs_compute','cmd_vs_steps','compact_accuracy','initial_final_correlation','paired_differences'):
        text+=f'![{name}](results/mattr_optimization/{out.name}/figures/{name}.png)\n\n'
    text+='No Gate5, additional sweeps, full-data follow-up or leaderboard submission.\n'
    (out/'GATE4_EAPIG_INITIALIZATION_REPORT.md').write_text(text,encoding='utf-8')
    (ROOT/'GATE4_EAPIG_INITIALIZATION_REPORT.md').write_text(text,encoding='utf-8')
