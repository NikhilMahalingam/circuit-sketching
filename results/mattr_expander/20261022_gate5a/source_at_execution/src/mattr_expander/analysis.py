"""Frozen selection rules and result analysis; no model forward calls."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from src.mattr_optimization.common import ROOT,sha,write_json,configure
from src.mattr_optimization.gate3 import baseline
from src.mattr_optimization.gate2_final_analysis import table
from src.mattr_optimization.gate4_report import job_cost

FRACTIONS=(.005,.01,.02,.05)


def protocol(out):
    path=out/'gate5a_protocol.json'
    if sha(path)!=(out/'gate5a_protocol.sha256').read_text().strip():raise ValueError('Protocol bytes changed.')
    return json.loads(path.read_text())


def folder(out,family,beta,seed):return out/f'jobs/{family}_b{beta:g}_s{seed}'


def quality(job,step,filename=None,curve_path=None):
    q=json.loads((filename or job/f'quality_{step}.json').read_text())
    c=pd.read_csv(curve_path or job/f'evaluation_step{step}/faithfulness_curve.csv')
    for f in FRACTIONS:
        row=c[np.isclose(c.fraction,f)].iloc[0];q[f'accuracy_{f}']=float(row.accuracy)
    return q


def base_quality(out,seed,step):
    if step==2500:
        job=out/f'baseline_2500_s{seed}'
        return quality(job,step,job/'quality.json')
    q=pd.read_csv(baseline(seed)/'quality.csv').query("split=='validation' and step==@step").iloc[0].to_dict()
    c=pd.read_csv(baseline(seed)/f'evaluation/validation_step{step}/faithfulness_curve.csv')
    for f in FRACTIONS:q[f'accuracy_{f}']=float(c[np.isclose(c.fraction,f)].iloc[0].accuracy)
    return q


def effective_scores(out,job,step):
    from .coupling import Coupling
    config=json.loads((job/'experiment_config.json').read_text())
    state=torch.load(job/f'checkpoints/step_{step:06d}.pt',weights_only=False,map_location='cpu')
    family=config['family'];idx=None
    if family not in ('zero','scale'):idx=torch.tensor(np.load(out/'graphs'/f'{family}_neighbors.npy'),dtype=torch.long)
    return Coupling(idx,config['beta'],'zero' if family=='zero' else 'scale' if family=='scale' else 'graph')(state['scores']).detach()


def setup_cost(seed):
    prior=ROOT/f'results/mattr_optimization/20261022_gate3/uniform_seed{seed}'
    c=job_cost(prior)
    return c['adaptation_seconds']-c['training_seconds']


def baseline_cost(out,seed,step):
    job=folder(out,'zero',0,seed)
    # Fresh identity replay at1000 provides matched instrumentation/cold setup.
    # Longer baselines retain Gate2's actual measured checkpoint training timer.
    if (job/'completed_1000.json').exists():
        c=job_cost(job,1000);setup=c['adaptation_seconds']-c['training_seconds']
        peak=c['peak_allocated_mib']
        if step==1000:return c['adaptation_seconds'],peak
    else:
        setup=setup_cost(seed)
        peak=float(pd.read_csv(baseline(seed)/'compute_accounting.csv').peak_allocated_mib.max())
    state=torch.load(baseline(seed)/f'checkpoints/step_{step:06d}.pt',weights_only=False,map_location='cpu')
    return setup+state['training_seconds'],peak


def candidate_cost(out,job,step):
    target=1000 if step<=1000 else 5000
    c=job_cost(job,target)
    if step in (500,2500):
        state=torch.load(job/f'checkpoints/step_{step:06d}.pt',weights_only=False,map_location='cpu')
        # Before continuation the raw checkpoint timer already accumulates the
        # screen prefix. Fixed cold/restart setup is separately retained in c.
        seconds=c['adaptation_seconds']-c['training_seconds']+state['training_seconds']
    else:seconds=c['adaptation_seconds']
    config=json.loads((job/'experiment_config.json').read_text())
    rows=json.loads((out/'graphs_metadata.json').read_text())
    construction=next((r['required_construction_seconds'] for r in rows if r['family']==config['family']),0.)
    return seconds+construction,c['peak_allocated_mib']


def compare(out,family,beta,seed,step,against=None,baseline_step=None):
    job=folder(out,family,beta,seed);q=quality(job,step);baseline_step=baseline_step or step
    if against is None:
        b=base_quality(out,seed,baseline_step);seconds,peak=baseline_cost(out,seed,baseline_step)
        label='official_Gate2'
    else:
        other=folder(out,against,beta,seed);b=quality(other,baseline_step)
        seconds,peak=candidate_cost(out,other,baseline_step);label=against
    actual,actual_peak=candidate_cost(out,job,step)
    row={**q,'reference':label,'baseline_step':baseline_step,'baseline_cpr':b['cpr'],'baseline_cmd':b['cmd'],
         'cpr_delta':q['cpr']-b['cpr'],'relative_cpr_gain':q['cpr']/b['cpr']-1,'cmd_delta':q['cmd']-b['cmd'],
         'accuracy_auc_delta':q['accuracy_auc']-b['accuracy_auc'],'adaptation_seconds':actual,
         'baseline_adaptation_seconds':seconds,'time_ratio':actual/seconds,'memory_ratio':actual_peak/peak}
    for f in FRACTIONS:row[f'accuracy_delta_{f}']=q[f'accuracy_{f}']-b[f'accuracy_{f}']
    return row


def guards(rows,p,efficiency=False,pilot=False):
    c=p['criteria'];e=c['efficiency']
    if efficiency:
        checks={'CPR_noninferiority':rows.relative_cpr_gain.mean()>=-e['CPR_loss_max'],
                'CMD_guard':rows.cmd_delta.mean()<=e['CMD_increase_max'],
                'compute_saving':rows.adaptation_seconds.mean()/rows.baseline_adaptation_seconds.mean()<=1-e['compute_saving_min']}
    else:
        checks={'CPR_gain':rows.relative_cpr_gain.mean()>=(.005 if pilot else c['primary_CPR_gain']),
          'CMD_guard':rows.cmd_delta.mean()<=max(0.,rows.cpr_delta.mean())+.05,
          'time_guard':rows.adaptation_seconds.mean()/rows.baseline_adaptation_seconds.mean()<=c['adaptation_time_ratio_max'],
          'memory_guard':rows.memory_ratio.mean()<=c['allocated_memory_ratio_max']}
        if not pilot:checks['positive_seeds']=int((rows.cpr_delta>0).sum())>=c['positive_seeds']
    checks['accuracy_auc_guard']=rows.accuracy_auc_delta.mean()>=-c['accuracy_auc_drop']
    for f in FRACTIONS:checks[f'compact_{f}']=rows[f'accuracy_delta_{f}'].mean()>=-c['compact_drop']
    return {'passed':all(bool(x) for x in checks.values()),'checks':{k:bool(v) for k,v in checks.items()}}


def save_decision(path,value):
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Frozen decision changed: '+str(path))
    if not path.exists():write_json(path,value)


def phase_b(out):
    p=protocol(out);eligible=[];records=[]
    for family in ('G0','G1','G2'):
        for beta in p['betas']:
            r=compare(out,family,beta,42,1000);row=pd.DataFrame([r])
            checks=guards(row,p,pilot=True)
            eff=guards(pd.DataFrame([compare(out,family,beta,42,1000,baseline_step=5000)]),p,efficiency=True)
            records.append({'family':family,'beta':beta,'primary_plausibility':checks,'efficiency':eff,
                            'relative_cpr_gain':r['relative_cpr_gain']})
            if checks['passed'] or eff['passed']:
                compact=float(np.mean([r[f'accuracy_delta_{f}'] for f in FRACTIONS]))
                eligible.append((r['relative_cpr_gain'],compact,beta,family))
    chosen=sorted(eligible,key=lambda x:(-x[0],-x[1],x[2],x[3]))[:2]
    selected=[{'family':r[3],'beta':r[2]} for r in chosen]
    save_decision(out/'phaseB_decision.json',{'exploratory':True,'selected':selected,'results':records,
                 'stopped':not bool(selected),'no_unregistered_sweep':True})
    return selected


def topology_checks(out,beta,step,p):
    results={}
    for comparator in ('G0','G1','scale'):
        rows=pd.DataFrame([compare(out,'G2',beta,s,step,against=comparator) for s in p['seeds']])
        checks={'relative_CPR_gain':rows.relative_cpr_gain.mean()>=.005,'positive_seeds':int((rows.cpr_delta>0).sum())>=2,
                'accuracy_auc':rows.accuracy_auc_delta.mean()>=-.02}
        for f in FRACTIONS:checks[f'compact_{f}']=rows[f'accuracy_delta_{f}'].mean()>=-.02
        results[comparator]={'passed':all(bool(v) for v in checks.values()),'checks':{k:bool(v) for k,v in checks.items()}}
    return results


def phase_c(out):
    p=protocol(out);selected=json.loads((out/'phaseB_decision.json').read_text())['selected'];eligible=[];results=[]
    for item in selected:
        family,beta=item['family'],item['beta']
        rows=pd.DataFrame([compare(out,family,beta,s,1000) for s in p['seeds']])
        primary=guards(rows,p);eff=guards(pd.DataFrame([compare(out,family,beta,s,1000,baseline_step=5000) for s in p['seeds']]),p,True)
        topo=topology_checks(out,beta,1000,p)
        results.append({**item,'primary_provisional':primary,'efficiency':eff,'G2_topology_controls':topo,
                        'mean_relative_cpr_gain':float(rows.relative_cpr_gain.mean())})
        if primary['passed'] or eff['passed']:eligible.append((float(rows.relative_cpr_gain.mean()),beta,family))
    eligible.sort(key=lambda x:(-x[0],x[1],x[2]));winner={'family':eligible[0][2],'beta':eligible[0][1]} if eligible else None
    save_decision(out/'phaseC_decision.json',{'winner':winner,'results':results,'development_only':True})
    return winner


def final_assessment(out,winner):
    p=protocol(out);family,beta=winner['family'],winner['beta']
    primary=guards(pd.DataFrame([compare(out,family,beta,s,5000) for s in p['seeds']]),p)
    efficiency=guards(pd.DataFrame([compare(out,family,beta,s,1000,baseline_step=5000) for s in p['seeds']]),p,True)
    topo=topology_checks(out,beta,5000,p)
    permutation=None
    if family=='G2':permutation=guards(pd.DataFrame([compare(out,'G2perm',beta,s,1000) for s in p['seeds']]),p)
    specific=family=='G2' and primary['passed'] and all(v['passed'] for v in topo.values()) and permutation is not None and permutation['passed']
    save_decision(out/'final_decision.json',{'winner':winner,'CPR_quality':primary,'efficiency':efficiency,
                  'topology_controls':topo,'permutation':permutation,'expander_specific':bool(specific),
                  'generalization_established':False})


def report(out):
    p=protocol(out);quality_rows=[];pairs=[];curves=[];costs=[];members=[];structural=[];diagnostics=[]
    configure()
    from eap.graph import Graph
    mapping=pd.read_csv(out/'edge_mapping.csv',keep_default_na=False)
    graph=Graph.from_model({'n_layers':12,'n_heads':12,'d_model':768,'parallel_attn_mlp':False})
    def add_structure(scores,family,beta,seed,step):
        f=mapping.copy();f['score']=scores.numpy();f['family']=family;f['beta']=beta;f['seed']=seed;f['step']=step
        f['source_type']=f.parent.map(lambda s:'attention' if s.startswith('a') else 'MLP' if s.startswith('m') else 'input')
        f['destination_type']=f.child.map(lambda s:'attention' if s.startswith('a') else 'MLP' if s.startswith('m') else 'logits')
        f['source_component']=f.parent;f['destination_component']=f.child
        for axis in ('source_type','destination_type','source_component','destination_component'):
            summary=f.groupby(['family','beta','seed','step',axis]).score.agg(['mean','std','min','max','count']).reset_index()
            summary['grouping']=axis;summary=summary.rename(columns={axis:'category'})
            structural.extend(summary.to_dict('records'))
    for seed in p['seeds']:
        for step in (500,1000,5000):
            q=base_quality(out,seed,step);quality_rows.append({**q,'family':'Gate2','beta':0,'seed':seed,'step':step})
            curve=pd.read_csv(baseline(seed)/f'evaluation/validation_step{step}/faithfulness_curve.csv')
            curves.extend(curve.assign(family='Gate2',beta=0,seed=seed,step=step).to_dict('records'))
            state=torch.load(baseline(seed)/f'checkpoints/step_{step:06d}.pt',weights_only=False,map_location='cpu')
            graph.scores[:]=-torch.inf;graph.scores[graph.real_edge_mask.bool()]=state['scores']
            for fraction in FRACTIONS:
                graph.apply_topn(int(fraction*32491),absolute=False,prune=True)
                members.extend({'family':'Gate2','beta':0,'seed':seed,'step':step,'fraction':fraction,'edge':e.name} for e in graph.edges.values() if e.in_graph)
            add_structure(state['scores'],'Gate2',0,seed,step)
    for job in sorted((out/'jobs').glob('*')):
        if not (job/'completed_1000.json').exists():continue
        config=json.loads((job/'experiment_config.json').read_text());family,beta,seed=config['family'],config['beta'],config['seed']
        costs.append({'job':job.name,**job_cost(job)})
        for f in job.glob('gradient_diagnostics_*.csv'):
            diagnostics.extend(pd.read_csv(f).assign(family=family,beta=beta,seed=seed).to_dict('records'))
        for path in sorted(job.glob('quality_*.json')):
            step=int(path.stem.split('_')[1]);q=quality(job,step)
            quality_rows.append(q)
            curve=pd.read_csv(job/f'evaluation_step{step}/faithfulness_curve.csv')
            curves.extend(curve.assign(family=family,beta=beta,seed=seed,step=step).to_dict('records'))
            if step!=2500 or (out/f'baseline_2500_s{seed}/completed.json').exists():pairs.append(compare(out,family,beta,seed,step))
            m=pd.read_csv(job/f'membership_{step}.csv');members.extend(m.assign(family=family,beta=beta,seed=seed).to_dict('records'))
            add_structure(effective_scores(out,job,step),family,beta,seed,step)
    for job in [out/'phaseA_smoke',*out.glob('baseline_2500_*')]:
        if (job/'process_summary.json').exists():costs.append({'job':job.name,**job_cost(job)})
    q=pd.DataFrame(quality_rows);c=pd.DataFrame(curves);d=pd.DataFrame(pairs);cost=pd.DataFrame(costs);m=pd.DataFrame(members)
    for name,frame in [('quality_by_step',q),('faithfulness_curves',c),('paired_results',d),('compute_accounting',cost),
                       ('gradient_diagnostics',pd.DataFrame(diagnostics)),('structural_score_statistics',pd.DataFrame(structural))]:
        frame.to_csv(out/f'{name}.csv',index=False)
    c[c.fraction.isin(FRACTIONS)].to_csv(out/'compact_accuracy.csv',index=False)
    m.to_csv(out/'retained_edges.csv',index=False)
    stability=[]
    for (family,beta,step,fraction),g in m.groupby(['family','beta','step','fraction']):
        for a,b in itertools.combinations(sorted(g.seed.unique()),2):
            x=set(g[g.seed==a].edge);y=set(g[g.seed==b].edge)
            stability.append({'family':family,'beta':beta,'step':step,'fraction':fraction,'seed_a':int(a),'seed_b':int(b),
                              'jaccard':len(x&y)/len(x|y) if x|y else 1.})
    stab=pd.DataFrame(stability);stab.to_csv(out/'ranking_stability.csv',index=False)
    topology=[];eff=[]
    for family in ('G0','G1','G2','scale','G2perm'):
        for beta in p['betas']:
            for seed in p['seeds']:
                job=folder(out,family,beta,seed)
                if not (job/'completed_1000.json').exists():continue
                eff.append(compare(out,family,beta,seed,1000,baseline_step=5000))
                if family=='G2':
                    for other in ('G0','G1','scale'):
                        if (folder(out,other,beta,seed)/'completed_1000.json').exists():topology.append(compare(out,family,beta,seed,1000,against=other))
    pd.DataFrame(topology).to_csv(out/'topology_comparisons.csv',index=False);pd.DataFrame(eff).to_csv(out/'efficiency_comparisons.csv',index=False)
    # Approximately time-matched ordinary checkpoints, no extra training/evaluation:
    # only previously evaluated500/1000/5000 baselines are eligible here.
    matched=[]
    for r in d[d.step==1000].to_dict('records') if not d.empty else []:
        candidates=[(abs(baseline_cost(out,int(r['seed']),s)[0]-r['adaptation_seconds']),s) for s in (500,1000,5000)]
        _,step=min(candidates);base=base_quality(out,int(r['seed']),step);seconds,_=baseline_cost(out,int(r['seed']),step)
        matched.append({'family':r['family'],'beta':r['beta'],'seed':r['seed'],'candidate_step':1000,'baseline_step':step,
          'candidate_CPR':r['cpr'],'baseline_CPR':base['cpr'],'candidate_seconds':r['adaptation_seconds'],
          'baseline_seconds':seconds,'absolute_time_mismatch':abs(seconds-r['adaptation_seconds'])})
    pd.DataFrame(matched).to_csv(out/'matched_compute.csv',index=False)
    spectral=pd.read_csv(out/'graph_spectral_statistics.csv');figures=out/'figures';figures.mkdir(exist_ok=True)
    def finish(name):
        plt.gcf().suptitle('Gate5A — development-cohort evidence',fontsize=10)
        plt.tight_layout();plt.savefig(figures/(name+'.png'),dpi=150);plt.close()
    for metric in ('cpr','cmd'):
        plt.figure(figsize=(9,4))
        for (family,beta),g in q.groupby(['family','beta']):
            v=g.groupby('step')[metric].mean();plt.plot(v.index,v.values,'o-',label=f'{family}, beta={beta:g}')
        plt.xlabel('Optimizer updates');plt.ylabel('Official '+metric.upper());plt.legend(fontsize=6);finish(metric+'_vs_updates')
    plt.figure(figsize=(9,4))
    if not d.empty:
        for (family,beta),g in d.groupby(['family','beta']):plt.scatter(g.adaptation_seconds,g.cpr,label=f'{family}, beta={beta:g}')
        plt.legend(fontsize=6)
    plt.xlabel('Total adaptation seconds (includes graph construction)');plt.ylabel('Official CPR');finish('cpr_vs_compute')
    plt.figure(figsize=(8,4))
    if not d.empty:
        for family in ('G0','G1','G2'):
            gap=float(spectral[spectral.family==family].iloc[0].laplacian_gap)
            for beta in p['betas']:
                values=d[(d.family==family)&(d.beta==beta)&(d.step==1000)]
                if not values.empty:plt.scatter([gap]*len(values),values.cpr,label=f'{family}, beta={beta:g}')
        plt.xscale('log');plt.legend(fontsize=7)
    plt.xlabel('Normalized Laplacian gap — one graph per family, exploratory');plt.ylabel('Official CPR at1000');finish('cpr_vs_gap')
    plt.figure(figsize=(9,4))
    for (family,beta),g in c[c.step==1000].groupby(['family','beta']):
        v=g.groupby('fraction').accuracy.mean();plt.plot(v.index,v.values,'o-',label=f'{family}, beta={beta:g}')
    plt.xscale('log');plt.xlabel('Retained edge fraction');plt.ylabel('Official margin>0 accuracy');plt.legend(fontsize=6);finish('compact_accuracy')
    plt.figure(figsize=(9,4))
    if not d.empty:
        for (family,beta),g in d[d.step==1000].groupby(['family','beta']):plt.plot(g.seed,g.cpr_delta,'o-',label=f'{family}, beta={beta:g}')
        plt.axhline(0,color='gray');plt.legend(fontsize=6)
    plt.xlabel('Paired training seed');plt.ylabel('CPR difference vs Gate2');finish('paired_cpr')
    plt.figure(figsize=(9,4))
    if not stab.empty:
        for (family,beta,step),g in stab.groupby(['family','beta','step']):
            v=g.groupby('fraction').jaccard.mean();plt.plot(v.index,v.values,'o-',label=f'{family}, b={beta:g}, step={step}')
        plt.legend(fontsize=6)
    plt.xscale('log');plt.xlabel('Retained fraction');plt.ylabel('Cross-seed retained-edge Jaccard');finish('ranking_stability')
    b=json.loads((out/'phaseB_decision.json').read_text()) if (out/'phaseB_decision.json').exists() else None
    cc=json.loads((out/'phaseC_decision.json').read_text()) if (out/'phaseC_decision.json').exists() else None
    final=json.loads((out/'final_decision.json').read_text()) if (out/'final_decision.json').exists() else None
    if b is None:outcome='Correctness/pilot execution is incomplete; no research conclusion.'
    elif not b['selected']:outcome='Negative PhaseB result: no graph/beta passed the registered plausibility/efficiency screen. PhaseC and PhaseD skipped; no extra sweep.'
    elif cc is None:outcome='Exploratory PhaseB selections only; paired PhaseC pending.'
    elif cc['winner'] is None:outcome='Negative paired PhaseC result. No registered winner; PhaseD skipped.'
    else:outcome='Paired development screening completed. Consult final frozen checks; no generalization/leaderboard claim.'
    text='# Gate5A: Expander-coupled Matryoshka Attribution\n\n'+outcome+'\n\n'
    text+=f'Protocol SHA-256 `{sha(out/"gate5a_protocol.json")}`; implementation parent `{p["parent_commit"]}`. Historical Gate3 criteria remain `{p["historical_gate3_criteria_sha256"]}`.\n\n'
    text+='## Hypothesis and implementation\n\nAn auxiliary degree8 graph couples all32,491 learnable transformer-edge scores: s_effective=(1-beta)s+beta Ps. It changes score optimization, never transformer-edge eligibility. Raw scores initialize at zero; official Adam/lr.05, uniform-k, temperature.5, implicit sigmoid-top-k, live edge patching, model-weight gradient behavior and float32 remain unchanged. Only trainer-local build_mask receives effective scores. Official evaluation/export uses effective scores; raw scores and original Adam/RNG states remain in local checkpoints for exact resume. For undirected regular graphs and beta<.5 the linear map is invertible: this is reparameterization/conditioning, not an enlarged hypothesis class.\n\n'
    text+='A custom symmetric linear backward uses the same fixed-degree gather/reduction as forward, P^T=P, avoiding CUDA scatter-add ordering. CPU/GPU values, gradients, finite differences, soft mass, beta0 exact trainer behavior and checkpoint continuation are tested. Graph8-neighbor indices are immutable during training. No N-square dense matrix is constructed.\n\n'
    text+='## Graph construction and measured spectra\n\nG0 is the +/-1..4 ring; G1 applies E//4=32,491 successful degree-preserving swaps; G2 is seeded NetworkX random regular. All use graph seed1701 and the identical canonical real-edge mapping across training seeds. G2 is labeled a candidate until its spectrum is measured. G0 uses analytic circulant eigenvalues; G1/G2 use sparse ARPACK and residual checks. Permutation2701 is fixed in advance and is used only for a selected G2 winner.\n\n'
    text+=table(spectral[['family','vertices','edges','degree_min','degree_max','components','laplacian_gap','nontrivial_adjacency_magnitude','generation_seconds','spectral_seconds','neighbor_int32_bytes']])+'\n\n'
    text+='Detailed sparse memory, generation audits, eigen residuals and adjacency hashes are in graph metadata. The tiny ring gap may round to0.0000 above; the CSV/JSON retains full precision. Canonical edge identities are saved and hashed, including a compressed mapping; no score reordering occurs.\n\n'
    text+='## Correctness and controls\n\nPhaseA combines CPU graph/spectrum/gradient/official-Adam tests, full-size GPU mixing comparisons, exact beta0 GPU30-update replay, the existing official hard-mask patching oracle checks at0/.1/.5/1, then G2 beta.25 smoke30 and official frozen evaluation. Every later variant matches exact Gate2 train/dev manifests, actual row draws and uniform-k draws. The fresh zero1000 calibration must reproduce Gate2 scores bitwise. Controls use identical diagnostic cadence and precision.\n\n'
    text+='## Actual held-out development results\n\n'
    if not d.empty:
        text+=table(d[['family','beta','seed','step','cpr','baseline_cpr','cpr_delta','relative_cpr_gain','cmd_delta','accuracy_auc_delta','time_ratio']])+'\n\n'
    text+='All PhaseB results have one seed and are exploratory. Only paired PhaseC/D comparisons can meet reproducibility gates; three seeds do not establish statistical significance. No causal spectral-gap inference from one graph instance. Lower CMD indicates closer full-model margins; a CPR gain due to margin amplification is not automatically better behavioral fidelity. Compact accuracy is official IOI margin>0, not full-vocabulary top1 accuracy.\n\n'
    text+='## Registered decisions\n\n```json\n'+json.dumps({'PhaseB':b,'PhaseC':cc,'final':final},indent=2)+'\n```\n\n'
    text+='## Compute and memory\n\n'+table(cost)+'\n\n'
    if not cost.empty:text+=f'Instrumented model stages: {cost.stage_seconds.sum()/60:.3f}min; real root forward batches {cost.forward_batches.sum():,}; example-forwards {cost.example_forwards.sum():,}; backward calls {cost.backward_calls.sum():,}; optimizer updates {cost.optimizer_updates.sum():,}.\n\n'
    launch=sum(json.loads(f.read_text())['seconds'] for f in out.rglob('launch_elapsed_*.json'))
    text+=f'Actual child-process timers sum to **{launch/60:.3f}min**, including all replay/smoke, rejected candidates and evaluations. Graph construction, CPU/GPU tests, spectra and reporting are separate research work. Deployable adaptation costs additionally charge the corresponding graph generation/validation; spectra are excluded from training-overhead comparisons and reported separately. CUDA timing uses synchronized monotonic perf_counter; upstream calendar progress logs can jump on WSL. Peak allocation is PyTorch allocator usage, CPU RSS is sampled, neither is total physical device occupancy.\n\n'
    text+='The matched_compute.csv comparison uses the nearest already-evaluated ordinary500/1000/5000 checkpoint by actual recorded time, reporting its timing mismatch explicitly. It is approximate, not an exactly time-matched rerun. Fresh zero1000 controls provide common instrumentation/setup; longer ordinary training timers come from Gate2. Partial500/2500 costs reconstruct measured prefix training plus fixed process/restart setup, not separately launched jobs. Every interrupted/rejected attempt remains charged.\n\n'
    text+='## Stability and structural diagnostics\n\nRetained sets use the official signed ranking and connectivity pruning. Pairwise cross-seed Jaccard is in ranking_stability.csv; a one-seed variant has no cross-seed stability estimate. Raw/effective gradient norms, variance, gradient concentration and successive rank changes are in gradient_diagnostics.csv. Structural score summaries group source/destination attention heads, MLPs and edge types; these are diagnostic, not tuning targets. Full retained edge lists and score vectors stay local.\n\n'
    text+='## Limitations and recommendation\n\nThe fixed128-prompt development cohort has been reused across gates. Final5000-step modified quality is untested if an earlier gate fails. No private/public test split, full-data multi-model run, deferred ranking refinement, swaps, hard-mask continuation or leaderboard submission is performed. Preserve any negative result; do not extend beta/topology sweeps. If the registered gates fail, retain ordinary MAttr as the reference and investigate the recorded gradient/compact-accuracy tradeoffs before proposing a separately preregistered study. A development success would still require independent validation before generalization or leaderboard claims.\n\n'
    for name in ('cpr_vs_updates','cpr_vs_compute','cpr_vs_gap','cmd_vs_updates','compact_accuracy','paired_cpr','ranking_stability'):
        text+=f'![{name}](results/mattr_expander/{out.name}/figures/{name}.png)\n\n'
    (ROOT/'GATE5A_EXPANDER_MATTR_REPORT.md').write_text(text,encoding='utf-8')
    (out/'GATE5A_EXPANDER_MATTR_REPORT.md').write_text(text.replace(f'results/mattr_expander/{out.name}/figures/','figures/'),encoding='utf-8')
