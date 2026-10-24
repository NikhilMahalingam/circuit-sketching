"""Offline aggregation; no model construction or forwards."""
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.mattr_optimization.common import ROOT,sha,write_json
from .core import protocol,endpoint_key,verified
from .statistics import contrasts,interval

def table(frame):
    from src.mattr_optimization.gate2_final_analysis import table as render
    return render(frame)

def expected(p):
    for cohort in ('primary','development'):
        for e in p['endpoints']:yield cohort,endpoint_key(e),e
        for e in p['eap_jobs']:yield cohort,e['key'],dict(method='eap_ig',dataset_size=e['dataset_size'],seed=e['seed'] if e['dataset_size']==128 else 'one_full_ranking',step=0)

def compute(out):
    rows=[]
    if (out/'preflight_cpu.json').exists():rows.append(dict(job='CPU_preflight',attempt='preflight',status='complete',wall_seconds=json.loads((out/'preflight_cpu.json').read_text())['seconds']))
    for path in (out/'training').rglob('reuse.json'):
        data=json.loads(path.read_text());rows.append(dict(job='reuse_export_'+path.parent.name,attempt='CPU_export',status='complete',wall_seconds=data.get('new_CPU_export_seconds',0.),optimizer_updates=0))
    for path in (out/'attempts').rglob('attempt.json'):
        a=json.loads(path.read_text());row=dict(job=a['key'],attempt=a['job_id'],status=a['status'],wall_seconds=a.get('seconds',0))
        accounting=path.parent/'compute_accounting.csv'
        if accounting.exists():
            frame=pd.read_csv(accounting)
            for col in ('forward_batches','example_forwards','backward_calls','optimizer_updates'):row[col]=frame[col].sum()
            for col in ('peak_allocated_mib','peak_reserved_mib','sampled_peak_cpu_rss_mib'):row[col]=frame[col].max()
            for stage in ('training','eap_ig','official_evaluation'):
                row[stage+'_seconds']=frame.loc[frame.stage==stage,'seconds'].sum()
        rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(out/'compute_summary.csv',index=False)
    reused=[]
    import torch
    for j in json.loads((out/'replacement_job_manifest.json').read_text()):
        if j['reuse_checkpoint']:
            state=torch.load(ROOT/j['reuse_checkpoint'],map_location='cpu',weights_only=False)
            reused.append(dict(job=j['key'],steps=j['reuse_step'],checkpoint=j['reuse_checkpoint'],
                historical_training_seconds=state.get('training_seconds',np.nan),scope='preexisting training timer; never charged as new Gate5F work; setup/evaluation differs across historical campaigns'))
    pd.DataFrame(reused).to_csv(out/'reused_training_cost.csv',index=False)
    serialization=[]
    for path in (out/'training').rglob('serialization.jsonl'):
        for line in path.read_text().splitlines():serialization.append(dict(job=path.parent.parent.name,**json.loads(line)))
    pd.DataFrame(serialization).to_csv(out/'checkpoint_serialization_cost.csv',index=False)
    return frame

def analyze(out,allow_incomplete=False):
    start=time.perf_counter();p=protocol(out);missing=[];records=[];metrics=[];caps=[];arrays={};templates={}
    for cohort,key,rec in expected(p):
        folder=out/'evaluations'/cohort/key
        if not verified(folder):missing.append(dict(cohort=cohort,key=key));continue
        result=json.loads((folder/'official_result.json').read_text())
        assert result['correctness_passed']
        records.append(({**rec,'cohort':cohort},folder))
        metrics.append({**rec,'cohort':cohort,**{k:result[k] for k in ('CPR','CMD','accuracy_log_AUC')}})
        frame=pd.read_csv(folder/'caps.csv');caps.append(frame.assign(**rec,cohort=cohort))
        per=pd.read_csv(folder/'cap_per_prompt.csv')
        if rec['method']!='eap_ig':
            for target,g in per.groupby('target'):
                g=g.sort_values('example')
                arrays[cohort,rec['method'],rec['dataset_size'],rec['step'],rec['seed'],target]=g.correct.to_numpy(dtype=float)
        templates[cohort]=pd.read_csv(out/f'cohort_{cohort}.csv').template_id.to_numpy()
    write_json(out/'completeness.json',dict(complete=not missing,missing=missing,registered_evaluation_jobs=64,completed=len(records)))
    if missing:
        body='# Gate 5F data scaling report\n\n**Incomplete; no experimental conclusion or classification.**\n\n'+table(pd.DataFrame(missing))+'\n'
        (out/'GATE5F_DATA_SCALING_REPORT.md').write_text(body)
        if not allow_incomplete:raise RuntimeError(f'{len(missing)} evaluation jobs missing; see completeness.json. No synthetic performance substituted.')
        return
    metrics=pd.DataFrame(metrics);caps=pd.concat(caps,ignore_index=True)
    metrics.to_csv(out/'official_metrics.csv',index=False);caps.to_csv(out/'accuracy_size_curves.csv',index=False)
    correct=all(json.loads((out/n).read_text())['passed'] for n in ('preflight_cpu.json','preflight_gpu.json','schedule_audit.json'))
    results={};factorial=[];seed_rows=[];exact=[];absolute=[];paired=[]
    for cohort in ('primary','development'):
        data={k[1:]:v for k,v in arrays.items() if k[0]==cohort}
        result,rows,draws=contrasts(data,templates[cohort],correct)
        # Only primary cohort is classified; development is explicitly descriptive.
        if cohort=='development':result.pop('classification');result['classification_used']=False
        results[cohort]=result
        factorial.extend([{**row,'cohort':cohort} for row in rows])
        for seed,effect in zip(range(42,47),result['per_seed']):
            h=data['h2',10000,5000,seed,116];b=data['baseline',10000,5000,seed,116]
            seed_rows.append(dict(cohort=cohort,seed=seed,effect=effect,baseline_accuracy=b.mean(),h2_accuracy=h.mean(),
                                  corrected=int(((h==1)&(b==0)).sum()),broken=int(((h==0)&(b==1)).sum())))
        for n,s in ((128,1000),(128,5000),(10000,1000),(10000,5000)):
            seeds=range(42,47) if (n,s)==(10000,5000) else (42,43,44)
            for target in p['caps']:
                same=[];attained=[];differences=[]
                for seed in seeds:
                    a=caps[(caps.cohort==cohort)&(caps.method=='h2')&(caps.dataset_size==n)&(caps.step==s)&(caps.seed==seed)&(caps.target==target)].iloc[0]
                    b=caps[(caps.cohort==cohort)&(caps.method=='baseline')&(caps.dataset_size==n)&(caps.step==s)&(caps.seed==seed)&(caps.target==target)].iloc[0]
                    if a.actual_edges==b.actual_edges:
                        same.append(seed);differences.append(data['h2',n,s,seed,target]-data['baseline',n,s,seed,target])
                    if a.actual_edges==b.actual_edges==target:attained.append(seed)
                    h=data['h2',n,s,seed,target];z=data['baseline',n,s,seed,target]
                    paired.append(dict(cohort=cohort,dataset_size=n,step=s,seed=seed,target=target,
                        baseline_actual=int(b.actual_edges),h2_actual=int(a.actual_edges),equal_count=bool(a.actual_edges==b.actual_edges),
                        both_attain_target=bool(a.actual_edges==b.actual_edges==target),corrected=int(((h==1)&(z==0)).sum()),broken=int(((h==0)&(z==1)).sum()),
                        **interval(h-z,draws)))
                row=dict(cohort=cohort,dataset_size=n,step=s,target=target,equal_count_seeds=json.dumps(same),both_attain_target_seeds=json.dumps(attained))
                if differences:row.update(interval(np.mean(differences,axis=0),draws))
                exact.append(row)
        for method in ('baseline','h2'):
            for r in (42,43,44):
                for s in (1000,5000):
                    absolute.append(dict(cohort=cohort,method=method,seed=r,contrast='data_effect_'+str(s),
                        **interval(data[method,10000,s,r,116]-data[method,128,s,r,116],draws)))
                for n in (128,10000):
                    absolute.append(dict(cohort=cohort,method=method,seed=r,contrast='step_effect_'+str(n),
                        **interval(data[method,n,5000,r,116]-data[method,n,1000,r,116],draws)))
            for name,changes in [(f'data_effect_{s}',np.stack([data[method,10000,s,r,116]-data[method,128,s,r,116] for r in (42,43,44)])) for s in (1000,5000)]+[(f'step_effect_{n}',np.stack([data[method,n,5000,r,116]-data[method,n,1000,r,116] for r in (42,43,44)])) for n in (128,10000)]:
                absolute.append(dict(cohort=cohort,method=method,seed='equal_seed_mean',contrast=name,**interval(changes.mean(axis=0),draws)))
    factorial=pd.DataFrame(factorial);factorial.to_csv(out/'factorial_contrasts.csv',index=False)
    pd.DataFrame(seed_rows).to_csv(out/'primary_seed_effects.csv',index=False)
    pd.DataFrame(exact).to_csv(out/'exact_size_subsets.csv',index=False)
    pd.DataFrame(absolute).to_csv(out/'absolute_method_changes.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_seed_effects.csv',index=False)
    write_json(out/'gate5f_decision.json',results)
    from .structure import analyze as structure
    structure(out,records)
    overlap=pd.read_csv(out/'known_head_overlap.csv');control=pd.read_csv(out/'topology_random_controls.csv')
    group=['method','dataset_size','step','seed','cohort','target']
    summary=control.groupby(group,dropna=False).mean(numeric_only=True).reset_index()
    merged=overlap.merge(summary,on=group,suffixes=('_observed','_control_mean'))
    for col in ('incident_known_fraction','known_heads'):
        merged[col+'_enrichment']=merged[col+'_observed']-merged[col+'_control_mean']
    merged.to_csv(out/'enrichment_vs_topology.csv',index=False)
    costs=compute(out);plots(out,caps,factorial,metrics,pd.DataFrame(seed_rows),overlap,costs)
    mean_table=caps[(caps.cohort=='primary')&(caps.target==116)&(caps.method!='eap_ig')&caps.seed.isin([42,43,44])].groupby(['method','dataset_size','step'])[['accuracy','actual_edges']].mean().reset_index()
    primary=results['primary'];inv=pd.read_csv(out/'checkpoint_inventory.csv')
    report=f'''# Gate 5F: data scaling and training duration

## Registered primary result

Classification: **{primary['classification']}**. Five-seed mean and prompt interval:

```json
{json.dumps(primary,indent=2)}
```

Only the fresh primary cohort decides this result. Gate5E cohort results are development comparisons. No historical Gate5C/5D/5E decisions were revised. No GPT-2 weights were trained.

## Design and provenance

Protocol SHA-256: `{sha(out/'gate5f_protocol.json')}`. Complete2×2 comparisons use seeds42–44; primary full-data5000 uses42–46. The additional seeds' intermediate checkpoints are not factorial observations. All28 registered training endpoints were evaluated on both512-prompt cohorts, with four EAP-IG rankings per cohort. The full-data EAP ranking is one ranking. Uniform-k, Adam.05, temperature.5, float32, iso masking and official live patching were unchanged. H2 gamma=0 uses all32491 edge residuals plus centered receiver components. Exact mapping hashes, checkpoint inventory, stream hashes and schedule tests are archived.

## Factorial accuracy table (three fixed seeds)

{table(mean_table)}

## Five-seed primary effects

{table(pd.DataFrame(seed_rows)[lambda x:x.cohort=='primary'])}

## Factorial contrasts

{table(factorial[(factorial.cohort=='primary')&(factorial.seed=='equal_seed_mean')])}

Absolute per-method data/duration changes (three fixed seeds):

{table(pd.DataFrame(absolute)[lambda x:(x.cohort=='primary')&(x.seed=='equal_seed_mean')])}

All per-condition/per-seed corrected and broken counts, actual-size differences and prompt intervals are in paired_seed_effects.csv.

All data effects are observed discovery-pool effects under the registered subset/optimizer sampling design. N128 seeds change both subset and optimizer draws; full-pool seeds change optimizer draws. Neither a three-seed nor a five-seed comparison is an independent set of prompt datasets. The5000 paired bootstrap draws (seed5644) resample exact template strata and use identical prompt indices for every method/condition/seed within each cohort. Intervals condition on fixed models and template mix; they do not establish significance across training seeds or unseen template families. Raw corrected/broken counts and per-seed ranges/SD/median are retained. Secondary intervals are descriptive and unadjusted.

## Exact counts and safeguards

This is an at-most116-edge primary endpoint. Exact equal-count subsets and both-attain-target flags are in exact_size_subsets.csv; all actual counts, nominal prefixes, shortfalls and mask hashes are in accuracy_size_curves.csv. No edges were artificially added. Larger-cap safeguards at296/400/640 and all frozen classification booleans are recorded above. A zero-crossing interval alone does not demonstrate equivalence. The full criteria and operational definitions were frozen before training.

## Official CPR, CMD and fidelity

{table(metrics)}

Every area uses the pinned official evaluator on the indicated cohort. Do not compare these512-row CPRs directly against Gate1 full-validation CPR as identical samples. Positive-margin correctness, full-vocabulary top1, margin magnitude, normalized faithfulness, accuracy log-AUC and CPR remain distinct. The plots expose cases where margin amplification changes CPR without corresponding compact correctness. Full nominal/per-prompt outputs are retained for each job.

## EAP-IG reference

The pinned EAP-IG-inputs implementation uses five left endpoints, batch4 and official mean-loss reduction. Equal-batch partitioning was verified in preflight. Changing batch size changes raw scale under this upstream reduction, so it was not changed. Full10000 attribution is one fixed-data ranking, not five seeds. Its attribution cost and each cohort's evaluation are charged separately.

## Known-head overlap and topology controls

The26-head reference is pinned to the authoritative ACDC author implementation and Wang et al., [Interpretability in the Wild, v1](https://arxiv.org/html/2211.00593v1). Known IOI classes and exact reference commit/hash were frozen. Overlap measures head representation and incident-edge fraction, not validated mechanistic equivalence. Sixty-four seeded within-layer head-relabeling automorphisms preserve Q/K/V identities together, actual size, layer structure, source/destination types and official connectivity. They test identity enrichment conditional on layer structure, not enrichment in known layers. All controls passed official pruning; achieved sizes/zero shortfalls and descriptive observed-minus-control means are archived. This analysis cannot change classification.

## Compute and reuse

{table(costs)}

CUDA-synchronized wall times are distinct from GPU utilization. Root forwards, example-forwards, backwards, updates, serialization, cold attempts, peak allocated/reserved CUDA memory and sampled CPU RSS are logged; allocation is not physical VRAM occupancy. Reused historical training costs are separate in reused_training_cost.csv, never relabeled as new work. Interrupted attempts are included. All rejected/incompatible reuse reasons are in checkpoint_inventory.csv. Cache hits execute no forwards and reuse immutable exact outputs only.

## Limitations and recommendation

These are fixed trained models, known IOI templates, capped circuits, and public validation data historically exposed by Gate1. The primary cohort is fresh relative to smaller campaigns only. There is no private-test or leaderboard claim. Classification and observed data/step contrasts determine whether additional independent validation of compact hierarchy is justified; CPR/fidelity tradeoffs and control results must accompany any claim. No further algorithm or gate is automatically authorized.
'''
    for name in ('accuracy_size','factorial','seed_effects','interaction','cpr_accuracy','eap_comparison','head_recall','compute'):
        report+=f'\n\n![{name}](figures/{name}.png)'
    (out/'GATE5F_DATA_SCALING_REPORT.md').write_text(report+'\n')
    (ROOT/'GATE5F_DATA_SCALING_REPORT.md').write_text(report.replace('(figures/',f'(results/mattr_data_scaling/{out.name}/figures/')+'\n')
    write_json(out/'analysis_metadata.json',dict(seconds=time.perf_counter()-start,model_forwards=0,complete=True))
    print('Analysis complete:',primary['classification'],flush=True)

def plots(out,caps,factorial,metrics,seeds,heads,costs):
    dest=out/'figures';dest.mkdir(exist_ok=True)
    def save(name):plt.tight_layout();plt.savefig(dest/(name+'.png'),dpi=150);plt.close()
    data=caps[caps.cohort=='primary']
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,(n,s) in zip(axes.flat,[(128,1000),(128,5000),(10000,1000),(10000,5000)]):
        for method in ('baseline','h2'):
            mean=data[(data.method==method)&(data.dataset_size==n)&(data.step==s)&data.seed.isin([42,43,44])].groupby('target')[['actual_edges','accuracy']].mean()
            ax.plot(mean.actual_edges,mean.accuracy,'o-',label=method)
        ax.set(title=f'N={n},updates={s}',xlabel='Actual edges (seed mean)',ylabel='Positive-margin accuracy');ax.legend()
    save('accuracy_size')
    f=factorial[(factorial.cohort=='primary')&(factorial.seed=='equal_seed_mean')]
    plt.figure(figsize=(8,4));names=['delta128_1000','delta128_5000','delta10000_1000','delta10000_5000']
    for i,name in enumerate(names):
        r=f[f.contrast==name].iloc[0];plt.errorbar(i,r['mean'],yerr=[[max(0,r['mean']-r.lower)],[max(0,r.upper-r['mean'])]],fmt='o',capsize=4)
    plt.xticks(range(4),names,rotation=12);plt.axhline(0,color='black');plt.ylabel('H2 minus baseline accuracy');save('factorial')
    plt.figure(figsize=(7,4));s=seeds[seeds.cohort=='primary'];plt.bar(s.seed.astype(str),s.effect);plt.axhline(0,color='black');plt.ylabel('Primary per-seed accuracy difference');save('seed_effects')
    plt.figure(figsize=(7,4))
    for n in (128,10000):
        rows=f.set_index('contrast');plt.plot([1000,5000],[rows.loc[f'delta{n}_1000','mean'],rows.loc[f'delta{n}_5000','mean']],'o-',label=f'N{n}')
    plt.xlabel('Optimizer updates');plt.ylabel('Hierarchy advantage');plt.legend();save('interaction')
    m=metrics.merge(data[data.target==116][['method','dataset_size','step','seed','accuracy']],on=['method','dataset_size','step','seed'])
    plt.figure(figsize=(8,5))
    for method,g in m[m.cohort=='primary'].groupby('method'):plt.scatter(g.CPR,g.accuracy,label=method)
    plt.xlabel('Official CPR');plt.ylabel('Accuracy at116-edge cap');plt.legend();save('cpr_accuracy')
    plt.figure(figsize=(8,5))
    for method,g in data[((data.dataset_size==10000)&(data.step==5000))|(data.method=='eap_ig')].groupby('method'):
        mean=g.groupby('target').accuracy.mean();plt.plot(mean.index,mean,'o-',label=method)
    plt.xlabel('Actual-edge cap');plt.ylabel('Positive-margin accuracy');plt.legend();save('eap_comparison')
    plt.figure(figsize=(9,5));columns=[c for c in heads if c.endswith('_recall')]
    for method,g in heads[heads.cohort=='primary'].groupby('method'):
        values=g.assign(mean_class_recall=g[columns].mean(axis=1)).groupby('target').mean_class_recall.mean()
        plt.plot(values.index,values,'o-',label=method)
    plt.xlabel('Actual-edge cap');plt.ylabel('Mean known-head class recall (descriptive pooled conditions)');plt.legend();save('head_recall')
    plt.figure(figsize=(10,5));g=costs.groupby('job').wall_seconds.sum();plt.bar(range(len(g)),g/60);plt.xticks(range(len(g)),g.index,rotation=90,fontsize=6);plt.ylabel('Observed wall minutes (all attempts)');save('compute')
